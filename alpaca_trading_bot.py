import os, asyncio, json, time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from contextlib import asynccontextmanager  # <—— NEU

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")  # e.g. https://your-service.onrender.com
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")  # optional; will be set on /start if empty

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"     # '1d' or '1h'
    lookback_days: int = 365
    rsiLen: int = 12
    rsiLow: float = 52.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    macdFast: int = 8
    macdSlow: int = 21
    macdSig: int = 11
    efiLen: int = 11
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0
    poll_minutes: int = 10
    live_enabled: bool = False

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_action_bar_index: Optional[int] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ========= Indicators =========
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=s.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=s.index)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

def macd(s: pd.Series, fast: int, slow: int, sig: int):
    fast_ema = ema(s, fast)
    slow_ema = ema(s, slow)
    line = fast_ema - slow_ema
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def efi_func(close: pd.Series, vol: pd.Series, efi_len: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, efi_len)

# ========= Data =========
def fetch_ohlcv(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days+5)
    df = yf.download(symbol, interval=interval, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    df.index = df.index.tz_convert("UTC") if df.index.tzinfo else df.index.tz_localize("UTC")
    df["time"] = df.index
    return df

def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsiLen)
    macd_line, macd_sig, _ = macd(out["close"], cfg.macdFast, cfg.macdSlow, cfg.macdSig)
    out["macd_line"], out["macd_sig"] = macd_line, macd_sig
    out["efi"] = efi_func(out["close"], out["volume"], cfg.efiLen)
    return out

# ========= Strategy Logic =========
def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.macdSlow, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    rsi_val = last["rsi"]
    macd_above = (last["macd_line"] > last["macd_sig"])
    rsi_rising = (last["rsi"] > prev["rsi"])
    rsi_falling = (last["rsi"] < prev["rsi"])
    efi_rising = (last["efi"] > prev["efi"])
    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising and macd_above
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling
    price = float(last["close"])
    o = float(last["open"])
    ts = last["time"]
    def sl(price_entry): return price_entry * (1 - cfg.slPerc/100.0)
    def tp(price_entry): return price_entry * (1 + cfg.tpPerc/100.0)
    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)
    if st.position_size == 0:
        if entry_cond:
            size = 1
            return {"action": "buy", "qty": size, "price": o, "time": str(ts), "reason": "rule_entry",
                    "sl": sl(o), "tp": tp(o)}
        else:
            return {"action": "none", "reason": "flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok
        cur_sl = sl(st.avg_price)
        cur_tp = tp(st.avg_price)
        hit_sl = price <= cur_sl
        hit_tp = price >= cur_tp
        if rsi_exit_ok:
            return {"action": "sell", "qty": st.position_size, "price": o, "time": str(ts), "reason": "rsi_exit"}
        if hit_sl:
            return {"action": "sell", "qty": st.position_size, "price": cur_sl, "time": str(ts), "reason": "stop_loss"}
        if hit_tp:
            return {"action": "sell", "qty": st.position_size, "price": cur_tp, "time": str(ts), "reason": "take_profit"}
        return {"action": "none", "reason": "hold"}

# ========= Telegram Bot =========
tg_app: Optional[Application] = None
tg_running = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        print("send error:", e)

# === Handlers (unverändert aus deinem Code) ===
# ... cmd_start / cmd_status / cmd_set / cmd_cfg / cmd_live / run_once_and_report / cmd_run / cmd_bt / on_message ...

# ========= Lifespan (ersetzt @app.on_event) =========
from contextlib import asynccontextmanager
import asyncio
import traceback

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startet Telegram im POLLING-Modus als Hintergrundtask (non-blocking),
    damit FastAPI parallel weiterläuft. Robust mit Fehler-Logs.
    """
    global tg_app, tg_running
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Handlers registrieren
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        # Hintergrund-Task: run_polling (blockierend) als Task starten
        # stop_signals=None => Render sendet keine Unix-Signale (Windows-ähnliche Umgebung)
        polling_task = asyncio.create_task(
            tg_app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                close_bot_session=True,
                drop_pending_updates=False,
                stop_signals=None
            )
        )
        tg_running = True
        print("🚀 Telegram POLLING gestartet")
    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()
        polling_task = None

    try:
        yield  # ---- App läuft hier ----
    finally:
        tg_running = False
        if polling_task and not polling_task.done():
            polling_task.cancel()
            try:
                await polling_task
            except Exception:
                pass
        print("🛑 Telegram POLLING gestoppt")


# ========= FastAPI App =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

# Telegram webhook endpoint
@app.post(f"/telegram/{WEBHOOK_SECRET}")
async def telegram_webhook(req: Request):
    if tg_app is None:
        raise HTTPException(503, "Bot not ready")
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

# Health check
@app.get("/")
async def root():
    return {"ok": True, "live": CONFIG.live_enabled, "symbol": CONFIG.symbol, "interval": CONFIG.interval}





# ========= HEALTH & DIAGNOSE ROUTES =========

@app.get("/")
async def root():
    """Health check: zeigt Basisstatus der Strategie."""
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval
    }

@app.get("/tick")
async def tick():
    """Cron-/Ping-freundlich: führt einen Strategy-Check aus, wenn live mode ON ist."""
    if not CONFIG.live_enabled:
        return {"ran": False, "reason": "live_disabled"}
    if CHAT_ID is None:
        return {"ran": False, "reason": "no_chat_id (use /start in Telegram)"}
    await run_once_and_report(CHAT_ID)
    return {"ran": True}

@app.get("/envcheck")
def envcheck():
    """Diagnose: zeigt ob die ENV Variablen vorhanden sind (ohne Secrets auszudrucken)."""
    def chk(k): 
        v = os.getenv(k)
        return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "BASE_URL": chk("BASE_URL"),
        "TELEGRAM_WEBHOOK_SECRET": chk("TELEGRAM_WEBHOOK_SECRET"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_API_BASE_URL": chk("APCA_API_BASE_URL"),
    }

