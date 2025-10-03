# alpaca_trading_bot.py
import os, json, asyncio, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# Telegram (PTB v20+)
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler, filters
)

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")  # nicht genutzt im Polling-Modus
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")  # optional

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"     # '1d' or '1h'
    lookback_days: int = 365

    # Pine-like inputs
    rsiLen: int = 12
    rsiLow: float = 52.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    macdFast: int = 8
    macdSlow: int = 21
    macdSig: int = 11
    efiLen: int = 11

    # Risk
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine
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
    df.index = df.index.tz_convert("UTC") if getattr(df.index, "tz", None) else df.index.tz_localize("UTC")
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

    # Bars since entry (approx.)
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

# ========= Telegram helpers & handlers =========
tg_app: Optional[Application] = None
tg_running = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        print("send error:", e)

async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status – zeigt Status\n"
        "/set key=value … – z.B. /set rsiLow=52 rsiHigh=68 sl=2 tp=4\n"
        "/run – einen Live-Check jetzt ausführen\n"
        "/live on|off – Live-Loop schalten\n"
        "/cfg – aktuelle Konfiguration\n"
        "/bt 90 – Backtest über 90 Tage\n"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbol: {CONFIG.symbol} {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} (alle {CONFIG.poll_minutes}m)\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Last: {STATE.last_status}"
    )

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1)
    k = k.strip(); v = v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k, k)
    if not hasattr(CONFIG, k):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):
        setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):
        setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):
        setattr(CONFIG, k, float(v))
    else:
        setattr(CONFIG, k, v)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /set key=value [key=value] …")
        return
    msgs = [set_from_kv(a) for a in context.args if "=" in a]
    await update.message.reply_text("\n".join(msgs))

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on oder /live off")
        return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def run_once_and_report(chat_id: str):
    df = fetch_ohlcv(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, "❌ Keine Daten vom Feed.")
        return
    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"
    if act["action"] == "buy":
        STATE.position_size = act["qty"]
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        await send(chat_id, f"🟢 LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f} TP={act['tp']:.4f}")
    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\nPnL={pnl*100:.2f}%")
        STATE.position_size = 0
        STATE.avg_price = 0.0
        STATE.entry_time = None
    else:
        await send(chat_id, f"ℹ️ {STATE.last_status}")

async def cmd_run(update, context):
    await run_once_and_report(str(update.effective_chat.id))

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    df = fetch_ohlcv(CONFIG.symbol, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten für Backtest.")
        return
    fdf = build_features(df, CONFIG)

    # simpler EoB-Backtest
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = row["rsi"]
        macd_above = row["macd_line"] > row["macd_sig"]
        rsi_rising = row["rsi"] > prev["rsi"]
        rsi_falling = row["rsi"] < prev["rsi"]
        efi_rising = row["efi"] > prev["efi"]
        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and rsi_rising and efi_rising and macd_above
        exitc = (rsi_val<CONFIG.rsiExit) and rsi_falling
        if pos==0 and entry:
            pos=1; avg=float(row["open"]); entries+=1
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            price = float(row["close"])
            stop = price<=sl; take = price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                r = (px-avg)/avg
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a = np.array(R); win=(a>0).mean()
        pf = (a[a>0].sum()) / (1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr = (eq**(365/max(1,days)) - 1)
        await update.message.reply_text(f"📈 Backtest {days}d: entries={entries}, exits={exits}\nWin={win:.2f} PF={pf:.2f} CAGR~{cagr*100:.2f}%")
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Lifespan (Polling, non-blocking) =========
from contextlib import asynccontextmanager

tg_app: Optional[Application]  # redeclare for type hints

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running
    polling_task = None
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        # Polling als Hintergrundtask
        polling_task = asyncio.create_task(tg_app.run_polling())
        tg_running = True
        print("🚀 Telegram POLLING gestartet")
    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        tg_running = False
        if polling_task and not polling_task.done():
            polling_task.cancel()
            try:
                await polling_task
            except Exception:
                pass
        try:
            await tg_app.shutdown()
        except Exception:
            pass
        print("🛑 Telegram POLLING gestoppt")

# ========= FastAPI App =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

# ========= HEALTH & DIAGNOSE ROUTES =========
@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval
    }

@app.head("/")
async def root_head():
    return

@app.get("/tick")
async def tick():
    if not CONFIG.live_enabled:
        return {"ran": False, "reason": "live_disabled"}
    if CHAT_ID is None:
        return {"ran": False, "reason": "no_chat_id (use /start in Telegram)"}
    await run_once_and_report(CHAT_ID)
    return {"ran": True}

@app.get("/envcheck")
def envcheck():
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
