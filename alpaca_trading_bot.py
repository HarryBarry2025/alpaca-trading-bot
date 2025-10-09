# alpaca_trading_bot.py
import os, io, json, time, asyncio, traceback
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.7)
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler, ContextTypes, filters
)
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")  # ungenutzt im Polling-Modus
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional, nur wenn data_provider=alpaca genutzt wird)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # z.B. https://paper-api.alpaca.markets

# ========= Config & State =========
class StratConfig(BaseModel):
    # Data / Engine
    symbol: str = "TQQQ"
    interval: str = "1h"     # '1d' oder '1h' / '15m' etc.
    lookback_days: int = 365

    data_provider: str = "alpaca"  # "alpaca" | "yahoo" | "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Strategy Inputs (TV-kompatible Varianten)
    rsiLen: int = 12
    rsiLow: float = 0.0      # explizit 0 (gewünscht)
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    macdFast: int = 8
    macdSlow: int = 21
    macdSig: int = 11
    efiLen: int = 11
    rsi_source: str = "close"  # "close" | "hlc3"

    # Risk / Exits
    slPerc: float = 1.0
    tpPerc: float = 400.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Loop / Live
    poll_minutes: int = 10
    live_enabled: bool = False

    # Market-time guard
    market_hours_only: bool = True
    use_us_holidays: bool = True

    # Trading (Paper via Alpaca)
    trade_enabled: bool = False
    tif: str = "day"  # "day" | "gtc" | "ioc" | "fok"
    sizing_mode: str = "shares"  # "shares" | "notional"
    fixed_shares: int = 1
    notional_usd: float = 100.0
    max_pos: int = 1  # einfache Ein-Pos Logik

class StratState(BaseModel):
    # naive single-position model (für Messaging)
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_action_bar_index: Optional[int] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ========= Timer / Background =========
TIMER_STATE = {
    "enabled": False,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": True,
    "last_reason": "init"
}
_TIMER_TASK: Optional[asyncio.Task] = None

# ========= Market hours & holidays =========
import zoneinfo
US_EASTERN = zoneinfo.ZoneInfo("America/New_York")

_US_HOLIDAYS_Y = {
    2025: {"2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
           "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25"},
    2024: {"2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27",
           "2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25"},
}

def is_us_holiday(dt_utc: datetime) -> bool:
    if not CONFIG.use_us_holidays:
        return False
    dt_et = dt_utc.astimezone(US_EASTERN)
    return dt_et.strftime("%Y-%m-%d") in _US_HOLIDAYS_Y.get(dt_et.year, set())

def is_rth(dt_utc: datetime) -> bool:
    """Regular Trading Hours NYSE/Nasdaq 09:30–16:00 ET, Mo–Fr, kein Feiertag."""
    if not CONFIG.market_hours_only:
        return True
    dt_et = dt_utc.astimezone(US_EASTERN)
    if dt_et.weekday() >= 5:  # Sa/So
        return False
    if is_us_holiday(dt_utc):
        return False
    t = dt_et.time()
    return dtime(9,30) <= t <= dtime(16,0)

# ========= Indicators (TV-kompatibel) =========
def ema(s: pd.Series, n: int) -> pd.Series:
    # TV: adjust=False entspricht EMA-Definition
    return s.ewm(span=n, adjust=False).mean()

def _rsi_source_series(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode.lower() == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    return df["close"]

def rsi_tv(df: pd.DataFrame, length: int, source: str = "close") -> pd.Series:
    src = _rsi_source_series(df, source)
    delta = src.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    # Wilder's smoothing via EMA (TradingView-äquivalent)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(df: pd.DataFrame, fast: int, slow: int, sig: int):
    line = ema(df["close"], fast) - ema(df["close"], slow)
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def efi_tv(df: pd.DataFrame, efi_len: int) -> pd.Series:
    raw = df["volume"] * (df["close"] - df["close"].shift(1))
    return ema(raw, efi_len)

# ========= Data Providers =========
from urllib.parse import quote

def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    stooq_sym = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={quote(stooq_sym)}&i=d"
    try:
        df = pd.read_csv(url)
        if df is None or df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if lookback_days > 0:
            df = df.iloc[-lookback_days:]
        df["time"] = df.index.tz_localize("UTC")
        # Vereinheitlichte Spaltennamen
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[stooq] fetch failed:", e)
        return pd.DataFrame()

def fetch_alpaca_ohlcv(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] library not available:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing API key/secret")
        return pd.DataFrame()

    interval = interval.lower()
    if interval in {"1h","60m"}:
        tf = TimeFrame(1, TimeFrameUnit.Hour)
    elif interval in {"1d","1day"}:
        tf = TimeFrame(1, TimeFrameUnit.Day)
    elif interval in {"15m"}:
        tf = TimeFrame(15, TimeFrameUnit.Minute)
    elif interval in {"5m"}:
        tf = TimeFrame(5, TimeFrameUnit.Minute)
    else:
        tf = TimeFrame(1, TimeFrameUnit.Hour)

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 60))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        adjustment=None,
        feed="iex",     # wichtig für freien/paper Zugriff
        limit=10000
    )

    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty frame (prüfe feed/interval/zugang)")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0)
            except Exception:
                pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.sort_index()
        df["time"] = df.index
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        # Manche Alpaca Frames kommen bereits in lower-case:
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns:
                # fallback mapping versuchen
                if c.capitalize() in df.columns:
                    df[c] = df[c.capitalize()]
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}

    # 1) Alpaca
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=f"{interval}")
            return df, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")

    # 2) Stooq EOD
    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        note.update(provider="Stooq→Yahoo", detail="Stooq leer; versuche Yahoo")

    # 3) Yahoo
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}
    is_intraday = interval.lower() in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1)
            except Exception:
                pass
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            try:
                df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            except Exception:
                pass
        df = df.sort_index()
        df["time"] = df.index
        return df

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True,  threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    last_err = None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries + 1):
            try:
                if method == "download":
                    tmp = yf.download(**kwargs)
                else:
                    tmp = yf.Ticker(symbol).history(**kwargs)
                df = _normalize(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}")
                    return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2 ** (attempt - 1)))

    # Yahoo Fallback 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # Optional Stooq Fallback
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] keine Daten für {symbol} ({interval}, period={period}). Last error: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)", "detail":"keine Daten"}

# ========= Features / Logic =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv(out, cfg.rsiLen, cfg.rsi_source)
    macd_line, macd_sig, _ = macd(out, cfg.macdFast, cfg.macdSlow, cfg.macdSig)
    out["macd_line"], out["macd_sig"] = macd_line, macd_sig
    out["efi"] = efi_tv(out, cfg.efiLen)
    return out

def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.macdSlow, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_val = last["rsi"]
    rsi_rising = (last["rsi"] > prev["rsi"])
    rsi_falling = (last["rsi"] < prev["rsi"])
    efi_rising = (last["efi"] > prev["efi"])

    # kein MACD-Filter (wie gewünscht)
    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = last["time"]

    def sl(p): return p * (1 - cfg.slPerc/100.0)
    def tp(p): return p * (1 + cfg.tpPerc/100.0)

    # Bars since entry (approx.)
    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if st.position_size == 0:
        if entry_cond:
            size = 1
            return {"action": "buy", "qty": size, "price": price_open, "time": str(ts),
                    "reason": "rule_entry", "sl": sl(price_open), "tp": tp(price_open)}
        else:
            return {"action": "none", "reason": "flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(st.avg_price)
        cur_tp = tp(st.avg_price)

        hit_sl = price_close <= cur_sl
        hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            return {"action": "sell", "qty": st.position_size, "price": price_open, "time": str(ts), "reason": "rsi_exit"}
        if hit_sl:
            return {"action": "sell", "qty": st.position_size, "price": cur_sl, "time": str(ts), "reason": "stop_loss"}
        if hit_tp:
            return {"action": "sell", "qty": st.position_size, "price": cur_tp, "time": str(ts), "reason": "take_profit"}

        return {"action": "none", "reason": "hold"}

# ========= Alpaca trading helpers =========
def alpaca_place_order(side: str, qty: Optional[int], notional: Optional[float], tif: str) -> str:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
    except Exception as e:
        return f"[alpaca] lib missing: {e}"

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY and APCA_API_BASE_URL):
        return "[alpaca] missing credentials or base url"

    try:
        client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True, url_override=APCA_API_BASE_URL)
        tif_enum = {
            "day":"DAY","gtc":"GTC","ioc":"IOC","fok":"FOK"
        }.get(tif.lower(), "DAY")
        req = MarketOrderRequest(
            symbol=CONFIG.symbol,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=TimeInForce(tif_enum),
            qty=qty if qty is not None else None,
            notional=notional if notional is not None else None
        )
        o = client.submit_order(req)
        return f"[alpaca] order {o.id} {side} ok"
    except Exception as e:
        return f"[alpaca] order error: {e}"

def alpaca_positions_text() -> str:
    try:
        from alpaca.trading.client import TradingClient
    except Exception as e:
        return f"[alpaca] lib missing: {e}"
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY and APCA_API_BASE_URL):
        return "[alpaca] missing credentials or base url"

    try:
        client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True, url_override=APCA_API_BASE_URL)
        poss = client.get_all_positions()
        if not poss:
            return "📭 Keine offenen Alpaca-Positionen."
        lines = ["📌 Alpaca Positionen:"]
        for p in poss:
            lines.append(f"- {p.symbol}: {p.qty} @ {p.avg_entry_price}  (Market={p.current_price})  UPL={p.unrealized_pl}")
        return "\n".join(lines)
    except Exception as e:
        return f"[alpaca] positions error: {e}"

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False  # Singleton-Schutz

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        if not text or not text.strip():
            text = "ℹ️ (leer)"
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except BadRequest as e:
        print("send badrequest:", e)
    except Exception as e:
        print("send error:", e)

def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider","")
    det  = note.get("detail","")
    if not prov: return ""
    if prov == "Yahoo":
        return f"📡 Daten: Yahoo ({det})"
    elif prov.startswith("Yahoo (Fallback"):
        return f"📡 Daten: {prov} – {det}"
    elif prov.startswith("Stooq"):
        return f"📡 Daten: {prov} – {det} (nur Daily)"
    else:
        return f"📡 Daten: {prov} – {det}"

# ========= Strategy runner =========
def _compute_indicators_for_row(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Hilfsfunktion für /ind Ausgabe, berechnet Werte der letzten Bar robust."""
    sub = df.iloc[:idx+1].copy()
    fdf = build_features(sub, CONFIG)
    row = fdf.iloc[-1]
    prev = fdf.iloc[-2] if len(fdf) >= 2 else row
    return {
        "rsi": float(row["rsi"]),
        "rsi_prev": float(prev["rsi"]),
        "efi": float(row["efi"]),
        "efi_prev": float(prev["efi"]),
        "macd": float(row["macd_line"]),
        "signal": float(row["macd_sig"])
    }

async def run_once_and_report(chat_id: str, place_order: bool=False):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten für {CONFIG.symbol} ({CONFIG.interval}). "
                            f"Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return

    note_msg = _friendly_data_note(note)
    if note_msg and ( "Fallback" in note_msg or "Stooq" in note_msg or "Alpaca" in note_msg or CONFIG.data_provider!="alpaca"):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    # Orders (optional)
    if place_order and CONFIG.trade_enabled and is_rth(datetime.now(timezone.utc)):
        if act["action"] == "buy":
            qty   = None
            notion= None
            if CONFIG.sizing_mode == "shares":
                qty = max(1, int(CONFIG.fixed_shares))
            else:
                notion = max(1.0, float(CONFIG.notional_usd))
            msg = alpaca_place_order("buy", qty, notion, CONFIG.tif)
            await send(chat_id, msg)
        elif act["action"] == "sell" and STATE.position_size > 0:
            # simple fully close
            qty = STATE.position_size
            msg = alpaca_place_order("sell", qty, None, CONFIG.tif)
            await send(chat_id, msg)

    # State-Nachrichten (immer)
    if act["action"] == "buy":
        STATE.position_size = act["qty"]
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        await send(chat_id, f"🟢 LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f}  TP={act['tp']:.4f}")
    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\nPnL={pnl*100:.2f}%")
        STATE.position_size = 0
        STATE.avg_price = 0.0
        STATE.entry_time = None
    else:
        await send(chat_id, f"ℹ️ {STATE.last_status}")

# ========= Background timer =========
async def _timer_loop():
    try:
        TIMER_STATE["running"] = True
        TIMER_STATE["poll_minutes"] = CONFIG.poll_minutes
        TIMER_STATE["market_hours_only"] = CONFIG.market_hours_only
        TIMER_STATE["next_due"] = datetime.now(timezone.utc).isoformat()
        while TIMER_STATE["enabled"]:
            now = datetime.now(timezone.utc)

            if CONFIG.market_hours_only and not is_rth(now):
                TIMER_STATE["last_reason"] = "skip_outside_RTH_or_holiday"
                TIMER_STATE["next_due"] = (now + timedelta(minutes=CONFIG.poll_minutes)).isoformat()
                await asyncio.sleep(CONFIG.poll_minutes * 60)
                continue

            if CHAT_ID:
                try:
                    await run_once_and_report(CHAT_ID, place_order=CONFIG.trade_enabled)
                    TIMER_STATE["last_run"] = now.isoformat()
                    TIMER_STATE["last_reason"] = "tick_ok"
                except Exception as e:
                    TIMER_STATE["last_reason"] = f"tick_error: {e}"
            else:
                TIMER_STATE["last_reason"] = "no_chat_id"

            TIMER_STATE["next_due"] = (now + timedelta(minutes=CONFIG.poll_minutes)).isoformat()
            await asyncio.sleep(CONFIG.poll_minutes * 60)
    finally:
        TIMER_STATE["running"] = False

# ========= Telegram Commands =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status – Statusübersicht\n"
        "/cfg – Konfiguration anzeigen\n"
        "/set key=value … – z.B. /set rsiLow=0 rsiHigh=68 slPerc=1 tpPerc=400 rsi_source=hlc3\n"
        "/run – jetzt einen Live-Check ausführen\n"
        "/live on|off – Live-Modus schalten\n"
        "/bt 90 – Backtest über 90 Tage\n"
        "/ind – Indikatorwerte (Alias: /sig)\n"
        "/positions – Alpaca-Positionen anzeigen\n"
        "/plot 200 – Diagramm von Close, RSI, EFI, MACD\n"
        "/timerstart – internen 10m-Timer starten\n"
        "/timerstop – internen Timer stoppen\n"
        "/timerstatus – Timer-Zustand anzeigen"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbol: {CONFIG.symbol}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}  poll={CONFIG.poll_minutes}m\n"
        f"RTH-only: {'ON' if CONFIG.market_hours_only else 'OFF'}  Holidays: {'ON' if CONFIG.use_us_holidays else 'OFF'}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'}  TIF={CONFIG.tif}  Sizing={CONFIG.sizing_mode}\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Timer: enabled={TIMER_STATE['enabled']} running={TIMER_STATE['running']}\n"
        f"Last run: {TIMER_STATE['last_run']}  Next due: {TIMER_STATE['next_due']}\n"
        f"Last: {STATE.last_status}"
    )

def set_from_kv(k: str, v: str) -> str:
    mapping = {
        "sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit",
        "cooldown":"minBarsInTrade"
    }
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
    # Timer-state spiegeln, falls relevant
    if k == "poll_minutes":
        TIMER_STATE["poll_minutes"] = getattr(CONFIG, k)
    if k == "market_hours_only":
        TIMER_STATE["market_hours_only"] = getattr(CONFIG, k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 slPerc=1 tpPerc=400\n"
            "/set rsi_source=hlc3 interval=1h lookback_days=365\n"
            "/set data_provider=alpaca trade_enabled=true tif=day sizing_mode=shares fixed_shares=1\n"
            "/set sizing_mode=notional notional_usd=100\n"
            "/set market_hours_only=true use_us_holidays=true\n"
            "/set poll_minutes=10\n"
        )
        return
    msgs, errors = [], []
    for a in context.args:
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errors.append(f"❌ Ungültig: „{a}“ (erwarte key=value)")
            continue
        k, v = a.split("=", 1)
        k = k.strip(); v=v.strip()
        try:
            msgs.append(set_from_kv(k, v))
        except Exception as e:
            errors.append(f"❌ Fehler bei „{a}“: {e}")

    out = []
    if msgs:
        out.append("✅ Übernommen:")
        out.extend(msgs)
    if errors:
        out.append("\n⚠️ Probleme:")
        out.extend(errors)
    await update.message.reply_text("\n".join(out).strip())

async def cmd_cfg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Nutze: /live on oder /live off")
        return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(alpaca_positions_text())

async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await run_once_and_report(str(update.effective_chat.id), place_order=CONFIG.trade_enabled)

async def cmd_bt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass

    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für Backtest. Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return
    note_msg = _friendly_data_note(note)
    if note_msg:
        await update.message.reply_text(note_msg)

    fdf = build_features(df, CONFIG)

    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = row["rsi"]
        rsi_rising = row["rsi"] > prev["rsi"]
        rsi_falling = row["rsi"] < prev["rsi"]
        efi_rising  = row["efi"] > prev["efi"]

        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and rsi_rising and efi_rising
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
        await update.message.reply_text(
            f"📈 Backtest {days}d\nTrades: {entries}/{exits}\n"
            f"WinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_ind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    row, prev = fdf.iloc[-1], fdf.iloc[-2]
    text = (f"🔎 Indikatoren ({CONFIG.symbol} {CONFIG.interval})\n"
            f"RSI({CONFIG.rsiLen},{CONFIG.rsi_source}): {row['rsi']:.2f} (prev {prev['rsi']:.2f})\n"
            f"EFI({CONFIG.efiLen}): {row['efi']:.2f} (prev {prev['efi']:.2f})\n"
            f"MACD({CONFIG.macdFast},{CONFIG.macdSlow},{CONFIG.macdSig}): line={row['macd_line']:.4f}  sig={row['macd_sig']:.4f}\n"
            f"{_friendly_data_note(note)}")
    await update.message.reply_text(text)

# /sig als Alias von /ind
async def cmd_sig(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_ind(update, context)

async def cmd_plot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bars = 200
    if context.args:
        try: bars = int(context.args[0])
        except: pass

    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    f = fdf.tail(max(50, bars))

    # 3 einfache Panels in einem Bild (Close / RSI / EFI / MACD auf 2 Achsen)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(f.index, f["close"])
    ax0.set_title(f"{CONFIG.symbol} Close ({CONFIG.interval}) – {_friendly_data_note(note)}")
    ax0.grid(True)

    ax1.plot(f.index, f["rsi"])
    ax1.axhline(CONFIG.rsiLow, linestyle="--")
    ax1.axhline(CONFIG.rsiHigh, linestyle="--")
    ax1.set_title(f"RSI({CONFIG.rsiLen}, src={CONFIG.rsi_source})")
    ax1.grid(True)

    ax2.plot(f.index, f["efi"], label="EFI")
    # Zweitachse für MACD
    ax2b = ax2.twinx()
    ax2b.plot(f.index, f["macd_line"], alpha=0.7)
    ax2b.plot(f.index, f["macd_sig"], alpha=0.7)
    ax2.set_title(f"EFI({CONFIG.efiLen}) & MACD({CONFIG.macdFast},{CONFIG.macdSlow},{CONFIG.macdSig})")
    ax2.grid(True)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    try:
        await tg_app.bot.send_photo(chat_id=update.effective_chat.id, photo=buf)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Plot-Fehler: {e}")

async def cmd_timerstart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _TIMER_TASK
    TIMER_STATE["enabled"] = True
    TIMER_STATE["market_hours_only"] = CONFIG.market_hours_only
    if _TIMER_TASK and not _TIMER_TASK.done():
        await update.message.reply_text("⏱ Timer läuft bereits.")
        return
    _TIMER_TASK = asyncio.create_task(_timer_loop())
    await update.message.reply_text("⏱ Timer gestartet.")

async def cmd_timerstop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _TIMER_TASK
    TIMER_STATE["enabled"] = False
    await update.message.reply_text("⏱ Timer gestoppt.")

async def cmd_timerstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = TIMER_STATE
    txt = (
        "⏱ Timer-Status\n"
        f"enabled: {s['enabled']}\n"
        f"running: {s['running']}\n"
        f"poll_minutes: {s['poll_minutes']}\n"
        f"last_run: {s['last_run']}\n"
        f"next_due: {s['next_due']}\n"
        f"market_hours_only: {s['market_hours_only']}\n"
        f"last_reason: {s['last_reason']}"
    )
    await update.message.reply_text(txt)

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Lifespan (PTB Polling ohne Loop-Konflikt) =========
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED, _TIMER_TASK
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Commands
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))      # Alias
        tg_app.add_handler(CommandHandler("positions", cmd_positions))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
        tg_app.add_handler(CommandHandler("timerstart",  cmd_timerstart))
        tg_app.add_handler(CommandHandler("timerstop",   cmd_timerstop))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()

        # Sicher Webhook löschen (Polling-Mode)
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht (drop_pending_updates=True)")
        except Exception as e:
            print("delete_webhook warn:", e)

        # Polling nur einmal starten
        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    print("▶️ starte Polling…")
                    await tg_app.updater.start_polling(
                        poll_interval=1.0,
                        timeout=10.0
                    )
                    POLLING_STARTED = True
                    print("✅ Polling läuft")
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. Retry in {delay}s…")
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc()
                    await asyncio.sleep(10)

        tg_running = True
        print("🚀 Telegram POLLING aktiv")

        # Optional: Auto-Timer beim Start aktivieren (falls gewünscht)
        # TIMER_STATE["enabled"] = True
        # if _TIMER_TASK is None or _TIMER_TASK.done():
        #     _TIMER_TASK = asyncio.create_task(_timer_loop())

    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        tg_running = False
        try:
            await tg_app.updater.stop()
        except Exception:
            pass
        try:
            await tg_app.stop()
        except Exception:
            pass
        try:
            await tg_app.shutdown()
        except Exception:
            pass
        TIMER_STATE["enabled"] = False
        try:
            if _TIMER_TASK:
                _TIMER_TASK.cancel()
        except Exception:
            pass
        POLLING_STARTED = False
        print("🛑 Telegram POLLING gestoppt")

# ========= FastAPI App =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

# ========= HTTP Routes =========
@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider
    }

@app.head("/")
async def root_head():
    return

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/tick")
async def tick():
    if not CONFIG.live_enabled:
        return {"ran": False, "reason": "live_disabled"}
    if CHAT_ID is None:
        return {"ran": False, "reason": "no_chat_id (use /start in Telegram)"}
    if CONFIG.market_hours_only and not is_rth(datetime.now(timezone.utc)):
        return {"ran": False, "reason": "outside_rth_or_holiday"}
    await run_once_and_report(CHAT_ID, place_order=CONFIG.trade_enabled)
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

@app.get("/tgstatus")
def tgstatus():
    return {
        "tg_running": tg_running,
        "polling_started": POLLING_STARTED
    }

# Optional: Webhook-Route (unbenutzt im Polling-Modus)
@app.post(f"/telegram/{WEBHOOK_SECRET}")
async def telegram_webhook(req: Request):
    if tg_app is None:
        raise HTTPException(503, "Bot not ready")
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
