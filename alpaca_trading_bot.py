# alpaca_trading_bot.py
import os, io, json, time, asyncio, traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    CommandHandler, MessageHandler, filters
)
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
# Trading-Optionen
AUTO_TRADE      = os.getenv("AUTO_TRADE", "false").lower() in ("1","true","on","yes")
SIZING_MODE     = os.getenv("SIZING_MODE", "shares")      # "shares" | "notional"
SHARES          = float(os.getenv("SHARES", "1"))
NOTIONAL_USD    = float(os.getenv("NOTIONAL_USD", "1000"))
ALPACA_TIF      = os.getenv("ALPACA_TIF", "day").lower()  # "day" | "gtc"
PAPER_TRADING   = os.getenv("PAPER_TRADING", "true").lower() in ("1","true","on","yes")

# ========= CONFIG / STATE =========
class StratConfig(BaseModel):
    # Engine / Daten
    symbols: List[str] = ["TQQQ"]    # Multi-Asset fähig (looped)
    symbol: str = "TQQQ"             # primary (für /set symbol=…)
    interval: str = "1h"             # '1h'/'1d' etc.
    lookback_days: int = 365
    data_provider: str = "alpaca"    # "alpaca" (default), "yahoo", "stooq_eod"
    allow_stooq_fallback: bool = True
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 1.5

    # Strategie-Parameter (TV-kompatibel: ohne MACD!)
    rsiLen: int = 12
    rsiLow: float = 0.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Timer / Laufzeit
    poll_minutes: int = 10
    live_enabled: bool = False
    market_hours_only: bool = True

class SymState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_status: str = "idle"

class TimerState(BaseModel):
    enabled: bool = True
    running: bool = False
    poll_minutes: int = 10
    last_run: Optional[str] = None
    next_due: Optional[str] = None
    market_hours_only: bool = True

CONFIG = StratConfig()
STATES: Dict[str, SymState] = {sym: SymState() for sym in CONFIG.symbols}
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None
TIMER = TimerState(poll_minutes=CONFIG.poll_minutes, market_hours_only=CONFIG.market_hours_only)

# ========= Indicators (TV-kompatibel) =========
def _rma(x: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA (Pine ta.rma)"""
    alpha = 1.0 / float(length)
    return x.ewm(alpha=alpha, adjust=False).mean()

def rsi_tv(close: pd.Series, length: int) -> pd.Series:
    """Pine ta.rsi mit Wilder RMA"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ru = _rma(up, length)
    rd = _rma(down, length)
    rs = ru / (rd.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(0.0)

def efi_tv(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """EFI = EMA( Volume * (Close - Close[1]) , length )"""
    raw = volume * (close - close.shift(1))
    return raw.ewm(span=length, adjust=False).mean()

# ========= Data Providers =========
from urllib.parse import quote

def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    stooq_sym = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={quote(stooq_sym)}&i=d"
    try:
        df = pd.read_csv(url)
        if df is None or df.empty: return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if lookback_days > 0:
            df = df.iloc[-lookback_days:]
        df["time"] = df.index.tz_localize("UTC")
        return df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})\
                 .rename(columns=str.lower)
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
    elif interval in {"1d"}:
        tf = TimeFrame(1, TimeFrameUnit.Day)
    elif interval in {"15m"}:
        tf = TimeFrame(15, TimeFrameUnit.Minute)
    elif interval in {"5m"}:
        tf = TimeFrame(5, TimeFrameUnit.Minute)
    else:
        tf = TimeFrame(1, TimeFrameUnit.Hour)

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 30))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        adjustment=None,
        feed="iex",
        limit=10000
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty bars")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except Exception: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df = df.sort_index()
        df["time"] = df.index
        # alpaca df columns already lower-case (open,high,low,close,volume)
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}

    # 1) Alpaca default
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=f"{interval}")
            return df, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")

    # 2) Yahoo
    intraday = interval.lower() in {"1m","2m","5m","15m","30m","60m","90m","1h"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except Exception: pass
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df = df.sort_index()
        df["time"] = df.index
        return df

    last_err = None
    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True, threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = _normalize(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}")
                    return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # 3) Fallback Yahoo 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 120)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # 4) Stooq EOD
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] no data for {symbol} {interval}. last_err={last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Features & Logic =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: SymState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 5:
        return {"action":"none","reason":"not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_val = float(last["rsi"])
    rsi_prev= float(prev["rsi"])
    efi_val = float(last["efi"])
    efi_prev= float(prev["efi"])

    rsi_rising  = rsi_val > rsi_prev
    rsi_falling = rsi_val < rsi_prev
    efi_rising  = efi_val > efi_prev

    entry_cond = (rsi_val > CONFIG.rsiLow) and (rsi_val < CONFIG.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < CONFIG.rsiExit) and rsi_falling

    o = float(last["open"])
    c = float(last["close"])
    ts = last["time"]

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    # bars since entry (approx.)
    bars_in_trade = 0
    if st.entry_time:
        since = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if st.position_size == 0:
        if entry_cond:
            return {"action":"buy","qty":1,"price":o,"time":str(ts),
                    "reason":"rule_entry","sl":sl(o),"tp":tp(o),
                    "rsi":rsi_val,"efi":efi_val}
        else:
            return {"action":"none","reason":"flat_no_entry","rsi":rsi_val,"efi":efi_val}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(st.avg_price)
        cur_tp = tp(st.avg_price)
        hit_sl = c <= cur_sl
        hit_tp = c >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","qty":st.position_size,"price":o,"time":str(ts),
                    "reason":"rsi_exit","rsi":rsi_val,"efi":efi_val}
        if hit_sl:
            return {"action":"sell","qty":st.position_size,"price":cur_sl,"time":str(ts),
                    "reason":"stop_loss","rsi":rsi_val,"efi":efi_val}
        if hit_tp:
            return {"action":"sell","qty":st.position_size,"price":cur_tp,"time":str(ts),
                    "reason":"take_profit","rsi":rsi_val,"efi":efi_val}

        return {"action":"none","reason":"hold","rsi":rsi_val,"efi":efi_val}

# ========= Market Hours (NYSE approx) =========
US_HOLIDAYS_2025 = {
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25"
}
def is_us_holiday(dt_utc: datetime) -> bool:
    d = dt_utc.astimezone(timezone(timedelta(hours=-5)))  # EST/EDT grob: -5 (keine DST-Perfektion)
    return d.strftime("%Y-%m-%d") in US_HOLIDAYS_2025

def is_market_open_now(dt_utc: Optional[datetime]=None) -> bool:
    dt_utc = dt_utc or datetime.now(timezone.utc)
    d_ny = dt_utc.astimezone(timezone(timedelta(hours=-5)))  # grob
    if d_ny.weekday() >= 5:  # Sa/So
        return False
    if is_us_holiday(dt_utc):
        return False
    # 9:30–16:00 NY local (vereinfachte Prüfung, keine DST-Feinheiten)
    h, m = d_ny.hour, d_ny.minute
    mins = h*60 + m
    return (9*60+30) <= mins < (16*60)

# ========= Alpaca Trading Helpers =========
def _alpaca_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=PAPER_TRADING)

def _tif_enum():
    from alpaca.trading.enums import TimeInForce
    return TimeInForce.GTC if ALPACA_TIF=="gtc" else TimeInForce.DAY

def submit_entry_order(symbol: str, sl_px: float, tp_px: float):
    """MARKET BUY mit optionalem BRACKET (TP/SL) je nach sl/tp Parametern."""
    from alpaca.trading.enums import OrderSide, OrderClass
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest

    qty = None; notional = None
    if SIZING_MODE == "notional":
        notional = NOTIONAL_USD
    else:
        qty = SHARES

    order_kwargs = dict(
        symbol=symbol,
        side=OrderSide.BUY,
        time_in_force=_tif_enum(),
    )
    if qty: order_kwargs["qty"] = qty
    if notional: order_kwargs["notional"] = notional

    # BRACKET, falls beide gesetzt
    tp_req = TakeProfitRequest(limit_price=round(tp_px, 4))
    sl_req = StopLossRequest(stop_price=round(sl_px, 4))
    order_kwargs.update(order_class=OrderClass.BRACKET, take_profit=tp_req, stop_loss=sl_req)

    cli = _alpaca_client()
    return cli.submit_order(order_data=MarketOrderRequest(**order_kwargs))

def submit_exit_all(symbol: str):
    """Schließt alle offenen Positionen im Symbol (Market SELL)."""
    cli = _alpaca_client()
    try:
        cli.close_position(symbol)
        return True
    except Exception as e:
        print("close_position error:", e)
        return False

def list_positions():
    try:
        cli = _alpaca_client()
        return cli.get_all_positions()
    except Exception as e:
        print("get_all_positions err:", e)
        return []

def list_orders(status="open", symbols: Optional[List[str]]=None, limit=50):
    try:
        from alpaca.trading.requests import GetOrdersRequest
        cli = _alpaca_client()
        req = GetOrdersRequest(status=status, symbols=symbols, limit=limit)
        return cli.get_orders(filter=req)
    except Exception as e:
        print("get_orders err:", e)
        return []

def cancel_all_orders():
    try:
        cli = _alpaca_client()
        cli.cancel_orders()
        return True
    except Exception as e:
        print("cancel_orders err:", e)
        return False

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text or "(empty)")
    except Exception as e:
        print("send error:", e)

def friendly_note(note: Dict[str,str]) -> str:
    p = note.get("provider","")
    d = note.get("detail","")
    if not p: return ""
    if "Fallback" in p or p.startswith("Stooq"):
        return f"📡 Daten: {p} – {d}"
    return f"📡 Daten: {p} – {d}"

# ========= Strategy run per symbol =========
def build_feats_for(symbol: str, interval: str, days: int):
    df, note = fetch_ohlcv_with_note(symbol, interval, days)
    if df.empty: return df, note, None
    fdf = build_features(df, CONFIG)
    return fdf, note, df

async def run_once_symbol(symbol: str, chat_id: Optional[str]=None):
    st = STATES.setdefault(symbol, SymState())
    fdf, note, raw = build_feats_for(symbol, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        if chat_id: await send(chat_id, f"❌ {symbol}: keine Daten. Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return {"symbol":symbol, "status":"nodata"}

    act = bar_logic(fdf, CONFIG, st)
    st.last_status = f"{act['action']} ({act['reason']})"
    info = f"RSI={act.get('rsi',np.nan):.2f} EFI={act.get('efi',np.nan):.0f}"

    if act["action"]=="buy":
        st.position_size = act["qty"]
        st.avg_price = float(act["price"])
        st.entry_time = act["time"]
        if AUTO_TRADE and APCA_API_KEY_ID and APCA_API_SECRET_KEY and CONFIG.data_provider.lower()=="alpaca" and is_market_open_now():
            try:
                order = submit_entry_order(symbol, sl_px=act["sl"], tp_px=act["tp"])
                oid = getattr(order, "id", "?")
                msg = f"🟢 {symbol} LONG @{st.avg_price:.4f} | SL={act['sl']:.4f} TP={act['tp']:.4f}\n{info}\n🧾 Alpaca Order={oid}"
            except Exception as e:
                msg = f"🟢 {symbol} LONG (paper) @{st.avg_price:.4f} | SL={act['sl']:.4f} TP={act['tp']:.4f}\n{info}\n⚠️ Alpaca Fehler: {e}"
        else:
            msg = f"🟢 {symbol} LONG (sim) @{st.avg_price:.4f} | SL={act['sl']:.4f} TP={act['tp']:.4f}\n{info}"
        if chat_id: await send(chat_id, msg)

    elif act["action"]=="sell" and st.position_size>0:
        exit_px = float(act["price"])
        pnl = (exit_px - st.avg_price)/st.avg_price*100.0
        if AUTO_TRADE and APCA_API_KEY_ID and APCA_API_SECRET_KEY and CONFIG.data_provider.lower()=="alpaca" and is_market_open_now():
            ok = submit_exit_all(symbol)
            extra = " (Alpaca close_position ok)" if ok else " (close_position fehlgeschlagen)"
        else:
            extra = ""
        if chat_id:
            await send(chat_id, f"🔴 {symbol} EXIT @{exit_px:.4f} [{act['reason']}] {extra}\nPnL={pnl:.2f}% | {info}")
        st.position_size = 0; st.avg_price=0.0; st.entry_time=None

    else:
        if chat_id: await send(chat_id, f"ℹ️ {symbol} {st.last_status} | {info}")

    return {"symbol":symbol,"status":st.last_status}

async def run_once_all(chat_id: Optional[str]=None):
    results = []
    for sym in CONFIG.symbols:
        results.append(await run_once_symbol(sym, chat_id=chat_id))
    return results

# ========= Plot helper =========
def make_plot_png(df: pd.DataFrame, symbol: str) -> bytes:
    fig = plt.figure(figsize=(9,5), dpi=150)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.plot(df.index, df["close"])
    ax1.set_title(f"{symbol} Close")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["rsi"], label="RSI")
    ax2.plot(df.index, df["efi"]/ max(1.0, np.nanstd(df["efi"])) , label="EFI (normiert)")
    ax2.axhline(CONFIG.rsiLow,  linestyle="--")
    ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.axhline(CONFIG.rsiExit, linestyle=":")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Verbunden.\n"
        "Befehle:\n"
        "/status /timerstatus /cfg /set key=value … /run /live on|off /bt [tage]\n"
        "/ind [SYM] /sig [SYM] /plot [SYM] /positions /orders /cancel [SYM] /cancel_all /trade on|off"
    )

async def cmd_status(update, context):
    sym = CONFIG.symbol
    st = STATES.get(sym, SymState())
    await update.message.reply_text(
        f"📊 Status\nSymbols: {', '.join(CONFIG.symbols)}  (primär: {sym})\n"
        f"Interval: {CONFIG.interval}  Lookback: {CONFIG.lookback_days}d\n"
        f"Provider: {CONFIG.data_provider}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}  AutoTrade: {'ON' if AUTO_TRADE else 'OFF'}\n"
        f"Pos[{sym}]: size={st.position_size} avg={st.avg_price:.4f} since={st.entry_time}\n"
        f"Timer: {'running' if TIMER.running else 'idle'} next={TIMER.next_due} marketOpen={is_market_open_now()}"
    )

async def cmd_timerstatus(update, context):
    await update.message.reply_text(json.dumps(TIMER.dict(), indent=2))

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1)
    k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc","tp":"tpPerc","samebar":"allowSameBarExit","cooldown":"minBarsInTrade"}
    k = mapping.get(k,k)
    target = CONFIG
    if k=="symbol":
        CONFIG.symbol = v
        if v not in CONFIG.symbols:
            CONFIG.symbols.append(v)
        if v not in STATES: STATES[v]=SymState()
        return f"✓ symbol={v} (Liste: {CONFIG.symbols})"
    if not hasattr(target, k):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(target, k)
    if isinstance(cur, bool):   setattr(target, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):  setattr(target, k, int(float(v)))
    elif isinstance(cur, float):setattr(target, k, float(v))
    elif isinstance(cur, list): setattr(target, k, [s.strip() for s in v.split(",") if s.strip()])
    else:                       setattr(target, k, v)
    if k=="poll_minutes": TIMER.poll_minutes = getattr(target,k)
    if k=="market_hours_only": TIMER.market_hours_only = getattr(target,k)
    return f"✓ {k}={getattr(target,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value …\n"
            "z.B.: /set data_provider=alpaca interval=1h lookback_days=365\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=2 tp=4\n"
            "/set symbols=TQQQ,QQQ /set symbol=TQQQ\n"
            "/set poll_minutes=10 market_hours_only=true"
        )
        return
    msgs=[]
    for a in context.args:
        if "=" not in a: msgs.append(f"❌ Ungültig: {a}"); continue
        msgs.append(set_from_kv(a))
    await update.message.reply_text("\n".join(msgs))

async def cmd_cfg(update, context):
    await update.message.reply_text(json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off")
        return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live={'ON' if on else 'OFF'}")

async def cmd_trade(update, context):
    global AUTO_TRADE
    if not context.args:
        await update.message.reply_text(f"AUTO_TRADE ist {'ON' if AUTO_TRADE else 'OFF'} – /trade on|off")
        return
    on = context.args[0].lower() in ("on","1","true")
    AUTO_TRADE = on
    await update.message.reply_text(f"AUTO_TRADE={'ON' if on else 'OFF'}")

async def cmd_run(update, context):
    sym = CONFIG.symbol
    await run_once_symbol(sym, str(update.effective_chat.id))

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    sym = CONFIG.symbol
    fdf, note, raw = build_feats_for(sym, CONFIG.interval, days)
    if fdf is None or fdf.empty:
        await update.message.reply_text(f"❌ Backtest: keine Daten ({note})")
        return
    pos=0; avg=0.0; eq=1.0; R=[]; ent=ex=0
    for i in range(2,len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val, rsi_prev = row["rsi"], prev["rsi"]
        efi_val, efi_prev = row["efi"], prev["efi"]
        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and (rsi_val>rsi_prev) and (efi_val>efi_prev)
        exitc = (rsi_val<CONFIG.rsiExit) and (rsi_val<rsi_prev)
        if pos==0 and entry:
            pos=1; avg=float(row["open"]); ent+=1
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            price = float(row["close"])
            stop = price<=sl; take = price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                r = (px-avg)/avg
                eq *= (1+r); R.append(r); ex+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean(); pf=(a[a>0].sum())/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(f"📈 BT {sym} {days}d: trades={ent}/{ex} win={win*100:.1f}% PF={pf:.2f} CAGR~{cagr*100:.2f}%\n{friendly_note(note)}")
    else:
        await update.message.reply_text("📉 BT: keine abgeschlossenen Trades.")

async def cmd_ind(update, context):
    sym = CONFIG.symbol if not context.args else context.args[0].upper()
    fdf, note, raw = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        await update.message.reply_text(f"❌ {sym}: keine Daten.")
        return
    last = fdf.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} Indikatoren\nRSI={last['rsi']:.2f}  EFI={last['efi']:.0f}\n{friendly_note(note)}"
    )

async def cmd_sig(update, context):
    sym = CONFIG.symbol if not context.args else context.args[0].upper()
    st = STATES.setdefault(sym, SymState())
    fdf, note, raw = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        await update.message.reply_text(f"❌ {sym}: keine Daten.")
        return
    act = bar_logic(fdf, CONFIG, st)
    await update.message.reply_text(
        f"🧭 {sym} Signal: {act['action']} ({act['reason']})\n"
        f"RSI={act.get('rsi',np.nan):.2f} EFI={act.get('efi',np.nan):.0f}\n{friendly_note(note)}"
    )

async def cmd_plot(update, context):
    sym = CONFIG.symbol if not context.args else context.args[0].upper()
    fdf, note, raw = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        await update.message.reply_text(f"❌ {sym}: keine Daten.")
        return
    png = make_plot_png(fdf.iloc[-300:], sym)
    try:
        await tg_app.bot.send_photo(chat_id=update.effective_chat.id, photo=png, caption=f"{sym} – Close/RSI/EFI\n{friendly_note(note)}")
    except Exception as e:
        await update.message.reply_text(f"Plot-Fehler: {e}")

async def cmd_positions(update, context):
    pos = list_positions()
    if not pos:
        await update.message.reply_text("Keine Alpaca Positionen.")
        return
    lines=["📦 Alpaca Positionen:"]
    for p in pos:
        lines.append(f"- {p.symbol}: qty={p.qty} avg={p.avg_entry_price} current={p.current_price} unrealized_pl={p.unrealized_pl}")
    await update.message.reply_text("\n".join(lines))

async def cmd_orders(update, context):
    orders = list_orders(status="open", symbols=None, limit=50)
    if not orders:
        await update.message.reply_text("Keine offenen Orders.")
        return
    lines=["🧾 Offene Orders:"]
    for o in orders:
        lines.append(f"- {o.symbol} {o.side} {o.qty or o.notional} tif={o.time_in_force} id={o.id} status={o.status}")
    await update.message.reply_text("\n".join(lines))

async def cmd_cancel(update, context):
    sym = CONFIG.symbol if not context.args else context.args[0].upper()
    # Einfach alle Orders canceln (Filter per Symbol oben möglich, SDK limitiert das)
    ok = cancel_all_orders()
    await update.message.reply_text(f"Cancel all orders: {'OK' if ok else 'Fehler'} (SDK-seitig kein Symbol-Filter)")

async def cmd_cancel_all(update, context):
    ok = cancel_all_orders()
    await update.message.reply_text(f"Cancel ALL orders: {'OK' if ok else 'Fehler'}")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Background Timer =========
async def timer_loop():
    global TIMER
    while True:
        try:
            TIMER.running = True
            if not CONFIG.live_enabled:
                TIMER.next_due = None
                await asyncio.sleep(5)
                continue
            # Market hours check
            if CONFIG.market_hours_only and not is_market_open_now():
                TIMER.last_run = datetime.now(timezone.utc).isoformat()
                TIMER.next_due = None
                await asyncio.sleep(30)
                continue

            # Fire one pass for all symbols
            if CHAT_ID:
                await run_once_all(CHAT_ID)
            else:
                await run_once_all(None)

            TIMER.last_run = datetime.now(timezone.utc).isoformat()
            due = datetime.now(timezone.utc) + timedelta(minutes=CONFIG.poll_minutes)
            TIMER.next_due = due.isoformat()

            # Sleep until next run
            await asyncio.sleep(CONFIG.poll_minutes * 60)
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(5)

# ========= Lifespan (Telegram Polling ohne Loop-Konflikte) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED, TIMER
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",        cmd_start))
        tg_app.add_handler(CommandHandler("status",       cmd_status))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("set",          cmd_set))
        tg_app.add_handler(CommandHandler("cfg",          cmd_cfg))
        tg_app.add_handler(CommandHandler("live",         cmd_live))
        tg_app.add_handler(CommandHandler("trade",        cmd_trade))
        tg_app.add_handler(CommandHandler("run",          cmd_run))
        tg_app.add_handler(CommandHandler("bt",           cmd_bt))
        tg_app.add_handler(CommandHandler("ind",          cmd_ind))
        tg_app.add_handler(CommandHandler("sig",          cmd_sig))
        tg_app.add_handler(CommandHandler("plot",         cmd_plot))
        tg_app.add_handler(CommandHandler("positions",    cmd_positions))
        tg_app.add_handler(CommandHandler("orders",       cmd_orders))
        tg_app.add_handler(CommandHandler("cancel",       cmd_cancel))
        tg_app.add_handler(CommandHandler("cancel_all",   cmd_cancel_all))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass

        # Polling
        if not POLLING_STARTED:
            delay=3
            while True:
                try:
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED=True
                    break
                except Conflict as e:
                    print(f"Conflict polling: {e}; retry in {delay}s")
                    await asyncio.sleep(delay)
                    delay=min(60, delay*2)

        # Timer starten
        TIMER.enabled = True
        TIMER.poll_minutes = CONFIG.poll_minutes
        asyncio.create_task(timer_loop())

    except Exception as e:
        print("lifespan init error:", e)
        traceback.print_exc()
    try:
        yield
    finally:
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
        print("shutdown complete")

# ========= FastAPI =========
app = FastAPI(title="TQQQ Strategy + Telegram (Alpaca default)", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "primary": CONFIG.symbol,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "live": CONFIG.live_enabled,
        "timer": TIMER.dict()
    }

@app.get("/tick")
async def tick():
    if CONFIG.market_hours_only and not is_market_open_now():
        return {"ran": False, "reason": "market_closed"}
    await run_once_all(CHAT_ID)
    TIMER.last_run = datetime.now(timezone.utc).isoformat()
    due = datetime.now(timezone.utc) + timedelta(minutes=CONFIG.poll_minutes)
    TIMER.next_due = due.isoformat()
    return {"ran": True, "next_due": TIMER.next_due}

@app.get("/timerstatus")
async def timerstatus():
    return TIMER.dict()

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v=os.getenv(k); return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "AUTO_TRADE": AUTO_TRADE,
        "SIZING_MODE": SIZING_MODE,
        "NOTIONAL_USD": NOTIONAL_USD,
        "SHARES": SHARES,
        "ALPACA_TIF": ALPACA_TIF,
        "PAPER_TRADING": PAPER_TRADING
    }

@app.head("/")
async def head_root():
    return

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
