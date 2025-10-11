# alpaca_trading_bot.py
import os, json, time, asyncio, traceback, io, base64
from datetime import datetime, timedelta, timezone, date
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Telegram (python-telegram-bot v20.x)
from telegram import Update, InputFile
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")  # ungenutzt im Polling-Modus
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # optional, für Trading-REST unerheblich

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    # Symbole/TF
    symbol: str = "TQQQ"
    interval: str = "1h"         # '1h' oder '1d' oder '15m'
    lookback_days: int = 365

    # Indikatoren (TV-kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0          # wie gewünscht: 0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk / SL/TP (Prozent relativ Einstieg)
    slPerc: float = 1.0
    tpPerc: float = 4.0

    # Engine
    poll_minutes: int = 10
    live_enabled: bool = False
    market_hours_only: bool = True   # Timer nur in Marktzeit

    # Datenquelle
    data_provider: str = "alpaca"    # 'alpaca' (default), 'yahoo', 'stooq_eod'
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Auto-Trading (Paper)
    auto_trade: bool = (os.getenv("AUTO_TRADE", "false").lower() in ("1","true","on","yes"))
    tif: str = os.getenv("ALPACA_TIF", "day").lower()     # 'day'|'gtc'
    sizing_mode: str = os.getenv("SIZING_MODE", "notional").lower()  # 'notional'|'shares'
    notional_usd: float = float(os.getenv("NOTIONAL_USD", "1000"))
    shares: int = int(float(os.getenv("SHARES", "1")))

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_action_bar_index: Optional[int] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ===== Timer-State =====
TIMER = {
    "enabled": True,                 # interner Timer AN
    "running": False,
    "poll_minutes": 10,
    "last_run": None,
    "next_due": None,
    "market_hours_only": True
}

# ========= US Market Hours + Holidays =========
import pytz
NY = pytz.timezone("America/New_York")

# Minimaler NYSE-Feiertags-Kalender (2024–2026) – observed
US_HOLIDAYS = {
    # 2024
    date(2024,1,1),  date(2024,1,15), date(2024,2,19),
    date(2024,3,29), date(2024,5,27), date(2024,6,19),
    date(2024,7,4),  date(2024,9,2),  date(2024,11,28),
    date(2024,12,25),
    # 2025
    date(2025,1,1),  date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19),
    date(2025,7,4),  date(2025,9,1),  date(2025,11,27),
    date(2025,12,25),
    # 2026
    date(2026,1,1),  date(2026,1,19), date(2026,2,16),
    date(2026,4,3),  date(2026,5,25), date(2026,6,19),
    date(2026,7,3),  date(2026,9,7),  date(2026,11,26),
    date(2026,12,25),
}

def is_us_market_open(now_utc: Optional[datetime] = None) -> bool:
    """Einfache NYSE Öffnungszeit (09:30–16:00 ET, Mo–Fr, ohne Holidays)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY)
    if now_ny.date() in US_HOLIDAYS:
        return False
    if now_ny.weekday() > 4:  # 0=Mon..6=Sun
        return False
    open_t  = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_ny <= close_t

# ========= Indicators (TV-kompatibel) =========
def ema_tv(s: pd.Series, length: int) -> pd.Series:
    """EMA wie in TV: alpha = 2/(len+1)"""
    return s.ewm(alpha=2/(length+1.0), adjust=False).mean()

def rsi_tv(close: pd.Series, length: int) -> pd.Series:
    """TV RSI (Wilder Glättung)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Wilder RMA = EMA mit alpha=1/len
    alpha = 1.0 / float(length)
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def efi_tv(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """EFI wie TV: EMA(length) auf raw = volume * (close - close[1])."""
    raw = volume * (close - close.shift(1))
    return ema_tv(raw, length)

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
        return df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
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
        feed="iex",
        limit=10000
    )

    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty frame")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0)
            except Exception:
                pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df = df.sort_index()
        df["time"] = df.index
        # Normiere Spaltennamen (Alpaca liefert bereits open/high/low/close/volume)
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_yahoo(symbol: str, interval: str, lookback_days: int, retries: int, backoff: float) -> pd.DataFrame:
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}
    is_intraday = interval in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1)
            except Exception:
                pass
        df = df.rename(columns=str.lower)
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
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
        for attempt in range(1, retries+1):
            try:
                if method == "download":
                    tmp = yf.download(**kwargs)
                else:
                    tmp = yf.Ticker(symbol).history(**kwargs)
                df = _norm(tmp)
                if not df.empty:
                    return df
            except Exception as e:
                last_err = e
            time.sleep(backoff * (2**(attempt-1)))
    # Fallback 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty:
            return df
    except Exception as e:
        last_err = e
    print("[yahoo] failed:", last_err)
    return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}

    # 1) Alpaca (default)
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=interval)
            return df, note
        note.update(provider="Alpaca → Yahoo", detail="alpaca leer; versuche Yahoo")

    # 2) Yahoo
    if CONFIG.data_provider.lower() in ("yahoo","alpaca"):
        df = fetch_yahoo(symbol, interval, lookback_days, CONFIG.yahoo_retries, CONFIG.yahoo_backoff_sec)
        if not df.empty:
            note.update(provider="Yahoo", detail=interval)
            return df, note
        note.update(provider="Yahoo → Stooq", detail="yahoo leer; versuche Stooq")

    # 3) Stooq (EOD)
    df = fetch_stooq_daily(symbol, max(lookback_days, 120))
    if not df.empty:
        note.update(provider="Stooq EOD (Fallback)", detail="1d")
        return df, note

    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Features & Strategy =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_val = float(last["rsi"])
    rsi_rising  = rsi_val > float(prev["rsi"])
    rsi_falling = rsi_val < float(prev["rsi"])
    efi_rising  = float(last["efi"]) > float(prev["efi"])

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price = float(last["close"])
    o = float(last["open"])
    ts = last["time"]

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    # Bars since entry (approx.)
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)
    else:
        bars_in_trade = 0

    if st.position_size == 0:
        if entry_cond:
            return {"action":"buy","qty":1,"price":o,"time":str(ts),"reason":"rule_entry",
                    "sl":sl(o),"tp":tp(o),
                    "ind": {"rsi":rsi_val,"efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        else:
            return {"action":"none","reason":"flat_no_entry",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
    else:
        cur_sl = sl(st.avg_price)
        cur_tp = tp(st.avg_price)
        hit_sl = price <= cur_sl
        hit_tp = price >= cur_tp
        if exit_cond:
            return {"action":"sell","qty":st.position_size,"price":o,"time":str(ts),"reason":"rsi_exit",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        if hit_sl:
            return {"action":"sell","qty":st.position_size,"price":cur_sl,"time":str(ts),"reason":"stop_loss",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        if hit_tp:
            return {"action":"sell","qty":st.position_size,"price":cur_tp,"time":str(ts),"reason":"take_profit",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        return {"action":"none","reason":"hold",
                "ind":{"rsi":rsi_val,"efi":float(last['efi']),"efi_prev":float(prev['efi'])}}

# ========= Alpaca Trading (Paper) =========
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, CancelOrdersRequest, TakeProfitRequest, StopLossRequest

def _tif_enum(tif: str) -> TimeInForce:
    return TimeInForce.GTC if (tif or "day").lower()=="gtc" else TimeInForce.DAY

def _trading_client() -> Optional[TradingClient]:
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] keys missing for trading")
        return None
    try:
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] TradingClient error:", e)
        return None

def place_bracket_buy(symbol: str, entry_px: float, sl_px: float, tp_px: float, cfg: StratConfig) -> str:
    tc = _trading_client()
    if tc is None: return "alpaca_client_none"
    try:
        kwargs = dict(
            symbol=symbol,
            side=OrderSide.BUY,
            time_in_force=_tif_enum(cfg.tif),
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(tp_px,4)),
            stop_loss=StopLossRequest(stop_price=round(sl_px,4))
        )
        if cfg.sizing_mode == "shares":
            kwargs["qty"] = max(1, int(cfg.shares))
        else:
            kwargs["notional"] = float(cfg.notional_usd)
        req = MarketOrderRequest(**kwargs)
        order = tc.submit_order(req)
        return f"ok:{order.id}"
    except Exception as e:
        print("[alpaca] buy error:", e)
        return f"err:{e}"

def place_market_sell(symbol: str, qty_shares: Optional[int], cfg: StratConfig) -> str:
    tc = _trading_client()
    if tc is None: return "alpaca_client_none"
    try:
        if not qty_shares or qty_shares <= 0:
            poss = tc.get_all_positions()
            qty_shares = 0
            for p in poss:
                if p.symbol.upper()==symbol.upper():
                    qty_shares = int(float(p.qty))
                    break
        if qty_shares <= 0:
            return "no_position"
        req = MarketOrderRequest(
            symbol=symbol, qty=qty_shares, side=OrderSide.SELL, time_in_force=_tif_enum(cfg.tif)
        )
        order = tc.submit_order(req)
        return f"ok:{order.id}"
    except Exception as e:
        print("[alpaca] sell error:", e)
        return f"err:{e}"

def list_recent_orders(limit: int = 20):
    tc = _trading_client()
    if tc is None: return "alpaca_client_none", []
    try:
        req = GetOrdersRequest(limit=limit)
        orders = tc.get_orders(req)
        out=[]
        for o in orders:
            out.append({
                "id": o.id, "symbol": o.symbol, "side": str(o.side),
                "qty": float(o.qty) if o.qty else None,
                "notional": float(o.notional) if o.notional else None,
                "status": str(o.status), "submitted_at": str(o.submitted_at)
            })
        return "ok", out
    except Exception as e:
        return f"err:{e}", []

def cancel_open_orders(symbol: Optional[str]=None) -> str:
    tc = _trading_client()
    if tc is None: return "alpaca_client_none"
    try:
        if symbol:
            status, orders = list_recent_orders(limit=200)
            if status!="ok": return status
            cnt=0
            for o in orders:
                if o["symbol"].upper()==symbol.upper() and o["status"] in ("new","accepted","pending_new"):
                    tc.cancel_order_by_id(o["id"]); cnt+=1
            return f"ok:{cnt}"
        else:
            tc.cancel_orders(CancelOrdersRequest())
            return "ok:all"
    except Exception as e:
        return f"err:{e}"

def fetch_positions_text() -> str:
    tc = _trading_client()
    if tc is None: return "alpaca_client_none"
    try:
        poss = tc.get_all_positions()
        if not poss:
            return "📭 Keine offenen Positionen."
        lines=["📌 Alpaca Positionen:"]
        for p in poss:
            lines.append(f"- {p.symbol}: qty={p.qty} avg={p.avg_entry_price} market={p.current_price} "
                         f"unrealized={p.unrealized_pl} ({p.unrealized_plpc})")
        return "\n".join(lines)
    except Exception as e:
        return f"err:{e}"

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text if text.strip() else "ℹ️ (leer)")
    except Exception as e:
        print("send error:", e)

# ========= Core run & reporting =========
def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if prov.startswith("Alpaca"): return f"📡 Daten: {prov} ({det})"
    if prov.startswith("Yahoo"):  return f"📡 Daten: {prov} ({det})"
    if "Stooq" in prov:          return f"📡 Daten: {prov} – {det} (nur Daily)"
    return f"📡 Daten: {prov} – {det}"

def _format_indicators(last_row: pd.Series, prev_row: pd.Series) -> str:
    rsi_v = float(last_row["rsi"]); efi_v = float(last_row["efi"]); efi_p = float(prev_row["efi"])
    efi_ch = efi_v - efi_p
    return (f"RSI({CONFIG.rsiLen})={rsi_v:.2f}  |  "
            f"EFI({CONFIG.efiLen})={efi_v:.2f} (Δ {efi_ch:+.2f})  |  close={float(last_row['close']):.4f}")

def _plot_chart(fdf: pd.DataFrame, title: str = "") -> bytes:
    """Erzeugt PNG mit Preis, RSI, EFI (letzte 120 Bars)."""
    D = fdf.iloc[-min(len(fdf), 120):].copy()
    fig = plt.figure(figsize=(10,6))
    # Preis
    ax1 = fig.add_subplot(311)
    ax1.plot(D["time"], D["close"])
    ax1.set_title(title or f"{CONFIG.symbol} ({CONFIG.interval}) – Close")
    ax1.grid(True, alpha=0.3)

    # RSI
    ax2 = fig.add_subplot(312)
    ax2.plot(D["time"], D["rsi"])
    ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.axhline(CONFIG.rsiLow,  linestyle="--")
    ax2.set_title(f"RSI({CONFIG.rsiLen})")
    ax2.grid(True, alpha=0.3)

    # EFI
    ax3 = fig.add_subplot(313)
    ax3.plot(D["time"], D["efi"])
    ax3.axhline(0.0, linestyle="--")
    ax3.set_title(f"EFI({CONFIG.efiLen})")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

async def run_once_and_report(chat_id: str, send_every_time_indicators: bool = False):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten: {CONFIG.symbol} ({CONFIG.interval}) – {note.get('provider','?')} {note.get('detail','')}")
        return

    note_msg = _friendly_data_note(note)
    if note_msg and ( "Fallback" in note_msg or CONFIG.data_provider!="alpaca"):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    last, prev = fdf.iloc[-1], fdf.iloc[-2]
    ind_text = _format_indicators(last, prev)

    # Indikator-Status IMMER im Timer versenden (wie gewünscht) oder nur bei /run
    if send_every_time_indicators:
        await send(chat_id, f"🧭 {ind_text}")

    # Signal/Trades
    if act["action"] == "buy":
        STATE.position_size = act["qty"]
        STATE.avg_price     = float(act["price"])
        STATE.entry_time    = act["time"]
        await send(chat_id, f"🟢 LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f}  TP={act['tp']:.4f}\n{ind_text}")

        if CONFIG.auto_trade:
            res = place_bracket_buy(CONFIG.symbol, STATE.avg_price, act["sl"], act["tp"], CONFIG)
            await send(chat_id, f"📬 Alpaca BUY (Bracket): {res}")

    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / max(1e-12, STATE.avg_price)
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\nPnL={pnl*100:.2f}%\n{ind_text}")

        if CONFIG.auto_trade:
            res = place_market_sell(CONFIG.symbol, STATE.position_size, CONFIG)
            await send(chat_id, f"📬 Alpaca SELL: {res}")

        STATE.position_size = 0
        STATE.avg_price     = 0.0
        STATE.entry_time    = None
    else:
        # Nur Status, falls nicht im Timer schon gesendet
        if not send_every_time_indicators:
            await send(chat_id, f"ℹ️ {STATE.last_status}\n{ind_text}")

# ========= Telegram Handlers =========
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/help – Liste aller Befehle\n"
        "/status – Status & Quelle\n"
        "/cfg – Konfiguration\n"
        "/set key=value …  (z.B. /set interval=1h rsiHigh=68 slPerc=1 tpPerc=4 sizing_mode=notional notional_usd=1000)\n"
        "/run – jetzt einen Zyklus ausführen\n"
        "/live on|off – Live-Schalter (Signal/Trade-Engine)\n"
        "/timerstatus – interner Timer-Status\n"
        "/sig – letztes Signal (Entry/Exit) & Indikatoren\n"
        "/ind – aktuelle RSI/EFI-Werte\n"
        "/plot – Chart (Preis+RSI+EFI)\n"
        "/bt [Tage] – Backtest (EoB, simpel)\n"
        "/trade on|off – Auto-Paper-Trading umschalten\n"
        "/orders – letzte Orders\n"
        "/cancel [SYMBOL] – offene Orders stornieren\n"
        "/positions – Alpaca Positionen\n"
        "/dump – Debug/ENV/Timer\n"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbol: {CONFIG.symbol}  TF: {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}  AutoTrade: {'ON' if CONFIG.auto_trade else 'OFF'}\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Datenquelle: {CONFIG.data_provider}\n"
        f"Timer: {'ON' if TIMER['enabled'] else 'OFF'}  running={TIMER['running']}  "
        f"last={TIMER['last_run']}  next={TIMER['next_due']}\n"
        f"NYSE Markt offen: {'JA' if is_us_market_open() else 'NEIN'}"
    )

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1)
    k = k.strip(); v = v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k, k)
    if not hasattr(CONFIG, k):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    try:
        if isinstance(cur, bool):
            setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
        elif isinstance(cur, int):
            setattr(CONFIG, k, int(float(v)))
        elif isinstance(cur, float):
            setattr(CONFIG, k, float(v))
        else:
            setattr(CONFIG, k, v)
        # Timer-Spiegel
        if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
        if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
        return f"✓ {k} = {getattr(CONFIG,k)}"
    except Exception as e:
        return f"❌ {k}: {e}"

async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set interval=1h lookback_days=365\n"
            "/set rsiLen=12 rsiLow=0 rsiHigh=68 rsiExit=48\n"
            "/set slPerc=1 tpPerc=4\n"
            "/set data_provider=alpaca  (oder yahoo)\n"
            "/set sizing_mode=notional notional_usd=1000  (oder: sizing_mode=shares shares=10)\n"
            "/set tif=day  (oder gtc)"
        )
        return
    msgs, errors = [], []
    for a in context.args:
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errors.append(f"❌ Ungültig: {a}")
            continue
        msgs.append(set_from_kv(a))
    out = []
    if msgs: out += ["✅ Übernommen:"] + msgs
    if errors: out += ["\n⚠️ Probleme:"] + errors
    await update.message.reply_text("\n".join(out).strip())

async def cmd_cfg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off")
        return
    CONFIG.live_enabled = context.args[0].lower() in ("on","1","true","start")
    await update.message.reply_text(f"Live = {'ON' if CONFIG.live_enabled else 'OFF'}")

async def cmd_timerstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(json.dumps({
        "enabled": TIMER["enabled"],
        "running": TIMER["running"],
        "poll_minutes": TIMER["poll_minutes"],
        "last_run": TIMER["last_run"],
        "next_due": TIMER["next_due"],
        "market_hours_only": TIMER["market_hours_only"]
    }, indent=2))

async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await run_once_and_report(str(update.effective_chat.id), send_every_time_indicators=False)

async def cmd_sig(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    last, prev = fdf.iloc[-1], fdf.iloc[-2]
    ind_text = _format_indicators(last, prev)
    await update.message.reply_text(f"Signal: {act['action']} ({act['reason']})\n{ind_text}")

async def cmd_ind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df, _ = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    last, prev = fdf.iloc[-1], fdf.iloc[-2]
    await update.message.reply_text(_format_indicators(last, prev))

async def cmd_plot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df, _ = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    png = _plot_chart(fdf, title=f"{CONFIG.symbol} {CONFIG.interval}")
    await tg_app.bot.send_photo(chat_id=update.effective_chat.id, photo=png)

async def cmd_bt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    days = 180
    if context.args:
        try: days=int(context.args[0])
        except: pass
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten ({note})")
        return
    fdf = build_features(df, CONFIG)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = float(row["rsi"])
        rsi_rising  = rsi_val > float(prev["rsi"])
        rsi_falling = rsi_val < float(prev["rsi"])
        efi_rising  = float(row["efi"]) > float(prev["efi"])
        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and rsi_rising and efi_rising
        exitc = (rsi_val<CONFIG.rsiExit) and rsi_falling
        if pos==0 and entry:
            pos=1; avg=float(row["open"]); entries+=1
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            price=float(row["close"])
            stop=price<=sl; take=price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                r  = (px-avg)/avg
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean()
        pf=(a[a>0].sum())/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(
            f"📈 Backtest {days}d  Trades: {entries}/{exits}\n"
            f"WinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"AUTO_TRADE = {'ON' if CONFIG.auto_trade else 'OFF'}\nNutze: /trade on|off")
        return
    CONFIG.auto_trade = context.args[0].lower() in ("on","1","true","start")
    await update.message.reply_text(f"AUTO_TRADE = {'ON' if CONFIG.auto_trade else 'OFF'}")

async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status, orders = list_recent_orders(limit=30)
    if status!="ok":
        await update.message.reply_text(f"❌ Orders: {status}")
        return
    if not orders:
        await update.message.reply_text("📭 Keine Orders gefunden.")
        return
    lines=["🧾 Letzte Orders:"]
    for o in orders:
        qty = o["qty"] if o["qty"] is not None else o["notional"]
        lines.append(f"- {o['symbol']} {o['side']} {qty}  [{o['status']}]  {o['submitted_at']}")
    await update.message.reply_text("\n".join(lines))

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sym=None
    if context.args: sym=context.args[0].upper()
    res = cancel_open_orders(sym)
    await update.message.reply_text(f"🧹 Cancel result: {res}")

async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = fetch_positions_text()
    await update.message.reply_text(txt)

async def cmd_dump(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payload = {
        "config": CONFIG.dict(),
        "state": STATE.dict(),
        "timer": TIMER,
        "env": {
            "APCA_API_KEY_ID": bool(APCA_API_KEY_ID),
            "APCA_API_SECRET_KEY": bool(APCA_API_SECRET_KEY),
            "DEFAULT_CHAT_ID_present": bool(DEFAULT_CHAT_ID),
        },
        "market_open_now": is_us_market_open()
    }
    await update.message.reply_text(json.dumps(payload, indent=2, default=str))

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unbekannter Befehl. /help für Hilfe")

# ========= Background Timer Task =========
async def _timer_loop():
    TIMER["running"] = True
    try:
        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)
            if CONFIG.live_enabled and CHAT_ID:
                market_ok = True
                if CONFIG.market_hours_only:
                    market_ok = is_us_market_open(now)
                if market_ok:
                    await run_once_and_report(CHAT_ID, send_every_time_indicators=True)
                    TIMER["last_run"] = datetime.now(timezone.utc).isoformat()
                else:
                    # Still update next_due even if not running in off-hours
                    pass
            # schedule next
            next_dt = datetime.now(timezone.utc) + timedelta(minutes=CONFIG.poll_minutes)
            TIMER["next_due"] = next_dt.isoformat()
            await asyncio.sleep(CONFIG.poll_minutes * 60)
    except asyncio.CancelledError:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        TIMER["running"] = False

_timer_task: Optional[asyncio.Task] = None

# ========= Lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED, _timer_task
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("help",    cmd_help))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("trade",   cmd_trade))
        tg_app.add_handler(CommandHandler("orders",  cmd_orders))
        tg_app.add_handler(CommandHandler("cancel",  cmd_cancel))
        tg_app.add_handler(CommandHandler("positions", cmd_positions))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass

        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED = True
                    break
                except Conflict as e:
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc()
                    await asyncio.sleep(10)

        tg_running = True

        # Timer vorbereiten
        TIMER["enabled"] = True
        TIMER["poll_minutes"] = CONFIG.poll_minutes
        TIMER["market_hours_only"] = CONFIG.market_hours_only
        if _timer_task is None or _timer_task.done():
            _timer_task = asyncio.create_task(_timer_loop())

    except Exception as e:
        print("❌ Telegram/Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        try:
            if _timer_task and not _timer_task.done():
                _timer_task.cancel()
                try: await _timer_task
                except: pass
        except Exception:
            pass
        try: await tg_app.updater.stop()
        except Exception: pass
        try: await tg_app.stop()
        except Exception: pass
        try: await tg_app.shutdown()
        except Exception: pass

# ========= FastAPI =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "last_run": TIMER["last_run"],
            "next_due": TIMER["next_due"],
            "market_hours_only": TIMER["market_hours_only"]
        }
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
    if CONFIG.market_hours_only and not is_us_market_open():
        return {"ran": False, "reason": "market_closed"}
    await run_once_and_report(CHAT_ID, send_every_time_indicators=True)
    return {"ran": True}

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v = os.getenv(k); return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_API_BASE_URL": chk("APCA_API_BASE_URL"),
    }

@app.get("/tgstatus")
def tgstatus():
    return {"tg_running": tg_running, "polling_started": POLLING_STARTED}
