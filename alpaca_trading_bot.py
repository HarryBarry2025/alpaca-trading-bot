# alpaca_trading_bot.py  —  V5.1.2
import os, io, json, math, time, asyncio, traceback, warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram
from telegram import Update, InputFile
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict

warnings.filterwarnings("ignore", category=UserWarning)

# ========= ENV & PATHS =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None
if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "iex").lower()  # 'sip' oder 'iex'

ENV_ENABLE_TRADE      = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER      = os.getenv("ENABLE_TIMER", "true").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES      = int(os.getenv("POLL_MINUTES", "60"))
ENV_MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

# Persistenz (PDT / Logs)
def ensure_writable_dir(preferred: str, fallback: str = "./data") -> Path:
    for d in [preferred, fallback]:
        p = Path(d)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".touch"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    return Path("./")


DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/appdata"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDT_FILE = DATA_DIR / "pdt_trades.json"

# ========= Helpers / Time =========

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def next_half_hour_utc(dt: datetime) -> datetime:
    # anchor at :30; if before :30 → same hour :30, otherwise next hour :30
    base = dt.replace(second=0, microsecond=0)
    if base.minute < 30:
        return base.replace(minute=30)
    return (base + timedelta(hours=1)).replace(minute=30)

def build_tv_1h_from_1m(df_1m: pd.DataFrame, rth_only: bool = True) -> pd.DataFrame:
    """
    Convert 1-minute bars to TV-like 1h bars:
    - Bars close at :30 UTC (9:30 ET open → 13:30 UTC)
    - Drop the last incomplete hour
    - Optionally restrict to RTH (Mo–Fr, 13:30–20:00 UTC closes)
    """
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()

    m = df_1m.copy()
    m = m.sort_index()

    # ensure datetime index is UTC
    if not isinstance(m.index, pd.DatetimeIndex):
        m.index = pd.to_datetime(m.index, utc=True)
    elif m.index.tz is None:
        m.index = m.index.tz_localize("UTC")
    else:
        m.index = m.index.tz_convert("UTC")

    # shift -30 min to make 1H bins close exactly on :30
    m.index = m.index - pd.Timedelta(minutes=30)

    # resample to 1H, right-closed/right-labeled
    h = m.resample("1H", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

    # shift index back to true close times (on :30)
    h.index = h.index + pd.Timedelta(minutes=30)

    # drop last incomplete bar: only keep bars <= last completed :30
    end = now_utc()
    last_complete_close = next_half_hour_utc(end) - timedelta(hours=1)
    h = h[h.index <= last_complete_close]

    # optional: RTH only (Mon–Fri, 13:30–20:00 UTC)
    if rth_only and not h.empty:
        hhmm = h.index.hour * 60 + h.index.minute
        mask = (
            (h.index.weekday < 5) &
            (hhmm >= (13*60 + 30)) &
            (hhmm <= (20*60))  # bar close 20:00 UTC
        )
        h = h[mask]

    if h.empty:
        return h

    h["time"] = h.index
    return h

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()

US_HOLIDAYS = {
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27",
    "2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25",
}

def next_business_day(dt: datetime) -> datetime:
    d = dt
    while True:
        d = (d + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in US_HOLIDAYS:
            return d

def in_rth_utc(dt: datetime) -> bool:
    if dt.weekday() >= 5: return False
    if dt.strftime("%Y-%m-%d") in US_HOLIDAYS: return False
    m = dt.hour*60 + dt.minute
    return (13*60+30) <= m <= (20*60)

def next_rth_halfhour(now: datetime, anchor_minute: int = 30) -> datetime:
    if now.weekday() >= 5 or now.strftime("%Y-%m-%d") in US_HOLIDAYS:
        d0 = next_business_day(now)
        return d0.replace(hour=13, minute=anchor_minute, second=2, microsecond=0)
    candidate = now.replace(minute=anchor_minute, second=2, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(hours=1)
    rth_start = now.replace(hour=13, minute=anchor_minute, second=2, microsecond=0)
    rth_end   = now.replace(hour=20, minute=0, second=0, microsecond=0)
    if candidate < rth_start: candidate = rth_start
    if candidate > rth_end:
        d0 = next_business_day(now)
        candidate = d0.replace(hour=13, minute=anchor_minute, second=2, microsecond=0)
    return candidate

# ========= Config / State =========
class SizerConfig(BaseModel):
    sizing_mode: str = "shares"             # shares|percent_equity|notional_usd|risk
    sizing_value: float = 1.0               # je nach Modus
    max_position_pct: float = 100.0         # begrenzt Positionsgröße vom Equity

class StratConfig(BaseModel):
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"     # 1h oder 1d (1h wird TV-synchron aggregiert)
    lookback_days: int = 365

    rsiLen: int = 12
    rsiLow: float = 0.0      # Default = 0 (gewünscht)
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = True
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY
    sync_hourly_at_minute: int = 30          # :30 UTC-Anchor (TV-kompatibel)

    data_provider: str = "alpaca"            # alpaca|yahoo|stooq_eod
    yahoo_retries: int = 2
    yahoo_backoff_sec: float = 1.0
    allow_stooq_fallback: bool = True

    trade_enabled: bool = ENV_ENABLE_TRADE
    sizer: SizerConfig = SizerConfig()

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}
    last_status: str = "idle"

# PDT persistent store
class PdtStore(BaseModel):
    trades: List[str] = []          # ISO timestamps of daytrades (sell same day as buy)
    hard_block: bool = False        # when >=3 trades last 5 biz days
    max_daytrades: int = 3

CONFIG = StratConfig()
STATE  = StratState()
PDT    = PdtStore()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= Indicators (TV) =========
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi_wilder(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / length
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

def efi_tv(close: pd.Series, vol: pd.Series, length: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, length)

# ========= Alpaca Data =========
def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int):
    interval = interval.lower()
    if interval in ("1h","60m"):
        # Pull fresh 1m and resample to TV-style 1h
        m = fetch_alpaca_1m(symbol, lookback_minutes=lookback_days*24*60 // 3)  # e.g., ~1/3 of minutes
        if not m.empty:
            h = build_tv_1h_from_1m(m, rth_only=CONFIG.market_hours_only)
            if not h.empty:
                return h, {"provider": f"Alpaca ({os.getenv('APCA_DATA_FEED','sip')})", "detail": "1m→1h TV-sync"}
        # fallbacks (Yahoo/Stooq) if needed...
        # ...
    else:
        # your existing “direct 1d” path
        # ...
        pass


def alpaca_data_client():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    except Exception as e:
        print("[alpaca] data client error:", e)
        return None

def fetch_alpaca_1m(symbol: str, lookback_minutes: int = 4000) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] 1m library not available:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing API key/secret")
        return pd.DataFrame()

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

    end = now_utc()
    start = end - timedelta(minutes=lookback_minutes)
    tf = TimeFrame(1, TimeFrameUnit.Minute)

    feed = os.getenv("APCA_DATA_FEED", "sip").lower()  # 'sip' if you have Algo+; else 'iex'
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=feed,
            limit=100000
        )
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] 1m empty frame")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.rename(columns=str.lower).sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = df.index.tz_convert("UTC")
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] 1m fetch failed:", e)
        return pd.DataFrame()

def resample_1h_tv(df_1m: pd.DataFrame, rth_only: bool = True, anchor_minute: int = 30,
                   drop_last_incomplete: bool = True) -> pd.DataFrame:
    if df_1m.empty: return df_1m
    d = df_1m.copy()
    # RTH filter (13:30–20:00 UTC)
    if rth_only:
        m = d.index.map(lambda t: t.hour*60 + t.minute)
        mask = (m >= 13*60+30) & (m <= 20*60)
        d = d[mask]
    # Anchor zu :30: schiebe 30 Minuten rückwärts, resample, schiebe zurück
    d = d.tz_convert("UTC")
    shifted = d.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=anchor_minute)
    o = shifted["open"].resample("1H").first()
    h = shifted["high"].resample("1H").max()
    l = shifted["low"].resample("1H").min()
    c = shifted["close"].resample("1H").last()
    v = shifted["volume"].resample("1H").sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    out.index = out.index + pd.Timedelta(minutes=anchor_minute)
    out.index.name = d.index.name
    out = out.dropna()
    if drop_last_incomplete:
        last_stamp = out.index[-1]
        # letzte fertige Kerze endet exakt auf :30 (z.B. 15:30, 16:30, …)
        # wenn aktuelle Zeit kleiner als this close + 1h → drop
        if last_stamp > (now_utc() - timedelta(minutes=59)):
            out = out.iloc[:-1] if len(out) > 0 else out
    out["time"] = out.index
    return out

# Yahoo/Stooq fallback
from urllib.parse import quote

def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    try:
        url = f"https://stooq.com/q/d/l/?s={quote(symbol.lower())}.us&i=d"
        df = pd.read_csv(url)
        if df is None or df.empty: return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if lookback_days > 0: df = df.iloc[-lookback_days:]
        df["time"] = df.index.tz_localize("UTC")
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[stooq] fetch failed:", e); return pd.DataFrame()

def fetch_yahoo(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    intraday = interval.lower() in {"1m","2m","5m","15m","30m","60m","90m","1h"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"
    def _norm(tmp):
        if tmp is None or len(tmp)==0: return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except Exception: pass
        if isinstance(df.index, pd.DatetimeIndex):
            try: df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            except Exception: pass
        df = df.sort_index()
        df["time"] = df.index
        return df
    last_err = None
    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=True, threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    for method, kw in tries:
        for attempt in range(1, 1+CONFIG.yahoo_retries):
            try:
                tmp = yf.download(**kw) if method=="download" else yf.Ticker(symbol).history(**kw)
                df = _norm(tmp)
                if not df.empty: return df
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,60)}d", auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty: return df
    except Exception as e:
        last_err = e
    print("[yahoo] empty. last_err:", last_err)
    return pd.DataFrame()

def fetch_ohlcv_tv_sync(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"", "detail":""}
    if CONFIG.data_provider.lower() == "alpaca":
        # 1) 1m vom Alpaca + resample
        m1 = fetch_alpaca_1m(symbol, lookback_days, APCA_DATA_FEED)
        if not m1.empty:
            if interval.lower() in ("1h","60m"):
                df = resample_1h_tv(m1, rth_only=True, anchor_minute=CONFIG.sync_hourly_at_minute,
                                    drop_last_incomplete=True)
                if not df.empty:
                    note.update(provider=f"Alpaca ({APCA_DATA_FEED})", detail="TV 1h (RTH, :30)")
                    return df, note
            else:
                # direkte Nutzung anderer TF nicht vorgesehen -> fallback Yahoo
                pass
        note.update(provider="Alpaca→Fallback", detail="1m leer; versuche Yahoo/Stooq")

    # Fallback: Yahoo 1h + (nicht perfekt TV-synchron), danach Stooq daily
    y = fetch_yahoo(symbol, "60m" if interval.lower() in ("1h","60m") else interval, lookback_days)
    if not y.empty and interval.lower() in ("1h","60m"):
        # erzwinge RTH + :30-Anchor via Neuaggregation
        y1m = fetch_yahoo(symbol, "1m", min(lookback_days, 30))
        if not y1m.empty:
            df = resample_1h_tv(y1m, rth_only=True, anchor_minute=CONFIG.sync_hourly_at_minute,
                                drop_last_incomplete=True)
            if not df.empty:
                note.update(provider="Yahoo (1m→1h TV)", detail="RTH, :30")
                return df, note
    # Stooq EOD
    if CONFIG.allow_stooq_fallback:
        s = fetch_stooq_daily(symbol, lookback_days)
        if not s.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return s, note

    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Features / Signals =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_wilder(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    out["rsi_rising"] = out["rsi"] > out["rsi"].shift(1)
    out["rsi_falling"] = out["rsi"] < out["rsi"].shift(1)
    out["efi_rising"] = out["efi"] > out["efi"].shift(1)
    out["entry_cond"] = (out["rsi"] > cfg.rsiLow) & (out["rsi"] < cfg.rsiHigh) & out["rsi_rising"] & out["efi_rising"]
    out["exit_cond"]  = (out["rsi"] < cfg.rsiExit) & out["rsi_falling"]
    return out

# ========= PDT Persistenz =========
def load_pdt():
    global PDT
    try:
        if PDT_FILE.exists():
            PDT = PdtStore(**json.loads(PDT_FILE.read_text(encoding="utf-8")))
    except Exception as e:
        print("PDT load error:", e)

def save_pdt():
    try:
        PDT_FILE.write_text(json.dumps(PDT.dict(), indent=2), encoding="utf-8")
    except Exception as e:
        print("PDT save error:", e)

def biz_days_ago(dt: datetime, days: int) -> datetime:
    d = dt
    n = 0
    while n < days:
        d = d - timedelta(days=1)
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in US_HOLIDAYS:
            n += 1
    return d

def pdt_window_trades(now: datetime) -> int:
    cutoff = biz_days_ago(now, 5)
    cnt = 0
    for iso in PDT.trades:
        try:
            t = datetime.fromisoformat(iso)
            if t >= cutoff: cnt += 1
        except:
            pass
    return cnt

def pdt_check_and_set_block(now: datetime):
    cnt = pdt_window_trades(now)
    PDT.hard_block = (cnt >= PDT.max_daytrades)
    save_pdt()

def pdt_register_daytrade(now: datetime):
    PDT.trades.append(to_iso(now))
    # keep last ~100
    if len(PDT.trades) > 100:
        PDT.trades = PDT.trades[-100:]
    pdt_check_and_set_block(now)

# ========= Trading (Sizer + Alpaca) =========
def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

def alpaca_account() -> Dict[str,Any]:
    c = alpaca_trading_client()
    if not c: return {}
    try:
        a = c.get_account()
        return {"status": a.status, "equity": float(a.equity), "cash": float(a.cash),
                "buying_power": float(a.buying_power), "multiplier": a.multiplier}
    except Exception as e:
        print("alpaca_account error:", e); return {}

def alpaca_positions() -> List[Dict[str,Any]]:
    c = alpaca_trading_client()
    if not c: return []
    try:
        ps = c.get_all_positions()
        out=[]
        for p in ps:
            out.append({"symbol":p.symbol,"qty":float(p.qty),"avg":float(p.avg_entry_price),
                        "last":float(p.current_price)})
        return out
    except Exception as e:
        print("alpaca_positions error:", e); return []

def size_order(sym: str, price: float, equity: float) -> int:
    s = CONFIG.sizer
    max_dol = equity * (CONFIG.sizer.max_position_pct/100.0)
    if s.sizing_mode == "shares":
        qty = int(max(0, math.floor(s.sizing_value)))
    elif s.sizing_mode == "percent_equity":
        notional = equity * (s.sizing_value/100.0)
        qty = int(max(0, math.floor(notional / max(1e-9, price))))
    elif s.sizing_mode == "notional_usd":
        qty = int(max(0, math.floor(s.sizing_value / max(1e-9, price))))
    elif s.sizing_mode == "risk":
        # sehr einfache Risk-Berechnung: sizing_value = %Risk pro Trade gegen SL
        sl = price * (1 - CONFIG.slPerc/100.0)
        risk_per_share = max(1e-6, price - sl)
        risk_budget = equity * (s.sizing_value/100.0)
        qty = int(max(0, math.floor(risk_budget / risk_per_share)))
    else:
        qty = 1
    # begrenze gegen max_position_pct
    qty = int(min(qty, math.floor(max_dol / max(1e-9, price))))
    return max(qty, 0)

async def place_market_order(sym: str, qty: int, side: str, tif: str = "day") -> str:
    c = alpaca_trading_client()
    if c is None: return "alpaca client not available"
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tif_map = {"day":TimeInForce.DAY,"gtc":TimeInForce.GTC,"opg":TimeInForce.OPG,"cls":TimeInForce.CLS,
                   "ioc":TimeInForce.IOC,"fok":TimeInForce.FOK}
        req = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif_map.get(tif.lower(), TimeInForce.DAY)
        )
        o = c.submit_order(order_data=req)
        return f"order_id={o.id}"
    except Exception as e:
        return f"alpaca order error: {e}"

# ========= Single-step (bar logic) =========
def apply_intrabar_sl_tp(row_open, row_high, row_low, avg, slPerc, tpPerc) -> Tuple[bool, str, float]:
    sl = avg * (1 - slPerc/100.0)
    tp = avg * (1 + tpPerc/100.0)
    hit_sl = row_low <= sl
    hit_tp = row_high >= tp
    if hit_sl and hit_tp:
        # konservativ: zuerst SL getroffen
        return True, "stop_loss", sl
    if hit_sl:
        return True, "stop_loss", sl
    if hit_tp:
        return True, "take_profit", tp
    return False, "", 0.0

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        return {"ok":False, "msg": f"❌ Keine Daten {sym} – {note}"}
    f = build_features(df, CONFIG)
    last, prev = f.iloc[-1], f.iloc[-2]

    pos = STATE.positions.get(sym, {"size":0, "avg":0.0, "entry_time":None})
    size, avg = pos["size"], pos["avg"]

    entry = bool(last["entry_cond"])
    exitc = bool(last["exit_cond"])

    action = {"action":"none","symbol":sym}
    if size == 0:
        if entry and not PDT.hard_block:
            eq = alpaca_account().get("equity", 10000.0)
            qty = size_order(sym, float(last["open"]), float(eq))
            if qty > 0:
                action = {"action":"buy","qty":qty,"px":float(last["open"]),"time":str(last["time"]),
                          "rsi":float(last["rsi"]),"efi":float(last["efi"])}
    else:
        # Intrabar SL/TP gegen Hoch/Tief der fertigen Bar
        hit, why, px = apply_intrabar_sl_tp(float(last["open"]), float(last["high"]), float(last["low"]),
                                            avg, CONFIG.slPerc, CONFIG.tpPerc)
        if hit:
            action = {"action":"sell","qty":size,"px":px,"time":str(last["time"]),"reason":why,
                      "rsi":float(last["rsi"]),"efi":float(last["efi"])}
        elif exitc:
            action = {"action":"sell","qty":size,"px":float(last["open"]),"time":str(last["time"]),"reason":"rsi_exit",
                      "rsi":float(last["rsi"]),"efi":float(last["efi"])}

    # Execute / Notify
    if send_signals and CHAT_ID:
        await tg_send(f"ℹ️ {sym} {CONFIG.interval}  rsi={float(last['rsi']):.2f}  efi={float(last['efi']):.2f}  • provider={note.get('provider','')}")

    if CONFIG.trade_enabled and action["action"] in ("buy","sell") and not PDT.hard_block:
        if CONFIG.market_hours_only and not in_rth_utc(now_utc()):
            if CHAT_ID: await tg_send("⛔ Markt geschlossen – kein Trade.")
        else:
            side = "buy" if action["action"]=="buy" else "sell"
            info = await place_market_order(sym, int(action["qty"]), side, "day")
            if CHAT_ID: await tg_send(f"🛒 {side.upper()} {sym} x{action['qty']} @ {action['px']:.4f} • {info}")

    # local sim position + PDT tracking
    if action["action"]=="buy" and size==0:
        STATE.positions[sym] = {"size":action["qty"],"avg":action["px"],"entry_time":action["time"]}
        if CHAT_ID and send_signals: await tg_send(f"🟢 LONG (sim) {sym} @ {action['px']:.4f}")
    elif action["action"]=="sell" and size>0:
        pnl = (action["px"] - avg)/max(1e-9, avg)
        # PDT: wenn entry/exit am selben Handelstag → count
        try:
            ent = datetime.fromisoformat(pos["entry_time"])
            ex  = datetime.fromisoformat(action["time"])
            if ent.date() == ex.date():
                pdt_register_daytrade(ex)
        except Exception: pass
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if CHAT_ID and send_signals: await tg_send(f"🔴 EXIT (sim) {sym} @ {action['px']:.4f}  {action.get('reason','')}  PnL={pnl*100:.2f}%")

    STATE.last_status = f"{sym}: {action['action']}"
    return {"ok":True,"act":action}

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
POLLING_STARTED = False

async def tg_send(text: str):
    try:
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text if text.strip() else "(leer)")
    except Exception as e:
        print("tg_send error:", e)

async def send_doc_bytes(data: bytes, filename: str, caption: str=""):
    try:
        bio = io.BytesIO(data); bio.name = filename; bio.seek(0)
        await tg_app.bot.send_document(chat_id=CHAT_ID, document=InputFile(bio), caption=caption)
    except Exception as e:
        print("send_doc error:", e)

async def send_png(fig, filename: str, caption: str=""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    await send_doc_bytes(buf.getvalue(), filename, caption)

# ========= Timer (V5.1.2 robust) =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}

def compute_next_due(now: datetime) -> datetime:
    tf = CONFIG.interval.lower()
    if tf in ("1h","60m"):
        return next_rth_halfhour(now, CONFIG.sync_hourly_at_minute)
    return now + timedelta(minutes=TIMER["poll_minutes"])

async def timer_loop():
    TIMER["running"] = True
    try:
        while TIMER["enabled"]:
            now = now_utc()

            # Market-hours gate (optional)
            if TIMER["market_hours_only"] and not is_market_open_now(now):
                await asyncio.sleep(15)
                continue

            # Compute next_due
            if CONFIG.interval.lower() in ("1h", "60m"):
                if TIMER["next_due"] is None:
                    TIMER["next_due"] = next_half_hour_utc(now).isoformat()
                due_dt = datetime.fromisoformat(TIMER["next_due"])
            else:
                if TIMER["next_due"] is None:
                    TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
                due_dt = datetime.fromisoformat(TIMER["next_due"])

            if now >= due_dt:
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                TIMER["last_run"] = now.isoformat()
                if CONFIG.interval.lower() in ("1h","60m"):
                    TIMER["next_due"] = next_half_hour_utc(now + timedelta(seconds=1)).isoformat()
                else:
                    TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()

            await asyncio.sleep(5)
    finally:
        TIMER["running"] = False

TIMER_TASK: Optional[asyncio.Task] = None

# ========= Telegram Commands =========

async def cmd_debugbars(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    tail = df.tail(6)
    lines = [f"{pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%M:%S %Z')} | close={r['close']:.4f}"
             for _, r in tail.iterrows()]
    await update.message.reply_text(
        "🧪 Debug Bars (last 6)\n" + "\n".join(lines) +
        "\n(Barzeiten = Abschlusszeiten; für 1h sollten sie auf :30 UTC enden.)"
    )


async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Verbunden.\n"
        "Befehle:\n"
        "/status, /cfg, /set key=value …\n"
        "/run, /bt [days], /wf [is oos]\n"
        "/sig, /ind, /plot, /dump, /dumpcsv [N]\n"
        "/trade on|off, /pos, /account\n"
        "/timer on|off, /timerstatus, /timerrunnow, /market on|off"
    )

async def cmd_status(update, context):
    acc = alpaca_account()
    pos_lines = [f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}" for s,p in STATE.positions.items()] or ["keine (sim)"]
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider} (feed={APCA_DATA_FEED})\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} • Timer: {'ON' if TIMER['enabled'] else 'OFF'}\n"
        f"Timer next_due={TIMER['next_due']} market_hours_only={TIMER['market_hours_only']}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper)\n"
        f"PDT: trades(last5d)={pdt_window_trades(now_utc())} hard_block={PDT.hard_block}\n"
        f"Account: {json.dumps(acc)}\n"
        f"Sim-Pos:\n" + "\n".join(pos_lines)
    )

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

def set_from_kv(kv: str) -> str:
    k,v = kv.split("=",1); k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k,k)
    if not hasattr(CONFIG, k): return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):   setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):  setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):setattr(CONFIG, k, float(v))
    elif isinstance(cur, list): setattr(CONFIG, k, [x.strip() for x in v.split(",") if x.strip()])
    elif isinstance(cur, SizerConfig):
        # allow /set sizer.sizing_mode=percent_equity  | sizer.sizing_value=50 | sizer.max_position_pct=100
        if "." not in v: return "❌ Nutze sizer.sizing_mode=… etc."
    else:
        setattr(CONFIG, k, v)
    if k == "poll_minutes": TIMER["poll_minutes"] = getattr(CONFIG,k)
    if k == "market_hours_only": TIMER["market_hours_only"] = getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value …\n"
            "z.B.: /set rsiLow=0 rsiHigh=68 rsiExit=48 sl=2 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set data_provider=alpaca\n"
            "/set sizer.sizing_mode=percent_equity\n"
            "/set sizer.sizing_value=50\n"
            "/set sizer.max_position_pct=100\n"
        ); return
    msgs, errs = [], []
    for a in context.args:
        if "=" not in a: errs.append(f"❌ {a}"); continue
        key,val = a.split("=",1)
        key=key.strip(); val=val.strip()
        if key.startswith("sizer."):
            sub = key.split(".",1)[1]
            if hasattr(CONFIG.sizer, sub):
                cur = getattr(CONFIG.sizer, sub)
                try:
                    if isinstance(cur, float): setattr(CONFIG.sizer, sub, float(val))
                    elif isinstance(cur, int): setattr(CONFIG.sizer, sub, int(float(val)))
                    else: setattr(CONFIG.sizer, sub, val)
                    msgs.append(f"✓ sizer.{sub} = {getattr(CONFIG.sizer, sub)}")
                except Exception as e:
                    errs.append(f"❌ {key}: {e}")
            else:
                errs.append(f"❌ unbekannter sizer key: {sub}")
        else:
            try:
                msgs.append(set_from_kv(a))
            except Exception as e:
                errs.append(f"❌ {a}: {e}")
    out = []
    if msgs: out += ["✅ Übernommen:"] + msgs
    if errs: out += ["⚠️ Probleme:"] + errs
    await update.message.reply_text("\n".join(out) if out else "Kein gültiges Paar.")

async def cmd_trade(update, context):
    if not context.args: await update.message.reply_text("Nutze: /trade on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.trade_enabled = on
    await update.message.reply_text(f"Trading = {'ON' if on else 'OFF'}")

async def cmd_pos(update, context):
    ps = alpaca_positions()
    if ps:
        lines = [f"{p['symbol']}: qty={p['qty']}, avg={p['avg']:.4f}, last={p['last']:.4f}" for p in ps]
        await update.message.reply_text("📦 Positionen\n" + "\n".join(lines))
    else:
        await update.message.reply_text("📦 Keine Positionen oder kein Zugriff.")

async def cmd_account(update, context):
    await update.message.reply_text("👤 Account\n" + json.dumps(alpaca_account(), indent=2))

async def cmd_run(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    await update.message.reply_text("✔️ Run done.")

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    sym = CONFIG.symbols[0]
    df, _ = fetch_ohlcv_tv_sync(sym, CONFIG.interval, days)
    if df.empty: await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG)
    eq=1.0; pos=0; avg=0.0; R=[]; entries=exits=0
    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        if pos==0 and bool(row["entry_cond"]):
            pos=1; avg=float(row["open"]); entries+=1
        elif pos==1:
            # intrabar SL/TP
            hit, why, px = apply_intrabar_sl_tp(float(row["open"]), float(row["high"]), float(row["low"]),
                                                avg, CONFIG.slPerc, CONFIG.tpPerc)
            exitc = bool(row["exit_cond"])
            if hit or exitc:
                px = px if hit else float(row["open"])
                r = (px-avg)/max(1e-9,avg)
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean(); pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(f"📈 Backtest {days}d  Trades={entries}/{exits}  Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\n"
                                        "ℹ️ EoB+Intrabar SL/TP, keine Fees/Slippage.")
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_wf(update, context):
    # sehr knapper Walk-Forward (Demo)
    args = context.args or []
    is_days = int(args[0]) if len(args)>=1 and args[0].isdigit() else 120
    oos_days= int(args[1]) if len(args)>=2 and args[1].isdigit() else 30
    sym = CONFIG.symbols[0]
    df, _ = fetch_ohlcv_tv_sync(sym, CONFIG.interval, is_days+oos_days+300)
    if df.empty: await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG)
    # split letzten (is+oos) Abschnitte
    IS = f.iloc[-(is_days+oos_days)*7: -oos_days*7]  # ~7 bars/day (RTH 6.5h)
    OOS= f.iloc[-oos_days*7:]
    def run_block(F):
        eq=1.0; pos=0; avg=0.0
        R=[]; entries=exits=0
        for i in range(2,len(F)):
            row, prev = F.iloc[i], F.iloc[i-1]
            if pos==0 and bool(row["entry_cond"]):
                pos=1; avg=float(row["open"]); entries+=1
            elif pos==1:
                hit, why, px = apply_intrabar_sl_tp(float(row["open"]), float(row["high"]), float(row["low"]),
                                                    avg, CONFIG.slPerc, CONFIG.tpPerc)
                exitc = bool(row["exit_cond"])
                if hit or exitc:
                    px = px if hit else float(row["open"])
                    r=(px-avg)/max(1e-9,avg)
                    eq*=(1+r); R.append(r); exits+=1
                    pos=0; avg=0.0
        a=np.array(R) if R else np.array([0.0])
        win=(a>0).mean(); pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        return {"trades":(entries,exits),"win":win,"pf":pf,"ret":eq}
    is_res = run_block(IS)
    oos_res= run_block(OOS)
    await update.message.reply_text(
        "🔁 Walk-Forward (Demo)\n"
        f"IS {is_days}d: trades={is_res['trades']} win={is_res['win']*100:.1f}% PF={is_res['pf']:.2f} Ret={is_res['ret']:.2f}x\n"
        f"OOS{oos_days}d: trades={oos_res['trades']} win={oos_res['win']*100:.1f}% PF={oos_res['pf']:.2f} Ret={oos_res['ret']:.2f}x"
    )

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, _ = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty: await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG); last = f.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} {CONFIG.interval}\n"
        f"rsi={float(last['rsi']):.2f} (rise={bool(last['rsi_rising'])})  "
        f"efi={float(last['efi']):.2f} (rise={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}"
    )

async def cmd_ind(update, context):
    await cmd_sig(update, context)

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    df, _ = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty: await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG).tail(300)
    fig, ax = plt.subplots(figsize=(10,6)); ax.plot(f.index, f["close"]); ax.set_title(f"{sym} Close"); ax.grid(True)
    fig2, ax2 = plt.subplots(figsize=(10,3)); ax2.plot(f.index, f["rsi"]); ax2.axhline(CONFIG.rsiLow, ls="--"); ax2.axhline(CONFIG.rsiHigh, ls="--"); ax2.set_title("RSI")
    fig3, ax3 = plt.subplots(figsize=(10,3)); ax3.plot(f.index, f["efi"]); ax3.set_title("EFI")
    await send_png(fig,  f"{sym}_{CONFIG.interval}_close.png","Close")
    await send_png(fig2, f"{sym}_{CONFIG.interval}_rsi.png","RSI")
    await send_png(fig3, f"{sym}_{CONFIG.interval}_efi.png","EFI")

def build_export(df: pd.DataFrame) -> pd.DataFrame:
    f = build_features(df, CONFIG)
    cols = ["open","high","low","close","volume","rsi","efi","rsi_rising","efi_rising","entry_cond","exit_cond","time"]
    return f[cols]

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty: await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG); last = f.iloc[-1]
    payload = {
        "symbol": sym, "interval": CONFIG.interval, "provider": note.get("provider",""),
        "time": str(last["time"]),
        "open": float(last["open"]), "high": float(last["high"]), "low": float(last["low"]), "close": float(last["close"]),
        "volume": float(last["volume"]),
        "rsi": float(last["rsi"]), "efi": float(last["efi"]),
        "rsi_rising": bool(last["rsi_rising"]), "efi_rising": bool(last["efi_rising"]),
        "entry_cond": bool(last["entry_cond"]), "exit_cond": bool(last["exit_cond"]),
        "slPerc": CONFIG.slPerc, "tpPerc": CONFIG.tpPerc
    }
    await update.message.reply_text("🧾 Dump\n" + json.dumps(payload, indent=2))

async def cmd_dumpcsv(update, context):
    sym = CONFIG.symbols[0]
    n = 300
    if context.args:
        try: n=max(1,int(context.args[0]))
        except: pass
    df, _ = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty: await update.message.reply_text("❌ Keine Daten."); return
    exp = build_export(df).tail(n)
    csv_bytes = exp.to_csv(index=True).encode("utf-8")
    await send_doc_bytes(csv_bytes, f"{sym}_{CONFIG.interval}_ind_{n}.csv", f"CSV {sym} n={n}")

async def cmd_timer(update, context):
    if not context.args: await update.message.reply_text("Nutze: /timer on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    TIMER["enabled"]=on
    await update.message.reply_text(f"Timer = {'ON' if on else 'OFF'}")

async def cmd_market(update, context):
    if not context.args: await update.message.reply_text("Nutze: /market on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    TIMER["market_hours_only"] = on
    CONFIG.market_hours_only   = on
    await update.message.reply_text(f"market_hours_only = {on}")

async def cmd_timerstatus(update, context):
    await update.message.reply_text("⏱️ Timer\n" + json.dumps({
        "enabled": TIMER["enabled"],
        "running": TIMER["running"],
        "poll_minutes": TIMER["poll_minutes"],
        "last_run": TIMER["last_run"],
        "next_due": TIMER["next_due"],
        "market_hours_only": TIMER["market_hours_only"]
    }, indent=2))

async def cmd_timerrunnow(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    now = now_utc()
    TIMER["last_run"] = to_iso(now)
    TIMER["next_due"] = to_iso(compute_next_due(now))
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def on_msg(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

async def cmd_debugbars(update, context):
    import pandas as pd
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); 
        return
    tail = df.tail(6)
    lines = [
        f"{pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%M:%S %Z')} | close={r['close']:.4f}"
        for _, r in tail.iterrows()
    ]
    await update.message.reply_text(
        "🧪 Debug Bars (last 6)\n" + "\n".join(lines) +
        "\n(1h sollte auf :30 UTC enden, z. B. 13:30, 14:30 …)"
    )
# --- LIVE toggle + Timer-Kopplung (optional) ---
async def cmd_live(update, context):
    """
    /live on|off
    Setzt CONFIG.live_enabled und startet/stoppt den Timer analog zu /timer.
    """
    global TIMER_TASK
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off")
        return
    on = context.args[0].lower() in ("on", "1", "true", "start")
    CONFIG.live_enabled = on
    # Timer spiegeln
    TIMER["enabled"] = on
    msg = [f"Live = {'ON' if on else 'OFF'}"]
    try:
        if on:
            if TIMER_TASK is None or TIMER_TASK.done():
                TIMER_TASK = asyncio.create_task(timer_loop())
                msg.append("⏱️ Timer gestartet")
        else:
            if TIMER_TASK and not TIMER_TASK.done():
                TIMER["enabled"] = False
                await asyncio.sleep(0.05)
                TIMER_TASK.cancel()
                TIMER_TASK = None
                msg.append("⏱️ Timer gestoppt")
    except Exception as e:
        msg.append(f"⚠️ Timer-Fehler: {e}")
    await update.message.reply_text(" • ".join(msg))
    
# ========= FastAPI lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED, TIMER_TASK
    # --- Startup ---
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Handlers registrieren (inkl. debugbars)
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("dumpcsv", cmd_dumpcsv))
        tg_app.add_handler(CommandHandler("trade",   cmd_trade))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(CommandHandler("account", cmd_account))
        tg_app.add_handler(CommandHandler("timer",        cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("timerrunnow",  cmd_timerrunnow))
        tg_app.add_handler(CommandHandler("debugbars",    cmd_debugbars))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()

        # Webhook entfernen (wir nutzen Polling)
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            print("delete_webhook warn:", e)

        # Polling nur einmal starten (mit Conflict-Retry)
        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED = True
                    print("✅ Telegram polling läuft")
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. retry in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc()
                    await asyncio.sleep(10)

        # Timer ggf. starten
        if TIMER.get("enabled") and TIMER_TASK is None:
            TIMER_TASK = asyncio.create_task(timer_loop())
            print("⏱️ Timer gestartet")

    except Exception as e:
        print("❌ Telegram startup error:", e)
        traceback.print_exc()

    # --- Serve ---
    try:
        yield
    finally:
        # --- Shutdown ---
        try:
            if TIMER_TASK:
                TIMER["enabled"] = False
                await asyncio.sleep(0.05)
                TIMER_TASK.cancel()
        except Exception:
            pass
        try:
            if tg_app and tg_app.updater:
                await tg_app.updater.stop()
        except Exception:
            pass
        try:
            if tg_app:
                await tg_app.stop()
                await tg_app.shutdown()
        except Exception:
            pass
        POLLING_STARTED = False
        TIMER["running"] = False
        print("🛑 Shutdown complete")

# ========= FastAPI =========
app = FastAPI(title="TQQQ Strategy + Telegram (V5.1.2)", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "feed": APCA_DATA_FEED,
        "live": CONFIG.live_enabled,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "poll_minutes": TIMER["poll_minutes"],
            "last_run": TIMER["last_run"],
            "next_due": TIMER["next_due"],
            "market_hours_only": TIMER["market_hours_only"]
        },
        "trade_enabled": CONFIG.trade_enabled
    }

@app.get("/tick")
async def tick():
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=False)
    now = now_utc()
    TIMER["last_run"]=to_iso(now)
    TIMER["next_due"]=to_iso(compute_next_due(now))
    return {"ran": True, "at": TIMER["last_run"]}

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v=os.getenv(k); return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_DATA_FEED": os.getenv("APCA_DATA_FEED",""),
        "ENABLE_TRADE": os.getenv("ENABLE_TRADE",""),
        "ENABLE_TIMER": os.getenv("ENABLE_TIMER",""),
        "POLL_MINUTES": os.getenv("POLL_MINUTES",""),
        "MARKET_HOURS_ONLY": os.getenv("MARKET_HOURS_ONLY",""),
    }

@app.get("/tgstatus")
def tgstatus():
    return {"polling_started": POLLING_STARTED}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
