# alpaca_trading_bot.py  — V5 (TV-sync, PDT, Trader-Sizer, Intrabar SL/TP, WF)
import os, io, json, time, asyncio, traceback, warnings, math
from datetime import datetime, timedelta, timezone, date
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.7)
from telegram import Update, InputFile
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler, filters
)
from telegram.error import Conflict, BadRequest

warnings.filterwarnings("ignore", category=UserWarning)

# ========= ENV & Defaults =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None
if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "sip").lower()  # "sip" (empfohlen) oder "iex"

# Timer/Trading Defaults
ENV_ENABLE_TRADE       = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER       = os.getenv("ENABLE_TIMER", "true").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES       = int(os.getenv("POLL_MINUTES", "10"))
ENV_MARKET_HOURS_ONLY  = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

# Persistenz-Dateien
PDT_FILE = "/mnt/data/pdt_trades.json"

# ========= Utility: US-Business-Days =========
US_HOLIDAYS = {
    # Beispiel-Sets (erweitern/aktualisieren nach Bedarf)
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27","2024-06-19","2024-07-04",
   "2024-09-02","2024-11-28","2024-12-25",
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26","2025-06-19","2025-07-04",
   "2025-09-01","2025-11-27","2025-12-25",
}
def is_us_business_day(d: date) -> bool:
    if d.strftime("%Y-%m-%d") in US_HOLIDAYS: return False
    return d.weekday() < 5

def last_n_business_days(n: int, ref: Optional[date] = None) -> List[date]:
    ref = ref or datetime.now(timezone.utc).date()
    out=[]
    cur = ref
    while len(out)<n:
        if is_us_business_day(cur): out.append(cur)
        cur -= timedelta(days=1)
    return out

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    # Engine
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"             # '1h' oder '1d'
    lookback_days: int = 365
    drop_last_incomplete: bool = True
    rth_only: bool = True            # nur Regular Trading Hours (NY 9:30–16:00)
    tv_anchor_halfhour: bool = True  # 1h-Bars bei :30 UTC schließen

    # Indikatoren (TV-kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0              # Default=0 (gewünscht)
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0
    intrabar_risk_check: bool = True  # SL/TP intrabar via High/Low

    # Live Scheduler
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = False
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY
    timer_sync_to_candle: bool = True # Timer mit Candle-Anker synchronisieren

    # Data Provider
    data_provider: str = "alpaca"    # "alpaca" | "yahoo" | "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Trading Toggle
    trade_enabled: bool = ENV_ENABLE_TRADE

    # Trader Sizer
    sizing_mode: str = "shares"      # 'shares'|'percent_equity'|'notional_usd'|'risk'
    sizing_value: float = 1.0
    max_position_pct: float = 100.0  # maximale Position in % vom Equity

    # Backtest/Walk-Forward
    bt_slippage_bp: float = 0.0      # Slippage in Basispunkten (1bp=0.01%)
    bt_fee_perc: float = 0.0         # Fee als % pro Trade (Roundtrip approx)

    # PDT
    pdt_hard_stop: bool = True       # harte Blockade neuer Daytrades
    pdt_max_daytrades_5bd: int = 3   # Non-PDT: max 3 DT in 5 Business Days

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}  # {symbol: {"size":int,"avg":float,"entry_time":str|None}}
    last_status: str = "idle"
    data_feed: str = APCA_DATA_FEED

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= Indicators (TV-kompatibel) =========
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi_tv_wilder(s: pd.Series, length: int = 14) -> pd.Series:
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

# ========= Alpaca Clients =========
def alpaca_data_client():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    except Exception as e:
        print("[alpaca-data] error:", e)
        return None

def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca-trading] error:", e)
        return None

# ========= TV-sync 1-Minute → 1h Bars (RTH, :30-UTC-Anker) =========
def floor_to_half_hour_utc(ts: pd.Timestamp) -> pd.Timestamp:
    # runde ab auf :30 oder :00 – TV-Anker: Bars enden bei :30, also Buckets [:30–:30)
    ts = ts.tz_convert("UTC")
    mm = 30 if ts.minute >= 30 else 0
    return ts.replace(minute=mm, second=0, microsecond=0)

def resample_1m_to_1h_tv_sync(df1m: pd.DataFrame, drop_last_incomplete: bool = True) -> pd.DataFrame:
    """
    Erwartet 1-min Bars in UTC. Baut 1h-Bars, die immer bei **:30 UTC** enden:
    Buckets: 13:30–14:30, 14:30–15:30, ...
    Außerdem RTH-Filter: 13:30–20:00 UTC (9:30–16:00 ET).
    """
    if df1m.empty: return df1m
    df = df1m.copy()

    # nur RTH
    if CONFIG.rth_only:
        idx = df.index.tz_convert("UTC")
        minutes = idx.hour * 60 + idx.minute
        mask = (minutes >= (13*60+30)) & (minutes <= (20*60))
        df = df.loc[mask]
        if df.empty: return df

    # Hilfsindex: "Endzeit" pro 1m-Bar auf :30-Raster
    # Wir lassen 1h-Kerzen bei :30 enden => jede Minute bekommt den nächsten :30-Zeitstempel als "close_bucket"
    idx = df.index.tz_convert("UTC")
    close_bucket = []
    for ts in idx:
        base = floor_to_half_hour_utc(ts)
        if ts >= base:
            # Minute gehört zur Periode (base .. base+1h]
            bucket_end = base + timedelta(hours=1)
        else:
            # theoretisch nicht nötig
            bucket_end = base
        close_bucket.append(bucket_end)
    df = df.copy()
    df["bucket_end"] = close_bucket

    # Resample by bucket_end (Groupby)
    gb = df.groupby("bucket_end")
    out = pd.DataFrame({
        "open": gb["open"].first(),
        "high": gb["high"].max(),
        "low": gb["low"].min(),
        "close": gb["close"].last(),
        "volume": gb["volume"].sum()
    })
    out.index = pd.to_datetime(out.index).tz_localize("UTC")

    # unvollständige letzte Kerze droppen
    if drop_last_incomplete:
        # "jetzt" zur nächsten :30 Runde
        now = datetime.now(timezone.utc)
        cur_bucket_end = floor_to_half_hour_utc(pd.Timestamp(now, tz="UTC")) + timedelta(hours=1)
        if len(out)>0 and out.index[-1] >= cur_bucket_end:
            out = out.iloc[:-1]

    out["time"] = out.index
    return out

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
        return df
    except Exception as e:
        print("[stooq] fetch failed:", e)
        return pd.DataFrame()

def fetch_alpaca_1m(symbol: str, lookback_days: int) -> pd.DataFrame:
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] library not available:", e)
        return pd.DataFrame()

    client = alpaca_data_client()
    if client is None:
        return pd.DataFrame()

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 10))
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start, end=end,
            feed=("SIP" if APCA_DATA_FEED=="sip" else "IEX"),
            limit=50000
        )
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.rename(columns=str.lower).sort_index()
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca 1m] fetch failed:", e)
        return pd.DataFrame()

def fetch_yahoo(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}
    is_intraday = interval in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except Exception: pass
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
        ("download", dict(tickers=symbol, interval=("60m" if interval=="1h" else interval),
                          period=period, auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=("60m" if interval=="1h" else interval),
                          period=period, auto_adjust=False, progress=False, prepost=True, threads=False)),
        ("history",  dict(period=period, interval=("60m" if interval=="1h" else interval),
                          auto_adjust=False, prepost=True)),
    ]
    last_err=None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = _norm(tmp)
                if not df.empty:
                    return df
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # 1d fallback
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 60)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty:
            return df
    except Exception as e:
        last_err = e

    print("[yahoo] failed:", last_err)
    return pd.DataFrame()

def fetch_ohlcv_tv_sync(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    Primär: Alpaca 1m → TV-synchrone 1h RTH; Sekundär: Yahoo 60m → (optional) auf TV-Anker anpassen; Tertiär: Stooq daily
    """
    note = {"provider":"", "detail":""}

    if interval == "1h":
        # 1) Alpaca 1m and resample
        if CONFIG.data_provider.lower() == "alpaca":
            m1 = fetch_alpaca_1m(symbol, lookback_days)
            if not m1.empty:
                h1 = resample_1m_to_1h_tv_sync(m1, drop_last_incomplete=CONFIG.drop_last_incomplete)
                if not h1.empty:
                    note.update(provider=f"Alpaca ({APCA_DATA_FEED})", detail="1h TV-sync from 1m")
                    return h1, note
            # -> Fallback
            note.update(provider=f"Alpaca→Yahoo", detail="fallback 60m")

        # 2) Yahoo 60m (nicht perfekt TV-ankert, aber nutzbar)
        yh = fetch_yahoo(symbol, "1h", lookback_days)
        if not yh.empty:
            # RTH grob filtern + letzte halbe Bar droppen (best effort)
            if CONFIG.rth_only:
                idx = yh.index.tz_convert("UTC")
                minutes = idx.hour*60 + idx.minute
                mask = (minutes >= (13*60+30)) & (minutes <= (20*60))
                yh = yh.loc[mask]
            if CONFIG.drop_last_incomplete and len(yh)>0:
                # heuristik: letzte Kerze in Zukunft?
                now = datetime.now(timezone.utc)
                last_ts = yh.index[-1]
                if last_ts > now:
                    yh = yh.iloc[:-1]
            yh["time"] = yh.index
            note.update(provider="Yahoo (60m)", detail="approx TV-sync")
            return yh, note

        # 3) Stooq EOD fallback
        if CONFIG.allow_stooq_fallback:
            dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
            if not dfe.empty:
                note.update(provider="Stooq EOD (Fallback)", detail="1d")
                return dfe, note

        return pd.DataFrame(), {"provider":"(leer)","detail":"no data (1h)"}

    else:
        # 1d direkt von Alpaca oder Yahoo
        if CONFIG.data_provider.lower() == "alpaca":
            # Alpaca daily via bars? Nutzen wir Yahoo direkt, um Komplexität gering zu halten.
            df = fetch_yahoo(symbol, "1d", lookback_days)
            if not df.empty:
                note.update(provider="Yahoo", detail="1d")
                return df, note

        df = fetch_yahoo(symbol, "1d", lookback_days)
        if not df.empty:
            note.update(provider="Yahoo", detail="1d")
            return df, note

        if CONFIG.allow_stooq_fallback:
            dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
            if not dfe.empty:
                note.update(provider="Stooq EOD (Fallback)", detail="1d")
                return dfe, note

        return pd.DataFrame(), {"provider":"(leer)","detail":"no data (1d)"}

# ========= Features / Signals =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv_wilder(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

def compute_signals_for_frame(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    f = build_features(df, cfg)
    f["rsi_rising"] = f["rsi"] > f["rsi"].shift(1)
    f["rsi_falling"] = f["rsi"] < f["rsi"].shift(1)
    f["efi_rising"] = f["efi"] > f["efi"].shift(1)
    # Entry/Exit (ohne MACD) — bestätigt
    f["entry_cond"] = (f["rsi"] > cfg.rsiLow) & (f["rsi"] < cfg.rsiHigh) & f["rsi_rising"] & f["efi_rising"]
    f["exit_cond"]  = (f["rsi"] < cfg.rsiExit) & f["rsi_falling"]
    return f

# ========= PDT Persistenz =========
def load_pdt() -> Dict[str, Any]:
    try:
        with open(PDT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"daytrades": []}  # list of {"open": "YYYY-MM-DD", "close": "YYYY-MM-DD"}

def save_pdt(obj: Dict[str,Any]):
    try:
        os.makedirs(os.path.dirname(PDT_FILE), exist_ok=True)
        with open(PDT_FILE, "w") as f:
            json.dump(obj, f)
    except Exception as e:
        print("[pdt] save error:", e)

def register_daytrade(open_dt: datetime, close_dt: datetime):
    p = load_pdt()
    p["daytrades"].append({"open": str(open_dt.date()), "close": str(close_dt.date())})
    # Cleanup: keep last ~200
    if len(p["daytrades"])>200:
        p["daytrades"] = p["daytrades"][-200:]
    save_pdt(p)

def pdt_count_last_5bd(ref: Optional[date] = None) -> int:
    p = load_pdt()
    last5 = {d.strftime("%Y-%m-%d") for d in last_n_business_days(5, ref)}
    cnt=0
    for tr in p.get("daytrades", []):
        # Daytrade = open und close am gleichen Kalendertag (US)
        if tr.get("open")==tr.get("close") and tr["close"] in last5:
            cnt += 1
    return cnt

def pdt_block_new_daytrade() -> bool:
    # harte Blockade, wenn nächste Position potenziell ein Daytrade wäre und Limit bereits erreicht
    if not CONFIG.pdt_hard_stop: return False
    # Wir blocken "Entry" wenn in den letzten 5 BD bereits >= pdt_max Daytrades
    cnt = pdt_count_last_5bd()
    return cnt >= CONFIG.pdt_max_daytrades_5bd

# ========= Trader Sizer =========
def alpaca_account_equity() -> Optional[float]:
    client = alpaca_trading_client()
    if client is None: return None
    try:
        acc = client.get_account()
        return float(acc.equity)
    except Exception as e:
        print("alpaca_account_equity error:", e); return None

def price_of_last(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1])

def position_size_for_signal(sym: str, df: pd.DataFrame) -> int:
    mode = CONFIG.sizing_mode.lower()
    val  = CONFIG.sizing_value
    px = price_of_last(df)
    # Equity-Basis
    eq = alpaca_account_equity()
    if eq is None:  # fallback: 10000 für Sizer
        eq = 10000.0
    max_shares_by_pct = math.floor((CONFIG.max_position_pct/100.0 * eq) / max(px, 1e-9))

    if mode == "shares":
        qty = int(val)
    elif mode == "percent_equity":
        notional = (val/100.0) * eq
        qty = max(1, math.floor(notional / max(px, 1e-9)))
    elif mode == "notional_usd":
        qty = max(1, math.floor(val / max(px, 1e-9)))
    elif mode == "risk":
        # einfacher Risk: val = %Equity, SL-Distanz = slPerc (%)
        risk_dollars = (val/100.0) * eq
        sl_dist = CONFIG.slPerc/100.0 * px
        qty = max(1, math.floor(risk_dollars / max(sl_dist, 1e-9)))
    else:
        qty = 1

    qty = max(1, min(qty, max_shares_by_pct if max_shares_by_pct>0 else qty))
    return qty

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
POLLING_STARTED = False

async def send_text(chat_id: str, text: str):
    if tg_app is None: return
    if not text or not text.strip():
        text = "ℹ️ (leer)"
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        print("send_text error:", e)

async def send_document_bytes(chat_id: str, data: bytes, filename: str, caption: str = ""):
    if tg_app is None: return
    try:
        bio = io.BytesIO(data); bio.name = filename; bio.seek(0)
        await tg_app.bot.send_document(chat_id=chat_id, document=InputFile(bio), caption=caption)
    except Exception as e:
        print("send_document error:", e)

async def send_png(chat_id: str, fig, filename: str, caption: str = ""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    await send_document_bytes(chat_id, buf.getvalue(), filename, caption)

# ========= Trading via Alpaca (Paper) =========
async def place_market_order(sym: str, qty: int, side: str, tif: str = "day") -> str:
    client = alpaca_trading_client()
    if client is None: return "alpaca client not available"
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tif_map = {
            "day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "opg": TimeInForce.OPG,
            "cls": TimeInForce.CLS, "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK
        }
        req = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif_map.get(tif.lower(), TimeInForce.DAY)
        )
        order = client.submit_order(order_data=req)
        return f"order_id={order.id}"
    except Exception as e:
        return f"alpaca order error: {e}"

def alpaca_positions() -> List[Dict[str,Any]]:
    client = alpaca_trading_client()
    if client is None: return []
    try:
        pos = client.get_all_positions()
        out=[]
        for p in pos:
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_price": float(p.current_price),
                "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0) or 0.0)
            })
        return out
    except Exception as e:
        print("alpaca_positions error:", e); return []

def alpaca_account() -> Dict[str,Any]:
    client = alpaca_trading_client()
    if client is None: return {}
    try:
        acc = client.get_account()
        return {
            "status": acc.status,
            "equity": float(acc.equity),
            "cash": float(acc.cash),
            "buying_power": float(acc.buying_power),
            "multiplier": acc.multiplier
        }
    except Exception as e:
        print("alpaca_account error:", e); return {}

# ========= Single-step logic =========
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    return f"📡 Daten: {prov} ({det})"

def sl_for(p): return p*(1-CONFIG.slPerc/100.0)
def tp_for(p): return p*(1+CONFIG.tpPerc/100.0)

def bar_logic_last(df: pd.DataFrame, cfg: StratConfig, sym: str) -> Dict[str,Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data","symbol":sym}

    f = compute_signals_for_frame(df, cfg)
    last = f.iloc[-1]

    price_open  = float(last["open"])
    price_close = float(last["close"])
    bar_low     = float(last["low"])
    bar_high    = float(last["high"])
    ts = last["time"]

    entry_cond = bool(last["entry_cond"])
    exit_cond  = bool(last["exit_cond"])

    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]; entry_time = pos["entry_time"]

    # bars in trade
    bars_in_trade=0
    if entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if size==0:
        if entry_cond:
            # PDT Hard Stop: blocke neuen Daytrade ggf.
            if pdt_block_new_daytrade():
                return {"action":"none","symbol":sym,"reason":"pdt_block",
                        "rsi":float(last["rsi"]), "efi":float(last["efi"])}
            q = position_size_for_signal(sym, df)
            return {"action":"buy","symbol":sym,"qty":q,"px":price_open,"time":str(ts),
                    "sl":sl_for(price_open),"tp":tp_for(price_open),"reason":"rule_entry",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        return {"action":"none","symbol":sym,"reason":"flat_no_entry",
                "rsi":float(last["rsi"]), "efi":float(last["efi"])}

    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade>0)
        cooldown_ok = (bars_in_trade>=cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl_for(avg); cur_tp = tp_for(avg)

        if cfg.intrabar_risk_check:
            hit_sl = bar_low  <= cur_sl
            hit_tp = bar_high >= cur_tp
        else:
            hit_sl = price_close <= cur_sl
            hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","symbol":sym,"qty":size,"px":price_open,"time":str(ts),"reason":"rsi_exit",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        if hit_sl:
            return {"action":"sell","symbol":sym,"qty":size,"px":cur_sl,"time":str(ts),"reason":"stop_loss",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        if hit_tp:
            return {"action":"sell","symbol":sym,"qty":size,"px":cur_tp,"time":str(ts),"reason":"take_profit",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        return {"action":"none","symbol":sym,"reason":"hold",
                "rsi":float(last["rsi"]), "efi":float(last["efi"])}

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        return {"ok":False, "msg": f"❌ Keine Daten für {sym}. {note}"}

    act = bar_logic_last(df, CONFIG, sym)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    # Telegram Info
    if send_signals and CHAT_ID:
        await send_text(CHAT_ID, f"ℹ️ {sym} {CONFIG.interval} rsi={act.get('rsi',np.nan):.2f} efi={act.get('efi',np.nan):.2f} • {act['reason']}")
        await send_text(CHAT_ID, friendly_note(note))

    # Trading (Paper) – blockiert automatisch bei PDT durch bar_logic_last
    if CONFIG.trade_enabled and act["action"] in ("buy","sell"):
        if CONFIG.market_hours_only and not is_market_open_now():
            if CHAT_ID:
                await send_text(CHAT_ID, "⛔ Markt geschlossen – kein Trade.")
        else:
            side = "buy" if act["action"]=="buy" else "sell"
            tif  = "day"
            info = await place_market_order(sym, int(act["qty"]), side, tif)
            if CHAT_ID:
                await send_text(CHAT_ID, f"🛒 {side.upper()} {sym} x{act['qty']} @ {act['px']:.4f} • {info}")

    # Sim-Position + PDT-Registrierung
    pos = STATE.positions.get(sym, {"size":0, "avg":0.0, "entry_time":None})
    if act["action"]=="buy" and pos["size"]==0:
        STATE.positions[sym] = {"size":act["qty"],"avg":act["px"],"entry_time":act["time"]}
        if CHAT_ID and send_signals:
            await send_text(CHAT_ID, f"🟢 LONG (sim) {sym} @ {act['px']:.4f} | SL={act.get('sl',np.nan):.4f} TP={act.get('tp',np.nan):.4f}")
    elif act["action"]=="sell" and pos["size"]>0:
        pnl = (act["px"] - pos["avg"]) / pos["avg"]
        # PDT-Erkennung: Daytrade wenn open/close same US business day
        try:
            open_day  = pd.to_datetime(pos["entry_time"]).tz_convert("UTC").date()
            close_day = pd.to_datetime(act["time"]).tz_convert("UTC").date()
            if open_day == close_day and is_us_business_day(open_day):
                register_daytrade(pd.to_datetime(pos["entry_time"]).to_pydatetime(), pd.to_datetime(act["time"]).to_pydatetime())
        except Exception:
            pass
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if CHAT_ID and send_signals:
            await send_text(CHAT_ID, f"🔴 EXIT (sim) {sym} @ {act['px']:.4f} • {act['reason']} • PnL={pnl*100:.2f}%")

    return {"ok":True,"act":act}

# ========= Market open (UTC window) =========
def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or datetime.now(timezone.utc)
    if not is_us_business_day(now.date()): return False
    hhmm = now.hour*60 + now.minute
    return (13*60+30) <= hhmm <= (20*60)

# ========= Timer (synchron auf :30 für 1h) =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}
TIMER_TASK: Optional[asyncio.Task] = None

def next_halfhour_anchor_utc(now: datetime) -> datetime:
    base = now.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    mm = 30 if now.minute < 30 else 0
    hour = now.hour if now.minute < 30 else (now.hour+1)
    anchor = base.replace(minute=mm, hour=hour)
    # für 1h TF laufen wir *auf* den Anchor
    return anchor

async def timer_loop():
    TIMER["running"] = True
    try:
        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)

            # Market-hours Filter
            if TIMER["market_hours_only"] and not is_market_open_now(now):
                TIMER["next_due"] = None
                await asyncio.sleep(30)
                continue

            # Candle-synced?
            if CONFIG.timer_sync_to_candle and CONFIG.interval=="1h":
                # Ziel: :30 UTC (13:30, 14:30, ...)
                anchor = next_halfhour_anchor_utc(now)
                # wenn jetzt genau Anchor? dann laufen — sonst schlafen bis Anchor
                if abs((anchor - now).total_seconds()) > 2:
                    sleep_s = max(1, (anchor - now).total_seconds())
                    await asyncio.sleep(sleep_s)
                    # nach Sleep sofort run
            else:
                # klassisch alle poll_minutes
                if TIMER["last_run"] and TIMER["next_due"]:
                    if now < pd.to_datetime(TIMER["next_due"]).to_pydatetime():
                        await asyncio.sleep(5)
                        continue

            # run
            for sym in CONFIG.symbols:
                await run_once_for_symbol(sym, send_signals=True)
            TIMER["last_run"] = now.isoformat()
            if CONFIG.timer_sync_to_candle and CONFIG.interval=="1h":
                # nächster Anchor + 1h
                next_anchor = next_halfhour_anchor_utc(now + timedelta(minutes=61))
                TIMER["next_due"] = next_anchor.isoformat()
            else:
                TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()

            await asyncio.sleep(2)
    finally:
        TIMER["running"] = False

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg, /set key=value …, /run, /live on|off, /bt [tage]\n"
        "/sig, /ind, /plot, /dump [csv [N]], /dumpcsv [N]\n"
        "/trade on|off, /pos, /account\n"
        "/timer on|off, /timerstatus, /timerrunnow\n"
        "/wf [is_days oos_days] – Walk-Forward/OOS\n"
        "/pdt – Zähler letzte 5 Handelstage\n"
    )

async def cmd_status(update, context):
    pos_lines=[]
    for s,p in STATE.positions.items():
        pos_lines.append(f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}")
    pos_txt = "\n".join(pos_lines) if pos_lines else "keine (sim)"

    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider}  Feed: {STATE.data_feed}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} • Timer: {'ON' if TIMER['enabled'] else 'OFF'} "
        f"(alle {TIMER['poll_minutes']}m, market-hours-only={TIMER['market_hours_only']})\n"
        f"RiskCheck: {'Intrabar (High/Low)' if CONFIG.intrabar_risk_check else 'EoB (Close)'}\n"
        f"Sizer: mode={CONFIG.sizing_mode} value={CONFIG.sizing_value} max%={CONFIG.max_position_pct}\n"
        f"LastStatus: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper)\n"
        f"PDT used (5BD): {pdt_count_last_5bd()}/{CONFIG.pdt_max_daytrades_5bd}"
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
    else:                       setattr(CONFIG, k, v)
    # Sync Timer
    if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=4\n"
            "/set interval=1h lookback_days=365 tv_anchor_halfhour=true\n"
            "/set sizing_mode=percent_equity sizing_value=10 max_position_pct=50\n"
            "/set intrabar_risk_check=true\n"
            "/set pdt_hard_stop=true\n"
        ); return
    msgs=[]; errs=[]
    for a in context.args:
        if "=" not in a: errs.append(f"❌ ungültig: {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = ("✅ Übernommen:\n" + "\n".join(msgs) + ("\n\n⚠️ Probleme:\n" + "\n".join(errs) if errs else "")).strip()
    await update.message.reply_text(txt)

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_trade(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /trade on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.trade_enabled = on
    await update.message.reply_text(f"Trading (Paper) = {'ON' if on else 'OFF'}")

async def cmd_pos(update, context):
    pos = alpaca_positions()
    if pos:
        lines = [f"{p['symbol']}: qty={p['qty']}, avg={p['avg_entry']:.4f}, last={p['market_price']:.4f}, UPL={p['unrealized_pl']:.2f}" for p in pos]
        await update.message.reply_text("📦 Alpaca Positionen\n" + "\n".join(lines))
    else:
        await update.message.reply_text("📦 Alpaca Positionen: keine oder Zugriff nicht möglich.")

async def cmd_account(update, context):
    acc = alpaca_account()
    if acc:
        await update.message.reply_text("👤 Alpaca Account\n" + json.dumps(acc, indent=2))
    else:
        await update.message.reply_text("👤 Alpaca Account: kein Zugriff.")

async def cmd_run(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für Backtest ({sym})."); return
    f = compute_signals_for_frame(df, CONFIG)

    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry = bool(row["entry_cond"])
        exitc = bool(row["exit_cond"])
        if pos==0 and entry:
            # PDT block im BT: simulativ (falls HardStop aktiv)
            if CONFIG.pdt_hard_stop and pdt_count_last_5bd(pd.to_datetime(row["time"]).date()) >= CONFIG.pdt_max_daytrades_5bd:
                continue
            # sizing
            px_open = float(row["open"])
            q = position_size_for_signal(sym, f.iloc[:i+1])
            pos=1; avg=px_open; entries+=1
        elif pos==1:
            sl = sl_for(avg); tp = tp_for(avg)
            if CONFIG.intrabar_risk_check:
                stop = float(row["low"])  <= sl
                take = float(row["high"]) >= tp
            else:
                price = float(row["close"])
                stop = price<=sl; take=price>=tp

            if exitc or stop or take:
                px = sl if stop else (tp if take else float(row["open"]))
                # Slippage/Fees
                slip = CONFIG.bt_slippage_bp/10000.0
                fee  = CONFIG.bt_fee_perc/100.0
                # long exit
                gross_r = (px/avg) - 1.0
                gross_r -= slip
                gross_r -= fee
                eq*= (1+gross_r); R.append(gross_r); exits+=1
                pos=0; avg=0.0
                # PDT-Registrierung (Backtest)
                try:
                    od = pd.to_datetime(prev["time"]).date()
                    cd = pd.to_datetime(row["time"]).date()
                    if od==cd and is_us_business_day(od):
                        # temporär zählen (keine Datei-IO im BT)
                        pass
                except Exception:
                    pass

    if R:
        a=np.array(R); win=(a>0).mean(); pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(
            f"📈 Backtest {days}d  Trades={entries}/{exits}  Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\n"
            f"ℹ️ Hinweis: Slippage={CONFIG.bt_slippage_bp}bp, Fees={CONFIG.bt_fee_perc}%"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_wf(update, context):
    # Walk-Forward mit einfachem Grid auf rsiLow/rsiHigh/rsiExit
    args = context.args or []
    is_days = 120
    oos_days = 30
    if len(args)>=1:
        try: is_days=int(args[0])
        except: pass
    if len(args)>=2:
        try: oos_days=int(args[1])
        except: pass
    sym = CONFIG.symbols[0]
    df_all, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df_all.empty or len(df_all)<(is_days+oos_days+50):
        await update.message.reply_text("❌ Zu wenige Daten für WF."); return

    # Split rollierend
    end_idx = len(df_all)
    start_idx = 0
    step = oos_days  # rollen um OOS-Fenster
    results=[]
    grid = []
    for rl in [0, 5, 10]:
        for rh in [60, 65, 68, 70]:
            for rx in [45, 48, 50]:
                if rl < rh:
                    grid.append((rl, rh, rx))

    def run_bt(df, rsiLow, rsiHigh, rsiExit):
        cfg = CONFIG.copy()
        cfg.rsiLow=rsiLow; cfg.rsiHigh=rsiHigh; cfg.rsiExit=rsiExit
        f = compute_signals_for_frame(df, cfg)
        pos=0; avg=0.0; eq=1.0; R=[]
        for i in range(2,len(f)):
            row, prev = f.iloc[i], f.iloc[i-1]
            if pos==0 and bool(row["entry_cond"]):
                if cfg.pdt_hard_stop and pdt_count_last_5bd(pd.to_datetime(row["time"]).date()) >= cfg.pdt_max_daytrades_5bd:
                    continue
                avg=float(row["open"]); pos=1
            elif pos==1:
                sl=sl_for(avg); tp=tp_for(avg)
                if cfg.intrabar_risk_check:
                    stop = float(row["low"])  <= sl
                    take = float(row["high"]) >= tp
                else:
                    price = float(row["close"])
                    stop = price<=sl; take=price>=tp
                exitc = bool(row["exit_cond"])
                if exitc or stop or take:
                    px = sl if stop else (tp if take else float(row["open"]))
                    slip = cfg.bt_slippage_bp/10000.0
                    fee  = cfg.bt_fee_perc/100.0
                    r=(px/avg-1)-slip-fee
                    eq*=(1+r); R.append(r); pos=0; avg=0.0
        if not R:
            return 0.0
        return np.prod(1+np.array(R))

    for i in range(start_idx, end_idx - (is_days+oos_days), step):
        is_df = df_all.iloc[i:i+is_days]
        oos_df = df_all.iloc[i+is_days:i+is_days+oos_days]
        if len(is_df)<20 or len(oos_df)<5: continue
        # Suche bestes Grid auf IS
        best=None; best_ret=-1
        for rl,rh,rx in grid:
            ret = run_bt(is_df, rl, rh, rx)
            if ret>best_ret:
                best_ret=ret; best=(rl,rh,rx)
        # Test auf OOS
        oos_ret = run_bt(oos_df, *best)
        results.append({"win": oos_ret, "best": best})

    if not results:
        await update.message.reply_text("❌ WF: keine Fenster berechnet."); return

    wins = [r["win"] for r in results]
    geo = np.prod(wins) if wins else 0.0
    await update.message.reply_text(
        f"🧪 Walk-Forward (IS={is_days}d, OOS={oos_days}d)\n"
        f"Fenster: {len(results)} | Geo.OOS-Faktor: {geo:.3f}\n"
        f"Median OOS-Faktor: {np.median(wins):.3f}\n"
        f"Grid: rsiLow∈{{0,5,10}}, rsiHigh∈{{60,65,68,70}}, rsiExit∈{{45,48,50}}"
    )

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG)
    last = f.iloc[-1]
    txt = (
        f"🔎 {sym} {CONFIG.interval}\n"
        f"rsi={last['rsi']:.2f} (rise={bool(last['rsi_rising'])})  "
        f"efi={last['efi']:.2f} (rise={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}\n"
        f"{friendly_note(note)}"
    ).strip()
    await update.message.reply_text(txt)

async def cmd_ind(update, context):
    await cmd_sig(update, context)

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    n = 300
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG).tail(n)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(f.index, f["close"], label="Close")
    ax.set_title(f"{sym} {CONFIG.interval} Close")
    ax.grid(True); ax.legend(loc="best")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(f.index, f["rsi"], label="RSI")
    ax2.axhline(CONFIG.rsiLow, linestyle="--")
    ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.set_title("RSI (Wilder)"); ax2.grid(True); ax2.legend(loc="best")
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(f.index, f["efi"], label="EFI")
    ax3.set_title("EFI (EMA(vol*Δclose))"); ax3.grid(True); ax3.legend(loc="best")

    cid = str(update.effective_chat.id)
    await send_png(cid, fig,  f"{sym}_{CONFIG.interval}_close.png", "📈 Close")
    await send_png(cid, fig2, f"{sym}_{CONFIG.interval}_rsi.png",   "📈 RSI")
    await send_png(cid, fig3, f"{sym}_{CONFIG.interval}_efi.png",   "📈 EFI")

def build_export_frame(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    f = compute_signals_for_frame(df, cfg)
    cols = ["open","high","low","close","volume","rsi","efi","rsi_rising","efi_rising","entry_cond","exit_cond","time"]
    return f[cols]

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    args = [a.lower() for a in (context.args or [])]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für {sym}."); return

    if args and args[0]=="csv":
        n = 300
        if len(args)>=2:
            try: n=max(1,int(args[1]))
            except: pass
        exp = build_export_frame(df, CONFIG).tail(n)
        csv_bytes = exp.to_csv(index=True).encode("utf-8")
        await send_document_bytes(str(update.effective_chat.id), csv_bytes,
                                  f"{sym}_{CONFIG.interval}_indicators_{n}.csv",
                                  caption=f"🧾 CSV (OHLCV + RSI/EFI + Entry/Exit) {sym} {CONFIG.interval} n={n}")
        return

    f = compute_signals_for_frame(df, CONFIG)
    last = f.iloc[-1]
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
    context.args = ["csv"] + (context.args or [])
    await cmd_dump(update, context)

async def cmd_pdt(update, context):
    cnt = pdt_count_last_5bd()
    await update.message.reply_text(f"📛 PDT: Daytrades in letzten 5 Handelstagen: {cnt}/{CONFIG.pdt_max_daytrades_5bd} • HardStop={'ON' if CONFIG.pdt_hard_stop else 'OFF'}")

async def cmd_timer(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /timer on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    TIMER["enabled"] = on
    await update.message.reply_text(f"Timer = {'ON' if on else 'OFF'}")

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
    now = datetime.now(timezone.utc)
    TIMER["last_run"] = now.isoformat()
    if CONFIG.timer_sync_to_candle and CONFIG.interval=="1h":
        TIMER["next_due"] = next_halfhour_anchor_utc(now + timedelta(minutes=61)).isoformat()
    else:
        TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= FastAPI lifespan (PTB polling & timer) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED, TIMER_TASK
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Handlers
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
        tg_app.add_handler(CommandHandler("wf",           cmd_wf))
        tg_app.add_handler(CommandHandler("pdt",          cmd_pdt))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass

        if not POLLING_STARTED:
            delay=5
            while True:
                try:
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED=True
                    print("✅ Telegram polling läuft")
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. retry in {delay}s")
                    await asyncio.sleep(delay); delay=min(delay*2,60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        # Timer ggf. starten
        if TIMER["enabled"] and TIMER_TASK is None:
            TIMER_TASK = asyncio.create_task(timer_loop())
            print("⏱️ Timer gestartet")

    except Exception as e:
        print("❌ Telegram startup error:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        try:
            if TIMER_TASK:
                TIMER["enabled"]=False
                await asyncio.sleep(0.1)
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
        POLLING_STARTED=False
        TIMER["running"]=False
        print("🛑 Shutdown complete")

# ========= FastAPI app & routes =========
app = FastAPI(title="TQQQ Strategy + Telegram (V5) TV-sync/PDT/Sizer", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "feed": STATE.data_feed,
        "live": CONFIG.live_enabled,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "poll_minutes": TIMER["poll_minutes"],
            "next_due": TIMER["next_due"]
        },
        "trade_enabled": CONFIG.trade_enabled
    }

@app.get("/tick")
async def tick():
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=False)
    now = datetime.now(timezone.utc)
    TIMER["last_run"]=now.isoformat()
    if CONFIG.timer_sync_to_candle and CONFIG.interval=="1h":
        TIMER["next_due"] = next_halfhour_anchor_utc(now + timedelta(minutes=61)).isoformat()
    else:
        TIMER["next_due"]=(now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
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
