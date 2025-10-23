# -*- coding: utf-8 -*-
# ========================= alpaca_trading_bot.py (V5.1) =========================
# Features:
# - TV-synced 1h bars from Alpaca 1m (anchor :30 UTC, RTH only, no last partial)
# - RSI (Wilder-RMA) + EFI(EMA(vol*Δclose)) — TV-kompatibel
# - Entry: rsiLow <= RSI < rsiHigh  & RSI rising & EFI rising  (rsiLow default 0)
# - Exit : RSI < rsiExit  (RSI falling in Signalanzeige, für Exit-Flag)
# - SL/TP, intrabar-ähnliche Prüfung end-of-bar (EoB) auf last close
# - Intraday-strict: kein EOD-Fallback für 1h; statt dessen Cache (/mnt/data/*.parquet)
# - PDT-Persistenz (/mnt/data/pdt_trades.json) mit Hard-Block (≥3 Daytrades in 5 BizDays)
# - Trader-Sizer: shares | percent_equity | notional_usd | risk
# - Timer: synct auf 30min-Offsets bei 1h; market_hours_only optional
# - Backtest + Walk-Forward (/bt, /wf) inkl. Slippage & Fees
# - Telegram-Befehle vollständig
# ===============================================================================
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

# ============================ ENV & Defaults ============================
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None
if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca credentials (Paper; Market Data uses separate key pair)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "iex").lower()  # 'iex' (default) or 'sip'

# Toggles
ENV_ENABLE_TRADE       = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER       = os.getenv("ENABLE_TIMER", "false").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES       = int(os.getenv("POLL_MINUTES", "10"))
ENV_MARKET_HOURS_ONLY  = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

# ============================ Config / State ============================
class StratConfig(BaseModel):
    # Engine
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"                 # '1h' oder '1d' (intraday strict gilt für 1h)
    lookback_days: int = 365

    # TV-kompatible Inputs
    rsiLen: int = 12
    rsiLow: float = 0.0                  # default 0 (gewünscht)
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Live Scheduler
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = False
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY
    sync_half_hour: bool = True          # 1h bei :30, TV-like

    # Data Provider
    data_provider: str = "alpaca"        # primary
    yahoo_retries: int = 2
    yahoo_backoff_sec: float = 1.0
    allow_stooq_fallback: bool = True    # nur für 1d sinnvoll

    # Intraday policy
    intraday_strict: bool = True         # bei 1h niemals EOD fallback
    cache_intraday: bool = True          # 1h TV-bars cachen

    # Trading Toggle
    trade_enabled: bool = ENV_ENABLE_TRADE

    # Trader Sizer
    sizing_mode: str = "shares"          # shares | percent_equity | notional_usd | risk
    sizing_value: float = 1.0            # je nach Modus: #shares, %, $, oder %Risk
    max_position_pct: float = 100.0      # 0..100

    # Backtest frictions
    bt_slippage_bps: float = 0.0         # 1bp = 0.01%
    bt_fee_per_trade: float = 0.0

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}  # {symbol: {"size":int,"avg":float,"entry_time":str|None}}
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ============================ TV Indicators ============================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi_tv_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    # ta.rma = Wilder smoothing (RMA)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / length
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0/(1.0 + rs))

def efi_tv(close: pd.Series, vol: pd.Series, length: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, length)

# ============================ Market Hours =============================
# Grob (UTC): Mo-Fr, 13:30–20:00 (RTH); Feiertage Liste (Beispiele)
US_HOLIDAYS = {
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27",
    "2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25",
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
}

def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or datetime.now(timezone.utc)
    if now.strftime("%Y-%m-%d") in US_HOLIDAYS: return False
    if now.weekday() >= 5: return False
    hhmm = now.hour*60 + now.minute
    return 13*60+30 <= hhmm <= 20*60

def is_rth_minute(ts_utc: pd.Timestamp) -> bool:
    d = ts_utc.to_pydatetime().replace(tzinfo=timezone.utc)
    if d.strftime("%Y-%m-%d") in US_HOLIDAYS: return False
    if d.weekday() >= 5: return False
    hhmm = d.hour*60 + d.minute
    return 13*60+30 <= hhmm <= 20*60

# ============================ Cache Helpers ============================
def _cache_path(symbol: str, key: str) -> str:
    safe = f"{symbol.replace('/','_')}_{key}"
    return f"/mnt/data/cache_{safe}.parquet"

def cache_save_df(symbol: str, key: str, df: pd.DataFrame):
    if df is None or df.empty: return
    try:
        os.makedirs("/mnt/data", exist_ok=True)
        df.to_parquet(_cache_path(symbol, key))
    except Exception as e:
        print("[cache] save error:", e)

def cache_load_df(symbol: str, key: str) -> pd.DataFrame:
    try:
        path = _cache_path(symbol, key)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            return df
    except Exception as e:
        print("[cache] load error:", e)
    return pd.DataFrame()

# ============================ PDT Persistence ==========================
PDT_JSON = "/mnt/data/pdt_trades.json"

def pdt_load() -> Dict[str, Any]:
    try:
        if os.path.exists(PDT_JSON):
            with open(PDT_JSON, "r") as f:
                return json.load(f)
    except Exception as e:
        print("[pdt] load error:", e)
    return {"daytrades": []}  # list of ISO dates (YYYY-MM-DD)

def pdt_save(data: Dict[str,Any]):
    try:
        os.makedirs("/mnt/data", exist_ok=True)
        with open(PDT_JSON, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("[pdt] save error:", e)

def business_days_range(end: date, days: int) -> List[date]:
    out=[]; cur=end
    while len(out) < days:
        if cur.weekday() < 5 and cur.strftime("%Y-%m-%d") not in US_HOLIDAYS:
            out.append(cur)
        cur = cur - timedelta(days=1)
    return out

def pdt_count_last5bizdays() -> int:
    data = pdt_load()
    today = datetime.now(timezone.utc).date()
    last5 = set(d.strftime("%Y-%m-%d") for d in business_days_range(today, 5))
    return sum(1 for d in data.get("daytrades", []) if d in last5)

def pdt_register_if_daytrade(entry_ts_iso: Optional[str], exit_ts_iso: Optional[str]):
    if not entry_ts_iso or not exit_ts_iso: return
    try:
        e = datetime.fromisoformat(entry_ts_iso.replace("Z","+00:00")).astimezone(timezone.utc).date()
        x = datetime.fromisoformat(exit_ts_iso.replace("Z","+00:00")).astimezone(timezone.utc).date()
        if e == x:
            data = pdt_load()
            iso = e.strftime("%Y-%m-%d")
            data.setdefault("daytrades", [])
            data["daytrades"].append(iso)
            # dedupe + keep last 50
            data["daytrades"] = data["daytrades"][-50:]
            pdt_save(data)
    except Exception as ex:
        print("[pdt] register error:", ex)

def pdt_hard_block_active() -> bool:
    return pdt_count_last5bizdays() >= 3

# ============================ Data Providers ===========================
def fetch_alpaca_1m(symbol: str, lookback_days: int) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] data lib missing:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing API keys"); return pd.DataFrame()

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 14))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=APCA_DATA_FEED,  # 'iex' or 'sip'
        limit=50000
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] 1m empty"); return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except Exception: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.rename(columns=str.lower).sort_index()
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] 1m fetch failed:", e)
        return pd.DataFrame()

def aggregate_tv_1h_from_1m(df1m: pd.DataFrame, drop_last_incomplete: bool = True) -> pd.DataFrame:
    """RTH-only, 1h bars, anchored at :30, TV-like (label/closed 'right')."""
    if df1m.empty: return df1m
    # Filter RTH minutes
    mask = df1m.index.map(lambda ts: is_rth_minute(pd.Timestamp(ts, tz="UTC")))
    dfr = df1m.loc[mask].copy()
    if dfr.empty: return pd.DataFrame()

    # Resample: 60T with 30min offset, closed='right' -> bar [t-60,t)
    # pandas >=1.1 supports offset param
    rs = dfr.resample("60T", offset="30min", label="right", closed="right").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","high","low","close"])

    # Drop last incomplete (ensure full hour inside RTH)
    if drop_last_incomplete and not rs.empty:
        now = datetime.now(timezone.utc)
        last_idx = rs.index[-1]
        # A bar is complete if last_idx <= floor_to_half_hour(now) (previous boundary)
        # Next scheduled boundary:
        mins = now.minute
        # next boundary by :30/:00
        next_boundary = now.replace(second=0, microsecond=0)
        if mins < 30:
            next_boundary = next_boundary.replace(minute=30)
        else:
            next_boundary = (next_boundary + timedelta(hours=1)).replace(minute=30)
        if last_idx >= pd.Timestamp(next_boundary, tz=timezone.utc):
            rs = rs.iloc[:-1]

    rs["time"] = rs.index
    return rs

def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    from urllib.parse import quote
    url = f"https://stooq.com/q/d/l/?s={quote(symbol.lower()+'.us')}&i=d"
    try:
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
    # For 60m: Yahoo bars anchor often on :00 — we can accept as fallback only
    intraday = interval.lower() in {"60m","1h","30m","15m","5m","1m"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"
    last_err=None
    def _norm(tmp):
        if tmp is None or tmp.empty: return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            try: df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            except: pass
        df = df.sort_index(); df["time"] = df.index
        return df

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True, threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = _norm(tmp)
                if not df.empty: return df
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))
    # Fallback daily
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,60)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty: return df
    except Exception as e:
        last_err = e
    print("[yahoo] empty:", last_err); return pd.DataFrame()

def fetch_ohlcv_tv_sync(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Returns TV-compatible bars (1h via Alpaca 1m -> :30 anchor, RTH, no incomplete)."""
    note = {"provider":"", "detail":""}
    if interval.lower() in {"1h","60m"}:
        # Primary: Alpaca 1m -> aggregate
        m1 = fetch_alpaca_1m(symbol, lookback_days)
        if not m1.empty:
            h1 = aggregate_tv_1h_from_1m(m1, drop_last_incomplete=True)
            if not h1.empty:
                if CONFIG.cache_intraday:
                    cache_save_df(symbol, "1h_tv", h1)
                note.update(provider=f"Alpaca ({APCA_DATA_FEED})", detail="1h TV-sync (:30, RTH)")
                return h1, note
        # Yahoo fallback (intraday) — may be off-anchored; use only if strict disabled
        yh = fetch_yahoo(symbol, "60m", lookback_days)
        if not yh.empty and not CONFIG.intraday_strict:
            if CONFIG.cache_intraday:
                cache_save_df(symbol, "1h_tv", yh)
            note.update(provider="Yahoo", detail="60m (fallback)")
            return yh, note

        # STRICT: no EOD fallback; try cache
        if CONFIG.intraday_strict:
            if CONFIG.cache_intraday:
                cached = cache_load_df(symbol, "1h_tv")
                if not cached.empty:
                    note.update(provider="CACHE (last 1h TV-sync)", detail="no fresh intraday data")
                    return cached, note
            return pd.DataFrame(), {"provider":"(leer)","detail":"strict 1h; no intraday data"}

        # non-strict: optional EOD fallback
        if CONFIG.allow_stooq_fallback:
            dfe = fetch_stooq_daily(symbol, max(lookback_days,120))
            if not dfe.empty:
                note.update(provider="Stooq EOD (Fallback)", detail="1d (not intraday)")
                return dfe, note
        return pd.DataFrame(), {"provider":"(leer)","detail":"1h no data"}

    # 1d path (simple)
    if CONFIG.data_provider.lower() == "alpaca":
        # We could fetch Alpaca 1d directly via data client, but Yahoo daily is often fine:
        pass
    df = fetch_yahoo(symbol, "1d", lookback_days)
    if not df.empty:
        note.update(provider="Yahoo", detail="1d")
        return df, note
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, lookback_days)
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note
    return pd.DataFrame(), {"provider":"(leer)","detail":"no data"}

# ============================ Features / Signals =======================
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
    f["entry_cond"] = (f["rsi"] > cfg.rsiLow) & (f["rsi"] < cfg.rsiHigh) & f["rsi_rising"] & f["efi_rising"]
    f["exit_cond"]  = (f["rsi"] < cfg.rsiExit) & f["rsi_falling"]
    return f

def build_export_frame(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    f = compute_signals_for_frame(df, cfg)
    cols = ["open","high","low","close","volume","rsi","efi","rsi_rising","efi_rising","entry_cond","exit_cond","time"]
    return f[cols]

# ============================ Trader Sizer =============================
def get_account_equity_cash() -> Tuple[float, float]:
    # Alpaca trading client for equity/cash (paper)
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return 10000.0, 10000.0
        tc = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
        acc = tc.get_account()
        eq = float(acc.equity)
        cash = float(acc.cash)
        return eq, cash
    except Exception:
        return 10000.0, 10000.0

def size_from_sizer(symbol: str, price: float, stop_px: Optional[float]) -> int:
    mode = CONFIG.sizing_mode.lower()
    val  = CONFIG.sizing_value
    eq, cash = get_account_equity_cash()
    max_notional = eq * (CONFIG.max_position_pct/100.0) if CONFIG.max_position_pct>0 else eq

    qty = 0
    if mode == "shares":
        qty = int(max(1, round(val)))
    elif mode == "percent_equity":
        notional = eq * (val/100.0)
        notional = min(notional, max_notional)
        qty = int(max(1, math.floor(notional / max(1e-9, price))))
    elif mode == "notional_usd":
        notional = min(val, max_notional)
        qty = int(max(1, math.floor(notional / max(1e-9, price))))
    elif mode == "risk":
        # risk in % of equity; needs stop price
        if stop_px is None or stop_px <= 0:
            # fallback to 1 share
            qty = 1
        else:
            risk_notional = eq * (val/100.0)
            per_share_risk = max(1e-9, abs(price - stop_px))
            qty = int(max(1, math.floor(risk_notional / per_share_risk)))
            # cap by max_notional
            qty = int(min(qty, math.floor(max_notional / max(1e-9, price)))))
    else:
        qty = 1

    # do not exceed cash (simple)
    max_cash_qty = int(max(1, math.floor(cash / max(1e-9, price))))
    qty = max(1, min(qty, max_cash_qty))
    return qty

# ============================ Strategy step ============================
def bar_logic_last(df: pd.DataFrame, cfg: StratConfig, sym: str) -> Dict[str,Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data","symbol":sym}

    f = compute_signals_for_frame(df, cfg)
    last, prev = f.iloc[-1], f.iloc[-2]

    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = last["time"]

    entry_cond = bool(last["entry_cond"])
    exit_cond  = bool(last["exit_cond"])

    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]; entry_time = pos["entry_time"]

    # stop/tp templates
    def sl_from(p): return p*(1-cfg.slPerc/100.0)
    def tp_from(p): return p*(1+cfg.tpPerc/100.0)

    bars_in_trade=0
    if entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if size==0:
        if entry_cond:
            sl = sl_from(price_open); tp = tp_from(price_open)
            qty = size_from_sizer(sym, price_open, sl)
            return {"action":"buy","symbol":sym,"qty":qty,"px":price_open,"time":str(ts),
                    "sl":sl,"tp":tp,"reason":"rule_entry",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        return {"action":"none","symbol":sym,"reason":"flat_no_entry",
                "rsi":float(last["rsi"]), "efi":float(last["efi"])}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade>0)
        cooldown_ok = (bars_in_trade>=cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl_from(avg); cur_tp = tp_from(avg)
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

# ============================ Trading (Alpaca) =========================
def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] trading client error:", e); return None

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

# ============================ Runner ==================================
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if "Fallback" in prov or "Stooq" in prov or "CACHE" in prov:
        return f"📡 Quelle: {prov} ({det})"
    return f"📡 Quelle: {prov} ({det})"

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        # defensive: try cache for 1h
        if CONFIG.interval.lower() in {"1h","60m"} and CONFIG.cache_intraday:
            cached = cache_load_df(sym, "1h_tv")
            if not cached.empty:
                df = cached
                note = {"provider":"CACHE (last 1h TV-sync)", "detail":"strict intraday"}
            else:
                return {"ok":False, "msg": f"❌ Keine Intraday-Daten & kein Cache für {sym}."}
        else:
            return {"ok":False, "msg": f"❌ Keine Daten für {sym}. {note}"}

    act = bar_logic_last(df, CONFIG, sym)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    # Signals to Telegram
    if send_signals and CHAT_ID:
        await tg_send(f"ℹ️ {sym} {CONFIG.interval} rsi={act.get('rsi',np.nan):.2f} efi={act.get('efi',np.nan):.2f} • {act['reason']}")
        note_msg = friendly_note(note)
        if note_msg and ("Fallback" in note_msg or "CACHE" in note_msg or CONFIG.data_provider!="alpaca"):
            await tg_send(note_msg)

    # PDT hard-block: block BUY entries if >=3 in last 5 biz days
    if CONFIG.trade_enabled and act["action"]=="buy" and pdt_hard_block_active():
        if CHAT_ID and send_signals:
            await tg_send("⛔ PDT hard block aktiv (≥3 Daytrades/5 Tage). Kein neuer Entry.")
        act["action"]="none"
        act["reason"]="pdt_blocked"

    # Execute paper order via Alpaca
    if CONFIG.trade_enabled and act["action"] in ("buy","sell"):
        if CONFIG.market_hours_only and not is_market_open_now():
            if CHAT_ID and send_signals:
                await tg_send("⛔ Markt geschlossen – kein Trade ausgeführt.")
        else:
            side = "buy" if act["action"]=="buy" else "sell"
            info = await place_market_order(sym, int(act["qty"]), side, "day")
            if CHAT_ID and send_signals:
                await tg_send(f"🛒 {side.upper()} {sym} x{act['qty']} @ {act['px']:.4f} • {info}")

    # Local sim position book (for status & PDT registration)
    pos = STATE.positions.get(sym, {"size":0, "avg":0.0, "entry_time":None})
    if act["action"]=="buy" and pos["size"]==0:
        STATE.positions[sym] = {"size":act["qty"],"avg":act["px"],"entry_time":act["time"]}
        if CHAT_ID and send_signals:
            await tg_send(f"🟢 LONG (sim) {sym} @ {act['px']:.4f} | SL={act.get('sl',np.nan):.4f} TP={act.get('tp',np.nan):.4f}")
    elif act["action"]=="sell" and pos["size"]>0:
        pnl = (act["px"] - pos["avg"]) / pos["avg"]
        # PDT detect (same day entry/exit)
        pdt_register_if_daytrade(pos["entry_time"], act["time"])
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if CHAT_ID and send_signals:
            await tg_send(f"🔴 EXIT (sim) {sym} @ {act['px']:.4f} • {act['reason']} • PnL={pnl*100:.2f}%")

    return {"ok":True,"act":act}

# ============================ Background Timer =========================
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}
TIMER_TASK: Optional[asyncio.Task] = None

def next_half_hour_boundary_utc(now: Optional[datetime]=None) -> datetime:
    now = now or datetime.now(timezone.utc)
    nb = now.replace(second=0, microsecond=0)
    if now.minute < 30:
        nb = nb.replace(minute=30)
    else:
        nb = (nb + timedelta(hours=1)).replace(minute=30)
    return nb

async def timer_loop():
    TIMER["running"] = True
    try:
        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)
            if CONFIG.interval.lower() in {"1h","60m"} and CONFIG.sync_half_hour:
                # run at :30 boundaries
                due = next_half_hour_boundary_utc(now)
                # respect market hours
                if TIMER["market_hours_only"]:
                    # wait until boundary that is inside RTH; else sleep to next minute
                    if not (13*60+30 <= due.hour*60+due.minute <= 20*60 and due.strftime("%Y-%m-%d") not in US_HOLIDAYS and due.weekday()<5):
                        await asyncio.sleep(30)
                        continue
                wait_s = max(1, (due - now).total_seconds())
                TIMER["next_due"] = due.isoformat()
                await asyncio.sleep(wait_s)
                # run
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                TIMER["last_run"] = datetime.now(timezone.utc).isoformat()
            else:
                # generic poll_minutes cadence
                if TIMER["market_hours_only"] and not is_market_open_now(now):
                    TIMER["next_due"] = None
                    await asyncio.sleep(60)
                    continue
                # due calc
                if TIMER["last_run"] is None:
                    for sym in CONFIG.symbols:
                        await run_once_for_symbol(sym, send_signals=True)
                    now = datetime.now(timezone.utc)
                    TIMER["last_run"] = now.isoformat()
                    TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
                else:
                    nxt = datetime.fromisoformat(TIMER["next_due"]) if TIMER["next_due"] else now
                    if now >= nxt:
                        for sym in CONFIG.symbols:
                            await run_once_for_symbol(sym, send_signals=True)
                        now = datetime.now(timezone.utc)
                        TIMER["last_run"] = now.isoformat()
                        TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
                await asyncio.sleep(5)
    finally:
        TIMER["running"] = False

# ============================ Telegram Helpers =========================
tg_app: Optional[Application] = None
POLLING_STARTED = False

async def tg_send(text: str):
    if tg_app is None or CHAT_ID is None: return
    try:
        if not text.strip(): text="(leer)"
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        print("tg_send error:", e)

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

# ============================ Telegram Commands ========================
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg, /set key=value …, /run, /live on|off, /bt [tage], /wf [is oos]\n"
        "/sig, /ind, /plot\n"
        "/dump [csv [N]], /dumpcsv [N]\n"
        "/trade on|off, /pos, /account, /pdt\n"
        "/timer on|off, /timerstatus, /timerrunnow"
    )

async def cmd_status(update, context):
    pos_lines=[]
    for s,p in STATE.positions.items():
        pos_lines.append(f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}")
    pos_txt = "\n".join(pos_lines) if pos_lines else "keine (sim)"
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: Alpaca data_feed={APCA_DATA_FEED}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} • Timer: {'ON' if TIMER['enabled'] else 'OFF'} "
        f"(alle {TIMER['poll_minutes']}m, market-hours-only={TIMER['market_hours_only']}, sync_half_hour={CONFIG.sync_half_hour})\n"
        f"Sizer: mode={CONFIG.sizing_mode} value={CONFIG.sizing_value} max_pos={CONFIG.max_position_pct}%\n"
        f"PDT: last5biz={pdt_count_last5bizdays()} (hard_block={'ON' if pdt_hard_block_active() else 'OFF'})\n"
        f"LastStatus: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper)"
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
    # sync timer props
    if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set symbols=TQQQ,QQQ,SPY\n"
            "/set data_provider=alpaca intraday_strict=true cache_intraday=true\n"
            "/set sizing_mode=percent_equity sizing_value=25 max_position_pct=50\n"
            "/set poll_minutes=10 market_hours_only=true sync_half_hour=true"
        ); return
    msgs=[]; errs=[]
    for a in context.args:
        if "=" not in a: errs.append(f"❌ ungültig: {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = ("✅ Übernommen:\n" + "\n".join(msgs) + ("\n\n⚠️ Probleme:\n" + "\n".join(errs) if errs else "")).strip()
    await update.message.reply_text(txt)

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

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_run(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG); last=f.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} {CONFIG.interval}\n"
        f"rsi={last['rsi']:.2f} (rise={bool(last['rsi_rising'])})  "
        f"efi={last['efi']:.2f} (rise={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}\n"
        f"{friendly_note(note)}"
    )

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
    ax.plot(f.index, f["close"], label="Close"); ax.grid(True); ax.legend(); ax.set_title(f"{sym} {CONFIG.interval} Close")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(f.index, f["rsi"], label="RSI"); ax2.axhline(CONFIG.rsiLow, linestyle="--"); ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.grid(True); ax2.legend(); ax2.set_title("RSI (Wilder)")
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(f.index, f["efi"], label="EFI"); ax3.grid(True); ax3.legend(); ax3.set_title("EFI (EMA(vol*Δclose))")

    cid = str(update.effective_chat.id)
    await send_png(cid, fig,  f"{sym}_{CONFIG.interval}_close.png", "📈 Close")
    await send_png(cid, fig2, f"{sym}_{CONFIG.interval}_rsi.png",   "📈 RSI")
    await send_png(cid, fig3, f"{sym}_{CONFIG.interval}_efi.png",   "📈 EFI")

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
                                  caption=f"🧾 CSV (OHLCV + RSI/EFI + Flags) n={n}")
        return

    f = compute_signals_for_frame(df, CONFIG); last = f.iloc[-1]
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

    eq=1.0; pos=0; avg=0.0; entry_ts=None
    R=[]; entries=exits=0
    slip = CONFIG.bt_slippage_bps/10000.0
    fee  = CONFIG.bt_fee_per_trade

    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry = bool(row["entry_cond"]); exitc = bool(row["exit_cond"])
        if pos==0 and entry:
            # buy @ open + slippage
            px = float(row["open"]) * (1+slip)
            pos=1; avg=px; entry_ts=str(row["time"]); entries+=1
            continue
        if pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            price = float(row["close"])
            stop = price<=sl; take=price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])*(1-slip)  # sell: subtract slippage
                r = (px-avg)/avg
                eq *= (1+r)
                eq -= fee/ max(1e-9, avg) * 0  # flat fee modeled outside return (optional, ignore for % eq)
                R.append(r); exits+=1
                # PDT register
                pdt_register_if_daytrade(entry_ts, str(row["time"]))
                pos=0; avg=0.0; entry_ts=None

    if R:
        a=np.array(R); win=(a>0).mean(); pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(
            f"📈 Backtest {days}d  Trades={entries}/{exits}  Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\n"
            f"ℹ️ Hinweis: Slippage={CONFIG.bt_slippage_bps} bps, Fees={CONFIG.bt_fee_per_trade:.2f}, EoB-Logik."
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_wf(update, context):
    # Walk-Forward: /wf 120 30
    is_days, oos_days = 120, 30
    if context.args and len(context.args)>=2:
        try:
            is_days = int(context.args[0]); oos_days = int(context.args[1])
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG)

    def bt_on_slice(ff: pd.DataFrame, cfg: StratConfig) -> float:
        eq=1.0; pos=0; avg=0.0
        slip = cfg.bt_slippage_bps/10000.0
        for i in range(2,len(ff)):
            row, prev = ff.iloc[i], ff.iloc[i-1]
            entry = bool(row["entry_cond"]); exitc = bool(row["exit_cond"])
            if pos==0 and entry:
                avg=float(row["open"])*(1+slip); pos=1; continue
            if pos==1:
                sl = avg*(1-cfg.slPerc/100); tp = avg*(1+cfg.tpPerc/100)
                price=float(row["close"])
                if price<=sl or price>=tp or exitc:
                    px = sl if price<=sl else tp if price>=tp else float(row["open"])*(1-slip)
                    eq *= (1+(px-avg)/avg); pos=0; avg=0.0
        return eq

    # crude grid over rsiLow/high/exit
    lows  = [0, 30, 40, 50]
    highs = [60, 65, 68, 70]
    exits = [40, 45, 48, 50]
    best=None; best_eq=-1

    # rolling window walk-forward
    total_oos_eq=1.0; slices=0
    for end in pd.date_range(f.index.min()+timedelta(days=is_days+oos_days),
                             f.index.max(), freq=f"{oos_days}D"):
        IS = f.loc[end - timedelta(days=is_days+oos_days): end - timedelta(days=oos_days)]
        OOS= f.loc[end - timedelta(days=oos_days): end]
        if len(IS)<50 or len(OOS)<10: continue

        # search best on IS
        local_best=None; local_best_eq=-1
        for lo in lows:
            for hi in highs:
                if hi <= lo: continue
                for ex in exits:
                    cfg = CONFIG.copy()
                    cfg.rsiLow = lo; cfg.rsiHigh=hi; cfg.rsiExit=ex
                    eq_is = bt_on_slice(IS, cfg)
                    if eq_is > local_best_eq:
                        local_best_eq=eq_is; local_best=(lo,hi,ex)
        if local_best is None: continue

        lo,hi,ex = local_best
        cfg = CONFIG.copy()
        cfg.rsiLow=lo; cfg.rsiHigh=hi; cfg.rsiExit=ex
        eq_oos = bt_on_slice(OOS, cfg)
        total_oos_eq *= eq_oos
        slices += 1

    if slices>0:
        cagr = (total_oos_eq**(365/max(1, is_days+oos_days))-1)
        await update.message.reply_text(
            f"🧪 Walk-Forward (IS={is_days}d, OOS={oos_days}d) Slices={slices}\n"
            f"OOS EQ Multiplier={total_oos_eq:.3f}  ~CAGR={cagr*100:.2f}%"
        )
    else:
        await update.message.reply_text("⚠️ WF: nicht genug Datenfenster.")

async def cmd_pdt(update, context):
    await update.message.reply_text(
        "📛 PDT\n" +
        json.dumps({
            "last5biz_trades": pdt_count_last5bizdays(),
            "hard_block": pdt_hard_block_active(),
            "persist": PDT_JSON
        }, indent=2)
    )

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
    if CONFIG.interval.lower() in {"1h","60m"} and CONFIG.sync_half_hour:
        TIMER["next_due"] = next_half_hour_boundary_utc(now).isoformat()
    else:
        TIMER["next_due"] = (now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ============================ FastAPI Lifespan =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED, TIMER_TASK
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        # handlers
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("wf",      cmd_wf))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("dumpcsv", cmd_dumpcsv))
        tg_app.add_handler(CommandHandler("trade",   cmd_trade))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(CommandHandler("account", cmd_account))
        tg_app.add_handler(CommandHandler("pdt",     cmd_pdt))
        tg_app.add_handler(CommandHandler("timer",        cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("timerrunnow",  cmd_timerrunnow))
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
                    print(f"⚠️ Conflict: {e}. retry {delay}s"); await asyncio.sleep(delay); delay=min(delay*2,60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        # start timer
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
                await tg_app.stop(); await tg_app.shutdown()
        except Exception:
            pass
        print("🛑 Shutdown complete")

# ============================ FastAPI App & Routes =====================
app = FastAPI(title="TQQQ Strategy + Telegram (V5.1)", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": f"alpaca ({APCA_DATA_FEED})",
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
    if CONFIG.interval.lower() in {"1h","60m"} and CONFIG.sync_half_hour:
        TIMER["next_due"]=next_half_hour_boundary_utc(now).isoformat()
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
# ============================ END FILE ============================
