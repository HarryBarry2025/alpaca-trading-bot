# alpaca_trading_bot.py
import os, io, json, time, asyncio, traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.7)
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict, BadRequest

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zoneinfo import ZoneInfo

# ----- optional US holidays -----
try:
    import holidays as _hol  # pip install holidays
    US_HOL = _hol.US()
except Exception:
    US_HOL = None  # stiller Fallback: keine Feiertage berücksichtigt

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")  # ungenutzt im Polling
BASE_URL        = os.getenv("BASE_URL", "")  # ungenutzt im Polling

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")  # Paper default

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    # Symbole & Data
    symbol: str = "TQQQ"
    interval: str = "1h"     # '1h' (intraday) oder '1d'
    lookback_days: int = 365

    # Indikator-Parameter (TV-kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0         # gewünscht: 0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    macdFast: int = 12          # nur Plot/Info; nicht im Entry genutzt
    macdSlow: int = 26
    macdSig: int = 9
    efiLen: int = 11

    # Risk (SL/TP in %)
    slPerc: float = 1.0
    tpPerc: float = 4.0

    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine
    poll_minutes: int = 10
    live_enabled: bool = False
    market_hours_only: bool = True

    # Data Provider
    data_provider: str = "alpaca"        # "alpaca" (default), "yahoo", "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Trading
    trade_enabled: bool = False
    time_in_force: str = "day"           # "day" oder "gtc"
    sizing_mode: str = "shares"          # "shares" | "notional"
    shares: int = 1
    notional_usd: float = 1000.0

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_status: str = "idle"

    # Timer
    timer_enabled: bool = True
    timer_running: bool = False
    last_run: Optional[str] = None
    next_due: Optional[str] = None

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= Indicators (TV-kompatibel) =========
def ema_tv(s: pd.Series, length: int) -> pd.Series:
    # TV-EMA entspricht pandas ewm(alpha=2/(len+1))
    return s.ewm(alpha=2.0/(length+1.0), adjust=False).mean()

def rsi_tv_wilder(close: pd.Series, length: int) -> pd.Series:
    # Wilder RSI (TV-kompatibel)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing == RMA == EMA mit alpha=1/length
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - 100/(1 + rs)
    rsi = rsi.fillna(50.0)
    return rsi

def macd_tv(close: pd.Series, fast: int, slow: int, sig: int):
    fast_ = ema_tv(close, fast)
    slow_ = ema_tv(close, slow)
    line = fast_ - slow_
    signal = ema_tv(line, sig)
    hist = line - signal
    return line, signal, hist

def efi_tv(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    raw = volume * (close - close.shift(1))
    return ema_tv(raw.fillna(0), length)

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
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
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
    start = end - timedelta(days=max(lookback_days, 60))

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
            print("[alpaca] empty frame (feed/plan/interval?)")
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
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}

    # Preferred provider
    prov = CONFIG.data_provider.lower()

    if prov == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=interval)
            return df, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")

    if prov == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        note.update(provider="Stooq→Yahoo", detail="Stooq leer; versuche Yahoo")

    # Yahoo
    is_intraday = interval.lower() in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _norm(tmp: pd.DataFrame) -> pd.DataFrame:
        if tmp is None or tmp.empty:
            return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
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
        df["time"] = df.index
        return df

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True, threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]

    last_err = None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                if method == "download":
                    tmp = yf.download(**kwargs)
                else:
                    tmp = yf.Ticker(symbol).history(**kwargs)
                df = _norm(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}")
                    return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # Yahoo Fallback: 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 60)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # Optional Stooq EOD
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] no data for {symbol} ({interval}, {period}). last_err={last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Features =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv_wilder(out["close"], cfg.rsiLen)
    macd_line, macd_sig, macd_hist = macd_tv(out["close"], cfg.macdFast, cfg.macdSlow, cfg.macdSig)
    out["macd_line"], out["macd_sig"], out["macd_hist"] = macd_line, macd_sig, macd_hist
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

# ========= Strategy (Entry ohne MACD) =========
def bar_logic(fdf: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if fdf.empty or len(fdf) < max(cfg.rsiLen, cfg.macdSlow, cfg.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data"}

    last = fdf.iloc[-1]
    prev = fdf.iloc[-2]

    rsi_val = last["rsi"]
    rsi_rising  = last["rsi"] > prev["rsi"]
    rsi_falling = last["rsi"] < prev["rsi"]
    efi_rising  = last["efi"] > prev["efi"]

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    o = float(last["open"])
    c = float(last["close"])
    ts = last["time"]

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    # Bars seit Entry (nur Info)
    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = fdf[fdf["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if st.position_size == 0:
        if entry_cond:
            size = 1
            return {"action":"buy","qty":size,"price":o,"time":str(ts),"reason":"rule_entry",
                    "sl":sl(o),"tp":tp(o)}
        return {"action":"none","reason":"flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(st.avg_price)
        cur_tp = tp(st.avg_price)
        hit_sl = c <= cur_sl
        hit_tp = c >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","qty":st.position_size,"price":o,"time":str(ts),"reason":"rsi_exit"}
        if hit_sl:
            return {"action":"sell","qty":st.position_size,"price":cur_sl,"time":str(ts),"reason":"stop_loss"}
        if hit_tp:
            return {"action":"sell","qty":st.position_size,"price":cur_tp,"time":str(ts),"reason":"take_profit"}
        return {"action":"none","reason":"hold"}

# ========= US Session / Holidays =========
def is_us_holiday(d_utc: datetime) -> bool:
    if US_HOL is None:
        return False
    d_ny = d_utc.astimezone(ZoneInfo("America/New_York")).date()
    return d_ny in US_HOL

def in_us_rth(now_utc: Optional[datetime] = None) -> bool:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    if is_us_holiday(now_utc):
        return False
    start = ny.replace(hour=9, minute=30, second=0, microsecond=0)
    end   = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= ny <= end

def us_session_mask(index_utc: pd.DatetimeIndex) -> pd.Series:
    if not isinstance(index_utc, pd.DatetimeIndex):
        index_utc = pd.DatetimeIndex(index_utc)
    ny = index_utc.tz_convert(ZoneInfo("America/New_York"))
    is_weekday = ny.weekday < 5
    start = ny.normalize() + pd.Timedelta(hours=9, minutes=30)
    end   = ny.normalize() + pd.Timedelta(hours=16)
    in_hours = (ny >= start) & (ny <= end)
    if US_HOL is not None:
        hol_mask = ny.date.astype("O").copy()
        hols = np.array([dt in US_HOL for dt in hol_mask], dtype=bool)
        return is_weekday & in_hours & (~hols)
    return is_weekday & in_hours

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
tg_running = False
poll_task: Optional[asyncio.Task] = None
timer_task: Optional[asyncio.Task] = None

async def send(chat_id: str, text: str):
    if tg_app is None:
        return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text or "ℹ️")
    except Exception as e:
        print("send error:", e)

def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if prov.startswith("Yahoo (Fallback"):
        return f"📡 Daten: {prov} – {det}"
    if "Stooq" in prov:
        return f"📡 Daten: {prov} – {det} (nur Daily)"
    return f"📡 Daten: {prov} – {det}"

# ========= Plot Helpers =========
def simulate_trades_for_plot(fdf: pd.DataFrame, cfg: StratConfig):
    rsi = fdf["rsi"]; efi = fdf["efi"]
    rsi_rising  = rsi > rsi.shift(1)
    rsi_falling = rsi < rsi.shift(1)
    efi_rising  = efi > efi.shift(1)
    entry = (rsi > cfg.rsiLow) & (rsi < cfg.rsiHigh) & rsi_rising & efi_rising
    exitc = (rsi < cfg.rsiExit) & rsi_falling

    pos = 0; avg = None
    entries, exits = [], []
    last_sl = last_tp = None

    for i in range(1, len(fdf)):
        row = fdf.iloc[i]
        ts, op, cl = row["time"], float(row["open"]), float(row["close"])
        if pos == 0 and bool(entry.iloc[i]):
            pos = 1; avg = op
            entries.append((ts, op))
            last_sl = avg*(1-cfg.slPerc/100.0)
            last_tp = avg*(1+cfg.tpPerc/100.0)
        elif pos == 1:
            stop = cl <= avg*(1-cfg.slPerc/100.0)
            take = cl >= avg*(1+cfg.tpPerc/100.0)
            rxi  = bool(exitc.iloc[i])
            if stop or take or rxi:
                px = avg*(1-cfg.slPerc/100.0) if stop else avg*(1+cfg.tpPerc/100.0) if take else op
                exits.append((ts, px))
                pos = 0; avg = None
                last_sl = last_tp = None
            else:
                last_sl = avg*(1-cfg.slPerc/100.0)
                last_tp = avg*(1+cfg.tpPerc/100.0)
    return {"entries":entries, "exits":exits, "last_sl":last_sl, "last_tp":last_tp}

def make_plot_image(fdf: pd.DataFrame, cfg: StratConfig, bars: int = 200, what: str = "all") -> bytes:
    df = fdf.tail(max(100, bars)).copy()
    ts = df["time"]
    mask = us_session_mask(df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(ts))
    sim = simulate_trades_for_plot(df, cfg)

    panels = []
    if what in ("all","price"): panels.append("price")
    if what in ("all","rsi"):   panels.append("rsi")
    if what in ("all","efi"):   panels.append("efi")
    if what in ("all","macd"):
        if "macd_line" in df and "macd_sig" in df and "macd_hist" in df:
            panels.append("macd")
    rows = len(panels) or 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 8), sharex=True)
    if rows == 1: axes = [axes]
    idx = 0

    if "price" in panels:
        ax = axes[idx]; idx += 1
        ax.plot(ts, df["close"], label=f"{cfg.symbol} Close")
        ax.fill_between(ts, df["close"].min(), df["close"].max(), where=~mask, color="gray", alpha=0.08, step="pre")
        # markers
        if sim["entries"]:
            e_ts = [t for t,_ in sim["entries"]]; e_px = [p for _,p in sim["entries"]]
            ax.scatter(e_ts, e_px, marker="^", color="tab:green", zorder=5, label="Entry")
        if sim["exits"]:
            x_ts = [t for t,_ in sim["exits"]]; x_px = [p for _,p in sim["exits"]]
            ax.scatter(x_ts, x_px, marker="v", color="tab:red", zorder=5, label="Exit")
        if sim["last_sl"] is not None: ax.axhline(sim["last_sl"], color="tab:red", linestyle="--", linewidth=1, label="SL")
        if sim["last_tp"] is not None: ax.axhline(sim["last_tp"], color="tab:green", linestyle="--", linewidth=1, label="TP")
        ax.set_title(f"{cfg.symbol} – Close ({cfg.interval}) • Grau=außerhalb RTH")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    if "rsi" in panels:
        ax = axes[idx]; idx += 1
        ax.plot(ts, df["rsi"], label=f"RSI({cfg.rsiLen})")
        ax.axhline(cfg.rsiLow,  color="tab:gray", linestyle="--", linewidth=1)
        ax.axhline(cfg.rsiHigh, color="tab:gray", linestyle="--", linewidth=1)
        ax.axhline(cfg.rsiExit, color="tab:red",  linestyle="--", linewidth=1, label="RSI Exit")
        ax.set_ylim(0, 100); ax.set_title("RSI")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    if "efi" in panels:
        ax = axes[idx]; idx += 1
        ax.plot(ts, df["efi"], label=f"EFI({cfg.efiLen})")
        ax.axhline(0.0, color="tab:gray", linewidth=1)
        ax.set_title("EFI")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    if "macd" in panels:
        ax = axes[idx]; idx += 1
        ax.plot(ts, df["macd_line"], label="MACD line")
        ax.plot(ts, df["macd_sig"],  label="Signal")
        ax.bar(ts, df["macd_hist"], width=0.8, alpha=0.3, label="Hist")
        ax.set_title("MACD")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    plt.tight_layout()
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()

# ========= Trading (Alpaca) =========
def alpaca_clients_or_none():
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        return None, None, None
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        trading_client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper="paper" in APCA_API_BASE_URL)
        return trading_client, MarketOrderRequest, (OrderSide, TimeInForce)
    except Exception as e:
        print("[alpaca] trading lib unavailable:", e)
        return None, None, None

async def place_alpaca_order(side: str, qty: Optional[int], notional: Optional[float], tif: str):
    trading_client, MarketOrderRequest, Enums = alpaca_clients_or_none()
    if trading_client is None:
        return False, "Alpaca client not ready"
    OrderSide, TimeInForce = Enums
    try:
        kwargs = dict(symbol=CONFIG.symbol,
                      side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
                      time_in_force=TimeInForce.DAY if tif.lower()=="day" else TimeInForce.GTC)
        if CONFIG.sizing_mode == "notional":
            kwargs["notional"] = float(notional or CONFIG.notional_usd)
        else:
            kwargs["qty"] = int(qty or CONFIG.shares)
        order_req = MarketOrderRequest(**kwargs)
        order = trading_client.submit_order(order_req)
        return True, f"Alpaca order ok: {order.id}"
    except Exception as e:
        return False, f"Alpaca order failed: {e}"

def get_alpaca_positions_text() -> str:
    trading_client, _, _ = alpaca_clients_or_none()
    if trading_client is None:
        return "Alpaca nicht konfiguriert."
    try:
        pos = trading_client.get_all_positions()
        if not pos:
            return "Keine offenen Positionen."
        lines = ["📦 Alpaca Positionen:"]
        for p in pos:
            lines.append(f"- {p.symbol}: {p.qty} @ {p.avg_entry_price}  (unrealized PnL ${p.unrealized_pl})")
        return "\n".join(lines)
    except Exception as e:
        return f"Fehler beim Abruf der Positionen: {e}"

# ========= One-run Engine =========
async def run_once_and_report(chat_id: str):
    # RTH check
    if CONFIG.market_hours_only and not in_us_rth():
        await send(chat_id, "⏸️ Außerhalb US-Handelszeit (09:30–16:00 ET). Kein Run.")
        return
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten. Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return

    note_msg = _friendly_data_note(note)
    if note_msg and ("Fallback" in note_msg or "Stooq" in note_msg or CONFIG.data_provider!="alpaca"):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    if act["action"] == "buy" and STATE.position_size == 0:
        STATE.position_size = act["qty"]
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        await send(chat_id, f"🟢 LONG @ {STATE.avg_price:.4f}  SL={act['sl']:.4f}  TP={act['tp']:.4f}")
        if CONFIG.trade_enabled:
            ok, msg = await place_alpaca_order("buy",
                                               qty=CONFIG.shares if CONFIG.sizing_mode=="shares" else None,
                                               notional=CONFIG.notional_usd if CONFIG.sizing_mode=="notional" else None,
                                               tif=CONFIG.time_in_force)
            await send(chat_id, ("✅ " if ok else "❌ ") + msg)

    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} [{act['reason']}]  PnL={pnl*100:.2f}%")
        if CONFIG.trade_enabled:
            ok, msg = await place_alpaca_order("sell",
                                               qty=STATE.position_size if CONFIG.sizing_mode=="shares" else None,
                                               notional=None if CONFIG.sizing_mode=="shares" else CONFIG.notional_usd,
                                               tif=CONFIG.time_in_force)
            await send(chat_id, ("✅ " if ok else "❌ ") + msg)
        STATE.position_size = 0
        STATE.avg_price = 0.0
        STATE.entry_time = None
    else:
        await send(chat_id, f"ℹ️ {STATE.last_status}")

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status – Status\n"
        "/cfg – Konfiguration\n"
        "/set key=value …  (z.B. /set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=4)\n"
        "/run – einmaliger Check jetzt\n"
        "/live on|off – Autolauf im Hintergrund\n"
        "/bt [tage] – Backtest\n"
        "/positions – Alpaca-Positionen\n"
        "/ind – Indikatorwerte (letzte Bar)\n"
        "/plot [bars] [all|price|rsi|efi|macd] – Chart mit Markern"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbol: {CONFIG.symbol}  TF={CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}  poll={CONFIG.poll_minutes}m\n"
        f"RTH-only: {'ON' if CONFIG.market_hours_only else 'OFF'}  Holidays: {'ON' if US_HOL else 'OFF'}\n"
        f"Trade: {'ON' if CONFIG.trade_enabled else 'OFF'}  TIF={CONFIG.time_in_force}  Sizing={CONFIG.sizing_mode}\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Timer: enabled={STATE.timer_enabled} running={STATE.timer_running}\n"
        f"Last run: {STATE.last_run}  Next due: {STATE.next_due}"
    )

def _set_one(k:str, v:str) -> str:
    mapping = {"sl":"slPerc","tp":"tpPerc","samebar":"allowSameBarExit","cooldown":"minBarsInTrade",
               "tif":"time_in_force","trade":"trade_enabled","rth":"market_hours_only"}
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
        return f"✓ {k} = {getattr(CONFIG,k)}"
    except Exception as e:
        return f"❌ {k}: {e}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set data_provider=alpaca | yahoo | stooq_eod\n"
            "/set trade=on tif=day sizing_mode=shares shares=1\n"
            "/set sizing_mode=notional notional_usd=1000"
        )
        return
    msgs = []
    for a in context.args:
        if "=" not in a: msgs.append(f"❌ Ungültig: {a}"); continue
        k,v = a.split("=",1)
        msgs.append(_set_one(k.strip(), v.strip()))
    await update.message.reply_text("\n".join(msgs))

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off")
        return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_trade(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /trade on|off")
        return
    on = context.args[0].lower() in ("on","1","true")
    CONFIG.trade_enabled = on
    await update.message.reply_text(f"Trading = {'ON' if on else 'OFF'}")

async def cmd_positions(update, context):
    await update.message.reply_text(get_alpaca_positions_text())

async def cmd_ind(update, context):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    row, prev = fdf.iloc[-1], fdf.iloc[-2]
    text = (
        f"📐 Indikatoren ({CONFIG.symbol} {CONFIG.interval})\n"
        f"Close={row['close']:.4f}\n"
        f"RSI({CONFIG.rsiLen})={row['rsi']:.2f}  Δ={row['rsi']-prev['rsi']:+.2f}\n"
        f"EFI({CONFIG.efiLen})={row['efi']:.2f}  Δ={row['efi']-prev['efi']:+.2f}\n"
        f"MACD={row['macd_line']:.4f}, Signal={row['macd_sig']:.4f}, Hist={row['macd_hist']:.4f}"
    )
    note_msg = _friendly_data_note(note)
    if note_msg: text += f"\n{note_msg}"
    await update.message.reply_text(text)

async def cmd_plot(update, context):
    bars = 200; what = "all"
    if context.args:
        for a in context.args:
            a=a.lower()
            if a.isdigit(): bars = max(50, int(a))
            elif a in ("all","price","rsi","efi","macd"): what=a
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten zum Plotten.")
        return
    fdf = build_features(df, CONFIG)
    png = make_plot_image(fdf, CONFIG, bars=bars, what=what)
    cap = _friendly_data_note(note) or f"{CONFIG.symbol} {CONFIG.interval} • {what}"
    try:
        await update.message.reply_photo(png, caption=cap)
    except Exception as e:
        await update.message.reply_text(f"❌ Plot-Sendefehler: {e}")

async def cmd_run(update, context):
    await run_once_and_report(str(update.effective_chat.id))

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten. Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return
    fdf = build_features(df, CONFIG)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = row["rsi"]
        rsi_rising  = row["rsi"] > prev["rsi"]
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
                r  = (px-avg)/avg
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean()
        pf=(a[a>0].sum())/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr = (eq**(365/max(1,days)) - 1)
        note_msg=_friendly_data_note(note)
        await update.message.reply_text(
            f"📈 Backtest {days}d  Trades {entries}/{exits}\n"
            f"Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\n"
            f"{note_msg}"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Background timer =========
async def background_timer_loop():
    STATE.timer_running = True
    try:
        due_at: Optional[datetime] = None
        while True:
            await asyncio.sleep(30)  # tick alle 30s
            if not CONFIG.live_enabled or not STATE.timer_enabled:
                continue
            if CONFIG.market_hours_only and not in_us_rth():
                continue
            now = datetime.now(timezone.utc)
            if due_at is None:
                due_at = now
            if now >= due_at:
                if CHAT_ID:
                    try:
                        await run_once_and_report(CHAT_ID)
                    except Exception:
                        traceback.print_exc()
                STATE.last_run = now.isoformat()
                due_at = now + timedelta(minutes=max(1, CONFIG.poll_minutes))
                STATE.next_due = due_at.isoformat()
    finally:
        STATE.timer_running = False

# ========= FastAPI lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, poll_task, timer_task
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",      cmd_start))
        tg_app.add_handler(CommandHandler("status",     cmd_status))
        tg_app.add_handler(CommandHandler("cfg",        cmd_cfg))
        tg_app.add_handler(CommandHandler("set",        cmd_set))
        tg_app.add_handler(CommandHandler("run",        cmd_run))
        tg_app.add_handler(CommandHandler("live",       cmd_live))
        tg_app.add_handler(CommandHandler("bt",         cmd_bt))
        tg_app.add_handler(CommandHandler("positions",  cmd_positions))
        tg_app.add_handler(CommandHandler("ind",        cmd_ind))
        tg_app.add_handler(CommandHandler("plot",       cmd_plot))
        tg_app.add_handler(CommandHandler("trade",      cmd_trade))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()

        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht")
        except Exception as e:
            print("delete_webhook warn:", e)

        # Polling starten (einfach, konfliktarm)
        async def _start_polling():
            delay = 5
            while True:
                try:
                    print("▶️ starte Telegram-Polling…")
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    print("✅ Polling läuft")
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e} – retry in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc()
                    await asyncio.sleep(10)

        poll_task = asyncio.create_task(_start_polling())

        # Background Timer
        timer_task = asyncio.create_task(background_timer_loop())

        tg_running = True
        yield
    finally:
        tg_running = False
        if poll_task:
            try: await tg_app.updater.stop()
            except Exception: pass
            poll_task.cancel()
        if timer_task:
            timer_task.cancel()
        try: await tg_app.stop()
        except Exception: pass
        try: await tg_app.shutdown()
        except Exception: pass
        print("🛑 Shutdown complete.")

# ========= FastAPI App & routes =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

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
    if CONFIG.market_hours_only and not in_us_rth():
        return {"ran": False, "reason": "outside_rth"}
    if CHAT_ID:
        await run_once_and_report(CHAT_ID)
        return {"ran": True}
    return {"ran": False, "reason": "no_chat_id"}

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
    return {
        "tg_running": tg_running
    }

@app.get("/timer")
def timer_status():
    return {
        "enabled": STATE.timer_enabled,
        "running": STATE.timer_running,
        "poll_minutes": CONFIG.poll_minutes,
        "last_run": STATE.last_run,
        "next_due": STATE.next_due,
        "market_hours_only": CONFIG.market_hours_only
    }
