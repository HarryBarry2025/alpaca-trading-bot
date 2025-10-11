# alpaca_trading_bot.py
import os, io, json, time, asyncio, traceback, math
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
from telegram.ext import (
    Application, ApplicationBuilder, ContextTypes,
    CommandHandler, MessageHandler, filters
)
from telegram.error import Conflict, BadRequest

# Optional: Matplotlib für /plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")  # ungenutzt im Polling-Modus
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")  # optional

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional, nur wenn trade_enabled oder data_provider=alpaca genutzt wird)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # z.B. https://paper-api.alpaca.markets

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    # Data / Engine
    symbol: str = "TQQQ"
    interval: str = "1h"
    lookback_days: int = 365

    data_provider: str = "alpaca"   # "alpaca" | "yahoo" | "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Strategy Inputs (TV-kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0     # explizit 0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
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
    max_pos: int = 1

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_action_bar_index: Optional[int] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ========= Timer State =========
TIMER_STATE: Dict[str, Any] = {
    "enabled": False,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only,
    "last_reason": "",
}
_timer_task: Optional[asyncio.Task] = None

# ========= US Market Hours & Holidays =========
def is_us_holiday(dt_utc: datetime) -> bool:
    if not CONFIG.use_us_holidays:
        return False
    try:
        import holidays
        y = dt_utc.year
        us_h = holidays.US(years=[y, y+1])
        # NYSE schließt auch an manchen Tagen vorzeitig, vereinfachen:
        return dt_utc.date() in us_h
    except Exception:
        # Minimal-Set häufigster Feiertage (Approx)
        y = dt_utc.year
        common = {
            # Neujahr (verschoben, wenn am WE) – grob: 1. Jan
            datetime(y, 1, 1).date(),
            # Independence Day
            datetime(y, 7, 4).date(),
            # Christmas
            datetime(y, 12, 25).date(),
            # Thanksgiving (4. Do. im Nov) – Approx: 22–28
        }
        if dt_utc.month == 11 and dt_utc.weekday() == 3 and 22 <= dt_utc.day <= 28:
            return True
        return dt_utc.date() in common

def is_rth(dt_utc: datetime) -> bool:
    # NYSE Regular Trading Hours: 13:30–20:00 UTC (9:30–16:00 ET), Mo–Fr außer Feiertage
    if dt_utc.weekday() >= 5:  # Sa/So
        return False
    if is_us_holiday(dt_utc):
        return False
    hhmm = dt_utc.hour * 60 + dt_utc.minute
    start = 13*60 + 30
    end   = 20*60
    return start <= hhmm < end

# ========= Indicators (TV-kompatibel) =========
def _ema_tv(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rsi_tv(df: pd.DataFrame, length: int, source: str = "close") -> pd.Series:
    if source == "hlc3" and {"high", "low", "close"}.issubset(df.columns):
        src = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        src = df["close"]
    delta = src.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder Smoothing (EMA mit alpha=1/length)
    alpha = 1.0 / float(length)
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def efi_tv(df: pd.DataFrame, length: int) -> pd.Series:
    # Elder’s Force Index (EMA der Force: Vol * ΔClose)
    close = df["close"]
    vol   = df["volume"].fillna(0.0)
    force = vol * (close - close.shift(1))
    return _ema_tv(force, length)

def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv(out, cfg.rsiLen, cfg.rsi_source)
    out["efi"] = efi_tv(out, cfg.efiLen)
    return out

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
        return df
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
    start = end - timedelta(days=max(lookback_days, 30))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        adjustment=None,
        feed="iex",   # wichtig für free/paper
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
        df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        # Alpaca liefert oft already lowercased, aber rename ist safe
        df["time"] = df.index
        cols = ["open","high","low","close","volume","time"]
        return df[[c for c in cols if c in df.columns]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}

    # Alpaca bevorzugt
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=f"{interval}")
            return df, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")

    # Stooq explizit
    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        note.update(provider="Stooq→Yahoo", detail="Stooq leer; versuche Yahoo")

    # Yahoo
    is_intraday = interval in intraday_set
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

    # Yahoo Fallback: 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # Optional Stooq-Fallback
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] keine Daten für {symbol} ({interval}, period={period}). Last error: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)", "detail":"keine Daten"}

# ========= Strategy Logic =========
def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_val     = float(last["rsi"])
    rsi_prev    = float(prev["rsi"])
    efi_val     = float(last["efi"])
    efi_prev    = float(prev["efi"])

    rsi_rising  = rsi_val > rsi_prev
    rsi_falling = rsi_val < rsi_prev
    efi_rising  = efi_val > efi_prev

    entry_cond = (rsi_val > CONFIG.rsiLow) and (rsi_val < CONFIG.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < CONFIG.rsiExit) and rsi_falling

    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = last["time"]

    def sl(p): return p * (1 - CONFIG.slPerc/100.0)
    def tp(p): return p * (1 + CONFIG.tpPerc/100.0)

    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if st.position_size == 0:
        if entry_cond:
            return {"action":"buy","qty":1,"price":price_open,"time":str(ts),
                    "reason":"rule_entry","sl":sl(price_open),"tp":tp(price_open)}
        return {"action":"none","reason":"flat_no_entry"}
    else:
        same_bar_ok = CONFIG.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= CONFIG.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(STATE.avg_price); cur_tp = tp(STATE.avg_price)
        hit_sl = price_close <= cur_sl
        hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","qty":STATE.position_size,"price":price_open,"time":str(ts),"reason":"rsi_exit"}
        if hit_sl:
            return {"action":"sell","qty":STATE.position_size,"price":cur_sl,"time":str(ts),"reason":"stop_loss"}
        if hit_tp:
            return {"action":"sell","qty":STATE.position_size,"price":cur_tp,"time":str(ts),"reason":"take_profit"}
        return {"action":"none","reason":"hold"}

# ========= Trading via Alpaca (optional) =========
def _alpaca_trader():
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None, None, None
        # paper=True unabhängig von APCA_API_BASE_URL
        client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
        return client, MarketOrderRequest, TimeInForce
    except Exception as e:
        print("[alpaca trade] library not available:", e)
        return None, None, None

async def place_order_alpaca(side: str, qty: Optional[int]=None, notional: Optional[float]=None) -> str:
    client, MarketOrderRequest, TimeInForce = _alpaca_trader()
    if client is None:
        return "alpaca_client_unavailable"

    tif_map = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }
    tif = tif_map.get(CONFIG.tif.lower(), TimeInForce.DAY)

    try:
        if CONFIG.sizing_mode == "notional":
            if notional is None: notional = CONFIG.notional_usd
            order_data = MarketOrderRequest(
                symbol=CONFIG.symbol,
                notional=float(notional),
                side=("buy" if side=="buy" else "sell"),
                time_in_force=tif
            )
        else:
            if qty is None: qty = CONFIG.fixed_shares
            order_data = MarketOrderRequest(
                symbol=CONFIG.symbol,
                qty=int(qty),
                side=("buy" if side=="buy" else "sell"),
                time_in_force=tif
            )
        order = client.submit_order(order_data)
        return f"alpaca_ok id={getattr(order,'id','?')}"
    except Exception as e:
        return f"alpaca_err: {e}"

def get_alpaca_positions() -> Tuple[str, Optional[pd.DataFrame]]:
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
        pos = client.get_all_positions()
        if not pos:
            return "keine Positionen", pd.DataFrame()
        rows = []
        for p in pos:
            rows.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_price": float(p.market_price),
                "unrealized_pl": float(p.unrealized_pl),
                "side": p.side
            })
        df = pd.DataFrame(rows)
        return "ok", df
    except Exception as e:
        return f"alpaca_err: {e}", None

# ========= Friendly data note =========
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

# ========= Indicator snapshot =========
def format_indicator_snapshot(fdf: pd.DataFrame, cfg: StratConfig) -> str:
    if len(fdf) < 2:
        return "🔎 Zu wenig Daten für Indikator-Snapshot."
    row, prev = fdf.iloc[-1], fdf.iloc[-2]
    rsi_val, rsi_prev = float(row["rsi"]), float(prev["rsi"])
    efi_val, efi_prev = float(row["efi"]), float(prev["efi"])
    up = "↑"; down = "↓"; flat = "→"
    rsi_arrow = up if rsi_val > rsi_prev else down if rsi_val < rsi_prev else flat
    efi_arrow = up if efi_val > efi_prev else down if efi_val < efi_prev else flat
    ts = row["time"]
    return (f"🕒 {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"RSI({cfg.rsiLen},{cfg.rsi_source}): {rsi_val:.2f} {rsi_arrow} (prev {rsi_prev:.2f})\n"
            f"EFI({cfg.efiLen}): {efi_val:.2f} {efi_arrow} (prev {efi_prev:.2f})")

# ========= Strategy run & reports =========
async def run_once_and_report(chat_id: str, place_order: bool=False):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten für {CONFIG.symbol} ({CONFIG.interval}). "
                            f"Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return
    note_msg = _friendly_data_note(note)
    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    # Indikatoren bei jedem Run zusätzlich anzeigen?
    # (Timer macht das ohnehin; hier lassen wir es minimal)
    if act["action"] == "buy":
        # Positionslogik lokal
        if STATE.position_size <= 0:
            STATE.position_size = act["qty"]
            STATE.avg_price = float(act["price"])
            STATE.entry_time = act["time"]

        txt = f"🟢 LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f}  TP={act['tp']:.4f}"
        if note_msg: txt = f"{note_msg}\n{txt}"

        # Alpaca Order
        if place_order:
            res = await place_order_alpaca("buy",
                                           qty=(STATE.position_size if CONFIG.sizing_mode=="shares" else None),
                                           notional=(CONFIG.notional_usd if CONFIG.sizing_mode=="notional" else None))
            txt += f"\n📬 Order: {res}"
        await send(chat_id, txt)

    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        txt = f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\nPnL={pnl*100:.2f}%"
        if place_order:
            res = await place_order_alpaca("sell",
                                           qty=(STATE.position_size if CONFIG.sizing_mode=="shares" else None),
                                           notional=(CONFIG.notional_usd if CONFIG.sizing_mode=="notional" else None))
            txt += f"\n📬 Order: {res}"
        await send(chat_id, txt)
        STATE.position_size = 0
        STATE.avg_price = 0.0
        STATE.entry_time = None
    else:
        txt = f"ℹ️ {STATE.last_status}"
        if note_msg and ("Fallback" in note_msg or CONFIG.data_provider!="alpaca"):
            txt = f"{note_msg}\n{txt}"
        await send(chat_id, txt)

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        if not text.strip():
            text = "ℹ️ (leer)"
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except BadRequest as e:
        print("send badrequest:", e)
    except Exception as e:
        print("send error:", e)

# ========= Timer Loop =========
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
                    # 1) Daten & Indikator-Meldung
                    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
                    if not df.empty:
                        fdf = build_features(df, CONFIG)
                        snap = format_indicator_snapshot(fdf, CONFIG)
                        note_msg = _friendly_data_note(note)
                        msg = f"📡 {note_msg}\n{snap}" if note_msg else snap
                        await send(CHAT_ID, msg)

                        # 2) Strategie-Tick + evtl. Order
                        await run_once_and_report(CHAT_ID, place_order=CONFIG.trade_enabled)
                        TIMER_STATE["last_reason"] = "tick_ok"
                    else:
                        await send(CHAT_ID, f"❌ Keine Daten für {CONFIG.symbol} ({CONFIG.interval}).")
                        TIMER_STATE["last_reason"] = "no_data"
                except Exception as e:
                    traceback.print_exc()
                    TIMER_STATE["last_reason"] = f"tick_error: {e}"
            else:
                TIMER_STATE["last_reason"] = "no_chat_id"

            TIMER_STATE["last_run"] = now.isoformat()
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
        "/status – Status & Quelle\n"
        "/set key=value … – z.B. /set rsiLow=0 rsiHigh=68 slPerc=1 tpPerc=400\n"
        "/run – sofortiger Strategie-Tick\n"
        "/live on|off – Live-Handel schalten (Orders via Alpaca)\n"
        "/cfg – aktuelle Konfiguration\n"
        "/bt 180 – Backtest (Tage)\n"
        "/ind – Indikator-Snapshot (RSI & EFI); /sig Alias\n"
        "/plot 200 – Chart Close/RSI/EFI\n"
        "/timerstart | /timerstop | /timerstatus – interner Timer\n"
        "/dump 200 – CSV der letzten N Bars mit RSI/EFI\n"
        "/envcheck – ENV Überblick\n"
        "/tgstatus – Bot Status\n"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbol: {CONFIG.symbol}  TF: {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} (alle {CONFIG.poll_minutes}m)\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Letzte Aktion: {STATE.last_status}\n"
        f"Datenquelle: {CONFIG.data_provider}\n"
        f"MarketHoursOnly: {CONFIG.market_hours_only} | US Holidays: {CONFIG.use_us_holidays}\n"
        f"TradeEnabled: {CONFIG.trade_enabled} | TIF: {CONFIG.tif} | Sizing: {CONFIG.sizing_mode}"
    )

def set_from_kv(k: str, v: str) -> str:
    mapping = {"sl":"slPerc","tp":"tpPerc","samebar":"allowSameBarExit","cooldown":"minBarsInTrade"}
    k = mapping.get(k.strip(), k.strip())
    v = v.strip()
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

async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 rsi_source=close\n"
            "/set slPerc=1 tpPerc=400\n"
            "/set interval=1h lookback_days=365\n"
            "/set data_provider=alpaca | yahoo | stooq_eod\n"
            "/set poll_minutes=10 market_hours_only=true use_us_holidays=true\n"
            "/set trade_enabled=false tif=day sizing_mode=shares fixed_shares=1\n"
            "/set sizing_mode=notional notional_usd=100 max_pos=1"
        )
        return
    msgs, errs = [], []
    for a in context.args:
        if "=" not in a:
            errs.append(f"❌ Ungültig: {a}")
            continue
        k, v = a.split("=", 1)
        msgs.append(set_from_kv(k, v))
    out = "\n".join(msgs + (["\n⚠️ Probleme:"] + errs if errs else []))
    if not out.strip():
        out = "❌ Keine gültigen Paare."
    await update.message.reply_text(out)

async def cmd_cfg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Nutze: /live on oder /live off")
        return
    on = context.args[0].lower() in ("on","1","true","start","yes")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

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
    fdf = build_features(df, CONFIG)

    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = float(row["rsi"]); rsi_prev = float(prev["rsi"])
        efi_val = float(row["efi"]); efi_prev = float(prev["efi"])
        rsi_rising = rsi_val > rsi_prev
        rsi_falling= rsi_val < rsi_prev
        efi_rising = efi_val > efi_prev
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
        base = f"📈 Backtest {days}d  Trades: {entries}/{exits}\nWinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%"
        if note_msg: base = f"{note_msg}\n{base}"
        await update.message.reply_text(base)
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_ind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG)
    text = format_indicator_snapshot(fdf, CONFIG)
    note_msg = _friendly_data_note(note)
    if note_msg:
        text = f"{note_msg}\n{text}"
    await update.message.reply_text(text)

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

    ax2.plot(f.index, f["efi"])
    ax2.set_title(f"EFI({CONFIG.efiLen})")
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
    global _timer_task
    if TIMER_STATE["enabled"]:
        await update.message.reply_text("⏱️ Timer ist bereits aktiv.")
        return
    TIMER_STATE["enabled"] = True
    if _timer_task is None or _timer_task.done():
        _timer_task = asyncio.create_task(_timer_loop())
    await update.message.reply_text(f"⏱️ Timer gestartet (alle {CONFIG.poll_minutes}m).")

async def cmd_timerstop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global _timer_task
    TIMER_STATE["enabled"] = False
    await update.message.reply_text("⏹️ Timer gestoppt.")

async def cmd_timerstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        json.dumps({
            "enabled": TIMER_STATE["enabled"],
            "running": TIMER_STATE["running"],
            "poll_minutes": TIMER_STATE["poll_minutes"],
            "last_run": TIMER_STATE["last_run"],
            "next_due": TIMER_STATE["next_due"],
            "market_hours_only": TIMER_STATE["market_hours_only"],
            "last_reason": TIMER_STATE.get("last_reason","")
        }, indent=2)
    )

async def cmd_dump(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = 200
    if context.args:
        try: n = int(context.args[0])
        except: pass
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten.")
        return
    fdf = build_features(df, CONFIG).tail(max(10, n))
    out = fdf[["time","open","high","low","close","volume","rsi","efi"]].copy()
    out["time"] = pd.to_datetime(out["time"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    bio = io.BytesIO(csv_bytes); bio.name = f"{CONFIG.symbol}_{CONFIG.interval}_dump.csv"
    await tg_app.bot.send_document(chat_id=update.effective_chat.id, document=bio,
                                   caption=_friendly_data_note(note) or "Dump")

async def cmd_envcheck(update: Update, context: ContextTypes.DEFAULT_TYPE):
    def chk(k):
        v = os.getenv(k); return {"present": bool(v), "len": len(v) if v else 0}
    await update.message.reply_text(json.dumps({
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "BASE_URL": chk("BASE_URL"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_API_BASE_URL": chk("APCA_API_BASE_URL"),
    }, indent=2))

async def cmd_tgstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(json.dumps({
        "tg_running": tg_running,
        "polling_started": POLLING_STARTED
    }, indent=2))

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Lifespan (Polling) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
        tg_app.add_handler(CommandHandler("timerstart", cmd_timerstart))
        tg_app.add_handler(CommandHandler("timerstop",  cmd_timerstop))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("envcheck",cmd_envcheck))
        tg_app.add_handler(CommandHandler("tgstatus",cmd_tgstatus))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()

        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht (drop_pending_updates=True)")
        except Exception as e:
            print("delete_webhook warn:", e)

        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    print("▶️ starte Polling…")
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
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
    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        # stop timer
        TIMER_STATE["enabled"] = False
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
        POLLING_STARTED = False
        print("🛑 Telegram POLLING gestoppt")

# ========= FastAPI App =========
app = FastAPI(title="TQQQ Strategy + Telegram (RSI/EFI, no MACD)", lifespan=lifespan)

# ========= HEALTH & DIAG =========
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
    if CONFIG.market_hours_only and not is_rth(datetime.now(timezone.utc)):
        return {"ran": False, "reason": "outside_rth_or_holiday"}
    if CHAT_ID is None:
        return {"ran": False, "reason": "no_chat_id (use /start in Telegram)"}
    # Indikatoren sofort vorher senden
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if not df.empty:
        fdf = build_features(df, CONFIG)
        snap = format_indicator_snapshot(fdf, CONFIG)
        note_msg = _friendly_data_note(note)
        msg = f"📡 {note_msg}\n{snap}" if note_msg else snap
        await send(CHAT_ID, msg)
    await run_once_and_report(CHAT_ID, place_order=CONFIG.trade_enabled)
    return {"ran": True}

@app.get("/timerstatus")
async def timerstatus():
    return {
        "enabled": TIMER_STATE["enabled"],
        "running": TIMER_STATE["running"],
        "poll_minutes": TIMER_STATE["poll_minutes"],
        "last_run": TIMER_STATE["last_run"],
        "next_due": TIMER_STATE["next_due"],
        "market_hours_only": TIMER_STATE["market_hours_only"],
        "last_reason": TIMER_STATE.get("last_reason","")
    }

@app.get("/envcheck")
def envcheck_http():
    def chk(k):
        v = os.getenv(k)
        return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "BASE_URL": chk("BASE_URL"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_API_BASE_URL": chk("APCA_API_BASE_URL"),
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
