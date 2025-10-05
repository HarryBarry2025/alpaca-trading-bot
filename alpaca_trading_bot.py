# alpaca_trading_bot.py
import os, json, time, asyncio, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.7)
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional, für Market Open/Equity/Orders)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # optional – trading client setzt default auf paper/live passend zum key

# ========= (Lazy) Imports Alpaca Trading/MarketData =========
alpaca_ok = True
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import Clock
    from alpaca.trading.requests import (
        MarketOrderRequest, GetCalendarRequest,
        TakeProfitRequest, StopLossRequest
    )
    from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception as _e:
    alpaca_ok = False
    print("[alpaca] optional libs not available:", _e)

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"           # '1d' oder '1h'
    lookback_days: int = 365

    # Pine-like Inputs
    rsiLen: int = 12
    rsiLow: float = 0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    macdFast: int = 8
    macdSlow: int = 21
    macdSig: int = 11
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 40.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine
    poll_minutes: int = 10
    live_enabled: bool = False

    # Data Engine
    data_provider: str = "yahoo"      # "yahoo" | "stooq_eod" | "alpaca"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Trading / Paper
    paper_trading: bool = False       # nur Messages = False; echte Paper-Orders = True
    submit_tp_sl: bool = False        # Bracket mitschicken (TP/SL)

    # ---- Position Sizing ----
    # 'fixed_qty' | 'notional' | 'pct_equity' | 'risk_per_trade'
    sizing_mode: str = "fixed_qty"

    # fixed_qty
    order_qty: float = 1.0

    # notional
    notional_usd: float = 1000.0

    # pct_equity
    pct_equity: float = 10.0          # % des Account-Equity

    # risk_per_trade
    risk_pct_equity: float = 1.0      # % Equity-Risiko pro Trade

    # Runden/Minima
    min_qty: float = 1.0
    round_lots: bool = False

    # ---- Time-in-Force ----
    tif: str = "DAY"                  # "DAY" | "GTC" | "IOC" | "FOK"

class StratState(BaseModel):
    position_size: float = 0.0
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
    d = s.diff()
    up = pd.Series(np.where(d > 0, d, 0.0), index=s.index)
    dn = pd.Series(np.where(d < 0, -d, 0.0), index=s.index)
    ru = up.ewm(span=n, adjust=False).mean()
    rd = dn.ewm(span=n, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100/(1+rs))

def macd(s: pd.Series, fast: int, slow: int, sig: int):
    f = ema(s, fast); sl = ema(s, slow)
    line = f - sl
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def efi_func(close: pd.Series, vol: pd.Series, efi_len: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, efi_len)

# ========= Alpaca helpers =========
def get_trading_client() -> Optional["TradingClient"]:
    if not alpaca_ok: return None
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        return None
    try:
        # paper=True/False wird intern anhand Key erkannt – BASE_URL optional
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    except Exception as e:
        print("[alpaca] TradingClient failed:", e)
        return None

def get_data_client() -> Optional["StockHistoricalDataClient"]:
    if not alpaca_ok: return None
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        return None
    try:
        return StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    except Exception as e:
        print("[alpaca] DataClient failed:", e)
        return None

# ========= Market Open (Clock + Calendar, Fallback RTH) =========
def _ny_now():
    return datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))

def is_rth_now_basic() -> bool:
    now = _ny_now()
    if now.weekday() > 4:  # Sa/So
        return False
    hhmm = now.hour*100 + now.minute
    return 930 <= hhmm < 1600

def alpaca_clock_is_open() -> Optional[bool]:
    tc = get_trading_client()
    if tc is None: return None
    try:
        clock: Clock = tc.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        print("[alpaca] get_clock failed:", e)
        return None

def is_us_holiday_today() -> Optional[bool]:
    tc = get_trading_client()
    if tc is None: return None
    try:
        today = datetime.now(timezone.utc).date()
        req = GetCalendarRequest(start=today, end=today)
        cals = tc.get_calendar(req)
        # Kein Kalender-Eintrag => kein Handel
        return False if cals else True
    except Exception as e:
        print("[alpaca] get_calendar failed:", e)
        return None

def is_market_open_now() -> bool:
    clk = alpaca_clock_is_open()
    if clk is True:
        hol = is_us_holiday_today()
        if hol is True: return False
        return True
    if clk is False:
        return False
    return is_rth_now_basic()

def get_account_equity() -> Optional[float]:
    tc = get_trading_client()
    if tc is None: return None
    try:
        acct = tc.get_account()
        return float(acct.equity)
    except Exception as e:
        print("[alpaca] get_account failed:", e)
        return None

def tif_enum_from_str(name: str) -> "TimeInForce":
    if not alpaca_ok: return None
    return {
        "DAY": TimeInForce.DAY,
        "GTC": TimeInForce.GTC,
        "IOC": TimeInForce.IOC,
        "FOK": TimeInForce.FOK,
    }.get(name.upper(), TimeInForce.DAY)

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
        if lookback_days > 0: df = df.iloc[-lookback_days:]
        df["time"] = df.index.tz_localize("UTC")
        return df
    except Exception as e:
        print("[stooq] fetch failed:", e)
        return pd.DataFrame()

def fetch_alpaca_ohlcv(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    dc = get_data_client()
    if dc is None: return pd.DataFrame()
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
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 30))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start, end=end,
        adjustment=None, feed="iex", limit=10000
    )
    try:
        bars = dc.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty data")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.sort_index()
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int):
    note = {"provider":"","detail":""}
    intraday = {"1m","2m","5m","15m","30m","60m","90m","1h"}

    # alpaca first if chosen
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=interval)
            return df, note
        note.update(provider="Alpaca→Yahoo", detail="no data; try Yahoo")

    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        note.update(provider="Stooq→Yahoo", detail="no data; try Yahoo")

    # Yahoo
    is_intraday = interval in intraday
    period = f"{min(lookback_days,730)}d" if is_intraday else f"{lookback_days}d"

    def _normalize(tmp):
        if tmp is None or tmp.empty: return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except: pass
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            try: df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            except: pass
        df = df.sort_index()
        df["time"] = df.index
        return df

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True, threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    last_err=None
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

    # Yahoo 1d fallback
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} failed")
            return df, note
    except Exception as e:
        last_err = e

    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days,120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] no data for {symbol} ({interval}). last_err={last_err}")
    return pd.DataFrame(), {"provider":"(empty)", "detail":"no data"}

# ========= Features & Logic =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsiLen)
    macd_line, macd_sig, _ = macd(out["close"], cfg.macdFast, cfg.macdSlow, cfg.macdSig)
    out["macd_line"], out["macd_sig"] = macd_line, macd_sig
    out["efi"] = efi_func(out["close"], out["volume"], cfg.efiLen)
    return out

def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.macdSlow, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi_val = last["rsi"]
    macd_above = last["macd_line"] > last["macd_sig"]
    rsi_rising = last["rsi"] > prev["rsi"]
    rsi_falling= last["rsi"] < prev["rsi"]
    efi_rising = last["efi"] > prev["efi"]

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price = float(last["close"]); o = float(last["open"]); ts = last["time"]
    sl = lambda px: px * (1 - cfg.slPerc/100.0)
    tp = lambda px: px * (1 + cfg.tpPerc/100.0)

    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if st.position_size <= 0:
        if entry_cond:
            return {"action":"buy", "qty":1, "price":o, "time":str(ts), "reason":"rule_entry",
                    "sl":sl(o), "tp":tp(o)}
        return {"action":"none","reason":"flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade>0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok
        cur_sl = sl(st.avg_price); cur_tp = tp(st.avg_price)
        if rsi_exit_ok:
            return {"action":"sell","qty":st.position_size,"price":o,"time":str(ts),"reason":"rsi_exit"}
        if price <= cur_sl:
            return {"action":"sell","qty":st.position_size,"price":cur_sl,"time":str(ts),"reason":"stop_loss"}
        if price >= cur_tp:
            return {"action":"sell","qty":st.position_size,"price":cur_tp,"time":str(ts),"reason":"take_profit"}
        return {"action":"none","reason":"hold"}

# ========= Sizing & Order helpers =========
def compute_order_size(entry_px: float, stop_px: Optional[float]) -> Tuple[Optional[float], Optional[float], str]:
    mode = CONFIG.sizing_mode.lower()
    q = n = None; note=""
    if mode == "fixed_qty":
        q = max(CONFIG.order_qty, CONFIG.min_qty)
        if CONFIG.round_lots: q = float(int(round(q)))
        note = f"fixed_qty={q:g}"
    elif mode == "notional":
        n = max(CONFIG.notional_usd, 1.0)
        note = f"notional=${n:,.0f}"
    elif mode == "pct_equity":
        eq = get_account_equity() or 10000.0
        n = max(eq * (CONFIG.pct_equity/100.0), 1.0)
        note = f"{CONFIG.pct_equity:.1f}% equity → notional ${n:,.0f}"
    elif mode == "risk_per_trade":
        eq = get_account_equity() or 10000.0
        risk_dollars = max(eq * (CONFIG.risk_pct_equity/100.0), 1.0)
        if stop_px is None or stop_px >= entry_px:
            n = max(CONFIG.notional_usd, risk_dollars)
            note = f"risk% fallback notional ${n:,.0f}"
        else:
            per_share_risk = entry_px - stop_px
            shares = risk_dollars / max(per_share_risk, 1e-6)
            q = max(shares, CONFIG.min_qty)
            if CONFIG.round_lots: q = float(int(round(q)))
            note = f"risk {CONFIG.risk_pct_equity:.1f}% → qty={q:g} (Δ={per_share_risk:.2f})"
    else:
        q = max(CONFIG.order_qty, CONFIG.min_qty); note = f"default fixed_qty={q:g}"
    return q, n, note

def place_market_order(symbol: str, side: str,
                       qty: float | None = None,
                       notional: float | None = None,
                       tp_px: float | None = None,
                       sl_px: float | None = None,
                       tif_str: str = None):
    tc = get_trading_client()
    if tc is None: raise RuntimeError("Alpaca TradingClient not available")
    if qty is None and notional is None:
        raise ValueError("qty or notional required")
    order_class = OrderClass.SIMPLE
    tp_req = sl_req = None
    if tp_px or sl_px:
        order_class = OrderClass.BRACKET
        tp_req = TakeProfitRequest(limit_price=float(tp_px)) if tp_px else None
        sl_req = StopLossRequest(stop_price=float(sl_px)) if sl_px else None
    tif = tif_enum_from_str(tif_str or CONFIG.tif)
    req = MarketOrderRequest(
        symbol=symbol,
        side=OrderSide.BUY if side.lower()=="buy" else OrderSide.SELL,
        time_in_force=tif,
        qty=qty if qty is not None else None,
        notional=notional if notional is not None else None,
        order_class=order_class,
        take_profit=tp_req,
        stop_loss=sl_req
    )
    return tc.submit_order(order_data=req)

def get_open_position_qty(symbol: str) -> Optional[float]:
    tc = get_trading_client()
    if tc is None: return None
    try:
        pos = tc.get_open_position(symbol)
        return float(pos.qty) if pos else 0.0
    except Exception:
        return None

# ========= Telegram =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text if text.strip() else "ℹ️ (leer)")
    except Exception as e:
        print("send error:", e)

async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "/status – Status\n"
        "/set key=value … (z.B. /set tif=GTC sizing_mode=notional notional_usd=2500)\n"
        "/run – einmal prüfen & handeln (nur RTH/Market open)\n"
        "/live on|off – Live-Loop\n"
        "/cfg – aktuelle Konfiguration\n"
        "/bt 90 – Backtest über 90 Tage"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbol: {CONFIG.symbol} {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} (alle {CONFIG.poll_minutes}m)\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Last: {STATE.last_status}\n"
        f"Datenquelle: {CONFIG.data_provider}\n"
        f"Paper: {'ON' if CONFIG.paper_trading else 'OFF'}\n"
        f"MarketOpen: {'YES' if is_market_open_now() else 'NO'}\n"
        f"TIF: {CONFIG.tif} | Sizing: {CONFIG.sizing_mode}"
    )

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1)
    k = k.strip(); v = v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k, k)
    if not hasattr(CONFIG, k): return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):   setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):  setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):setattr(CONFIG, k, float(v))
    else:                       setattr(CONFIG, k, v)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value …\n"
            "Bsp: /set tif=GTC sizing_mode=risk_per_trade risk_pct_equity=1.0 sl=2 tp=4\n"
            "Bsp: /set data_provider=alpaca paper_trading=true submit_tp_sl=true"
        ); return
    msgs, errors = [], []
    for a in context.args:
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errors.append(f"❌ Ungültig: {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errors.append(f"❌ {a}: {e}")
    out = []
    if msgs: out += ["✅ Übernommen:"] + msgs
    if errors: out += ["\n⚠️ Probleme:"] + errors
    await update.message.reply_text("\n".join(out) if out else "❌ Keine gültigen key=value-Paare.")

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if prov == "Yahoo": return f"📡 Daten: Yahoo ({det})"
    if prov.startswith("Yahoo (Fallback"): return f"📡 Daten: {prov} – {det}"
    if prov.startswith("Stooq"): return f"📡 Daten: {prov} – {det} (EOD)"
    return f"📡 Daten: {prov} – {det}"

# ===== main run/report =====
async def run_once_and_report(chat_id: str):
    if not is_market_open_now():
        await send(chat_id, "⏸️ US-Börse geschlossen (Feiertag/RTH außerhalb).")
        return

    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten: {note.get('provider','?')} – {note.get('detail','')}")
        return

    note_msg = _friendly_data_note(note)
    if note_msg and ("Fallback" in note_msg or "Stooq" in note_msg or CONFIG.data_provider!="yahoo"):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    if act["action"] == "buy":
        entry_px = float(act["price"])
        tp_px = float(act.get("tp")) if act.get("tp") else None
        sl_px = float(act.get("sl")) if act.get("sl") else None

        qty, notional, note_size = compute_order_size(entry_px, sl_px)

        if CONFIG.paper_trading and alpaca_ok and get_trading_client() is not None:
            try:
                order = place_market_order(
                    symbol=CONFIG.symbol, side="buy",
                    qty=qty, notional=notional,
                    tp_px=(tp_px if CONFIG.submit_tp_sl else None),
                    sl_px=(sl_px if CONFIG.submit_tp_sl else None),
                    tif_str=CONFIG.tif
                )
                human = f"qty {qty:g}" if qty is not None else f"notional ${notional:,.0f}"
                await send(chat_id, f"🟢 PAPER BUY {CONFIG.symbol} ({human}, {note_size}, TIF={CONFIG.tif})")
            except Exception as e:
                await send(chat_id, f"❌ Paper-Buy fehlgeschlagen: {e}")

        STATE.position_size = float(qty if qty is not None else (notional/entry_px if notional else 1))
        STATE.avg_price = entry_px
        STATE.entry_time = act["time"]
        line2 = (f"\nSL={sl_px:.4f}  TP={tp_px:.4f}" if sl_px and tp_px else "")
        await send(chat_id, f"Signal: LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol}){line2}")

    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\nPnL={pnl*100:.2f}%")
        STATE.position_size = 0.0
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
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für Backtest. Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return
    note_msg = _friendly_data_note(note)
    if note_msg: await update.message.reply_text(note_msg)

    fdf = build_features(df, CONFIG)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = row["rsi"]; macd_above = row["macd_line"] > row["macd_sig"]
        rsi_rising = row["rsi"] > prev["rsi"]; rsi_falling = row["rsi"] < prev["rsi"]
        efi_rising  = row["efi"] > prev["efi"]
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
                r = (px-avg)/avg; eq*= (1+r); R.append(r); exits+=1
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

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Lifespan (Polling ohne Loop-Konflikt) =========
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
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize(); await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht (drop_pending_updates=True)")
        except Exception as e:
            print("delete_webhook warn:", e)

        if not POLLING_STARTED:
            delay=5
            while True:
                try:
                    print("▶️ starte Polling…")
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED=True
                    print("✅ Polling läuft"); break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. Retry in {delay}s…")
                    await asyncio.sleep(delay); delay=min(delay*2,60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        tg_running = True
        print("🚀 Telegram POLLING aktiv")
    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e); traceback.print_exc()
    try:
        yield
    finally:
        tg_running=False
        try: await tg_app.updater.stop()
        except: pass
        try: await tg_app.stop()
        except: pass
        try: await tg_app.shutdown()
        except: pass
        POLLING_STARTED=False
        print("🛑 Telegram POLLING gestoppt")

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
        "market_open": is_market_open_now()
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
    if not is_market_open_now():
        return {"ran": False, "reason": "market_closed"}
    await run_once_and_report(CHAT_ID)
    return {"ran": True}

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v = os.getenv(k); return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_API_BASE_URL": chk("APCA_API_BASE_URL"),
    }

@app.get("/tgstatus")
def tgstatus():
    return {"tg_running": tg_running, "polling_started": POLLING_STARTED}

@app.post(f"/telegram/{WEBHOOK_SECRET}")
async def telegram_webhook(req: Request):
    if tg_app is None:
        raise HTTPException(503, "Bot not ready")
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
