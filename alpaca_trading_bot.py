# alpaca_trading_bot.py
import os, io, json, time, asyncio, traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (python-telegram-bot v20.x)
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler, CallbackContext, filters
)
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional; für Handel & Daten)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # z.B. https://paper-api.alpaca.markets

# ========= Konfig/State =========
class StratConfig(BaseModel):
    # Multi-Asset Support
    symbols: List[str] = ["TQQQ"]          # zuerst genutztes Symbol ist "Hauptsymbol"
    interval: str = "1h"                   # "1h" (intraday) oder "1d"
    lookback_days: int = 365

    # TV-kompatible Inputs
    rsiLen: int = 12
    rsiLow: float = 0.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine / Timer
    poll_minutes: int = 10
    live_enabled: bool = False
    market_hours_only: bool = True

    # Daten
    data_provider: str = "alpaca"          # "alpaca", "yahoo", "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Trading (Alpaca)
    trading_enabled: bool = False          # True = sende Orders zu Alpaca
    sizing_mode: str = "fraction"          # "fraction" | "notional" | "shares"
    position_fraction: float = 0.25        # 25% des Cash
    notional_usd: float = 1000.0           # für sizing_mode="notional"
    shares: int = 1                        # für sizing_mode="shares"
    time_in_force: str = "day"             # "day" oder "gtc"

class StratState(BaseModel):
    # pro Symbol separater Positions- und Timer-Status
    positions: Dict[str, Dict[str, Any]] = {}
    last_status: str = "idle"

class TimerState(BaseModel):
    enabled: bool = True
    running: bool = False
    poll_minutes: int = 10
    last_run: Optional[str] = None
    next_due: Optional[str] = None
    market_hours_only: bool = True

CONFIG = StratConfig()
STATE  = StratState()
TIMER  = TimerState(poll_minutes=CONFIG.poll_minutes, market_hours_only=CONFIG.market_hours_only)
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= US-Markzeiten & Feiertage =========
import pytz
NY = pytz.timezone("America/New_York")

# Minimal-Feiertage (kann erweitert werden)
def _nyse_holidays(year: int) -> set:
    # Sehr einfache Auswahl; für produktiv: 'pandas_market_calendars' verwenden
    # New Year's Day, MLK, Presidents, Good Friday (vereinfacht ignoriert), Memorial, Juneteenth, Independence, Labor, Thanksgiving, Christmas
    fixed = {
        (1,1), (6,19), (7,4), (12,25),
    }
    # bewegliche grob (kein perfekter Kalender)
    # Wir markieren NICHT alle korrekt – genügt als "nur-handelszeit grob" Filter.
    return set((year, m, d) for (m,d) in [(1,1),(6,19),(7,4),(12,25)])

def is_market_open_now(dt_utc: Optional[datetime]=None) -> bool:
    dt_utc = dt_utc or datetime.now(timezone.utc)
    now_ny = dt_utc.astimezone(NY)
    if now_ny.weekday() >= 5:
        return False
    y, m, d = now_ny.year, now_ny.month, now_ny.day
    if (y,m,d) in _nyse_holidays(y):
        return False
    # Handelszeit 09:30–16:00 ET
    open_t  = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_ny <= close_t

# ========= Indikatoren (TV-kompatibel) =========
def rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def rsi_tv(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = rma(up, length)
    avg_loss = rma(down, length)
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def efi_tv(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    raw = volume * (close - close.shift(1))
    return raw.ewm(span=length, adjust=False).mean()

# ========= Datenquellen =========
def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    from urllib.parse import quote
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
    elif interval in {"15m"}:
        tf = TimeFrame(15, TimeFrameUnit.Minute)
    elif interval in {"5m"}:
        tf = TimeFrame(5, TimeFrameUnit.Minute)
    else:
        tf = TimeFrame(1, TimeFrameUnit.Day)

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
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]].sort_index()
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_yahoo(symbol: str, interval: str, lookback_days: int, retries=3, backoff=2.0) -> pd.DataFrame:
    intraday = interval.lower() in {"1m","2m","5m","15m","30m","60m","90m","1h"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except: pass
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            except: pass
        df["time"] = df.index
        return df.sort_index()

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=True,  threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    last_err = None
    for method, kwargs in tries:
        for attempt in range(1, retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = normalize(tmp)
                if not df.empty:
                    return df
            except Exception as e:
                last_err = e
            time.sleep(backoff * (2**(attempt-1)))
    # Fallback 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = normalize(tmp)
        if not df.empty:
            return df
    except Exception as e:
        last_err = e
    print("[yahoo] failed:", last_err)
    return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}

    if CONFIG.data_provider.lower()=="alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=interval)
            return df, note
        # Fallbacks:
        df = fetch_yahoo(symbol, interval, lookback_days, CONFIG.yahoo_retries, CONFIG.yahoo_backoff_sec)
        if not df.empty:
            note.update(provider="Alpaca→Yahoo", detail=interval)
            return df, note
        if CONFIG.allow_stooq_fallback:
            df = fetch_stooq_daily(symbol, lookback_days)
            if not df.empty:
                note.update(provider="Alpaca→Stooq EOD (Fallback)", detail="1d")
                return df, note
        return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

    if CONFIG.data_provider.lower()=="yahoo":
        df = fetch_yahoo(symbol, interval, lookback_days, CONFIG.yahoo_retries, CONFIG.yahoo_backoff_sec)
        if not df.empty:
            note.update(provider="Yahoo", detail=interval)
            return df, note
        if CONFIG.allow_stooq_fallback:
            df = fetch_stooq_daily(symbol, lookback_days)
            if not df.empty:
                note.update(provider="Yahoo→Stooq EOD (Fallback)", detail="1d")
                return df, note
        return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

    if CONFIG.data_provider.lower()=="stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        # Fallback Yahoo:
        df = fetch_yahoo(symbol, interval, lookback_days, CONFIG.yahoo_retries, CONFIG.yahoo_backoff_sec)
        if not df.empty:
            note.update(provider="Stooq→Yahoo", detail=interval)
            return df, note
        return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

    return pd.DataFrame(), {"provider":"(leer)","detail":"unbekannter provider"}

def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider","")
    det  = note.get("detail","")
    if not prov: return ""
    if "Fallback" in prov or "Stooq" in prov or "Alpaca→" in prov:
        return f"📡 Daten: {prov} – {det}"
    return f"📡 Daten: {prov} ({det})"

# ========= Feature-Builder =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

def build_feats_for(symbol: str, interval: str, lookback_days: int):
    df, note = fetch_ohlcv_with_note(symbol, interval, lookback_days)
    if df is None or df.empty:
        return pd.DataFrame(), note, None
    fdf = build_features(df, CONFIG)
    return fdf, note, df

# ========= Strategie-Logik =========
def bar_logic(fdf: pd.DataFrame, cfg: StratConfig, sym: str) -> Dict[str, Any]:
    if fdf.empty or len(fdf) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data","symbol":sym}

    last = fdf.iloc[-1]
    prev = fdf.iloc[-2]

    rsi_val = float(last["rsi"])
    rsi_rising  = last["rsi"] > prev["rsi"]
    rsi_falling = last["rsi"] < prev["rsi"]
    efi_rising  = last["efi"] > prev["efi"]

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = str(last["time"])

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]; entry_time = pos["entry_time"]

    # Bars in Trade (näherungsweise)
    bars_in_trade = 0
    if entry_time is not None:
        since_entry = fdf[fdf["time"] >= pd.to_datetime(entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if size == 0:
        if entry_cond:
            return {"action":"buy","symbol":sym,"qty":1,"price":price_open,"time":ts,"reason":"rule_entry",
                    "sl":sl(price_open),"tp":tp(price_open),
                    "rsi":rsi_val,"efi":float(last['efi'])}
        return {"action":"none","reason":"flat_no_entry","symbol":sym,
                "rsi":rsi_val,"efi":float(last['efi'])}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(avg)
        cur_tp = tp(avg)
        hit_sl = price_close <= cur_sl
        hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","symbol":sym,"qty":size,"price":price_open,"time":ts,"reason":"rsi_exit",
                    "rsi":rsi_val,"efi":float(last['efi'])}
        if hit_sl:
            return {"action":"sell","symbol":sym,"qty":size,"price":cur_sl,"time":ts,"reason":"stop_loss",
                    "rsi":rsi_val,"efi":float(last['efi'])}
        if hit_tp:
            return {"action":"sell","symbol":sym,"qty":size,"price":cur_tp,"time":ts,"reason":"take_profit",
                    "rsi":rsi_val,"efi":float(last['efi'])}
        return {"action":"none","reason":"hold","symbol":sym,
                "rsi":rsi_val,"efi":float(last['efi'])}

# ========= Alpaca Trading (optional) =========
def alpaca_client_or_none():
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        return None, "missing_keys"
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import TimeInForce
        client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=("paper" in (APCA_API_BASE_URL or "").lower()))
        return client, None
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None, str(e)

def alpaca_positions_text() -> str:
    client, err = alpaca_client_or_none()
    if client is None:
        return f"❌ Alpaca nicht konfiguriert ({err})."
    try:
        pos = client.get_all_positions()
        if not pos:
            return "📦 Keine offenen Alpaca-Positionen."
        lines = ["📦 Alpaca Positionen:"]
        for p in pos:
            lines.append(f"- {p.symbol}: {p.qty} @ {p.avg_entry_price}  (unrealized PnL: {p.unrealized_pl})")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Fehler beim Abrufen der Positionen: {e}"

def place_order_alpaca(symbol: str, side: str, price: float) -> str:
    client, err = alpaca_client_or_none()
    if client is None:
        return f"❌ Alpaca nicht konfiguriert ({err}). Order simuliert."

    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        tif = TimeInForce.DAY if CONFIG.time_in_force.lower()=="day" else TimeInForce.GTC

        # Sizing
        qty=None; notional=None
        if CONFIG.sizing_mode=="fraction":
            account = client.get_account()
            cash = float(account.cash)
            notional = max(1.0, cash * float(CONFIG.position_fraction))
        elif CONFIG.sizing_mode=="notional":
            notional = float(CONFIG.notional_usd)
        else:
            qty = max(1, int(CONFIG.shares))

        req = MarketOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif,
            qty=qty,
            notional=notional
        )
        order = client.submit_order(req)
        return f"🪙 Alpaca Order {order.id} {side.upper()} {symbol} qty={qty} notional={notional}"
    except Exception as e:
        return f"❌ Alpaca Order-Fehler: {e}"

# ========= Telegram Setup =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text or " ")
    except Exception as e:
        print("send error:", e)

# ========= Telegram Commands =========
async def cmd_start(update: Update, context: CallbackContext):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status  – Zustand & Datenquelle\n"
        "/cfg     – Konfiguration anzeigen\n"
        "/set key=value … (z.B. /set interval=1h poll_minutes=10 data_provider=alpaca)\n"
        "/live on|off – Hintergrundtimer schalten\n"
        "/run     – sofortiger Check (alle Symbole)\n"
        "/bt [Tage] [SYM] – Backtest EoB\n"
        "/ind [SYM] – letzte Indikatorwerte\n"
        "/sig [SYM] – Entry/Exit-Begründung\n"
        "/plot [SYM] – PNG mit Kurs/RSI/EFI\n"
        "/dump [SYM] [rows] – CSV-Export\n"
        "/positions – Alpaca Positionen\n"
        "/timerstatus – Status des Timer-Tasks"
    )

async def cmd_status(update: Update, context: CallbackContext):
    p = STATE.positions.get(CONFIG.symbols[0], {"size":0,"avg":0.0,"entry_time":None})
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbole: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}  Timer: {'RUNNING' if TIMER.running else 'STOP'}\n"
        f"Pos {CONFIG.symbols[0]}: {p['size']} @ {p['avg']}\n"
        f"Timer last={TIMER.last_run} next={TIMER.next_due}\n"
        f"Datenquelle: {CONFIG.data_provider}"
    )

async def cmd_cfg(update: Update, context: CallbackContext):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

def _apply_set_key(k: str, v: str) -> str:
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k, k)
    if not hasattr(CONFIG, k):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):
        val = v.lower() in ("1","true","on","yes")
    elif isinstance(cur, int):
        val = int(float(v))
    elif isinstance(cur, float):
        val = float(v)
    elif isinstance(cur, list):
        val = [s.strip().upper() for s in v.split(",") if s.strip()]
    else:
        val = v
    setattr(CONFIG, k, val)
    if k=="poll_minutes": TIMER.poll_minutes = int(float(val))
    if k=="market_hours_only": TIMER.market_hours_only = bool(val)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value …\nBeispiele:\n"
            "/set interval=1h lookback_days=365\n"
            "/set symbols=TQQQ,QQQ data_provider=alpaca\n"
            "/set trading_enabled=true sizing_mode=fraction position_fraction=0.25 time_in_force=day\n"
            "/set slPerc=1 tpPerc=4 rsiLow=0 rsiHigh=68 rsiExit=48"
        ); return
    msgs, errs = [], []
    for a in context.args:
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errs.append(f"❌ Ungültig: {a}"); continue
        k,v = a.split("=",1)
        try:
            msgs.append(_apply_set_key(k.strip(), v.strip()))
        except Exception as e:
            errs.append(f"❌ {k}: {e}")
    out = []
    if msgs: out+=["✅ Übernommen:"]+msgs
    if errs: out+=["\n⚠️ Probleme:"]+errs
    await update.message.reply_text("\n".join(out).strip())

async def cmd_live(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def run_once_for_symbol(sym: str) -> str:
    fdf, note, _ = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        return f"❌ {sym}: keine Daten. {friendly_note(note)}"

    act = bar_logic(fdf, CONFIG, sym)
    STATE.last_status = f"{act['action']} ({act['reason']})"
    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})

    msg_note = friendly_note(note)
    if act["action"]=="buy" and pos["size"]==0:
        # Setze interne Position
        STATE.positions[sym] = {"size":1, "avg":act["price"], "entry_time":act["time"]}
        txt = f"🟢 LONG {sym} @ {act['price']:.4f}\nSL={act['sl']:.4f} TP={act['tp']:.4f}\n{msg_note}"
        # Optional: echte Order
        if CONFIG.trading_enabled:
            txt += "\n" + place_order_alpaca(sym, "buy", act["price"])
        return txt
    elif act["action"]=="sell" and pos["size"]>0:
        exit_px = act["price"]
        pnl = (exit_px - pos["avg"]) / max(1e-9, pos["avg"])
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        txt = f"🔴 EXIT {sym} @ {exit_px:.4f} [{act['reason']}]\nPnL={pnl*100:.2f}%\n{msg_note}"
        if CONFIG.trading_enabled:
            txt += "\n" + place_order_alpaca(sym, "sell", exit_px)
        return txt
    else:
        return f"ℹ️ {sym}: {act['action']} ({act['reason']})  RSI={act.get('rsi'):.2f} EFI={act.get('efi'):.2f}\n{msg_note}"

async def cmd_run(update: Update, context: CallbackContext):
    lines=[]
    for sym in CONFIG.symbols:
        lines.append(await run_once_for_symbol(sym))
    await update.message.reply_text("\n\n".join(lines))

async def cmd_bt(update: Update, context: CallbackContext):
    days = 180; sym = CONFIG.symbols[0]
    if context.args:
        try:
            days = int(context.args[0])
            if len(context.args) > 1:
                sym = context.args[1].upper()
        except:
            sym = context.args[0].upper()
    fdf, note, _ = build_feats_for(sym, CONFIG.interval, days)
    if fdf.empty:
        await update.message.reply_text(f"❌ Backtest: keine Daten. {friendly_note(note)}"); return
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = row["rsi"]; rsi_ris = row["rsi"]>prev["rsi"]; rsi_fal = row["rsi"]<prev["rsi"]
        efi_ris = row["efi"]>prev["efi"]
        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and rsi_ris and efi_ris
        exitc = (rsi_val<CONFIG.rsiExit) and rsi_fal
        if pos==0 and entry:
            pos=1; avg=float(row["open"]); entries+=1
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            price = float(row["close"]); stop=price<=sl; take=price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                r = (px-avg)/avg
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean()
        pf=(a[a>0].sum())/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(
            f"📈 Backtest {sym} {days}d (TF={CONFIG.interval})\nTrades {entries}/{exits}\nWin {win*100:.1f}%  PF {pf:.2f}  CAGR~{cagr*100:.2f}%\n{friendly_note(note)}"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_ind(update: Update, context: CallbackContext):
    sym = CONFIG.symbols[0]
    if context.args: sym = context.args[0].upper()
    fdf, note, _ = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf.empty:
        await update.message.reply_text(f"❌ {sym}: keine Daten. {friendly_note(note)}"); return
    last=fdf.iloc[-1]; prev=fdf.iloc[-2]
    await update.message.reply_text(
        f"🔎 {sym} Indikatoren (letzter Bar)\n"
        f"Close={last['close']:.4f}\n"
        f"RSI={last['rsi']:.2f} (Δ {last['rsi']-prev['rsi']:+.2f})\n"
        f"EFI={last['efi']:.2f} (Δ {last['efi']-prev['efi']:+.2f})\n"
        + friendly_note(note)
    )

async def cmd_sig(update: Update, context: CallbackContext):
    sym = CONFIG.symbols[0]
    if context.args: sym = context.args[0].upper()
    fdf, note, _ = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf.empty:
        await update.message.reply_text(f"❌ {sym}: keine Daten. {friendly_note(note)}"); return
    act = bar_logic(fdf, CONFIG, sym)
    await update.message.reply_text(
        f"🧭 {sym} Signal: {act['action']} ({act['reason']})\n"
        f"RSI={act.get('rsi',np.nan):.2f}  EFI={act.get('efi',np.nan):.2f}\n" + friendly_note(note)
    )

def make_plot_png(sym: str, bars: int = 300) -> bytes:
    fdf, note, _ = build_feats_for(sym, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        return b""
    df = fdf.tail(bars).copy()
    t = pd.to_datetime(df["time"]).dt.tz_convert("UTC")
    fig, axes = plt.subplots(3, 1, figsize=(10,7), sharex=True)
    axes[0].plot(t, df["close"]); axes[0].set_title(f"{sym} Close")
    axes[1].plot(t, df["rsi"]); axes[1].axhline(CONFIG.rsiLow, ls="--"); axes[1].axhline(CONFIG.rsiHigh, ls="--"); axes[1].set_title("RSI (TV/RMA)")
    axes[2].plot(t, df["efi"]); axes[2].axhline(0.0, ls="--"); axes[2].set_title("EFI (EMA)")
    plt.tight_layout()
    bio = io.BytesIO(); plt.savefig(bio, format="png", dpi=140); plt.close(fig)
    bio.seek(0); return bio.read()

async def cmd_plot(update: Update, context: CallbackContext):
    sym = CONFIG.symbols[0]
    if context.args: sym = context.args[0].upper()
    png = make_plot_png(sym)
    if not png:
        await update.message.reply_text(f"❌ {sym}: keine Daten für Plot."); return
    await tg_app.bot.send_photo(chat_id=update.effective_chat.id, photo=png, caption=f"{sym} – Kurs/RSI/EFI")

def make_dump_csv_bytes(symbol: str, rows: int = 500) -> Tuple[bytes, str]:
    fdf, note, _ = build_feats_for(symbol, CONFIG.interval, CONFIG.lookback_days)
    if fdf is None or fdf.empty:
        return b"", f"❌ {symbol}: keine Daten zum Export."
    cols = ["time","open","high","low","close","volume","rsi","efi"]
    for c in cols:
        if c not in fdf.columns:
            fdf[c] = np.nan
    out = fdf[cols].tail(max(1, rows)).copy()
    out["time"] = pd.to_datetime(out["time"]).dt.tz_convert("UTC")
    csv = out.to_csv(index=False).encode("utf-8")
    note_msg = friendly_note(note)
    return csv, f"📄 Dump {symbol} ({len(out)} Zeilen)\n{note_msg}"

async def cmd_dump(update: Update, context: CallbackContext):
    sym = CONFIG.symbols[0]; rows = 500
    if context.args:
        sym = context.args[0].upper()
        if len(context.args) > 1:
            try: rows = int(context.args[1])
            except: rows = 500
    csv, caption = make_dump_csv_bytes(sym, rows)
    if not csv:
        await update.message.reply_text(caption); return
    await tg_app.bot.send_document(
        chat_id=update.effective_chat.id,
        document=csv,
        filename=f"{sym}_{CONFIG.interval}_dump.csv",
        caption=caption
    )

async def cmd_positions(update: Update, context: CallbackContext):
    await update.message.reply_text(alpaca_positions_text())

async def cmd_timerstatus(update: Update, context: CallbackContext):
    await update.message.reply_text(json.dumps(TIMER.dict(), indent=2))

async def on_message(update: Update, context: CallbackContext):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Hintergrund-Timer =========
async def timer_loop():
    while True:
        try:
            TIMER.running = True
            TIMER.last_run = datetime.now(timezone.utc).isoformat()
            # next_due vorab setzen
            TIMER.next_due = (datetime.now(timezone.utc) + timedelta(minutes=CONFIG.poll_minutes)).isoformat()

            if CONFIG.live_enabled:
                allowed = (not CONFIG.market_hours_only) or is_market_open_now()
                if allowed and CHAT_ID:
                    # Alle Symbole prüfen & Indikatorwerte melden (alle 10m)
                    lines = []
                    for sym in CONFIG.symbols:
                        lines.append(await run_once_for_symbol(sym))
                    await send(CHAT_ID, "\n\n".join(lines))
            await asyncio.sleep(max(60, CONFIG.poll_minutes*60))
        except Exception as e:
            print("timer_loop error:", e)
            await asyncio.sleep(30)

# ========= Lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",        cmd_start))
        tg_app.add_handler(CommandHandler("status",       cmd_status))
        tg_app.add_handler(CommandHandler("cfg",          cmd_cfg))
        tg_app.add_handler(CommandHandler("set",          cmd_set))
        tg_app.add_handler(CommandHandler("live",         cmd_live))
        tg_app.add_handler(CommandHandler("run",          cmd_run))
        tg_app.add_handler(CommandHandler("bt",           cmd_bt))
        tg_app.add_handler(CommandHandler("ind",          cmd_ind))
        tg_app.add_handler(CommandHandler("sig",          cmd_sig))
        tg_app.add_handler(CommandHandler("plot",         cmd_plot))
        tg_app.add_handler(CommandHandler("dump",         cmd_dump))
        tg_app.add_handler(CommandHandler("positions",    cmd_positions))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            print("delete_webhook warn:", e)

        if not POLLING_STARTED:
            delay=5
            while True:
                try:
                    print("▶️ starte Polling…")
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED=True; print("✅ Polling läuft"); break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. Retry in {delay}s"); await asyncio.sleep(delay); delay=min(delay*2,60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        # Timer starten
        asyncio.create_task(timer_loop())
        tg_running = True
    except Exception as e:
        print("❌ Telegram Startup Fehler:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        tg_running = False
        try: await tg_app.updater.stop()
        except: pass
        try: await tg_app.stop()
        except: pass
        try: await tg_app.shutdown()
        except: pass
        print("🛑 Telegram gestoppt")

# ========= FastAPI =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbols": CONFIG.symbols,
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
    lines=[]
    for sym in CONFIG.symbols:
        lines.append(await run_once_for_symbol(sym))
    await send(CHAT_ID, "\n\n".join(lines))
    return {"ran": True}

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v = os.getenv(k)
        return {"present": bool(v), "len": len(v) if v else 0}
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
        "tg_running": tg_running,
        "polling_started": POLLING_STARTED
    }
