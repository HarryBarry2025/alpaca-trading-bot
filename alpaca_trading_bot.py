# alpaca_trading_bot.py  —  V5.1+ (TV-sync :30, SIP, PDT persist, Sizer, BT/WF, CSV/Plot)
# Python 3.11+, FastAPI, python-telegram-bot 20.x, alpaca-py 0.25+, yfinance, pandas, numpy, matplotlib

import os, io, json, time, math, asyncio, traceback, warnings
from datetime import datetime, timedelta, timezone
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
from telegram.error import Conflict

warnings.filterwarnings("ignore", category=UserWarning)

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None
if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "sip").lower()  # sip/iex

ENV_ENABLE_TRADE      = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER      = os.getenv("ENABLE_TIMER", "false").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES      = int(os.getenv("POLL_MINUTES", "60"))
ENV_MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

PDT_JSON_PATH = "/mnt/data/pdt_trades.json"  # daytrade persist

# ========= CONFIG / STATE =========
class StratConfig(BaseModel):
    # Engine
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"           # "1h" empfohlen (TV-synchron, :30 anchor)
    lookback_days: int = 365

    # Indicators (TV compatible)
    rsiLen: int = 12
    rsiLow: float = 0.0            # <- Default wie gewünscht
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Sizer
    sizing_mode: str = "percent_equity"    # "shares" | "percent_equity" | "notional_usd" | "risk"
    sizing_value: float = 100.0            # per Modus interpretiert
    max_position_pct: float = 100.0        # absolute Obergrenze pro Asset vs. Equity (0..100)

    # Fees / Slippage (Backtests)
    fee_per_order: float = 0.0
    slippage_bps: float = 0.0              # 1 bps = 0.01%

    # PDT
    pdt_soft_block: bool = True
    pdt_hard_block: bool = True

    # Timer
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = False
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY

    # Data Provider
    data_provider: str = "alpaca"   # default: alpaca (sip), fallback: yahoo -> stooq
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 1.5
    allow_stooq_fallback: bool = True

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}  # sim pos per symbol
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= HELPERS =========
def tz_utc_now() -> datetime:
    return datetime.now(timezone.utc)

def is_market_open_rth(dt_utc: Optional[datetime]=None) -> bool:
    """RTH grob: Mo–Fr, 13:30–20:00 UTC (9:30–16:00 ET); Holiday-Set minimal."""
    now = dt_utc or tz_utc_now()
    # simple holiday list (extend if needed)
    holidays = {
        "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
        "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
        "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27",
        "2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25",
    }
    if now.strftime("%Y-%m-%d") in holidays:
        return False
    if now.weekday() >= 5:
        return False
    hhmm = now.hour*60 + now.minute
    return 13*60+30 <= hhmm <= 20*60

# ========= INDICATORS (TV compatible) =========
def rsi_tv_wilder(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / max(1, length)
    rma_up = up.ewm(alpha=alpha, adjust=False).mean()
    rma_dn = down.ewm(alpha=alpha, adjust=False).mean()
    rs = rma_up / (rma_dn + 1e-12)
    return 100 - (100/(1+rs))

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def efi_tv(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    raw = volume * (close - close.shift(1))
    return ema(raw, max(1, length))

# ========= DATA FETCH & TV-SYNC =========
def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    from urllib.parse import quote
    url = f"https://stooq.com/q/d/l/?s={quote(symbol.lower())}.us&i=d"
    try:
        df = pd.read_csv(url)
        if df is None or df.empty: return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["time"] = df.index.tz_localize("UTC")
        if lookback_days>0:
            df = df.iloc[-lookback_days:]
        return df
    except Exception as e:
        print("[stooq] failed:", e)
        return pd.DataFrame()

def fetch_alpaca_minutes(symbol: str, start: datetime, end: datetime, feed: str) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] lib missing:", e)
        return pd.DataFrame()
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] keys missing"); return pd.DataFrame()
    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start, end=end, limit=100000,
        feed="sip" if feed=="sip" else "iex"
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
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
        df = df.rename(columns=str.lower).sort_index()
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch minutes failed:", e)
        return pd.DataFrame()

def filter_rth_minutes(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty: return dfm
    idx = dfm.index
    # RTH UTC window: 13:30 .. 20:00 inclusive starts, exclude extended
    mask = (idx.weekday < 5) & \
           ((idx.hour*60 + idx.minute) >= (13*60+30)) & \
           ((idx.hour*60 + idx.minute) <= (20*60))
    return dfm.loc[mask]

def aggregate_minutes_to_hour_anchored_30(dfm: pd.DataFrame, drop_last_incomplete: bool=True) -> pd.DataFrame:
    """
    Aggregiert 1-Minuten-Bars zu 1h-Bars, geankert auf :30 (z.B. 13:30–14:30).
    """
    if dfm.empty: return dfm
    # shift -30min, resample '1H', dann zurück-shiften
    shifted = dfm.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=30)
    agg = shifted.resample("1H", label="right", closed="right").agg({
        "open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum", "time":"last"
    }).dropna(how="any")
    # zurückshiften
    agg.index = agg.index + pd.Timedelta(minutes=30)
    agg["time"] = agg.index
    if drop_last_incomplete:
        # letzte fertige Stunde sicherstellen: wir nehmen nur bars deren "time" <= now - 1s (und mit mind. 1-Min in der Stunde)
        now = tz_utc_now()
        agg = agg.loc[agg.index <= (now.replace(second=0, microsecond=0) - pd.Timedelta(seconds=1))]
    return agg

def fetch_yahoo_intraday(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    # direkte 1h von Yahoo (nicht perfekt TV-sync), nutzen wir nur als Not-Fallback
    try:
        df = yf.download(symbol, interval=interval, period=f"{min(lookback_days,730)}d",
                         auto_adjust=False, progress=False, prepost=False, threads=False)
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns=str.lower)
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df["time"] = df.index
        return df
    except Exception as e:
        print("[yahoo] failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_tv_sync(symbol: str, interval: str, lookback_days: int, rth_only: bool=True) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    TV-synchrone 1h Bars aus Alpaca-Minuten → :30 anchor, RTH-Filter, drop incomplete.
    Fallback: Yahoo 1h (nicht 100% TV-sync), Stooq (1d).
    """
    note = {"provider":"","detail":""}
    if interval.lower() not in ("1h","60m"):
        # nur 1h unterstützen wir mit TV-Sync. Für 1d nehmen wir Alpaca daily (oder Yahoo/Stooq).
        # Für Kürze: fallback auf Yahoo 1d
        dfd = fetch_yahoo_intraday(symbol, "1d", lookback_days)
        if not dfd.empty:
            note.update(provider="Yahoo (1d)", detail="not TV-sync")
            return dfd, note
        dfe = fetch_stooq_daily(symbol, lookback_days)
        if not dfe.empty:
            note.update(provider="Stooq EOD (1d)", detail="fallback")
            return dfe, note
        return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

    end = tz_utc_now()
    start = end - timedelta(days=max(lookback_days, 60))
    feed = "sip" if APCA_DATA_FEED=="sip" else "iex"
    dfm = fetch_alpaca_minutes(symbol, start, end, feed=feed)

    if not dfm.empty and rth_only:
        dfm = filter_rth_minutes(dfm)

    if not dfm.empty:
        dfh = aggregate_minutes_to_hour_anchored_30(dfm, drop_last_incomplete=True)
        if not dfh.empty:
            note.update(provider=f"Alpaca ({feed})", detail="1h (:30 anchor, RTH filter)" if rth_only else "1h (:30 anchor)")
            return dfh, note

    # Fallback Yahoo (1h, NICHT TV-sync)
    dfy = fetch_yahoo_intraday(symbol, "1h", lookback_days)
    if not dfy.empty:
        note.update(provider="Yahoo (1h)", detail="fallback (not TV-sync)")
        return dfy, note

    # Stooq EOD
    dfe = fetch_stooq_daily(symbol, lookback_days)
    if not dfe.empty:
        note.update(provider="Stooq EOD (1d)", detail="fallback")
        return dfe, note

    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= FEATURES / SIGNALS =========
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv_wilder(out["close"], CONFIG.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], CONFIG.efiLen)
    out["rsi_rising"] = out["rsi"] > out["rsi"].shift(1)
    out["efi_rising"] = out["efi"] > out["efi"].shift(1)
    out["entry_cond"] = (out["rsi"] > CONFIG.rsiLow) & (out["rsi"] < CONFIG.rsiHigh) & out["rsi_rising"] & out["efi_rising"]
    out["exit_cond"]  = (out["rsi"] < CONFIG.rsiExit) & (~out["rsi_rising"])
    return out

# ========= PDT PERSISTENCE =========
def load_pdt_state() -> Dict[str,Any]:
    try:
        if os.path.exists(PDT_JSON_PATH):
            with open(PDT_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"trades": []}  # list of {"date":"YYYY-MM-DD","symbol":"TQQQ"}

def save_pdt_state(state: Dict[str,Any]) -> None:
    os.makedirs(os.path.dirname(PDT_JSON_PATH), exist_ok=True)
    with open(PDT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def pdt_can_trade_today() -> Tuple[bool, Dict[str,Any]]:
    """Returns (allowed, info). 5-day window, max 3 daytrades."""
    st = load_pdt_state()
    today = tz_utc_now().date()
    window_start = today - timedelta(days=5)
    recent = [t for t in st["trades"] if window_start <= datetime.fromisoformat(t["date"]).date() <= today]
    count = len(recent)
    allowed = count < 3
    return allowed, {"today":str(today), "window":f"{window_start}..{today}", "recent":recent, "count":count}

def pdt_register_daytrade(symbol: str) -> None:
    st = load_pdt_state()
    today = tz_utc_now().date()
    st["trades"].append({"date": today.isoformat(), "symbol": symbol})
    # keep last 90 entries max
    st["trades"] = st["trades"][-90:]
    save_pdt_state(st)

# ========= SIZER & ALPACA TRADING =========
def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

def alpaca_get_equity_cash() -> Tuple[float,float]:
    client = alpaca_trading_client()
    if client is None: return 0.0, 0.0
    try:
        acc = client.get_account()
        return float(acc.equity), float(acc.cash)
    except Exception:
        return 0.0, 0.0

def compute_order_qty(symbol: str, price: float) -> int:
    equity, cash = alpaca_get_equity_cash()
    if price <= 0: return 0

    # Max pos notional by equity cap
    max_notional_by_cap = (CONFIG.max_position_pct/100.0) * max(1.0, equity)

    if CONFIG.sizing_mode == "shares":
        qty = int(max(0, math.floor(CONFIG.sizing_value)))
        max_qty_cap = int(max(0, math.floor(max_notional_by_cap / price)))
        return max(0, min(qty, max_qty_cap))

    elif CONFIG.sizing_mode == "percent_equity":
        target_notional = (CONFIG.sizing_value/100.0) * max(1.0, equity)
        target_notional = min(target_notional, max_notional_by_cap)
        qty = int(max(0, math.floor(target_notional / price)))
        return qty

    elif CONFIG.sizing_mode == "notional_usd":
        target_notional = min(CONFIG.sizing_value, max_notional_by_cap)
        qty = int(max(0, math.floor(target_notional / price)))
        return qty

    elif CONFIG.sizing_mode == "risk":
        # Simple placeholder: use percent_equity logic for now
        target_notional = (CONFIG.sizing_value/100.0) * max(1.0, equity)
        target_notional = min(target_notional, max_notional_by_cap)
        qty = int(max(0, math.floor(target_notional / price)))
        return qty

    # default
    target_notional = (CONFIG.sizing_value/100.0) * max(1.0, equity)
    target_notional = min(target_notional, max_notional_by_cap)
    return int(max(0, math.floor(target_notional / price)))

async def place_market_order(symbol: str, qty: int, side: str, tif: str = "day") -> str:
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
            symbol=symbol, qty=qty,
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
        print("alpaca_positions error:", e)
        return []

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

# ========= CORE STEP =========
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if prov:
        return f"Quelle: {prov} ({det})"
    return ""

async def run_once_for_symbol(sym: str, send_signals: bool=True) -> Dict[str,Any]:
    rth = CONFIG.market_hours_only
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=rth)
    if df.empty:
        return {"ok":False, "msg": f"❌ Keine Daten für {sym}. {note}"}

    f = build_features(df)
    last = f.iloc[-1]; prev = f.iloc[-2] if len(f)>=2 else last

    action = "none"; reason = "hold"
    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]

    entry_cond = bool(last["entry_cond"])
    exit_cond  = bool(last["exit_cond"])

    price_open  = float(last["open"])
    price_close = float(last["close"])

    def sl(p): return p*(1-CONFIG.slPerc/100.0)
    def tp(p): return p*(1+CONFIG.tpPerc/100.0)

    if size == 0:
        if entry_cond:
            # PDT hard stop?
            if CONFIG.pdt_hard_block:
                can_trade, info = pdt_can_trade_today()
                if not can_trade:
                    action, reason = "none", f"pdt_block ({info['count']} daytrades in 5d)"
                else:
                    q = compute_order_qty(sym, price_open)
                    if q>0:
                        action, reason = "buy", "rule_entry"
                    else:
                        action, reason = "none", "qty=0"
            else:
                q = compute_order_qty(sym, price_open)
                if q>0:
                    action, reason = "buy", "rule_entry"
                else:
                    action, reason = "none", "qty=0"
        else:
            action, reason = "none", "flat_no_entry"
    else:
        # RSI Exit
        bars_in_trade = 0
        if pos["entry_time"] is not None:
            since = f[f["time"] >= pd.to_datetime(pos["entry_time"], utc=True)]
            bars_in_trade = max(0, len(since)-1)
        same_bar_ok = CONFIG.allowSameBarExit or (bars_in_trade>0)
        cooldown_ok = (bars_in_trade >= CONFIG.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(avg); cur_tp = tp(avg)
        hit_sl = price_close <= cur_sl
        hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            action, reason = "sell", "rsi_exit"
        elif hit_sl:
            action, reason = "sell", "stop_loss"
        elif hit_tp:
            action, reason = "sell", "take_profit"
        else:
            action, reason = "none", "hold"

    STATE.last_status = f"{sym}: {action} ({reason})"

    # Telegram info
    if send_signals and CHAT_ID:
        await tg_send(f"ℹ️ {sym} {CONFIG.interval} rsi={last['rsi']:.2f} efi={last['efi']:.2f} • {reason}\n{friendly_note(note)}")

    # Execute / Simulate
    if action == "buy":
        q = compute_order_qty(sym, price_open)
        if CONFIG.pdt_hard_block:
            can_trade, info = pdt_can_trade_today()
            if not can_trade:
                if CHAT_ID: await tg_send(f"⛔ PDT: Blockiert ({info['count']} Daytrades/5d). Kein Kauf.")
            else:
                # Live trade?
                if CONFIG.trade_enabled:
                    info_s = await place_market_order(sym, q, "buy", tif="day")
                    if CHAT_ID: await tg_send(f"🛒 BUY {sym} x{q} @ ~{price_open:.4f} • {info_s}")
                # Sim pos:
                STATE.positions[sym] = {"size":q,"avg":price_open,"entry_time":str(last["time"])}
                if CHAT_ID and not CONFIG.trade_enabled:
                    await tg_send(f"🟢 LONG (sim) {sym} x{q} @ {price_open:.4f} | SL={sl(price_open):.4f} TP={tp(price_open):.4f}")
        else:
            if CONFIG.trade_enabled:
                info_s = await place_market_order(sym, q, "buy", tif="day")
                if CHAT_ID: await tg_send(f"🛒 BUY {sym} x{q} @ ~{price_open:.4f} • {info_s}")
            STATE.positions[sym] = {"size":q,"avg":price_open,"entry_time":str(last["time"])}

    elif action == "sell" and size>0:
        if CONFIG.trade_enabled:
            info_s = await place_market_order(sym, size, "sell", tif="day")
            if CHAT_ID: await tg_send(f"🛒 SELL {sym} x{size} @ ~{price_close:.4f} • {info_s}")
        pnl = (price_close-avg)/max(1e-9,avg)
        if CHAT_ID and not CONFIG.trade_enabled:
            await tg_send(f"🔴 EXIT (sim) {sym} @ {price_close:.4f} • {reason} • PnL={pnl*100:.2f}%")
        # PDT register (simple: any same-day round trip)
        try:
            ent_date = datetime.fromisoformat(STATE.positions[sym]["entry_time"]).date()
            if ent_date == tz_utc_now().date():
                pdt_register_daytrade(sym)
        except Exception:
            pass
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}

    return {"ok":True,"action":action,"reason":reason}

# ========= TELEGRAM =========
tg_app: Optional[Application] = None
POLLING_STARTED = False

async def tg_send(text: str):
    global tg_app
    if tg_app is None or not CHAT_ID: return
    await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)

# Commands
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Verbunden.\n"
        "Befehle:\n"
        "/status /cfg /set key=value … /run /sig /ind /plot /dump [/csv N] /dumpcsv [N]\n"
        "/bt [days] /wf [is oos]\n"
        "/trade on|off /pos /account\n"
        "/timer on|off /timerstatus /timerrunnow /market on|off\n"
        "/tgstatus"
    )

async def cmd_status(update, context):
    pos_lines=[]
    for s,p in STATE.positions.items():
        pos_lines.append(f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}")
    pos_txt="\n".join(pos_lines) if pos_lines else "keine (sim)"
    acc = alpaca_account()
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider} ({APCA_DATA_FEED})\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} •"
        f" Timer: {'ON' if TIMER['enabled'] else 'OFF'} (alle {TIMER['poll_minutes']}m, market-hours-only={TIMER['market_hours_only']})\n"
        f"Sizer: {CONFIG.sizing_mode}={CONFIG.sizing_value}  max_pos={CONFIG.max_position_pct}%\n"
        f"PDT: soft={CONFIG.pdt_soft_block} hard={CONFIG.pdt_hard_block}\n"
        f"Last: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}\n"
        f"Account: {json.dumps(acc, indent=2) if acc else 'n/a'}"
    )

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

def set_from_kv(kv: str) -> str:
    k,v = kv.split("=",1); k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k,k)
    if not hasattr(CONFIG, k):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):   setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):  setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):setattr(CONFIG, k, float(v))
    elif isinstance(cur, list): setattr(CONFIG, k, [x.strip() for x in v.split(",") if x.strip()])
    else:                       setattr(CONFIG, k, v)
    if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value …  z.B.\n"
            "/set interval=1h lookback_days=365\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=2 tp=4\n"
            "/set sizing_mode=percent_equity sizing_value=100 max_position_pct=100\n"
            "/set poll_minutes=60 market_hours_only=true\n"
            "/set pdt_soft_block=true pdt_hard_block=true"
        ); return
    msgs=[]; errs=[]
    for a in context.args:
        if "=" not in a: errs.append(f"❌ ungültig: {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = ("✅ Übernommen:\n" + "\n".join(msgs) + ("\n\n⚠️ Probleme:\n" + "\n".join(errs) if errs else "")).strip()
    await update.message.reply_text(txt)

async def cmd_run(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.market_hours_only)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df)
    last = f.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} {CONFIG.interval}\n"
        f"RSI={last['rsi']:.2f} (rising={bool(last['rsi_rising'])})  "
        f"EFI={last['efi']:.2f} (rising={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}\n"
        f"{friendly_note({'provider':note['provider'],'detail':note['detail']})}"
    )

async def cmd_ind(update, context):
    await cmd_sig(update, context)

async def _send_plot(chat_id: str, fig, name: str, caption: str):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    await tg_app.bot.send_document(chat_id=chat_id, document=InputFile(buf, filename=name), caption=caption)

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.market_hours_only)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df).tail(300)

    # Close
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(f.index, f["close"], label="Close")
    ax.set_title(f"{sym} {CONFIG.interval} Close"); ax.grid(True); ax.legend()
    await _send_plot(str(update.effective_chat.id), fig, f"{sym}_{CONFIG.interval}_close.png", "📈 Close")

    # RSI
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(f.index, f["rsi"], label="RSI")
    ax2.axhline(CONFIG.rsiLow, linestyle="--"); ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.set_title("RSI (Wilder)"); ax2.grid(True); ax2.legend()
    await _send_plot(str(update.effective_chat.id), fig2, f"{sym}_{CONFIG.interval}_rsi.png", "📈 RSI")

    # EFI
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(f.index, f["efi"], label="EFI"); ax3.grid(True); ax3.legend()
    ax3.set_title("EFI (EMA(vol*Δclose))")
    await _send_plot(str(update.effective_chat.id), fig3, f"{sym}_{CONFIG.interval}_efi.png", "📈 EFI")

def build_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = build_features(df)
    cols = ["open","high","low","close","volume","rsi","efi","rsi_rising","efi_rising","entry_cond","exit_cond","time"]
    return f[cols]

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    args = [a.lower() for a in (context.args or [])]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.market_hours_only)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    if args and args[0]=="csv":
        n=300
        if len(args)>=2:
            try: n=max(1,int(args[1]))
            except: pass
        exp = build_export_frame(df).tail(n)
        data = exp.to_csv(index=True).encode("utf-8")
        bio = io.BytesIO(data); bio.name=f"{sym}_{CONFIG.interval}_indicators_{n}.csv"; bio.seek(0)
        await tg_app.bot.send_document(chat_id=str(update.effective_chat.id), document=InputFile(bio), caption="🧾 CSV Export")
        return
    f = build_features(df)
    last = f.iloc[-1]
    payload = {
        "symbol": sym, "interval": CONFIG.interval, "provider": note["provider"],
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

# ======= Backtest & Walk-Forward (mit PDT, Fees, Slippage) =======
def apply_slippage(price: float, bps: float, side: str) -> float:
    if bps <= 0: return price
    # bei Buy verteuern, Sell verbilligen
    factor = (1 + bps/10000.0) if side=="buy" else (1 - bps/10000.0)
    return price * factor

def backtest_series(df: pd.DataFrame) -> Dict[str,Any]:
    f = build_features(df)
    eq=1.0; pos_size=0; avg=0.0
    fees=CONFIG.fee_per_order; slip=CONFIG.slippage_bps
    trades=0; wins=0; R=[]

    # PDT sim
    pdt_state={"trades":[]}

    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry = bool(row["entry_cond"])
        exitc = bool(row["exit_cond"])
        o = float(row["open"]); c=float(row["close"])

        if pos_size==0 and entry:
            # PDT check
            today = pd.Timestamp(row["time"]).date()
            window_start = today - timedelta(days=5)
            recent = [t for t in pdt_state["trades"] if window_start <= t["date"] <= today]
            if CONFIG.pdt_hard_block and len(recent) >= 3:
                pass  # blocked
            else:
                # size proportional zu eq (sim)
                price = apply_slippage(o, slip, "buy")
                notional = 1.0  # wir handeln "ein Anteil" relativ => eq wird mit (1+r) skaliert
                pos_size = 1
                avg = price
                eq -= fees/10000.0  # mini-impact; fees in eq-terms
        elif pos_size==1:
            sl = avg*(1-CONFIG.slPerc/100.0); tp=avg*(1+CONFIG.tpPerc/100.0)
            stop = c<=sl; take=c>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                price = apply_slippage(px, slip, "sell")
                r = (price-avg)/avg
                eq *= (1+r)
                eq -= fees/10000.0
                trades+=1
                if r>0: wins+=1
                # PDT reg (round trip same day)
                ent_day = pd.Timestamp(f.iloc[i-1]["time"]).date()
                if ent_day == pd.Timestamp(row["time"]).date():
                    pdt_state["trades"].append({"date": pd.Timestamp(row["time"]).date()})
                pos_size=0; avg=0.0

    out = {"trades":trades, "wins":wins, "winrate": (wins/max(1,trades)), "eq":eq}
    return out

def walk_forward(df: pd.DataFrame, is_days: int, oos_days: int) -> Dict[str,Any]:
    # Simplified WF: grid über (rsiLow,rsiHigh,rsiExit), eval is-> pick best -> run oos mit best params
    base = CONFIG.dict()
    f_all = build_features(df).copy()
    dates = f_all.index.normalize().unique()
    chunks=[]
    start_idx=0
    while True:
        is_end = start_idx + is_days
        oos_end = is_end + oos_days
        if oos_end > len(dates): break
        is_mask = (f_all.index.normalize()>=dates[start_idx]) & (f_all.index.normalize()<dates[is_end])
        oos_mask= (f_all.index.normalize()>=dates[is_end])   & (f_all.index.normalize()<dates[oos_end])
        df_is  = f_all.loc[is_mask]
        df_oos = f_all.loc[oos_mask]
        # grid
        grid=[]
        for rlow in [0, 40, 45, 50]:
            for rhigh in [60, 65, 68, 70]:
                for rexit in [40, 45, 48, 50]:
                    CONFIG.rsiLow, CONFIG.rsiHigh, CONFIG.rsiExit = float(rlow), float(rhigh), float(rexit)
                    res = backtest_series(df_is)
                    grid.append((res["eq"], rlow, rhigh, rexit))
        grid.sort(reverse=True, key=lambda x: x[0])
        best = grid[0]
        CONFIG.rsiLow, CONFIG.rsiHigh, CONFIG.rsiExit = float(best[1]), float(best[2]), float(best[3])
        res_oos = backtest_series(df_oos)
        chunks.append({"IS":best, "OOS":res_oos})
        start_idx += oos_days
    # restore
    CONFIG.__dict__.update(base)
    return {"chunks":chunks}

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, days, rth_only=CONFIG.market_hours_only)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten für Backtest."); return
    res = backtest_series(df)
    # CAGR grob
    years = max(1e-9, days/365.0)
    cagr = (res["eq"] ** (1/years)) - 1 if res["eq"]>0 else -1
    await update.message.reply_text(
        f"📈 Backtest {days}d  Trades={res['trades']}  Win={res['winrate']*100:.1f}%  CAGR~{cagr*100:.2f}%\n"
        f"ℹ️ Hinweis: Fees={CONFIG.fee_per_order}, Slippage={CONFIG.slippage_bps}bps, EoB-Logik, PDT berücksichtigt."
    )

async def cmd_wf(update, context):
    is_days, oos_days = 120, 30
    if context.args and len(context.args)>=2:
        try:
            is_days = int(context.args[0]); oos_days = int(context.args[1])
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.market_hours_only)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    res = walk_forward(df, is_days, oos_days)
    await update.message.reply_text("🏃 Walk-Forward (IS/OOS)\n" + json.dumps(res, indent=2, default=str))

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

async def cmd_timer(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /timer on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    TIMER["enabled"] = on
    if on and TIMER_TASK is None:
        # falls Timer noch nicht läuft → starten
        start_timer_task()
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
    now = tz_utc_now()
    TIMER["last_run"]=now.isoformat()
    TIMER["next_due"]=(now + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def cmd_market(update, context):
    if not context.args:
        await update.message.reply_text(f"Market-hours-only ist {'ON' if TIMER['market_hours_only'] else 'OFF'}\nNutze: /market on|off")
        return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.market_hours_only = on
    TIMER["market_hours_only"] = on
    await update.message.reply_text(f"Market-hours-only = {'ON' if on else 'OFF'}")

async def cmd_tgstatus(update, context):
    await update.message.reply_text(json.dumps({"polling_started": POLLING_STARTED}, indent=2))

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= TIMER (sync to :30 for 1h) =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}
TIMER_TASK: Optional[asyncio.Task] = None

def next_half_hour_utc(now: datetime) -> datetime:
    # nächste :30 oder :30 + n*60
    minute = now.minute
    # align to 30-minute grid, then +60min for next bar close
    # we want to trigger shortly after bar close at :30
    # compute last :30 boundary:
    base_min = 30 if minute >= 30 else 0
    base = now.replace(minute=base_min, second=0, microsecond=0)
    if base_min == 30:
        # next close is base + 60min => :30 next hour
        nxt = base + timedelta(hours=1)
    else:
        # minute < 30 → next close is today :30
        nxt = base + timedelta(minutes=30)
    return nxt

async def timer_loop():
    TIMER["running"] = True
    try:
        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)

            # Initialisierung: wenn noch nie gelaufen oder next_due fehlt → sofort 1x laufen, dann next_due setzen
            if TIMER.get("last_run") is None or TIMER.get("next_due") is None:
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                TIMER["last_run"] = now.isoformat()
                TIMER["next_due"] = _align_next_due(now).isoformat()
                await asyncio.sleep(5)
                continue

            # Market Hours Filter: NICHT vor Initialisierung anwenden
            if TIMER.get("market_hours_only", False) and not is_market_open_now():
                # Schlafen, aber next_due nicht löschen – beim nächsten Open wird geprüft
                await asyncio.sleep(30)
                continue

            # Fälligkeit prüfen
            try:
                due_dt = datetime.fromisoformat(TIMER["next_due"])
            except Exception:
                # Falls korrupt → neu setzen
                due_dt = _align_next_due(now)
                TIMER["next_due"] = due_dt.isoformat()

            if now >= due_dt:
                # Run ausführen
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                now = datetime.now(timezone.utc)
                TIMER["last_run"] = now.isoformat()
                TIMER["next_due"] = _align_next_due(now).isoformat()

            await asyncio.sleep(5)

    finally:
        TIMER["running"] = False

def start_timer_task():
    global TIMER_TASK
    if TIMER_TASK is None or TIMER_TASK.done():
        TIMER_TASK = asyncio.create_task(timer_loop())

from datetime import datetime, timedelta, timezone

def _align_next_due(now_utc: datetime) -> datetime:
    """
    Liefert das nächste Fälligkeitsdatum:
    - 1h-Intervall: immer zur vollen halben Stunde (:30) in UTC (TV-konform)
    - 15m/5m: analog auf Candle-Grenzen ausrichtbar (hier Fokus 1h)

    Berücksichtigt Market Hours:
      - Wenn RTH-only und gerade Closed => auf den nächsten RTH-Slot ausrichten.
    """
    interval = CONFIG.interval.lower()
    rth_only = TIMER.get("market_hours_only", False)

    def next_rth_anchor(base: datetime) -> datetime:
        # RTH grob (13:30–20:00 UTC), nächste :30 finden
        d = base
        # springe falls vor 13:30 auf 13:30; falls nach 20:00 auf nächsten Tag 13:30
        hhmm = d.hour * 60 + d.minute
        if hhmm < 13*60 + 30:
            d = d.replace(hour=13, minute=30, second=0, microsecond=0)
        elif hhmm >= 20*60:
            d = (d + timedelta(days=1)).replace(hour=13, minute=30, second=0, microsecond=0)
        # auf nächste :30 runden
        if d.minute < 30:
            d = d.replace(minute=30, second=0, microsecond=0)
        else:
            # nächste volle Stunde :30
            d = (d + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0)
        # Sa/So → auf Montag 13:30
        while d.weekday() >= 5:
            d = (d + timedelta(days=1)).replace(hour=13, minute=30, second=0, microsecond=0)
        return d

    # 1) Intervall-spezifisch ausrichten
    if interval in ("1h", "60m"):
        # nächster :30-Anker
        base = now_utc
        if now_utc.minute < 30:
            due = now_utc.replace(minute=30, second=0, microsecond=0)
        else:
            due = (now_utc + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0)
    else:
        # generischer Fallback: jetzt + poll_minutes
        due = now_utc + timedelta(minutes=TIMER["poll_minutes"])

    # 2) Market Hours berücksichtigen
    if rth_only and not is_market_open_now(due):
        due = next_rth_anchor(due)

    return due

# ========= FASTAPI LIFESPAN =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("dumpcsv", cmd_dumpcsv))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("wf",      cmd_wf))
        tg_app.add_handler(CommandHandler("trade",   cmd_trade))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(CommandHandler("account", cmd_account))
        tg_app.add_handler(CommandHandler("timer",        cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("timerrunnow",  cmd_timerrunnow))
        tg_app.add_handler(CommandHandler("market",       cmd_market))
        tg_app.add_handler(CommandHandler("tgstatus",     cmd_tgstatus))
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
        if TIMER["enabled"]:
            start_timer_task()
            print("⏱️ Timer gestartet (TV :30 sync für 1h)")
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

# ========= FASTAPI APP =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": f"{CONFIG.data_provider} ({APCA_DATA_FEED})",
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
    now = tz_utc_now()
    TIMER["last_run"]=now.isoformat()
    TIMER["next_due"]=next_half_hour_utc(now).isoformat()
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
