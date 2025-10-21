# alpaca_trading_bot.py
# ——————————————————————————————————————————————————————————
# Features:
# - TV-kompatibler RSI (Wilder RMA) & EFI
# - Alpaca (default) + Yahoo -> Stooq Fallback
# - TV-synchroner Timer (1h -> :30, 15m -> :00/:15/:30/:45, 1d -> 20:00 UTC)
# - Walk-Forward/OOS (/wf) mit einfacher Grid-Search
# - Telegram-Bot (PTB v20.7): /status /cfg /set /run /live /bt /sig /ind /plot /dump /dumpcsv
#                            /trade /pos /account /timer /timerstatus /timerrunnow /syncdebug /wf
# - Paper-Trading via Alpaca (Market-Order)
# ——————————————————————————————————————————————————————————

import os, io, json, time, asyncio, traceback, warnings
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.x)
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

# Alpaca (Paper)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
ALPACA_DATA_FEED    = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()  # 'iex' (free) oder 'sip' (kostenpflichtig)

# Trading- & Timer-Flags
ENV_ENABLE_TRADE      = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER      = os.getenv("ENABLE_TIMER", "false").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES      = int(os.getenv("POLL_MINUTES", "10"))
ENV_MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    symbols: List[str] = ["TQQQ"]    # Multi-Asset
    interval: str = "1h"             # '1h','15m','1d' …
    lookback_days: int = 365

    # TV-kompatible Inputs (Wilder RSI / EFI)
    rsiLen: int = 12
    rsiLow: float = 0.0         # explizit Default 0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine / Timer
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = False
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY
    candle_sync: bool = True     # Timer an Candle-Kante ausrichten

    # Data Provider
    data_provider: str = "alpaca"  # 'alpaca' (default), 'yahoo', 'stooq_eod'
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Trading Toggle
    trade_enabled: bool = ENV_ENABLE_TRADE
    alpaca_tif: str = "day"  # 'day','gtc','opg','cls','ioc','fok'

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}  # {symbol: {"size":int,"avg":float,"entry_time":str|None}}
    last_status: str = "idle"

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

# ========= US Market Hours (grob) =========
def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or datetime.now(timezone.utc)
    # rudimentär: Mo–Fr, 13:30–20:00 UTC (9:30–16:00 ET) – ohne Sondertage
    if now.weekday() >= 5:
        return False
    mins = now.hour*60 + now.minute
    return 13*60+30 <= mins <= 20*60

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

    iv = interval.lower()
    if iv in {"1h","60m"}:
        tf = TimeFrame(1, TimeFrameUnit.Hour)
    elif iv in {"15m"}:
        tf = TimeFrame(15, TimeFrameUnit.Minute)
    elif iv in {"1d","1day"}:
        tf = TimeFrame(1, TimeFrameUnit.Day)
    else:
        tf = TimeFrame(1, TimeFrameUnit.Hour)

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 90))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        feed=ALPACA_DATA_FEED,   # 'iex' (free) oder 'sip' (abo)
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
        df = df.sort_index()
        df = df.rename(columns=str.lower)
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    # Primär: Alpaca
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider=f"Alpaca ({ALPACA_DATA_FEED})", detail=interval)
            return df, note
        note.update(provider=f"Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")

    # Yahoo
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}
    is_intraday = interval.lower() in intraday_set
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
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True,  threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    last_err=None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = _norm(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}")
                    return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # Yahoo 1d fallback
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 120)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # Stooq fallback
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 180))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] empty for {symbol} ({interval}). Last error: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Features & Signals =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv_wilder(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

def compute_signals_for_frame(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    f = build_features(df, cfg)
    f["rsi_rising"] = f["rsi"] > f["rsi"].shift(1)
    f["efi_rising"] = f["efi"] > f["efi"].shift(1)
    f["entry_cond"] = (f["rsi"] > cfg.rsiLow) & (f["rsi"] < cfg.rsiHigh) & f["rsi_rising"] & f["efi_rising"]
    f["exit_cond"]  = (f["rsi"] < cfg.rsiExit) & (~f["rsi_rising"])
    return f

def build_export_frame(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    f = compute_signals_for_frame(df, cfg)
    cols = ["open","high","low","close","volume","rsi","efi","rsi_rising","efi_rising","entry_cond","exit_cond","time"]
    return f[cols]

# ========= Strategy step =========
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

    def sl(p): return p*(1-cfg.slPerc/100.0)
    def tp(p): return p*(1+cfg.tpPerc/100.0)

    bars_in_trade=0
    if entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if size==0:
        if entry_cond:
            q = 1
            return {"action":"buy","symbol":sym,"qty":q,"px":price_open,"time":str(ts),
                    "sl":sl(price_open),"tp":tp(price_open),"reason":"rule_entry",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        return {"action":"none","symbol":sym,"reason":"flat_no_entry",
                "rsi":float(last["rsi"]), "efi":float(last["efi"])}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade>0)
        cooldown_ok = (bars_in_trade>=cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(avg); cur_tp = tp(avg)
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
def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

async def place_market_order(sym: str, qty: int, side: str, tif: str = None) -> str:
    client = alpaca_trading_client()
    if client is None: return "alpaca client not available"
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tif_map = {
            "day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "opg": TimeInForce.OPG,
            "cls": TimeInForce.CLS, "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK
        }
        tif_eff = tif or CONFIG.alpaca_tif
        req = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif_map.get(tif_eff.lower(), TimeInForce.DAY)
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

# ========= Single-step runner =========
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    return f"📡 Datenquelle: {prov} ({det})"

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        return {"ok":False, "msg": f"❌ Keine Daten für {sym}. {note}"}

    act = bar_logic_last(df, CONFIG, sym)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    if send_signals and CHAT_ID:
        await send_text(CHAT_ID, f"ℹ️ {sym} {CONFIG.interval} rsi={act.get('rsi',np.nan):.2f} efi={act.get('efi',np.nan):.2f} • {act['reason']}")
        # Fallback-Hinweis nur bei Nicht-Standard
        if "Alpaca" not in note.get("provider",""):
            await send_text(CHAT_ID, friendly_note(note))

    # Trading (Paper) wenn erlaubt
    if CONFIG.trade_enabled and act["action"] in ("buy","sell"):
        if CONFIG.market_hours_only and not is_market_open_now():
            if CHAT_ID:
                await send_text(CHAT_ID, "⛔ Markt geschlossen – kein Trade ausgeführt.")
        else:
            side = "buy" if act["action"]=="buy" else "sell"
            info = await place_market_order(sym, int(act["qty"]), side)
            if CHAT_ID:
                await send_text(CHAT_ID, f"🛒 {side.upper()} {sym} x{act['qty']} @ {act['px']:.4f} • {info}")

    # Sim-Pos für Status
    pos = STATE.positions.get(sym, {"size":0, "avg":0.0, "entry_time":None})
    if act["action"]=="buy" and pos["size"]==0:
        STATE.positions[sym] = {"size":act["qty"],"avg":act["px"],"entry_time":act["time"]}
        if CHAT_ID and send_signals:
            await send_text(CHAT_ID, f"🟢 LONG (sim) {sym} @ {act['px']:.4f} | SL={act.get('sl',np.nan):.4f} TP={act.get('tp',np.nan):.4f}")
    elif act["action"]=="sell" and pos["size"]>0:
        pnl = (act["px"] - pos["avg"]) / pos["avg"]
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if CHAT_ID and send_signals:
            await send_text(CHAT_ID, f"🔴 EXIT (sim) {sym} @ {act['px']:.4f} • {act['reason']} • PnL={pnl*100:.2f}%")

    return {"ok":True,"act":act}

# ========= Timer mit TV-Sync =========
def next_tv_aligned_due(now: datetime, interval: str, minutes_step: int) -> datetime:
    """Berechne nächste Ausführungszeit an der TV-Kante."""
    iv = interval.lower()
    if iv in ("60m","1h"):
        # Ziel ist :30 jeder Stunde (TradingView 1h Bar schließt :30 UTC)
        base = now.replace(second=0, microsecond=0)
        minute = base.minute
        if minute < 30:
            due = base.replace(minute=30)
        elif minute == 30 and base.second == 0:
            due = base + timedelta(hours=1)  # nächste :30
        else:
            # >30 -> zur nächsten vollen Stunde und dann :30 ergänzen
            due = (base + timedelta(hours=1)).replace(minute=30)
        return due
    if iv in ("15m","15"):
        # 00/15/30/45
        base = now.replace(second=0, microsecond=0)
        m = base.minute
        step = 15
        add = (step - (m % step)) % step
        due = base + timedelta(minutes=add or step)
        return due
    if iv in ("1d","1day","d"):
        # 20:00 UTC (NYSE close)
        base = now.astimezone(timezone.utc)
        target = base.replace(hour=20, minute=0, second=0, microsecond=0)
        if base >= target:
            target = target + timedelta(days=1)
        return target
    # generisch: Minutenraster
    base = now.replace(second=0, microsecond=0)
    step = minutes_step
    add = (step - (base.minute % step)) % step
    return base + timedelta(minutes=add or step)

TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}
TIMER_TASK: Optional[asyncio.Task] = None

async def timer_loop():
    TIMER["running"] = True
    try:
        # Initial an die Kante setzen
        now = datetime.now(timezone.utc)
        if CONFIG.candle_sync:
            due = next_tv_aligned_due(now, CONFIG.interval, TIMER["poll_minutes"])
        else:
            due = now + timedelta(minutes=TIMER["poll_minutes"])
        TIMER["next_due"] = due.isoformat()

        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)

            if TIMER["market_hours_only"] and not is_market_open_now(now):
                # warte 60s und prüfe erneut
                await asyncio.sleep(60)
                continue

            # Fällig?
            if now >= pd.to_datetime(TIMER["next_due"]):
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                TIMER["last_run"] = now.isoformat()
                if CONFIG.candle_sync:
                    due = next_tv_aligned_due(now + timedelta(seconds=1), CONFIG.interval, TIMER["poll_minutes"])
                else:
                    due = now + timedelta(minutes=TIMER["poll_minutes"])
                TIMER["next_due"] = due.isoformat()

            await asyncio.sleep(1)
    finally:
        TIMER["running"] = False

# ========= Walk-Forward / OOS =========
def backtest_simple(f: pd.DataFrame, cfg: StratConfig) -> Dict[str, Any]:
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry = bool(row["entry_cond"])
        exitc = bool(row["exit_cond"])
        if pos==0 and entry:
            pos=1; avg=float(row["open"]); entries+=1
        elif pos==1:
            sl = avg*(1-cfg.slPerc/100); tp = avg*(1+cfg.tpPerc/100)
            price = float(row["close"])
            stop = price<=sl; take=price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                r = (px-avg)/avg
                eq *= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    res = {"trades": entries, "exits": exits, "equity": eq}
    if R:
        a = np.array(R)
        win = (a>0).mean()
        pf  = a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
    else:
        win = 0.0; pf = 0.0
    res.update({"win":win, "pf":pf})
    return res

def walk_forward_oos(df: pd.DataFrame, cfg: StratConfig, is_days: int, oos_days: int,
                     grid: List[Tuple[float,float,float]]) -> Dict[str,Any]:
    """
    Rollendes Walk-Forward:
      - Split in Blöcke von IS (Train) + OOS (Test), über gesamten Zeitraum
      - Auf IS wird Grid-Search über (rsiLow, rsiHigh, rsiExit) gemacht
      - OOS mit den besten IS-Parametern evaluiert
    """
    res_blocks=[]
    df = df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC")
    start = df.index[0]
    end   = df.index[-1]

    # iteriere Tagesfenster anhand Index-Zeit
    cur_start = df.index[0]
    while True:
        is_end_time  = df[df["date"] < (df["date"][0] + pd.Timedelta(days=is_days))].index[-1] if len(df)>0 else None
        if is_end_time is None: break
        is_mask  = (df.index >= df.index[0]) & (df.index <= is_end_time)
        is_df    = df.loc[is_mask]
        # OOS Start = direkt nach IS-Ende
        try:
            oos_start_idx = df.index.get_loc(is_end_time) + 1
        except Exception:
            break
        if oos_start_idx >= len(df): break
        oos_end_idx = min(len(df)-1, oos_start_idx + max(1, oos_days*24*60//15 if cfg.interval=="15m" else oos_days))
        oos_df = df.iloc[oos_start_idx:oos_end_idx+1]
        if len(is_df)<50 or len(oos_df)<10: break

        # Grid auf IS
        best=None; best_eq=-1
        base_cfg = cfg.copy()
        for (lo, hi, ex) in grid:
            base_cfg.rsiLow  = lo
            base_cfg.rsiHigh = hi
            base_cfg.rsiExit = ex
            f_is  = compute_signals_for_frame(is_df, base_cfg)
            met   = backtest_simple(f_is, base_cfg)
            if met["equity"]>best_eq:
                best_eq = met["equity"]; best=(lo,hi,ex,met)

        # OOS mit best
        if best is None: break
        lo,hi,ex,_ = best
        best_cfg = cfg.copy()
        best_cfg.rsiLow, best_cfg.rsiHigh, best_cfg.rsiExit = lo,hi,ex
        f_oos = compute_signals_for_frame(oos_df, best_cfg)
        met_oos = backtest_simple(f_oos, best_cfg)

        res_blocks.append({
            "is_period": [str(is_df["time"].iloc[0]), str(is_df["time"].iloc[-1])],
            "oos_period": [str(oos_df["time"].iloc[0]), str(oos_df["time"].iloc[-1])],
            "best_params": {"rsiLow":lo,"rsiHigh":hi,"rsiExit":ex},
            "is_metrics": best[3],
            "oos_metrics": met_oos
        })

        # Fenster weiterschieben: OOS-Ende -> neue Basis
        if oos_end_idx+1 >= len(df): break
        df = df.iloc[oos_end_idx+1:]  # restlicher Teil

    return {"blocks": res_blocks}

# ========= TV-Sync Debug =========
def _is_tv_close(ts_utc: pd.Timestamp, interval: str) -> bool:
    if not isinstance(ts_utc, pd.Timestamp):
        ts_utc = pd.to_datetime(ts_utc, utc=True)
    m = ts_utc.minute; h = ts_utc.hour
    iv = interval.lower()
    if iv in ("60m","1h"):
        return m in (30, 0)  # toleranter Check (Datenquellen variieren)
    if iv in ("15m","15"):
        return m in (0,15,30,45)
    if iv in ("1d","1day","d"):
        return (h,m)==(20,0) # 20:00 UTC
    try:
        if iv.endswith("m"):
            step=int(iv[:-1]); return m%step==0
    except: pass
    return True

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg, /set key=value …\n"
        "/run, /live on|off, /bt [tage]\n"
        "/sig, /ind, /plot\n"
        "/dump [csv [N]], /dumpcsv [N]\n"
        "/trade on|off, /pos, /account\n"
        "/timer on|off, /timerstatus, /timerrunnow\n"
        "/syncdebug [N], /wf [is_days oos_days]"
    )

async def cmd_status(update, context):
    pos_lines=[]
    for s,p in STATE.positions.items():
        pos_lines.append(f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}")
    pos_txt = "\n".join(pos_lines) if pos_lines else "keine (sim)"
    acc = alpaca_account()
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider} (Alpaca feed={ALPACA_DATA_FEED})\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} • Timer: {'ON' if TIMER['enabled'] else 'OFF'} "
        f"(alle {TIMER['poll_minutes']}m, market-hours-only={TIMER['market_hours_only']}, candle_sync={CONFIG.candle_sync})\n"
        f"LastStatus: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper) TIF={CONFIG.alpaca_tif}\n"
        f"Account: {json.dumps(acc) if acc else 'n/a'}"
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
            "/set data_provider=alpaca\n"
            "/set poll_minutes=10 market_hours_only=true candle_sync=true\n"
            "/set alpaca_tif=day trade_enabled=false"
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
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für Backtest ({sym})."); return
    f = compute_signals_for_frame(df, CONFIG)
    met = backtest_simple(f, CONFIG)
    cagr = (met["equity"]**(365/max(1,days)) - 1)
    await update.message.reply_text(
        f"📈 Backtest {days}d  Trades={met['trades']}/{met['exits']}  "
        f"Win={met['win']*100:.1f}%  PF={met['pf']:.2f}  CAGR~{cagr*100:.2f}%\n"
        f"ℹ️ Hinweis: Kein Slippage/Fees; EoB-Logik – eher optimistisch."
    )

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG)
    last = f.iloc[-1]
    txt = (
        f"🔎 {sym} {CONFIG.interval}\n"
        f"rsi={last['rsi']:.2f} (rising={bool(last['rsi_rising'])})  "
        f"efi={last['efi']:.2f} (rising={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}"
    ).strip()
    await update.message.reply_text(txt)

async def cmd_ind(update, context):
    await cmd_sig(update, context)

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    n = 300
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG).tail(n)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(f.index, f["close"], label="Close")
    ax.set_title(f"{sym} {CONFIG.interval} Close")
    ax.grid(True); ax.legend(loc="best")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(f.index, f["rsi"], label="RSI (Wilder)")
    ax2.axhline(CONFIG.rsiLow, linestyle="--"); ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.grid(True); ax2.legend(loc="best")
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(f.index, f["efi"], label="EFI")
    ax3.grid(True); ax3.legend(loc="best")

    chat_id = str(update.effective_chat.id)
    await send_png(chat_id, fig,  f"{sym}_{CONFIG.interval}_close.png", "📈 Close")
    await send_png(chat_id, fig2, f"{sym}_{CONFIG.interval}_rsi.png",   "📈 RSI")
    await send_png(chat_id, fig3, f"{sym}_{CONFIG.interval}_efi.png",   "📈 EFI")

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    args = [a.lower() for a in (context.args or [])]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
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
        "market_hours_only": TIMER["market_hours_only"],
        "candle_sync": CONFIG.candle_sync
    }, indent=2))

async def cmd_timerrunnow(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    now = datetime.now(timezone.utc)
    TIMER["last_run"] = now.isoformat()
    if CONFIG.candle_sync:
        due = next_tv_aligned_due(now + timedelta(seconds=1), CONFIG.interval, TIMER["poll_minutes"])
    else:
        due = now + timedelta(minutes=TIMER["poll_minutes"])
    TIMER["next_due"] = due.isoformat()
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def cmd_syncdebug(update, context):
    n = 12
    if context.args:
        try:
            n = max(1, int(context.args[0]))
        except:
            pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten ({sym} {CONFIG.interval}).")
        return

    f = compute_signals_for_frame(df, CONFIG).tail(n)
    if not isinstance(f.index, pd.DatetimeIndex):
        f.index = pd.to_datetime(f.index, utc=True)
    else:
        try:
            f.index = f.index.tz_convert("UTC") if f.index.tz is not None else f.index.tz_localize("UTC")
        except Exception:
            pass

    lines=[]; ok_cnt=0
    for ts, row in f.iterrows():
        ts_iso = ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        ok = _is_tv_close(ts, CONFIG.interval)
        if ok: ok_cnt += 1
        lines.append(f"{ts_iso} | rsi={row['rsi']:.2f} efi={row['efi']:.2f} | TV-edge={'OK' if ok else 'NO'}")
    pct = (ok_cnt/len(f))*100.0
    header = (
        f"🧭 TV-Sync-Debug  {sym}  {CONFIG.interval}\n"
        f"Quelle: {note.get('provider','?')} ({note.get('detail','')})\n"
        f"TV-Edge OK: {ok_cnt}/{len(f)} = {pct:.1f}%\n"
        f"— letzte {len(f)} Bars —"
    )
    msg = header + "\n" + "\n".join(lines)
    if len(msg) > 3500:
        await send_document_bytes(str(update.effective_chat.id),
                                  ("\n".join(lines)).encode("utf-8"),
                                  f"{sym}_{CONFIG.interval}_syncdebug.txt",
                                  caption=header)
    else:
        await update.message.reply_text(msg)

async def cmd_wf(update, context):
    # /wf [is_days oos_days]
    is_days, oos_days = 120, 30
    if context.args:
        try:
            if len(context.args)>=1: is_days = max(10, int(context.args[0]))
            if len(context.args)>=2: oos_days = max(5,  int(context.args[1]))
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return

    # kleines Grid (anpassbar)
    lo_grid  = [0.0, 30.0, 40.0, 50.0]
    hi_grid  = [60.0, 65.0, 68.0, 70.0]
    ex_grid  = [40.0, 45.0, 48.0, 50.0]
    grid = [(lo,hi,ex) for lo in lo_grid for hi in hi_grid if lo<hi for ex in ex_grid]

    res = walk_forward_oos(df, CONFIG, is_days, oos_days, grid)
    if not res["blocks"]:
        await update.message.reply_text("⚠️ Zu wenig Daten für Walk-Forward.")
        return

    lines=["📐 Walk-Forward/OOS Ergebnis"]
    for i,b in enumerate(res["blocks"],1):
        is_m = b["is_metrics"]; oos_m = b["oos_metrics"]; par = b["best_params"]
        lines.append(
            f"Block {i}: IS {par} -> OOS "
            f"trades={oos_m['trades']}, win={oos_m['win']*100:.1f}%, pf={oos_m['pf']:.2f}, eq={oos_m['equity']:.3f}"
        )
    txt = "\n".join(lines)
    await update.message.reply_text(txt if len(txt)<3500 else txt[:3490]+"…")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= FastAPI lifespan (PTB polling & timer) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED, TIMER_TASK
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Handlers registrieren
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
        tg_app.add_handler(CommandHandler("syncdebug",    cmd_syncdebug))
        tg_app.add_handler(CommandHandler("wf",           cmd_wf))
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
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED=True
                    print("✅ Telegram polling läuft")
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. retry in {delay}s")
                    await asyncio.sleep(delay); delay=min(delay*2,60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        # Timer starten, falls aktiviert
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
app = FastAPI(title="TQQQ Strategy + Telegram (RSI/EFI + WF)", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": f"{CONFIG.data_provider} (feed={ALPACA_DATA_FEED})",
        "live": CONFIG.live_enabled,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "poll_minutes": TIMER["poll_minutes"],
            "next_due": TIMER["next_due"],
            "candle_sync": CONFIG.candle_sync
        },
        "trade_enabled": CONFIG.trade_enabled
    }

@app.get("/tick")
async def tick():
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=False)
    now = datetime.now(timezone.utc)
    TIMER["last_run"]=now.isoformat()
    if CONFIG.candle_sync:
        due = next_tv_aligned_due(now + timedelta(seconds=1), CONFIG.interval, TIMER["poll_minutes"])
    else:
        due = now + timedelta(minutes=TIMER["poll_minutes"])
    TIMER["next_due"] = due.isoformat()
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
        "ALPACA_DATA_FEED": os.getenv("ALPACA_DATA_FEED", ""),
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
