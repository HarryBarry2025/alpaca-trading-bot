# alpaca_trading_bot.py  — V5.1.1
import os, io, json, time, math, asyncio, traceback, warnings
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

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

# ========= ENV & Paths =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "iex").lower()  # 'iex' oder 'sip'

ENV_ENABLE_TRADE       = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER       = os.getenv("ENABLE_TIMER", "true").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES       = int(os.getenv("POLL_MINUTES", "60"))  # 60 passend für 1h-TF
ENV_MARKET_HOURS_ONLY  = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDT_FILE = DATA_DIR / "pdt_trades.json"

# ========= Helpers =========
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    # Engine
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"                # '1h' empfohlen, wird aus 1m aufgebaut
    lookback_days: int = 365
    drop_last_incomplete: bool = True   # keine angefangenen Bars

    # Indikatoren (TV-kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0                  # explizit 0 als Default
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk & Exits
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Live Scheduler
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = True
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY

    # Data Provider
    data_provider: str = "alpaca"        # "alpaca" (default), "yahoo", "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 1.5
    allow_stooq_fallback: bool = True

    # Trading Toggle & Sizer
    trade_enabled: bool = ENV_ENABLE_TRADE
    sizing_mode: str = "shares"          # 'shares' | 'percent_equity' | 'notional_usd' | 'risk'
    sizing_value: float = 1.0            # je nach Modus interpretiert
    max_position_pct: float = 100.0      # Kappung pro-Trade

    # Slippage/Fees (Backtest)
    bt_slippage_bps: float = 0.0
    bt_fees_per_trade: float = 0.0

    # Timer-Anchor (TV Sync)
    sync_hourly_at_minute: int = 30      # 1h Bars enden bei :30 UTC

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}  # {sym: {"size":int,"avg":float,"entry_time":str}}
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= Indicators (TV-kompatibel) =========
def rsi_wilder(s: pd.Series, length: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / length
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def efi_tv(close: pd.Series, vol: pd.Series, length: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return raw.ewm(span=length, adjust=False).mean()

# ========= US Market Hours + Holidays =========
US_HOLIDAYS = {
    # 2024
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27",
    "2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25",
    # 2025
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
}

def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or now_utc()
    d = now.strftime("%Y-%m-%d")
    if d in US_HOLIDAYS:
        return False
    if now.weekday() >= 5:  # Sa/So
        return False
    hhmm = now.hour * 60 + now.minute
    # 13:30–20:00 UTC (RTH)
    return 13*60 + 30 <= hhmm <= 20*60

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

def fetch_alpaca_minute(symbol: str, days: int, include_extended: bool=False) -> pd.DataFrame:
    """ Minutebars via Alpaca Market Data v2 """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] library not available:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing api keys")
        return pd.DataFrame()

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    end   = now_utc()
    start = end - timedelta(days=max(days, 60))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start, end=end,
        feed=APCA_DATA_FEED if APCA_DATA_FEED in ("iex","sip") else "iex",
        limit=50000,
        adjustment=None
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] minute empty")
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
        # Filter RTH falls nötig – ABER: wir resamplen später und droppen Offhours aus RSI/EFI, nicht hier
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch minute failed:", e)
        return pd.DataFrame()

def fetch_yahoo_intraday(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """ Yahoo fallback (direktes Intervall) """
    try:
        df = yf.download(
            tickers=symbol, interval=interval, period=f"{min(lookback_days, 60)}d",
            auto_adjust=False, progress=False, prepost=False, threads=False
        )
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns=str.lower)
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            try:
                df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            except Exception:
                pass
        df["time"] = df.index
        return df
    except Exception as e:
        print("[yahoo] intraday failed:", e)
        return pd.DataFrame()

def resample_to_tv_hourly(df_1m: pd.DataFrame, drop_last_incomplete=True,
                          only_rth=True, anchor_minute=30) -> pd.DataFrame:
    """
    Baue 1h-Bars aus 1-Minuten-Bars mit :30 Anchor, nur RTH (13:30–20:00 UTC).
    """
    if df_1m.empty: return df_1m

    # optional: nur RTH Minuten behalten (damit RSI/EFI nur RTH-Minuten sehen)
    if only_rth:
        i = df_1m.index
        hhmm = i.hour * 60 + i.minute
        mask_day = (i.weekday < 5)
        mask_time = (hhmm >= 13*60+30) & (hhmm <= 20*60)  # inkl. 20:00
        df_1m = df_1m[mask_day & mask_time]

    # Anker: erste Bar endet z. B. 13:30–14:30→close 14:30 (UTC)
    # Wir nutzen Grouper with base minute offset (=anchor).
    # In Pandas 2.0+ kann 'offset' genutzt werden:
    offset = f"{anchor_minute}min"  # '30min'
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    r = df_1m.resample("60T", origin="epoch", offset=offset).agg(ohlc).dropna(how="any")
    r.index = r.index.tz_localize("UTC") if r.index.tz is None else r.index.tz_convert("UTC")

    if drop_last_incomplete:
        # wir verwerfen die letzte Bar, falls ihr Endzeitpunkt nach 'jetzt' liegt (angefangene Stunde)
        last_idx = r.index.max()
        if last_idx and last_idx > now_utc():
            r = r.iloc[:-1]

    r["time"] = r.index
    return r

def fetch_ohlcv_tv_hourly(symbol: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    Bevorzugt Alpaca 1m → TV-1h; Fallback Yahoo 1h; Fallback Stooq 1d.
    """
    note = {"provider":"","detail":""}

    # 1) Alpaca 1m → 1h
    if CONFIG.data_provider.lower() == "alpaca":
        m = fetch_alpaca_minute(symbol, lookback_days, include_extended=False)
        if not m.empty:
            h = resample_to_tv_hourly(
                m, drop_last_incomplete=CONFIG.drop_last_incomplete,
                only_rth=True, anchor_minute=CONFIG.sync_hourly_at_minute
            )
            if not h.empty:
                note.update(provider=f"Alpaca ({APCA_DATA_FEED})", detail="1h (from 1m, RTH, :30 anchor)")
                return h, note
        # Fallback:
        note.update(provider="Alpaca→Yahoo", detail="fallback 1h")

    # 2) Yahoo direkt 1h
    y = fetch_yahoo_intraday(symbol, "60m", lookback_days)
    if not y.empty:
        # Yahoo 1h ist meist :30 anchor RTH-nah, dennoch RTH-Filter + drop last
        y = resample_to_tv_hourly(
            y, drop_last_incomplete=CONFIG.drop_last_incomplete,
            only_rth=True, anchor_minute=CONFIG.sync_hourly_at_minute
        )
        if not y.empty:
            note.update(provider="Yahoo", detail="1h (resampled to RTH)")
            return y, note

    # 3) Stooq daily (nur Chart/BT notfalls)
    if CONFIG.allow_stooq_fallback:
        s = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not s.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return s, note

    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

def fetch_ohlcv_with_note(symbol: str) -> Tuple[pd.DataFrame, Dict[str,str]]:
    if CONFIG.interval.lower() in ("1h","60m"):
        return fetch_ohlcv_tv_hourly(symbol, CONFIG.lookback_days)
    # als Alternative 1d (falls umgestellt)
    # hier könntest du noch 1d-RTH implementieren; fürs Projekt bleibt Fokus 1h
    return fetch_ohlcv_tv_hourly(symbol, CONFIG.lookback_days)

# ========= Features & Signals =========
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_wilder(out["close"], CONFIG.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], CONFIG.efiLen)
    out["rsi_rising"] = out["rsi"] > out["rsi"].shift(1)
    out["rsi_falling"] = out["rsi"] < out["rsi"].shift(1)
    out["efi_rising"] = out["efi"] > out["efi"].shift(1)
    out["entry_cond"] = (out["rsi"] > CONFIG.rsiLow) & (out["rsi"] < CONFIG.rsiHigh) & out["rsi_rising"] & out["efi_rising"]
    out["exit_cond"]  = (out["rsi"] < CONFIG.rsiExit) & out["rsi_falling"]
    return out

# ========= PDT Persistenz & Logik =========
def load_pdt_state() -> Dict[str, Any]:
    if PDT_FILE.exists():
        try:
            return json.loads(PDT_FILE.read_text())
        except Exception:
            pass
    return {"trades": []}  # list of ISO timestamps for day-trades (flat in same day)

def save_pdt_state(state: Dict[str, Any]):
    try:
        PDT_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print("save_pdt_state error:", e)

def purge_old_pdt(state: Dict[str, Any]):
    """ Keep only last 5 business days """
    now = now_utc().date()
    # naive: keep last 10 calendar days to be safe
    keep_after = now - timedelta(days=10)
    new = []
    for t in state.get("trades", []):
        try:
            d = datetime.fromisoformat(t).date()
            if d >= keep_after:
                new.append(t)
        except:
            pass
    state["trades"] = new

def pdt_is_blocked(state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Hard Block nach 3 Daytrades in 5 Handelstagen.
    """
    purge_old_pdt(state)
    # Zähle Trades innerhalb der letzen 5 Handelstage (grob: 7 Kalendertage)
    now = now_utc()
    lower = now - timedelta(days=7)
    recent = [t for t in state.get("trades", []) if datetime.fromisoformat(t) >= lower]
    count = len(recent)
    blocked = count >= 3
    return blocked, {"recent_count": count, "window_days": 7}

def log_pdt_trade():
    st = load_pdt_state()
    st.setdefault("trades", []).append(now_utc().isoformat())
    purge_old_pdt(st)
    save_pdt_state(st)

# ========= Trader Sizer =========
def get_account_equity() -> Optional[float]:
    try:
        from alpaca.trading.client import TradingClient
        tc = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
        acc = tc.get_account()
        return float(acc.equity)
    except Exception:
        return None

def compute_order_qty(symbol: str, price: float) -> int:
    mode = CONFIG.sizing_mode
    val  = CONFIG.sizing_value
    max_pct = max(1e-9, min(100.0, CONFIG.max_position_pct))

    qty = 0.0
    if mode == "shares":
        qty = float(val)
    elif mode == "percent_equity":
        eq = get_account_equity() or 10000.0
        target_notional = eq * (val/100.0)
        qty = target_notional / max(1e-9, price)
    elif mode == "notional_usd":
        qty = float(val) / max(1e-9, price)
    elif mode == "risk":
        # val = Risiko in USD; Stop = slPerc%
        risk_per_share = price * (CONFIG.slPerc/100.0)
        if risk_per_share > 0:
            qty = float(val) / risk_per_share
        else:
            qty = 0.0
    else:
        qty = 1.0

    # Max Position Cap je Trade
    eq = get_account_equity() or 10000.0
    max_notional = eq * (max_pct/100.0)
    qty = min(qty, math.floor(max_notional / max(1e-9, price)))
    return max(0, int(qty))

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

# ========= Single-step strategy (per symbol) =========
def bar_logic_last(df: pd.DataFrame, sym: str) -> Dict[str,Any]:
    if df.empty or len(df) < max(CONFIG.rsiLen, CONFIG.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data","symbol":sym}

    f = build_features(df)
    last, prev = f.iloc[-1], f.iloc[-2]

    price_close = float(last["close"])
    price_open  = float(last["open"])
    bar_high    = float(last["high"])
    bar_low     = float(last["low"])
    ts = last["time"]

    entry_cond = bool(last["entry_cond"])
    exit_cond  = bool(last["exit_cond"])

    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]; entry_time = pos["entry_time"]

    def sl(p): return p*(1-CONFIG.slPerc/100.0)
    def tp(p): return p*(1+CONFIG.tpPerc/100.0)

    bars_in_trade=0
    if entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if size==0:
        if entry_cond:
            # sizing
            qty = compute_order_qty(sym, price_open)
            if qty <= 0:
                return {"action":"none","symbol":sym,"reason":"qty=0",
                        "rsi":float(last["rsi"]), "efi":float(last["efi"])}
            return {"action":"buy","symbol":sym,"qty":qty,"px":price_open,"time":str(ts),
                    "sl":sl(price_open),"tp":tp(price_open),"reason":"rule_entry",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        return {"action":"none","symbol":sym,"reason":"flat_no_entry",
                "rsi":float(last["rsi"]), "efi":float(last["efi"])}
    else:
        same_bar_ok = CONFIG.allowSameBarExit or (bars_in_trade>0)
        cooldown_ok = (bars_in_trade>=CONFIG.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(avg); cur_tp = tp(avg)

        # Intrabar SL/TP: High/Low-Check
        hit_sl = bar_low  <= cur_sl
        hit_tp = bar_high >= cur_tp

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
    if not text or not text.strip(): text = "ℹ️ (leer)"
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

# ========= One run =========
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    return f"📡 Quelle: {prov} ({det})"

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    # PDT Hard-Block prüfen (nur auf neue Entries anwenden)
    pdt_state = load_pdt_state()
    pdt_blocked, pdt_info = pdt_is_blocked(pdt_state)

    df, note = fetch_ohlcv_with_note(sym)
    if df.empty:
        msg = f"❌ Keine Daten für {sym}. {note}"
        if send_signals and CHAT_ID:
            await send_text(CHAT_ID, msg)
        return {"ok":False,"msg":msg}

    act = bar_logic_last(df, sym)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    # Info & Indikator-Status
    if send_signals and CHAT_ID:
        await send_text(CHAT_ID, f"ℹ️ {sym} {CONFIG.interval} rsi={act.get('rsi',np.nan):.2f} efi={act.get('efi',np.nan):.2f} • {act['reason']}")
        note_msg = friendly_note(note)
        if note_msg:
            await send_text(CHAT_ID, note_msg)

    # Trading (Paper) – nur wenn frei, Markt ggf. offen, und keine PDT-Sperre auf Entries
    if CONFIG.trade_enabled and act["action"] in ("buy","sell"):
        if CONFIG.market_hours_only and not is_market_open_now():
            if CHAT_ID: await send_text(CHAT_ID, "⛔ Markt geschlossen – kein Trade ausgeführt.")
        else:
            side = "buy" if act["action"]=="buy" else "sell"

            # PDT-Hard-Stop: blockiere nur neue LONG-Entries
            if side=="buy" and pdt_blocked:
                if CHAT_ID:
                    await send_text(CHAT_ID, f"⛔ PDT-Block aktiv (Trades in 5d: {pdt_info['recent_count']}). Kein neuer Entry.")
            else:
                if side=="buy":
                    qty = act["qty"]
                else:
                    qty = STATE.positions.get(sym, {"size":0})["size"] or act.get("qty", 0)

                if qty > 0:
                    info = await place_market_order(sym, int(qty), side, "day")
                    if CHAT_ID:
                        await send_text(CHAT_ID, f"🛒 {side.upper()} {sym} x{int(qty)} @ {act['px']:.4f} • {info}")

    # Lokale Sim-Position (für Anzeige /status)
    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    if act["action"]=="buy" and pos["size"]==0:
        # Bei Entry: PDT-Tracking vorbereiten (wird beim Exit geloggt)
        STATE.positions[sym] = {"size":act["qty"],"avg":act["px"],"entry_time":act["time"]}
        if CHAT_ID and send_signals:
            await send_text(CHAT_ID, f"🟢 LONG (sim) {sym} @ {act['px']:.4f} | SL={act.get('sl',np.nan):.4f} TP={act.get('tp',np.nan):.4f}")
    elif act["action"]=="sell" and pos["size"]>0:
        pnl = (act["px"] - pos["avg"]) / pos["avg"]
        # PDT-Logik: Entry & Exit am selben Kalendertag? -> Count als Daytrade
        try:
            entry_day = datetime.fromisoformat(pos["entry_time"]).date()
            exit_day  = datetime.fromisoformat(act["time"]).date()
            if entry_day == exit_day:
                log_pdt_trade()
        except Exception:
            pass

        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if CHAT_ID and send_signals:
            await send_text(CHAT_ID, f"🔴 EXIT (sim) {sym} @ {act['px']:.4f} • {act['reason']} • PnL={pnl*100:.2f}%")

    return {"ok":True,"act":act}

# ========= Background Timer =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}

def compute_next_due(now: datetime) -> datetime:
    """
    Für 1h-TF: auf die nächste :30 UTC runden.
    """
    if CONFIG.interval.lower() in ("1h","60m"):
        minute_anchor = CONFIG.sync_hourly_at_minute  # 30
        # nächste volle Stunde + :30
        nxt = now.replace(minute=minute_anchor, second=0, microsecond=0)
        if now.minute >= minute_anchor:
            nxt = nxt + timedelta(hours=1)
        # stelle sicher, dass nxt in der Zukunft liegt
        if nxt <= now:
            nxt = nxt + timedelta(hours=1)
        return nxt
    else:
        # generisch: poll_minutes
        return now + timedelta(minutes=TIMER["poll_minutes"])

async def timer_loop():
    TIMER["running"] = True
    try:
        # Beim Start direkt auf den nächsten Due setzen
        now = now_utc()
        TIMER["next_due"] = to_iso(compute_next_due(now))
        while TIMER["enabled"]:
            await asyncio.sleep(1)
            now = now_utc()

            # Market Hours Gate
            if TIMER["market_hours_only"] and not is_market_open_now(now):
                # Während Off-Hours schieben wir den next_due auf nächste Handelszeit
                # aber lassen den loop laufen
                await asyncio.sleep(10)
                continue

            # fälliger Tick?
            nd = TIMER["next_due"]
            if nd:
                due = datetime.fromisoformat(nd)
                if now < due:
                    await asyncio.sleep(1)
                    continue

            # Run
            for sym in CONFIG.symbols:
                await run_once_for_symbol(sym, send_signals=True)

            TIMER["last_run"] = to_iso(now)
            nxt = compute_next_due(now)
            TIMER["next_due"] = to_iso(nxt)
    finally:
        TIMER["running"] = False

TIMER_TASK: Optional[asyncio.Task] = None

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg, /set key=value …, /run, /live on|off, /bt [tage]\n"
        "/wf [is_days oos_days]\n"
        "/sig, /ind, /plot\n"
        "/dump [csv [N]], /dumpcsv [N]\n"
        "/trade on|off, /pos, /account\n"
        "/timer on|off, /timerstatus, /timerrunnow\n"
        "/market on|off (RTH-Filter)"
    )

async def cmd_status(update, context):
    pos_lines=[]
    for s,p in STATE.positions.items():
        pos_lines.append(f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}")
    pos_txt = "\n".join(pos_lines) if pos_lines else "keine (sim)"

    pdt = load_pdt_state()
    blocked, info = pdt_is_blocked(pdt)

    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider} (feed={APCA_DATA_FEED})\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}\n"
        f"Timer: {'ON' if TIMER['enabled'] else 'OFF'} (next_due={TIMER['next_due']})\n"
        f"RTH only: {TIMER['market_hours_only']}\n"
        f"Sizer: mode={CONFIG.sizing_mode} value={CONFIG.sizing_value} max%={CONFIG.max_position_pct}\n"
        f"LastStatus: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper)\n"
        f"PDT: blocked={blocked} recent={info['recent_count']} in {info['window_days']}d"
    )

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

def set_from_kv(kv: str) -> str:
    k,v = kv.split("=",1); k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade",
               "sizing":"sizing_mode", "size":"sizing_value", "maxpct":"max_position_pct"}
    k = mapping.get(k,k)
    if not hasattr(CONFIG, k): return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):   setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):  setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):setattr(CONFIG, k, float(v))
    elif isinstance(cur, list): setattr(CONFIG, k, [x.strip() for x in v.split(",") if x.strip()])
    else:                       setattr(CONFIG, k, v)
    # sync mit TIMER
    if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "z.B. /set rsiLow=0 rsiHigh=68 rsiExit=48 sl=2 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set symbols=TQQQ,QQQ\n"
            "/set data_provider=alpaca\n"
            "/set sizing_mode=percent_equity sizing_value=100 max_position_pct=100\n"
            "/set poll_minutes=60 market_hours_only=true"
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
    df, _ = fetch_ohlcv_with_note(sym)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für Backtest ({sym})."); return
    f = build_features(df)

    # Backtest mit Intrabar SL/TP (High/Low), EoB-Entry/Exit
    eq=1.0; pos=0; avg=0.0; R=[]; entries=exits=0
    slip = CONFIG.bt_slippage_bps/10000.0
    fee  = CONFIG.bt_fees_per_trade

    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry = bool(row["entry_cond"])
        exitc = bool(row["exit_cond"])
        if pos==0 and entry:
            # Preis inkl. slippage
            px = float(row["open"]) * (1+slip)
            pos=1; avg=px; entries+=1
            eq -= fee/ max(1.0, eq)
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            low=float(row["low"]); high=float(row["high"])
            stop = low<=sl; take = high>=tp
            do_exit = exitc or stop or take
            if do_exit:
                px = sl if stop else tp if take else float(row["open"])*(1-slip)
                r = (px-avg)/avg
                eq*= (1+r)
                eq -= fee/ max(1.0, eq)
                R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean(); pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(f"📈 Backtest {days}d  Trades={entries}/{exits}  Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\nℹ️ Slippage={CONFIG.bt_slippage_bps} bps, Fees={CONFIG.bt_fees_per_trade}")
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_wf(update, context):
    # Walk-Forward Quick-Variante (ein Chunk)
    is_days, oos_days = 120, 30
    if context.args and len(context.args)>=2:
        try:
            is_days = int(context.args[0]); oos_days = int(context.args[1])
        except: pass
    sym = CONFIG.symbols[0]
    df, _ = fetch_ohlcv_with_note(sym)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten für WF."); return

    # Split
    end = df.index.max()
    is_start = end - timedelta(days=is_days+oos_days)
    oos_start = end - timedelta(days=oos_days)
    IS = df[df.index>=is_start]; IS = IS[IS.index<oos_start]
    OOS= df[df.index>=oos_start]
    if IS.empty or OOS.empty:
        await update.message.reply_text("❌ Zu wenig Daten für WF."); return

    # Simple Grid über RSI-Zonen
    best=None; best_eq=-1
    for low in (0, 45, 50, 52):
        for high in (60, 65, 68, 70):
            if high <= low: continue
            cfg_snapshot = CONFIG.copy()
            CONFIG.rsiLow = float(low); CONFIG.rsiHigh = float(high)

            def backtest(frame: pd.DataFrame) -> float:
                f = build_features(frame)
                eq=1.0; pos=0; avg=0.0
                for i in range(2,len(f)):
                    row=f.iloc[i]
                    if pos==0 and bool(row["entry_cond"]):
                        pos=1; avg=float(row["open"])
                    elif pos==1:
                        sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
                        low=float(row["low"]); high=float(row["high"]); exitc=bool(row["exit_cond"])
                        stop = low<=sl; take = high>=tp
                        if exitc or stop or take:
                            px = sl if stop else tp if take else float(row["open"])
                            eq*= (1+(px-avg)/avg); pos=0; avg=0.0
                return eq

            eq_is = backtest(IS)
            if eq_is > best_eq:
                best_eq=eq_is; best={"low":low,"high":high,"eq_is":eq_is}
            CONFIG = cfg_snapshot  # restore

    # OOS mit Best-Params
    if best is None:
        await update.message.reply_text("❌ WF: kein best gefunden."); return
    CONFIG.rsiLow=float(best["low"]); CONFIG.rsiHigh=float(best["high"])
    fO=build_features(OOS)
    eq=1.0; pos=0; avg=0.0
    for i in range(2,len(fO)):
        row=fO.iloc[i]
        if pos==0 and bool(row["entry_cond"]):
            pos=1; avg=float(row["open"])
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            low=float(row["low"]); high=float(row["high"]); exitc=bool(row["exit_cond"])
            stop = low<=sl; take = high>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                eq*= (1+(px-avg)/avg); pos=0; avg=0.0

    await update.message.reply_text(
        f"🧪 WF (IS/OOS)  IS={is_days}d OOS={oos_days}d\n"
        f"Best RSI-Zone: [{best['low']}, {best['high']}]  IS_Equity={best['eq_is']:.3f}  OOS_Equity={eq:.3f}"
    )

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df)
    last = f.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} {CONFIG.interval}\n"
        f"rsi={last['rsi']:.2f} (↑={bool(last['rsi_rising'])})  "
        f"efi={last['efi']:.2f} (↑={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}\n"
        f"{friendly_note(note)}"
    )

async def cmd_ind(update, context):
    await cmd_sig(update, context)

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    n = 300
    df, _ = fetch_ohlcv_with_note(sym)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df).tail(n)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(f.index, f["close"], label="Close")
    ax.set_title(f"{sym} {CONFIG.interval} Close"); ax.grid(True); ax.legend(loc="best")

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

def build_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = build_features(df)
    cols = ["open","high","low","close","volume","rsi","efi","rsi_rising","efi_rising","entry_cond","exit_cond","time"]
    return f[cols]

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    args = [a.lower() for a in (context.args or [])]
    df, note = fetch_ohlcv_with_note(sym)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für {sym}."); return

    if args and args[0]=="csv":
        n = 300
        if len(args)>=2:
            try: n=max(1,int(args[1]))
            except: pass
        exp = build_export_frame(df).tail(n)
        csv_bytes = exp.to_csv(index=True).encode("utf-8")
        await send_document_bytes(str(update.effective_chat.id), csv_bytes,
                                  f"{sym}_{CONFIG.interval}_indicators_{n}.csv",
                                  caption=f"🧾 CSV (OHLCV + RSI/EFI + Entry/Exit) {sym} {CONFIG.interval} n={n}")
        return

    f = build_features(df)
    last = f.iloc[-1]
    payload = {
        "symbol": sym, "interval": CONFIG.interval,
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
        "market_hours_only": TIMER["market_hours_only"]
    }, indent=2))

async def cmd_timerrunnow(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    now = now_utc()
    TIMER["last_run"] = to_iso(now)
    TIMER["next_due"] = to_iso(compute_next_due(now))
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def cmd_market(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /market on|off (RTH-Filter)"); return
    on = context.args[0].lower() in ("on","1","true","start")
    TIMER["market_hours_only"] = on
    CONFIG.market_hours_only = on
    await update.message.reply_text(f"RTH-Filter = {'ON' if on else 'OFF'}")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= FastAPI lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, POLLING_STARTED, TIMER_TASK
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()

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
        tg_app.add_handler(CommandHandler("timer",        cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("timerrunnow",  cmd_timerrunnow))
        tg_app.add_handler(CommandHandler("market",       cmd_market))
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

        # Timer starten
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
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

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
        "DATA_DIR": str(DATA_DIR),
        "PDT_FILE_exists": PDT_FILE.exists()
    }

@app.get("/tgstatus")
def tgstatus():
    return {"polling_started": POLLING_STARTED}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
