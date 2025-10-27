# alpaca_trading_bot.py  — V5.1.3 (stable baseline)
# Start-Cmd (Render): uvicorn alpaca_trading_bot:app --host 0.0.0.0 --port ${PORT:-8000}

import os, io, json, time, math, asyncio, traceback, warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (python-telegram-bot v20.7)
from telegram import Update, InputFile
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict

warnings.filterwarnings("ignore", category=UserWarning)

# ========= ENV & Paths =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID") or None

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = (os.getenv("APCA_DATA_FEED") or "sip").lower()  # 'sip' empfohlen (Abo nötig), sonst 'iex'

ENV_ENABLE_TRADE      = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER      = os.getenv("ENABLE_TIMER", "false").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES      = int(os.getenv("POLL_MINUTES", "60"))
ENV_MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

# persistente Ablage NUR in /tmp (Render erlaubt), nicht /mnt/data
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDT_JSON = DATA_DIR / "pdt_trades.json"

# ========= Config & State =========
class StratConfig(BaseModel):
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"
    lookback_days: int = 365

    # TV-kompatible Inputs
    rsiLen: int = 12
    rsiLow: float = 0.0          # explizit default 0 erlaubt
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Sizing (einfach & robust)
    sizing_mode: str = "shares"      # shares | percent_equity | notional_usd | risk
    sizing_value: float = 1.0        # je nach Modus interpretiert
    max_position_pct: float = 100.0  # 0..100

    # Engine / Timer
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = False
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY
    candle_align_30: bool = True     # 1h-Bars auf :30 UTC geankert (TradingView-kompatibel)
    drop_last_incomplete: bool = True

    # Data Provider
    data_provider: str = "alpaca"    # primär Alpaca (m1 -> resample)
    yahoo_retries: int = 2
    yahoo_backoff_sec: float = 1.5
    allow_stooq_fallback: bool = True

    # Trading Toggle
    trade_enabled: bool = ENV_ENABLE_TRADE

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}  # {sym: {"size":int,"avg":float,"entry_time":str|None}}
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= PDT Persistenz =========
def load_pdt() -> Dict[str, Any]:
    if PDT_JSON.exists():
        try:
            return json.loads(PDT_JSON.read_text())
        except:
            pass
    return {"trades": []}  # list of {"date":"YYYY-MM-DD","symbol":"TQQQ","buy_time": "...", "sell_time":"..."}

def save_pdt(d: Dict[str, Any]):
    try:
        PDT_JSON.write_text(json.dumps(d, indent=2))
    except Exception as e:
        print("save_pdt error:", e)

def five_business_days_dates(ref_utc: datetime) -> List[str]:
    # roll back 10 days and take last 5 weekdays unique by date
    ds = []
    cur = ref_utc.date()
    for i in range(10):
        d = (cur - timedelta(days=i))
        if (datetime(d.year,d.month,d.day).weekday() < 5):
            ds.append(str(d))
        if len(ds) >= 5: break
    return list(reversed(ds))

def pdt_count_in_window(now_utc: datetime) -> int:
    d = load_pdt()
    window = set(five_business_days_dates(now_utc))
    cnt = 0
    for t in d.get("trades", []):
        d0 = t.get("date")
        if d0 in window:
            cnt += 1
    return cnt

def pdt_record_daytrade(sym: str, buy_time: str, sell_time: str):
    try:
        bt = pd.to_datetime(buy_time)
        st = pd.to_datetime(sell_time)
        if bt.date() == st.date():  # nur intraday
            d = load_pdt()
            d["trades"].append({
                "date": str(bt.date()),
                "symbol": sym,
                "buy_time": buy_time,
                "sell_time": sell_time
            })
            # optional kürzen
            if len(d["trades"]) > 2000:
                d["trades"] = d["trades"][-2000:]
            save_pdt(d)
    except Exception as e:
        print("pdt_record_daytrade error:", e)

def pdt_block_allow(now_utc: datetime, hard_stop: bool = True) -> Tuple[bool, str]:
    """True=allow, False=block. Alpaca-ähnliche Regel: max 3 Daytrades in 5 Handelstagen."""
    cnt = pdt_count_in_window(now_utc)
    if cnt >= 3:
        return (False, f"PDT hard-stop: {cnt} DTs/5d erreicht")
    return (True, f"PDT okay: {cnt}/3 in 5d")

# ========= Indicators (TV-kompatibel) =========
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi_tv_rma(s: pd.Series, length: int = 14) -> pd.Series:
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

# ========= Market hours (grob) =========
def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or datetime.now(timezone.utc)
    # Mo–Fr
    if now.weekday() >= 5:
        return False
    # Kernhandelszeit 13:30–20:00 UTC (9:30–16:00 ET) — grob
    hhmm = now.hour*60 + now.minute
    return 13*60+30 <= hhmm <= 20*60

# ========= Data: Alpaca m1 + Resample → TV-synced 1h =========
def fetch_alpaca_1m(symbol: str, feed: Optional[str] = None, days: int = 30) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] lib missing:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing keys")
        return pd.DataFrame()

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(days, 10))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start, end=end,
        feed=(feed or APCA_DATA_FEED),
        limit=50000
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.rename(columns=str.lower).sort_index()
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch 1m error:", e)
        return pd.DataFrame()

def tv_resample_1h_from_m1(m1: pd.DataFrame, drop_last_incomplete: bool = True, align_30: bool = True,
                           rth_only: bool = True) -> pd.DataFrame:
    if m1.empty:
        return m1
    df = m1.copy()
    # RTH-Filter: 13:30–20:00 UTC
    if rth_only:
        idx = df.index
        hhmm = idx.hour*60 + idx.minute
        mask = (hhmm >= (13*60+30)) & (hhmm <= (20*60))
        df = df.loc[mask]

    # auf :30 ausrichten (so dass 13:30–14:29 → close @14:30 etc)
    # Trick: verschiebe Index um -30 Min, resample auf volle Stunde, schiebe zurück
    if align_30:
        df = df.tz_convert("UTC")
        df.index = df.index - pd.Timedelta(minutes=30)

    o = df["open"].resample("1H").first()
    h = df["high"].resample("1H").max()
    l = df["low"].resample("1H").min()
    c = df["close"].resample("1H").last()
    v = df["volume"].resample("1H").sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]

    # incomplete letzte Bar droppen?
    if drop_last_incomplete:
        # Wenn letzte (verschobene) 1h-Bar noch nicht „voll“ ist: droppen
        last_ts = out.index[-1]
        # aktuelle verschobene Zeit:
        now_shifted = (datetime.now(timezone.utc) - pd.Timedelta(minutes=30)).replace(second=0, microsecond=0)
        # eine Bar ist „voll“, wenn last_ts < floor(now_shifted, 1h)
        full_cutoff = now_shifted.replace(minute=0)
        if last_ts >= full_cutoff:
            out = out.iloc[:-1]

    # Zeitstempel wieder um +30 min zurück
    if align_30:
        out.index = out.index + pd.Timedelta(minutes=30)

    out = out.dropna(how="any")
    out["time"] = out.index
    return out

def fetch_ohlcv_tv_sync(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Primär: Alpaca m1 → TV-synced 1h bars; Fallback: Yahoo / Stooq."""
    interval = interval.lower()
    if interval != "1h":
        # Für andere TFs: simplere Wege (hier nur 1h TV-synced im Fokus)
        return pd.DataFrame(), {"provider": "(only 1h tv-sync)", "detail": ""}

    # Alpaca m1 holen
    m1 = fetch_alpaca_1m(symbol, APCA_DATA_FEED, days=min(lookback_days, 90))
    if not m1.empty:
        h1 = tv_resample_1h_from_m1(m1,
                                    drop_last_incomplete=CONFIG.drop_last_incomplete,
                                    align_30=CONFIG.candle_align_30,
                                    rth_only=CONFIG.market_hours_only)
        if not h1.empty:
            # lookback
            if lookback_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                h1 = h1.loc[h1.index >= cutoff]
            return h1, {"provider": f"Alpaca ({APCA_DATA_FEED})", "detail": "1m→1h TV-sync"}

    # Fallback: Yahoo 1h (kann Offsets anders setzen)
    try:
        tmp = yf.download(symbol, interval="60m", period=f"{min(lookback_days, 730)}d",
                          auto_adjust=False, progress=False, prepost=not CONFIG.market_hours_only,
                          threads=False)
        if tmp is not None and not tmp.empty:
            df = tmp.rename(columns=str.lower).sort_index()
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df["time"] = df.index
            return df[["open","high","low","close","volume","time"]], {"provider":"Yahoo (60m)","detail":"fallback"}
    except Exception as e:
        pass

    # Stooq daily fallback (als Notnagel)
    try:
        from urllib.parse import quote
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        df = pd.read_csv(url)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["time"] = df.index.tz_localize("UTC")
            return df[["open","high","low","close","volume","time"]], {"provider":"Stooq (1d)","detail":"fallback"}
    except Exception as e:
        pass

    return pd.DataFrame(), {"provider":"(leer)", "detail":"no data"}

# ========= Features & Signals =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv_rma(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    out["rsi_rising"] = out["rsi"] > out["rsi"].shift(1)
    out["rsi_falling"]= out["rsi"] < out["rsi"].shift(1)
    out["efi_rising"] = out["efi"] > out["efi"].shift(1)
    out["entry_cond"] = (out["rsi"] > cfg.rsiLow) & (out["rsi"] < cfg.rsiHigh) & out["rsi_rising"] & out["efi_rising"]
    out["exit_cond"]  = (out["rsi"] < cfg.rsiExit) & out["rsi_falling"]
    return out

def bar_logic_last(df: pd.DataFrame, cfg: StratConfig, sym: str) -> Dict[str,Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data","symbol":sym}

    f = build_features(df, cfg)
    last, prev = f.iloc[-1], f.iloc[-2]
    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = last["time"]

    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]; entry_time = pos["entry_time"]

    def sl(p): return p*(1-cfg.slPerc/100.0)
    def tp(p): return p*(1+cfg.tpPerc/100.0)

    # Bars since entry (EOB approx)
    bars_in_trade=0
    if entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    entry_cond = bool(last["entry_cond"])
    exit_cond  = bool(last["exit_cond"])

    if size==0:
        if entry_cond:
            q = 1  # sizing minimal (kann per /set geändert werden; vgl. trade_size())
            return {"action":"buy","symbol":sym,"qty":q,"px":price_open,"time":str(ts),
                    "sl":sl(price_open),"tp":tp(price_open),"reason":"rule_entry",
                    "rsi":float(last["rsi"]), "efi":float(last["efi"])}
        return {"action":"none","symbol":sym,"reason":"flat_no_entry",
                "rsi":float(last["rsi"]), "efi":float(last["efi"])}

    # already in position
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

async def send_text(chat_id: str, text: str):
    if tg_app is None: return
    if not text or not text.strip(): text="(leer)"
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

# ========= Trading (Alpaca, Paper) =========
def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

def trade_size(sym: str, price: float) -> int:
    mode = CONFIG.sizing_mode.lower()
    val  = CONFIG.sizing_value
    # einfache Sizer (ohne Account-Equity-Abfrage):
    if mode == "shares":
        qty = int(max(1, int(val)))
    elif mode == "notional_usd":
        qty = int(max(1, math.floor(val / max(1e-9, price))))
    elif mode == "percent_equity":
        # ohne Konto-Abfrage: setze Equity=10_000 implizit
        eq = 10_000.0
        notional = eq * (float(val)/100.0)
        qty = int(max(1, math.floor(notional / max(1e-9, price))))
    elif mode == "risk":
        # primitive: val = $-Risiko pro Trade, SL-Distanz = slPerc%
        risk_per_share = price * (CONFIG.slPerc/100.0)
        qty = int(max(1, math.floor(val / max(1e-9, risk_per_share))))
    else:
        qty = 1
    return max(1, qty)

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

# ========= Runner =========
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    return f"📡 Quelle: {prov} ({det})"

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        return {"ok":False, "msg": f"❌ Keine Daten für {sym}. {note}"}

    act = bar_logic_last(df, CONFIG, sym)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    if send_signals and CHAT_ID:
        await send_text(CHAT_ID,
            f"ℹ️ {sym} {CONFIG.interval} rsi={act.get('rsi',np.nan):.2f} efi={act.get('efi',np.nan):.2f} • {act['reason']}")

    # Trading mit PDT-Hard-Stop
    if CONFIG.trade_enabled and act["action"] in ("buy","sell"):
        now = datetime.now(timezone.utc)
        allow, why = pdt_block_allow(now, hard_stop=True)
        if not allow:
            if CHAT_ID: await send_text(CHAT_ID, f"⛔ {why} – Trade blockiert.")
        else:
            # optional market_hours_only beachten
            if CONFIG.market_hours_only and not is_market_open_now(now):
                if CHAT_ID: await send_text(CHAT_ID, "⛔ Markt geschlossen – kein Trade.")
            else:
                side = "buy" if act["action"]=="buy" else "sell"
                px   = float(act["px"])
                qty  = trade_size(sym, px)
                info = await place_market_order(sym, qty, side, "day")
                if CHAT_ID: await send_text(CHAT_ID, f"🛒 {side.upper()} {sym} x{qty} • {info}")

    # Lokale Sim-Position (für Anzeige & PDT-Tracking)
    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    if act["action"]=="buy" and pos["size"]==0:
        STATE.positions[sym] = {"size":act["qty"],"avg":act["px"],"entry_time":act["time"]}
        if send_signals and CHAT_ID:
            await send_text(CHAT_ID, f"🟢 LONG (sim) {sym} @ {act['px']:.4f} | SL={act.get('sl',np.nan):.4f} TP={act.get('tp',np.nan):.4f}")
    elif act["action"]=="sell" and pos["size"]>0:
        pnl = (act["px"] - pos["avg"]) / pos["avg"]
        # PDT-Record (nur wenn am selben Tag)
        try:
            pdt_record_daytrade(sym, pos["entry_time"], act["time"])
        except: pass
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if send_signals and CHAT_ID:
            await send_text(CHAT_ID, f"🔴 EXIT (sim) {sym} @ {act['px']:.4f} • {act['reason']} • PnL={pnl*100:.2f}%")

    return {"ok":True,"act":act}

# ========= Timer (mit 1h-:30 Sync) =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,  # typisch 60
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}

def next_half_hour_anchor(now_utc: datetime) -> datetime:
    # nächste :30:00 oder :30 + n*60
    minute = now_utc.minute
    base = now_utc.replace(second=0, microsecond=0)
    # Wenn genau :30, sofort jetzt (aber +60m weiterplanen)
    if minute == 30:
        return base
    # sonst zu nächster :30 vorrücken
    if minute < 30:
        return base.replace(minute=30)
    else:
        # zur nächsten vollen Stunde + :30
        return (base + timedelta(hours=1)).replace(minute=30)

async def timer_loop():
    TIMER["running"] = True
    try:
        # Initial next_due so setzen, dass 1h-bars auf :30 anstoßen
        now = datetime.now(timezone.utc)
        TIMER["next_due"] = next_half_hour_anchor(now).isoformat()

        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)

            # Market hours Filter (sofort weiter schlafen, aber next_due NICHT nullen)
            if TIMER["market_hours_only"] and not is_market_open_now(now):
                await asyncio.sleep(15)
                continue

            # due?
            nd = pd.to_datetime(TIMER["next_due"]) if TIMER["next_due"] else None
            if nd is None or now >= nd:
                # laufen
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                TIMER["last_run"] = now.isoformat()
                # nächster Slot exakt +60 min ab letztem Anker
                anchor = next_half_hour_anchor(now)
                next_slot = anchor + timedelta(minutes=TIMER["poll_minutes"])
                TIMER["next_due"] = next_slot.isoformat()
            await asyncio.sleep(5)
    finally:
        TIMER["running"] = False

TIMER_TASK: Optional[asyncio.Task] = None

# ========= Telegram Commands =========
async def cmd_start(update: Update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await send_text(CHAT_ID,
        "🤖 Verbunden.\n"
        "Cmds: /status /cfg /set key=value … /run /sig /dump /timer on|off /timerstatus /timerrunnow "
        "/trade on|off /pos /account"
    )

async def cmd_status(update: Update, context):
    pos_lines = [f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}" for s,p in STATE.positions.items()]
    pos_txt = "\n".join(pos_lines) if pos_lines else "keine (sim)"
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}\n"
        f"TF: {CONFIG.interval} | Provider: Alpaca ({APCA_DATA_FEED})\n"
        f"Timer: {'ON' if TIMER['enabled'] else 'OFF'} running={TIMER['running']} next={TIMER['next_due']}\n"
        f"MarketHoursOnly={TIMER['market_hours_only']} | poll={TIMER['poll_minutes']}m\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper)\n"
        f"Last: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}"
    )

async def cmd_cfg(update: Update, context):
    await update.message.reply_text("⚙️ Konfig:\n"+json.dumps(CONFIG.dict(), indent=2))

def set_from_kv(kv: str) -> str:
    k,v = kv.split("=",1); k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k,k)
    if not hasattr(CONFIG, k): return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG,k)
    if isinstance(cur,bool):   setattr(CONFIG,k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur,int):  setattr(CONFIG,k, int(float(v)))
    elif isinstance(cur,float):setattr(CONFIG,k, float(v))
    elif isinstance(cur,list): setattr(CONFIG,k, [x.strip() for x in v.split(",") if x.strip()])
    else:                      setattr(CONFIG,k, v)
    if k=="poll_minutes": TIMER["poll_minutes"] = getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update: Update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value …\n"
            "z.B. /set rsiLow=0 rsiHigh=68 rsiExit=48 sl=2 tp=4\n"
            "/set sizing_mode=shares sizing_value=1\n"
            "/set poll_minutes=60 market_hours_only=true"
        ); return
    msgs=[]; errs=[]
    for a in context.args:
        if "=" not in a: errs.append(f"❌ {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = ("✅ Übernommen:\n"+"\n".join(msgs)+("\n\n⚠️\n"+ "\n".join(errs) if errs else "")).strip()
    await update.message.reply_text(txt)

async def cmd_run(update: Update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    await update.message.reply_text("✅ run done")

async def cmd_sig(update: Update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG)
    last = f.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} {CONFIG.interval}\n"
        f"rsi={last['rsi']:.2f} (rise={bool(last['rsi_rising'])})  "
        f"efi={last['efi']:.2f} (rise={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])} exit={bool(last['exit_cond'])}\n"
        f"{friendly_note(note)}"
    )

async def cmd_dump(update: Update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = build_features(df, CONFIG)
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
    await update.message.reply_text("🧾 Dump\n"+json.dumps(payload, indent=2))

async def cmd_timer(update: Update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /timer on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    TIMER["enabled"] = on
    await update.message.reply_text(f"Timer = {'ON' if on else 'OFF'}")

async def cmd_timerstatus(update: Update, context):
    await update.message.reply_text("⏱️ Timer\n"+json.dumps({
        "enabled": TIMER["enabled"],
        "running": TIMER["running"],
        "poll_minutes": TIMER["poll_minutes"],
        "last_run": TIMER["last_run"],
        "next_due": TIMER["next_due"],
        "market_hours_only": TIMER["market_hours_only"],
    }, indent=2))

async def cmd_timerrunnow(update: Update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    now = datetime.now(timezone.utc)
    TIMER["last_run"] = now.isoformat()
    # nächsten Slot exakt auf :30 + poll
    anchor = next_half_hour_anchor(now)
    next_slot = anchor + timedelta(minutes=TIMER["poll_minutes"])
    TIMER["next_due"] = next_slot.isoformat()
    await update.message.reply_text("⏱️ Timer-Run done.")

async def cmd_trade(update: Update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /trade on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.trade_enabled = on
    await update.message.reply_text(f"Trading (Paper) = {'ON' if on else 'OFF'}")

async def cmd_pos(update: Update, context):
    pos = alpaca_positions()
    if pos:
        lines = [f"{p['symbol']}: qty={p['qty']}, avg={p['avg_entry']:.4f}, last={p['market_price']:.4f}, UPL={p['unrealized_pl']:.2f}" for p in pos]
        await update.message.reply_text("📦 Alpaca Positionen\n" + "\n".join(lines))
    else:
        await update.message.reply_text("📦 Alpaca Positionen: keine / kein Zugriff.")

async def cmd_account(update: Update, context):
    acc = alpaca_account()
    if acc:
        await update.message.reply_text("👤 Alpaca Account\n" + json.dumps(acc, indent=2))
    else:
        await update.message.reply_text("👤 Alpaca Account: kein Zugriff.")

async def on_message(update: Update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= FastAPI lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, TIMER_TASK
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("timer",        cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("timerrunnow",  cmd_timerrunnow))
        tg_app.add_handler(CommandHandler("trade",   cmd_trade))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(CommandHandler("account", cmd_account))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            print("delete_webhook warn:", e)

        # Polling starten (einfach)
        started=False; delay=5
        while not started:
            try:
                await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                started=True
                print("✅ Telegram polling läuft")
            except Conflict as e:
                print(f"⚠️ Conflict: {e} – retry in {delay}s")
                await asyncio.sleep(delay); delay=min(delay*2,60)

        # Timer ggf. starten
        if TIMER["enabled"] and (TIMER_TASK is None or TIMER_TASK.done()):
            TIMER_TASK = asyncio.create_task(timer_loop())
            print("⏱️ Timer gestartet")

    except Exception as e:
        print("❌ Telegram startup error:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        try:
            if TIMER_TASK and not TIMER_TASK.done():
                TIMER["enabled"]=False
                await asyncio.sleep(0.1)
                TIMER_TASK.cancel()
        except: pass
        try:
            if tg_app and tg_app.updater:
                await tg_app.updater.stop()
        except: pass
        try:
            if tg_app:
                await tg_app.stop()
                await tg_app.shutdown()
        except: pass
        print("🛑 Shutdown complete")

# ========= FastAPI app & routes =========
app = FastAPI(title="TV-synced RSI/EFI + Telegram", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": f"Alpaca ({APCA_DATA_FEED})",
        "live": CONFIG.live_enabled,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "poll_minutes": TIMER["poll_minutes"],
            "last_run": TIMER["last_run"],
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
    anchor = next_half_hour_anchor(now)
    TIMER["next_due"]=(anchor + timedelta(minutes=TIMER["poll_minutes"])).isoformat()
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
        "PDT_JSON_exists": PDT_JSON.exists()
    }

@app.get("/tgstatus")
def tgstatus():
    return {"polling_started": True if tg_app and tg_app.updater else False}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
