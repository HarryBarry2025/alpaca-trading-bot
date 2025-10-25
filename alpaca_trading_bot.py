# alpaca_trading_bot.py  (V5.1.1 consolidated)
# (see previous cell for full description header)
import os, io, json, math, time, asyncio, traceback, warnings, pathlib
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

# ========= ENV & Defaults =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca (Paper)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "iex").lower()  # "sip" or "iex"

# Toggles & timer
ENV_ENABLE_TRADE      = os.getenv("ENABLE_TRADE", "false").lower() in ("1","true","on","yes")
ENV_ENABLE_TIMER      = os.getenv("ENABLE_TIMER", "false").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES      = int(os.getenv("POLL_MINUTES", "60"))
ENV_MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

# Persistence paths
from pathlib import Path

DATA_DIR = Path("./data")  # Render-kompatibel
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDT_JSON = DATA_DIR / "pdt_trades.json"

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    # Engine
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"
    lookback_days: int = 365

    # TV-compatible indicators
    rsiLen: int = 12
    rsiLow: float = 0.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Timer / Market hours
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = False
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY
    rth_only: bool = True

    # Data Provider
    data_provider: str = "alpaca"
    yahoo_retries: int = 2
    yahoo_backoff_sec: float = 1.2
    allow_stooq_fallback: bool = True
    alpaca_feed: str = APCA_DATA_FEED

    # Trading Toggle & Sizer
    trade_enabled: bool = ENV_ENABLE_TRADE
    sizing_mode: str = "shares"            # "shares","percent_equity","notional_usd","risk"
    sizing_value: float = 1.0
    max_position_pct: float = 100.0

    # Slippage/Fees
    bt_slippage_bps: float = 1.0
    bt_fee_per_trade: float = 0.0

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID

# ========= Indicators =========
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

# ========= US Market Hours =========
def is_us_holiday(dt_utc: datetime) -> bool:
    y = dt_utc.year
    fixed = {f"{y}-01-01", f"{y}-07-04", f"{y}-12-25"}
    known = {
        "2024-01-15","2024-02-19","2024-03-29","2024-05-27","2024-06-19","2024-09-02","2024-11-28",
        "2025-01-20","2025-02-17","2025-04-18","2025-05-26","2025-06-19","2025-09-01","2025-11-27"
    }
    ds = dt_utc.strftime("%Y-%m-%d")
    return ds in fixed or ds in known

def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or datetime.now(timezone.utc)
    if is_us_holiday(now): return False
    if now.weekday() >= 5: return False
    m = now.hour*60 + now.minute
    return 13*60+30 <= m <= 20*60

# ========= Data =========
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

def fetch_alpaca_bars(symbol: str, minutes: int, lookback_days: int) -> pd.DataFrame:
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

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 60))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=("SIP" if CONFIG.alpaca_feed=="sip" else "IEX"),
        limit=50000
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty minute frame")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except Exception: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df = df.sort_index()
        df = df.rename(columns=str.lower)
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def yahoo_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    tmp = yf.download(tickers=symbol, interval=interval, period=period,
                      auto_adjust=False, progress=False, prepost=False, threads=False)
    if tmp is None or tmp.empty: return pd.DataFrame()
    tmp = tmp.rename(columns=str.lower)
    if isinstance(tmp.columns, pd.MultiIndex):
        try: tmp = tmp.xs(symbol, axis=1)
        except Exception: pass
    idx = tmp.index
    if isinstance(idx, pd.DatetimeIndex):
        try: tmp.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        except Exception: pass
    tmp["time"] = tmp.index
    return tmp

def fetch_ohlcv_tv_synced(symbol: str, interval: str, lookback_days: int,
                          rth_only: bool = True, drop_last_incomplete: bool = True
                          ) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"", "detail":""}
    interval = interval.lower()

    if interval == "1m":
        base_min = 1; assemble = False
    elif interval == "15m":
        base_min = 1; assemble = True; target = 15
    elif interval == "1h":
        base_min = 1; assemble = True; target = 60
    elif interval == "1d":
        base_min = None; assemble = False
    else:
        base_min = 1; assemble = True; target = 60

    df = pd.DataFrame()
    if CONFIG.data_provider.lower() == "alpaca":
        if interval != "1d":
            df = fetch_alpaca_bars(symbol, base_min, lookback_days)
            if not df.empty:
                note.update(provider=f"Alpaca ({CONFIG.alpaca_feed})", detail=f"{interval}")
                if rth_only:
                    m = df.index.minute + df.index.hour*60
                    rth_mask = (m >= (13*60+30)) & (m <= (20*60))
                    df = df.loc[rth_mask]
                if assemble:
                    if target == 60:
                        minutes_since_midnight = df.index.hour*60 + df.index.minute
                        offset = (minutes_since_midnight - 810) % 60
                        start_stamp = df.index - pd.to_timedelta(offset, unit="m")
                        df = df.assign(_g=start_stamp)
                        agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                        g = df.groupby("_g", sort=True).agg(agg).dropna()
                        g.index = pd.DatetimeIndex(g.index).tz_localize("UTC")
                        g["time"] = g.index + pd.to_timedelta(60, "m")
                        if drop_last_incomplete:
                            last_end = g.index[-1] + pd.to_timedelta(60, "m")
                            if last_end > datetime.now(timezone.utc):
                                g = g.iloc[:-1]
                        df = g
                    elif target == 15:
                        minutes_since_midnight = df.index.hour*60 + df.index.minute
                        offset = (minutes_since_midnight - 810) % 15
                        start_stamp = df.index - pd.to_timedelta(offset, unit="m")
                        df = df.assign(_g=start_stamp)
                        agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                        g = df.groupby("_g", sort=True).agg(agg).dropna()
                        g.index = pd.DatetimeIndex(g.index).tz_localize("UTC")
                        g["time"] = g.index + pd.to_timedelta(15, "m")
                        if drop_last_incomplete:
                            last_end = g.index[-1] + pd.to_timedelta(15, "m")
                            if last_end > datetime.now(timezone.utc):
                                g = g.iloc[:-1]
                        df = g
                return df, note

    if interval in {"1m","15m","60m","1h"}:
        period = f"{min(lookback_days, 730)}d"
        try:
            yf_interval = {"1m":"1m","15m":"15m","1h":"60m"}.get(interval, "60m")
            tmp = yahoo_download(symbol, yf_interval, period)
            if not tmp.empty:
                note.update(provider="Yahoo", detail=f"{interval}")
                return tmp, note
        except Exception:
            pass

    dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
    if not dfe.empty:
        note.update(provider="Stooq EOD (Fallback)", detail="1d")
        return dfe, note

    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Feature & Signals =========
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

# ========= PDT =========
def load_pdt_trades() -> List[Dict[str,Any]]:
    if PDT_JSON.exists():
        try:
            return json.loads(PDT_JSON.read_text())
        except Exception:
            return []
    return []

def save_pdt_trades(trades: List[Dict[str,Any]]):
    try:
        PDT_JSON.write_text(json.dumps(trades, indent=2))
    except Exception as e:
        print("save_pdt_trades error:", e)

def prune_pdt_window(trades: List[Dict[str,Any]], now_utc: datetime) -> List[Dict[str,Any]]:
    cutoff = now_utc - timedelta(days=5)
    return [t for t in trades if datetime.fromisoformat(t["time"]) >= cutoff]

def count_day_trades(trades: List[Dict[str,Any]]) -> int:
    return len(trades)

def pdt_block_active() -> bool:
    tr = prune_pdt_window(load_pdt_trades(), datetime.now(timezone.utc))
    return count_day_trades(tr) >= 3

def record_day_trade(symbol: str):
    tr = prune_pdt_window(load_pdt_trades(), datetime.now(timezone.utc))
    tr.append({"time": datetime.now(timezone.utc).isoformat(), "symbol": symbol})
    save_pdt_trades(tr)

# ========= Trading via Alpaca =========
def alpaca_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

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

def compute_order_qty(sym: str, price: float) -> int:
    mode = CONFIG.sizing_mode.lower()
    v = CONFIG.sizing_value
    acc = alpaca_account() if CONFIG.trade_enabled else {}
    equity = acc.get("equity", 10000.0)
    max_notional = float(equity) * (CONFIG.max_position_pct/100.0)

    qty = 0.0
    if mode == "shares":
        qty = v
    elif mode == "percent_equity":
        qty = float(equity) * (v/100.0) / max(1e-9, price)
    elif mode == "notional_usd":
        qty = float(v) / max(1e-9, price)
    elif mode == "risk":
        sl_price = price*(1-CONFIG.slPerc/100.0)
        risk_per_share = max(1e-9, price - sl_price)
        qty = (float(equity)*(v/100.0))/risk_per_share
    else:
        qty = 1.0

    qty = min(qty, math.floor(max_notional / max(1e-9, price)))
    qty = int(max(0, math.floor(qty)))
    return qty

async def place_market_order(sym: str, qty: int, side: str, tif: str = "day") -> str:
    client = alpaca_trading_client()
    if client is None: return "alpaca client not available"
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tif_map = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "opg": TimeInForce.OPG,
                   "cls": TimeInForce.CLS, "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK}
        req = MarketOrderRequest(
            symbol=sym, qty=qty,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif_map.get(tif.lower(), TimeInForce.DAY)
        )
        order = client.submit_order(order_data=req)
        return f"order_id={order.id}"
    except Exception as e:
        return f"alpaca order error: {e}"

# ========= Runner =========
def friendly_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if "Fallback" in prov or "Stooq" in prov or "Alpaca" in prov:
        return f"📡 Quelle: {prov} – {det}"
    return f"📡 Quelle: {prov} ({det})"

def intrabar_hit(sl: float, tp: float, bar_high: float, bar_low: float) -> Tuple[bool,bool]:
    hit_tp = bar_high >= tp
    hit_sl = bar_low  <= sl
    if hit_tp and hit_sl:
        return True, True
    return hit_sl, hit_tp

async def run_once_for_symbol(sym: str, send_signals: bool = True) -> Dict[str,Any]:
    df, note = fetch_ohlcv_tv_synced(sym, CONFIG.interval, CONFIG.lookback_days,
                                     rth_only=CONFIG.rth_only, drop_last_incomplete=True)
    if df.empty:
        return {"ok":False, "msg": f"❌ Keine Daten für {sym}. {note}"}

    f = compute_signals_for_frame(df, CONFIG)
    if len(f) < 3:
        return {"ok":False, "msg":"zu wenige Bars"}

    last, prev = f.iloc[-1], f.iloc[-2]
    px_open  = float(last["open"])
    px_close = float(last["close"])
    bar_high = float(last["high"])
    bar_low  = float(last["low"])
    ts       = last["time"]

    entry_cond = bool(last["entry_cond"])
    exit_cond  = bool(last["exit_cond"])

    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size = pos["size"]; avg = pos["avg"]; entry_time = pos["entry_time"]

    def sl(p): return p*(1-CONFIG.slPerc/100.0)
    def tp(p): return p*(1+CONFIG.tpPerc/100.0)

    if send_signals and CHAT_ID:
        await tg_send(f"ℹ️ {sym} {CONFIG.interval} rsi={last['rsi']:.2f} efi={last['efi']:.2f} • entry={entry_cond} exit={exit_cond}\n{friendly_note(note)}")

    if CONFIG.trade_enabled and pdt_block_active():
        if CHAT_ID:
            await tg_send("⛔ PDT: blockiert (≥3 Daytrades in 5 Tagen). Keine neuen Trades.")
        allow_new_orders = False
    else:
        allow_new_orders = CONFIG.trade_enabled

    if size == 0:
        if entry_cond:
            qty = compute_order_qty(sym, px_open)
            if allow_new_orders and qty > 0 and is_market_open_now():
                info = await place_market_order(sym, qty, "buy", "day")
                if CHAT_ID and send_signals:
                    await tg_send(f"🛒 BUY {sym} x{qty} @ ~{px_open:.4f} • {info}")
                record_day_trade(sym)
            STATE.positions[sym] = {"size":qty if qty>0 else 1, "avg":px_open, "entry_time":str(ts)}
            if CHAT_ID and send_signals and qty<=0:
                await tg_send(f"🟢 LONG (sim) {sym} @ {px_open:.4f}")
    else:
        cur_sl = sl(avg); cur_tp = tp(avg)
        hit_sl, hit_tp = intrabar_hit(cur_sl, cur_tp, bar_high, bar_low)

        sell_reason = None
        sell_px     = None
        if hit_sl:
            sell_reason="stop_loss"; sell_px=cur_sl
        elif hit_tp:
            sell_reason="take_profit"; sell_px=cur_tp
        elif exit_cond:
            sell_reason="rsi_exit"; sell_px=px_open

        if sell_reason:
            if allow_new_orders and is_market_open_now():
                info = await place_market_order(sym, size, "sell", "day")
                if CHAT_ID and send_signals:
                    await tg_send(f"🛒 SELL {sym} x{size} @ ~{sell_px:.4f} • {sell_reason} • {info}")
                record_day_trade(sym)
            pnl = (sell_px - avg) / max(1e-9, avg)
            if CHAT_ID and send_signals:
                await tg_send(f"🔴 EXIT (sim) {sym} @ {sell_px:.4f} • {sell_reason} • PnL={pnl*100:.2f}%")
            STATE.positions[sym] = {"size":0, "avg":0.0, "entry_time":None}

    STATE.last_status = f"{sym}: pos={STATE.positions.get(sym,{}).get('size',0)} @ {STATE.positions.get(sym,{}).get('avg',0.0):.4f}"
    return {"ok":True}

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
POLLING_STARTED = False

async def tg_send(text: str):
    if tg_app is None or CHAT_ID is None: return
    if not text.strip(): text="(leer)"
    try:
        await tg_app.bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        print("tg_send error:", e)

async def send_document_bytes(data: bytes, filename: str, caption: str = ""):
    if tg_app is None or CHAT_ID is None: return
    try:
        bio = io.BytesIO(data); bio.name = filename; bio.seek(0)
        await tg_app.bot.send_document(chat_id=CHAT_ID, document=InputFile(bio), caption=caption)
    except Exception as e:
        print("send_document error:", e)

async def send_png(fig, filename: str, caption: str = ""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    await send_document_bytes(buf.getvalue(), filename, caption)

# ========= Timer =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}

def next_tf_due(now: datetime, interval: str) -> datetime:
    if interval == "1h":
        nxt = now.replace(minute=30, second=0, microsecond=0)
        if now.minute >= 30:
            nxt = (nxt + timedelta(hours=1))
        return nxt
    if interval == "15m":
        base = now.replace(second=0, microsecond=0)
        # next quarter after :30 anchor
        mins = base.minute
        add = (15 - ((mins - 30) % 15)) % 15
        if add == 0: add = 15
        return base + timedelta(minutes=add)
    if interval == "1m":
        return now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    base = now.replace(hour=20, minute=1, second=0, microsecond=0)
    if now >= base:
        base = base + timedelta(days=1)
    return base

async def timer_loop():
    TIMER["running"] = True
    try:
        while TIMER["enabled"]:
            now = datetime.now(timezone.utc)
            if TIMER["market_hours_only"] and not is_market_open_now(now):
                TIMER["next_due"] = None
                await asyncio.sleep(30)
                continue

            if TIMER["next_due"] is None:
                due = next_tf_due(now, CONFIG.interval)
                TIMER["next_due"] = due.isoformat()

            due_dt = datetime.fromisoformat(TIMER["next_due"])
            if now >= due_dt:
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, send_signals=True)
                TIMER["last_run"] = datetime.now(timezone.utc).isoformat()
                TIMER["next_due"] = next_tf_due(datetime.now(timezone.utc), CONFIG.interval).isoformat()
            await asyncio.sleep(1)
    finally:
        TIMER["running"] = False

TIMER_TASK: Optional[asyncio.Task] = None

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Verbunden.\n"
        "/status, /cfg, /set key=value …, /run, /live on|off\n"
        "/bt [tage], /wf [is_days oos_days]\n"
        "/sig, /ind, /plot\n"
        "/dump [csv [N]], /dumpcsv [N]\n"
        "/trade on|off, /pos, /account\n"
        "/timer on|off, /timerstatus, /timerrunnow, /market on|off\n"
        "/ptd, /ptdreset"
    )

async def cmd_status(update, context):
    pos_lines=[]
    for s,p in STATE.positions.items():
        pos_lines.append(f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}")
    pos_txt = "\n".join(pos_lines) if pos_lines else "keine (sim)"
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider} (Alpaca feed={CONFIG.alpaca_feed})\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} • Timer: {'ON' if TIMER['enabled'] else 'OFF'} "
        f"(market-hours-only={TIMER['market_hours_only']})\n"
        f"LastStatus: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}\n"
        f"Trading: {'ON' if CONFIG.trade_enabled else 'OFF'} (Paper)\n"
        f"Sizing: {CONFIG.sizing_mode}={CONFIG.sizing_value}  max_position_pct={CONFIG.max_position_pct}%"
    )

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

def set_from_kv(k: str, v: str) -> str:
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
    k = mapping.get(k, k)
    if not hasattr(CONFIG, k): return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):   setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):  setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):setattr(CONFIG, k, float(v))
    elif isinstance(cur, list): setattr(CONFIG, k, [x.strip() for x in v.split(",") if x.strip()])
    else:                       setattr(CONFIG, k, v)
    if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    if k=="alpaca_feed": CONFIG.alpaca_feed = str(getattr(CONFIG,k)).lower()
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "z.B. /set rsiLow=0 rsiHigh=68 rsiExit=48 sl=2 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set symbols=TQQQ SPY\n"
            "/set data_provider=alpaca  alpaca_feed=sip\n"
            "/set sizing_mode=percent_equity sizing_value=100 max_position_pct=100\n"
            "/set rth_only=true market_hours_only=true"
        ); return
    msgs=[]; errs=[]
    for a in context.args:
        if "=" not in a: errs.append(f"❌ ungültig: {a}"); continue
        k,v = a.split("=",1); k=k.strip(); v=v.strip()
        try: msgs.append(set_from_kv(k,v))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    out = ("✅ Übernommen:\n" + "\n".join(msgs) + ("\n\n⚠️ Probleme:\n" + "\n".join(errs) if errs else "")).strip()
    await update.message.reply_text(out)

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_market(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /market on|off (market-hours-only)"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.market_hours_only = on
    TIMER["market_hours_only"] = on
    await update.message.reply_text(f"market_hours_only = {'true' if on else 'false'}")

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
    df, note = fetch_ohlcv_tv_synced(sym, CONFIG.interval, days, rth_only=CONFIG.rth_only, drop_last_incomplete=True)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für Backtest ({sym})."); return
    f = compute_signals_for_frame(df, CONFIG)

    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    fee = CONFIG.bt_fee_per_trade
    slip = CONFIG.bt_slippage_bps/10000.0

    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry = bool(row["entry_cond"]); exitc = bool(row["exit_cond"])
        if pos==0 and entry:
            pos=1; avg=float(row["open"])*(1+slip); entries+=1; eq-=fee/10000.0
        elif pos==1:
            slp = avg*(1-CONFIG.slPerc/100); tpp = avg*(1+CONFIG.tpPerc/100)
            bh, bl = float(row["high"]), float(row["low"])
            hit_sl = bl<=slp; hit_tp = bh>=tpp
            if hit_sl or hit_tp or exitc:
                px = slp if hit_sl else tpp if hit_tp else float(row["open"])*(1-slip)
                r = (px-avg)/avg
                eq*= (1+r); eq-=fee/10000.0
                R.append(r); exits+=1
                pos=0; avg=0.0

    if R:
        a=np.array(R); win=(a>0).mean()
        pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(
            f"📈 Backtest {days}d  Trades={entries}/{exits}  Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\n"
            f"ℹ️ Hinweis: Slippage={CONFIG.bt_slippage_bps}bps, Fee=${CONFIG.bt_fee_per_trade}"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_wf(update, context):
    is_days, oos_days = 120, 30
    if context.args and len(context.args)>=2:
        try:
            is_days = int(context.args[0]); oos_days = int(context.args[1])
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_synced(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.rth_only, drop_last_incomplete=True)
    if df.empty or len(df)<(is_days+oos_days+10):
        await update.message.reply_text("❌ Zu wenige Daten für Walk-Forward."); return
    f = compute_signals_for_frame(df, CONFIG)

    lows  = [0, 30, 40, 50]
    highs = [60, 68, 70]
    exits = [40, 45, 48, 50]

    # split into rolling windows by index
    chunks = []
    start_idx = 0
    while True:
        is_end_idx = start_idx + is_days
        oos_end_idx= is_end_idx + oos_days
        if oos_end_idx >= len(f): break
        is_df  = f.iloc[start_idx:is_end_idx]
        oos_df = f.iloc[is_end_idx:oos_end_idx]
        start_idx += oos_days
        if len(is_df)<20 or len(oos_df)<5: continue

        best=(None,-1e9)
        for lo in lows:
            for hi in highs:
                if hi<=lo: continue
                for ex in exits:
                    eq=1.0; pos=0; avg=0.0
                    for j in range(2,len(is_df)):
                        row=is_df.iloc[j]
                        entry=bool((row["rsi"]>lo)&(row["rsi"]<hi)&(row["rsi"]>is_df["rsi"].iloc[j-1])&(row["efi"]>is_df["efi"].iloc[j-1]))
                        exitc=bool((row["rsi"]<ex)&(row["rsi"]<is_df["rsi"].iloc[j-1]))
                        if pos==0 and entry:
                            pos=1; avg=float(row["open"])
                        elif pos==1:
                            slp=avg*(1-CONFIG.slPerc/100); tpp=avg*(1+CONFIG.tpPerc/100)
                            hit_sl=row["low"]<=slp; hit_tp=row["high"]>=tpp
                            if hit_sl or hit_tp or exitc:
                                px=slp if hit_sl else tpp if hit_tp else float(row["open"])
                                r=(px-avg)/avg; eq*=(1+r); pos=0; avg=0.0
                    if eq>best[1]: best=((lo,hi,ex),eq)

        if best[0] is None: continue
        lo,hi,ex=best[0]
        eq=1.0; pos=0; avg=0.0; R=[]
        for j in range(2,len(oos_df)):
            row=oos_df.iloc[j]
            entry=bool((row["rsi"]>lo)&(row["rsi"]<hi)&(row["rsi"]>oos_df["rsi"].iloc[j-1])&(row["efi"]>oos_df["efi"].iloc[j-1]))
            exitc=bool((row["rsi"]<ex)&(row["rsi"]<oos_df["rsi"].iloc[j-1]))
            if pos==0 and entry:
                pos=1; avg=float(row["open"])
            elif pos==1:
                slp=avg*(1-CONFIG.slPerc/100); tpp=avg*(1+CONFIG.tpPerc/100)
                hit_sl=row["low"]<=slp; hit_tp=row["high"]>=tpp
                if hit_sl or hit_tp or exitc:
                    px=slp if hit_sl else tpp if hit_tp else float(row["open"])
                    r=(px-avg)/avg; eq*=(1+r); R.append(r); pos=0; avg=0.0
        chunks.append({"is_days":is_days, "oos_days":oos_days,
                       "best":[lo,hi,ex], "oos_eq":eq, "oos_trades":len(R),
                       "oos_win":float((np.array(R)>0).mean()) if R else None})

    await update.message.reply_text("Walk-Forward (IS/OOS)\n" + json.dumps({"chunks":chunks}, indent=2))

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_tv_synced(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.rth_only, drop_last_incomplete=True)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    f = compute_signals_for_frame(df, CONFIG)
    last = f.iloc[-1]
    txt = (
        f"🔎 {sym} {CONFIG.interval}\n"
        f"RSI={last['rsi']:.2f} (rising={bool(last['rsi_rising'])})  "
        f"EFI={last['efi']:.2f} (rising={bool(last['efi_rising'])})\n"
        f"entry={bool(last['entry_cond'])}  exit={bool(last['exit_cond'])}\n"
        f"{friendly_note({'provider': f'Alpaca ({CONFIG.alpaca_feed})', 'detail': CONFIG.interval})}"
    ).strip()
    await update.message.reply_text(txt)

async def cmd_ind(update, context):
    await cmd_sig(update, context)

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    n = 300
    df, note = fetch_ohlcv_tv_synced(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.rth_only, drop_last_incomplete=True)
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

    await send_png(fig,  f"{sym}_{CONFIG.interval}_close.png", "📈 Close")
    await send_png(fig2, f"{sym}_{CONFIG.interval}_rsi.png",   "📈 RSI")
    await send_png(fig3, f"{sym}_{CONFIG.interval}_efi.png",   "📈 EFI")

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    args = [a.lower() for a in (context.args or [])]
    df, note = fetch_ohlcv_tv_synced(sym, CONFIG.interval, CONFIG.lookback_days, rth_only=CONFIG.rth_only, drop_last_incomplete=True)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten für {sym}."); return

    if args and args[0]=="csv":
        n = 300
        if len(args)>=2:
            try: n=max(1,int(args[1]))
            except: pass
        exp = build_export_frame(df, CONFIG).tail(n)
        csv_bytes = exp.to_csv(index=True).encode("utf-8")
        await send_document_bytes(csv_bytes,
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
        "market_hours_only": TIMER["market_hours_only"]
    }, indent=2))

async def cmd_timerrunnow(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True)
    now = datetime.now(timezone.utc)
    TIMER["last_run"] = now.isoformat()
    TIMER["next_due"] = next_tf_due(now, CONFIG.interval).isoformat()
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def cmd_ptd(update, context):
    tr = prune_pdt_window(load_pdt_trades(), datetime.now(timezone.utc))
    await update.message.reply_text("PDT\n" + json.dumps({"count_5d":len(tr), "records":tr}, indent=2))

async def cmd_ptdreset(update, context):
    save_pdt_trades([])
    await update.message.reply_text("PDT Reset: ok (persistenz geleert).")

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
        tg_app.add_handler(CommandHandler("market",  cmd_market))
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
        tg_app.add_handler(CommandHandler("ptd",     cmd_ptd))
        tg_app.add_handler(CommandHandler("ptdreset",cmd_ptdreset))
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

# ========= FastAPI app =========
app = FastAPI(title="Algo (V5.1.1) TV-synced", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "alpaca_feed": CONFIG.alpaca_feed,
        "live": CONFIG.live_enabled,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "poll_minutes": TIMER["poll_minutes"],
            "last_run": TIMER["last_run"],
            "next_due": TIMER["next_due"],
            "market_hours_only": TIMER["market_hours_only"]
        },
        "trade_enabled": CONFIG.trade_enabled,
        "sizer": {"mode": CONFIG.sizing_mode, "value": CONFIG.sizing_value, "max_position_pct": CONFIG.max_position_pct}
    }

@app.get("/tick")
async def tick():
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=False)
    now = datetime.now(timezone.utc)
    TIMER["last_run"]=now.isoformat()
    TIMER["next_due"]=next_tf_due(now, CONFIG.interval).isoformat()
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
