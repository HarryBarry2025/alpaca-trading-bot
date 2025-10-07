# alpaca_trading_bot.py
import os, json, time, asyncio, traceback
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.7)
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict, BadRequest

# --------- ENV ----------
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")                # ungenutzt im Polling-Modus
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")
if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional für MarktDaten/Orders)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")   # z.B. https://paper-api.alpaca.markets

# ---------- Konfig & State ----------
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"          # '1d', '1h', '15m' ...
    lookback_days: int = 365

    # Strategie-Inputs (MACD entfernt, rsiLow=0 wie gewünscht)
    rsiLen: int = 12
    rsiLow: float = 0.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk / Exits
    slPerc: float = 1.0
    tpPerc: float = 400.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine / Timer
    poll_minutes: int = 10
    live_enabled: bool = False
    timer_enabled: bool = False
    market_hours_only: bool = True

    # Positionsgröße & TIF für Alpaca
    qty_mode: str = "usd"         # "usd" oder "shares"
    qty_value: float = 1000.0     # USD pro Trade oder Anzahl Shares
    tif: str = "day"              # "day" oder "gtc"

    # Datenquelle
    data_provider: str = "alpaca" # "alpaca", "yahoo", "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ---------- US-Marktzeiten / Feiertage ----------
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")

US_HOLIDAYS_2025 = {
    date(2025,1,1),  # New Year's Day
    date(2025,1,20), # MLK Day
    date(2025,2,17), # Washington's Birthday
    date(2025,4,18), # Good Friday (NYSE)
    date(2025,5,26), # Memorial Day
    date(2025,6,19), # Juneteenth
    date(2025,7,4),  # Independence Day
    date(2025,9,1),  # Labor Day
    date(2025,11,27),# Thanksgiving
    date(2025,12,25) # Christmas
}

def is_us_market_open_now(now_utc: Optional[datetime]=None) -> bool:
    """Einfache Regular-Hours-Gate (Mo–Fr, 9:30–16:00 ET, excl. Holidays)."""
    if not CONFIG.market_hours_only:
        return True
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    now_et = now_utc.astimezone(ET)
    if now_et.weekday() >= 5:
        return False
    if now_et.date() in US_HOLIDAYS_2025:
        return False
    t = now_et.time()
    return (t >= datetime(2000,1,1,9,30,tzinfo=ET).time()) and (t <= datetime(2000,1,1,16,0,tzinfo=ET).time())

# ---------- Indikatoren ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int=14) -> pd.Series:
    delta = s.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=s.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=s.index)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

def efi_func(close: pd.Series, vol: pd.Series, efi_len: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, efi_len)

# ---------- Datenprovider ----------
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
    end   = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
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
        if not bars or bars.df is None or bars.df.empty: return pd.DataFrame()
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
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    # 1) Alpaca
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=interval); return df, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")
    # 2) Stooq EOD
    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d"); return df, note
        note.update(provider="Stooq→Yahoo", detail="Stooq leer; versuche Yahoo")

    # 3) Yahoo + Retries
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}
    is_intraday = interval in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _normalize(tmp: pd.DataFrame) -> pd.DataFrame:
        if tmp is None or tmp.empty: return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except Exception: pass
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            except Exception: pass
        df["time"] = df.index
        return df.sort_index()

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True,  threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    last_err = None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = _normalize(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}"); return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # Yahoo Fallback: 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} failed"); return df, note
    except Exception as e:
        last_err = e

    # Optional Stooq Failover
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days,120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d"); return dfe, note

    print(f"[fetch_ohlcv] no data for {symbol} ({interval}, {period}). Last err: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ---------- Features ----------
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsiLen)
    out["efi"] = efi_func(out["close"], out["volume"], cfg.efiLen)
    return out

# ---------- Strategie-Logik ----------
def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]; prev = df.iloc[-2]
    rsi_val = float(last["rsi"])
    rsi_rising  = rsi_val > float(prev["rsi"])
    rsi_falling = rsi_val < float(prev["rsi"])
    efi_rising  = float(last["efi"]) > float(prev["efi"])

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = last["time"]

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    # Bars since entry
    bars_in_trade = 0
    if st.entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if st.position_size == 0:
        if entry_cond:
            return {"action":"buy", "qty":1, "price":price_open, "time":str(ts),
                    "reason":"rule_entry", "sl":sl(price_open), "tp":tp(price_open)}
        return {"action":"none", "reason":"flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = bars_in_trade >= cfg.minBarsInTrade
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(STATE.avg_price)
        cur_tp = tp(STATE.avg_price)
        hit_sl = price_close <= cur_sl
        hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell", "qty":st.position_size, "price":price_open, "time":str(ts), "reason":"rsi_exit"}
        if hit_sl:
            return {"action":"sell", "qty":st.position_size, "price":cur_sl, "time":str(ts), "reason":"stop_loss"}
        if hit_tp:
            return {"action":"sell", "qty":st.position_size, "price":cur_tp, "time":str(ts), "reason":"take_profit"}
        return {"action":"none", "reason":"hold"}

# ---------- Alpaca Trading / Positions ----------
def get_alpaca_trading_client():
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY and APCA_API_BASE_URL):
        return None
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper="paper" in APCA_API_BASE_URL)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

def alpaca_place_order(side: str, qty_shares: int, tif: str) -> str:
    client = get_alpaca_trading_client()
    if client is None: return "Alpaca trading client not configured."
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tif_enum = TimeInForce.DAY if tif.lower()=="day" else TimeInForce.GTC
        req = MarketOrderRequest(
            symbol=CONFIG.symbol,
            qty=qty_shares,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif_enum
        )
        order = client.submit_order(req)
        return f"Alpaca order {order.id} {side.upper()} {qty_shares} {CONFIG.symbol} TIF={tif}"
    except Exception as e:
        return f"[alpaca] order error: {e}"

def alpaca_positions_report() -> str:
    client = get_alpaca_trading_client()
    if client is None: return "Alpaca trading client not configured."
    try:
        poss = client.get_all_positions()
        if not poss: return "Keine offenen Alpaca-Positionen."
        lines = ["📦 Alpaca Positionen:"]
        for p in poss:
            lines.append(f"- {p.symbol}: {p.qty} @ {p.avg_entry_price} (market={p.current_price}) PnL={p.unrealized_pl} ({p.unrealized_plpc})")
        return "\n".join(lines)
    except Exception as e:
        return f"[alpaca] positions error: {e}"

def compute_shares_from_config(price: float) -> int:
    if CONFIG.qty_mode.lower()=="shares":
        return max(1, int(round(CONFIG.qty_value)))
    # usd
    shares = int(max(1, CONFIG.qty_value // max(0.01, price)))
    return shares

# ---------- Telegram Helper ----------
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text if text.strip() else "ℹ️ (leer)")
    except Exception as e:
        print("send error:", e)

def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if prov.startswith("Yahoo (Fallback"): return f"📡 Daten: {prov} – {det}"
    if "Stooq" in prov:                    return f"📡 Daten: {prov} – {det} (nur Daily)"
    return f"📡 Daten: {prov} – {det}"

# ---------- Telegram Commands ----------
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg\n"
        "/set key=value …  (z.B. /set data_provider=alpaca interval=1h poll_minutes=10)\n"
        "/run, /bt 180\n"
        "/timer on|off, /timerstatus\n"
        "/ind 5, /pos\n"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        f"📊 Status\nSymbol: {CONFIG.symbol} {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}\n"
        f"Timer: {'ON' if CONFIG.timer_enabled else 'OFF'} alle {CONFIG.poll_minutes}m\n"
        f"US-Regular-Hours-Gate: {'ON' if CONFIG.market_hours_only else 'OFF'}\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Last: {STATE.last_status}\nDatenquelle: {CONFIG.data_provider}"
    )

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1); k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc","tp":"tpPerc","samebar":"allowSameBarExit","cooldown":"minBarsInTrade"}
    k = mapping.get(k,k)
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
            "Nutze: /set key=value [key=value] …\n"
            "z.B. /set data_provider=alpaca interval=1h poll_minutes=10 market_hours_only=true\n"
            "/set qty_mode=usd qty_value=1000 tif=day"
        ); return
    msgs, errs = [], []
    for a in context.args:
        if "=" not in a: errs.append(f"❌ Ungültig: {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = "\n".join((["✅ Übernommen:"]+msgs)+(["\n⚠️ Probleme:"]+errs if errs else []))
    await update.message.reply_text(txt.strip())

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n"+json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

# alpaca_trading_bot.py
import os, json, time, asyncio, traceback
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (PTB v20.7)
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict, BadRequest

# --------- ENV ----------
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")                # ungenutzt im Polling-Modus
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")
if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional für MarktDaten/Orders)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")   # z.B. https://paper-api.alpaca.markets

# ---------- Konfig & State ----------
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"          # '1d', '1h', '15m' ...
    lookback_days: int = 365

    # Strategie-Inputs (MACD entfernt, rsiLow=0 wie gewünscht)
    rsiLen: int = 12
    rsiLow: float = 0.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk / Exits
    slPerc: float = 1.0
    tpPerc: float = 400.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine / Timer
    poll_minutes: int = 10
    live_enabled: bool = False
    timer_enabled: bool = False
    market_hours_only: bool = True

    # Positionsgröße & TIF für Alpaca
    qty_mode: str = "usd"         # "usd" oder "shares"
    qty_value: float = 1000.0     # USD pro Trade oder Anzahl Shares
    tif: str = "day"              # "day" oder "gtc"

    # Datenquelle
    data_provider: str = "alpaca" # "alpaca", "yahoo", "stooq_eod"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ---------- US-Marktzeiten / Feiertage ----------
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")

US_HOLIDAYS_2025 = {
    date(2025,1,1),  # New Year's Day
    date(2025,1,20), # MLK Day
    date(2025,2,17), # Washington's Birthday
    date(2025,4,18), # Good Friday (NYSE)
    date(2025,5,26), # Memorial Day
    date(2025,6,19), # Juneteenth
    date(2025,7,4),  # Independence Day
    date(2025,9,1),  # Labor Day
    date(2025,11,27),# Thanksgiving
    date(2025,12,25) # Christmas
}

def is_us_market_open_now(now_utc: Optional[datetime]=None) -> bool:
    """Einfache Regular-Hours-Gate (Mo–Fr, 9:30–16:00 ET, excl. Holidays)."""
    if not CONFIG.market_hours_only:
        return True
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    now_et = now_utc.astimezone(ET)
    if now_et.weekday() >= 5:
        return False
    if now_et.date() in US_HOLIDAYS_2025:
        return False
    t = now_et.time()
    return (t >= datetime(2000,1,1,9,30,tzinfo=ET).time()) and (t <= datetime(2000,1,1,16,0,tzinfo=ET).time())

# ---------- Indikatoren ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int=14) -> pd.Series:
    delta = s.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=s.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=s.index)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

def efi_func(close: pd.Series, vol: pd.Series, efi_len: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, efi_len)

# ---------- Datenprovider ----------
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
    end   = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
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
        if not bars or bars.df is None or bars.df.empty: return pd.DataFrame()
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
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    # 1) Alpaca
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=interval); return df, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")
    # 2) Stooq EOD
    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d"); return df, note
        note.update(provider="Stooq→Yahoo", detail="Stooq leer; versuche Yahoo")

    # 3) Yahoo + Retries
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}
    is_intraday = interval in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _normalize(tmp: pd.DataFrame) -> pd.DataFrame:
        if tmp is None or tmp.empty: return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except Exception: pass
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            except Exception: pass
        df["time"] = df.index
        return df.sort_index()

    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True,  threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=True)),
    ]
    last_err = None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                tmp = yf.download(**kwargs) if method=="download" else yf.Ticker(symbol).history(**kwargs)
                df = _normalize(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}"); return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # Yahoo Fallback: 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} failed"); return df, note
    except Exception as e:
        last_err = e

    # Optional Stooq Failover
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days,120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d"); return dfe, note

    print(f"[fetch_ohlcv] no data for {symbol} ({interval}, {period}). Last err: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ---------- Features ----------
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsiLen)
    out["efi"] = efi_func(out["close"], out["volume"], cfg.efiLen)
    return out

# ---------- Strategie-Logik ----------
def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]; prev = df.iloc[-2]
    rsi_val = float(last["rsi"])
    rsi_rising  = rsi_val > float(prev["rsi"])
    rsi_falling = rsi_val < float(prev["rsi"])
    efi_rising  = float(last["efi"]) > float(prev["efi"])

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price_close = float(last["close"])
    price_open  = float(last["open"])
    ts = last["time"]

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    # Bars since entry
    bars_in_trade = 0
    if st.entry_time is not None:
        since = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since)-1)

    if st.position_size == 0:
        if entry_cond:
            return {"action":"buy", "qty":1, "price":price_open, "time":str(ts),
                    "reason":"rule_entry", "sl":sl(price_open), "tp":tp(price_open)}
        return {"action":"none", "reason":"flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = bars_in_trade >= cfg.minBarsInTrade
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(STATE.avg_price)
        cur_tp = tp(STATE.avg_price)
        hit_sl = price_close <= cur_sl
        hit_tp = price_close >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell", "qty":st.position_size, "price":price_open, "time":str(ts), "reason":"rsi_exit"}
        if hit_sl:
            return {"action":"sell", "qty":st.position_size, "price":cur_sl, "time":str(ts), "reason":"stop_loss"}
        if hit_tp:
            return {"action":"sell", "qty":st.position_size, "price":cur_tp, "time":str(ts), "reason":"take_profit"}
        return {"action":"none", "reason":"hold"}

# ---------- Alpaca Trading / Positions ----------
def get_alpaca_trading_client():
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY and APCA_API_BASE_URL):
        return None
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper="paper" in APCA_API_BASE_URL)
    except Exception as e:
        print("[alpaca] trading client error:", e)
        return None

def alpaca_place_order(side: str, qty_shares: int, tif: str) -> str:
    client = get_alpaca_trading_client()
    if client is None: return "Alpaca trading client not configured."
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        tif_enum = TimeInForce.DAY if tif.lower()=="day" else TimeInForce.GTC
        req = MarketOrderRequest(
            symbol=CONFIG.symbol,
            qty=qty_shares,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif_enum
        )
        order = client.submit_order(req)
        return f"Alpaca order {order.id} {side.upper()} {qty_shares} {CONFIG.symbol} TIF={tif}"
    except Exception as e:
        return f"[alpaca] order error: {e}"

def alpaca_positions_report() -> str:
    client = get_alpaca_trading_client()
    if client is None: return "Alpaca trading client not configured."
    try:
        poss = client.get_all_positions()
        if not poss: return "Keine offenen Alpaca-Positionen."
        lines = ["📦 Alpaca Positionen:"]
        for p in poss:
            lines.append(f"- {p.symbol}: {p.qty} @ {p.avg_entry_price} (market={p.current_price}) PnL={p.unrealized_pl} ({p.unrealized_plpc})")
        return "\n".join(lines)
    except Exception as e:
        return f"[alpaca] positions error: {e}"

def compute_shares_from_config(price: float) -> int:
    if CONFIG.qty_mode.lower()=="shares":
        return max(1, int(round(CONFIG.qty_value)))
    # usd
    shares = int(max(1, CONFIG.qty_value // max(0.01, price)))
    return shares

# ---------- Telegram Helper ----------
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text if text.strip() else "ℹ️ (leer)")
    except Exception as e:
        print("send error:", e)

def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if prov.startswith("Yahoo (Fallback"): return f"📡 Daten: {prov} – {det}"
    if "Stooq" in prov:                    return f"📡 Daten: {prov} – {det} (nur Daily)"
    return f"📡 Daten: {prov} – {det}"

# ---------- Telegram Commands ----------
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg\n"
        "/set key=value …  (z.B. /set data_provider=alpaca interval=1h poll_minutes=10)\n"
        "/run, /bt 180\n"
        "/timer on|off, /timerstatus\n"
        "/ind 5, /pos\n"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        f"📊 Status\nSymbol: {CONFIG.symbol} {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}\n"
        f"Timer: {'ON' if CONFIG.timer_enabled else 'OFF'} alle {CONFIG.poll_minutes}m\n"
        f"US-Regular-Hours-Gate: {'ON' if CONFIG.market_hours_only else 'OFF'}\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Last: {STATE.last_status}\nDatenquelle: {CONFIG.data_provider}"
    )

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1); k=k.strip(); v=v.strip()
    mapping = {"sl":"slPerc","tp":"tpPerc","samebar":"allowSameBarExit","cooldown":"minBarsInTrade"}
    k = mapping.get(k,k)
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
            "Nutze: /set key=value [key=value] …\n"
            "z.B. /set data_provider=alpaca interval=1h poll_minutes=10 market_hours_only=true\n"
            "/set qty_mode=usd qty_value=1000 tif=day"
        ); return
    msgs, errs = [], []
    for a in context.args:
        if "=" not in a: errs.append(f"❌ Ungültig: {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = "\n".join((["✅ Übernommen:"]+msgs)+(["\n⚠️ Probleme:"]+errs if errs else []))
    await update.message.reply_text(txt.strip())

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n"+json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_timer(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /timer on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.timer_enabled = on
    await update.message.reply_text(f"Timer = {'ON' if on else 'OFF'}")

async def cmd_timerstatus(update, context):
    await update.message.reply_text(json.dumps(TIMER.status(), indent=2, default=str))

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
    msg = _friendly_data_note(note)
    if msg: await update.message.reply_text(msg)

    fdf = build_features(df, CONFIG)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = float(row["rsi"])
        rsi_rising  = rsi_val > float(prev["rsi"])
        rsi_falling = rsi_val < float(prev["rsi"])
        efi_rising  = float(row["efi"]) > float(prev["efi"])
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
                eq *= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean()
        pf=(a[a>0].sum())/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days)) - 1)
        await update.message.reply_text(f"📈 Backtest {days}d Trades={entries}/{exits}\nWin={win*100:.1f}% PF={pf:.2f} CAGR~{cagr*100:.2f}%")
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_ind(update, context):
    n=5
    if context.args:
        try: n=int(context.args[0])
        except: pass
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten für /ind."); return
    fdf = build_features(df, CONFIG).tail(max(2,n))
    lines = ["🧮 Indikatoren (jüngste Bars):"]
    for ts, row in fdf.iterrows():
        lines.append(f"- {ts}: close={row['close']:.4f} RSI={row['rsi']:.2f} EFI={row['efi']:.2f}")
    await update.message.reply_text("\n".join(lines))

async def cmd_pos(update, context):
    await update.message.reply_text(alpaca_positions_report())

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ---------- Run & Report ----------
async def run_once_and_report(chat_id: str):
    # Marktzeit-Gate
    if not is_us_market_open_now():
        STATE.last_status = "market_closed"
        await send(chat_id, "⏰ Market closed (US Regular Hours)."); return

    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten ({note.get('provider','?')} – {note.get('detail','')})"); return

    msg = _friendly_data_note(note)
    if msg and ("Fallback" in msg or CONFIG.data_provider!="alpaca"):
        await send(chat_id, msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    # Positionsgröße
    last_price = float(fdf.iloc[-1]["close"])
    shares = compute_shares_from_config(last_price)

    if act["action"] == "buy" and CONFIG.live_enabled:
        # Lokal
        STATE.position_size = shares
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        # Alpaca-Order (optional)
        alp = alpaca_place_order("buy", shares, CONFIG.tif)
        await send(chat_id, f"🟢 LONG {shares} @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f} TP={act['tp']:.4f}\n{alp}")
        await send(chat_id, alpaca_positions_report())

    elif act["action"] == "sell" and STATE.position_size > 0 and CONFIG.live_enabled:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / max(1e-9, STATE.avg_price)
        alp = alpaca_place_order("sell", STATE.position_size, CONFIG.tif)
        await send(chat_id, f"🔴 EXIT {STATE.position_size} @ {exit_px:.4f} [{act['reason']}]\nPnL={pnl*100:.2f}%\n{alp}")
        STATE.position_size = 0; STATE.avg_price=0.0; STATE.entry_time=None
        await send(chat_id, alpaca_positions_report())
    else:
        await send(chat_id, f"ℹ️ {STATE.last_status}")

# ---------- Background Timer ----------
class BackgroundTimer:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._next_due: Optional[datetime] = None
        self._running: bool = False

    def status(self):
        return {
            "enabled": CONFIG.timer_enabled,
            "running": self._running,
            "poll_minutes": CONFIG.poll_minutes,
            "last_run": self._last_run,
            "next_due": self._next_due,
            "market_hours_only": CONFIG.market_hours_only
        }

    async def _loop(self):
        self._running = True
        try:
            while CONFIG.timer_enabled:
                now = datetime.utcnow()
                if self._next_due is None:
                    self._next_due = now + timedelta(minutes=CONFIG.poll_minutes)
                # fällig?
                if now >= self._next_due and CONFIG.live_enabled and CHAT_ID:
                    await run_once_and_report(CHAT_ID)
                    self._last_run = datetime.utcnow()
                    self._next_due = self._last_run + timedelta(minutes=CONFIG.poll_minutes)
                await asyncio.sleep(60)
        finally:
            self._running = False
            self._task = None

    def ensure_running(self):
        if CONFIG.timer_enabled and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        CONFIG.timer_enabled = False
        t = self._task
        if t and not t.done():
            t.cancel()
            try: await t
            except: pass
        self._task = None
        self._running = False

TIMER = BackgroundTimer()

# ---------- Lifespan (Telegram Polling + Timer) ----------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("timer",   cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            print("delete_webhook warn:", e)

        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED = True
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e} – retry in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        tg_running = True
        # Timer ggf. starten
        TIMER.ensure_running()

    except Exception as e:
        print("❌ Telegram startup error:", e); traceback.print_exc()

    try:
        yield
    finally:
        await TIMER.stop()
        try: await tg_app.updater.stop()
        except: pass
        try: await tg_app.stop()
        except: pass
        try: await tg_app.shutdown()
        except: pass

# ---------- FastAPI ----------
app = FastAPI(title="TQQQ Strategy + Telegram (Timer)", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "timer": CONFIG.timer_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider
    }

@app.get("/tick")
async def tick():
    if not CONFIG.live_enabled:   return {"ran": False, "reason": "live_disabled"}
    if not CHAT_ID:               return {"ran": False, "reason": "no_chat_id (/start)"}
    await run_once_and_report(CHAT_ID); return {"ran": True}

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
    return {"tg_running": tg_running, "polling_started": POLLING_STARTED}

@app.get("/timerstatus")
def timerstatus():
    return TIMER.status()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


async def cmd_timerstatus(update, context):
    await update.message.reply_text(json.dumps(TIMER.status(), indent=2, default=str))

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
    msg = _friendly_data_note(note)
    if msg: await update.message.reply_text(msg)

    fdf = build_features(df, CONFIG)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = float(row["rsi"])
        rsi_rising  = rsi_val > float(prev["rsi"])
        rsi_falling = rsi_val < float(prev["rsi"])
        efi_rising  = float(row["efi"]) > float(prev["efi"])
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
                eq *= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean()
        pf=(a[a>0].sum())/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days)) - 1)
        await update.message.reply_text(f"📈 Backtest {days}d Trades={entries}/{exits}\nWin={win*100:.1f}% PF={pf:.2f} CAGR~{cagr*100:.2f}%")
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_ind(update, context):
    n=5
    if context.args:
        try: n=int(context.args[0])
        except: pass
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten für /ind."); return
    fdf = build_features(df, CONFIG).tail(max(2,n))
    lines = ["🧮 Indikatoren (jüngste Bars):"]
    for ts, row in fdf.iterrows():
        lines.append(f"- {ts}: close={row['close']:.4f} RSI={row['rsi']:.2f} EFI={row['efi']:.2f}")
    await update.message.reply_text("\n".join(lines))

async def cmd_pos(update, context):
    await update.message.reply_text(alpaca_positions_report())

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ---------- Run & Report ----------
async def run_once_and_report(chat_id: str):
    # Marktzeit-Gate
    if not is_us_market_open_now():
        STATE.last_status = "market_closed"
        await send(chat_id, "⏰ Market closed (US Regular Hours)."); return

    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten ({note.get('provider','?')} – {note.get('detail','')})"); return

    msg = _friendly_data_note(note)
    if msg and ("Fallback" in msg or CONFIG.data_provider!="alpaca"):
        await send(chat_id, msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    # Positionsgröße
    last_price = float(fdf.iloc[-1]["close"])
    shares = compute_shares_from_config(last_price)

    if act["action"] == "buy" and CONFIG.live_enabled:
        # Lokal
        STATE.position_size = shares
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        # Alpaca-Order (optional)
        alp = alpaca_place_order("buy", shares, CONFIG.tif)
        await send(chat_id, f"🟢 LONG {shares} @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f} TP={act['tp']:.4f}\n{alp}")
        await send(chat_id, alpaca_positions_report())

    elif act["action"] == "sell" and STATE.position_size > 0 and CONFIG.live_enabled:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / max(1e-9, STATE.avg_price)
        alp = alpaca_place_order("sell", STATE.position_size, CONFIG.tif)
        await send(chat_id, f"🔴 EXIT {STATE.position_size} @ {exit_px:.4f} [{act['reason']}]\nPnL={pnl*100:.2f}%\n{alp}")
        STATE.position_size = 0; STATE.avg_price=0.0; STATE.entry_time=None
        await send(chat_id, alpaca_positions_report())
    else:
        await send(chat_id, f"ℹ️ {STATE.last_status}")

# ---------- Background Timer ----------
class BackgroundTimer:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._next_due: Optional[datetime] = None
        self._running: bool = False

    def status(self):
        return {
            "enabled": CONFIG.timer_enabled,
            "running": self._running,
            "poll_minutes": CONFIG.poll_minutes,
            "last_run": self._last_run,
            "next_due": self._next_due,
            "market_hours_only": CONFIG.market_hours_only
        }

    async def _loop(self):
        self._running = True
        try:
            while CONFIG.timer_enabled:
                now = datetime.utcnow()
                if self._next_due is None:
                    self._next_due = now + timedelta(minutes=CONFIG.poll_minutes)
                # fällig?
                if now >= self._next_due and CONFIG.live_enabled and CHAT_ID:
                    await run_once_and_report(CHAT_ID)
                    self._last_run = datetime.utcnow()
                    self._next_due = self._last_run + timedelta(minutes=CONFIG.poll_minutes)
                await asyncio.sleep(60)
        finally:
            self._running = False
            self._task = None

    def ensure_running(self):
        if CONFIG.timer_enabled and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        CONFIG.timer_enabled = False
        t = self._task
        if t and not t.done():
            t.cancel()
            try: await t
            except: pass
        self._task = None
        self._running = False

TIMER = BackgroundTimer()

# ---------- Lifespan (Telegram Polling + Timer) ----------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("timer",   cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            print("delete_webhook warn:", e)

        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED = True
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e} – retry in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        tg_running = True
        # Timer ggf. starten
        TIMER.ensure_running()

    except Exception as e:
        print("❌ Telegram startup error:", e); traceback.print_exc()

    try:
        yield
    finally:
        await TIMER.stop()
        try: await tg_app.updater.stop()
        except: pass
        try: await tg_app.stop()
        except: pass
        try: await tg_app.shutdown()
        except: pass

# ---------- FastAPI ----------
app = FastAPI(title="TQQQ Strategy + Telegram (Timer)", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "timer": CONFIG.timer_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider
    }

@app.get("/tick")
async def tick():
    if not CONFIG.live_enabled:   return {"ran": False, "reason": "live_disabled"}
    if not CHAT_ID:               return {"ran": False, "reason": "no_chat_id (/start)"}
    await run_once_and_report(CHAT_ID); return {"ran": True}

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
    return {"tg_running": tg_running, "polling_started": POLLING_STARTED}

@app.get("/timerstatus")
def timerstatus():
    return TIMER.status()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
