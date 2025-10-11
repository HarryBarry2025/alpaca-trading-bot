# alpaca_trading_bot.py
import os, json, time, asyncio, traceback, io, base64
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

# Telegram (PTB v20.7)
from telegram import Update, InputFile
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict, BadRequest

# -------- ENV
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # paper: https://paper-api.alpaca.markets

# -------- Strategy Config / State
class StratConfig(BaseModel):
    # Engine
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"       # "1h", "15m", "1d"
    lookback_days: int = 365
    data_provider: str = "alpaca"   # "alpaca" (default), "yahoo", "stooq_eod"
    yahoo_retries: int = 2
    yahoo_backoff_sec: float = 1.5
    allow_stooq_fallback: bool = True

    # Indicators (TV kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0       # wie gewünscht: 0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Trading / Orders
    trading_enabled: bool = False
    sizing_mode: str = "fraction"      # "fraction" | "notional" | "shares"
    position_fraction: float = 0.25    # bei fraction
    notional_usd: float = 1000.0       # bei notional
    shares: int = 1                    # bei shares
    time_in_force: str = "day"         # "day" | "gtc"

    # Timer
    poll_minutes: int = 10
    live_enabled: bool = False
    market_hours_only: bool = True

class PositionState(BaseModel):
    size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None

class StratState(BaseModel):
    positions: Dict[str, PositionState] = {}
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# -------- US Market Calendar (nyse)
def is_us_trading_time(dt_utc: Optional[datetime] = None) -> bool:
    """
    Approximiert US-Regular Hours (09:30–16:00 ET) und US-Feiertage.
    Verwendet pandas_market_calendars, fällt sonst auf einfache Uhrzeitprüfung zurück.
    """
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        now = dt_utc or datetime.now(timezone.utc)
        start = (now - timedelta(days=2)).date()
        end = (now + timedelta(days=2)).date()
        sched = cal.schedule(start_date=start, end_date=end)
        # ist heute Handelstag?
        today = now.astimezone(timezone.utc).date()
        if sched.empty or pd.Timestamp(today) not in sched.index:
            return False
        # Session times in UTC
        open_utc = sched.loc[pd.Timestamp(today), "market_open"].to_pydatetime().replace(tzinfo=timezone.utc)
        close_utc = sched.loc[pd.Timestamp(today), "market_close"].to_pydatetime().replace(tzinfo=timezone.utc)
        return open_utc <= now <= close_utc
    except Exception:
        # Fallback: Uhrzeitfenster 13:30–20:00 UTC (entspricht 9:30–16:00 ET Standard, ohne DST-Korrektur)
        now = dt_utc or datetime.now(timezone.utc)
        hhmm = now.hour*60 + now.minute
        return 13*60+30 <= hhmm <= 20*60

# -------- Indicators (TV-kompatibel)
def rma(series: pd.Series, length: int) -> pd.Series:
    """TV RMA (Wilder's) mit alpha=1/length; implementiert als EWM(alpha=1/length, adjust=False)"""
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()

def rsi_tv(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = rma(up, length)
    roll_down = rma(down, length)
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def efi_tv(close: pd.Series, vol: pd.Series, length: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, length)

# -------- Data Providers
from urllib.parse import quote

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # flatten if symbol-level present
        try:
            first = df.columns.levels[0][0]
            df = df.xs(first, axis=1)
        except Exception:
            pass
    df = df.rename(columns=str.lower)
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df["time"] = df.index
    return df

def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    stooq_sym = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={quote(stooq_sym)}&i=d"
    try:
        dfe = pd.read_csv(url)
        if dfe is None or dfe.empty:
            return pd.DataFrame()
        dfe.columns = [c.lower() for c in dfe.columns]
        dfe["date"] = pd.to_datetime(dfe["date"])
        dfe = dfe.set_index("date").sort_index()
        if lookback_days > 0:
            dfe = dfe.iloc[-lookback_days:]
        dfe["time"] = dfe.index.tz_localize("UTC")
        dfe = dfe.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        return dfe[["open","high","low","close","volume","time"]]
    except Exception:
        return pd.DataFrame()

def fetch_alpaca_ohlcv(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] lib not available:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing api keys")
        return pd.DataFrame()

    interval = interval.lower()
    if interval in {"1h","60m"}:
        tf = TimeFrame(1, TimeFrameUnit.Hour)
    elif interval in {"15m"}:
        tf = TimeFrame(15, TimeFrameUnit.Minute)
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
        feed="iex",
        limit=10000,
        adjustment=None
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty df")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except Exception: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume",
                                "open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        df = df.sort_index()
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}

    if CONFIG.data_provider.lower() == "alpaca":
        dfa = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not dfa.empty:
            note.update(provider="Alpaca", detail=interval)
            return dfa, note
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer")

    if CONFIG.data_provider.lower() == "stooq_eod":
        dfe = fetch_stooq_daily(symbol, lookback_days)
        if not dfe.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return dfe, note
        note.update(provider="Stooq→Yahoo", detail="leer")

    # Yahoo tries
    intraday = interval.lower() in {"1m","2m","5m","15m","30m","60m","90m","1h"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"
    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=True,  threads=False)),
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
                df = _normalize_df(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}")
                    return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # Yahoo fallback 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days,30)}d", auto_adjust=False, progress=False, threads=False)
        df = _normalize_df(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} failed")
            return df, note
    except Exception as e:
        last_err = e

    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] empty for {symbol} ({interval}, period={period}). last_err={last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"no data"}

# -------- Features / Strategy
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv(out["close"], cfg.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], cfg.efiLen)
    return out

def bar_logic(df: pd.DataFrame, cfg: StratConfig, st_pos: PositionState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data"}

    last = df.iloc[-1]; prev = df.iloc[-2]
    rsi_val = float(last["rsi"])
    rsi_rising  = last["rsi"] > prev["rsi"]
    rsi_falling = last["rsi"] < prev["rsi"]
    efi_rising  = last["efi"] > prev["efi"]

    entry = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exitc = (rsi_val < cfg.rsiExit) and rsi_falling

    price_o = float(last["open"])
    price_c = float(last["close"])
    ts = last["time"]

    def sl(px): return px*(1 - cfg.slPerc/100.0)
    def tp(px): return px*(1 + cfg.tpPerc/100.0)

    if st_pos.size == 0:
        if entry:
            return {"action":"buy","qty":1,"price":price_o,"time":str(ts),"reason":"rule_entry",
                    "sl":sl(price_o),"tp":tp(price_o),
                    "ind":{"rsi":rsi_val,"efi":float(last['efi'])}}
        return {"action":"none","reason":"flat_no_entry","ind":{"rsi":rsi_val,"efi":float(last['efi'])}}
    else:
        # bars in trade (approx)
        bars_in_trade = 0
        if st_pos.entry_time is not None:
            since = df[df["time"] >= pd.to_datetime(st_pos.entry_time, utc=True)]
            bars_in_trade = max(0, len(since)-1)
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exitc and same_bar_ok and cooldown_ok

        cur_sl = sl(st_pos.avg_price); cur_tp = tp(st_pos.avg_price)
        hit_sl = price_c <= cur_sl
        hit_tp = price_c >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","qty":st_pos.size,"price":price_o,"time":str(ts),"reason":"rsi_exit",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi'])}}
        if hit_sl:
            return {"action":"sell","qty":st_pos.size,"price":cur_sl,"time":str(ts),"reason":"stop_loss",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi'])}}
        if hit_tp:
            return {"action":"sell","qty":st_pos.size,"price":cur_tp,"time":str(ts),"reason":"take_profit",
                    "ind":{"rsi":rsi_val,"efi":float(last['efi'])}}
        return {"action":"none","reason":"hold","ind":{"rsi":rsi_val,"efi":float(last['efi'])}}

# -------- Alpaca trading (orders & positions)
def _is_paper() -> bool:
    return "paper" in (APCA_API_BASE_URL or "").lower()

def _alpaca_clients():
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    trade_client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=_is_paper())
    data_client  = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    return trade_client, data_client

def place_order_alpaca(symbol: str, side: str, notional: Optional[float], qty: Optional[int]) -> str:
    """
    side: 'buy'|'sell'
    Echte Orders nur, wenn Keys vorhanden. notional ODER qty setzen.
    """
    try:
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return "⚠️ Alpaca Keys fehlen – keine Order gesendet."
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        trade_client, _ = _alpaca_clients()
        tif = TimeInForce.DAY if CONFIG.time_in_force.lower()=="day" else TimeInForce.GTC
        req = MarketOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
            time_in_force=tif,
            qty=qty,
            notional=notional
        )
        order = trade_client.submit_order(req)
        return f"📝 Alpaca Order: {order.id} {side.upper()} {symbol} tif={CONFIG.time_in_force}"
    except Exception as e:
        return f"❌ Alpaca Order-Fehler: {e}"

def alpaca_positions_text() -> str:
    try:
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return "⚠️ Alpaca Keys fehlen – keine Positionsabfrage."
        from alpaca.trading.client import TradingClient
        trade_client, _ = _alpaca_clients()
        poss = trade_client.get_all_positions()
        if not poss:
            return "📭 Keine offenen Positionen bei Alpaca."
        lines = ["📌 Alpaca Positionen:"]
        for p in poss:
            lines.append(
                f"- {p.symbol}: {p.qty} @ {p.avg_entry_price}  (Mkt={p.current_price}, UPL={p.unrealized_pl})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Positionsfehler: {e}"

# -------- Helpers
def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if not prov: return ""
    if prov.startswith("Yahoo"): return f"📡 Daten: {prov} – {det}"
    if "Stooq" in prov: return f"📡 Daten: {prov} – {det} (nur Daily)"
    return f"📡 Daten: {prov} – {det}"

def _sizing_from_config(last_price: float, side: str) -> Tuple[Optional[float], Optional[int]]:
    mode = CONFIG.sizing_mode.lower()
    if mode == "fraction":
        # Bruchteil der Account-Equity via notional: wir approximieren hier notional = fraction * 10000
        # (Ohne Account-API. Optional: Account abrufen und Equity verwenden.)
        notional = max(1.0, CONFIG.position_fraction * 10000.0)
        return float(notional), None
    elif mode == "notional":
        return float(max(1.0, CONFIG.notional_usd)), None
    else:  # shares
        return None, int(max(1, CONFIG.shares))

def ensure_pos_state(sym: str) -> PositionState:
    if sym not in STATE.positions:
        STATE.positions[sym] = PositionState()
    return STATE.positions[sym]

# -------- Core run (one symbol)
async def run_once_for_symbol(sym: str, chat_id: Optional[str]) -> str:
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        msg = f"❌ Keine Daten für {sym} ({CONFIG.interval}). Quelle: {note.get('provider','?')} – {note.get('detail','')}"
        if chat_id: await send(chat_id, msg)
        return msg

    note_msg = _friendly_data_note(note)
    if chat_id and (note_msg and ("Fallback" in note_msg or CONFIG.data_provider!="alpaca")):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    pos = ensure_pos_state(sym)
    act = bar_logic(fdf, CONFIG, pos)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    # Telemetry: Indikatorwerte melden (alle 10 Min via Timer oder /run)
    rsi_val = act.get("ind",{}).get("rsi", float('nan'))
    efi_val = act.get("ind",{}).get("efi", float('nan'))
    info = f"ℹ️ {sym} {CONFIG.interval}  RSI={rsi_val:.2f}  EFI={efi_val:.0f}"

    if act["action"] == "buy" and pos.size == 0:
        pos.size = act["qty"]; pos.avg_price = float(act["price"]); pos.entry_time = act["time"]
        text = f"🟢 LONG {sym} @ {pos.avg_price:.4f}\nSL={act['sl']:.4f}  TP={act['tp']:.4f}\n{info}"
        if CONFIG.trading_enabled:
            notional, qty = _sizing_from_config(pos.avg_price, "buy")
            text += "\n" + place_order_alpaca(sym, "buy", notional, qty)
        if chat_id: await send(chat_id, text)
        return text

    if act["action"] == "sell" and pos.size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - pos.avg_price) / pos.avg_price
        text = f"🔴 EXIT {sym} @ {exit_px:.4f} [{act['reason']}]\nPnL={pnl*100:.2f}%\n{info}"
        if CONFIG.trading_enabled:
            notional, qty = _sizing_from_config(pos.avg_price, "sell")
            text += "\n" + place_order_alpaca(sym, "sell", notional, qty)
        # reset
        pos.size = 0; pos.avg_price = 0.0; pos.entry_time = None
        if chat_id:
            await send(chat_id, text)
            await send(chat_id, alpaca_positions_text())
        return text

    if chat_id: await send(chat_id, f"{info}\n{STATE.last_status}")
    return f"{info} / {STATE.last_status}"

# -------- Telegram glue
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        if not text.strip(): text = "ℹ️ (leer)"
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        print("send error:", e)

async def send_photo_bytes(chat_id: str, png_bytes: bytes, caption: str=""):
    if tg_app is None: return
    try:
        bio = io.BytesIO(png_bytes); bio.name = "plot.png"; bio.seek(0)
        await tg_app.bot.send_photo(chat_id=chat_id, photo=InputFile(bio), caption=caption)
    except Exception as e:
        print("send_photo error:", e)

# ----- Commands
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/help – Hilfe\n"
        "/status – Status\n"
        "/cfg – aktuelle Konfiguration\n"
        "/set key=value … (z.B. /set interval=1h rsiHigh=68 slPerc=1 tpPerc=4)\n"
        "/run – jetzt prüfen\n"
        "/live on|off – Timer an/aus\n"
        "/timerstatus – Timer Zustand\n"
        "/bt [Tage] – Backtest\n"
        "/positions – Alpaca Positionen\n"
        "/trade on|off|status – echtes Trading an/aus/Status\n"
        "/ind – aktuelle Indikatorwerte\n"
        "/sig – aktuelles Entry/Exit-Signal (verbale Diagnose)\n"
        "/dump – letzte Bar + alle Indikatoren\n"
        "/plot – RSI & EFI Plot senden"
    )

async def cmd_help(update, context):
    await cmd_start(update, context)

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1)
    k = k.strip(); v = v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade", "tif":"time_in_force"}
    k = mapping.get(k, k)
    if not hasattr(CONFIG, k):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, k)
    if isinstance(cur, bool):
        setattr(CONFIG, k, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):
        setattr(CONFIG, k, int(float(v)))
    elif isinstance(cur, float):
        setattr(CONFIG, k, float(v))
    elif isinstance(cur, list):
        # allow comma-separated symbols
        if k == "symbols":
            syms = [s.strip().upper() for s in v.split(",") if s.strip()]
            setattr(CONFIG, k, syms or CONFIG.symbols)
        else:
            setattr(CONFIG, k, v.split(","))
    else:
        setattr(CONFIG, k, v)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Bsp: /set interval=1h rsiHigh=68 slPerc=1 tpPerc=4 data_provider=alpaca\n"
            "Bsp: /set symbols=TQQQ,QQQ"
        ); return
    msgs, errors = [], []
    for a in context.args:
        a = a.strip()
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errors.append(f"❌ Ungültig: „{a}“"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errors.append(f"❌ Fehler bei „{a}“: {e}")
    out = []
    if msgs: out += ["✅ Übernommen:"] + msgs
    if errors: out += ["\n⚠️ Probleme:"] + errors
    await update.message.reply_text("\n".join(out).strip())

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_status(update, context):
    lines = [
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} (alle {CONFIG.poll_minutes}m, MktStd: {'Ja' if CONFIG.market_hours_only else 'Nein'})",
        f"Trading: {'ON' if CONFIG.trading_enabled else 'OFF'} ({'Paper' if _is_paper() else 'Live'})",
        f"Provider: {CONFIG.data_provider}  Interval: {CONFIG.interval}",
        f"Symbols: {', '.join(CONFIG.symbols)}",
        f"Last: {STATE.last_status}",
    ]
    await update.message.reply_text("📊 Status\n" + "\n".join(lines))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on | /live off"); return
    on = context.args[0].lower() in {"on","1","true","start"}
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_timerstatus(update, context):
    await update.message.reply_text(json.dumps(TIMER.dict(), indent=2))

async def cmd_run(update, context):
    chat = str(update.effective_chat.id)
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, chat)

async def cmd_positions(update, context):
    await update.message.reply_text(alpaca_positions_text())

async def cmd_trade(update, context):
    args = [a.lower() for a in context.args] if context.args else []
    if not args or args[0] in {"status","state"}:
        await update.message.reply_text(
            "🧩 Trading-Status\n"
            f"Trading: {'ON' if CONFIG.trading_enabled else 'OFF'}\n"
            f"Modus: {'Paper' if _is_paper() else 'Live'}\n"
            f"Sizing: {CONFIG.sizing_mode} (fraction={CONFIG.position_fraction}, notional={CONFIG.notional_usd}, shares={CONFIG.shares})\n"
            f"TIF: {CONFIG.time_in_force.upper()}\n"
            f"Nur US-Handelszeiten: {'Ja' if CONFIG.market_hours_only else 'Nein'}"
        ); return
    if args[0] not in {"on","off"}:
        await update.message.reply_text("Nutze: /trade on | /trade off | /trade status"); return
    CONFIG.trading_enabled = (args[0]=="on")
    await update.message.reply_text(("✅ Trading AKTIVIERT." if CONFIG.trading_enabled else "⛔ Trading DEAKTIVIERT.")
                                    + f" Modus: {'Paper' if _is_paper() else 'Live'}")

async def cmd_ind(update, context):
    # Zeigt aktuelle Indikatorwerte des ersten Symbols
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    fdf = build_features(df, CONFIG)
    last = fdf.iloc[-1]
    await update.message.reply_text(
        f"🔎 {sym} {CONFIG.interval}\n"
        f"RSI({CONFIG.rsiLen})={last['rsi']:.2f}\nEFI({CONFIG.efiLen})={last['efi']:.0f}"
    )

async def cmd_sig(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    fdf = build_features(df, CONFIG)
    pos = ensure_pos_state(sym)
    act = bar_logic(fdf, CONFIG, pos)
    await update.message.reply_text(f"🧭 Signal {sym}: {act['action']} ({act['reason']})\n{act.get('ind',{})}")

async def cmd_dump(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    fdf = build_features(df, CONFIG)
    last = fdf.iloc[-1]
    payload = {
        "symbol": sym, "interval": CONFIG.interval, "provider": note.get("provider",""),
        "time": str(last["time"]),
        "open": float(last["open"]), "high": float(last["high"]), "low": float(last["low"]), "close": float(last["close"]),
        "volume": float(last["volume"]),
        "rsi": float(last["rsi"]), "efi": float(last["efi"]),
        "slPerc": CONFIG.slPerc, "tpPerc": CONFIG.tpPerc
    }
    await update.message.reply_text("🧾 Dump\n" + json.dumps(payload, indent=2))

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, days)
    if df.empty:
        await update.message.reply_text(f"❌ Keine Daten ({note})."); return
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
                r = (px-avg)/avg
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if R:
        a = np.array(R); win=(a>0).mean()
        pf = (a[a>0].sum()) / (1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr = (eq**(365/max(1,days)) - 1)
        await update.message.reply_text(
            f"📈 Backtest {days}d\nTrades: {entries}/{exits}\nWinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def cmd_plot(update, context):
    sym = CONFIG.symbols[0]
    df, note = fetch_ohlcv_with_note(sym, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    fdf = build_features(df, CONFIG).tail(300)  # 300 Bars für Plot
    # Plot (zwei Achsen getrennt)
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(fdf.index, fdf["close"]); ax1.set_title(f"{sym} Close")
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(fdf.index, fdf["rsi"], label=f"RSI({CONFIG.rsiLen})")
    ax2.axhline(CONFIG.rsiLow, linestyle="--"); ax2.axhline(CONFIG.rsiHigh, linestyle="--")
    ax2.set_ylim(0,100); ax2.legend(loc="upper left"); ax2.set_title(f"RSI + EFI len={CONFIG.efiLen}")
    ax2b = ax2.twinx()
    ax2b.plot(fdf.index, fdf["efi"], alpha=0.5); ax2b.set_ylabel("EFI")
    fig.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=120); plt.close(fig)
    await send_photo_bytes(str(update.effective_chat.id), buf.getvalue(), caption=f"{sym} {CONFIG.interval} – RSI/EFI")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /help für Hilfe")

# -------- Background Timer
class TimerState(BaseModel):
    enabled: bool = False
    running: bool = False
    poll_minutes: int = 10
    last_run: Optional[str] = None
    next_due: Optional[str] = None
    market_hours_only: bool = True

TIMER = TimerState(enabled=False, running=False, poll_minutes=CONFIG.poll_minutes,
                   market_hours_only=CONFIG.market_hours_only)

async def timer_loop():
    global TIMER
    while True:
        try:
            if not TIMER.enabled:
                await asyncio.sleep(2); continue
            # market hours gate
            if TIMER.market_hours_only and not is_us_trading_time():
                # schlafe 60s und versuche erneut
                TIMER.running = False
                TIMER.next_due = None
                await asyncio.sleep(60); continue
            TIMER.running = True
            now = datetime.now(timezone.utc)
            if CHAT_ID:
                for sym in CONFIG.symbols:
                    await run_once_for_symbol(sym, CHAT_ID)
            TIMER.last_run = now.isoformat()
            TIMER.next_due = (now + timedelta(minutes=TIMER.poll_minutes)).isoformat()
            await asyncio.sleep(TIMER.poll_minutes*60)
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(5)

# -------- Lifespan (Polling)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED, TIMER
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        # Handlers
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("help",    cmd_help))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("positions", cmd_positions))
        tg_app.add_handler(CommandHandler("trade",   cmd_trade))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("plot",    cmd_plot))
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
                    print("▶️ starte Polling…")
                    await tg_app.updater.start_polling(poll_interval=1.0, timeout=10.0)
                    POLLING_STARTED = True; print("✅ Polling läuft"); break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. Retry in {delay}s…"); await asyncio.sleep(delay); delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc(); await asyncio.sleep(10)

        tg_running = True
        # Timer initialisieren
        TIMER.enabled = CONFIG.live_enabled
        TIMER.poll_minutes = CONFIG.poll_minutes
        TIMER.market_hours_only = CONFIG.market_hours_only
        asyncio.create_task(timer_loop())
        print("🚀 Telegram & Timer aktiv")
    except Exception as e:
        print("❌ Telegram-Startup Fehler:", e); traceback.print_exc()

    try:
        yield
    finally:
        try:
            await tg_app.updater.stop()
        except Exception: pass
        try:
            await tg_app.stop()
        except Exception: pass
        try:
            await tg_app.shutdown()
        except Exception: pass
        print("🛑 Bot gestoppt")

# -------- FastAPI app
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
async def head_root():
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
    out = []
    for sym in CONFIG.symbols:
        msg = await run_once_for_symbol(sym, CHAT_ID)
        out.append({sym: msg})
    return {"ran": True, "details": out}

@app.get("/timerstatus")
async def http_timerstatus():
    return TIMER.dict()

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v = os.getenv(k); return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "APCA_API_KEY_ID": chk("APCA_API_KEY_ID"),
        "APCA_API_SECRET_KEY": chk("APCA_API_SECRET_KEY"),
        "APCA_API_BASE_URL": chk("APCA_API_BASE_URL"),
        "DEFAULT_CHAT_ID": chk("DEFAULT_CHAT_ID"),
    }

# optional Webhook (ungenutzt im Polling)
@app.post(f"/telegram/{WEBHOOK_SECRET}")
async def telegram_webhook(req: Request):
    if tg_app is None:
        raise HTTPException(503, "Bot not ready")
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
