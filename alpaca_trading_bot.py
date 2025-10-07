# alpaca_trading_bot.py
import os, json, time, asyncio, traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

# Telegram (python-telegram-bot v20.7)
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")
BASE_URL        = os.getenv("BASE_URL", "")  # ungenutzt (Polling-Modus)

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca (optional: für Daten + PaperTrading)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")  # Paper-API

# ========= CONFIG & STATE =========
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"       # '1h'/'1d' (Yahoo Intraday via period)
    lookback_days: int = 365

    # Pine-like Inputs (MACD bewusst entfernt)
    rsiLen: int = 12
    rsiLow: float = 0.0         # wie gewünscht
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine
    poll_minutes: int = 10
    live_enabled: bool = False

    # Data Engine
    data_provider: str = "yahoo"          # "yahoo", "stooq_eod", "alpaca"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Timer
    timer_enabled: bool = True
    market_hours_only: bool = True

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID or None

# ========= INDICATORS =========
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
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

# ========= DATA PROVIDERS =========
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
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
        return df[["open","high","low","close","volume","time"]]
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
        feed="iex",
        limit=10000
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] empty frame (prüfe feed/interval/zugang)")
            return pd.DataFrame()
        df = bars.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            try: df = df.xs(symbol, level=0)
            except: pass
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        df = df.sort_index()
        df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        if "open" not in df.columns:
            df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    # Alpaca
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=f"{interval}")
            return df, note
        note.update(provider="Alpaca → Fallback", detail="versuche Yahoo/Stooq")

    # Stooq EOD
    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        note.update(provider="Stooq → Fallback", detail="versuche Yahoo")

    # Yahoo (mit Retries/Backoff)
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(symbol, axis=1)
            except: pass
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            except Exception:
                pass
        df = df.sort_index()
        df["time"] = df.index
        # nur relevante Columns
        cols = [c for c in ["open","high","low","close","volume","time"] if c in df.columns]
        return df[cols]

    intraday = interval.lower() in {"1m","2m","5m","15m","30m","60m","90m","1h"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"
    tries = [
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=False, threads=False)),
        ("download", dict(tickers=symbol, interval=interval, period=period,
                          auto_adjust=False, progress=False, prepost=True, threads=False)),
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

    # Yahoo Fallback 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # Stooq Fallback
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] keine Daten für {symbol} ({interval}, period={period}). Last error: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)", "detail":"keine Daten"}

# ========= FEATURES & STRATEGY =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsiLen)
    out["efi"] = efi_func(out["close"], out["volume"], cfg.efiLen)
    return out

def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]; prev = df.iloc[-2]
    rsi_val = float(last["rsi"])
    rsi_rising = last["rsi"] > prev["rsi"]
    rsi_falling = last["rsi"] < prev["rsi"]
    efi_rising  = last["efi"] > prev["efi"]

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    o = float(last["open"])
    c = float(last["close"])
    ts = last["time"]

    def sl(px): return px * (1 - cfg.slPerc/100.0)
    def tp(px): return px * (1 + cfg.tpPerc/100.0)

    # Bars since entry (approx)
    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if st.position_size == 0:
        if entry_cond:
            return {"action":"buy","qty":1,"price":o,"time":str(ts),"reason":"rule_entry",
                    "sl":sl(o),"tp":tp(o),
                    "indicators":{"rsi":rsi_val,"rsi_prev":float(prev['rsi']),
                                  "efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        return {"action":"none","reason":"flat_no_entry",
                "indicators":{"rsi":rsi_val,"rsi_prev":float(prev['rsi']),
                              "efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(st.avg_price); cur_tp = tp(st.avg_price)
        hit_sl = c <= cur_sl; hit_tp = c >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","qty":st.position_size,"price":o,"time":str(ts),"reason":"rsi_exit",
                    "indicators":{"rsi":rsi_val,"rsi_prev":float(prev['rsi']),
                                  "efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        if hit_sl:
            return {"action":"sell","qty":st.position_size,"price":cur_sl,"time":str(ts),"reason":"stop_loss",
                    "indicators":{"rsi":rsi_val,"rsi_prev":float(prev['rsi']),
                                  "efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        if hit_tp:
            return {"action":"sell","qty":st.position_size,"price":cur_tp,"time":str(ts),"reason":"take_profit",
                    "indicators":{"rsi":rsi_val,"rsi_prev":float(prev['rsi']),
                                  "efi":float(last['efi']),"efi_prev":float(prev['efi'])}}
        return {"action":"none","reason":"hold",
                "indicators":{"rsi":rsi_val,"rsi_prev":float(prev['rsi']),
                              "efi":float(last['efi']),"efi_prev":float(prev['efi'])}}

# ========= MARKET HOURS / US HOLIDAYS =========
US_HOLIDAYS: List[str] = [
    # YYYY-MM-DD (ungefähre Hauptfeiertage; Börsenkalender kann abweichen)
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27",
    "2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25"
]
def is_us_holiday(now_utc: datetime) -> bool:
    d = now_utc.astimezone(timezone(timedelta(hours=-5))).date()  # ET (naiv – ohne DST)
    s = d.isoformat()
    return s in US_HOLIDAYS

def is_us_regular_hours(now_utc: datetime) -> bool:
    # Simple ET Window 09:30–16:00, mit ungefährem DST-Handling:
    # Wir schätzen ET = UTC-4 (Sommer) oder UTC-5 (Winter) grob anhand des Datumsbereichs.
    y = now_utc.year
    # grob: DST mid-Mar ... early-Nov
    dst_on = (datetime(y,3,10,tzinfo=timezone.utc) <= now_utc <= datetime(y,11,10,tzinfo=timezone.utc))
    offset = -4 if dst_on else -5
    et = now_utc.astimezone(timezone(timedelta(hours=offset)))
    wd = et.weekday()  # 0=Mo ... 6=So
    if wd >= 5:
        return False
    if is_us_holiday(now_utc):
        return False
    hm = et.hour*60 + et.minute
    open_min = 9*60 + 30
    close_min = 16*60
    return open_min <= hm <= close_min

# ========= TELEGRAM HELPERS =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text if text.strip() else "ℹ️ (leer)")
    except Exception as e:
        print("send error:", e)

# ========= ALPACA BROKER HELPERS (Positions/Orders) =========
def get_alpaca_trade_client():
    try:
        from alpaca.trading.client import TradingClient
        if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
            return None
        # paper=True nutzt automatisch die Paper-API; Base URL per ENV
        return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
    except Exception as e:
        print("[alpaca] TradingClient error:", e)
        return None

async def report_positions(chat_id: str):
    client = get_alpaca_trade_client()
    if client is None:
        await send(chat_id, "ℹ️ Alpaca nicht konfiguriert (keine API-Keys).")
        return
    try:
        positions = client.get_all_positions()
        if not positions:
            await send(chat_id, "📭 Keine offenen Alpaca-Positionen.")
            return
        lines = ["📌 Alpaca offene Positionen:"]
        for p in positions:
            # p.avg_entry_price, p.current_price, p.qty, p.symbol, p.unrealized_pl
            lines.append(
                f"- {p.symbol}: {p.qty} @ {float(p.avg_entry_price):.4f} "
                f"→ {float(p.current_price):.4f}  (UPnL {float(p.unrealized_pl):.2f}$)"
            )
        await send(chat_id, "\n".join(lines))
    except Exception as e:
        await send(chat_id, f"❌ Konnte Positionen nicht laden: {e}")

# ========= STRATEGY RUN =========
def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider",""); det = note.get("detail","")
    if prov.startswith("Yahoo"): return f"📡 Daten: {prov} ({det})"
    if "Fallback" in prov: return f"📡 Daten: {prov} – {det}"
    if "Stooq" in prov: return f"📡 Daten: {prov} – {det} (nur Daily)"
    if prov: return f"📡 Daten: {prov} – {det}"
    return ""

async def run_once_and_report(chat_id: str):
    now = datetime.now(timezone.utc)
    if CONFIG.market_hours_only and not is_us_regular_hours(now):
        await send(chat_id, "⏸️ Markt geschlossen (US Regular Hours 09:30–16:00 ET).")
        return

    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten für {CONFIG.symbol} ({CONFIG.interval}). "
                            f"Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return

    note_msg = _friendly_data_note(note)
    if note_msg and ( "Fallback" in note_msg or "Stooq" in note_msg or CONFIG.data_provider!="yahoo"):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    # Indikatoren-Status für Transparenz
    ind = act.get("indicators", {})
    ind_msg = (f"RSI={ind.get('rsi', np.nan):.2f} (prev {ind.get('rsi_prev', np.nan):.2f}) | "
               f"EFI={ind.get('efi', np.nan):.2f} (prev {ind.get('efi_prev', np.nan):.2f})")

    if act["action"] == "buy":
        STATE.position_size = act["qty"]
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        await send(chat_id, f"🟢 LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol})\n"
                            f"SL={act['sl']:.4f}  TP={act['tp']:.4f}\n{ind_msg}")
    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\n"
                            f"PnL={pnl*100:.2f}%\n{ind_msg}")
        STATE.position_size = 0
        STATE.avg_price = 0.0
        STATE.entry_time = None
    else:
        await send(chat_id, f"ℹ️ {STATE.last_status}\n{ind_msg}")

# ========= BACKTEST (einfacher EoB) =========
async def run_backtest(days: int) -> str:
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, days)
    if df.empty:
        return f"❌ Keine Daten für Backtest. Quelle: {note.get('provider','?')} – {note.get('detail','')}"
    fdf = build_features(df, CONFIG)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = float(row["rsi"])
        rsi_rising = row["rsi"] > prev["rsi"]
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
                eq *= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0
    if not R:
        return "📉 Backtest: keine abgeschlossenen Trades."
    a = np.array(R); win=(a>0).mean()
    pf = (a[a>0].sum()) / (1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
    cagr = (eq**(365/max(1,days)) - 1)
    note_msg = _friendly_data_note(note)
    return (f"{note_msg}\n"
            f"📈 Backtest {days}d\n"
            f"Trades: {entries}/{exits}\n"
            f"WinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%")

# ========= TIMER (BACKGROUND LOOP) =========
class TimerLoop:
    def __init__(self):
        self.enabled = True
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.last_run: Optional[datetime] = None
        self.next_due: Optional[datetime] = None

    async def _loop(self):
        self.running = True
        try:
            while self.enabled:
                now = datetime.now(timezone.utc)
                # planen
                self.next_due = now + timedelta(minutes=CONFIG.poll_minutes)
                # run (nur wenn live + market ok)
                if CONFIG.live_enabled:
                    if (not CONFIG.market_hours_only) or is_us_regular_hours(now):
                        if CHAT_ID:
                            try:
                                await run_once_and_report(CHAT_ID)
                                self.last_run = datetime.now(timezone.utc)
                            except Exception:
                                traceback.print_exc()
                        else:
                            print("Timer: keine CHAT_ID (nutze /start).")
                    else:
                        # optional: loggen
                        pass
                await asyncio.sleep(CONFIG.poll_minutes * 60)
        finally:
            self.running = False
            self.task = None
            self.next_due = None

    def ensure_running(self):
        if not self.enabled:
            return
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._loop())
            self.running = True

    async def stop(self):
        self.enabled = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.running = False
        self.task = None
        self.next_due = None

TIMER = TimerLoop()

# ========= TELEGRAM HANDLERS =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status – Status & Datenquelle\n"
        "/set key=value … (z.B. /set interval=1h rsiHigh=68 sl=1 tp=4)\n"
        "/run – jetzt eine Auswertung\n"
        "/live on|off – Live Modus\n"
        "/timer on|off – Hintergrund-Timer an/aus\n"
        "/timerstatus – Timer Zustand\n"
        "/bt 90 – Backtest 90 Tage\n"
        "/positions – Alpaca offene Positionen"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbol: {CONFIG.symbol}  TF: {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} | Timer: {'ON' if TIMER.enabled else 'OFF'}\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Letzte Aktion: {STATE.last_status}\n"
        f"Datenquelle: {CONFIG.data_provider}\n"
        f"US Regular Hours only: {'ON' if CONFIG.market_hours_only else 'OFF'}"
    )

def set_from_kv(kv: str) -> str:
    k, v = kv.split("=", 1)
    k = k.strip(); v = v.strip()
    mapping = {"sl":"slPerc", "tp":"tpPerc", "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade"}
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
    else:
        setattr(CONFIG, k, v)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set data_provider=yahoo  (oder alpaca|stooq_eod)\n"
            "/set poll_minutes=10 market_hours_only=true"
        )
        return
    msgs, errs = [], []
    for a in context.args:
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errs.append(f"❌ Ungültig: {a}"); continue
        try:
            msgs.append(set_from_kv(a))
        except Exception as e:
            errs.append(f"❌ Fehler bei {a}: {e}")
    out = []
    if msgs: out += ["✅ Übernommen:"] + msgs
    if errs: out += ["\n⚠️ Probleme:"] + errs
    await update.message.reply_text("\n".join(out).strip())

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    if on and CONFIG.timer_enabled:
        TIMER.enabled = True
        TIMER.ensure_running()
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

async def cmd_timer(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /timer on|off"); return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.timer_enabled = on
    TIMER.enabled = on
    if on:
        TIMER.ensure_running()
        await update.message.reply_text("Timer = ON")
    else:
        await TIMER.stop()
        await update.message.reply_text("Timer = OFF")

async def cmd_timerstatus(update, context):
    def fmt(ts: Optional[datetime]) -> str:
        return ts.isoformat() if ts else "null"
    await update.message.reply_text(json.dumps({
        "enabled": TIMER.enabled,
        "running": TIMER.running,
        "poll_minutes": CONFIG.poll_minutes,
        "last_run": fmt(TIMER.last_run),
        "next_due": fmt(TIMER.next_due),
        "market_hours_only": CONFIG.market_hours_only
    }, indent=2))

async def cmd_run(update, context):
    await run_once_and_report(str(update.effective_chat.id))

async def cmd_bt(update, context):
    days = 180
    if context.args:
        try: days = int(context.args[0])
        except: pass
    msg = await run_backtest(days)
    await update.message.reply_text(msg)

async def cmd_positions(update, context):
    await report_positions(str(update.effective_chat.id))

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= LIFESPAN (PTB Polling + Timer-Autostart) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_status))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("timer",   cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus", cmd_timerstatus))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("positions", cmd_positions))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht (drop_pending_updates=True)")
        except Exception as e:
            print("delete_webhook warn:", e)

        # Polling starten (mit Konflikt-Retry)
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

        # Timer beim Deploy automatisch starten, wenn so konfiguriert
        if CONFIG.timer_enabled and CONFIG.live_enabled:
            TIMER.enabled = True
            TIMER.ensure_running()

    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        tg_running = False
        try:
            await tg_app.updater.stop()
        except Exception: pass
        try:
            await tg_app.stop()
        except Exception: pass
        try:
            await tg_app.shutdown()
        except Exception: pass
        try:
            await TIMER.stop()
        except Exception: pass
        POLLING_STARTED = False
        print("🛑 Telegram POLLING gestoppt")

# ========= FASTAPI =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "timer_enabled": CONFIG.timer_enabled
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
    await run_once_and_report(CHAT_ID)
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
    return {"tg_running": tg_running, "polling_started": POLLING_STARTED}
