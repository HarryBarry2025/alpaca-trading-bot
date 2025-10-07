# alpaca_trading_bot.py
import os, json, time, asyncio, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response

# Telegram (PTB v20.7)
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler, filters
)
from telegram.error import Conflict, BadRequest

# ========= ENV =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL        = os.getenv("BASE_URL", "")  # ungenutzt im Polling-Modus
WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret-path")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "")  # optional

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# Alpaca ENV (optional, nur für Positions-/Order-Funktionen)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

# ========= Strategy Config / State =========
class StratConfig:
    symbol: str = "TQQQ"
    interval: str = "1h"     # '1d' oder '1h'
    lookback_days: int = 365

    # Inputs (Pine-like)
    rsiLen: int = 12
    rsiLow: float = 0.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk
    slPerc: float = 1.0
    tpPerc: float = 400.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Engine
    poll_minutes: int = 10
    live_enabled: bool = False
    use_background_timer: bool = False   # optional interner Timer

    # Data Engine
    data_provider: str = "yahoo"          # "yahoo", "stooq_eod", "alpaca"
    yahoo_retries: int = 3
    yahoo_backoff_sec: float = 2.0
    allow_stooq_fallback: bool = True

    # Execution / Bars
    strict_closed_bar: bool = True        # Signal nur auf fertiger Bar
    execution: str = "next_open"          # "next_open" oder "close"
    session_filter: bool = True           # nur NYSE Regular Session (Mo–Fr 09:30–16:00)

CONFIG = StratConfig()

class StratState:
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_action_bar_index: Optional[int] = None
    last_status: str = "idle"

STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ========= Indicators =========
def rma(s: pd.Series, length: int) -> pd.Series:
    # Wilder's RMA
    return s.ewm(alpha=1/length, adjust=False).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = rma(up, length)
    roll_down = rma(down, length)
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

def efi_func(close: pd.Series, vol: pd.Series, efi_len: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return raw.ewm(span=efi_len, adjust=False).mean()

# ========= Data Providers =========
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
    elif interval == "15m":
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
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] fetch failed:", e)
        return pd.DataFrame()

def fetch_ohlcv_with_note(symbol: str, interval: str, lookback_days: int) -> Tuple[pd.DataFrame, Dict[str,str]]:
    note = {"provider":"","detail":""}
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}

    # Alpaca explizit
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=f"{interval}")
            return df, note
        note.update(provider="Alpaca->Yahoo", detail="alpaca leer; versuche Yahoo")

    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        note.update(provider="Stooq->Yahoo", detail="stooq leer; versuche Yahoo")

    # Yahoo
    is_intraday = interval.lower() in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _norm(tmp: pd.DataFrame) -> pd.DataFrame:
        if tmp is None or tmp.empty:
            return pd.DataFrame()
        df = tmp.rename(columns=str.lower)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1)
            except Exception:
                pass
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

    last_err = None
    for method, kwargs in tries:
        for attempt in range(1, CONFIG.yahoo_retries+1):
            try:
                if method == "download":
                    tmp = yf.download(**kwargs)
                else:
                    tmp = yf.Ticker(symbol).history(**kwargs)
                df = _norm(tmp)
                if not df.empty:
                    note.update(provider="Yahoo", detail=f"{interval} period={period}")
                    return df, note
            except Exception as e:
                last_err = e
            time.sleep(CONFIG.yahoo_backoff_sec * (2**(attempt-1)))

    # Yahoo Fallback 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _norm(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] no data for {symbol} ({interval}, {period}). last_err={last_err}")
    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Helpers =========
def filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    try:
        ny = df.index.tz_convert("America/New_York")
    except Exception:
        ny = df.index.tz_localize("America/New_York")
    # nur Werktage
    df = df[(ny.weekday < 5)]
    # nur 09:30–16:00
    idx = ny.indexer_between_time("09:30", "16:00", include_end=True)
    return df.iloc[idx]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_wilder(out["close"], CONFIG.rsiLen)
    out["efi"] = efi_func(out["close"], out["volume"], CONFIG.efiLen)
    return out

# ========= Strategy Logic =========
def evaluate_conditions(fdf: pd.DataFrame) -> Dict[str, Any]:
    """Nimmt bereits Session-gefilterte Daten; arbeitet auf fertiger Bar, sofern configured."""
    if len(fdf) < max(CONFIG.rsiLen, CONFIG.efiLen) + 3:
        return {"ok": False, "reason":"not_enough_data"}

    if CONFIG.strict_closed_bar:
        last = fdf.iloc[-2]   # letzte fertige Bar
        prev = fdf.iloc[-3]
        next_bar = fdf.iloc[-1]  # Bar, zu deren Open ggf. gehandelt wird
        exec_time = next_bar["time"]
        exec_open = float(next_bar["open"])
    else:
        last = fdf.iloc[-1]
        prev = fdf.iloc[-2]
        exec_time = last["time"]
        exec_open = float(last["open"])

    rsi_val = float(last["rsi"])
    efi_val = float(last["efi"])
    rsi_rising  = rsi_val > float(prev["rsi"])
    rsi_falling = rsi_val < float(prev["rsi"])
    efi_rising  = efi_val > float(prev["efi"])

    entry_cond = (rsi_val > CONFIG.rsiLow) and (rsi_val < CONFIG.rsiHigh) and rsi_rising and efi_rising
    exit_cond  = (rsi_val < CONFIG.rsiExit) and rsi_falling

    return {
        "ok": True,
        "rsi": rsi_val, "efi": efi_val,
        "rsi_rising": rsi_rising, "efi_rising": efi_rising,
        "rsi_falling": rsi_falling,
        "entry": entry_cond,
        "exit": exit_cond,
        "exec_time": exec_time,
        "exec_open": exec_open,
        "bar_time": last["time"],
        "bar_close": float(last["close"])
    }

def bar_logic(fdf: pd.DataFrame) -> Dict[str, Any]:
    ev = evaluate_conditions(fdf)
    if not ev.get("ok", False):
        return {"action":"none","reason":ev.get("reason","eval_failed")}

    # SL/TP helpers
    def sl(px): return px * (1 - CONFIG.slPerc/100.0)
    def tp(px): return px * (1 + CONFIG.tpPerc/100.0)

    # Für Execution-Modell
    if CONFIG.execution == "next_open":
        price_for_entry = ev["exec_open"]
        time_for_entry  = ev["exec_time"]
    else:  # "close"
        price_for_entry = ev["bar_close"]
        time_for_entry  = ev["bar_time"]

    # Cooldown
    bars_in_trade = 0
    if STATE.entry_time is not None:
        since_entry = fdf[fdf["time"] >= pd.to_datetime(STATE.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if STATE.position_size == 0:
        if ev["entry"]:
            size = 1
            return {"action":"buy","qty":size,"price":price_for_entry,"time":str(time_for_entry),
                    "reason":"rule_entry_closed_bar" if CONFIG.strict_closed_bar else "rule_entry_intrabar",
                    "sl": sl(price_for_entry), "tp": tp(price_for_entry)}
        return {"action":"none","reason":"flat_no_entry"}

    else:
        same_bar_ok = CONFIG.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= CONFIG.minBarsInTrade)
        rsi_exit_ok = ev["exit"] and same_bar_ok and cooldown_ok

        cur_sl = sl(STATE.avg_price)
        cur_tp = tp(STATE.avg_price)
        last_close = ev["bar_close"]

        hit_sl = last_close <= cur_sl
        hit_tp = last_close >= cur_tp

        if rsi_exit_ok:
            return {"action":"sell","qty":STATE.position_size,"price":price_for_entry,"time":str(time_for_entry),"reason":"rsi_exit"}
        if hit_sl:
            return {"action":"sell","qty":STATE.position_size,"price":cur_sl,"time":str(ev["bar_time"]),"reason":"stop_loss"}
        if hit_tp:
            return {"action":"sell","qty":STATE.position_size,"price":cur_tp,"time":str(ev["bar_time"]),"reason":"take_profit"}
        return {"action":"none","reason":"hold"}

# ========= Telegram helpers & handlers =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text or "ℹ️ (leer)")
    except Exception as e:
        print("send error:", e)

def _friendly_data_note(note: Dict[str,str]) -> str:
    prov = note.get("provider","")
    det  = note.get("detail","")
    if not prov: return ""
    if prov == "Yahoo":
        return f"📡 Daten: Yahoo ({det})"
    elif prov.startswith("Yahoo (Fallback"):
        return f"📡 Daten: {prov} – {det}"
    elif prov.startswith("Stooq"):
        return f"📡 Daten: {prov} – {det} (nur Daily)"
    else:
        return f"📡 Daten: {prov} – {det}"

async def fetch_frame_with_features() -> Tuple[pd.DataFrame, Dict[str,str]]:
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        return df, note
    if CONFIG.session_filter:
        df = filter_regular_session(df)
    fdf = build_features(df)
    return fdf, note

async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status – Status & Datenquelle\n"
        "/set key=value – z.B. /set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=400\n"
        "/run – Strategie einmal ausführen (EoB-Logik)\n"
        "/live on|off – Live-Loop schalten\n"
        "/cfg – aktuelle Konfiguration\n"
        "/bt 120 – Backtest über 120 Tage\n"
        "/pos – Alpaca Positionen senden (Paper/Live je nach Account)\n"
        "/ind [N] – letzte N fertige Bars mit RSI/EFI\n"
        "/sig – Detailauswertung der Entry/Exit-Bedingungen"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbol: {CONFIG.symbol}  TF: {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'}  (Timer: {'ON' if CONFIG.use_background_timer else 'OFF'}, {CONFIG.poll_minutes}m)\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Last: {STATE.last_status}\n"
        f"Daten: {CONFIG.data_provider}, SessionFilter: {CONFIG.session_filter}\n"
        f"Exec: {CONFIG.execution}, ClosedBar={CONFIG.strict_closed_bar}"
    )

def _set_from_kv(k: str, v: str) -> str:
    mapping = {
        "sl":"slPerc", "tp":"tpPerc",
        "samebar":"allowSameBarExit", "cooldown":"minBarsInTrade",
        "closed":"strict_closed_bar", "session":"session_filter",
        "exec":"execution"
    }
    attr = mapping.get(k, k)
    if not hasattr(CONFIG, attr):
        return f"❌ unbekannter Key: {k}"
    cur = getattr(CONFIG, attr)
    if isinstance(cur, bool):
        setattr(CONFIG, attr, v.lower() in ("1","true","on","yes"))
    elif isinstance(cur, int):
        setattr(CONFIG, attr, int(float(v)))
    elif isinstance(cur, float):
        setattr(CONFIG, attr, float(v))
    else:
        setattr(CONFIG, attr, v)
    return f"✓ {attr} = {getattr(CONFIG, attr)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text(
            "Nutze: /set key=value [key=value] …\n"
            "Beispiele:\n"
            "/set rsiLow=0 rsiHigh=68 rsiExit=48 sl=1 tp=400\n"
            "/set interval=1h lookback_days=365\n"
            "/set data_provider=yahoo session=true closed=true exec=next_open\n"
        )
        return
    msgs=[]
    for a in context.args:
        if "=" not in a or a.startswith("=") or a.endswith("="):
            msgs.append(f"❌ Ungültig: {a}")
            continue
        k,v = a.split("=",1)
        msgs.append(_set_from_kv(k.strip(), v.strip()))
    await update.message.reply_text("\n".join(msgs))

async def cmd_cfg(update, context):
    cfg = {
        "symbol": CONFIG.symbol,
        "interval": CONFIG.interval,
        "lookback_days": CONFIG.lookback_days,
        "rsiLen": CONFIG.rsiLen, "rsiLow": CONFIG.rsiLow, "rsiHigh": CONFIG.rsiHigh, "rsiExit": CONFIG.rsiExit,
        "efiLen": CONFIG.efiLen,
        "slPerc": CONFIG.slPerc, "tpPerc": CONFIG.tpPerc,
        "allowSameBarExit": CONFIG.allowSameBarExit, "minBarsInTrade": CONFIG.minBarsInTrade,
        "poll_minutes": CONFIG.poll_minutes, "live_enabled": CONFIG.live_enabled,
        "use_background_timer": CONFIG.use_background_timer,
        "data_provider": CONFIG.data_provider, "allow_stooq_fallback": CONFIG.allow_stooq_fallback,
        "strict_closed_bar": CONFIG.strict_closed_bar, "execution": CONFIG.execution,
        "session_filter": CONFIG.session_filter
    }
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(cfg, indent=2))

async def run_once_and_report(chat_id: str):
    fdf, note = await fetch_frame_with_features()
    if fdf.empty:
        await send(chat_id, f"❌ Keine Daten ({CONFIG.symbol}/{CONFIG.interval}) – {note.get('provider')} {note.get('detail')}")
        return
    note_msg = _friendly_data_note(note)
    if note_msg and ("Fallback" in note_msg or "Stooq" in note_msg or "Alpaca" in note_msg or CONFIG.data_provider!="yahoo"):
        await send(chat_id, note_msg)

    act = bar_logic(fdf)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    if act["action"] == "buy" and STATE.position_size == 0:
        STATE.position_size = act["qty"]
        STATE.avg_price = float(act["price"])
        STATE.entry_time = act["time"]
        await send(chat_id, f"🟢 LONG @ {STATE.avg_price:.4f} ({CONFIG.symbol})\nSL={act['sl']:.4f}  TP={act['tp']:.4f}")
    elif act["action"] == "sell" and STATE.position_size > 0:
        exit_px = float(act["price"])
        pnl = (exit_px - STATE.avg_price) / STATE.avg_price
        await send(chat_id, f"🔴 EXIT @ {exit_px:.4f} ({CONFIG.symbol}) [{act['reason']}]\nPnL={pnl*100:.2f}%")
        STATE.position_size = 0
        STATE.avg_price = 0.0
        STATE.entry_time = None
    else:
        # Diagnostik kurz dazu
        ev = evaluate_conditions(fdf)
        diag = (f"RSI={ev['rsi']:.2f} (↑{ev['rsi_rising']})  "
                f"EFI={ev['efi']:.0f} (↑{ev['efi_rising']})  "
                f"Entry={ev['entry']} Exit={ev['exit']}")
        await send(chat_id, f"ℹ️ {STATE.last_status}\n{diag}")

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
    if CONFIG.session_filter:
        df = filter_regular_session(df)
    fdf = build_features(df)

    # Backtest mit Pine-typischer Semantik: Signal auf closed bar, Entry "next_open"
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(3, len(fdf)-1):
        last, prev = fdf.iloc[i-1], fdf.iloc[i-2]  # fertige Bar i-1, davor i-2
        rsi_val = float(last["rsi"])
        efi_rise = float(last["efi"]) > float(prev["efi"])
        rsi_rise = rsi_val > float(prev["rsi"])
        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and rsi_rise and efi_rise
        exitc = (rsi_val<CONFIG.rsiExit) and (rsi_val < float(prev["rsi"]))

        if pos==0 and entry:
            avg = float(fdf.iloc[i]["open"])  # next open
            pos=1; entries+=1
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp = avg*(1+CONFIG.tpPerc/100)
            price = float(last["close"])
            stop = price<=sl; take = price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(fdf.iloc[i]["open"])
                r = (px-avg)/avg
                eq*= (1+r); R.append(r); exits+=1
                pos=0; avg=0.0

    if R:
        a = np.array(R); win=(a>0).mean()
        pf = (a[a>0].sum()) / (1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr = (eq**(365/max(1,days)) - 1)
        await update.message.reply_text(
            f"📈 Backtest {days}d\nTrades: {entries}/{exits}\n"
            f"WinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\n"
            f"{_friendly_data_note(note)}"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

# ---- NEW: /ind  (letzte N fertige Bars mit Indikatoren)
async def cmd_ind(update, context):
    N = 5
    if context.args:
        try: N = max(1, min(50, int(context.args[0])))
        except: pass
    fdf, note = await fetch_frame_with_features()
    if fdf.empty or len(fdf) < N+1:
        await update.message.reply_text("Zu wenige Daten.")
        return
    # nur fertige Bars
    g = fdf.iloc[:-1].copy().tail(N)
    g["efi_rising"] = g["efi"].diff() > 0
    g["rsi_rising"] = g["rsi"].diff() > 0

    lines = [f"Letzte {N} fertige Bars (NYSE Zeit):"]
    for _, r in g.iterrows():
        t = pd.to_datetime(r["time"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
        lines.append(f"{t}  C={r['close']:.2f}  RSI={r['rsi']:.1f} (↑{bool(r['rsi_rising'])})  "
                     f"EFI={r['efi']:.0f} (↑{bool(r['efi_rising'])})")
    await update.message.reply_text("\n".join(lines))

# ---- NEW: /sig (Detailauswertung aktuelle geschlossene Bar)
async def cmd_sig(update, context):
    fdf, note = await fetch_frame_with_features()
    if fdf.empty:
        await update.message.reply_text("Keine Daten.")
        return
    ev = evaluate_conditions(fdf)
    if not ev.get("ok", False):
        await update.message.reply_text(f"Eval-Fehler: {ev.get('reason')}")
        return
    t_bar  = pd.to_datetime(ev["bar_time"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
    t_exec = pd.to_datetime(ev["exec_time"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
    msg = (
        f"🧪 Signal-Check (geschlossene Bar)\n"
        f"Bar: {t_bar}\n"
        f"RSI={ev['rsi']:.2f}  (rising={ev['rsi_rising']})\n"
        f"EFI={ev['efi']:.0f}  (rising={ev['efi_rising']})\n"
        f"Entry={ev['entry']}  Exit={ev['exit']}\n"
        f"Ausführung ({CONFIG.execution}): {t_exec}  PX~{ev['exec_open']:.4f}\n"
        f"{_friendly_data_note({'provider':CONFIG.data_provider,'detail':CONFIG.interval})}"
    )
    await update.message.reply_text(msg)

# ---- Alpaca Positionen /pos
def _alpaca_positions() -> List[Dict[str, Any]]:
    try:
        from alpaca.trading.client import TradingClient
    except Exception as e:
        print("[alpaca] trading lib missing", e)
        return []
    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        return []
    try:
        client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=True)
        positions = client.get_all_positions()
        out=[]
        for p in positions:
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl)
            })
        return out
    except Exception as e:
        print("[alpaca] positions error:", e)
        return []

async def cmd_pos(update, context):
    pos = _alpaca_positions()
    if not pos:
        await update.message.reply_text("Keine Alpaca-Positionen (oder keine Credentials).")
        return
    lines = ["📦 Alpaca Positionen:"]
    for p in pos:
        lines.append(f"{p['symbol']}: {p['qty']} @ {p['avg_entry']:.4f}  MV={p['market_value']:.2f}  UPL={p['unrealized_pl']:.2f}")
    await update.message.reply_text("\n".join(lines))

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Background Timer =========
_timer_task: Optional[asyncio.Task] = None

async def timer_loop():
    while CONFIG.use_background_timer:
        try:
            if CONFIG.live_enabled and CHAT_ID:
                await run_once_and_report(CHAT_ID)
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(max(60, CONFIG.poll_minutes * 60))

# ========= Lifespan (Polling ohne Loop-Konflikte) =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_app, tg_running, POLLING_STARTED, _timer_task
    try:
        tg_app = ApplicationBuilder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("status",  cmd_status))
        tg_app.add_handler(CommandHandler("set",     cmd_set))
        tg_app.add_handler(CommandHandler("cfg",     cmd_cfg))
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    lambda u,c: cmd_set(u, type("C",(),{"args":["live_enabled="+("on" if c.args and c.args[0].lower() in ("on","1","true") else "off")]}))))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("pos",     cmd_pos))
        tg_app.add_handler(CommandHandler("ind",     cmd_ind))
        tg_app.add_handler(CommandHandler("sig",     cmd_sig))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()

        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht (drop_pending_updates=True)")
        except Exception as e:
            print("delete_webhook warn:", e)

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
        if CONFIG.use_background_timer and _timer_task is None:
            _timer_task = asyncio.create_task(timer_loop())
        print("🚀 Telegram POLLING aktiv")
    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        tg_running = False
        CONFIG.use_background_timer = False
        try:
            if _timer_task:
                _timer_task.cancel()
        except Exception:
            pass
        try:
            await tg_app.updater.stop()
        except Exception:
            pass
        try:
            await tg_app.stop()
        except Exception:
            pass
        try:
            await tg_app.shutdown()
        except Exception:
            pass
        POLLING_STARTED = False
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
        "timer": CONFIG.use_background_timer,
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
        return {"ran": False, "reason": "no_chat_id (use /start)"}
    await run_once_and_report(CHAT_ID)
    return {"ran": True}

@app.get("/tgstatus")
def tgstatus():
    return {"tg_running": tg_running, "polling_started": POLLING_STARTED}
