# alpaca_trading_bot.py
import os, json, time, asyncio, traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel

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

# Alpaca ENV (optional, nur wenn data_provider=alpaca genutzt wird)
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_API_BASE_URL   = os.getenv("APCA_API_BASE_URL", "")  # z.B. https://paper-api.alpaca.markets

# ========= Strategy Config / State =========
class StratConfig(BaseModel):
    symbol: str = "TQQQ"
    interval: str = "1h"     # '1d' oder '1h' (Yahoo Intraday via period)
    lookback_days: int = 365

    # Pine-like Inputs
    rsiLen: int = 12
    rsiLow: float = 52.0
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    macdFast: int = 8
    macdSlow: int = 21
    macdSig: int = 11
    efiLen: int = 11

    # Risk
    slPerc: float = 2.0
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
    allow_stooq_fallback: bool = True     # bei Yahoo-Fehler auf Stooq (daily) fallen?

class StratState(BaseModel):
    position_size: int = 0
    avg_price: float = 0.0
    entry_time: Optional[str] = None
    last_action_bar_index: Optional[int] = None
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID if DEFAULT_CHAT_ID else None

# ========= Indicators =========
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

def macd(s: pd.Series, fast: int, slow: int, sig: int):
    fast_ema = ema(s, fast)
    slow_ema = ema(s, slow)
    line = fast_ema - slow_ema
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def efi_func(close: pd.Series, vol: pd.Series, efi_len: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, efi_len)

# ========= Data Providers =========
from urllib.parse import quote

def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    """EOD (Daily) von Stooq (CSV). Nur Tagesdaten."""
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
    """
    Intraday/Daily via Alpaca Market Data (benötigt gültige Alpaca-ENV).
    Unterstützte Intervalle: '1Min','5Min','15Min','1Hour','1Day'.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] library not available:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY and APCA_API_BASE_URL):
        print("[alpaca] missing credentials/base url")
        return pd.DataFrame()

    # Map Interval
    interval_map = {
        "1m":"1Min","2m":"1Min","5m":"5Min","15m":"15Min","30m":"30Min",
        "60m":"1Hour","90m":"1Hour","1h":"1Hour","1d":"1Day","1D":"1Day"
    }
    alp_tf = interval_map.get(interval, "1Hour")
    if alp_tf == "30Min":
        # Alpaca hat 30Min als 30Min nicht im Enum – wir approximieren mit 15Min & resample, hier wählen wir 15Min:
        alp_tf = "15Min"

    # TimeFrame Objekt
    if alp_tf.endswith("Min"):
        unit = TimeFrameUnit.Minute
        mult = int(alp_tf.replace("Min",""))
    elif alp_tf == "1Hour":
        unit = TimeFrameUnit.Hour; mult = 1
    else:
        unit = TimeFrameUnit.Day; mult = 1

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 30))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(mult, unit),
        start=start,
        end=end,
        adjustment=None
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            return pd.DataFrame()
        df = bars.df.copy()
        # MultiIndex (symbol, timestamp) → wir ziehen Symbolebene ab:
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0)
            except Exception:
                pass
        df = df.rename(columns={
            "open":"open","high":"high","low":"low","close":"close","volume":"volume"
        })
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
    """
    Robuster Download:
      - Provider == yahoo: Intraday via period, Retries + Backoff, Ticker().history-Fallback,
                           letzter Fallback Yahoo 1d; optional Stooq EOD
      - Provider == stooq_eod: Stooq daily
      - Provider == alpaca: Alpaca Market Data
    Gibt (df, note) zurück – note beschreibt Quelle/Fallbacks freundlich.
    """
    note = {"provider":"","detail":""}
    intraday_set = {"1m","2m","5m","15m","30m","60m","90m","1h"}

    # Alpaca-Provider
    if CONFIG.data_provider.lower() == "alpaca":
        df = fetch_alpaca_ohlcv(symbol, interval, lookback_days)
        if not df.empty:
            note.update(provider="Alpaca", detail=f"{interval}")
            return df, note
        # Fallback auf Yahoo
        note.update(provider="Alpaca→Yahoo", detail="Alpaca leer; versuche Yahoo")

    # Stooq-Provider (EOD)
    if CONFIG.data_provider.lower() == "stooq_eod":
        df = fetch_stooq_daily(symbol, lookback_days)
        if not df.empty:
            note.update(provider="Stooq EOD", detail="1d")
            return df, note
        # Fallback auf Yahoo
        note.update(provider="Stooq→Yahoo", detail="Stooq leer; versuche Yahoo")

    # Yahoo-Provider
    is_intraday = interval in intraday_set
    period = f"{min(lookback_days, 730)}d" if is_intraday else f"{lookback_days}d"

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
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
            # Backoff
            sleep_s = CONFIG.yahoo_backoff_sec * (2 ** (attempt - 1))
            time.sleep(sleep_s)

    # Yahoo Fallback: 1d
    try:
        tmp = yf.download(symbol, interval="1d", period=f"{max(lookback_days, 30)}d",
                          auto_adjust=False, progress=False, prepost=False, threads=False)
        df = _normalize(tmp)
        if not df.empty:
            note.update(provider="Yahoo (Fallback 1d)", detail=f"intraday {interval} fehlgeschlagen")
            return df, note
    except Exception as e:
        last_err = e

    # Optional Stooq-Fallback
    if CONFIG.allow_stooq_fallback:
        dfe = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not dfe.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return dfe, note

    print(f"[fetch_ohlcv] keine Daten für {symbol} ({interval}, period={period}). Last error: {last_err}")
    return pd.DataFrame(), {"provider":"(leer)", "detail":"keine Daten"}

# ========= Feature-Builder =========
def build_features(df: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsiLen)
    macd_line, macd_sig, _ = macd(out["close"], cfg.macdFast, cfg.macdSlow, cfg.macdSig)
    out["macd_line"], out["macd_sig"] = macd_line, macd_sig
    out["efi"] = efi_func(out["close"], out["volume"], cfg.efiLen)
    return out

# ========= Strategy Logic =========
def bar_logic(df: pd.DataFrame, cfg: StratConfig, st: StratState) -> Dict[str, Any]:
    if df.empty or len(df) < max(cfg.rsiLen, cfg.macdSlow, cfg.efiLen) + 3:
        return {"action": "none", "reason": "not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_val = last["rsi"]
    macd_above = (last["macd_line"] > last["macd_sig"])
    rsi_rising = (last["rsi"] > prev["rsi"])
    rsi_falling = (last["rsi"] < prev["rsi"])
    efi_rising = (last["efi"] > prev["efi"])

    entry_cond = (rsi_val > cfg.rsiLow) and (rsi_val < cfg.rsiHigh) and rsi_rising and efi_rising and macd_above
    exit_cond  = (rsi_val < cfg.rsiExit) and rsi_falling

    price = float(last["close"])
    o = float(last["open"])
    ts = last["time"]

    def sl(price_entry): return price_entry * (1 - cfg.slPerc/100.0)
    def tp(price_entry): return price_entry * (1 + cfg.tpPerc/100.0)

    # Bars since entry (approx.)
    bars_in_trade = 0
    if st.entry_time is not None:
        since_entry = df[df["time"] >= pd.to_datetime(st.entry_time, utc=True)]
        bars_in_trade = max(0, len(since_entry)-1)

    if st.position_size == 0:
        if entry_cond:
            size = 1
            return {"action": "buy", "qty": size, "price": o, "time": str(ts), "reason": "rule_entry",
                    "sl": sl(o), "tp": tp(o)}
        else:
            return {"action": "none", "reason": "flat_no_entry"}
    else:
        same_bar_ok = cfg.allowSameBarExit or (bars_in_trade > 0)
        cooldown_ok = (bars_in_trade >= cfg.minBarsInTrade)
        rsi_exit_ok = exit_cond and same_bar_ok and cooldown_ok

        cur_sl = sl(st.avg_price)
        cur_tp = tp(st.avg_price)

        hit_sl = price <= cur_sl
        hit_tp = price >= cur_tp

        if rsi_exit_ok:
            return {"action": "sell", "qty": st.position_size, "price": o, "time": str(ts), "reason": "rsi_exit"}
        if hit_sl:
            return {"action": "sell", "qty": st.position_size, "price": cur_sl, "time": str(ts), "reason": "stop_loss"}
        if hit_tp:
            return {"action": "sell", "qty": st.position_size, "price": cur_tp, "time": str(ts), "reason": "take_profit"}

        return {"action": "none", "reason": "hold"}

# ========= Telegram helpers & handlers =========
tg_app: Optional[Application] = None
tg_running = False
POLLING_STARTED = False  # Singleton-Schutz

async def send(chat_id: str, text: str):
    if tg_app is None: return
    try:
        if not text.strip():
            text = "ℹ️ (leer)"
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except BadRequest as e:
        # Falls trotzdem leer oder zu lang etc.
        print("send badrequest:", e)
    except Exception as e:
        print("send error:", e)

async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Bot verbunden.\n"
        "Befehle:\n"
        "/status – zeigt Status\n"
        "/set key=value … – z.B. /set rsiLow=52 rsiHigh=68 sl=2 tp=4\n"
        "/run – einen Live-Check jetzt ausführen\n"
        "/live on|off – Live-Loop schalten\n"
        "/cfg – aktuelle Konfiguration\n"
        "/bt 90 – Backtest über 90 Tage\n"
    )

async def cmd_status(update, context):
    await update.message.reply_text(
        f"📊 Status\n"
        f"Symbol: {CONFIG.symbol} {CONFIG.interval}\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} (alle {CONFIG.poll_minutes}m)\n"
        f"Pos: {STATE.position_size} @ {STATE.avg_price:.4f} (seit {STATE.entry_time})\n"
        f"Last: {STATE.last_status}\n"
        f"Datenquelle: {CONFIG.data_provider}"
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
            "/set rsiLow=52 rsiHigh=68 sl=2 tp=4\n"
            "/set interval=1h lookback_days=365\n"
            "/set data_provider=yahoo allow_stooq_fallback=true\n"
            "/set data_provider=alpaca   # benötigt Alpaca ENV"
        )
        return

    msgs, errors = [], []
    for a in context.args:
        a = a.strip()
        if "=" not in a or a.startswith("=") or a.endswith("="):
            errors.append(f"❌ Ungültig: „{a}“ (erwarte key=value)")
            continue
        try:
            msgs.append(set_from_kv(a))
        except Exception as e:
            errors.append(f"❌ Fehler bei „{a}“: {e}")

    out = []
    if msgs:
        out.append("✅ Übernommen:")
        out.extend(msgs)
    if errors:
        out.append("\n⚠️ Probleme:")
        out.extend(errors)
    if not out:
        out = [
            "❌ Keine gültigen key=value-Paare erkannt.",
            "Beispiele:",
            "/set rsiLow=52 rsiHigh=68 sl=2 tp=4",
            "/set interval=1h lookback_days=365",
        ]
    await update.message.reply_text("\n".join(out).strip())

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

async def cmd_live(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /live on oder /live off")
        return
    on = context.args[0].lower() in ("on","1","true","start")
    CONFIG.live_enabled = on
    await update.message.reply_text(f"Live = {'ON' if on else 'OFF'}")

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

async def run_once_and_report(chat_id: str):
    df, note = fetch_ohlcv_with_note(CONFIG.symbol, CONFIG.interval, CONFIG.lookback_days)
    if df.empty:
        await send(chat_id, f"❌ Keine Daten für {CONFIG.symbol} ({CONFIG.interval}). "
                            f"Quelle: {note.get('provider','?')} – {note.get('detail','')}")
        return

    # Nutzerfreundlicher Hinweis zur Datenlage (nur senden, wenn Fallback oder nicht Standard)
    note_msg = _friendly_data_note(note)
    if note_msg and ( "Fallback" in note_msg or "Stooq" in note_msg or "Alpaca" in note_msg or CONFIG.data_provider!="yahoo"):
        await send(chat_id, note_msg)

    fdf = build_features(df, CONFIG)
    act = bar_logic(fdf, CONFIG, STATE)
    STATE.last_status = f"{act['action']} ({act['reason']})"

    if act["action"] == "buy":
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
        await send(chat_id, f"ℹ️ {STATE.last_status}")

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
    # Hinweis zur Datenquelle (freundlich)
    note_msg = _friendly_data_note(note)
    if note_msg:
        await update.message.reply_text(note_msg)

    fdf = build_features(df, CONFIG)

    # simpler EoB-Backtest
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2, len(fdf)):
        row, prev = fdf.iloc[i], fdf.iloc[i-1]
        rsi_val = row["rsi"]
        macd_above = row["macd_line"] > row["macd_sig"]
        rsi_rising = row["rsi"] > prev["rsi"]
        rsi_falling = row["rsi"] < prev["rsi"]
        efi_rising  = row["efi"] > prev["efi"]
        entry = (rsi_val>CONFIG.rsiLow) and (rsi_val<CONFIG.rsiHigh) and rsi_rising and efi_rising and macd_above
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
            f"📈 Backtest {days}d\nTrades: {entries}/{exits} (open trades nicht gezählt)\n"
            f"WinRate={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%"
        )
    else:
        await update.message.reply_text("📉 Backtest: keine abgeschlossenen Trades.")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= Lifespan (Polling, PTB 20.7, ohne Loop-Konflikte) =========
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
        tg_app.add_handler(CommandHandler("run",     cmd_run))
        tg_app.add_handler(CommandHandler("live",    cmd_live))
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await tg_app.initialize()
        await tg_app.start()

        # Sicherstellen, dass KEIN Webhook aktiv ist, alte Updates verwerfen (verhindert Doppelverarbeitung)
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=True)
            print("ℹ️ Webhook gelöscht (drop_pending_updates=True)")
        except Exception as e:
            print("delete_webhook warn:", e)

        # Polling nur einmal starten
        if not POLLING_STARTED:
            delay = 5
            while True:
                try:
                    print("▶️ starte Polling…")
                    await tg_app.updater.start_polling(
                        poll_interval=1.0,
                        timeout=10.0
                    )
                    POLLING_STARTED = True
                    print("✅ Polling läuft")
                    break
                except Conflict as e:
                    print(f"⚠️ Conflict: {e}. Läuft noch eine andere Instanz? Retry in {delay}s…")
                    await asyncio.sleep(delay)
                    delay = min(delay*2, 60)
                except Exception:
                    traceback.print_exc()
                    await asyncio.sleep(10)

        tg_running = True
        print("🚀 Telegram POLLING aktiv")
    except Exception as e:
        print("❌ Fehler beim Telegram-Startup:", e)
        traceback.print_exc()

    try:
        yield
    finally:
        tg_running = False
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

# ========= FastAPI App =========
app = FastAPI(title="TQQQ Strategy + Telegram", lifespan=lifespan)

# ========= HEALTH & DIAGNOSE ROUTES =========
@app.get("/")
async def root():
    return {
        "ok": True,
        "live": CONFIG.live_enabled,
        "symbol": CONFIG.symbol,
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
    await run_once_and_report(CHAT_ID)
    return {"ran": True}

@app.get("/envcheck")
def envcheck():
    def chk(k):
        v = os.getenv(k)
        return {"present": bool(v), "len": len(v) if v else 0}
    return {
        "TELEGRAM_BOT_TOKEN": chk("TELEGRAM_BOT_TOKEN"),
        "BASE_URL": chk("BASE_URL"),
        "TELEGRAM_WEBHOOK_SECRET": chk("TELEGRAM_WEBHOOK_SECRET"),
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

# Optional: Webhook-Route beibehalten (nicht genutzt im Polling-Modus, aber schadet nicht)
@app.post(f"/telegram/{WEBHOOK_SECRET}")
async def telegram_webhook(req: Request):
    if tg_app is None:
        raise HTTPException(503, "Bot not ready")
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
