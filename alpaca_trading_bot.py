# alpaca_trading_bot.py
# V5.1 — Timer-Revive Patches: Heartbeat, :30 Anchoring, Grace Delay, Drop Incomplete, DEFAULT_CHAT_ID persistence

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
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.error import Conflict

warnings.filterwarnings("ignore", category=UserWarning)

# ========= ENV & Defaults =========
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_CHAT_ID = os.getenv("DEFAULT_CHAT_ID", "") or None  # <- neu: Persistenz über Deploys
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED      = os.getenv("APCA_DATA_FEED", "iex")  # sip|iex|us

ENV_ENABLE_TIMER      = os.getenv("ENABLE_TIMER", "true").lower() in ("1","true","on","yes")
ENV_POLL_MINUTES      = int(os.getenv("POLL_MINUTES", "10"))
ENV_MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() in ("1","true","on","yes")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

# ========= Config/State =========
class StratConfig(BaseModel):
    symbols: List[str] = ["TQQQ"]
    interval: str = "1h"           # unterstützt: "1h" (TV-synchron), "1d"
    lookback_days: int = 365

    # Indikatoren (TV-kompatibel)
    rsiLen: int = 12
    rsiLow: float = 0.0            # <- Default wie gewünscht
    rsiHigh: float = 68.0
    rsiExit: float = 48.0
    efiLen: int = 11

    # Risk (nur für Meldungen/Backtest-Logik hier; Live-Order-Teil optional)
    slPerc: float = 2.0
    tpPerc: float = 4.0
    allowSameBarExit: bool = False
    minBarsInTrade: int = 0

    # Timer/Live
    poll_minutes: int = ENV_POLL_MINUTES
    live_enabled: bool = True
    market_hours_only: bool = ENV_MARKET_HOURS_ONLY

    # Daten
    data_provider: str = "alpaca"  # "alpaca" (default), "yahoo", "stooq_eod"
    yahoo_retries: int = 2
    yahoo_backoff_sec: float = 1.5
    allow_stooq_fallback: bool = True

class StratState(BaseModel):
    positions: Dict[str, Dict[str, Any]] = {}
    last_status: str = "idle"

CONFIG = StratConfig()
STATE  = StratState()
CHAT_ID: Optional[str] = DEFAULT_CHAT_ID  # <- wird auf /start aktualisiert, aber existiert sofort via ENV

# ========= Indicators =========
def rsi_tv_wilder(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / max(1, length)
    roll_up = up.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def efi_tv(close: pd.Series, vol: pd.Series, length: int) -> pd.Series:
    raw = vol * (close - close.shift(1))
    return ema(raw, length)

# ========= Market Hours rough check (UTC 13:30–20:00; Mo–Fr; kleine Holiday-Liste) =========
def is_market_open_now(dt_utc: Optional[datetime] = None) -> bool:
    now = dt_utc or datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Sa/So
        return False
    hhmm = now.hour * 60 + now.minute
    return 13*60+30 <= hhmm <= 20*60  # 13:30–20:00 UTC ~ 9:30–16:00 ET

# ========= Data: Alpaca 1-Minute + TV-Sync Resample =========
def fetch_alpaca_minutes(symbol: str, lookback_days: int, prepost: bool = False) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print("[alpaca] data lib not available:", e)
        return pd.DataFrame()

    if not (APCA_API_KEY_ID and APCA_API_SECRET_KEY):
        print("[alpaca] missing API creds")
        return pd.DataFrame()

    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(lookback_days, 5))  # Minuten brauchen weniger Tage

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=APCA_DATA_FEED,
        limit=100000
    )
    try:
        bars = client.get_stock_bars(req)
        if not bars or bars.df is None or bars.df.empty:
            print("[alpaca] minutes empty")
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
        # Pre/Post filtern (RTH: 13:30–20:00 UTC)
        if not prepost:
            df = df.between_time("13:30", "20:00")
        df["time"] = df.index
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[alpaca] minutes fetch failed:", e)
        return pd.DataFrame()

def resample_to_hour_tv_sync(min_df: pd.DataFrame, drop_last_incomplete: bool = True) -> pd.DataFrame:
    """
    TV-Sync 1h: Candle endet um hh:30 UTC. Dazu wird auf 60-min Blöcke geankert auf :30.
    """
    if min_df.empty:
        return min_df
    df = min_df.copy()
    # Anchor: verschiebe Index minus 30 Min, resample 60T, dann zurück schieben
    idx = df.index
    shifted = df.copy()
    shifted.index = idx - pd.Timedelta(minutes=30)

    o = shifted["open"].resample("60T", label="right", closed="right").first()
    h = shifted["high"].resample("60T", label="right", closed="right").max()
    l = shifted["low"].resample("60T", label="right", closed="right").min()
    c = shifted["close"].resample("60T", label="right", closed="right").last()
    v = shifted["volume"].resample("60T", label="right", closed="right").sum()

    out = pd.DataFrame({"open":o, "high":h, "low":l, "close":c, "volume":v})
    # zurück auf echte Candle-Endzeit (add 30m)
    out.index = out.index + pd.Timedelta(minutes=30)
    out = out.dropna(how="any")
    if drop_last_incomplete:
        # Entferne die letzte Zeile, wenn ihre Endzeit in der Zukunft liegt
        if len(out) > 0 and out.index[-1] > datetime.now(timezone.utc):
            out = out.iloc[:-1]
        # Sicherstellen, dass die letzte Candle exakt auf :30 endet (falls der letzte Block noch nicht voll war)
        if len(out) > 0 and out.index[-1].minute != 30:
            out = out.iloc[:-1]
    out["time"] = out.index
    return out

# Yahoo & Stooq Fallbacks (Daily oder Intraday, ohne :30-Ausrichtung – nur Fallback)
from urllib.parse import quote
def fetch_stooq_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={quote(symbol.lower()+'.us')}&i=d"
    try:
        df = pd.read_csv(url)
        if df is None or df.empty: return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if lookback_days > 0:
            df = df.iloc[-lookback_days:]
        df["time"] = df.index.tz_localize("UTC")
        return df[["open","high","low","close","volume","time"]]
    except Exception as e:
        print("[stooq] fail:", e)
        return pd.DataFrame()

def fetch_yahoo(symbol: str, interval: str, lookback_days: int, prepost: bool) -> pd.DataFrame:
    intraday = interval in {"1m","2m","5m","15m","30m","60m","90m","1h"}
    period = f"{min(lookback_days, 730)}d" if intraday else f"{lookback_days}d"
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns=str.lower).sort_index()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df["time"] = df.index
        return df
    last_err=None
    tries=[
        ("download", dict(tickers=symbol, interval=interval, period=period, auto_adjust=False, progress=False, prepost=prepost, threads=False)),
        ("history",  dict(period=period, interval=interval, auto_adjust=False, prepost=prepost)),
    ]
    for m,kw in tries:
        try:
            tmp = yf.download(**kw) if m=="download" else yf.Ticker(symbol).history(**kw)
            out = _norm(tmp)
            if not out.empty: return out[["open","high","low","close","volume","time"]]
        except Exception as e:
            last_err=e
            time.sleep(1.0)
    print("[yahoo] fail:", last_err)
    return pd.DataFrame()

def fetch_ohlcv_tv_sync(symbol: str, interval: str, lookback_days: int, prepost: bool, drop_last_incomplete: bool) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    Primär: Alpaca 1m → TV-Sync 1h (Bars enden :30 UTC, RTH-only wenn prepost=False).
    Fallback: Yahoo (gleiches interval) → Stooq daily.
    """
    note={"provider":"","detail":""}

    if CONFIG.data_provider.lower()=="alpaca":
        if interval.lower()=="1h":
            mdf = fetch_alpaca_minutes(symbol, lookback_days, prepost=prepost)
            if not mdf.empty:
                hdf = resample_to_hour_tv_sync(mdf, drop_last_incomplete=drop_last_incomplete)
                if not hdf.empty:
                    note.update(provider=f"Alpaca ({APCA_DATA_FEED})", detail="1h (TV-sync :30, RTH)" if not prepost else "1h (TV-sync :30, ExtHours)")
                    return hdf, note
            note.update(provider="Alpaca→Yahoo", detail="minutes empty; trying Yahoo")
        else:
            # daily via Yahoo (Alpaca daily not TV-special here)
            pass

    # Yahoo fallback
    ydf = fetch_yahoo(symbol, "1h" if interval.lower()=="1h" else interval, lookback_days, prepost=prepost)
    if not ydf.empty:
        note.update(provider="Yahoo", detail=f"{interval} (no TV-sync)")
        # Yahoo 1h endet meist vollstündig, NICHT :30 – nur Fallback, daher Unterschiede möglich
        # Letzte incomplete streichen:
        if drop_last_incomplete and len(ydf)>0 and ydf.index[-1] > datetime.now(timezone.utc):
            ydf = ydf.iloc[:-1]
        return ydf, note

    # Stooq EOD
    if CONFIG.allow_stooq_fallback:
        sdf = fetch_stooq_daily(symbol, max(lookback_days, 120))
        if not sdf.empty:
            note.update(provider="Stooq EOD (Fallback)", detail="1d")
            return sdf, note

    return pd.DataFrame(), {"provider":"(leer)","detail":"keine Daten"}

# ========= Feature & Signals =========
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_tv_wilder(out["close"], CONFIG.rsiLen)
    out["efi"] = efi_tv(out["close"], out["volume"], CONFIG.efiLen)
    out["rsi_rising"] = out["rsi"] > out["rsi"].shift(1)
    out["rsi_falling"] = out["rsi"] < out["rsi"].shift(1)
    out["efi_rising"] = out["efi"] > out["efi"].shift(1)
    out["entry_cond"] = (out["rsi"] > CONFIG.rsiLow) & (out["rsi"] < CONFIG.rsiHigh) & out["rsi_rising"] & out["efi_rising"]
    out["exit_cond"]  = (out["rsi"] < CONFIG.rsiExit) & out["rsi_falling"]
    return out

def bar_logic_last(df: pd.DataFrame, sym: str) -> Dict[str,Any]:
    if df.empty or len(df) < max(CONFIG.rsiLen, CONFIG.efiLen) + 3:
        return {"action":"none","reason":"not_enough_data","symbol":sym}

    f = build_features(df)
    last = f.iloc[-1]
    prev = f.iloc[-2]

    action="none"; reason="hold"; px=float(last["open"])
    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    size=pos["size"]; avg=pos["avg"]

    def sl(p): return p*(1-CONFIG.slPerc/100.0)
    def tp(p): return p*(1+CONFIG.tpPerc/100.0)

    if size==0:
        if bool(last["entry_cond"]):
            action="buy"; reason="rule_entry"; px=float(last["open"])
            return {"action":action,"reason":reason,"symbol":sym,"qty":1,"px":px,"time":str(last["time"]),
                    "rsi":float(last["rsi"]),"efi":float(last["efi"]),"sl":sl(px),"tp":tp(px)}
        return {"action":"none","reason":"flat_no_entry","symbol":sym,"rsi":float(last["rsi"]),"efi":float(last["efi"])}
    else:
        cur_sl = sl(avg); cur_tp = tp(avg)
        price_close = float(last["close"])
        if bool(last["exit_cond"]):
            action="sell"; reason="rsi_exit"; px=float(last["open"])
        elif price_close <= cur_sl:
            action="sell"; reason="stop_loss"; px=cur_sl
        elif price_close >= cur_tp:
            action="sell"; reason="take_profit"; px=cur_tp
        return {"action":action,"reason":reason,"symbol":sym,"qty":size,"px":px,"time":str(last["time"]),
                "rsi":float(last["rsi"]),"efi":float(last["efi"])}

# ========= Telegram helpers =========
tg_app: Optional[Application] = None
POLLING_STARTED=False

async def send_text(chat_id: str, text: str):
    if tg_app is None: return
    if not text or not text.strip(): text="(leer)"
    try:
        await tg_app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        print("send_text error:", e)

async def send_document_bytes(chat_id: str, data: bytes, filename: str, caption: str = ""):
    if tg_app is None: return
    bio = io.BytesIO(data); bio.name = filename; bio.seek(0)
    try:
        await tg_app.bot.send_document(chat_id=chat_id, document=InputFile(bio), caption=caption)
    except Exception as e:
        print("send_document error:", e)

# ========= Single step runner (mit Heartbeat immer) =========
def friendly_note(note: Dict[str,str]) -> str:
    if not note: return ""
    prov=note.get("provider",""); det=note.get("detail","")
    if not prov: return ""
    return f"📡 Quelle: {prov} ({det})"

async def run_once_for_symbol(sym: str, *, send_signals: bool, prepost: bool):
    # Achtung: 8s Grace nach :30 (wird im Timer aufgerufen, /run nutzt kein Grace)
    df, note = fetch_ohlcv_tv_sync(
        symbol=sym,
        interval=CONFIG.interval,
        lookback_days=CONFIG.lookback_days,
        prepost=prepost,
        drop_last_incomplete=True
    )
    if df.empty:
        if send_signals and CHAT_ID:
            await send_text(CHAT_ID, f"❌ Keine Daten für {sym}. {friendly_note(note)}")
        return {"ok":False,"reason":"no_data","note":note}

    act = bar_logic_last(df, sym)
    STATE.last_status = f"{sym}: {act['action']} ({act['reason']})"

    # HEARTBEAT: immer senden
    if send_signals and CHAT_ID:
        await send_text(
            CHAT_ID,
            f"⏱ {sym} {CONFIG.interval} | rsi={act.get('rsi', float('nan')):.2f} "
            f"efi={act.get('efi', float('nan')):.2f} • {act['reason']}"
        )
        # Hinweis bei Fallback/abweichender Quelle
        if note.get("provider","").startswith(("Alpaca→","Stooq","Yahoo")):
            await send_text(CHAT_ID, friendly_note(note))

    # simple Sim-Pos zur Statusanzeige
    pos = STATE.positions.get(sym, {"size":0,"avg":0.0,"entry_time":None})
    if act["action"]=="buy" and pos["size"]==0:
        STATE.positions[sym] = {"size":act["qty"],"avg":act["px"],"entry_time":act["time"]}
        if send_signals and CHAT_ID:
            await send_text(CHAT_ID, f"🟢 LONG (sim) {sym} @ {act['px']:.4f} | SL={act.get('sl',math.nan):.4f} TP={act.get('tp',math.nan):.4f}")
    elif act["action"]=="sell" and pos["size"]>0:
        pnl = (act["px"]-pos["avg"])/max(1e-9,pos["avg"])
        STATE.positions[sym] = {"size":0,"avg":0.0,"entry_time":None}
        if send_signals and CHAT_ID:
            await send_text(CHAT_ID, f"🔴 EXIT (sim) {sym} @ {act['px']:.4f} • {act['reason']} • PnL={pnl*100:.2f}%")

    return {"ok":True,"act":act}

# ========= Timer (mit :30-Anchoring, Grace, Marktfilter, Fehler-Meldungen) =========
TIMER = {
    "enabled": ENV_ENABLE_TIMER,
    "running": False,
    "poll_minutes": CONFIG.poll_minutes,
    "last_run": None,
    "next_due": None,
    "market_hours_only": CONFIG.market_hours_only
}
TIMER_TASK: Optional[asyncio.Task] = None

def seconds_until_next_half_hour(now: Optional[datetime]=None) -> int:
    """Nächster Trigger genau bei hh:30 (für 1h TF)."""
    now = now or datetime.now(timezone.utc)
    # nächste :30 oder nächste Stunde :30
    minute = now.minute
    second = now.second
    # Ziel-Minute = 30, aber wenn schon nach :30 => nächste Stunde :30
    target = now.replace(second=0, microsecond=0)
    if minute < 30:
        target = target.replace(minute=30)
    else:
        # nächste Stunde
        target = (target + timedelta(hours=1)).replace(minute=30)
    return max(0, int((target - now).total_seconds()))

async def timer_loop():
    TIMER["running"] = True
    try:
        if TIMER["enabled"] and CHAT_ID is None:
            print("⚠️ TIMER enabled but CHAT_ID is None. Set DEFAULT_CHAT_ID or send /start once.")

        while TIMER["enabled"]:
            # Warten bis zur nächsten :30 (bei 1h). Für andere TF könntest du eine ähnliche Logik bauen.
            if CONFIG.interval.lower()=="1h":
                wait_s = seconds_until_next_half_hour()
            else:
                # Fallback: poll_minutes
                wait_s = max(5, TIMER["poll_minutes"]*60)

            # Marktfilter: wenn nur RTH erwünscht und gerade geschlossen, dann kurze Pause & retry
            if TIMER["market_hours_only"] and not is_market_open_now():
                TIMER["next_due"] = None
                await asyncio.sleep(min(wait_s, 60))  # minütlich prüfen
                continue

            # Schlaf bis due (aber nicht zu lange hängen wenn disabled wird)
            steps = max(1, wait_s // 5)
            for _ in range(steps):
                if not TIMER["enabled"]: break
                await asyncio.sleep(5)
            if not TIMER["enabled"]: break

            # Grace-Delay nach :30, damit Daten da sind
            await asyncio.sleep(8)

            now = datetime.now(timezone.utc)
            # Run alle Symbole
            for sym in CONFIG.symbols:
                try:
                    await run_once_for_symbol(sym, send_signals=True, prepost=False)
                except Exception as e:
                    print(f"[timer] run {sym} error:", e)
                    if CHAT_ID:
                        await send_text(CHAT_ID, f"⚠️ Timer-Run {sym} error: {e}")

            TIMER["last_run"] = now.isoformat()
            # nächste :30 in etwa 3600s
            TIMER["next_due"] = (now + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0).isoformat()
    finally:
        TIMER["running"] = False

# ========= Telegram Commands =========
async def cmd_start(update, context):
    global CHAT_ID
    CHAT_ID = str(update.effective_chat.id)
    await update.message.reply_text(
        "🤖 Bot verbunden.\n"
        "Befehle:\n"
        "/status, /cfg, /set key=value …, /run, /bt [tage]\n"
        "/timer on|off, /timerstatus, /timerrunnow\n"
        "/dump [csv [N]]"
    )

async def cmd_status(update, context):
    pos_lines=[f"{s}: size={p['size']} avg={p['avg']:.4f} since={p['entry_time']}" for s,p in STATE.positions.items()]
    pos_txt="\n".join(pos_lines) if pos_lines else "keine (sim)"
    await update.message.reply_text(
        "📊 Status\n"
        f"Symbols: {', '.join(CONFIG.symbols)}  TF={CONFIG.interval}\n"
        f"Provider: {CONFIG.data_provider} (Alpaca feed={APCA_DATA_FEED})\n"
        f"Live: {'ON' if CONFIG.live_enabled else 'OFF'} • Timer: {'ON' if TIMER['enabled'] else 'OFF'} "
        f"(market-hours-only={TIMER['market_hours_only']})\n"
        f"LastStatus: {STATE.last_status}\n"
        f"Sim-Pos:\n{pos_txt}"
    )

async def cmd_cfg(update, context):
    await update.message.reply_text("⚙️ Konfiguration:\n" + json.dumps(CONFIG.dict(), indent=2))

def set_from_kv(kv: str) -> str:
    k,v = kv.split("=",1); k=k.strip(); v=v.strip()
    mapping={"sl":"slPerc","tp":"tpPerc","samebar":"allowSameBarExit","cooldown":"minBarsInTrade"}
    k=mapping.get(k,k)
    if not hasattr(CONFIG,k): return f"❌ unbekannter Key: {k}"
    cur=getattr(CONFIG,k)
    if isinstance(cur,bool):   setattr(CONFIG,k,v.lower() in ("1","true","on","yes"))
    elif isinstance(cur,int):  setattr(CONFIG,k,int(float(v)))
    elif isinstance(cur,float):setattr(CONFIG,k,float(v))
    elif isinstance(cur,list): setattr(CONFIG,k,[x.strip() for x in v.split(",") if x.strip()])
    else:                      setattr(CONFIG,k,v)
    if k=="poll_minutes": TIMER["poll_minutes"]=getattr(CONFIG,k)
    if k=="market_hours_only": TIMER["market_hours_only"]=getattr(CONFIG,k)
    return f"✓ {k} = {getattr(CONFIG,k)}"

async def cmd_set(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /set key=value …  z.B. /set rsiLow=0 rsiHigh=68 rsiExit=48 poll_minutes=10")
        return
    msgs=[]; errs=[]
    for a in context.args:
        if "=" not in a: errs.append(f"❌ {a}"); continue
        try: msgs.append(set_from_kv(a))
        except Exception as e: errs.append(f"❌ {a}: {e}")
    txt = ("✅ Übernommen:\n" + "\n".join(msgs) + ("\n\n⚠️ Probleme:\n" + "\n".join(errs) if errs else "")).strip()
    await update.message.reply_text(txt)

async def cmd_run(update, context):
    # /run sendet IMMER — unabhängig von Marktzeiten — und ohne Grace
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True, prepost=False)

async def cmd_bt(update, context):
    days=180
    if context.args:
        try: days=int(context.args[0])
        except: pass
    sym=CONFIG.symbols[0]
    # Einfacher Backtest auf TV-sync Daten
    mdf = fetch_alpaca_minutes(sym, days, prepost=False)
    if mdf.empty:
        await update.message.reply_text("❌ Keine Minutendaten für Backtest."); return
    df = resample_to_hour_tv_sync(mdf, drop_last_incomplete=True)
    if df.empty:
        await update.message.reply_text("❌ Keine 1h-Daten nach Resample."); return
    f = build_features(df)
    pos=0; avg=0.0; eq=1.0; R=[]; entries=exits=0
    for i in range(2,len(f)):
        row, prev = f.iloc[i], f.iloc[i-1]
        entry=bool(row["entry_cond"]); exitc=bool(row["exit_cond"])
        if pos==0 and entry:
            pos=1; avg=float(row["open"]); entries+=1
        elif pos==1:
            sl = avg*(1-CONFIG.slPerc/100); tp=avg*(1+CONFIG.tpPerc/100)
            price=float(row["close"])
            stop=price<=sl; take=price>=tp
            if exitc or stop or take:
                px = sl if stop else tp if take else float(row["open"])
                r=(px-avg)/avg; eq*=(1+r); R.append(r); exits+=1; pos=0; avg=0.0
    if R:
        a=np.array(R); win=(a>0).mean()
        pf=a[a>0].sum()/(1e-9 + -a[a<0].sum() if (a<0).any() else 1e-9)
        cagr=(eq**(365/max(1,days))-1)
        await update.message.reply_text(f"📈 Backtest {days}d  Trades={entries}/{exits}  Win={win*100:.1f}%  PF={pf:.2f}  CAGR~{cagr*100:.2f}%\nℹ️ EoB, keine Fees/Slippage.")
    else:
        await update.message.reply_text("📉 Backtest: keine Trades.")

async def cmd_dump(update, context):
    sym=CONFIG.symbols[0]
    args=[a.lower() for a in (context.args or [])]
    df, note = fetch_ohlcv_tv_sync(sym, CONFIG.interval, CONFIG.lookback_days, prepost=False, drop_last_incomplete=True)
    if df.empty:
        await update.message.reply_text("❌ Keine Daten."); return
    if args and args[0]=="csv":
        n = 300
        if len(args)>=2:
            try: n=max(1,int(args[1]))
            except: pass
        exp = build_features(df).tail(n)
        csv = exp.to_csv(index=True).encode("utf-8")
        await send_document_bytes(str(update.effective_chat.id), csv, f"{sym}_{CONFIG.interval}_indicators_{n}.csv", "🧾 CSV (TV-sync)")
        return
    last = build_features(df).iloc[-1]
    payload = {
        "symbol": sym, "interval": CONFIG.interval,
        "time": str(last.name),
        "close": float(last["close"]),
        "rsi": float(last["rsi"]),
        "efi": float(last["efi"]),
        "entry_cond": bool(last["entry_cond"]),
        "exit_cond": bool(last["exit_cond"]),
        "note": note
    }
    await update.message.reply_text("🧾 Dump\n" + json.dumps(payload, indent=2))

async def cmd_timer(update, context):
    if not context.args:
        await update.message.reply_text("Nutze: /timer on|off"); return
    on=context.args[0].lower() in ("on","1","true","start")
    TIMER["enabled"]=on
    await update.message.reply_text(f"Timer = {'ON' if on else 'OFF'}")

async def cmd_timerstatus(update, context):
    await update.message.reply_text("⏱️ Timer\n" + json.dumps({
        "enabled": TIMER["enabled"],
        "running": TIMER["running"],
        "poll_minutes": TIMER["poll_minutes"],
        "last_run": TIMER["last_run"],
        "next_due": TIMER["next_due"],
        "market_hours_only": TIMER["market_hours_only"],
        "chat_id_present": bool(CHAT_ID)
    }, indent=2))

async def cmd_timerrunnow(update, context):
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=True, prepost=False)
    now = datetime.now(timezone.utc)
    TIMER["last_run"]=now.isoformat()
    TIMER["next_due"]=(now + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0).isoformat()
    await update.message.reply_text("⏱️ Timer-Run ausgeführt.")

async def on_message(update, context):
    await update.message.reply_text("Unbekannter Befehl. /start für Hilfe")

# ========= FastAPI lifespan (PTB polling & timer) =========
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
        tg_app.add_handler(CommandHandler("bt",      cmd_bt))
        tg_app.add_handler(CommandHandler("dump",    cmd_dump))
        tg_app.add_handler(CommandHandler("timer",        cmd_timer))
        tg_app.add_handler(CommandHandler("timerstatus",  cmd_timerstatus))
        tg_app.add_handler(CommandHandler("timerrunnow",  cmd_timerrunnow))
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

        # Timer starten, wenn enabled
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
                await tg_app.stop(); await tg_app.shutdown()
        except Exception:
            pass
        POLLING_STARTED=False
        TIMER["running"]=False
        print("🛑 Shutdown complete")

# ========= FastAPI app & routes =========
app = FastAPI(title="TV-Sync 1h + Timer Heartbeat", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "ok": True,
        "symbols": CONFIG.symbols,
        "interval": CONFIG.interval,
        "provider": CONFIG.data_provider,
        "alpaca_feed": APCA_DATA_FEED,
        "live": CONFIG.live_enabled,
        "timer": {
            "enabled": TIMER["enabled"],
            "running": TIMER["running"],
            "poll_minutes": TIMER["poll_minutes"],
            "next_due": TIMER["next_due"]
        }
    }

@app.get("/tick")
async def tick():
    # manueller Trigger (ohne Grace, ohne Marktfilter)
    for sym in CONFIG.symbols:
        await run_once_for_symbol(sym, send_signals=False, prepost=False)
    now = datetime.now(timezone.utc)
    TIMER["last_run"]=now.isoformat()
    TIMER["next_due"]=(now + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0).isoformat()
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
        "ENABLE_TIMER": os.getenv("ENABLE_TIMER",""),
        "POLL_MINUTES": os.getenv("POLL_MINUTES",""),
        "MARKET_HOURS_ONLY": os.getenv("MARKET_HOURS_ONLY",""),
    }

@app.get("/tgstatus")
def tgstatus():
    return {"polling_started": POLLING_STARTED, "chat_id_present": bool(CHAT_ID)}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
