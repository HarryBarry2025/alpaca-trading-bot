import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import datetime as dt
import requests
import os

API_KEY = os.getenv("APCA_API_KEY")
API_SECRET = os.getenv("APCA_API_SECRET")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "TQQQ"
VIX_SYMBOL = "VIXY"
TIMEFRAME = TimeFrame.Hour
LOOKBACK = 100

api = REST(API_KEY, API_SECRET, BASE_URL)

def get_data(symbol, timeframe, lookback):
    start = (dt.datetime.now() - dt.timedelta(hours=lookback + 5)).replace(microsecond=0).isoformat() + 'Z'
    bars = api.get_bars(symbol, timeframe, start=start).df
    # Manche Alpaca-Versionen liefern keine 'symbol'-Spalte zurück, wenn nur 1 Symbol abgefragt wird
    if 'symbol' in bars.columns:
        return bars[bars['symbol'] == symbol].copy()
    else:
        return bars.copy()

def elder_not_red(df):
    ema = EMAIndicator(df['close'], window=13).ema_indicator()
    macd = MACD(df['close'], 8, 21, 11)
    hist = macd.macd_diff()
    return (ema > ema.shift(1)) | (hist > hist.shift(1))

def send_telegram(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        r = requests.post(url, data=data)
        if r.status_code != 200:
            print(f"Telegram send error: {r.text}")

def check_signals():
    df = get_data(SYMBOL, TIMEFRAME, LOOKBACK)
    vix = get_data(VIX_SYMBOL, TimeFrame.Minute, 15)

    df['rsi'] = RSIIndicator(df['close'], 12).rsi()
    macd = MACD(df['close'], 8, 21, 11)
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    df['hist'] = macd.macd_diff()
    df['ema'] = EMAIndicator(df['close'], window=13).ema_indicator()
    df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
    df['impulse_ok'] = elder_not_red(df)

    vix['vix_rising'] = vix['close'] > vix['close'].shift(1)
    vix_ok = not vix.iloc[-1]['vix_rising']

    latest = df.iloc[-1]
    entry = (
        latest['macd'] > latest['signal'] and
        52 < latest['rsi'] < 64 and
        latest['rsi_rising'] and
        latest['impulse_ok'] and
        vix_ok
    )
    exit = latest['rsi'] < 48

    try:
        position = api.get_open_position(SYMBOL)
        in_position = True
    except:
        in_position = False

    if entry and not in_position:
        api.submit_order(symbol=SYMBOL, qty=1, side='buy', type='market', time_in_force='gtc')
        send_telegram(f"🚀 BUY {SYMBOL} at {latest['close']:.2f}")
    elif exit and in_position:
        api.submit_order(symbol=SYMBOL, qty=1, side='sell', type='market', time_in_force='gtc')
        send_telegram(f"🔻 SELL {SYMBOL} at {latest['close']:.2f}")
    else:
        print("⏳ No action")

if __name__ == "__main__":
    check_signals()
