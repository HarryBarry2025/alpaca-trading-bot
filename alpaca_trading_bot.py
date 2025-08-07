import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

SYMBOL = "TQQQ"
VIX_SYMBOL = "VIXY"
LOOKBACK_DAYS = 60

def get_data_yf(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period)
        if df.empty or 'Close' not in df.columns:
            raise ValueError(f"No valid data for {symbol} with interval={interval}, period={period}")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[!] Data fetch error for {symbol} ({interval}): {e}")
        return pd.DataFrame()

def elder_not_red(df, ema_length, macd_fast, macd_slow, macd_signal):
    close_series = df['Close'].squeeze()
    ema = EMAIndicator(close_series, window=ema_length).ema_indicator()
    macd = MACD(close_series, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    hist = macd.macd_diff()
    return (ema > ema.shift(1)) | (hist > hist.shift(1))

def backtest(params, timeframe):
    rsi_len, rsi_entry_min, rsi_entry_max, rsi_exit, macd_f, macd_s, macd_sig, ema_len = params
    df = get_data_yf(SYMBOL, timeframe, f"{LOOKBACK_DAYS}d")
    vix = get_data_yf(VIX_SYMBOL, "15m", "5d")

    if df.empty or vix.empty:
        return -np.inf

    close_series = df['Close'].squeeze()

    df['rsi'] = RSIIndicator(close=close_series, window=rsi_len).rsi()
    macd = MACD(close=close_series, window_fast=macd_f, window_slow=macd_s, window_sign=macd_sig)
    df['macd'] = macd.macd()
    df['signal'] = macd.macd_signal()
    df['hist'] = macd.macd_diff()
    df['ema'] = EMAIndicator(close=close_series, window=ema_len).ema_indicator()
    df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
    df['impulse_ok'] = elder_not_red(df, ema_len, macd_f, macd_s, macd_sig)
    vix['vix_rising'] = vix['Close'] > vix['Close'].shift(1)

    df.dropna(inplace=True)
    if df.empty:
        return -np.inf

    position = False
    equity = 1_000
    balance = equity
    shares = 0

    for i in range(1, len(df)):
        latest = df.iloc[i]
        vix_ok = not vix.iloc[-1]['vix_rising'] if len(vix) > 1 else True

        try:
            entry = (
                latest['macd'] > latest['signal'] and
                rsi_entry_min < latest['rsi'] < rsi_entry_max and
                latest['rsi_rising'] is True and
                latest['impulse_ok'] is True and
                vix_ok
            )
        except Exception:
            entry = False

        exit = latest['rsi'] < rsi_exit

        if entry and not position:
            buy_price = latest['Close']
            shares = balance / buy_price
            position = True
        elif exit and position:
            sell_price = latest['Close']
            balance = shares * sell_price
            shares = 0
            position = False

    final_value = balance if not position else shares * df.iloc[-1]['Close']
    total_return = (final_value / equity - 1) * 100
    return total_return

def optimize():
    best_result = -np.inf
    best_params = None
    results = []
    timeframes = ["15m", "30m", "1h", "4h"]  # Updated intervals

    rsi_lens = [10, 12, 14]
    rsi_entries = [(50, 65), (52, 64), (54, 62)]
    rsi_exits = [45, 48, 50]
    macd_fast_vals = [8, 12]
    macd_slow_vals = [21, 26]
    macd_signal_vals = [9, 11]
    ema_lens = [13, 20]

    for tf in timeframes:
        param_grid = product(rsi_lens, rsi_entries, rsi_exits, macd_fast_vals, macd_slow_vals, macd_signal_vals, ema_lens)

        for rsi_len, (entry_min, entry_max), rsi_exit, macd_f, macd_s, macd_sig, ema_len in param_grid:
            try:
                result = backtest((rsi_len, entry_min, entry_max, rsi_exit, macd_f, macd_s, macd_sig, ema_len), tf)
                results.append({
                    "timeframe": tf,
                    "rsi": rsi_len,
                    "rsi_min": entry_min,
                    "rsi_max": entry_max,
                    "rsi_exit": rsi_exit,
                    "macd_fast": macd_f,
                    "macd_slow": macd_s,
                    "macd_signal": macd_sig,
                    "ema": ema_len,
                    "return": result
                })
                if result > best_result:
                    best_result = result
                    best_params = (rsi_len, entry_min, entry_max, rsi_exit, macd_f, macd_s, macd_sig, ema_len, tf)
                print(f"TF={tf} Tested RSI={rsi_len}, Entry=({entry_min}-{entry_max}), Exit={rsi_exit}, MACD=({macd_f},{macd_s},{macd_sig}), EMA={ema_len} → Return={result:.2f}%")
            except Exception as e:
                print(f"[!] Error on TF={tf} with params {(rsi_len, entry_min, entry_max, rsi_exit, macd_f, macd_s, macd_sig, ema_len)}: {e}")

    if not results:
        print("No valid backtest results.")
        return

    print("\nBest Result:")
    print(f"Return: {best_result:.2f}%")
    print(f"Params: RSI={best_params[0]}, Entry=({best_params[1]}, {best_params[2]}), Exit={best_params[3]}, MACD=({best_params[4]},{best_params[5]},{best_params[6]}), EMA={best_params[7]}, Timeframe={best_params[8]}")

    # Tabular results
    df_results = pd.DataFrame(results)
    print("\nTop 10 Results:")
    print(df_results.sort_values(by="return", ascending=False).head(10))

    # Heatmap: Timeframe vs MACD Fast
    pivot = df_results.pivot_table(index="timeframe", columns="macd_fast", values="return", aggfunc=np.max)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Max Return Heatmap: Timeframe vs MACD Fast")
    plt.xlabel("MACD Fast")
    plt.ylabel("Timeframe")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    optimize()
