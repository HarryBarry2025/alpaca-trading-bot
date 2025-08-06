{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;}
{\colortbl;\red255\green255\blue255;\red147\green0\blue147;\red253\green144\blue145;\red12\green12\blue12;
\red0\green0\blue0;\red133\green85\blue4;\red66\green147\blue62;}
{\*\expandedcolortbl;;\cssrgb\c65098\c14902\c64314;\cssrgb\c100000\c64314\c63529;\cssrgb\c5098\c5098\c5098;
\cssrgb\c0\c0\c0;\cssrgb\c59608\c40784\c392;\cssrgb\c31373\c63137\c30980;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
import\cf4  pandas \cf2 as\cf4  pd\
\cf2 import\cf4  numpy \cf2 as\cf4  np\
\cf2 from\cf4  alpaca_trade_api.rest \cf2 import\cf4  REST, TimeFrame\
\cf2 from\cf4  ta.momentum \cf2 import\cf4  RSIIndicator\
\cf2 from\cf4  ta.trend \cf2 import\cf4  MACD, EMAIndicator\
\cf2 import\cf4  datetime \cf2 as\cf4  dt\
\cf2 import\cf4  requests\
\cf2 import\cf4  os\
\
API_KEY = os.\cf6 getenv\cf4 (\cf7 "APCA_API_KEY"\cf4 )\
API_SECRET = os.\cf6 getenv\cf4 (\cf7 "APCA_API_SECRET"\cf4 )\
BASE_URL = os.\cf6 getenv\cf4 (\cf7 "APCA_API_BASE_URL"\cf4 , \cf7 "https://paper-api.alpaca.markets"\cf4 )\
TELEGRAM_TOKEN = os.\cf6 getenv\cf4 (\cf7 "TELEGRAM_TOKEN"\cf4 )\
TELEGRAM_CHAT_ID = os.\cf6 getenv\cf4 (\cf7 "TELEGRAM_CHAT_ID"\cf4 )\
\
SYMBOL = \cf7 "TQQQ"\cf4 \
VIX_SYMBOL = \cf7 "VIXY"\cf4 \
TIMEFRAME = TimeFrame.\cf6 Hour\cf4 \
LOOKBACK = \cf6 100\cf4 \
\
api = REST(API_KEY, API_SECRET, BASE_URL)\
\
\cf2 def\cf4  get_data(symbol, timeframe, lookback):\
    start = (dt.\cf6 datetime\cf4 .\cf6 now\cf4 () - dt.\cf6 timedelta\cf4 (hours=lookback + \cf6 5\cf4 )).\cf6 replace\cf4 (microsecond=\cf6 0\cf4 ).\cf6 isoformat\cf4 () + \cf7 'Z'\cf4 \
    bars = api.\cf6 get_bars\cf4 (symbol, timeframe, start=start).\cf6 df\cf4 \
    \cf2 return\cf4  bars[bars[\cf7 'symbol'\cf4 ] == symbol].\cf6 copy\cf4 ()\
\
\cf2 def\cf4  elder_not_red(df):\
    ema = EMAIndicator(df[\cf7 'close'\cf4 ], window=\cf6 13\cf4 ).\cf6 ema_indicator\cf4 ()\
    macd = MACD(df[\cf7 'close'\cf4 ], \cf6 8\cf4 , \cf6 21\cf4 , \cf6 11\cf4 )\
    hist = macd.\cf6 macd_diff\cf4 ()\
    \cf2 return\cf4  (ema > ema.\cf6 shift\cf4 (\cf6 1\cf4 )) | (hist > hist.\cf6 shift\cf4 (\cf6 1\cf4 ))\
\
\cf2 def\cf4  send_telegram(msg):\
    \cf2 if\cf4  TELEGRAM_TOKEN \cf2 and\cf4  TELEGRAM_CHAT_ID:\
        url = \cf7 f"https://api.telegram.org/bot\cf4 \{TELEGRAM_TOKEN\}\cf7 /sendMessage"\cf4 \
        data = \{\cf7 "chat_id"\cf4 : TELEGRAM_CHAT_ID, \cf7 "text"\cf4 : msg\}\
        r = requests.\cf6 post\cf4 (url, data=data)\
        \cf2 if\cf4  r.\cf6 status_code\cf4  != \cf6 200\cf4 :\
            print(\cf7 f"Telegram send error: \cf4 \{r.\cf6 text\cf4 \}\cf7 "\cf4 )\
\
\cf2 def\cf4  check_signals():\
    df = get_data(SYMBOL, TIMEFRAME, LOOKBACK)\
    vix = get_data(VIX_SYMBOL, TimeFrame.\cf6 Minute\cf4 , \cf6 15\cf4 )\
\
    df[\cf7 'rsi'\cf4 ] = RSIIndicator(df[\cf7 'close'\cf4 ], \cf6 12\cf4 ).\cf6 rsi\cf4 ()\
    macd = MACD(df[\cf7 'close'\cf4 ], \cf6 8\cf4 , \cf6 21\cf4 , \cf6 11\cf4 )\
    df[\cf7 'macd'\cf4 ] = macd.\cf6 macd\cf4 ()\
    df[\cf7 'signal'\cf4 ] = macd.\cf6 macd_signal\cf4 ()\
    df[\cf7 'hist'\cf4 ] = macd.\cf6 macd_diff\cf4 ()\
    df[\cf7 'ema'\cf4 ] = EMAIndicator(df[\cf7 'close'\cf4 ], window=\cf6 13\cf4 ).\cf6 ema_indicator\cf4 ()\
    df[\cf7 'rsi_rising'\cf4 ] = df[\cf7 'rsi'\cf4 ] > df[\cf7 'rsi'\cf4 ].\cf6 shift\cf4 (\cf6 1\cf4 )\
    df[\cf7 'impulse_ok'\cf4 ] = elder_not_red(df)\
\
    vix[\cf7 'vix_rising'\cf4 ] = vix[\cf7 'close'\cf4 ] > vix[\cf7 'close'\cf4 ].\cf6 shift\cf4 (\cf6 1\cf4 )\
    vix_ok = \cf2 not\cf4  vix.\cf6 iloc\cf4 [-\cf6 1\cf4 ][\cf7 'vix_rising'\cf4 ]\
\
    latest = df.\cf6 iloc\cf4 [-\cf6 1\cf4 ]\
    entry = (\
        latest[\cf7 'macd'\cf4 ] > latest[\cf7 'signal'\cf4 ] \cf2 and\cf4 \
        \cf6 52\cf4  < latest[\cf7 'rsi'\cf4 ] < \cf6 64\cf4  \cf2 and\cf4 \
        latest[\cf7 'rsi_rising'\cf4 ] \cf2 and\cf4 \
        latest[\cf7 'impulse_ok'\cf4 ] \cf2 and\cf4 \
        vix_ok\
    )\
    exit = latest[\cf7 'rsi'\cf4 ] < \cf6 48\cf4 \
\
    \cf2 try\cf4 :\
        position = api.\cf6 get_open_position\cf4 (SYMBOL)\
        in_position = \cf6 True\cf4 \
    \cf2 except\cf4 :\
        in_position = \cf6 False\cf4 \
\
    \cf2 if\cf4  entry \cf2 and\cf4  \cf2 not\cf4  in_position:\
        api.\cf6 submit_order\cf4 (symbol=SYMBOL, qty=\cf6 1\cf4 , side=\cf7 'buy'\cf4 , type=\cf7 'market'\cf4 , time_in_force=\cf7 'gtc'\cf4 )\
        send_telegram(\cf7 f"\uc0\u55357 \u56960  BUY \cf4 \{SYMBOL\}\cf7  at \cf4 \{latest[\cf7 'close'\cf4 ]:\cf2 .2f\cf4 \}\cf7 "\cf4 )\
    \cf2 elif\cf4  exit \cf2 and\cf4  in_position:\
        api.\cf6 submit_order\cf4 (symbol=SYMBOL, qty=\cf6 1\cf4 , side=\cf7 'sell'\cf4 , type=\cf7 'market'\cf4 , time_in_force=\cf7 'gtc'\cf4 )\
        send_telegram(\cf7 f"\uc0\u55357 \u56635  SELL \cf4 \{SYMBOL\}\cf7  at \cf4 \{latest[\cf7 'close'\cf4 ]:\cf2 .2f\cf4 \}\cf7 "\cf4 )\
    \cf2 else\cf4 :\
        print(\cf7 "\uc0\u9203  No action"\cf4 )\
\
\cf2 if\cf4  __name__ == \cf7 "__main__"\cf4 :\
    check_signals()}