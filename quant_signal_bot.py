#!/usr/bin/env python

import os, sys, time, asyncio, threading, json
import ccxt
import pandas as pd
import pandas_ta as pta
import talib
import numpy as np
from telegram import Bot
from flask import Flask
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import pickle
import warnings
import aiohttp
import ccxt.async_support as ccxt_async 
from tenacity import retry, wait_exponential, stop_after_attempt , retry_if_exception_type
import httpx
warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
CONFIG = {
    "telegram_token": os.environ.get("BOT_TOKEN", "YourBotToken"),
    "chat_id": os.environ.get("CHANNEL_ID", "YourChannelID"),
    "pairs": [
        ("SOLUSDT", "BTCUSDT"), ("BNBUSDT", "BTCUSDT"), ("ADAUSDT", "BTCUSDT"), ("XRPUSDT", "BTCUSDT"), 
        ("TONUSDT", "BTCUSDT"),("TRBUSDT", "BTCUSDT"), 
    ],
    "timeframes": ["15m", "1h", "4h", "1d"],
    "conf_threshold": 85,
    "atr_sl_mult": 1.8,
    "atr_tp_mult": 2.5,
    "adx_threshold": 25,
    "vol_mult": 2.0,
    "signal_lifetime_sec": 6*3600,
    "smt_window": 3,
    "log_file": "signals_log.jsonl",
    "ml_enabled": True,
    "ml_threshold": 0.65,
    "market_regime_enabled": True,
    "multi_tf_confluence": True,
    "volatility_filter": True,
    "order_flow_analysis": True,
    "scan_interval": 600
}

bot_status = {"status": "starting", "last_update": time.time(), "signals_count": 0}

# Validate configuration
for k in ("telegram_token", "chat_id"):
    if not CONFIG[k] or CONFIG[k].startswith("<"):
        print(f"ERROR: Please set {k} in env or config!")
        sys.exit(1)

# Initialize components
bot = Bot(token=CONFIG["telegram_token"])

exchange = ccxt_async.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})

app = Flask(__name__)


@app.route('/ping')
def home():
    return json.dumps({
        "message": "Crypto Quant Trading Bot is Running",
        "status": bot_status["status"],
        "uptime": time.time() - bot_status["last_update"],
    })
@app.route('/health')
def health(): return "ok"
def run_flask(): app.run(host="0.0.0.0", port=10000)

# Global variables
open_signals = {}
signal_history = []
semaphore = asyncio.Semaphore(10)  # Control concurrent requests

# ================== DATA FETCHING ==================
@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
async def fetch_ohlcv(symbol, timeframe, limit=300):
    
    async with semaphore: # global asyncio.Semaphore(10)
        data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        try:
          df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
          df['ts']=pd.to_datetime(df['ts'],unit='ms')
          df.set_index('ts',inplace=True)
          return df
        except Exception as e:
         print(f"Error fetching data for {symbol} {timeframe}: {e}")
         return None
        

# ================== ADVANCED INDICATORS ==================
def advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    # Basic EMAs
    df['ema20'] = pta.ema(df['close'], 20)
    df['ema50'] = pta.ema(df['close'], 50)
    df['ema200'] = pta.ema(df['close'], 200)
    
    # RSI variants
    df['rsi'] = pta.rsi(df['close'], 14)
    df['rsi_fast'] = pta.rsi(df['close'], 7)
    df['rsi_slow'] = pta.rsi(df['close'], 21)
    
    # MACD
    macd = pta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macds'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bb = pta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_middle'] = bb['BBM_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Stochastic
    stoch = pta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Williams %R
    df['williams_r'] = pta.willr(df['high'], df['low'], df['close'])
    
    # ADX and DMI
    adx_data = pta.adx(df['high'], df['low'], df['close'])
    df['adx'] = adx_data['ADX_14']
    df['dmp'] = adx_data['DMP_14']
    df['dmn'] = adx_data['DMN_14']
    
    # ATR and volatility
    df['atr'] = pta.atr(df['high'], df['low'], df['close'])
    df['atr_pct'] = df['atr'] / df['close'] * 100
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    
    # Volume indicators
    vol_ma = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] > vol_ma * CONFIG['vol_mult']
    df['vol_ratio'] = df['volume'] / vol_ma
    df['obv'] = pta.obv(df['close'], df['volume'])
    df['cmf'] = pta.cmf(df['high'], df['low'], df['close'], df['volume'])
    df['mfi'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'])
    
    # Price action metrics
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Momentum
    df['roc'] = pta.roc(df['close'], length=10)
    df['cci'] = pta.cci(df['high'], df['low'], df['close'])
    df = df.fillna(method='ffill').fillna(0)
    return df

def add_candle_patterns(df):
    """Add candlestick pattern recognition"""
    o,h,l,c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    patterns = [
        ("hammer", talib.CDLHAMMER), ("engulfing", talib.CDLENGULFING),
        ("doji", talib.CDLDOJI), ("shooting_star", talib.CDLSHOOTINGSTAR),
        ("morning_star", talib.CDLMORNINGSTAR), ("evening_star", talib.CDLEVENINGSTAR),
        ("three_white_soldiers", talib.CDL3WHITESOLDIERS), ("three_black_crows", talib.CDL3BLACKCROWS),
        ("marubozu", talib.CDLMARUBOZU), ("harami", talib.CDLHARAMI),
        ("piercing", talib.CDLPIERCING), ("dark_cloud_cover", talib.CDLDARKCLOUDCOVER)
    ]
    
    for name, func in patterns:
        df[name] = func(o, h, l, c)
    
    # Custom patterns
    df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['outside_bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    
    return df

def order_flow_analysis(df):
    """Analyze order flow patterns"""
    # Price rejection patterns
    df['upper_rejection'] = (df['high'] == df['high'].rolling(5).max()) & (df['close'] < df['high'] * 0.98)
    df['lower_rejection'] = (df['low'] == df['low'].rolling(5).min()) & (df['close'] > df['low'] * 1.02)
    
    # Buying/selling pressure
    df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    
    return df

# ================== CHART PATTERN DETECTION ==================
def detect_double_top(df, tol=0.01):
    close = df['close']
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    idxs = close[peaks].index[-2:]
    return (abs(close.loc[idxs[0]] - close.loc[idxs[1]]) / close.loc[idxs[0]] < tol) if len(idxs) == 2 else False

def detect_double_bottom(df, tol=0.01):
    close = df['close']
    valleys = (close.shift(1) > close) & (close.shift(-1) > close)
    idxs = close[valleys].index[-2:]
    return (abs(close.loc[idxs[0]] - close.loc[idxs[1]]) / close.loc[idxs[0]] < tol) if len(idxs) == 2 else False

def detect_triple_top(df, tol=0.01):
    close = df['close']
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    idxs = close[peaks].index[-3:]
    if len(idxs) == 3:
        v1, v2, v3 = close.loc[idxs]
        return abs(v1-v2)/v1 < tol and abs(v2-v3)/v2 < tol
    return False

def detect_triple_bottom(df, tol=0.01):
    close = df['close']
    valleys = (close.shift(1) > close) & (close.shift(-1) > close)
    idxs = close[valleys].index[-3:]
    if len(idxs) == 3:
        v1, v2, v3 = close.loc[idxs]
        return abs(v1-v2)/v1 < tol and abs(v2-v3)/v2 < tol
    return False

def detect_rectangle(df, tol=0.015, lookback=20):
    s = df['close'][-lookback:]
    u, l = s.max(), s.min()
    return ((s <= u + tol*u) & (s >= l - tol*u)).all()

def detect_head_and_shoulders(df):
    highs = df['high']
    lh = highs.rolling(3).apply(lambda x: x[1] > x[0] and x[1] > x[2], raw=True).fillna(0)
    p = highs[lh > 0].tail(3)
    return len(p) == 3 and p.iloc[1] > p.iloc[0] and p.iloc[1] > p.iloc[2]

def detect_inv_head_shoulders(df):
    lows = df['low']
    lh = lows.rolling(3).apply(lambda x: x[1] < x[0] and x[1] < x[2], raw=True).fillna(0)
    t = lows[lh > 0].tail(3)
    return len(t) == 3 and t.iloc[1] < t.iloc[0] and t.iloc[1] < t.iloc[2]

def detect_cup_and_handle(df, lookback=45):
    cl = df['close'][-lookback:]
    mid = lookback // 2
    return cl.iloc[0] > cl.min() and cl.iloc[mid] == cl.min() and cl.iloc[-1] > cl.iloc[mid]

def detect_rising_wedge(df, window=20):
    close = df['close'][-window:]
    return (close.is_monotonic_increasing and close.diff().min() > 0 and 
            close.diff().max() / close.diff().min() < 4)

def detect_falling_wedge(df, window=20):
    close = df['close'][-window:]
    return (close.is_monotonic_decreasing and close.diff().max() < 0 and 
            abs(close.diff().min() / close.diff().max()) < 4)

def detect_ascending_triangle(df, lookback=30, tol=0.01):
    highs = df['high'][-lookback:]
    resist = highs.max()
    close = df['close'][-lookback:]
    troughs = (close.shift(1) > close) & (close.shift(-1) > close)
    lows = close[troughs]
    return any(abs(resist - l) / resist < tol for l in lows)

def detect_descending_triangle(df, lookback=30, tol=0.01):
    lows = df['low'][-lookback:]
    supp = lows.min()
    close = df['close'][-lookback:]
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    highs = close[peaks]
    return any(abs(h - supp) / supp < tol for h in highs)

def detect_broadening(df, lookback=40):
    high = df['high'][-lookback:]
    low = df['low'][-lookback:]
    return high.is_monotonic_increasing and low.is_monotonic_decreasing

def get_chart_patterns(df):
    """Detect all chart patterns"""
    patterns = []
    if detect_double_top(df): patterns.append("Double Top")
    if detect_double_bottom(df): patterns.append("Double Bottom")
    if detect_triple_top(df): patterns.append("Triple Top")
    if detect_triple_bottom(df): patterns.append("Triple Bottom")
    if detect_rectangle(df): patterns.append("Rectangle")
    if detect_head_and_shoulders(df): patterns.append("Head & Shoulders")
    if detect_inv_head_shoulders(df): patterns.append("Inv Head & Shoulders")
    if detect_cup_and_handle(df): patterns.append("Cup & Handle")
    if detect_rising_wedge(df): patterns.append("Rising Wedge")
    if detect_falling_wedge(df): patterns.append("Falling Wedge")
    if detect_ascending_triangle(df): patterns.append("Ascending Triangle")
    if detect_descending_triangle(df): patterns.append("Descending Triangle")
    if detect_broadening(df): patterns.append("Broadening")
    return patterns

# ================== SMC & STRUCTURE ANALYSIS ==================
def detect_fvg(df, window=3):
    """Detect Fair Value Gaps"""
    gaps = []
    for i in range(window, len(df) - window):
        hi_prev = df['high'].iloc[i-1]
        lo_next = df['low'].iloc[i+1]
        if df['low'].iloc[i] > hi_prev:
            gaps.append(('bullish', i))
        elif df['high'].iloc[i] < lo_next:
            gaps.append(('bearish', i))
    return gaps

def find_sr_zones(df, bins=30):
    """Find support/resistance zones through price clustering"""
    prices = df['close'][-100:]
    counts, edges = pd.cut(prices, bins, retbins=True, labels=False)
    levels = []
    for b in range(bins):
        hits = prices[counts == b]
        if len(hits) > 3:
            levels.append(hits.mean())
    return sorted(set(round(l, 3) for l in levels))

def detect_bos_choch(df, lookback=20):
    """Detect Break of Structure / Change of Character"""
    closes = df['close'][-lookback:]
    if closes.iloc[-1] > closes.iloc[:-1].max():
        return "up_bos"
    if closes.iloc[-1] < closes.iloc[:-1].min():
        return "down_bos"
    return None

def find_fractals(df, window=3):
    """Find fractal highs and lows"""
    highs = (df['high'] == df['high'].rolling(window, center=True).max())
    lows = (df['low'] == df['low'].rolling(window, center=True).min())
    return highs, lows

def smt_divergence(df_main, df_ref, window=3):
    """Detect Smart Money divergence"""
    highs_main, lows_main = find_fractals(df_main, window)
    highs_ref, lows_ref = find_fractals(df_ref, window)
    divergences = []
    
    for i in range(window, len(df_main) - window):
        # Bullish divergence
        if lows_main.iloc[i] and lows_ref.iloc[i]:
            pm = df_main['low'].iloc[i-window:i][lows_main.iloc[i-window:i]].min() if any(lows_main.iloc[i-window:i]) else None
            pr = df_ref['low'].iloc[i-window:i][lows_ref.iloc[i-window:i]].min() if any(lows_ref.iloc[i-window:i]) else None
            cm = df_main['low'].iloc[i]
            cr = df_ref['low'].iloc[i]
            if pm is not None and pr is not None and cm > pm and cr < pr:
                divergences.append(('bullish', i))
        
        # Bearish divergence
        if highs_main.iloc[i] and highs_ref.iloc[i]:
            pmh = df_main['high'].iloc[i-window:i][highs_main.iloc[i-window:i]].max() if any(highs_main.iloc[i-window:i]) else None
            prh = df_ref['high'].iloc[i-window:i][highs_ref.iloc[i-window:i]].max() if any(highs_ref.iloc[i-window:i]) else None
            cmh = df_main['high'].iloc[i]
            crh = df_ref['high'].iloc[i]
            if pmh is not None and prh is not None and cmh < pmh and crh > prh:
                divergences.append(('bearish', i))
    
    if divergences:
        kind, idx = divergences[-1]
        if idx == len(df_main) - window - 1:
            return kind
    return None

# ================== MARKET REGIME DETECTION ==================
def detect_market_regime(df):
    """Detect market regime: trending, ranging, volatile"""
    adx_avg = df['adx'].tail(10).mean()
    volatility_avg = df['volatility'].tail(10).mean()
    bb_width_avg = df['bb_width'].tail(10).mean()
    volatility_ma = df['volatility'].rolling(50).mean().iloc[-1]
    bb_width_ma = df['bb_width'].rolling(50).mean().iloc[-1]
    
    if adx_avg > 30 and volatility_avg < volatility_ma:
        return "trending"
    elif bb_width_avg < bb_width_ma * 0.8:
        return "ranging"
    elif volatility_avg > volatility_ma * 1.5:
        return "volatile"
    else:
        return "neutral"

async def multi_timeframe_confluence(symbol, ref_symbol):
    """Check confluence across multiple timeframes"""
    confluence_score = 0
    tf_data = {}

    for tf in ["1h", "4h", "1d"]:
        try:
            df = await fetch_ohlcv(symbol, tf, limit=100)
            df = advanced_indicators(df)
            tf_data[tf] = df

            last = df.iloc[-1]

            # ✅ Skip if any EMA is NaN
            if pd.isna(last['ema20']) or pd.isna(last['ema50']) or pd.isna(last['ema200']):
                print(f"[SKIP] Missing EMA values for {symbol}-{tf}")
                continue

            # Trend confluence (safe now)
            if last['ema20'] > last['ema50'] > last['ema200']:
                confluence_score += 1
            elif last['ema20'] < last['ema50'] < last['ema200']:
                confluence_score -= 1

        except Exception as e:
            print(f"Multi-TF error for {symbol}-{tf}: {e}")

    return confluence_score, tf_data

# ================== COMPLETE ML CLASSIFIER WITH ALL THREE ALGORITHMS ==================
class CompleteCryptoMLClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model_name = None
        self.performance_history = {}
        
    def create_features(self, df, chart_patterns, fvg, choch, sr_levels):
        """Create comprehensive feature set for ML"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        features = {
            # Technical indicators (20+ features)
            'rsi': last['rsi'], 'rsi_fast': last['rsi_fast'], 'rsi_slow': last['rsi_slow'],
            'macd': last['macd'], 'macds': last['macds'], 'macd_hist': last['macd_hist'],
            'bb_position': last['bb_position'], 'bb_width': last['bb_width'],
            'stoch_k': last['stoch_k'], 'stoch_d': last['stoch_d'],
            'williams_r': last['williams_r'], 'adx': last['adx'],
            'atr_pct': last['atr_pct'], 'volatility': last['volatility'],
            'vol_ratio': last['vol_ratio'], 'cmf': last['cmf'], 'mfi': last['mfi'],
            'roc': last['roc'], 'cci': last['cci'],
            'dmp': last['dmp'], 'dmn': last['dmn'],
            
            # Price action (10+ features)
            'body_size': last['body_size'], 'upper_shadow': last['upper_shadow'],
            'lower_shadow': last['lower_shadow'], 'range_pct': last['range_pct'],
            'buying_pressure': last['buying_pressure'], 'selling_pressure': last['selling_pressure'],
            'gap_up': 1 if last['open'] > prev['close'] * 1.002 else 0,
            'gap_down': 1 if last['open'] < prev['close'] * 0.998 else 0,
            'higher_high': 1 if last['high'] > prev['high'] else 0,
            'lower_low': 1 if last['low'] < prev['low'] else 0,
            
            # Trend features (10+ features)
            'ema_alignment': 1 if last['ema20'] > last['ema50'] > last['ema200'] else 
                           -1 if last['ema20'] < last['ema50'] < last['ema200'] else 0,
            'price_vs_ema20': (last['close'] - last['ema20']) / last['ema20'],
            'price_vs_ema200': (last['close'] - last['ema200']) / last['ema200'],
            'ema20_slope': (last['ema20'] - prev['ema20']) / prev['ema20'],
            'momentum_1': (last['close'] - prev['close']) / prev['close'],
            'momentum_5': (last['close'] - df['close'].iloc[-6]) / df['close'].iloc[-6] if len(df) >= 6 else 0,
            'price_position_range': (last['close'] - df['low'].tail(20).min()) / (df['high'].tail(20).max() - df['low'].tail(20).min()),
            'consecutive_up': len([1 for i in range(min(5, len(df)-1)) if df['close'].iloc[-(i+1)] > df['close'].iloc[-(i+2)]]),
            'consecutive_down': len([1 for i in range(min(5, len(df)-1)) if df['close'].iloc[-(i+1)] < df['close'].iloc[-(i+2)]]),
            'macd_crossover': 1 if last['macd'] > last['macds'] and prev['macd'] <= prev['macds'] else 0,
            
            # Pattern features (15+ features)
            'double_top': 1 if "Double Top" in chart_patterns else 0,
            'double_bottom': 1 if "Double Bottom" in chart_patterns else 0,
            'triple_top': 1 if "Triple Top" in chart_patterns else 0,
            'triple_bottom': 1 if "Triple Bottom" in chart_patterns else 0,
            'head_shoulders': 1 if "Head & Shoulders" in chart_patterns else 0,
            'inv_head_shoulders': 1 if "Inv Head & Shoulders" in chart_patterns else 0,
            'rectangle': 1 if "Rectangle" in chart_patterns else 0,
            'cup_handle': 1 if "Cup & Handle" in chart_patterns else 0,
            'rising_wedge': 1 if "Rising Wedge" in chart_patterns else 0,
            'falling_wedge': 1 if "Falling Wedge" in chart_patterns else 0,
            'ascending_triangle': 1 if "Ascending Triangle" in chart_patterns else 0,
            'descending_triangle': 1 if "Descending Triangle" in chart_patterns else 0,
            'broadening': 1 if "Broadening" in chart_patterns else 0,
            'pattern_count': len(chart_patterns),
            'pattern_strength': min(len(chart_patterns), 3) / 3,  # Normalize to 0-1
            
            # Candlestick patterns (15+ features)
            'hammer': 1 if last['hammer'] == 100 else 0,
            'engulfing_bull': 1 if last['engulfing'] == 100 else 0,
            'engulfing_bear': 1 if last['engulfing'] == -100 else 0,
            'doji': 1 if last['doji'] == 100 else 0,
            'shooting_star': 1 if last['shooting_star'] == -100 else 0,
            'morning_star': 1 if last['morning_star'] == 100 else 0,
            'evening_star': 1 if last['evening_star'] == -100 else 0,
            'three_white_soldiers': 1 if last['three_white_soldiers'] == 100 else 0,
            'three_black_crows': 1 if last['three_black_crows'] == -100 else 0,
            'marubozu_bull': 1 if last['marubozu'] == 100 else 0,
            'marubozu_bear': 1 if last['marubozu'] == -100 else 0,
            'harami_bull': 1 if last['harami'] > 0 else 0,
            'harami_bear': 1 if last['harami'] < 0 else 0,
            'piercing': 1 if last['piercing'] > 0 else 0,
            'dark_cloud': 1 if last['dark_cloud_cover'] < 0 else 0,
            'inside_bar': 1 if last['inside_bar'] else 0,
            'outside_bar': 1 if last['outside_bar'] else 0,
            
            # SMC features (10+ features)
            'bos_up': 1 if choch == "up_bos" else 0,
            'bos_down': 1 if choch == "down_bos" else 0,
            'fvg_bullish': 1 if fvg and fvg[-1][0] == "bullish" else 0,
            'fvg_bearish': 1 if fvg and fvg[-1][0] == "bearish" else 0,
            'fvg_count': len(fvg) if fvg else 0,
            'near_resistance': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl > last['close']) else 0,
            'near_support': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl < last['close']) else 0,
            'above_key_level': len([lvl for lvl in sr_levels if lvl < last['close']]),
            'below_key_level': len([lvl for lvl in sr_levels if lvl > last['close']]),
            'sr_zone_strength': min(len(sr_levels), 10) / 10,  # Normalize to 0-1
            
            # Volume features (8+ features)
            'vol_spike': 1 if last['vol_spike'] else 0,
            'obv_trend': 1 if last['obv'] > prev['obv'] else 0,
            'volume_trend_5': 1 if df['volume'].tail(5).mean() > df['volume'].tail(10).mean() else 0,
            'volume_above_avg': 1 if last['vol_ratio'] > 1.5 else 0,
            'cmf_bullish': 1 if last['cmf'] > 0.1 else 0,
            'cmf_bearish': 1 if last['cmf'] < -0.1 else 0,
            'mfi_oversold': 1 if last['mfi'] < 20 else 0,
            'mfi_overbought': 1 if last['mfi'] > 80 else 0,
            
            # Oscillator features (10+ features)
            'rsi_oversold': 1 if last['rsi'] < 30 else 0,
            'rsi_overbought': 1 if last['rsi'] > 70 else 0,
            'rsi_divergence': 1 if last['rsi'] > prev['rsi'] and last['close'] < prev['close'] else 0,
            'stoch_oversold': 1 if last['stoch_k'] < 20 else 0,
            'stoch_overbought': 1 if last['stoch_k'] > 80 else 0,
            'stoch_crossover': 1 if last['stoch_k'] > last['stoch_d'] and prev['stoch_k'] <= prev['stoch_d'] else 0,
            'williams_oversold': 1 if last['williams_r'] > -20 else 0,
            'williams_overbought': 1 if last['williams_r'] < -80 else 0,
            'cci_oversold': 1 if last['cci'] < -100 else 0,
            'cci_overbought': 1 if last['cci'] > 100 else 0,
            
            # Market structure features (5+ features)
            'trend_strength': min(abs(last['adx']), 100) / 100,  # Normalize to 0-1
            'volatility_percentile': min(last['atr_pct'], 10) / 10,  # Normalize to 0-1
            'bb_squeeze': 1 if last['bb_width'] < df['bb_width'].rolling(20).mean().iloc[-1] * 0.8 else 0,
            'price_extreme': 1 if last['bb_position'] > 0.9 or last['bb_position'] < 0.1 else 0,
            'range_expansion': 1 if last['range_pct'] > df['range_pct'].rolling(20).mean().iloc[-1] * 1.5 else 0,
        }
        
        return features
    
    def prepare_training_data(self, historical_signals):
        """Prepare training data from historical signals"""
        X, y = [], []
        
        for signal in historical_signals:
            if 'features' in signal and 'outcome' in signal:
                features = signal['features']
                if isinstance(features, dict):
                    # Convert to consistent feature vector
                    if not self.feature_names:
                        self.feature_names = sorted(features.keys())
                    
                    feature_vector = [features.get(name, 0) for name in self.feature_names]
                    X.append(feature_vector)
                    y.append(signal['outcome'])
        
        return np.array(X), np.array(y)
    
    def train_all_models(self, X, y):
        """Train all three ML models and select the best one"""
        if len(X) < 50:
            print(f"[ML] Not enough training data: {len(X)} samples. Need at least 50.")
            return False
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define all models to train
            models_to_train = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    min_samples_split=10,
                    random_state=42
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            }
            
            best_model = None
            best_score = 0
            
            print("[ML] Training and comparing all models...")
            
            for name, model in models_to_train.items():
                try:
                    print(f"[ML] Training {name}...")
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    y_pred = model.predict(X_test_scaled)
                    try:
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        auc_score = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc_score = 0
                    
                    # Store performance
                    self.performance_history[name] = {
                        'train_score': train_score,
                        'test_score': test_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'auc_score': auc_score
                    }
                    
                    print(f"[ML] {name} Results:")
                    print(f"    Train: {train_score:.3f}, Test: {test_score:.3f}")
                    print(f"    CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, AUC: {auc_score:.3f}")
                    
                    # Select best model based on combined score (CV + test score)
                    combined_score = (cv_scores.mean() + test_score) / 2
                    if combined_score > best_score and test_score > 0.55:  # Minimum threshold
                        best_score = combined_score
                        best_model = (name, model, scaler)
                        self.best_model_name = name
                    
                except Exception as e:
                    print(f"[ML] Error training {name}: {e}")
            
            if best_model:
                model_name, model, scaler = best_model
                self.models['best'] = model
                self.scalers['best'] = scaler
                
                print(f"\n[ML] 🏆 Best Model: {model_name} (Score: {best_score:.3f})")
                
                # Detailed evaluation of best model
                y_pred = model.predict(X_test_scaled)
                
                print(f"\n[ML] Detailed {model_name} Performance:")
                print("Classification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                print(confusion_matrix(y_test, y_pred))
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_') and len(self.feature_names) > 0:
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    print(f"\n[ML] Top 10 Most Important Features for {model_name}:")
                    for i in range(min(10, len(indices))):
                        feature_idx = indices[i]
                        if feature_idx < len(self.feature_names):
                            print(f"  {i+1}. {self.feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
                
                return True
            else:
                print("[ML] No model met the minimum performance threshold")
                return False
                
        except Exception as e:
            print(f"[ML] Training error: {e}")
            return False
    
    def predict(self, features):
        """Make prediction with confidence score"""
        if 'best' not in self.models:
            return 0.5, 0
        
        try:
            feature_vector = []
            if isinstance(features, dict):
                # Convert dict to feature vector (ensure consistent ordering)
                for name in self.feature_names:
                    value = features.get(name, 0)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(value)
                    else:
                        feature_vector.append(0)
            else:
                feature_vector = features
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scalers['best'].transform(feature_vector)
            
            # Predict
            probability = self.models['best'].predict_proba(feature_vector_scaled)[0][1]
            prediction = self.models['best'].predict(feature_vector_scaled)[0]
            
            return probability, prediction
            
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return 0.5, 0
    
    def save_models(self, filepath='complete_crypto_ml_models.pkl'):
        """Save all trained models"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'best_model_name': self.best_model_name,
                'performance_history': self.performance_history
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"[ML] All models saved to {filepath}")
            return True
        except Exception as e:
            print(f"[ML] Error saving models: {e}")
            return False
    
    def load_models(self, filepath='complete_crypto_ml_models.pkl'):
        """Load pre-trained models"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_names = model_data.get('feature_names', [])
            self.best_model_name = model_data.get('best_model_name', 'Unknown')
            self.performance_history = model_data.get('performance_history', {})
            
            print(f"[ML] Models loaded from {filepath}")
            print(f"[ML] Best model: {self.best_model_name}")
            if self.best_model_name in self.performance_history:
                perf = self.performance_history[self.best_model_name]
                print(f"[ML] Performance - Test: {perf.get('test_score', 0):.3f}, AUC: {perf.get('auc_score', 0):.3f}")
            return True
        except Exception as e:
            print(f"[ML] Error loading models: {e}")
            return False

# Initialize complete ML classifier
ml_classifier = CompleteCryptoMLClassifier()

def simulate_historical_outcomes():
    """Simulate historical outcomes for demonstration (replace with real data)"""
    historical_signals = []
    np.random.seed(42)
    
    for i in range(500):  # 500 historical signals
        # Create synthetic features that mirror real features
        features = {
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.uniform(-0.1, 0.1),
            'bb_position': np.random.uniform(0, 1),
            'adx': np.random.uniform(10, 50),
            'vol_ratio': np.random.uniform(0.5, 3.0),
            'ema_alignment': np.random.choice([-1, 0, 1]),
            'momentum_1': np.random.uniform(-0.05, 0.05),
            'double_bottom': np.random.choice([0, 1], p=[0.9, 0.1]),
            'hammer': np.random.choice([0, 1], p=[0.95, 0.05]),
            'bos_up': np.random.choice([0, 1], p=[0.8, 0.2]),
            'near_support': np.random.choice([0, 1], p=[0.7, 0.3]),
            'vol_spike': np.random.choice([0, 1], p=[0.8, 0.2]),
            'stoch_oversold': np.random.choice([0, 1], p=[0.85, 0.15]),
            'trend_strength': np.random.uniform(0, 1),
            'volatility_percentile': np.random.uniform(0, 1),
            # Add more features to match the comprehensive feature set
            'rsi_fast': np.random.uniform(20, 80),
            'rsi_slow': np.random.uniform(20, 80),
            'macds': np.random.uniform(-0.1, 0.1),
            'macd_hist': np.random.uniform(-0.05, 0.05),
        }
        
        # Add remaining features with default values
        for feature_name in [
            'bb_width', 'stoch_k', 'stoch_d', 'williams_r', 'atr_pct', 'volatility',
            'cmf', 'mfi', 'roc', 'cci', 'dmp', 'dmn', 'body_size', 'upper_shadow',
            'lower_shadow', 'range_pct', 'buying_pressure', 'selling_pressure',
            'gap_up', 'gap_down', 'higher_high', 'lower_low', 'price_vs_ema20',
            'price_vs_ema200', 'ema20_slope', 'momentum_5', 'price_position_range',
            'consecutive_up', 'consecutive_down', 'macd_crossover', 'double_top',
            'triple_top', 'triple_bottom', 'head_shoulders', 'inv_head_shoulders',
            'rectangle', 'cup_handle', 'rising_wedge', 'falling_wedge',
            'ascending_triangle', 'descending_triangle', 'broadening', 'pattern_count',
            'pattern_strength', 'engulfing_bull', 'engulfing_bear', 'doji',
            'shooting_star', 'morning_star', 'evening_star', 'three_white_soldiers',
            'three_black_crows', 'marubozu_bull', 'marubozu_bear', 'harami_bull',
            'harami_bear', 'piercing', 'dark_cloud', 'inside_bar', 'outside_bar',
            'bos_down', 'fvg_bullish', 'fvg_bearish', 'fvg_count', 'near_resistance',
            'above_key_level', 'below_key_level', 'sr_zone_strength', 'obv_trend',
            'volume_trend_5', 'volume_above_avg', 'cmf_bullish', 'cmf_bearish',
            'mfi_oversold', 'mfi_overbought', 'rsi_oversold', 'rsi_overbought',
            'rsi_divergence', 'stoch_overbought', 'stoch_crossover', 'williams_oversold',
            'williams_overbought', 'cci_oversold', 'cci_overbought', 'bb_squeeze',
            'price_extreme', 'range_expansion'
        ]:
            if feature_name not in features:
                features[feature_name] = np.random.uniform(0, 1) if 'ratio' in feature_name or 'percentile' in feature_name else np.random.choice([0, 1])
        
        # Simulate outcome based on feature logic (replace with real outcomes)
        score = 0
        if features['rsi'] < 30: score += 1
        if features['ema_alignment'] == 1: score += 1
        if features['double_bottom'] == 1: score += 1
        if features['hammer'] == 1: score += 1
        if features['vol_spike'] == 1: score += 1
        if features['bos_up'] == 1: score += 1
        
        outcome = 1 if score >= 3 and np.random.random() > 0.3 else 0
        
        historical_signals.append({
            'features': features,
            'outcome': outcome,
            'timestamp': time.time() - (500 - i) * 3600
        })
    
    return historical_signals

# ================== ENHANCED SIGNAL GENERATION ==================
def enhanced_signal_score(df, direction, chart_patterns, fvg, choch, sr_levels, regime, confluence_score):
    """Enhanced scoring with regime and confluence filters"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    patterns = []
    
    # Base technical score
    if direction == "LONG":
        if last['ema20'] > last['ema50'] > last['ema200']: score += 20
        if 30 < last['rsi'] < 70: score += 10
        if last['macd'] > last['macds'] and prev['macd'] < prev['macds']: score += 15
        if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 80: score += 8
        if last['adx'] > CONFIG['adx_threshold']: score += 12
        if last['vol_ratio'] > 1.5: score += 10
        if last['bb_position'] < 0.2: score += 8
        if last['buying_pressure'] > 0.6: score += 10
        
        # Pattern scoring
        if last['hammer'] == 100: score += 12; patterns.append("Hammer")
        if last['engulfing'] == 100: score += 15; patterns.append("Engulfing")
        if "Double Bottom" in chart_patterns: score += 16; patterns.append("Double Bottom")
        if "Inv Head & Shoulders" in chart_patterns: score += 20; patterns.append("Inv H&S")
        if choch == "up_bos": score += 15; patterns.append("BOS(up)")
        if fvg and fvg[-1][0] == "bullish": score += 12; patterns.append("Bull FVG")
        
    else:  # SHORT
        if last['ema20'] < last['ema50'] < last['ema200']: score += 20
        if 30 < last['rsi'] < 70: score += 10
        if last['macd'] < last['macds'] and prev['macd'] > prev['macds']: score += 15
        if last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > 20: score += 8
        if last['adx'] > CONFIG['adx_threshold']: score += 12
        if last['vol_ratio'] > 1.5: score += 10
        if last['bb_position'] > 0.8: score += 8
        if last['selling_pressure'] > 0.6: score += 10
        
        # Pattern scoring
        if last['shooting_star'] == -100: score += 12; patterns.append("Shooting Star")
        if last['engulfing'] == -100: score += 15; patterns.append("Bear Engulfing")
        if "Double Top" in chart_patterns: score += 16; patterns.append("Double Top")
        if "Head & Shoulders" in chart_patterns: score += 20; patterns.append("H&S")
        if choch == "down_bos": score += 15; patterns.append("BOS(down)")
        if fvg and fvg[-1][0] == "bearish": score += 12; patterns.append("Bear FVG")
    
    # Market regime adjustment
    if regime == "trending":
        score *= 1.2
    elif regime == "ranging":
        score *= 0.8
    elif regime == "volatile":
        score *= 0.9
    
    # Multi-timeframe confluence
    if CONFIG['multi_tf_confluence']:
        if abs(confluence_score) > 1:
            score *= 1.15
        else:
            score *= 0.9
    
    # SR zone bonus
    near_sr = any(abs(last['close']-lvl)/last['close']<0.015 for lvl in sr_levels)
    if near_sr: 
        score += 7
        patterns.append("SR Zone")
    
    return int(score), patterns

def dynamic_risk_management(df, entry, direction, regime):
    """Dynamic risk management based on market conditions"""
    atr = df['atr'].iloc[-1] or 1
    
    # Adjust multipliers based on regime
    if regime == "volatile":
        sl_mult = CONFIG['atr_sl_mult'] * 1.5
        tp_mult = CONFIG['atr_tp_mult'] * 1.3
    elif regime == "ranging":
        sl_mult = CONFIG['atr_sl_mult'] * 0.8
        tp_mult = CONFIG['atr_tp_mult'] * 0.9
    else:
        sl_mult = CONFIG['atr_sl_mult']
        tp_mult = CONFIG['atr_tp_mult']
    
    if direction == "LONG":
        sl = entry - sl_mult * atr
        tps = [entry + tp_mult * atr * i for i in (1, 1.5, 2)]
    else:
        sl = entry + sl_mult * atr
        tps = [entry - tp_mult * atr * i for i in (1, 1.5, 2)]
    
    return sl, tps

def get_enhanced_signal(df, chart_patterns, fvg, choch, sr_levels, smt_result, regime, confluence_score):
    """Enhanced signal generation with complete ML filtering"""
    for direction in ("LONG", "SHORT"):
        # SMT filter
        if smt_result and not ((direction == "LONG" and smt_result == "bullish") or 
                              (direction == "SHORT" and smt_result == "bearish")):
            continue
        
        # Calculate traditional score
        total_score, patterns = enhanced_signal_score(df, direction, chart_patterns, fvg, 
                                                    choch, sr_levels, regime, confluence_score)
        
        if total_score >= CONFIG['conf_threshold']:
            last = df.iloc[-1]
            entry = last['close']
            
            # Complete ML filtering with all three algorithms
            if CONFIG['ml_enabled']:
                features = ml_classifier.create_features(df, chart_patterns, fvg, choch, sr_levels)
                ml_confidence, ml_prediction = ml_classifier.predict(features)
                
                print(f"[ML] {direction} signal - Model: {ml_classifier.best_model_name} | Confidence: {ml_confidence:.3f}")
                
                if ml_confidence < CONFIG['ml_threshold']:
                    print(f"[ML] {direction} signal filtered out (confidence: {ml_confidence:.3f} < {CONFIG['ml_threshold']})")
                    continue
                
                # Combine scores
                final_score = total_score * ml_confidence
            else:
                ml_confidence = 0.5
                final_score = total_score
            
            # Dynamic risk management
            sl, tps = dynamic_risk_management(df, entry, direction, regime)
            rr = round(abs(tps[0]-entry)/abs(entry-sl), 2) if abs(entry-sl) > 0 else 0
            
            return {
                "direction": direction, "entry": entry, "sl": sl, "tps": tps,
                "score": int(final_score), "ml_confidence": ml_confidence,
                "ml_model": ml_classifier.best_model_name,
                "traditional_score": total_score, "patterns": patterns,
                "timestamp": time.time(), "smt": smt_result, "rr": rr,
                "regime": regime, "confluence": confluence_score,
                "fvg": str(fvg[-1]) if fvg else "None", "choch": choch,
                "sr": sr_levels, "atr": last['atr'], "bar_ts": str(df.index[-1])
            }
    
    return None

# ================== TELEGRAM & LOGGING ==================
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    reraise=True
)
async def send_telegram(text):
    """Ultra-reliable Telegram message sender"""
    try:
        # Try direct connection first
        await bot.send_message(
            chat_id=CONFIG["chat_id"],
            text=text,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30
        )
    except httpx.ConnectError as e:
        print(f"Connection failed, will retry: {e}")
        raise
    except httpx.TimeoutException as e:
        print(f"Timeout occurred, will retry: {e}")
        raise
    except Exception as e:
        print(f"Unexpected Telegram error: {e}")
        # Don't retry for other errors
        return False
    return True

def log_signal(signal):
    """Log signal to file"""
    with open(CONFIG['log_file'], 'a') as f:
        f.write(json.dumps(signal, default=str) + "\n")

def clean_stale_signals(signals):
    """Remove stale signals"""
    now = time.time()
    return {k: v for k, v in signals.items() 
            if now - v['timestamp'] < CONFIG['signal_lifetime_sec']}

def update_signal_outcome(signal_key, outcome):
    """Update signal outcome for ML training"""
    global signal_history
    for signal in signal_history:
        if signal.get('key') == signal_key:
            signal['outcome'] = outcome
            break

# ================== MAIN MONITORING LOOP ==================
async def monitor():
    """Main monitoring loop with complete ML integration and heartbeat messages"""
    global open_signals, signal_history
    
    print("[STARTUP] 🚀 Initializing Complete ML Crypto Quant Bot...")
    print(f"[ML] Algorithms: RandomForest, GradientBoosting, XGBoost")
    
    # Heartbeat control
    heartbeats_sent = 0
    CYCLE_COUNT = 0      # increments every scan
    
    # Load ML model
    if CONFIG['ml_enabled']:
        if not ml_classifier.load_models():
            print("[ML] No pre-trained model found. Training new models...")
            historical_signals = simulate_historical_outcomes()
            if len(historical_signals) >= 100:
                X, y = ml_classifier.prepare_training_data(historical_signals)
                if ml_classifier.train_all_models(X, y):
                    ml_classifier.save_models()
                else:
                    print("[ML] Model training failed. Running without ML.")
            else:
                print("[ML] Not enough historical data for training.")
    
    retrain_counter = 0
    
    while True:
        CYCLE_COUNT += 1   # heartbeat counter
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting enhanced ML analysis cycle...")
            
            # Clean stale signals
            open_signals = clean_stale_signals(open_signals)
            
            # Periodic ML retraining
            if CONFIG['ml_enabled'] and len(signal_history) > 50:
                retrain_counter += 1
                if retrain_counter % 48 == 0:
                    print("[ML] Periodic model retraining with new data...")
                    outcomes = [s for s in signal_history if 'outcome' in s]
                    if len(outcomes) >= 100:
                        X, y = ml_classifier.prepare_training_data(outcomes)
                        if ml_classifier.train_all_models(X, y):
                            ml_classifier.save_models()
            
            # --- scan each pair / timeframe ---
            for symbol, ref_symbol in CONFIG["pairs"]:
                for tf in CONFIG["timeframes"]:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analyzing {symbol}-{tf}...")
                    try:
                        df      = await fetch_ohlcv(symbol, tf)
                        df_ref  = await  fetch_ohlcv(ref_symbol, tf)
                        if not df.index.equals(df_ref.index):
                            df_ref = df_ref.reindex(df.index, method='nearest').fillna(method='ffill')
                            
                        df = advanced_indicators(df)
                        df = add_candle_patterns(df)
                        df = order_flow_analysis(df)
                        
                        regime            = detect_market_regime(df)
                        confluence_score, _ = await multi_timeframe_confluence(symbol, ref_symbol)
                        chart_patterns    = get_chart_patterns(df)
                        fvg               = detect_fvg(df)
                        sr_levels         = find_sr_zones(df)
                        choch             = detect_bos_choch(df)
                        smt               = smt_divergence(df, df_ref, window=CONFIG["smt_window"])
                        
                        print(f"[ANALYSIS] {symbol}-{tf}: Regime={regime}, Confluence={confluence_score}, Patterns={len(chart_patterns)}")
                        
                        sig = get_enhanced_signal(df, chart_patterns, fvg, choch,
                                                  sr_levels, smt, regime, confluence_score)
                        price = df['close'].iloc[-1]
                        
                        if sig:
                            sig_key = f"{symbol}-{tf}-{sig['direction']}"
                            sig['key'] = sig_key
                            bar_ts = sig['bar_ts']
                            
                            print(f"[SIGNAL] {symbol} {tf} {sig['direction']} | Score: {sig['score']} | ML({sig['ml_model']}): {sig['ml_confidence']:.3f}")
                            
                            # duplicate check
                            if sig_key in open_signals and open_signals[sig_key].get('bar_ts') == bar_ts:
                                continue
                            
                            # SL / TP handler for open positions
                            if sig_key in open_signals:
                                signal = open_signals[sig_key]
                                if signal['direction'] == 'LONG':
                                    if price <= signal['sl']:
                                        await send_telegram(f"📉 {symbol} LONG Stop Loss hit at {price:.4f} ({tf})")
                                        update_signal_outcome(sig_key, 0)
                                        del open_signals[sig_key]
                                        continue
                                    elif any(price >= tp for tp in signal['tps']):
                                        await send_telegram(f"📈 {symbol} LONG Take Profit hit at {price:.4f} ({tf})")
                                        update_signal_outcome(sig_key, 1)
                                        del open_signals[sig_key]
                                        continue
                                else:  # SHORT
                                    if price >= signal['sl']:
                                        await send_telegram(f"📈 {symbol} SHORT Stop Loss hit at {price:.4f} ({tf})")
                                        update_signal_outcome(sig_key, 0)
                                        del open_signals[sig_key]
                                        continue
                                    elif any(price <= tp for tp in signal['tps']):
                                        await send_telegram(f"📉 {symbol} SHORT Take Profit hit at {price:.4f} ({tf})")
                                        update_signal_outcome(sig_key, 1)
                                        del open_signals[sig_key]
                                        continue
                            
                            # new signal
                            open_signals[sig_key] = sig
                            signal_history.append(sig)
                            log_signal(sig)
                            
                            tps_str    = ', '.join(f"{tp:.4f}" for tp in sig['tps'])
                            patterns_str = ', '.join(sig['patterns'][:3]) if sig['patterns'] else "None"
                            
                            msg = (f"🤖 ML-POWERED SIGNAL\n"
                                   f"📊 {symbol} {tf}: {sig['direction']}\n"
                                   f"🏆 Model: {sig['ml_model']} | 🎯 ML Confidence: {sig['ml_confidence']:.2f}\n"
                                   f"⚡ Score: {sig['score']} (Base: {sig['traditional_score']}) | 📈 Regime: {regime}\n"
                                   f"💰 Entry: {sig['entry']:.4f} | 🛑 SL: {sig['sl']:.4f}\n"
                                   f"🎯 TPs: {tps_str} | ⚡ RR: {sig['rr']}\n"
                                   f"📋 Patterns: {patterns_str}\n"
                                   f"🔄 Confluence: {confluence_score} | ATR: {sig['atr']:.4f}")
                            
                            if sig['smt']: msg += f"\n📊 SMT: {sig['smt']}"
                            if sig['choch']: msg += f" | 🔄 BOS: {sig['choch']}"
                            await send_telegram(msg)
                            
                        else:
                            print(f"[NO SIGNAL] {symbol}-{tf} | Regime: {regime}")
                    
                    except Exception as e:
                        print(f"[ERROR] {symbol}-{tf}: {e}")
            
            # --------------------------------------
            # Heartbeat / no-signal summary
            # --------------------------------------
            open_cnt   = len(open_signals)
            total_cnt  = len(signal_history)
            ml_cnt     = len([s for s in signal_history
                              if s.get('ml_confidence', 0) > CONFIG['ml_threshold']])
            
            if open_cnt == 0:
                msg = (f"💤 No active signals – bot scanning...\n"
                       f"📊 Total generated: {total_cnt} | ML-filtered: {ml_cnt}\n"
                       f"⏱️ Next scan in {CONFIG['scan_interval']}s")
            else:
                msg = (f"🔄 Scan complete – {open_cnt} active signal(s)\n"
                       f"📊 Total generated: {total_cnt} | ML-filtered: {ml_cnt}")
            
            await send_telegram(msg)
            
            print(f"[CYCLE COMPLETE #{CYCLE_COUNT}] Open: {open_cnt} | ML signals: {ml_cnt}/{total_cnt}")
            if ml_classifier.best_model_name:
                print(f"[ML STATUS] Best Model: {ml_classifier.best_model_name}")
            
            print(f"[SLEEP] Waiting {CONFIG['scan_interval']}s...")
            await exchange.close()
            await asyncio.sleep(CONFIG['scan_interval'])
            
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            await asyncio.sleep(60)

# ================== MAIN ENTRY POINT ==================
if __name__ == "__main__":
    print("🚀 Complete ML Crypto Quant Trading Bot")
    print("🤖 RandomForest + GradientBoosting + XGBoost")
    print("=" * 60)
    print(f"Pairs: {len(CONFIG['pairs'])}")
    print(f"Timeframes: {CONFIG['timeframes']}")
    print(f"ML Enabled: {CONFIG['ml_enabled']}")
    print(f"ML Threshold: {CONFIG['ml_threshold']}")
    print(f"Scan Interval: {CONFIG['scan_interval']}s")
    print("=" * 60)
    
    # Start Flask health endpoint
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Start main monitoring loop
    asyncio.run(monitor())
