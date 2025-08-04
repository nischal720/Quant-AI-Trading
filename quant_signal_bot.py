#!/usr/bin/env python

import os, sys, time, asyncio, threading, json
import ccxt
import pandas as pd
import pandas_ta as pta
import talib
import numpy as np
from telegram import Bot
from flask import Flask
from datetime import datetime, timedelta
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
        ("TONUSDT", "BTCUSDT"),("SUIUSDT", "BTCUSDT"), ("LINKUSDT", "BTCUSDT"),("SEIUSDT", "BTCUSDT"),("DOGEUSDT", "BTCUSDT"),("POLUSDT", "BTCUSDT"),
    ],
    "timeframes": ["1h", "4h", "1d"],
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
    "scan_interval": 300,
    "live_data_interval": 5  # Seconds between live data checks
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
live_data_cache = {}  # For storing live market data

# ================== DATA FETCHING ==================
@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
async def fetch_ohlcv(symbol, timeframe, limit=300):
    async with semaphore:
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('ts', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol} {timeframe}: {e}")
            return None

async def fetch_live_price(symbol):
    """Fetch live price using Binance API"""
    try:
        ticker = await exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"Error fetching live price for {symbol}: {e}")
        return None

# ================== ADVANCED INDICATORS ==================
def advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    # Basic EMAs
    df['ema20'] = pta.ema(df['close'], 20)
    df['ema50'] = pta.ema(df['close'], 50)
    df['ema200'] = pta.ema(df['close'], 200)
    
    # Golden/Death Cross detection
    df['golden_cross'] = (df['ema50'] > df['ema200']) & (df['ema50'].shift(1) <= df['ema200'].shift(1))
    df['death_cross'] = (df['ema50'] < df['ema200']) & (df['ema50'].shift(1) >= df['ema200'].shift(1))
    
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

# ================== GOLDEN CROSS DETECTION ==================
def detect_golden_cross(df):
    """Detect golden cross (EMA50 crossing above EMA200) and death cross"""
    if len(df) < 2:
        return None
        
    prev_ema50 = df['ema50'].iloc[-2]
    prev_ema200 = df['ema200'].iloc[-2]
    current_ema50 = df['ema50'].iloc[-1]
    current_ema200 = df['ema200'].iloc[-1]
    
    if prev_ema50 <= prev_ema200 and current_ema50 > current_ema200:
        return "golden"
    elif prev_ema50 >= prev_ema200 and current_ema50 < current_ema200:
        return "death"
    return None

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

            # Skip if any EMA is NaN
            if pd.isna(last['ema20']) or pd.isna(last['ema50']) or pd.isna(last['ema200']):
                print(f"[SKIP] Missing EMA values for {symbol}-{tf}")
                continue

            # Trend confluence
            if last['ema20'] > last['ema50'] > last['ema200']:
                confluence_score += 1
            elif last['ema20'] < last['ema50'] < last['ema200']:
                confluence_score -= 1

        except Exception as e:
            print(f"Multi-TF error for {symbol}-{tf}: {e}")

    return confluence_score, tf_data

# ================== OPTIMIZED ML CLASSIFIER ==================
class OptimizedCryptoMLClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model_name = None
        self.performance_history = {}
        self.last_retrained = 0
        
    def create_features(self, df, chart_patterns, fvg, choch, sr_levels, golden_cross):
        """Create optimized feature set for ML"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Focus on most predictive features
        features = {
            # Trend and momentum
            'ema_alignment': 1 if last['ema20'] > last['ema50'] > last['ema200'] else 
                           -1 if last['ema20'] < last['ema50'] < last['ema200'] else 0,
            'golden_cross': 1 if golden_cross == "golden" else 0,
            'death_cross': 1 if golden_cross == "death" else 0,
            'price_vs_ema200': (last['close'] - last['ema200']) / last['ema200'],
            'adx': last['adx'],
            'rsi': last['rsi'],
            'macd_hist': last['macd_hist'],
            
            # Volatility and volume
            'atr_pct': last['atr_pct'],
            'vol_ratio': last['vol_ratio'],
            'vol_spike': 1 if last['vol_spike'] else 0,
            
            # Price action
            'bb_position': last['bb_position'],
            'body_size': last['body_size'],
            'buying_pressure': last['buying_pressure'],
            
            # Key patterns
            'double_top': 1 if "Double Top" in chart_patterns else 0,
            'double_bottom': 1 if "Double Bottom" in chart_patterns else 0,
            'head_shoulders': 1 if "Head & Shoulders" in chart_patterns else 0,
            'inv_head_shoulders': 1 if "Inv Head & Shoulders" in chart_patterns else 0,
            'hammer': 1 if last['hammer'] == 100 else 0,
            'engulfing_bull': 1 if last['engulfing'] == 100 else 0,
            'engulfing_bear': 1 if last['engulfing'] == -100 else 0,
            
            # Market structure
            'bos_up': 1 if choch == "up_bos" else 0,
            'bos_down': 1 if choch == "down_bos" else 0,
            'near_support': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl < last['close']) else 0,
            'near_resistance': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl > last['close']) else 0,
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
    
    def train_model(self, X, y):
        """Train and select best model with hyperparameter tuning"""
        if len(X) < 100:
            print(f"[ML] Not enough training data: {len(X)} samples. Need at least 100.")
            return False
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost with optimized parameters
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            print("[ML] Training XGBoost model...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Store performance
            self.performance_history = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'auc_score': auc_score
            }
            
            print(f"[ML] XGBoost Results:")
            print(f"    Train: {train_score:.3f}, Test: {test_score:.3f}")
            print(f"    CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, AUC: {auc_score:.3f}")
            
            # Only keep if meets minimum threshold
            if test_score > 0.6 and auc_score > 0.65:
                self.models['best'] = model
                self.scalers['best'] = scaler
                self.best_model_name = "XGBoost"
                self.last_retrained = time.time()
                
                print(f"\n[ML] 🏆 Model trained successfully (Score: {test_score:.3f})")
                
                # Feature importance
                if hasattr(model, 'feature_importances_') and len(self.feature_names) > 0:
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    print(f"\n[ML] Top 10 Most Important Features:")
                    for i in range(min(10, len(indices))):
                        feature_idx = indices[i]
                        if feature_idx < len(self.feature_names):
                            print(f"  {i+1}. {self.feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
                
                return True
            else:
                print("[ML] Model did not meet performance threshold")
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
                # Convert dict to feature vector
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
    
    def save_model(self, filepath='optimized_crypto_ml_model.pkl'):
        """Save trained model"""
        try:
            model_data = {
                'model': self.models.get('best'),
                'scaler': self.scalers.get('best'),
                'feature_names': self.feature_names,
                'performance': self.performance_history,
                'last_retrained': self.last_retrained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"[ML] Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"[ML] Error saving model: {e}")
            return False
    
    def load_model(self, filepath='optimized_crypto_ml_model.pkl'):
        """Load pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models['best'] = model_data.get('model')
            self.scalers['best'] = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.performance_history = model_data.get('performance', {})
            self.last_retrained = model_data.get('last_retrained', 0)
            
            print(f"[ML] Model loaded from {filepath}")
            if self.performance_history:
                print(f"[ML] Performance - Test: {self.performance_history.get('test_score', 0):.3f}, AUC: {self.performance_history.get('auc_score', 0):.3f}")
            return True
        except Exception as e:
            print(f"[ML] Error loading model: {e}")
            return False

# Initialize optimized ML classifier
ml_classifier = OptimizedCryptoMLClassifier()

def simulate_historical_outcomes():
    """Simulate historical outcomes for demonstration"""
    historical_signals = []
    np.random.seed(42)
    
    for i in range(1000):  # 1000 historical signals
        features = {
            'ema_alignment': np.random.choice([-1, 0, 1]),
            'golden_cross': np.random.choice([0, 1], p=[0.95, 0.05]),
            'death_cross': np.random.choice([0, 1], p=[0.95, 0.05]),
            'price_vs_ema200': np.random.uniform(-0.1, 0.1),
            'adx': np.random.uniform(10, 50),
            'rsi': np.random.uniform(20, 80),
            'macd_hist': np.random.uniform(-0.05, 0.05),
            'atr_pct': np.random.uniform(0.5, 5),
            'vol_ratio': np.random.uniform(0.5, 3.0),
            'vol_spike': np.random.choice([0, 1], p=[0.8, 0.2]),
            'bb_position': np.random.uniform(0, 1),
            'body_size': np.random.uniform(0, 0.1),
            'buying_pressure': np.random.uniform(0.3, 0.7),
            'double_top': np.random.choice([0, 1], p=[0.9, 0.1]),
            'double_bottom': np.random.choice([0, 1], p=[0.9, 0.1]),
            'head_shoulders': np.random.choice([0, 1], p=[0.95, 0.05]),
            'inv_head_shoulders': np.random.choice([0, 1], p=[0.95, 0.05]),
            'hammer': np.random.choice([0, 1], p=[0.95, 0.05]),
            'engulfing_bull': np.random.choice([0, 1], p=[0.95, 0.05]),
            'engulfing_bear': np.random.choice([0, 1], p=[0.95, 0.05]),
            'bos_up': np.random.choice([0, 1], p=[0.9, 0.1]),
            'bos_down': np.random.choice([0, 1], p=[0.9, 0.1]),
            'near_support': np.random.choice([0, 1], p=[0.7, 0.3]),
            'near_resistance': np.random.choice([0, 1], p=[0.7, 0.3]),
        }
        
        # Simulate outcome based on feature logic
        score = 0
        if features['golden_cross'] == 1: score += 3
        if features['ema_alignment'] == 1: score += 2
        if features['double_bottom'] == 1: score += 2
        if features['hammer'] == 1: score += 1
        if features['vol_spike'] == 1: score += 1
        if features['bos_up'] == 1: score += 2
        
        outcome = 1 if score >= 5 and np.random.random() > 0.3 else 0
        
        historical_signals.append({
            'features': features,
            'outcome': outcome,
            'timestamp': time.time() - (1000 - i) * 3600
        })
    
    return historical_signals

# ================== SIGNAL MANAGEMENT ==================
def enhanced_signal_score(df, direction, chart_patterns, fvg, choch, sr_levels, regime, confluence_score, golden_cross):
    """Enhanced scoring with golden cross integration"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    patterns = []
    
    # Golden cross bonus
    if direction == "LONG" and golden_cross == "golden":
        score += 25
        patterns.append("Golden Cross")
    elif direction == "SHORT" and golden_cross == "death":
        score += 25
        patterns.append("Death Cross")
    
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

def get_enhanced_signal(df, chart_patterns, fvg, choch, sr_levels, smt_result, regime, confluence_score, golden_cross):
    """Enhanced signal generation with ML filtering"""
    for direction in ("LONG", "SHORT"):
        # SMT filter
        if smt_result and not ((direction == "LONG" and smt_result == "bullish") or 
                              (direction == "SHORT" and smt_result == "bearish")):
            continue
        
        # Calculate traditional score
        total_score, patterns = enhanced_signal_score(df, direction, chart_patterns, fvg, 
                                                    choch, sr_levels, regime, confluence_score, golden_cross)
        
        if total_score >= CONFIG['conf_threshold']:
            last = df.iloc[-1]
            entry = last['close']
            
            # ML filtering
            if CONFIG['ml_enabled']:
                features = ml_classifier.create_features(df, chart_patterns, fvg, choch, sr_levels, golden_cross)
                ml_confidence, ml_prediction = ml_classifier.predict(features)
                
                print(f"[ML] {direction} signal - Confidence: {ml_confidence:.3f}")
                
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
                "traditional_score": total_score, "patterns": patterns,
                "timestamp": time.time(), "smt": smt_result, "rr": rr,
                "regime": regime, "confluence": confluence_score,
                "fvg": str(fvg[-1]) if fvg else "None", "choch": choch,
                "sr": sr_levels, "atr": last['atr'], "bar_ts": str(df.index[-1]),
                "golden_cross": golden_cross
            }
    
    return None

# ================== LIVE DATA MONITORING ==================
async def monitor_live_data():
    """Continuously monitor live prices for SL/TP hits"""
    global open_signals
    
    while True:
        try:
            # Refresh live prices
            for symbol, _ in CONFIG["pairs"]:
                live_price = await fetch_live_price(symbol)
                if live_price:
                    live_data_cache[symbol] = live_price
            
            # Check SL/TP for all open signals
            current_time = time.time()
            signals_to_remove = []
            
            for sig_key, signal in list(open_signals.items()):
                symbol = sig_key.split('-')[0]
                live_price = live_data_cache.get(symbol)
                
                if not live_price:
                    continue
                
                # Check if signal is expired
                if current_time - signal['timestamp'] > CONFIG['signal_lifetime_sec']:
                    signals_to_remove.append(sig_key)
                    await send_telegram(f"⌛ {symbol} signal expired ({signal['direction']})")
                    update_signal_outcome(sig_key, 0)  # Mark as expired
                    continue
                
                # Check SL/TP hits
                direction = signal['direction']
                entry = signal['entry']
                sl = signal['sl']
                tps = signal['tps']
                
                if direction == "LONG":
                    if live_price <= sl:
                        await send_telegram(f"📉 {symbol} LONG Stop Loss hit at {live_price:.4f}")
                        update_signal_outcome(sig_key, 0)
                        signals_to_remove.append(sig_key)
                    else:
                        for i, tp in enumerate(tps):
                            if live_price >= tp:
                                await send_telegram(f"📈 {symbol} LONG Take Profit #{i+1} hit at {live_price:.4f}")
                                if i == len(tps) - 1:  # Final TP
                                    update_signal_outcome(sig_key, 1)
                                    signals_to_remove.append(sig_key)
                else:  # SHORT
                    if live_price >= sl:
                        await send_telegram(f"📈 {symbol} SHORT Stop Loss hit at {live_price:.4f}")
                        update_signal_outcome(sig_key, 0)
                        signals_to_remove.append(sig_key)
                    else:
                        for i, tp in enumerate(tps):
                            if live_price <= tp:
                                await send_telegram(f"📉 {symbol} SHORT Take Profit #{i+1} hit at {live_price:.4f}")
                                if i == len(tps) - 1:  # Final TP
                                    update_signal_outcome(sig_key, 1)
                                    signals_to_remove.append(sig_key)
            
            # Remove completed signals
            for sig_key in set(signals_to_remove):
                if sig_key in open_signals:
                    del open_signals[sig_key]
            
            await asyncio.sleep(CONFIG['live_data_interval'])
            
        except Exception as e:
            print(f"[LIVE MONITOR ERROR] {e}")
            await asyncio.sleep(10)

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
        await bot.send_message(
            chat_id=CONFIG["chat_id"],
            text=text,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30
        )
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

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
    """Main monitoring loop with golden cross and ML integration"""
    global open_signals, signal_history
    
    print("[STARTUP] 🚀 Initializing Enhanced Crypto Quant Bot...")
    print(f"[ML] Optimized XGBoost classifier")
    
    # Heartbeat control
    heartbeats_sent = 0
    CYCLE_COUNT = 0
    
    # Load ML model
    if CONFIG['ml_enabled']:
        if not ml_classifier.load_model():
            print("[ML] No pre-trained model found. Training new model...")
            historical_signals = simulate_historical_outcomes()
            if len(historical_signals) >= 100:
                X, y = ml_classifier.prepare_training_data(historical_signals)
                if ml_classifier.train_model(X, y):
                    ml_classifier.save_model()
                else:
                    print("[ML] Model training failed. Running without ML.")
            else:
                print("[ML] Not enough historical data for training.")
    
    # Start live data monitoring
    asyncio.create_task(monitor_live_data())
    
    retrain_counter = 0
    
    while True:
        CYCLE_COUNT += 1
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting analysis cycle #{CYCLE_COUNT}...")
            
            # Clean stale signals
            open_signals = clean_stale_signals(open_signals)
            
            # Periodic ML retraining
            if CONFIG['ml_enabled'] and len(signal_history) > 100:
                retrain_counter += 1
                if retrain_counter % 24 == 0:  # Retrain every 24 cycles
                    print("[ML] Periodic model retraining with new data...")
                    outcomes = [s for s in signal_history if 'outcome' in s]
                    if len(outcomes) >= 100:
                        X, y = ml_classifier.prepare_training_data(outcomes)
                        if ml_classifier.train_model(X, y):
                            ml_classifier.save_model()
            
            # Process each pair and timeframe
            for symbol, ref_symbol in CONFIG["pairs"]:
                for tf in CONFIG["timeframes"]:
                    try:
                        df = await fetch_ohlcv(symbol, tf)
                        df_ref = await fetch_ohlcv(ref_symbol, tf)
                        
                        # Handle data alignment
                        if not df.index.equals(df_ref.index):
                            df_ref = df_ref.reindex(df.index, method='nearest').fillna(method='ffill')
                            
                        df = advanced_indicators(df)
                        df = add_candle_patterns(df)
                        df = order_flow_analysis(df)
                        
                        # Detect key patterns and regimes
                        regime = detect_market_regime(df)
                        confluence_score, _ = await multi_timeframe_confluence(symbol, ref_symbol)
                        chart_patterns = get_chart_patterns(df)
                        fvg = detect_fvg(df)
                        sr_levels = find_sr_zones(df)
                        choch = detect_bos_choch(df)
                        smt = smt_divergence(df, df_ref, window=CONFIG["smt_window"])
                        golden_cross = detect_golden_cross(df)
                        
                        # Generate signal
                        sig = get_enhanced_signal(df, chart_patterns, fvg, choch,
                                                sr_levels, smt, regime, confluence_score, golden_cross)
                        
                        if sig:
                            sig_key = f"{symbol}-{tf}-{sig['direction']}"
                            sig['key'] = sig_key
                            bar_ts = sig['bar_ts']
                            
                            # Skip duplicate signals
                            if sig_key in open_signals and open_signals[sig_key].get('bar_ts') == bar_ts:
                                continue
                            
                            # Add golden cross info to message
                            cross_info = ""
                            if golden_cross == "golden":
                                cross_info = " | ✨ Golden Cross"
                            elif golden_cross == "death":
                                cross_info = " | ☠️ Death Cross"
                            
                            print(f"[SIGNAL] {symbol} {tf} {sig['direction']} | Score: {sig['score']} | ML: {sig['ml_confidence']:.3f}{cross_info}")
                            
                            # Store and notify
                            open_signals[sig_key] = sig
                            signal_history.append(sig)
                            log_signal(sig)
                            
                            tps_str = ', '.join(f"{tp:.4f}" for tp in sig['tps'])
                            patterns_str = ', '.join(sig['patterns'][:3]) if sig['patterns'] else "None"
                            
                            msg = (f"🤖 ENHANCED SIGNAL\n"
                                   f"📊 {symbol} {tf}: {sig['direction']}{cross_info}\n"
                                   f"🏆 ML Confidence: {sig['ml_confidence']:.2f} | ⚡ Score: {sig['score']}\n"
                                   f"💰 Entry: {sig['entry']:.4f} | 🛑 SL: {sig['sl']:.4f}\n"
                                   f"🎯 TPs: {tps_str} | ⚡ RR: {sig['rr']}\n"
                                   f"📋 Patterns: {patterns_str}")
                            
                            if sig['smt']: 
                                msg += f"\n📊 SMT: {sig['smt']}"
                            await send_telegram(msg)
                    
                    except Exception as e:
                        print(f"[ERROR] {symbol}-{tf}: {e}")
            
            # Send heartbeat
            open_cnt = len(open_signals)
            total_signals = len(signal_history)
            ml_signals = len([s for s in signal_history if s.get('ml_confidence', 0) > CONFIG['ml_threshold']])
            
            if CYCLE_COUNT % 6 == 0:  # Send status every 6 cycles
                status_msg = (f"📊 BOT STATUS\n"
                              f"🔄 Cycle: #{CYCLE_COUNT} | ⏱ Uptime: {timedelta(seconds=int(time.time()-bot_status['last_update']))}\n"
                              f"📈 Active Signals: {open_cnt}\n"
                              f"📋 Total Signals: {total_signals} | 🤖 ML Signals: {ml_signals}")
                if ml_classifier.performance_history:
                    perf = ml_classifier.performance_history
                    status_msg += f"\n🤖 ML Perf: Test {perf.get('test_score', 0):.3f} | AUC {perf.get('auc_score', 0):.3f}"
                await send_telegram(status_msg)
            
            print(f"[CYCLE COMPLETE] Open: {open_cnt} | Total: {total_signals} | Next in {CONFIG['scan_interval']}s")
            await exchange.close()
            await asyncio.sleep(CONFIG['scan_interval'])
            
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            await asyncio.sleep(60)

# ================== MAIN ENTRY POINT ==================
if __name__ == "__main__":
    print("🚀 Enhanced Crypto Quant Trading Bot")
    print("✨ Golden Cross Detection | 🤖 Optimized ML | 📊 Live Monitoring")
    print("=" * 60)
    print(f"Pairs: {len(CONFIG['pairs'])}")
    print(f"Timeframes: {CONFIG['timeframes']}")
    print(f"ML Enabled: {CONFIG['ml_enabled']}")
    print(f"Live Data Interval: {CONFIG['live_data_interval']}s")
    print("=" * 60)
    
    # Start Flask health endpoint
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Start main monitoring loop
    asyncio.run(monitor())