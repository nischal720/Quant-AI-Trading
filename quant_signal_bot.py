import os, sys, time, asyncio, threading, json
import ccxt.async_support as ccxt_async
import pandas as pd
import pandas_ta as pta
import talib
import numpy as np
from telegram import Bot
from flask import Flask, request
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle
import warnings
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from cryptography.fernet import Fernet
import logging
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict
import re
from functools import wraps
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import CollectionInvalid
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# Configuration with defaults and validation
CONFIG = {
    "telegram_token": os.environ.get("BOT_TOKEN", "YOUR_TELEGRAM_TOKEN_HERE"),
    "chat_id": os.environ.get("CHANNEL_ID", "YOUR_CHAT_ID_HERE"),
    "exchanges": {
        "binance": {
            "apiKey": os.environ.get('BINANCE_API_KEY', ''),
            "secret": os.environ.get('BINANCE_SECRET', ''),
            "enableRateLimit": True,
            "rateLimit": 10000,
            "urls": {
                'api': {
                    'public': 'https://fapi.binance.com/fapi/v1',
                    'private': 'https://fapi.binance.com/fapi/v1',
                    'backup': 'https://api.binance.com/api/v3'
                }
            },
            "options": {
                'defaultType': 'future',
                'defaultSubType': 'linear',
                'adjustForTimeDifference': True,
                'recvWindow': 10000
            }
        }
    },
    "pairs": ["BNB/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT", "TON/USDT:USDT"],
    "timeframes": ["1h","2h", "4h", "1d"],
    "conf_threshold": 80,
    "atr_sl_mult": 1.8,
    "atr_tp_mult": 2.5,
    "adx_threshold": 25,
    "vol_mult": 2.0,
    "signal_lifetime_sec": 6*3600,
    "ml_enabled": True,
    "ml_threshold": 0.65,
    "market_regime_enabled": True,
    "multi_tf_confluence": True,
    "volatility_filter": True,
    "order_flow_analysis": True,
    "scan_interval": 60,
    "live_data_interval": 5,
    "cache_ttl": 120,
    "encryption_key": os.environ.get("ENCRYPTION_KEY", Fernet.generate_key().decode()),
    "flask_rate_limit": {"requests": 100, "window": 60},
    "telegram_rate_limit": {"requests": 20, "window": 60},
    "indicator_plugins": ["ema", "rsi", "macd", "bbands", "stoch", "willr", "adx", "atr", "volatility", "obv", "cmf", "mfi", "candle_patterns", "order_flow", "vwap", "trendline"],
    "scoring_plugins": ["ema_alignment", "golden_cross", "rsi", "macd", "stoch", "adx", "volume", "bbands", "order_flow", "chart_patterns", "smc", "vwap", "trendline_breakout"],
    "min_volume_multiple": 0.5,
    "min_body_ratio": 0.3,
    "tf_weights": {"15m": 0.5,  "1h": 1.0, "2h": 1.2, "4h": 1.5, "1d": 2.0},
    "economic_calendar_url": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "event_impact_threshold": "high",
    "event_window_hours": 1,
    "event_currencies": ["USD"],
    "mongodb": {
        "uri": os.environ.get("MONGODB_URI", ""),
        "database": "crypto_trading_bot",
        "collections": {
            "signals": "signals",
            "ohlcv": "ohlcv",
            "model_performance": "model_performance",
            "economic_events": "economic_events",
            "ml_models": "ml_models",
            "bot_state": "bot_state"
        }
    }
}

# Global state
bot_status = {"status": "starting", "last_update": time.time(), "signals_count": 0}
semaphore = asyncio.Semaphore(20)  # Increased for better parallelism
live_data_cache = {}
invalid_symbols = set()
fernet = Fernet(CONFIG["encryption_key"].encode())

# Rate limiting storage
flask_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})
telegram_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})

# Initialize components
bot = Bot(token=CONFIG["telegram_token"])
app = Flask(__name__)
mongo_client = AsyncIOMotorClient(CONFIG["mongodb"]["uri"])
db = mongo_client[CONFIG["mongodb"]["database"]]

# ================== SECURITY ENHANCEMENTS ==================
def rate_limit(limit_type: str):
    def decorator(f):
        @wraps(f)
        async def wrapped(*args, **kwargs):
            client_ip = request.remote_addr if limit_type == "flask" else "telegram"
            limits = CONFIG[f"{limit_type}_rate_limit"]
            now = time.time()
            rate_info = flask_rate_limits[client_ip] if limit_type == "flask" else telegram_rate_limits[client_ip]
            
            if now > rate_info["reset_time"]:
                rate_info["count"] = 0
                rate_info["reset_time"] = now + limits["window"]
            
            if rate_info["count"] >= limits["requests"]:
                logger.warning(f"Rate limit exceeded for {client_ip} ({limit_type})")
                if limit_type == "flask":
                    return json.dumps({"error": "Rate limit exceeded"}), 429
                return False
            
            rate_info["count"] += 1
            return await f(*args, **kwargs)
        return wrapped
    return decorator

def sanitize_string(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,:;@#$%^&*()_+-=]', '', text)
    return text[:500]

@app.route('/ping')
@rate_limit("flask")
def home():
    return json.dumps({
        "message": "Crypto Quant Trading Bot is Running",
        "status": bot_status["status"],
        "uptime": time.time() - bot_status["last_update"],
    })

@app.route('/health')
@rate_limit("flask")
def health():
    return "ok"

@app.route('/mongo_health')
@rate_limit("flask")
async def mongo_health():
    try:
        await db.command("ping")
        return json.dumps({"status": "MongoDB OK"})
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return json.dumps({"error": "MongoDB connection failed"}), 500

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# ================== CONFIG VALIDATION ==================
def validate_config():
    required = ["telegram_token", "chat_id", "pairs", "timeframes", "indicator_plugins", "scoring_plugins"]
    for key in required:
        if not CONFIG[key] or (isinstance(CONFIG[key], str) and CONFIG[key].startswith("<")):
            logger.error(f"Missing or invalid {key} in configuration")
            sys.exit(1)
    if not all(isinstance(p, str) and p.endswith('USDT') for p in CONFIG["pairs"]):
        logger.error("Invalid trading pairs format")
        sys.exit(1)
    if not all(tf in ["1m", "5m", "15m", "1h","2h", "4h", "12h", "1d"] for tf in CONFIG["timeframes"]):
        logger.error("Invalid timeframes")
        sys.exit(1)
    logger.info("Configuration validated successfully")

# ================== PLUGIN SYSTEM ==================
class IndicatorPlugin:
    def __init__(self, name: str):
        self.name = name

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class ScoringPlugin:
    def __init__(self, name: str):
        self.name = name

    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        raise NotImplementedError

class EMAPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema20'] = pta.ema(df['close'], 20)
        df['ema50'] = pta.ema(df['close'], 50)
        df['ema200'] = pta.ema(df['close'], 200)
        df['golden_cross'] = (df['ema50'] > df['ema200']) & (df['ema50'].shift(1) <= df['ema200'].shift(1))
        df['death_cross'] = (df['ema50'] < df['ema200']) & (df['ema50'].shift(1) >= df['ema200'].shift(1))
        return df

class RSIPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = pta.rsi(df['close'], 14)
        df['rsi_fast'] = pta.rsi(df['close'], 7)
        df['rsi_slow'] = pta.rsi(df['close'], 21)
        return df

class MACDPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        macd = pta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macds'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        return df

class BollingerBandsPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = pta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

class StochasticPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        stoch = pta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        return df

class WilliamsRPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['williams_r'] = pta.willr(df['high'], df['low'], df['close'])
        return df

class ADXPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        adx_data = pta.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx_data['ADX_14']
        df['dmp'] = adx_data['DMP_14']
        df['dmn'] = adx_data['DMN_14']
        return df

class ATRPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['atr'] = pta.atr(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close'] * 100
        return df

class VolatilityPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        vol_ma = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] > vol_ma * CONFIG['vol_mult']
        df['vol_ratio'] = df['volume'] / vol_ma
        return df

class OBVPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['obv'] = pta.obv(df['close'], df['volume'])
        return df

class CMFPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['cmf'] = pta.cmf(df['high'], df['low'], df['close'], df['volume'])
        return df

class MFIPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['mfi'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'])
        return df

class CandlePatternsPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
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
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['outside_bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        return df

class OrderFlowPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['upper_rejection'] = (df['high'] == df['high'].rolling(5).max()) & (df['close'] < df['high'] * 0.98)
        df['lower_rejection'] = (df['low'] == df['low'].rolling(5).min()) & (df['close'] > df['low'] * 1.02)
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        return df

class VWAPPlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

class TrendLinePlugin(IndicatorPlugin):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Find pivot highs and lows
        df['pivot_low'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
        df['pivot_high'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])

        n = len(df)
        x = np.arange(n)

        # Uptrend line (support): fit on pivot lows
        pivot_low_indices = np.where(df['pivot_low'])[0]
        if len(pivot_low_indices) >= 2:
            coef = np.polyfit(pivot_low_indices, df['low'].iloc[pivot_low_indices], 1)
            df['up_trend_line'] = coef[0] * x + coef[1]
        else:
            df['up_trend_line'] = np.nan

        # Downtrend line (resistance): fit on pivot highs
        pivot_high_indices = np.where(df['pivot_high'])[0]
        if len(pivot_high_indices) >= 2:
            coef = np.polyfit(pivot_high_indices, df['high'].iloc[pivot_high_indices], 1)
            df['down_trend_line'] = coef[0] * x + coef[1]
        else:
            df['down_trend_line'] = np.nan

        # Breakout
        df['up_breakout'] = (df['close'] > df['down_trend_line']) & (df['close'].shift(1) <= df['down_trend_line'].shift(1))
        df['down_breakout'] = (df['close'] < df['up_trend_line']) & (df['close'].shift(1) >= df['up_trend_line'].shift(1))

        # False breakout: breakout but reverses in next 3 bars
        df['false_up_breakout'] = False
        df['false_down_breakout'] = False
        for i in range(len(df) - 3):
            if df['up_breakout'].iloc[i]:
                if df['close'].iloc[i+3] < df['down_trend_line'].iloc[i+3]:
                    df.loc[df.index[i], 'false_up_breakout'] = True
            if df['down_breakout'].iloc[i]:
                if df['close'].iloc[i+3] > df['up_trend_line'].iloc[i+3]:
                    df.loc[df.index[i], 'false_down_breakout'] = True

        return df

class CustomIndicatorPlugin(IndicatorPlugin):
    def __init__(self, name: str, compute_fn: Callable[[pd.DataFrame], pd.DataFrame]):
        super().__init__(name)
        self.compute_fn = compute_fn

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_fn(df)

# Scoring Plugins
class EMAAlignmentPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['ema20'] > last['ema50'] > last['ema200']:
            score += 20
            patterns.append("EMA Alignment")
        elif direction == "SHORT" and last['ema20'] < last['ema50'] < last['ema200']:
            score += 20
            patterns.append("EMA Alignment")
        return score, patterns

class GoldenCrossPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, golden_cross: str, **context) -> Tuple[int, List[str]]:
        score, patterns = 0, []
        if direction == "LONG" and golden_cross == "golden":
            score += 25
            patterns.append("Golden Cross")
        elif direction == "SHORT" and golden_cross == "death":
            score += 25
            patterns.append("Death Cross")
        return score, patterns

class RSIPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if 30 < last['rsi'] < 70:
            score += 10
            patterns.append("RSI Neutral")
        return score, patterns

class MACDPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last, prev = df.iloc[-1], df.iloc[-2]
        score, patterns = 0, []
        if direction == "LONG" and last['macd'] > last['macds'] and prev['macd'] < prev['macds']:
            score += 15
            patterns.append("MACD Crossover")
        elif direction == "SHORT" and last['macd'] < last['macds'] and prev['macd'] > prev['macds']:
            score += 15
            patterns.append("MACD Crossover")
        return score, patterns

class StochasticPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 80:
            score += 8
            patterns.append("Stochastic Bullish")
        elif direction == "SHORT" and last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > 20:
            score += 8
            patterns.append("Stochastic Bearish")
        return score, patterns

class ADXPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        threshold = context.get("adx_threshold", CONFIG['adx_threshold'])
        if last['adx'] > threshold:
            score += 12
            patterns.append("ADX Strong Trend")
        return score, patterns

class VolumePluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if last['vol_ratio'] > 1.5:
            score += 10
            patterns.append("Volume Spike")
        if last['vol_spike']:
            score += 5
            patterns.append("High Volume")
        return score, patterns

class BollingerBandsPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['bb_position'] < 0.2:
            score += 8
            patterns.append("BB Lower Band")
        elif direction == "SHORT" and last['bb_position'] > 0.8:
            score += 8
            patterns.append("BB Upper Band")
        return score, patterns

class OrderFlowPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['buying_pressure'] > 0.6:
            score += 10
            patterns.append("Buying Pressure")
        elif direction == "SHORT" and last['selling_pressure'] > 0.6:
            score += 10
            patterns.append("Selling Pressure")
        return score, patterns

class ChartPatternsPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, chart_patterns: List[str], **context) -> Tuple[int, List[str]]:
        score, patterns = 0, []
        if direction == "LONG":
            if "Double Bottom" in chart_patterns:
                score += 16
                patterns.append("Double Bottom")
            if "Inv Head & Shoulders" in chart_patterns:
                score += 20
                patterns.append("Inv H&S")
            if "Triple Bottom" in chart_patterns:
                score += 16
                patterns.append("Triple Bottom")
            if "Ascending Triangle" in chart_patterns:
                score += 15
                patterns.append("Ascending Triangle")
            if "Falling Wedge" in chart_patterns:
                score += 15
                patterns.append("Falling Wedge")
            if "Cup & Handle" in chart_patterns:
                score += 18
                patterns.append("Cup & Handle")
            if "Pipe Bottom" in chart_patterns:
                score += 16
                patterns.append("Pipe Bottom")
            if "Gap Up" in chart_patterns:
                score += 10
                patterns.append("Gap Up")
            if "Narrow Range" in chart_patterns:
                score += 12
                patterns.append("Narrow Range")
            if "Flag" in chart_patterns or "Pennant" in chart_patterns:
                score += 14  # Continuation in up trend
                patterns.append("Flag/Pennant")
        elif direction == "SHORT":
            if "Double Top" in chart_patterns:
                score += 16
                patterns.append("Double Top")
            if "Head & Shoulders" in chart_patterns:
                score += 20
                patterns.append("H&S")
            if "Triple Top" in chart_patterns:
                score += 16
                patterns.append("Triple Top")
            if "Descending Triangle" in chart_patterns:
                score += 15
                patterns.append("Descending Triangle")
            if "Rising Wedge" in chart_patterns:
                score += 15
                patterns.append("Rising Wedge")
            if "Gap Down" in chart_patterns:
                score += 10
                patterns.append("Gap Down")
            if "Flag" in chart_patterns or "Pennant" in chart_patterns:
                score += 14  # Continuation in down trend
                patterns.append("Flag/Pennant")
        return score, patterns

class SMCPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, fvg: List, choch: str, sr_levels: List[float], **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG":
            if choch == "up_bos":
                score += 15
                patterns.append("BOS(up)")
            if fvg and fvg[-1][0] == "bullish":
                score += 12
                patterns.append("Bull FVG")
        elif direction == "SHORT":
            if choch == "down_bos":
                score += 15
                patterns.append("BOS(down)")
            if fvg and fvg[-1][0] == "bearish":
                score += 12
                patterns.append("Bear FVG")
        if any(abs(last['close']-lvl)/last['close']<0.015 for lvl in sr_levels):
            score += 7
            patterns.append("SR Zone")
        return score, patterns

class VWAPPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last.get('vwap') and last['close'] > last['vwap']:
            score += 10
            patterns.append("Above VWAP")
        elif direction == "SHORT" and last.get('vwap') and last['close'] < last['vwap']:
            score += 10
            patterns.append("Below VWAP")
        return score, patterns

class TrendLineBreakoutPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last.get('up_breakout', False):
            score += 15
            patterns.append("Uptrend Breakout")
        elif direction == "SHORT" and last.get('down_breakout', False):
            score += 15
            patterns.append("Downtrend Breakout")
        # Check recent false breakouts
        prev = df.iloc[-10:-1]
        if direction == "LONG" and any(prev.get('false_up_breakout', False)):
            score -= 10
            patterns.append("Recent False Up Breakout")
        elif direction == "SHORT" and any(prev.get('false_down_breakout', False)):
            score -= 10
            patterns.append("Recent False Down Breakout")
        return score, patterns

# Plugin Registry
INDICATOR_PLUGINS = {
    "ema": EMAPlugin("ema"),
    "rsi": RSIPlugin("rsi"),
    "macd": MACDPlugin("macd"),
    "bbands": BollingerBandsPlugin("bbands"),
    "stoch": StochasticPlugin("stoch"),
    "willr": WilliamsRPlugin("willr"),
    "adx": ADXPlugin("adx"),
    "atr": ATRPlugin("atr"),
    "volatility": VolatilityPlugin("volatility"),
    "obv": OBVPlugin("obv"),
    "cmf": CMFPlugin("cmf"),
    "mfi": MFIPlugin("mfi"),
    "candle_patterns": CandlePatternsPlugin("candle_patterns"),
    "order_flow": OrderFlowPlugin("order_flow"),
    "vwap": VWAPPlugin("vwap"),
    "trendline": TrendLinePlugin("trendline")
}

SCORING_PLUGINS = {
    "ema_alignment": EMAAlignmentPlugin("ema_alignment"),
    "golden_cross": GoldenCrossPlugin("golden_cross"),
    "rsi": RSIPluginScoring("rsi"),
    "macd": MACDPluginScoring("macd"),
    "stoch": StochasticPluginScoring("stoch"),
    "adx": ADXPluginScoring("adx"),
    "volume": VolumePluginScoring("volume"),
    "bbands": BollingerBandsPluginScoring("bbands"),
    "order_flow": OrderFlowPluginScoring("order_flow"),
    "chart_patterns": ChartPatternsPluginScoring("chart_patterns"),
    "smc": SMCPluginScoring("smc"),
    "vwap": VWAPPluginScoring("vwap"),
    "trendline_breakout": TrendLineBreakoutPlugin("trendline_breakout")
}

# ================== MONGODB INITIALIZATION ==================
async def init_mongo_indexes():
    for col_name in CONFIG["mongodb"]["collections"].values():
        try:
            await db.create_collection(col_name)
            logger.info(f"Collection '{col_name}' created.")
        except CollectionInvalid:
            logger.info(f"Collection '{col_name}' already exists.")
        except Exception as e:
            logger.warning(f"Could not create collection '{col_name}': {str(e)}")

    signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
    ohlcv_collection = db[CONFIG["mongodb"]["collections"]["ohlcv"]]
    economic_collection = db[CONFIG["mongodb"]["collections"]["economic_events"]]
    performance_collection = db[CONFIG["mongodb"]["collections"]["model_performance"]]
    ml_models_collection = db[CONFIG["mongodb"]["collections"]["ml_models"]]
    bot_state_collection = db[CONFIG["mongodb"]["collections"]["bot_state"]]

    # FIXED: Create proper indexes with correct fields
    await ohlcv_collection.create_index([("cache_key", 1)])
    await ohlcv_collection.create_index([("timestamp", 1)], expireAfterSeconds=CONFIG["cache_ttl"])
    
    await signal_collection.create_index([("key", 1), ("timestamp", -1)])
    await signal_collection.create_index([("timestamp", 1)], expireAfterSeconds=30*24*60*60)
    
    await economic_collection.create_index([("date", -1)])
    await performance_collection.create_index([("timestamp", -1)])
    await ml_models_collection.create_index([("model_id", 1)], unique=True)
    await bot_state_collection.create_index([("name", 1)], unique=True)
    
    logger.info("MongoDB indexes and TTLs created")

# ================== STATE MANAGEMENT ==================
async def save_bot_state():
    try:
        state_collection = db[CONFIG["mongodb"]["collections"]["bot_state"]]
        await state_collection.update_one(
            {"name": "global_state"},
            {"$set": {
                "bot_status": bot_status,
                "invalid_symbols": list(invalid_symbols),
                "last_update": datetime.now(timezone.utc)
            }},
            upsert=True
        )
        logger.info("Bot state saved to MongoDB")
    except Exception as e:
        logger.error(f"Error saving bot state: {str(e)}")

async def load_bot_state():
    try:
        state_collection = db[CONFIG["mongodb"]["collections"]["bot_state"]]
        state = await state_collection.find_one({"name": "global_state"})
        if state:
            global bot_status, invalid_symbols
            bot_status = state.get("bot_status", bot_status)
            invalid_symbols = set(state.get("invalid_symbols", []))
            logger.info("Bot state loaded from MongoDB")
            return True
        return False
    except Exception as e:
        logger.error(f"Error loading bot state: {str(e)}")
        return False

# ================== DATA FETCHING ==================
async def test_exchange_connectivity(exchange: ccxt_async.Exchange, url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/exchangeInfo", ssl=False, timeout=10) as response:
                if response.status == 200:
                    logger.info(f"Connectivity test for {exchange.id}: SUCCESS")
                    return True
                logger.error(f"Connectivity test for {exchange.id} failed: HTTP {response.status}")
                return False
    except Exception as e:
        logger.error(f"Connectivity test for {exchange.id} failed: {str(e)}")
        return False

@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(15))
async def validate_symbols(exchange: ccxt_async.Exchange) -> List[str]:
    try:
        await exchange.load_markets(reload=True)
        markets = exchange.markets
        valid_pairs = []
        for symbol in CONFIG["pairs"]:
            if symbol in markets:  # Check if symbol exists in markets
                market = markets[symbol]
                if market.get('active') and market.get('swap') and market.get('linear'):
                    valid_pairs.append(symbol)
                    invalid_symbols.discard(symbol)
                    logger.info(f"Validated symbol: {symbol}")
                else:
                    logger.warning(f"Symbol {symbol} not a linear perpetual future")
                    invalid_symbols.add(symbol)
            else:
                logger.warning(f"Symbol {symbol} not found in markets")
                invalid_symbols.add(symbol)
        if not valid_pairs:
            logger.error("No valid trading symbols found")
        logger.info(f"Valid pairs: {valid_pairs}")
        return valid_pairs
    except Exception as e:
        logger.error(f"Failed to validate symbols: {str(e)}")
        raise
    finally:
        await exchange.close()

@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(15))
async def fetch_ohlcv_single(exchange: ccxt_async.Exchange, symbol: str, timeframe: str, limit: int = 300) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}-{timeframe}"
    ohlcv_collection = db[CONFIG["mongodb"]["collections"]["ohlcv"]]

    # Check MongoDB cache first
    try:
        cache_entry = await ohlcv_collection.find_one({"cache_key": cache_key})
        if cache_entry and time.time() - cache_entry["timestamp"] < CONFIG["cache_ttl"]:
            df = pd.DataFrame(cache_entry["data"])
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            logger.info(f"Using MongoDB cached OHLCV data for {symbol}-{timeframe}")
            return df
    except Exception as e:
        logger.error(f"Error checking MongoDB cache for {cache_key}: {str(e)}")

    # Fetch from exchange if not cached
    async with semaphore:
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('ts', inplace=True)

            # Save to MongoDB
            try:
                await ohlcv_collection.update_one(
                    {"cache_key": cache_key},
                    {"$set": {
                        "cache_key": cache_key,
                        "timestamp": time.time(),
                        "data": df.reset_index().to_dict('records')
                    }},
                    upsert=True
                )
                logger.info(f"Saved OHLCV data to MongoDB for {cache_key}")
            except Exception as e:
                logger.error(f"Error saving OHLCV to MongoDB for {cache_key}: {str(e)}")

            logger.info(f"Fetched {len(df)} rows for {symbol}-{timeframe}")
            return df
        except ccxt_async.RateLimitExceeded as e:
            logger.error(f"Rate limit exceeded for {symbol} {timeframe}: {str(e)}")
            return None
        except ccxt_async.NetworkError as e:
            logger.error(f"Network error for {symbol} {timeframe}: {str(e)}")
            return None
        except ccxt_async.ExchangeError as e:
            logger.error(f"Exchange error for {symbol} {timeframe}: {str(e)}")
            invalid_symbols.add(symbol)
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol} {timeframe}: {str(e)}")
            return None

async def fetch_live_price(exchange: ccxt_async.Exchange, symbol: str) -> Optional[float]:
    if symbol in invalid_symbols:
        logger.info(f"Skipping invalid symbol: {symbol}")
        return None
    try:
        ticker = await exchange.fetch_ticker(symbol)
        live_data_cache[symbol] = {"price": ticker['last'], "timestamp": time.time()}
        return ticker['last']
    except Exception as e:
        logger.error(f"Live price fetch error for {symbol}: {str(e)}")
        return None

# ================== ECONOMIC CALENDAR INTEGRATION ==================
async def fetch_economic_calendar() -> List[Dict]:
    economic_collection = db[CONFIG["mongodb"]["collections"]["economic_events"]]

    try:
        cursor = economic_collection.find().sort("date", -1)
        events = await cursor.to_list(length=1000)
        if events:
            logger.info(f"Using cached economic events from MongoDB ({len(events)} events)")
            return events
    except Exception as e:
        logger.error(f"Error fetching economic events from MongoDB: {str(e)}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(CONFIG["economic_calendar_url"], timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    events = [
                        {**event, "date": datetime.strptime(event["date"], '%Y-%m-%dT%H:%M:%S%z')}
                        for event in data
                        if event.get('impact', '').lower() == CONFIG["event_impact_threshold"]
                        and event.get('currency', '') in CONFIG.get("event_currencies", ["USD"])
                    ]
                    # Save to MongoDB
                    if events:
                        await economic_collection.delete_many({})
                        await economic_collection.insert_many(events)
                        logger.info(f"Saved {len(events)} economic events to MongoDB")
                    return events
                else:
                    logger.error(f"Failed to fetch economic calendar: HTTP {response.status}")
                    return []
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {str(e)}")
        return []

async def is_near_event(current_time: datetime) -> bool:
    events = await fetch_economic_calendar()
    for event in events:
        try:
            event_time = event['date']
            time_diff_hours = abs((current_time - event_time.replace(tzinfo=timezone.utc)).total_seconds() / 3600)
            if time_diff_hours <= CONFIG["event_window_hours"]:
                logger.info(f"Near event detected: {event['title']} at {event['date']} (Impact: {event['impact']}, Diff: {time_diff_hours:.2f} hours)")
                return True
        except ValueError as e:
            logger.error(f"Failed to parse event date '{event.get('date', 'N/A')}': {str(e)}")
    return False

# ================== INDICATOR AND SCORING MANAGEMENT ==================
def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for plugin_name in CONFIG["indicator_plugins"]:
        if plugin_name in INDICATOR_PLUGINS:
            try:
                df = INDICATOR_PLUGINS[plugin_name].compute(df)
            except Exception as e:
                logger.error(f"Error applying indicator {plugin_name}: {str(e)}")
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['roc'] = pta.roc(df['close'], length=10)
    df['cci'] = pta.cci(df['high'], df['low'], df['close'])
    # Add bar quality
    vol_ma = df['volume'].rolling(20).mean()
    df['is_quality_bar'] = (df['volume'] >= CONFIG["min_volume_multiple"] * vol_ma) & (df['body_size'] >= CONFIG["min_body_ratio"] * df['range_pct'])
    key_columns = ['ema20', 'ema50', 'ema200', 'rsi', 'macd', 'stoch_k', 'adx', 'volume', 'bb_position', 'buying_pressure']
    for col in key_columns:
        if col in df and df[col].isnull().all():
            logger.warning(f"All values in {col} are NaN")
    return df.fillna(method='ffill').fillna(0)

# ================== SIGNAL GENERATION HELPERS ==================
def detect_golden_cross(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2 or 'ema50' not in df or 'ema200' not in df:
        return None
    prev_ema50, prev_ema200 = df['ema50'].iloc[-2], df['ema200'].iloc[-2]
    curr_ema50, curr_ema200 = df['ema50'].iloc[-1], df['ema200'].iloc[-1]
    if prev_ema50 <= prev_ema200 and curr_ema50 > curr_ema200:
        return "golden"
    elif prev_ema50 >= prev_ema200 and curr_ema50 < curr_ema200:
        return "death"
    return None

def get_chart_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    tol, lookback = 0.01, 20
    window, lookback_long = 20, 40
    
    close = df['close']
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    valleys = (close.shift(1) > close) & (close.shift(-1) > close)
    peak_idxs = close[peaks].index[-3:]  
    valley_idxs = close[valleys].index[-3:]
    
    if len(peak_idxs) >= 2 and abs(close.loc[peak_idxs[-2]] - close.loc[peak_idxs[-1]]) / close.loc[peak_idxs[-2]] < tol:
        patterns.append("Double Top")
    if len(valley_idxs) >= 2 and abs(close.loc[valley_idxs[-2]] - close.loc[valley_idxs[-1]]) / close.loc[valley_idxs[-2]] < tol:
        patterns.append("Double Bottom")
    
    if len(peak_idxs) == 3:
        v1, v2, v3 = close.loc[peak_idxs]
        if abs(v1-v2)/v1 < tol and abs(v2-v3)/v2 < tol:
            patterns.append("Triple Top")
    if len(valley_idxs) == 3:
        v1, v2, v3 = close.loc[valley_idxs]
        if abs(v1-v2)/v1 < tol and abs(v2-v3)/v2 < tol:
            patterns.append("Triple Bottom")
    
    s = df['close'][-lookback:]
    if ((s <= s.max() + 0.015*s.max()) & (s >= s.min() - 0.015*s.min())).all():
        patterns.append("Rectangle")
    
    highs = df['high']
    lh = highs.rolling(3, center=True).apply(lambda x: x[1] > x[0] and x[1] > x[2], raw=True).fillna(0)
    p = highs[lh > 0].tail(3)
    if len(p) == 3 and p.iloc[1] > p.iloc[0] and p.iloc[1] > p.iloc[2]:
        patterns.append("Head & Shoulders")
    
    lows = df['low']
    lh = lows.rolling(3, center=True).apply(lambda x: x[1] < x[0] and x[1] < x[2], raw=True).fillna(0)
    t = lows[lh > 0].tail(3)
    if len(t) == 3 and t.iloc[1] < t.iloc[0] and t.iloc[1] < t.iloc[2]:
        patterns.append("Inv Head & Shoulders")
    
    cl = df['close'][-45:]
    mid = 45 // 2
    if cl.iloc[0] > cl.min() and cl.iloc[mid] == cl.min() and cl.iloc[-1] > cl.iloc[mid]:
        patterns.append("Cup & Handle")
    
    close_window = df['close'][-window:]
    if all(close_window.diff()[1:] > 0) and close_window.diff().max() / close_window.diff().min() < 4:
        patterns.append("Rising Wedge")
    if all(close_window.diff()[1:] < 0) and abs(close_window.diff().min() / close_window.diff().max()) < 4:
        patterns.append("Falling Wedge")
    
    highs = df['high'][-30:]
    lows = df['low'][-30:]
    resist, supp = highs.max(), lows.min()
    close = df['close'][-30:]
    troughs = (close.shift(1) > close) & (close.shift(-1) > close)
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    if any(abs(resist - l) / resist < tol for l in close[troughs]):
        patterns.append("Ascending Triangle")
    if any(abs(h - supp) / supp < tol for h in close[peaks]):
        patterns.append("Descending Triangle")
    
    high, low = df['high'][-lookback_long:], df['low'][-lookback_long:]
    if all(high.diff()[1:] > 0) and all(low.diff()[1:] < 0):
        patterns.append("Broadening")
    
    # Flag and Pennant
    if len(df) > 10:
        recent_highs = df['high'][-5:]
        recent_lows = df['low'][-5:]
        x = np.arange(len(recent_highs))
        slope_high = np.corrcoef(x, recent_highs)[0,1] * recent_highs.std() / x.std()
        slope_low = np.corrcoef(x, recent_lows)[0,1] * recent_lows.std() / x.std()
        if abs(slope_high - slope_low) < 0.01 and abs(slope_high) > 0:  # Parallel
            patterns.append("Flag")
        if abs(slope_high) > 0 and abs(slope_low) > 0 and slope_high * slope_low < 0:  # Converging
            patterns.append("Pennant")
    
    # Gaps
    if len(df) > 1:
        if df['open'].iloc[-1] > df['high'].iloc[-2]:
            patterns.append("Gap Up")
        if df['open'].iloc[-1] < df['low'].iloc[-2]:
            patterns.append("Gap Down")
    
    # Pipe Bottom
    if len(df) > 2:
        ranges = df['high'] - df['low']
        if ranges.iloc[-2] > ranges.iloc[:-2].max() and ranges.iloc[-1] > ranges.iloc[:-2].max():
            if df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] > (df['high'].iloc[-1] + df['low'].iloc[-1]) / 2:
                patterns.append("Pipe Bottom")
    
    # Narrow Range
    if len(df) > 4:
        ranges = df['high'] - df['low']
        if ranges.iloc[-1] < min(ranges.iloc[-4:-1]):
            patterns.append("Narrow Range")
    
    return patterns

def detect_fvg(df: pd.DataFrame, window: int = 3) -> List[Tuple[str, int]]:
    gaps = []
    for i in range(window, len(df) - window):
        hi_prev = df['high'].iloc[i-1]
        lo_next = df['low'].iloc[i+1]
        if df['low'].iloc[i] > hi_prev:
            gaps.append(('bullish', i))
        elif df['high'].iloc[i] < lo_next:
            gaps.append(('bearish', i))
    return gaps

def find_sr_zones(df: pd.DataFrame, bins: int = 30) -> List[float]:
    prices = df['close'][-100:]
    counts, edges = pd.cut(prices, bins, retbins=True, labels=False)
    levels = []
    for b in range(bins):
        hits = prices[counts == b]
        if len(hits) > 3:
            levels.append(hits.mean())
    return sorted(set(round(l, 3) for l in levels))

def detect_bos_choch(df: pd.DataFrame, lookback: int = 20) -> Optional[str]:
    closes = df['close'][-lookback:]
    if closes.iloc[-1] > closes.iloc[:-1].max():
        return "up_bos"
    if closes.iloc[-1] < closes.iloc[:-1].min():
        return "down_bos"
    return None

def detect_market_regime(df: pd.DataFrame) -> str:
    adx_avg = df['adx'].tail(10).mean() if 'adx' in df else 0
    volatility_avg = df['volatility'].tail(10).mean() if 'volatility' in df else 0
    bb_width_avg = df['bb_width'].tail(10).mean() if 'bb_width' in df else 0
    volatility_ma = df['volatility'].rolling(50).mean().iloc[-1] if 'volatility' in df else 1
    bb_width_ma = df['bb_width'].rolling(50).mean().iloc[-1] if 'bb_width' in df else 1
    
    if adx_avg > 30 and volatility_avg < volatility_ma:
        return "trending"
    elif bb_width_avg < bb_width_ma * 0.8:
        return "ranging"
    elif volatility_avg > volatility_ma * 1.5:
        return "volatile"
    else:
        return "neutral"

async def multi_timeframe_confluence(exchange: ccxt_async.Exchange, symbol: str) -> Tuple[int, Dict]:
    confluence_score = 0
    tf_data = {}
    tasks = [fetch_and_analyze_tf(exchange, symbol, tf) for tf in CONFIG["timeframes"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for tf, result in zip(CONFIG["timeframes"], results):
        if isinstance(result, Exception):
            logger.error(f"Multi-TF error for {symbol}-{tf}: {str(result)}")
            continue
        if result:
            tf_data[tf], score = result
            weight = CONFIG["tf_weights"].get(tf, 1.0)
            confluence_score += score * weight
    return int(confluence_score), tf_data

async def fetch_and_analyze_tf(exchange: ccxt_async.Exchange, symbol: str, timeframe: str) -> Tuple[Optional[pd.DataFrame], int]:
    try:
        df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=300)
        if df is None or len(df) < 200:
            logger.info(f"Insufficient data for {symbol}-{timeframe}: {len(df) if df is not None else 0} rows")
            return None, 0
        df = apply_indicators(df)
        last = df.iloc[-1]
        if pd.isna(last.get('ema20', np.nan)) or pd.isna(last.get('ema50', np.nan)) or pd.isna(last.get('ema200', np.nan)):
            logger.info(f"Missing EMA values for {symbol}-{timeframe}")
            return df, 0
        score = 1 if last['ema20'] > last['ema50'] > last['ema200'] else -1 if last['ema20'] < last['ema50'] < last['ema200'] else 0
        return df, score
    except Exception as e:
        logger.error(f"Error analyzing {symbol}-{timeframe}: {str(e)}")
        return None, 0

# ================== MACHINE LEARNING MODEL ==================
class OptimizedCryptoMLClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model_name = None
        self.performance_history = {}
        self.last_retrained = 0

    def create_features(self, df: pd.DataFrame, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], golden_cross: str, regime: str) -> Dict:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        recent_false_up = 1 if df['false_up_breakout'].tail(5).any() else 0
        recent_false_down = 1 if df['false_down_breakout'].tail(5).any() else 0
        features = {
            'ema_alignment': 1 if last.get('ema20', 0) > last.get('ema50', 0) > last.get('ema200', 0) else 
                           -1 if last.get('ema20', 0) < last.get('ema50', 0) < last.get('ema200', 0) else 0,
            'golden_cross': 1 if golden_cross == "golden" else 0,
            'death_cross': 1 if golden_cross == "death" else 0,
            'price_vs_ema200': (last['close'] - last.get('ema200', last['close'])) / last['close'] if last.get('ema200') else 0,
            'adx': last.get('adx', 0),
            'rsi': last.get('rsi', 50),
            'macd_hist': last.get('macd_hist', 0),
            'atr_pct': last.get('atr_pct', 0),
            'vol_ratio': last.get('vol_ratio', 1),
            'vol_spike': 1 if last.get('vol_spike', False) else 0,
            'bb_position': last.get('bb_position', 0.5),
            'body_size': last.get('body_size', 0),
            'buying_pressure': last.get('buying_pressure', 0.5),
            'double_top': 1 if "Double Top" in chart_patterns else 0,
            'double_bottom': 1 if "Double Bottom" in chart_patterns else 0,
            'head_shoulders': 1 if "Head & Shoulders" in chart_patterns else 0,
            'inv_head_shoulders': 1 if "Inv Head & Shoulders" in chart_patterns else 0,
            'hammer': 1 if last.get('hammer', 0) == 100 else 0,
            'engulfing_bull': 1 if last.get('engulfing', 0) == 100 else 0,
            'engulfing_bear': 1 if last.get('engulfing', 0) == -100 else 0,
            'bos_up': 1 if choch == "up_bos" else 0,
            'bos_down': 1 if choch == "down_bos" else 0,
            'near_support': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl < last['close']) else 0,
            'near_resistance': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl > last['close']) else 0,
            'vwap_signal': 1 if last.get('vwap') and last['close'] > last['vwap'] else -1 if last.get('vwap') and last['close'] < last['vwap'] else 0,
            # Composite features
            'rsi_macd_bull': 1 if last['rsi'] < 30 and last['macd_hist'] > 0 else 0,
            'rsi_macd_bear': 1 if last['rsi'] > 70 and last['macd_hist'] < 0 else 0,
            'stoch_rsi_bull': 1 if last['stoch_k'] < 20 and last['rsi'] < 30 else 0,
            'stoch_rsi_bear': 1 if last['stoch_k'] > 80 and last['rsi'] > 70 else 0,
            # Regime features (one-hot)
            'regime_trending': 1 if regime == "trending" else 0,
            'regime_ranging': 1 if regime == "ranging" else 0,
            'regime_volatile': 1 if regime == "volatile" else 0,
            'regime_neutral': 1 if regime == "neutral" else 0,
            # New features from PDF patterns
            'pipe_bottom': 1 if "Pipe Bottom" in chart_patterns else 0,
            'gap_up': 1 if "Gap Up" in chart_patterns else 0,
            'gap_down': 1 if "Gap Down" in chart_patterns else 0,
            'narrow_range': 1 if "Narrow Range" in chart_patterns else 0,
            'flag': 1 if "Flag" in chart_patterns else 0,
            'pennant': 1 if "Pennant" in chart_patterns else 0,
            'triple_top': 1 if "Triple Top" in chart_patterns else 0,
            'triple_bottom': 1 if "Triple Bottom" in chart_patterns else 0,
            # Trendline features
            'up_breakout': 1 if last.get('up_breakout', False) else 0,
            'down_breakout': 1 if last.get('down_breakout', False) else 0,
            'recent_false_up': recent_false_up,
            'recent_false_down': recent_false_down,
        }
        return features

    def prepare_training_data(self, historical_signals: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for signal in historical_signals:
            if 'features' in signal and 'outcome' in signal:
                features = signal['features']
                if isinstance(features, dict):
                    if not self.feature_names:
                        self.feature_names = sorted(features.keys())
                    feature_vector = [features.get(name, 0) for name in self.feature_names]
                    X.append(feature_vector)
                    y.append(signal['outcome'])
        return np.array(X), np.array(y)

    async def train_model(self, exchange: ccxt_async.Exchange, symbol: str, timeframe: str) -> bool:
        df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=1500)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for ML training: {symbol}-{timeframe} ({len(df) if df is not None else 0} rows)")
            return False

        df = apply_indicators(df)
        historical_signals = []

        original_threshold = CONFIG['conf_threshold']
        CONFIG['conf_threshold'] = 50  # Lower threshold for generating training data
        try:
            for i in range(100, len(df) - 50):
                window = df.iloc[i-100:i]
                regime = detect_market_regime(window)
                chart_patterns = get_chart_patterns(window)
                fvg = detect_fvg(window)
                choch = detect_bos_choch(window)
                sr_levels = find_sr_zones(window)
                golden_cross = detect_golden_cross(window)
                signal = get_enhanced_signal(window, chart_patterns, fvg, choch, sr_levels,
                                             regime, 0, golden_cross, symbol)
                if signal:
                    future_price = df['close'].iloc[i+50]
                    outcome = 1 if (signal['direction'] == "LONG" and future_price > signal['entry']) or \
                                  (signal['direction'] == "SHORT" and future_price < signal['entry']) else 0
                    historical_signals.append({
                        'features': self.create_features(window, chart_patterns, fvg, choch, sr_levels, golden_cross, regime),
                        'outcome': outcome,
                        'timestamp': time.time()
                    })
        finally:
            CONFIG['conf_threshold'] = original_threshold

        if len(historical_signals) < 100:
            logger.warning(f"Not enough training signals: {len(historical_signals)}")
            return False

        X, y = self.prepare_training_data(historical_signals)
        if len(X) < 100:
            logger.warning(f"Not enough training data: {len(X)} samples")
            return False

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )

            logger.info("Training XGBoost model...")
            model.fit(X_train_scaled, y_train)

            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)

            self.performance_history = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'auc_score': auc_score
            }

            logger.info(f"XGBoost Results: Train: {train_score:.3f}, Test: {test_score:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, AUC: {auc_score:.3f}")

            if test_score > 0.6 and auc_score > 0.65:
                self.models['best'] = model
                self.scalers['best'] = scaler
                self.best_model_name = "XGBoost"
                self.last_retrained = time.time()
                logger.info(f"Model trained successfully (Score: {test_score:.3f})")
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    logger.info("Top 5 Most Important Features:")
                    for i in range(min(5, len(indices))):
                        logger.info(f"  {i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                return True
            else:
                logger.warning("Model did not meet performance threshold")
                return False
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False

    def predict(self, features: Dict) -> Tuple[float, int]:
        if 'best' not in self.models:
            return 0.5, 0
        try:
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scalers['best'].transform(feature_vector)
            probability = self.models['best'].predict_proba(feature_vector_scaled)[0][1]
            prediction = self.models['best'].predict(feature_vector_scaled)[0]
            return probability, prediction
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return 0.5, 0

    async def save_model(self, model_id: str = "primary") -> bool:
        try:
            model_data = {
                'model': self.models.get('best'),
                'scaler': self.scalers.get('best'),
                'feature_names': self.feature_names,
                'performance': self.performance_history,
                'last_retrained': self.last_retrained,
                'timestamp': datetime.now(timezone.utc)
            }
            
            serialized = pickle.dumps(model_data)
            encrypted_data = fernet.encrypt(serialized)
            
            ml_models = db[CONFIG["mongodb"]["collections"]["ml_models"]]
            await ml_models.update_one(
                {"model_id": model_id},
                {"$set": {
                    "model_id": model_id,
                    "data": encrypted_data,
                    "timestamp": datetime.now(timezone.utc),
                    "performance": self.performance_history
                }},
                upsert=True
            )
            logger.info(f"Model saved to MongoDB with ID: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to MongoDB: {str(e)}")
            return False

    async def load_model(self, model_id: str = "primary") -> bool:
        try:
            ml_models = db[CONFIG["mongodb"]["collections"]["ml_models"]]
            model_doc = await ml_models.find_one({"model_id": model_id})
            if model_doc is None:
                logger.warning(f"No model found in MongoDB with ID: {model_id}")
                return False
            encrypted_data = model_doc['data']
            serialized = fernet.decrypt(encrypted_data)
            model_data = pickle.loads(serialized)
            
            self.models['best'] = model_data.get('model')
            self.scalers['best'] = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.performance_history = model_data.get('performance', {})
            self.last_retrained = model_data.get('last_retrained', 0)
            logger.info(f"Model loaded from MongoDB: {model_id}")
            if self.performance_history:
                logger.info(f"ML Performance: Test {self.performance_history.get('test_score', 0):.3f}, AUC {self.performance_history.get('auc_score', 0):.3f}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from MongoDB: {str(e)}")
            return False

ml_classifier = OptimizedCryptoMLClassifier()

async def ensure_ml_model(exchange: ccxt_async.Exchange, valid_pairs: List[str], timeframes: List[str]) -> bool:
    if await ml_classifier.load_model():
        logger.info("Existing ML model loaded successfully")
        return True

    logger.info("No pre-trained model found. Attempting to train new model...")
    for symbol in valid_pairs:
        for timeframe in timeframes:
            logger.info(f"Training ML model with {symbol} on {timeframe}")
            try:
                if await ml_classifier.train_model(exchange, symbol, timeframe):
                    await ml_classifier.save_model()
                    logger.info(f"Model trained and saved successfully for {symbol}-{timeframe}")
                    return True
                else:
                    logger.warning(f"Model training failed for {symbol}-{timeframe}")
            except Exception as e:
                logger.error(f"Error training model for {symbol}-{timeframe}: {str(e)}")
    logger.error("Failed to train model with any symbol-timeframe combination. Disabling ML.")
    CONFIG["ml_enabled"] = False
    return False

# ================== SIGNAL MANAGEMENT ==================
def enhanced_signal_score(df: pd.DataFrame, direction: str, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], regime: str, confluence_score: int, golden_cross: str) -> Tuple[int, List[str]]:
    total_score, all_patterns = 0, []
    adx_threshold = CONFIG['adx_threshold']
    if regime == "trending":
        adx_threshold -= 5  # Lower threshold in trending markets
    elif regime == "ranging":
        adx_threshold += 5  # Higher in ranging
    elif regime == "volatile":
        adx_threshold += 3  # Adjust for volatility
    
    context = {
        "chart_patterns": chart_patterns,
        "fvg": fvg,
        "choch": choch,
        "sr_levels": sr_levels,
        "regime": regime,
        "confluence_score": confluence_score,
        "golden_cross": golden_cross,
        "adx_threshold": adx_threshold
    }
    
    for plugin_name in CONFIG["scoring_plugins"]:
        if plugin_name in SCORING_PLUGINS:
            try:
                score, patterns = SCORING_PLUGINS[plugin_name].score(df, direction, **context)
                total_score += score
                all_patterns.extend(patterns)
                logger.info(f"{plugin_name} score: {score}, patterns: {patterns}")
            except Exception as e:
                logger.error(f"Error applying scoring plugin {plugin_name}: {str(e)}")
    
    if regime == "trending":
        total_score *= 1.2
    elif regime == "ranging":
        total_score *= 0.8
    elif regime == "volatile":
        total_score *= 0.9

    if CONFIG['multi_tf_confluence'] and abs(confluence_score) > 1:
        total_score *= 1.15
    else:
        total_score *= 0.9

    return int(total_score), list(set(all_patterns))

def dynamic_risk_management(df: pd.DataFrame, entry: float, direction: str, regime: str) -> Tuple[float, List[float]]:
    atr = df.get('atr', pd.Series([1])).iloc[-1]
    if regime == "volatile":
        sl_mult = CONFIG['atr_sl_mult'] * 1.5
        tp_mult = CONFIG['atr_tp_mult'] * 1.3
    elif regime == "ranging":
        sl_mult = CONFIG['atr_sl_mult'] * 0.8
        tp_mult = CONFIG['atr_tp_mult'] * 0.9
    else:
        sl_mult = CONFIG['atr_sl_mult']
        tp_mult = CONFIG['atr_tp_mult']
    
    sl = entry - sl_mult * atr if direction == "LONG" else entry + sl_mult * atr
    tps = [entry + tp_mult * atr * i for i in (1, 1.5, 2)] if direction == "LONG" else [entry - tp_mult * atr * i for i in (1, 1.5, 2)]
    return sl, tps

def get_enhanced_signal(df: pd.DataFrame, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], regime: str, confluence_score: int, golden_cross: str, symbol: str) -> Optional[Dict]:
    if not df['is_quality_bar'].iloc[-1]:
        logger.info(f"{symbol}: Skipping signal generation due to low-quality bar")
        return None
    
    logger.info(f"Generating signal for {symbol}: Patterns={chart_patterns}, FVG={fvg}, CHOCH={choch}, SR={sr_levels}, Regime={regime}, Confluence={confluence_score}, GoldenCross={golden_cross}")
    for direction in ("LONG", "SHORT"):
        total_score, patterns = enhanced_signal_score(df, direction, chart_patterns, fvg, choch, sr_levels, regime, confluence_score, golden_cross)
        logger.info(f"{symbol}:{direction} Score: {total_score}, Patterns: {patterns}")
        if total_score >= CONFIG['conf_threshold']:
            last = df.iloc[-1]
            entry = last['close']
            if CONFIG['ml_enabled']:
                features = ml_classifier.create_features(df, chart_patterns, fvg, choch, sr_levels, golden_cross, regime)
                ml_confidence, ml_prediction = ml_classifier.predict(features)
                logger.info(f"{direction} signal - ML Confidence: {ml_confidence:.3f}")
                if ml_confidence < CONFIG['ml_threshold']:
                    logger.info(f"{direction} signal filtered out (confidence: {ml_confidence:.3f} < {CONFIG['ml_threshold']})")
                    continue
                final_score = total_score * ml_confidence
            else:
                ml_confidence = 0.5
                final_score = total_score
            sl, tps = dynamic_risk_management(df, entry, direction, regime)
            rr = round(abs(tps[0]-entry)/abs(entry-sl), 2) if abs(entry-sl) > 0 else 0
            # Pattern-based target
            pattern_target = None
            if chart_patterns:
                height = df['close'].max() - df['close'].min()
                if direction == "LONG":
                    pattern_target = entry + height
                else:
                    pattern_target = entry - height
            return {
                "direction": direction, "entry": entry, "sl": sl, "tps": tps,
                "score": int(final_score), "ml_confidence": ml_confidence,
                "traditional_score": total_score, "patterns": patterns,
                "timestamp": time.time(), "rr": rr,
                "regime": regime, "confluence": confluence_score,
                "fvg": str(fvg[-1]) if fvg else "None", "choch": choch,
                "sr": sr_levels, "atr": last.get('atr', 0), "bar_ts": str(df.index[-1]),
                "golden_cross": golden_cross, "pattern_target": pattern_target
            }
        else:
            logger.info(f"{symbol}:{direction} Score {total_score} below threshold {CONFIG['conf_threshold']}")
    return None

# ================== TELEGRAM & SIGNAL MANAGEMENT ==================
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
@rate_limit("telegram")
async def send_telegram(text: str) -> bool:
    sanitized_text = sanitize_string(text)
    if not sanitized_text:
        logger.error("Invalid or empty Telegram message after sanitization")
        return False
    try:
        await bot.send_message(
            chat_id=CONFIG["chat_id"],
            text=sanitized_text,
            read_timeout=30,
            write_timeout=30,
            connect_timeout=30
        )
        return True
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")
        return False

async def log_signal(signal: Dict):
    try:
        signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
        signal_copy = signal.copy()
        signal_copy["timestamp"] = datetime.fromtimestamp(signal_copy["timestamp"], tz=timezone.utc)
        signal_copy["bar_ts"] = pd.to_datetime(signal_copy["bar_ts"])
        await signal_collection.insert_one(signal_copy)
        logger.info(f"Signal saved to MongoDB for {signal['key']}")
    except Exception as e:
        logger.error(f"Error saving signal to MongoDB: {str(e)}")

async def get_signal_history(symbol: str = None, timeframe: str = None, limit: int = 1000) -> List[Dict]:
    try:
        signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
        query = {}
        if symbol:
            query["key"] = {"$regex": f"^{symbol}-"}
        if timeframe:
            query["key"] = {"$regex": f"-{timeframe}-"}
        cursor = signal_collection.find(query).sort("timestamp", -1).limit(limit)
        signals = await cursor.to_list(length=limit)
        for signal in signals:
            signal["_id"] = str(signal["_id"])
            signal["timestamp"] = signal["timestamp"].timestamp()
            signal["bar_ts"] = signal["bar_ts"].timestamp()
        return signals
    except Exception as e:
        logger.error(f"Error fetching signal history: {str(e)}")
        return []

async def get_open_signals() -> Dict[str, Dict]:
    try:
        signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
        cursor = signal_collection.find({"status": "open"}).sort("timestamp", -1)
        signals = await cursor.to_list(length=1000)
        return {s["key"]: {**s, "_id": str(s["_id"]), "timestamp": s["timestamp"].timestamp(), "bar_ts": s["bar_ts"].timestamp()} for s in signals}
    except Exception as e:
        logger.error(f"Error fetching open signals: {str(e)}")
        return {}

async def update_open_signal(signal: Dict, status: str = "open"):
    try:
        signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
        await signal_collection.update_one(
            {"key": signal["key"], "timestamp": datetime.fromtimestamp(signal["timestamp"], tz=timezone.utc)},
            {"$set": {"status": status}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error updating signal status: {str(e)}")

async def update_signal_outcome(signal_key: str, outcome: int):
    try:
        signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
        await signal_collection.update_one(
            {"key": signal_key},
            {"$set": {"outcome": outcome}}
        )
    except Exception as e:
        logger.error(f"Error updating signal outcome: {str(e)}")

async def clean_stale_signals(open_signals: Dict) -> Dict:
    now = time.time()
    signals_to_remove = []
    for sig_key, signal in open_signals.items():
        if now - signal["timestamp"] > CONFIG["signal_lifetime_sec"]:
            signals_to_remove.append(sig_key)
            await update_open_signal(signal, status="expired")
            await send_telegram(f"⌛ {sig_key.split('-')[0]} signal expired ({signal['direction']})")
            await update_signal_outcome(sig_key, 0)
    for sig_key in signals_to_remove:
        open_signals.pop(sig_key, None)
    return open_signals

async def check_storage_usage():
    stats = await db.command("dbStats")
    storage_mb = stats["storageSize"] / (1024 * 1024)
    logger.info(f"Current storage usage: {storage_mb:.2f} MB")
    return storage_mb

async def analyze_signal_performance(symbol: str = None):
    try:
        signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
        query = {"outcome": {"$exists": True}}
        if symbol:
            query["key"] = {"$regex": f"^{symbol}-"}
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": {"$arrayElemAt": [{"$split": ["$key", "-"]}, 0]},
                "total": {"$sum": 1},
                "wins": {"$sum": {"$cond": [{"$eq": ["$outcome", 1]}, 1, 0]}}
            }},
            {"$project": {
                "symbol": "$_id",
                "win_rate": {"$divide": ["$wins", "$total"]},
                "total": 1
            }}
        ]
        results = await signal_collection.aggregate(pipeline).to_list(length=100)
        for res in results:
            logger.info(f"Symbol: {res['symbol']}, Win Rate: {res['win_rate']:.2%}, Total Signals: {res['total']}")
    except Exception as e:
        logger.error(f"Error analyzing signal performance: {str(e)}")

# ================== LIVE DATA MONITORING ==================
async def monitor_live_data(exchange: ccxt_async.Exchange):
    while True:
        try:
            for symbol in CONFIG["pairs"]:
                if symbol in invalid_symbols:
                    continue
                live_price = await fetch_live_price(exchange, symbol)
                if live_price:
                    live_data_cache[symbol] = {"price": live_price, "timestamp": time.time()}

            current_time = time.time()
            open_signals = await get_open_signals()
            open_signals = await clean_stale_signals(open_signals)
            
            signals_to_remove = []
            for sig_key, signal in open_signals.items():
                symbol = sig_key.split('-')[0]
                if symbol in invalid_symbols:
                    signals_to_remove.append(sig_key)
                    continue
                live_price = live_data_cache.get(symbol, {}).get("price")
                if not live_price:
                    continue
                
                direction = signal['direction']
                entry = signal['entry']
                sl = signal['sl']
                tps = signal['tps']
                
                if direction == "LONG":
                    if live_price <= sl:
                        await send_telegram(f"📉 {symbol} LONG Stop Loss hit at {live_price:.4f}")
                        await update_signal_outcome(sig_key, 0)
                        signals_to_remove.append(sig_key)
                    else:
                        for i, tp in enumerate(tps):
                            if live_price >= tp:
                                await send_telegram(f"📈 {symbol} LONG Take Profit #{i+1} hit at {live_price:.4f}")
                                if i == len(tps) - 1:
                                    await update_signal_outcome(sig_key, 1)
                                    signals_to_remove.append(sig_key)
                else:  # SHORT
                    if live_price >= sl:
                        await send_telegram(f"📈 {symbol} SHORT Stop Loss hit at {live_price:.4f}")
                        await update_signal_outcome(sig_key, 0)
                        signals_to_remove.append(sig_key)
                    else:
                        for i, tp in enumerate(tps):
                            if live_price <= tp:
                                await send_telegram(f"📉 {symbol} SHORT Take Profit #{i+1} hit at {live_price:.4f}")
                                if i == len(tps) - 1:
                                    await update_signal_outcome(sig_key, 1)
                                    signals_to_remove.append(sig_key)
            
            for sig_key in set(signals_to_remove):
                if sig_key in open_signals:
                    await update_open_signal(open_signals[sig_key], status="closed")
            
            await asyncio.sleep(CONFIG['live_data_interval'])
        except Exception as e:
            logger.error(f"Live monitor error: {str(e)}")
            await asyncio.sleep(10)

# ================== PROCESSING FUNCTION ==================
async def process_symbol_tf(exchange, symbol, tf, open_signals):
    try:
        df = await fetch_ohlcv_single(exchange, symbol, tf, limit=300)
        if df is None or len(df) < 200:
            logger.info(f"Data fetch failed or insufficient for {symbol}-{tf}: {len(df) if df is not None else 'None'} rows")
            return
        df = apply_indicators(df)
        regime = detect_market_regime(df)
        confluence_score, _ = await multi_timeframe_confluence(exchange, symbol)
        chart_patterns = get_chart_patterns(df)
        fvg = detect_fvg(df)
        sr_levels = find_sr_zones(df)
        choch = detect_bos_choch(df)
        golden_cross = detect_golden_cross(df)
        sig = get_enhanced_signal(df, chart_patterns, fvg, choch, sr_levels, regime, confluence_score, golden_cross, symbol)
        if sig:
            sig_key = f"{symbol}-{tf}-{sig['direction']}"
            sig['key'] = sig_key
            bar_ts = sig['bar_ts']
            if sig_key in open_signals and open_signals[sig_key].get('bar_ts') == bar_ts:
                return
            cross_info = " | ✨ Golden Cross" if golden_cross == "golden" else " | ☠️ Death Cross" if golden_cross == "death" else ""
            logger.info(f"SIGNAL: {symbol} {tf} {sig['direction']} | Score: {sig['score']} | ML: {sig['ml_confidence']:.3f}{cross_info}")
            await update_open_signal(sig, status="open")
            await log_signal(sig)
            tps_str = ', '.join(f"{tp:.4f}" for tp in sig['tps'])
            patterns_str = ', '.join(sig['patterns'][:3]) if sig['patterns'] else "None"
            msg = (f"🤖 ENHANCED SIGNAL\n"
                   f"📊 {symbol} {tf}: {sig['direction']}{cross_info}\n"
                   f"🏆 ML Confidence: {sig['ml_confidence']:.2f} | ⚡ Score: {sig['score']}\n"
                   f"💰 Entry: {sig['entry']:.4f} | 🛑 SL: {sig['sl']:.4f}\n"
                   f"🎯 TPs: {tps_str} | ⚡ RR: {sig['rr']}\n"
                   f"📋 Patterns: {patterns_str}")
            if sig.get('pattern_target'):
                msg += f"\n🎯 Pattern Target: {sig['pattern_target']:.4f}"
            await send_telegram(msg)
            bot_status["signals_count"] += 1
    except Exception as e:
        logger.error(f"Error processing {symbol}-{tf}: {str(e)}")

# ================== MAIN MONITORING LOOP ==================
async def monitor():
    validate_config()
    exchange = ccxt_async.binance(CONFIG["exchanges"]["binance"])
    
    if not await test_exchange_connectivity(exchange, CONFIG["exchanges"]["binance"]["urls"]["api"]["public"]):
        logger.error("Cannot connect to Binance API. Exiting...")
        sys.exit(1)

    await init_mongo_indexes()
    await load_bot_state()

    valid_pairs = await validate_symbols(exchange)
    CONFIG["pairs"] = valid_pairs
    logger.info(f"Valid pairs: {valid_pairs}")
    if not valid_pairs:
        logger.error("No valid symbols to monitor. Exiting...")
        sys.exit(1)

    if CONFIG['ml_enabled']:
        await ensure_ml_model(exchange, valid_pairs, CONFIG["timeframes"])

    economic_events = await fetch_economic_calendar()

    asyncio.create_task(monitor_live_data(exchange))
    cycle_count = 0
    retrain_counter = 0

    while True:
        cycle_count += 1
        logger.info(f"Starting analysis cycle #{cycle_count}")
        open_signals = await get_open_signals()
        open_signals = await clean_stale_signals(open_signals)

        if CONFIG['ml_enabled'] and len(await get_signal_history()) > 100:
            retrain_counter += 1
            if retrain_counter % 24 == 0:
                logger.info("Periodic model retraining...")
                outcomes = [s for s in await get_signal_history() if 'outcome' in s]
                if len(outcomes) >= 100:
                    if await ml_classifier.train_model(exchange, valid_pairs[0], CONFIG["timeframes"][0]):
                        await ml_classifier.save_model()

        current_dt = datetime.now(timezone.utc)
        if await is_near_event(current_dt):
            logger.info("Skipping signal generation due to nearby high-impact economic event")
            await asyncio.sleep(CONFIG['scan_interval'])
            continue

        # Parallel processing of all symbol-timeframe pairs
        process_tasks = []
        for tf in CONFIG["timeframes"]:
            for symbol in CONFIG["pairs"]:
                if symbol not in invalid_symbols:
                    process_tasks.append(process_symbol_tf(exchange, symbol, tf, open_signals))

        await asyncio.gather(*process_tasks, return_exceptions=True)
        
        if cycle_count % 6 == 0:
            signal_history = await get_signal_history(limit=1000)
            status_msg = (f"📊 BOT STATUS\n"
                          f"🔄 Cycle: #{cycle_count} | ⏱ Uptime: {timedelta(seconds=int(time.time()-bot_status['last_update']))}\n"
                          f"📈 Active Signals: {len(open_signals)}\n"
                          f"📋 Total Signals: {len(signal_history)} | 🤖 ML Signals: {len([s for s in signal_history if s.get('ml_confidence', 0) > CONFIG['ml_threshold']])}\n"
                          f"🚫 Invalid Symbols: {len(invalid_symbols)}")
            if ml_classifier.performance_history:
                perf = ml_classifier.performance_history
                status_msg += f"\n🤖 ML Perf: Test {perf.get('test_score', 0):.3f} | AUC {perf.get('auc_score', 0):.3f}"
            await send_telegram(status_msg)
            await check_storage_usage()
        
        if cycle_count % 12 == 0:
            await analyze_signal_performance()
            
        await save_bot_state()
        logger.info(f"CYCLE COMPLETE: Open: {len(open_signals)} | Total: {len(signal_history)} | Invalid Symbols: {len(invalid_symbols)}")
        await asyncio.sleep(CONFIG['scan_interval'])

# ================== MAIN ENTRY POINT ==================
if __name__ == "__main__":
    logger.info("🚀 Enhanced Crypto Quant Trading Bot")
    logger.info("✨ MongoDB Storage | Plugin System | Parallel Multi-TF")
    logger.info(f"Symbols: {len(CONFIG['pairs'])}")
    logger.info(f"Timeframes: {CONFIG['timeframes']}")
    logger.info(f"ML Enabled: {CONFIG['ml_enabled']}")
    
    threading.Thread(target=run_flask, daemon=True).start()
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        asyncio.run(save_bot_state())
    finally:
        exchange = ccxt_async.binance(CONFIG["exchanges"]["binance"])
        exchange.close()
        mongo_client.close()
        logger.info("Resources released. Goodbye!")