import os, sys, time, asyncio, threading, json
import ccxt.async_support as ccxt_async
import pandas as pd
import pandas_ta as pta
import talib
import numpy as np
from telegram import Bot
from flask import Flask, request
from datetime import datetime, timedelta
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
    "conf_threshold": 50,  # Lowered for testing
    "atr_sl_mult": 1.8,
    "atr_tp_mult": 2.5,
    "adx_threshold": 25,
    "vol_mult": 2.0,
    "signal_lifetime_sec": 6*3600,
    "log_file": "signals_log.jsonl",
    "ml_enabled": True,
    "ml_threshold": 0.65,
    "market_regime_enabled": True,
    "multi_tf_confluence": True,
    "volatility_filter": True,
    "order_flow_analysis": True,
    "scan_interval": 120,
    "live_data_interval": 15,
    "cache_ttl": 300,
    "encryption_key": os.environ.get("ENCRYPTION_KEY", Fernet.generate_key().decode()),
    "flask_rate_limit": {"requests": 100, "window": 60},
    "telegram_rate_limit": {"requests": 20, "window": 60},
    "indicator_plugins": ["ema", "rsi", "macd", "bbands", "stoch", "willr", "adx", "atr", "volatility", "obv", "cmf", "mfi", "candle_patterns", "order_flow", "vwap"],
    "scoring_plugins": ["ema_alignment", "golden_cross", "rsi", "macd", "stoch", "adx", "volume", "bbands", "order_flow", "chart_patterns", "smc", "vwap"]
}

# Global state
bot_status = {"status": "starting", "last_update": time.time(), "signals_count": 0}
open_signals = {}
signal_history = []
semaphore = asyncio.Semaphore(10)
live_data_cache = {}
ohlcv_cache = {}
invalid_symbols = set()
fernet = Fernet(CONFIG["encryption_key"].encode())

# Rate limiting storage
flask_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})
telegram_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})

# Initialize components
bot = Bot(token=CONFIG["telegram_token"])
app = Flask(__name__)

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

# Indicator Plugins
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
        if last['adx'] > CONFIG['adx_threshold']:
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
        elif direction == "SHORT":
            if "Double Top" in chart_patterns:
                score += 16
                patterns.append("Double Top")
            if "Head & Shoulders" in chart_patterns:
                score += 20
                patterns.append("H&S")
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
    "vwap": VWAPPlugin("vwap")
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
    "vwap": VWAPPluginScoring("vwap")
}

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
        perpetual_markets = [
            symbol for symbol, market in markets.items()
            if market.get('active', False) and
               market.get('type') == 'swap' and
               market.get('linear', False) and
               symbol.endswith('USDT')
        ]
        valid_pairs = []
        for symbol in CONFIG["pairs"]:
            if symbol in perpetual_markets:
                valid_pairs.append(symbol)
                invalid_symbols.discard(symbol)
                logger.info(f"Validated symbol: {symbol}")
            else:
                logger.warning(f"Symbol {symbol} not available in perpetual futures")
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
async def fetch_ohlcv_batch(exchange: ccxt_async.Exchange, symbols: List[str], timeframe: str, limit: int = 300) -> Dict[str, Optional[pd.DataFrame]]:
    results = {}
    async with semaphore:
        tasks = [fetch_ohlcv_single(exchange, symbol, timeframe, limit) for symbol in symbols if symbol not in invalid_symbols]
        logger.info(f"Fetching data for {len(tasks)} symbols (skipped {len(invalid_symbols)} invalid symbols)")
        ohlcv_data = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, data in zip([s for s in symbols if s not in invalid_symbols], ohlcv_data):
            results[symbol] = data if not isinstance(data, Exception) else None
        return results

async def fetch_ohlcv_single(exchange: ccxt_async.Exchange, symbol: str, timeframe: str, limit: int = 300) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}-{timeframe}"
    if cache_key in ohlcv_cache and time.time() - ohlcv_cache[cache_key]["timestamp"] < CONFIG["cache_ttl"]:
        logger.info(f"Using cached OHLCV data for {symbol}-{timeframe}")
        return ohlcv_cache[cache_key]["data"]
    
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        ohlcv_cache[cache_key] = {"data": df, "timestamp": time.time()}
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
    key_columns = ['ema20', 'ema50', 'ema200', 'rsi', 'macd', 'stoch_k', 'adx', 'volume', 'bb_position', 'buying_pressure']
    for col in key_columns:
        if col in df and df[col].isnull().all():
            logger.warning(f"All values in {col} are NaN")
    return df.fillna(method='ffill').fillna(0)

# ================== GOLDEN CROSS DETECTION ==================
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

# ================== CHART PATTERN DETECTION ==================
def get_chart_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    tol, lookback = 0.01, 20
    window, lookback_long = 20, 40
    
    close = df['close']
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    valleys = (close.shift(1) > close) & (close.shift(-1) > close)
    peak_idxs = close[peaks].index[-2:]
    valley_idxs = close[valleys].index[-2:]
    if len(peak_idxs) == 2 and abs(close.loc[peak_idxs[0]] - close.loc[peak_idxs[1]]) / close.loc[peak_idxs[0]] < tol:
        patterns.append("Double Top")
    if len(valley_idxs) == 2 and abs(close.loc[valley_idxs[0]] - close.loc[valley_idxs[1]]) / close.loc[valley_idxs[0]] < tol:
        patterns.append("Double Bottom")
    
    peak_idxs = close[peaks].index[-3:]
    valley_idxs = close[valleys].index[-3:]
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
    lh = highs.rolling(3).apply(lambda x: x[1] > x[0] and x[1] > x[2], raw=True).fillna(0)
    p = highs[lh > 0].tail(3)
    if len(p) == 3 and p.iloc[1] > p.iloc[0] and p.iloc[1] > p.iloc[2]:
        patterns.append("Head & Shoulders")
    
    lows = df['low']
    lh = lows.rolling(3).apply(lambda x: x[1] < x[0] and x[1] < x[2], raw=True).fillna(0)
    t = lows[lh > 0].tail(3)
    if len(t) == 3 and t.iloc[1] < t.iloc[0] and t.iloc[1] < t.iloc[2]:
        patterns.append("Inv Head & Shoulders")
    
    cl = df['close'][-45:]
    mid = 45 // 2
    if cl.iloc[0] > cl.min() and cl.iloc[mid] == cl.min() and cl.iloc[-1] > cl.iloc[mid]:
        patterns.append("Cup & Handle")
    
    close_window = df['close'][-window:]
    if close_window.is_monotonic_increasing and close_window.diff().min() > 0 and close_window.diff().max() / close_window.diff().min() < 4:
        patterns.append("Rising Wedge")
    if close_window.is_monotonic_decreasing and close_window.diff().max() < 0 and abs(close_window.diff().min() / close_window.diff().max()) < 4:
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
    if high.is_monotonic_increasing and low.is_monotonic_decreasing:
        patterns.append("Broadening")
    
    return patterns

# ================== SMC & STRUCTURE ANALYSIS ==================
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

# ================== MARKET REGIME DETECTION ==================
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

# ================== MULTI-TIMEFRAME CONFLUENCE ==================
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
            confluence_score += score
    return confluence_score, tf_data

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

# ================== OPTIMIZED ML CLASSIFIER ==================
class OptimizedCryptoMLClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model_name = None
        self.performance_history = {}
        self.last_retrained = 0

    def create_features(self, df: pd.DataFrame, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], golden_cross: str) -> Dict:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
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
            'vwap_signal': 1 if last.get('vwap') and last['close'] > last['vwap'] else -1 if last.get('vwap') and last['close'] < last['vwap'] else 0
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
        df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=1000)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for ML training: {symbol}-{timeframe}")
            return False

        df = apply_indicators(df)
        historical_signals = []

        for i in range(100, len(df) - 50):
            window = df.iloc[i-100:i]
            chart_patterns = get_chart_patterns(window)
            fvg = detect_fvg(window)
            choch = detect_bos_choch(window)
            sr_levels = find_sr_zones(window)
            golden_cross = detect_golden_cross(window)
            signal = get_enhanced_signal(window, chart_patterns, fvg, choch, sr_levels,
                                         detect_market_regime(window), 0, golden_cross, symbol)
            if signal:
                future_price = df['close'].iloc[i+50]
                outcome = 1 if (signal['direction'] == "LONG" and future_price > signal['entry']) or \
                              (signal['direction'] == "SHORT" and future_price < signal['entry']) else 0
                historical_signals.append({
                    'features': self.create_features(window, chart_patterns, fvg, choch, sr_levels, golden_cross),
                    'outcome': outcome,
                    'timestamp': time.time()
                })

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

    def save_model(self, filepath: str = 'optimized_crypto_ml_model.pkl') -> bool:
        try:
            model_data = pickle.dumps({
                'model': self.models.get('best'),
                'scaler': self.scalers.get('best'),
                'feature_names': self.feature_names,
                'performance': self.performance_history,
                'last_retrained': self.last_retrained
            })
            encrypted_data = fernet.encrypt(model_data)
            with open(filepath, 'wb') as f:
                f.write(encrypted_data)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filepath: str = 'optimized_crypto_ml_model.pkl') -> bool:
        try:
            with open(filepath, 'rb') as f:
                encrypted_data = f.read()
            model_data = pickle.loads(fernet.decrypt(encrypted_data))
            self.models['best'] = model_data.get('model')
            self.scalers['best'] = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.performance_history = model_data.get('performance', {})
            self.last_retrained = model_data.get('last_retrained', 0)
            logger.info(f"Model loaded from {filepath}")
            if self.performance_history:
                logger.info(f"ML Performance: Test {self.performance_history.get('test_score', 0):.3f}, AUC {self.performance_history.get('auc_score', 0):.3f}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

ml_classifier = OptimizedCryptoMLClassifier()

# ================== SIGNAL MANAGEMENT ==================
def enhanced_signal_score(df: pd.DataFrame, direction: str, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], regime: str, confluence_score: int, golden_cross: str) -> Tuple[int, List[str]]:
    total_score, all_patterns = 0, []
    context = {
        "chart_patterns": chart_patterns,
        "fvg": fvg,
        "choch": choch,
        "sr_levels": sr_levels,
        "regime": regime,
        "confluence_score": confluence_score,
        "golden_cross": golden_cross
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
    logger.info(f"Generating signal for {symbol}: Patterns={chart_patterns}, FVG={fvg}, CHOCH={choch}, SR={sr_levels}, Regime={regime}, Confluence={confluence_score}, GoldenCross={golden_cross}")
    for direction in ("LONG", "SHORT"):
        total_score, patterns = enhanced_signal_score(df, direction, chart_patterns, fvg, choch, sr_levels, regime, confluence_score, golden_cross)
        logger.info(f"{symbol}:{direction} Score: {total_score}, Patterns: {patterns}")
        if total_score >= CONFIG['conf_threshold']:
            last = df.iloc[-1]
            entry = last['close']
            if CONFIG['ml_enabled']:
                features = ml_classifier.create_features(df, chart_patterns, fvg, choch, sr_levels, golden_cross)
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
            return {
                "direction": direction, "entry": entry, "sl": sl, "tps": tps,
                "score": int(final_score), "ml_confidence": ml_confidence,
                "traditional_score": total_score, "patterns": patterns,
                "timestamp": time.time(), "rr": rr,
                "regime": regime, "confluence": confluence_score,
                "fvg": str(fvg[-1]) if fvg else "None", "choch": choch,
                "sr": sr_levels, "atr": last.get('atr', 0), "bar_ts": str(df.index[-1]),
                "golden_cross": golden_cross
            }
        else:
            logger.info(f"{symbol}:{direction} Score {total_score} below threshold {CONFIG['conf_threshold']}")
    return None

# ================== LIVE DATA MONITORING ==================
async def monitor_live_data(exchange: ccxt_async.Exchange):
    while True:
        try:
            for symbol in CONFIG["pairs"]:
                if symbol in invalid_symbols:
                    continue
                live_price = await fetch_live_price(exchange, symbol)
                if live_price:
                    live_data_cache[symbol]["price"] = live_price

            current_time = time.time()
            signals_to_remove = []
            for sig_key, signal in list(open_signals.items()):
                symbol = sig_key.split('-')[0]
                if symbol in invalid_symbols:
                    signals_to_remove.append(sig_key)
                    continue
                live_price = live_data_cache.get(symbol, {}).get("price")
                if not live_price:
                    continue
                if current_time - signal['timestamp'] > CONFIG['signal_lifetime_sec']:
                    signals_to_remove.append(sig_key)
                    await send_telegram(f"⌛ {symbol} signal expired ({signal['direction']})")
                    update_signal_outcome(sig_key, 0)
                    continue
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
                                if i == len(tps) - 1:
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
                                if i == len(tps) - 1:
                                    update_signal_outcome(sig_key, 1)
                                    signals_to_remove.append(sig_key)
            
            for sig_key in set(signals_to_remove):
                open_signals.pop(sig_key, None)
            await asyncio.sleep(CONFIG['live_data_interval'])
        except Exception as e:
            logger.error(f"Live monitor error: {str(e)}")
            await asyncio.sleep(10)

# ================== TELEGRAM & LOGGING ==================
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

def log_signal(signal: Dict):
    with open(CONFIG['log_file'], 'a') as f:
        f.write(json.dumps(signal, default=str) + "\n")

def clean_stale_signals(signals: Dict) -> Dict:
    now = time.time()
    return {k: v for k, v in signals.items() if now - v['timestamp'] < CONFIG['signal_lifetime_sec']}

def update_signal_outcome(signal_key: str, outcome: int):
    for signal in signal_history:
        if signal.get('key') == signal_key:
            signal['outcome'] = outcome
            break

# ================== MAIN MONITORING LOOP ==================
async def monitor():
    global open_signals, signal_history
    validate_config()
    exchange = ccxt_async.binance(CONFIG["exchanges"]["binance"])
    
    if not await test_exchange_connectivity(exchange, CONFIG["exchanges"]["binance"]["urls"]["api"]["public"]):
        logger.error("Cannot connect to Binance API. Exiting...")
        sys.exit(1)

    valid_pairs = await validate_symbols(exchange)
    CONFIG["pairs"] = valid_pairs
    logger.info(f"Valid pairs: {valid_pairs}")
    if not valid_pairs:
        logger.error("No valid symbols to monitor. Exiting...")
        sys.exit(1)

    if CONFIG['ml_enabled']:
        if not ml_classifier.load_model():
            logger.info("No pre-trained model found. Training new model...")
            if await ml_classifier.train_model(exchange, valid_pairs[0], CONFIG["timeframes"][0]):
                ml_classifier.save_model()
            else:
                logger.warning("Model training failed. Running without ML.")
                CONFIG["ml_enabled"] = False

    asyncio.create_task(monitor_live_data(exchange))
    cycle_count = 0
    retrain_counter = 0

    while True:
        cycle_count += 1
        logger.info(f"Starting analysis cycle #{cycle_count}")
        open_signals = clean_stale_signals(open_signals)

        if CONFIG['ml_enabled'] and len(signal_history) > 100:
            retrain_counter += 1
            if retrain_counter % 24 == 0:
                logger.info("Periodic model retraining...")
                outcomes = [s for s in signal_history if 'outcome' in s]
                if len(outcomes) >= 100:
                    if await ml_classifier.train_model(exchange, valid_pairs[0], CONFIG["timeframes"][0]):
                        ml_classifier.save_model()

        for symbol_batch in [CONFIG["pairs"][i:i+10] for i in range(0, len(CONFIG["pairs"]), 10)]:
            for tf in CONFIG["timeframes"]:
                try:
                    ohlcv_batch = await fetch_ohlcv_batch(exchange, symbol_batch, tf)
                    for symbol, df in ohlcv_batch.items():
                        if df is None or len(df) < 200:
                            logger.info(f"Data fetch failed or insufficient for {symbol}-{tf}: {len(df) if df is not None else 'None'} rows")
                            continue
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
                                continue
                            cross_info = " | ✨ Golden Cross" if golden_cross == "golden" else " | ☠️ Death Cross" if golden_cross == "death" else ""
                            logger.info(f"SIGNAL: {symbol} {tf} {sig['direction']} | Score: {sig['score']} | ML: {sig['ml_confidence']:.3f}{cross_info}")
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
                            await send_telegram(msg)
                except Exception as e:
                    logger.error(f"Error processing batch {symbol_batch}-{tf}: {str(e)}")
                await asyncio.sleep(3)
        
        if cycle_count % 6 == 0:
            status_msg = (f"📊 BOT STATUS\n"
                          f"🔄 Cycle: #{cycle_count} | ⏱ Uptime: {timedelta(seconds=int(time.time()-bot_status['last_update']))}\n"
                          f"📈 Active Signals: {len(open_signals)}\n"
                          f"📋 Total Signals: {len(signal_history)} | 🤖 ML Signals: {len([s for s in signal_history if s.get('ml_confidence', 0) > CONFIG['ml_threshold']])}\n"
                          f"🚫 Invalid Symbols: {len(invalid_symbols)}")
            if ml_classifier.performance_history:
                perf = ml_classifier.performance_history
                status_msg += f"\n🤖 ML Perf: Test {perf.get('test_score', 0):.3f} | AUC {perf.get('auc_score', 0):.3f}"
            await send_telegram(status_msg)
        
        logger.info(f"CYCLE COMPLETE: Open: {len(open_signals)} | Total: {len(signal_history)} | Invalid Symbols: {len(invalid_symbols)}")
        await exchange.close()
        await asyncio.sleep(CONFIG['scan_interval'])

async def list_perpetual_futures_pairs(exchange: ccxt_async.Exchange) -> List[str]:
    try:
        await exchange.load_markets(reload=True)
        markets = exchange.markets
        perpetual_pairs = [
            symbol for symbol, market in markets.items()
            if market.get('active', False) and
               market.get('type') == 'swap' and
               market.get('linear', False) and
               symbol.endswith('USDT')
        ]
        logger.info(f"Available Binance Perpetual Futures Pairs ({len(perpetual_pairs)}): {perpetual_pairs[:20]}")
        return perpetual_pairs
    except Exception as e:
        logger.error(f"Failed to list perpetual futures pairs: {str(e)}")
        return []
    finally:
        await exchange.close()

# ================== MAIN ENTRY POINT ==================
if __name__ == "__main__":
    logger.info("🚀 Enhanced Crypto Quant Trading Bot")
    logger.info("✨ Plugin System | Parallel Multi-TF | Secure Flask/Telegram")
    logger.info(f"Symbols: {len(CONFIG['pairs'])}")
    logger.info(f"Timeframes: {CONFIG['timeframes']}")
    logger.info(f"ML Enabled: {CONFIG['ml_enabled']}")
    logger.info(f"Indicator Plugins: {CONFIG['indicator_plugins']}")
    logger.info(f"Scoring Plugins: {CONFIG['scoring_plugins']}")
    
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(monitor())