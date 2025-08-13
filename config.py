# config.py
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import logging
from typing import Dict, List

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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
    "timeframes": ["1h", "2h", "4h", "1d"],
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
    "indicator_plugins": ["ema", "rsi", "macd", "bbands", "stoch", "willr", "adx", "atr", "volatility", "obv", "cmf", "mfi", "candle_patterns", "order_flow", "vwap", "trendline", "liquidity", "volume_profile"],
    "scoring_plugins": ["ema_alignment", "golden_cross", "rsi", "macd", "stoch", "adx", "volume", "bbands", "order_flow", "chart_patterns", "smc", "vwap", "trendline_breakout", "liquidity", "volume_profile"],
    "min_volume_multiple": 0.5,
    "min_body_ratio": 0.3,
    "tf_weights": {"1h": 1.0, "2h": 1.2, "4h": 1.5, "1d": 2.0},
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
    },
    "anomaly_vol_threshold": 3.0,
    "health_check_interval": 300,
    "risk_per_trade": 0.01,  # 1% risk per trade
    "max_correlation": 0.8,  # Max correlation for portfolio
    "circuit_breaker_vol_mult": 5.0,  # Pause if volatility > this * avg
    "optuna_trials": 50,  # For hyperparameter optimization
}

def validate_config():
    required = ["telegram_token", "chat_id", "pairs", "timeframes", "indicator_plugins", "scoring_plugins"]
    for key in required:
        if not CONFIG[key] or (isinstance(CONFIG[key], str) and CONFIG[key].startswith("<")):
            logger.error(f"Missing or invalid {key} in configuration")
            raise ValueError(f"Missing or invalid {key}")
    if not all(isinstance(p, str) and p.endswith('USDT') for p in CONFIG["pairs"]):
        logger.error("Invalid trading pairs format")
        raise ValueError("Invalid trading pairs format")
    if not all(tf in ["1m", "5m", "15m", "1h", "2h", "4h", "12h", "1d"] for tf in CONFIG["timeframes"]):
        logger.error("Invalid timeframes")
        raise ValueError("Invalid timeframes")
    logger.info("Configuration validated successfully")