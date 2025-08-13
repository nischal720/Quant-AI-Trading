# database.py
from motor.motor_asyncio import AsyncIOMotorClient
from config import CONFIG
from utils import logger
from pymongo.errors import CollectionInvalid
from datetime import datetime, timezone
import pickle
from cryptography.fernet import Fernet
from typing import Dict, List
import pandas as pd
import time

mongo_client = AsyncIOMotorClient(CONFIG["mongodb"]["uri"])
db = mongo_client[CONFIG["mongodb"]["database"]]
fernet = Fernet(CONFIG["encryption_key"].encode())
ohlcv_collection = db[CONFIG["mongodb"]["collections"]["ohlcv"]]
signal_collection = db[CONFIG["mongodb"]["collections"]["signals"]]
economic_collection = db[CONFIG["mongodb"]["collections"]["economic_events"]]
performance_collection = db[CONFIG["mongodb"]["collections"]["model_performance"]]
ml_models_collection = db[CONFIG["mongodb"]["collections"]["ml_models"]]
bot_state_collection = db[CONFIG["mongodb"]["collections"]["bot_state"]]

async def init_mongo_indexes():
    for col_name in CONFIG["mongodb"]["collections"].values():
        try:
            await db.create_collection(col_name)
            logger.info(f"Collection '{col_name}' created.")
        except CollectionInvalid:
            logger.info(f"Collection '{col_name}' already exists.")
        except Exception as e:
            logger.warning(f"Could not create collection '{col_name}': {str(e)}")

    await ohlcv_collection.create_index([("cache_key", 1)])
    await ohlcv_collection.create_index([("timestamp", 1)], expireAfterSeconds=CONFIG["cache_ttl"])
    await signal_collection.create_index([("key", 1), ("timestamp", -1)])
    await signal_collection.create_index([("timestamp", 1)], expireAfterSeconds=30*24*60*60)
    await economic_collection.create_index([("date", -1)])
    await performance_collection.create_index([("timestamp", -1)])
    await ml_models_collection.create_index([("model_id", 1)], unique=True)
    await bot_state_collection.create_index([("name", 1)], unique=True)
    logger.info("MongoDB indexes and TTLs created")

async def save_bot_state(bot_status: Dict, invalid_symbols: set):
    try:
        await bot_state_collection.update_one(
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

async def load_bot_state() -> tuple:
    try:
        state = await bot_state_collection.find_one({"name": "global_state"})
        if state:
            bot_status = state.get("bot_status", {})
            invalid_symbols = set(state.get("invalid_symbols", []))
            logger.info("Bot state loaded from MongoDB")
            return bot_status, invalid_symbols
        return {}, set()
    except Exception as e:
        logger.error(f"Error loading bot state: {str(e)}")
        return {}, set()

async def log_signal(signal: Dict):
    try:
        signal_copy = signal.copy()
        signal_copy["timestamp"] = datetime.fromtimestamp(signal_copy["timestamp"], tz=timezone.utc)
        signal_copy["bar_ts"] = pd.to_datetime(signal_copy["bar_ts"])
        await signal_collection.insert_one(signal_copy)
        logger.info(f"Signal saved to MongoDB for {signal['key']}")
    except Exception as e:
        logger.error(f"Error saving signal to MongoDB: {str(e)}")

async def get_signal_history(symbol: str = None, timeframe: str = None, limit: int = 1000) -> List[Dict]:
    try:
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
        cursor = signal_collection.find({"status": "open"}).sort("timestamp", -1)
        signals = await cursor.to_list(length=1000)
        return {s["key"]: {**s, "_id": str(s["_id"]), "timestamp": s["timestamp"].timestamp(), "bar_ts": s["bar_ts"].timestamp()} for s in signals}
    except Exception as e:
        logger.error(f"Error fetching open signals: {str(e)}")
        return {}

async def update_open_signal(signal: Dict, status: str = "open"):
    try:
        await signal_collection.update_one(
            {"key": signal["key"], "timestamp": datetime.fromtimestamp(signal["timestamp"], tz=timezone.utc)},
            {"$set": {"status": status}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error updating signal status: {str(e)}")

async def update_signal_outcome(signal_key: str, outcome: int):
    try:
        await signal_collection.update_one(
            {"key": signal_key},
            {"$set": {"outcome": outcome}}
        )
    except Exception as e:
        logger.error(f"Error updating signal outcome: {str(e)}")

async def check_storage_usage():
    stats = await db.command("dbStats")
    storage_mb = stats["storageSize"] / (1024 * 1024)
    logger.info(f"Current storage usage: {storage_mb:.2f} MB")
    return storage_mb

async def clean_stale_signals(open_signals: Dict[str, Dict]) -> Dict[str, Dict]:
    try:
        current_time = time.time()
        signals_to_remove = []
        for sig_key, signal in open_signals.items():
            signal_age = current_time - signal['timestamp']
            if signal_age > CONFIG['signal_lifetime_sec']:
                await update_open_signal(signal, status="closed")
                signals_to_remove.append(sig_key)
                logger.info(f"Closed stale signal: {sig_key} (age: {signal_age:.0f}s)")

        # Remove stale signals from the dictionary
        updated_signals = {k: v for k, v in open_signals.items() if k not in signals_to_remove}
        logger.info(f"Cleaned {len(signals_to_remove)} stale signals. Remaining open signals: {len(updated_signals)}")
        return updated_signals
    except Exception as e:
        logger.error(f"Error cleaning stale signals: {str(e)}")
        return open_signals