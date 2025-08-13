# exchange.py
import ccxt.async_support as ccxt_async
import pandas as pd
from typing import Dict, List, Optional
from config import CONFIG
from utils import logger, retry, wait_exponential, stop_after_attempt
from database import db, ohlcv_collection
import asyncio
import time
import aiohttp

semaphore = asyncio.Semaphore(20)
live_data_cache = {}
invalid_symbols = set()

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
            if symbol in markets:
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

    async with semaphore:
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('ts', inplace=True)

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
        except Exception as e:
            logger.error(f"Fetch OHLCV error for {symbol} {timeframe}: {str(e)}")
            invalid_symbols.add(symbol)
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

async def fetch_tick_data(exchange: ccxt_async.Exchange, symbol: str, limit: int = 200) -> Dict:
    try:
        trades = await exchange.fetch_trades(symbol, limit=limit)
        buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
        sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
        return {'buy_vol': buy_vol, 'sell_vol': sell_vol, 'tick_imbalance': (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)}
    except Exception as e:
        logger.error(f"Tick data fetch error for {symbol}: {str(e)}")
        return {'buy_vol': 0, 'sell_vol': 0, 'tick_imbalance': 0}

async def setup_websockets(exchange: ccxt_async.Exchange):
    while True:
        try:
            symbols = [s for s in CONFIG["pairs"] if s not in invalid_symbols]
            if not symbols:
                await asyncio.sleep(10)
                continue
            ticker_tasks = [exchange.watch_ticker(symbol) for symbol in symbols]
            while True:
                results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
                for symbol, result in zip(symbols, results):
                    if not isinstance(result, Exception):
                        live_data_cache[symbol] = {"price": result.get('last', 0), "timestamp": time.time()}
                    else:
                        logger.warning(f"WS ticker error for {symbol}: {str(result)}")
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"WS reconnection needed: {str(e)}")
            await asyncio.sleep(5)