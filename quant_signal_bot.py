# main.py
import asyncio,sys,time,threading,json
import threading
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, json
from telegram import Bot
import ccxt.async_support as ccxt_async
import pandas as pd
import psutil
from config import CONFIG, validate_config
from utils import logger, rate_limit, sanitize_string
from exchange import test_exchange_connectivity, validate_symbols, setup_websockets, fetch_ohlcv_single, fetch_live_price, fetch_tick_data, live_data_cache, invalid_symbols
from database import init_mongo_indexes, save_bot_state, load_bot_state, log_signal, get_open_signals, clean_stale_signals, check_storage_usage, get_signal_history, update_open_signal, update_signal_outcome,mongo_client,ml_models_collection
from indicators import apply_indicators, detect_golden_cross, get_chart_patterns, detect_fvg, find_sr_zones, detect_bos_choch, detect_market_regime, get_adaptive_tf_weights
from scoring import get_enhanced_signal
from ml import ml_classifier, ensure_ml_model
from utils import fetch_json
from typing import List, Dict, Optional, Tuple
import numpy as np


app = Flask(__name__)
bot = Bot(token=CONFIG["telegram_token"])
bot_status = {"status": "starting", "last_update": time.time(), "signals_count": 0}


@app.route('/ping')
@rate_limit("flask")
async def home():
    return json.dumps({
        "message": "Crypto Quant Trading Bot is Running",
        "status": bot_status["status"],
        "uptime": time.time() - bot_status["last_update"],
    })

@app.route('/health')
@rate_limit("flask")
async def health():
    return "ok"

@app.route('/mongo_health')
@rate_limit("flask")
async def mongo_health():
    from database import db
    try:
        await db.command("ping")
        return json.dumps({"status": "MongoDB OK"})
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return json.dumps({"error": "MongoDB connection failed"}), 500

def run_flask():
    app.run(host="0.0.0.0", port=10000, use_reloader=False)

async def fetch_economic_calendar() -> List[Dict]:
    from database import economic_collection
    try:
        cursor = economic_collection.find().sort("date", -1)
        events = await cursor.to_list(length=1000)
        if events:
            logger.info(f"Using cached economic events from MongoDB ({len(events)} events)")
            return events
    except Exception as e:
        logger.error(f"Error fetching economic events from MongoDB: {str(e)}")

    try:
        data = await fetch_json(CONFIG["economic_calendar_url"])
        events = [
            {**event, "date": datetime.strptime(event["date"], '%Y-%m-%dT%H:%M:%S%z')}
            for event in data
            if event.get('impact', '').lower() == CONFIG["event_impact_threshold"]
            and event.get('currency', '') in CONFIG.get("event_currencies", ["USD"])
        ]
        if events:
            await economic_collection.delete_many({})
            await economic_collection.insert_many(events)
            logger.info(f"Saved {len(events)} economic events to MongoDB")
        return events
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

async def multi_timeframe_confluence(exchange: ccxt_async.Exchange, symbol: str) -> Tuple[int, Dict]:
    confluence_score = 0
    tf_data = {}
    tasks = [fetch_and_analyze_tf(exchange, symbol, tf) for tf in CONFIG["timeframes"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    df_primary = await fetch_ohlcv_single(exchange, symbol, CONFIG["timeframes"][0])
    adaptive_weights = get_adaptive_tf_weights(df_primary) if df_primary is not None else CONFIG["tf_weights"]
    
    for tf, result in zip(CONFIG["timeframes"], results):
        if isinstance(result, Exception):
            logger.error(f"Multi-TF error for {symbol}-{tf}: {str(result)}")
            continue
        if result:
            tf_data[tf], score = result
            weight = adaptive_weights.get(tf, 1.0)
            confluence_score += score * weight
    return int(confluence_score), tf_data

async def fetch_and_analyze_tf(exchange: ccxt_async.Exchange, symbol: str, timeframe: str) -> Tuple[Optional[pd.DataFrame], int]:
    try:
        df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=300)
        if df is None or len(df) < 200:
            logger.info(f"Insufficient data for {symbol}-{timeframe}: {len(df) if df is not None else 0} rows")
            return None, 0
        df = await apply_indicators(exchange, symbol, df)
        last = df.iloc[-1]
        if pd.isna(last.get('ema20', np.nan)) or pd.isna(last.get('ema50', np.nan)) or pd.isna(last.get('ema200', np.nan)):
            logger.info(f"Missing EMA values for {symbol}-{timeframe}")
            return df, 0
        score = 1 if last['ema20'] > last['ema50'] > last['ema200'] else -1 if last['ema20'] < last['ema50'] < last['ema200'] else 0
        return df, score
    except Exception as e:
        logger.error(f"Error analyzing {symbol}-{timeframe}: {str(e)}")
        return None, 0
    
async def send_telegram(message: str):
    from utils import rate_limit
    try:
        @rate_limit("telegram")
        async def send_message():
            await bot.send_message(chat_id=CONFIG["chat_id"], text=sanitize_string(message))
            logger.info(f"Telegram message sent: {message[:50]}...")
            return True
        return await send_message()
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}")
        return False
async def detect_anomaly(df: pd.DataFrame, symbol: str):
    if 'volatility' in df:
        vol_ma = df['volatility'].rolling(50).mean().iloc[-1]
        if df['volatility'].iloc[-1] > CONFIG['anomaly_vol_threshold'] * vol_ma:
            await send_telegram(f"⚠️ Anomaly: High volatility in {symbol}")
            logger.warning(f"Anomaly detected in {symbol}: Volatility {df['volatility'].iloc[-1]} > {CONFIG['anomaly_vol_threshold']} * {vol_ma}")

async def process_symbol_tf(exchange, symbol, tf, open_signals):
    try:
        df = await fetch_ohlcv_single(exchange, symbol, tf, limit=300)
        if df is None or len(df) < 200:
            logger.info(f"Data fetch failed or insufficient for {symbol}-{tf}: {len(df) if df is not None else 'None'} rows")
            return
        df = await apply_indicators(exchange, symbol, df)
        await detect_anomaly(df, symbol)
        tick_data = await fetch_tick_data(exchange, symbol)
        df.loc[df.index[-1], 'tick_imbalance'] = tick_data['tick_imbalance']
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

async def monitor():
    validate_config()
    exchange = ccxt_async.binance(CONFIG["exchanges"]["binance"])
    
    if not await test_exchange_connectivity(exchange, CONFIG["exchanges"]["binance"]["urls"]["api"]["public"]):
        logger.error("Cannot connect to Binance API. Exiting...")
        sys.exit(1)

    await init_mongo_indexes()
    bot_status, invalid_symbols = await load_bot_state()

    valid_pairs = await validate_symbols(exchange)
    CONFIG["pairs"] = valid_pairs
    logger.info(f"Valid pairs: {valid_pairs}")
    if not valid_pairs:
        logger.error("No valid symbols to monitor. Exiting...")
        sys.exit(1)

    if CONFIG['ml_enabled']:
        await ensure_ml_model(exchange, valid_pairs, CONFIG["timeframes"])

    asyncio.create_task(setup_websockets(exchange))
    asyncio.create_task(monitor_live_data(exchange))
    asyncio.create_task(system_health_monitor())
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
            await optimize_params()
            
        await save_bot_state(bot_status, invalid_symbols)
        logger.info(f"CYCLE COMPLETE: Open: {len(open_signals)} | Total: {len(signal_history)} | Invalid Symbols: {len(invalid_symbols)}")
        await asyncio.sleep(CONFIG['scan_interval'])

async def system_health_monitor():
    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            if cpu > 90 or mem > 90:
                await send_telegram(f"⚠️ System Health Alert: CPU {cpu}% | MEM {mem}%")
            logger.info(f"Health: CPU {cpu}% | MEM {mem}%")
            await asyncio.sleep(CONFIG['health_check_interval'])
        except Exception as e:
            logger.error(f"Health monitor error: {str(e)}")

async def analyze_signal_performance(symbol: str = None):
    from database import signal_collection
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

async def optimize_params():
    history = await get_signal_history(limit=200)
    recent = [s for s in history if 'outcome' in s][-50:]
    if len(recent) < 20:
        return
    win_rate = sum(s['outcome'] for s in recent) / len(recent)
    if win_rate < 0.5:
        CONFIG['conf_threshold'] = min(90, CONFIG['conf_threshold'] + 5)
    elif win_rate > 0.6:
        CONFIG['conf_threshold'] = max(60, CONFIG['conf_threshold'] - 5)
    logger.info(f"Self-tuned conf_threshold to {CONFIG['conf_threshold']} based on recent win_rate {win_rate:.2f}")

    # For indicator params, simple example for atr_mult
    def objective(x):
        atr_sl_mult = x[0]
        wins = 0
        for sig in recent:
            entry = sig['entry']
            sl = entry - atr_sl_mult * sig['atr'] if sig['direction'] == "LONG" else entry + atr_sl_mult * sig['atr']
            # To properly optimize, would need to simulate if sl hit before tp, but since we have outcome, perhaps adjust based on losses
            if sig['outcome'] == 0:  # Assume loss, perhaps widen sl
                wins += 0  # Simplified
        return -wins / len(recent) if len(recent) > 0 else 0

    from scipy.optimize import minimize
    res = minimize(objective, [CONFIG['atr_sl_mult']], bounds=[(1.0, 3.0)], method='L-BFGS-B')
    CONFIG['atr_sl_mult'] = res.x[0]
    logger.info(f"Optimized atr_sl_mult to {CONFIG['atr_sl_mult']:.2f}")

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
        asyncio.run(save_bot_state(bot_status, invalid_symbols))
    finally:
        exchange = ccxt_async.binance(CONFIG["exchanges"]["binance"])
        exchange.close()
        mongo_client.close()
        logger.info("Resources released. Goodbye!")