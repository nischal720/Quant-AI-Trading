# backtest.py
import pandas as pd
import numpy as np
from typing import Dict, List
from config import CONFIG
from utils import logger
from exchange import fetch_ohlcv_single
from indicators import apply_indicators, detect_golden_cross, get_chart_patterns, detect_fvg, find_sr_zones, detect_bos_choch, detect_market_regime
from scoring import get_enhanced_signal
from risk_management import calculate_position_size, dynamic_risk_management, circuit_breaker
import ccxt.async_support as ccxt_async
import asyncio

async def backtest_symbol(exchange: ccxt_async.Exchange, symbol: str, timeframe: str, start_date: str, end_date: str) -> Dict:
    df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=1500)
    if df is None or len(df) < 200:
        logger.warning(f"Insufficient data for backtesting {symbol}-{timeframe}")
        return {"trades": [], "metrics": {}}

    df = await apply_indicators(exchange, symbol, df)
    trades = []
    account_balance = 10000
    position = None

    for i in range(100, len(df) - 50):
        window = df.iloc[:i]
        regime = detect_market_regime(window)
        chart_patterns = get_chart_patterns(window)
        fvg = detect_fvg(window)
        choch = detect_bos_choch(window)
        sr_levels = find_sr_zones(window)
        golden_cross = detect_golden_cross(window)
        signal = get_enhanced_signal(window, chart_patterns, fvg, choch, sr_levels, regime, 0, golden_cross, symbol)

        if signal and not await circuit_breaker(exchange, symbol, window):
            if position and position['symbol'] == symbol:
                continue
            entry = signal['entry']
            sl, tps = dynamic_risk_management(window, entry, signal['direction'], regime)
            position_size = calculate_position_size(window, entry, sl, account_balance)
            signal['position_size'] = position_size
            signal['entry_time'] = window.index[-1]
            position = signal

        if position:
            current_price = df['close'].iloc[i]
            if position['direction'] == "LONG":
                if current_price <= position['sl']:
                    profit = (position['sl'] - position['entry']) * position['position_size']
                    account_balance += profit
                    trades.append({"symbol": symbol, "direction": "LONG", "entry": position['entry'], "exit": position['sl'], "profit": profit, "outcome": 0})
                    position = None
                elif current_price >= position['tps'][-1]:
                    profit = (position['tps'][-1] - position['entry']) * position['position_size']
                    account_balance += profit
                    trades.append({"symbol": symbol, "direction": "LONG", "entry": position['entry'], "exit": position['tps'][-1], "profit": profit, "outcome": 1})
                    position = None
            else:
                if current_price >= position['sl']:
                    profit = (position['entry'] - position['sl']) * position['position_size']
                    account_balance += profit
                    trades.append({"symbol": symbol, "direction": "SHORT", "entry": position['entry'], "exit": position['sl'], "profit": profit, "outcome": 0})
                    position = None
                elif current_price <= position['tps'][-1]:
                    profit = (position['entry'] - position['tps'][-1]) * position['position_size']
                    account_balance += profit
                    trades.append({"symbol": symbol, "direction": "SHORT", "entry": position['entry'], "exit": position['tps'][-1], "profit": profit, "outcome": 1})
                    position = None

    win_rate = sum(1 for t in trades if t['outcome'] == 1) / len(trades) if trades else 0
    total_profit = sum(t['profit'] for t in trades)
    metrics = {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "total_profit": total_profit,
        "final_balance": account_balance,
        "sharpe_ratio": np.mean([t['profit'] for t in trades]) / np.std([t['profit'] for t in trades]) if trades else 0
    }
    logger.info(f"Backtest results for {symbol}-{timeframe}: {metrics}")
    return {"trades": trades, "metrics": metrics}

async def run_backtest(exchange: ccxt_async.Exchange, symbols: List[str], timeframes: List[str]):
    results = {}
    for symbol in symbols:
        for timeframe in timeframes:
            result = await backtest_symbol(exchange, symbol, timeframe, "2023-01-01", "2023-12-31")
            results[f"{symbol}-{timeframe}"] = result
    return results