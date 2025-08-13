import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config import CONFIG
from utils import logger
from exchange import fetch_ohlcv_single
import ccxt.async_support as ccxt_async

def calculate_position_size(df: pd.DataFrame, entry: float, sl: float, account_balance: float = 10000) -> float:
    """Calculate position size based on volatility-adjusted risk"""
    atr = df.get('atr', pd.Series([1])).iloc[-1]
    risk_per_trade = CONFIG['risk_per_trade']
    risk_amount = account_balance * risk_per_trade
    risk_per_unit = abs(entry - sl)
    if risk_per_unit == 0:
        logger.warning("Zero risk per unit, using default position size")
        return 0.01  # Minimum lot size
    position_size = risk_amount / risk_per_unit
    # Adjust for volatility
    volatility = df['volatility'].iloc[-1] if 'volatility' in df else 1
    vol_adjustment = 1 / (1 + volatility)
    return max(0.01, position_size * vol_adjustment)

async def check_correlations(exchange: ccxt_async.Exchange, symbols: List[str], timeframe: str = "1d") -> Dict[str, float]:
    """Calculate correlations between assets to avoid overexposure"""
    correlations = {}
    dfs = {}
    for symbol in symbols:
        df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=100)
        if df is not None:
            dfs[symbol] = df['close']
    
    if len(dfs) < 2:
        return correlations
    
    df_combined = pd.DataFrame(dfs)
    corr_matrix = df_combined.pct_change().corr()
    
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            if sym1 in corr_matrix and sym2 in corr_matrix:
                corr = corr_matrix.loc[sym1, sym2]
                correlations[f"{sym1}-{sym2}"] = corr
                if corr > CONFIG['max_correlation']:
                    logger.warning(f"High correlation between {sym1} and {sym2}: {corr:.2f}")
    return correlations

def dynamic_risk_management(df: pd.DataFrame, entry: float, direction: str, regime: str) -> Tuple[float, List[float]]:
    """Dynamic SL and TP calculation with volatility adjustment"""
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

async def circuit_breaker(exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> bool:
    """Pause trading if extreme volatility detected"""
    if 'volatility' not in df:
        return False
    vol_ma = df['volatility'].rolling(50).mean().iloc[-1]
    if df['volatility'].iloc[-1] > CONFIG['circuit_breaker_vol_mult'] * vol_ma:
        logger.warning(f"Circuit breaker triggered for {symbol}: Volatility {df['volatility'].iloc[-1]} > {CONFIG['circuit_breaker_vol_mult']} * {vol_ma}")
        return True
    return False
