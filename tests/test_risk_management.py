# tests/test_risk_management.py
import pytest
import pandas as pd
import numpy as np
from risk_management import calculate_position_size, check_correlations, dynamic_risk_management, circuit_breaker
import ccxt.async_support as ccxt_async
import asyncio

@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    df = pd.DataFrame({
        'close': np.random.uniform(90, 110, 100),
        'atr': np.random.uniform(0.5, 2, 100),
        'volatility': np.random.uniform(0.01, 0.05, 100)
    }, index=dates)
    return df

def test_position_size(sample_df):
    entry = 100
    sl = 98
    size = calculate_position_size(sample_df, entry, sl)
    assert size > 0
    assert size <= 10000 / abs(entry - sl)  # Respect risk per trade

@pytest.mark.asyncio
async def test_correlations():
    exchange = ccxt_async.binance()
    symbols = ["BNB/USDT:USDT", "XRP/USDT:USDT"]
    correlations = await check_correlations(exchange, symbols)
    assert isinstance(correlations, dict)

def test_dynamic_risk_management(sample_df):
    entry = 100
    sl, tps = dynamic_risk_management(sample_df, entry, "LONG", "trending")
    assert sl < entry
    assert all(tp > entry for tp in tps)

@pytest.mark.asyncio
async def test_circuit_breaker(sample_df):
    exchange = ccxt_async.binance()
    sample_df['volatility'] = 10  # High volatility
    result = await circuit_breaker(exchange, "BTC/USDT", sample_df)
    assert result  # Should trigger