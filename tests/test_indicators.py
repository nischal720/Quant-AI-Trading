# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from indicators import EMAPlugin, RSIPlugin, apply_indicators
import ccxt.async_support as ccxt_async
import asyncio

@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    df = pd.DataFrame({
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(95, 115, 100),
        'low': np.random.uniform(85, 105, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    return df

@pytest.mark.asyncio
async def test_ema_plugin(sample_df):
    plugin = EMAPlugin("ema")
    exchange = ccxt_async.binance()
    result = await plugin.compute(exchange, "BTC/USDT", sample_df)
    assert 'ema20' in result.columns
    assert 'ema50' in result.columns
    assert 'ema200' in result.columns
    assert 'golden_cross' in result.columns
    assert not result['ema20'].isna().all()

@pytest.mark.asyncio
async def test_rsi_plugin(sample_df):
    plugin = RSIPlugin("rsi")
    exchange = ccxt_async.binance()
    result = await plugin.compute(exchange, "BTC/USDT", sample_df)
    assert 'rsi' in result.columns
    assert 'rsi_fast' in result.columns
    assert 'rsi_slow' in result.columns
    assert not result['rsi'].isna().all()

@pytest.mark.asyncio
async def test_apply_indicators(sample_df):
    exchange = ccxt_async.binance()
    result = await apply_indicators(exchange, "BTC/USDT", sample_df)
    assert 'body_size' in result.columns
    assert 'is_quality_bar' in result.columns
    assert len(result) == len(sample_df)