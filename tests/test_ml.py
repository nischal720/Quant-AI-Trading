# tests/test_ml.py
import pytest
import pandas as pd
import numpy as np
from ml import OptimizedCryptoMLClassifier
from indicators import get_chart_patterns, detect_fvg, find_sr_zones, detect_bos_choch, detect_market_regime
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
        'volume': np.random.uniform(1000, 5000, 100),
        'ema20': np.random.uniform(90, 110, 100),
        'ema50': np.random.uniform(90, 110, 100),
        'ema200': np.random.uniform(90, 110, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'macd_hist': np.random.uniform(-1, 1, 100),
        'atr_pct': np.random.uniform(0.01, 0.05, 100),
        'vol_ratio': np.random.uniform(0.5, 2, 100),
        'vol_spike': np.random.choice([True, False], 100),
        'bb_position': np.random.uniform(0, 1, 100),
        'buying_pressure': np.random.uniform(0, 1, 100),
        'false_up_breakout': np.random.choice([True, False], 100),
        'false_down_breakout': np.random.choice([True, False], 100),
    }, index=dates)
    return df

@pytest.mark.asyncio
async def test_ml_features(sample_df):
    ml = OptimizedCryptoMLClassifier()
    chart_patterns = get_chart_patterns(sample_df)
    fvg = detect_fvg(sample_df)
    sr_levels = find_sr_zones(sample_df)
    choch = detect_bos_choch(sample_df)
    golden_cross = "golden"
    regime = detect_market_regime(sample_df)
    features = ml.create_features(sample_df, chart_patterns, fvg, choch, sr_levels, golden_cross, regime)
    assert len(features) > 30  # Check for enhanced feature set
    assert 'rsi_momentum' in features
    assert 'volatility_change' in features

@pytest.mark.asyncio
async def test_ml_train(sample_df):
    ml = OptimizedCryptoMLClassifier()
    exchange = ccxt_async.binance()
    historical_signals = [
        {'features': ml.create_features(sample_df, [], [], "", [], "golden", "trending"), 'outcome': 1},
        {'features': ml.create_features(sample_df, [], [], "", [], "death", "ranging"), 'outcome': 0}
    ]
    X, y = ml.prepare_training_data(historical_signals)
    assert X.shape[0] == 2
    assert y.shape[0] == 2