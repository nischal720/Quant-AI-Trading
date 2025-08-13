# indicators.py
import pandas as pd
import pandas_ta as pta
import talib
import numpy as np
import ccxt.async_support as ccxt_async
from typing import List, Optional,Tuple,Dict
from config import CONFIG
from utils import logger

class IndicatorPlugin:
    def __init__(self, name: str):
        self.name = name

    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class EMAPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['ema20'] = pta.ema(df['close'], 20)
        df['ema50'] = pta.ema(df['close'], 50)
        df['ema200'] = pta.ema(df['close'], 200)
        df['golden_cross'] = (df['ema50'] > df['ema200']) & (df['ema50'].shift(1) <= df['ema200'].shift(1))
        df['death_cross'] = (df['ema50'] < df['ema200']) & (df['ema50'].shift(1) >= df['ema200'].shift(1))
        return df

class RSIPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = pta.rsi(df['close'], 14)
        df['rsi_fast'] = pta.rsi(df['close'], 7)
        df['rsi_slow'] = pta.rsi(df['close'], 21)
        return df

class MACDPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        macd = pta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macds'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        return df

class BollingerBandsPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        bb = pta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

class StochasticPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        stoch = pta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        return df

class WilliamsRPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['williams_r'] = pta.willr(df['high'], df['low'], df['close'])
        return df

class ADXPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        adx_data = pta.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx_data['ADX_14']
        df['dmp'] = adx_data['DMP_14']
        df['dmn'] = adx_data['DMN_14']
        return df

class ATRPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['atr'] = pta.atr(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close'] * 100
        return df

class VolatilityPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        vol_ma = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] > vol_ma * CONFIG['vol_mult']
        df['vol_ratio'] = df['volume'] / vol_ma
        return df

class OBVPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['obv'] = pta.obv(df['close'], df['volume'])
        return df

class CMFPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['cmf'] = pta.cmf(df['high'], df['low'], df['close'], df['volume'])
        return df

class MFIPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['mfi'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'])
        return df

class CandlePatternsPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
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
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df['upper_rejection'] = (df['high'] == df['high'].rolling(5).max()) & (df['close'] < df['high'] * 0.98)
        df['lower_rejection'] = (df['low'] == df['low'].rolling(5).min()) & (df['close'] > df['low'] * 1.02)
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        return df

class VWAPPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

class TrendLinePlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        # Find pivot highs and lows
        df['pivot_low'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
        df['pivot_high'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])

        n = len(df)
        x = np.arange(n)

        # Uptrend line (support): fit on pivot lows
        pivot_low_indices = np.where(df['pivot_low'])[0]
        if len(pivot_low_indices) >= 2:
            coef = np.polyfit(pivot_low_indices, df['low'].iloc[pivot_low_indices], 1)
            df['up_trend_line'] = coef[0] * x + coef[1]
        else:
            df['up_trend_line'] = np.nan

        # Downtrend line (resistance): fit on pivot highs
        pivot_high_indices = np.where(df['pivot_high'])[0]
        if len(pivot_high_indices) >= 2:
            coef = np.polyfit(pivot_high_indices, df['high'].iloc[pivot_high_indices], 1)
            df['down_trend_line'] = coef[0] * x + coef[1]
        else:
            df['down_trend_line'] = np.nan

        # Breakout
        df['up_breakout'] = (df['close'] > df['down_trend_line']) & (df['close'].shift(1) <= df['down_trend_line'].shift(1))
        df['down_breakout'] = (df['close'] < df['up_trend_line']) & (df['close'].shift(1) >= df['up_trend_line'].shift(1))

        # False breakout: breakout but reverses in next 3 bars
        df['false_up_breakout'] = False
        df['false_down_breakout'] = False
        for i in range(len(df) - 3):
            if df['up_breakout'].iloc[i]:
                if df['close'].iloc[i+3] < df['down_trend_line'].iloc[i+3]:
                    df.loc[df.index[i], 'false_up_breakout'] = True
            if df['down_breakout'].iloc[i]:
                if df['close'].iloc[i+3] > df['up_trend_line'].iloc[i+3]:
                    df.loc[df.index[i], 'false_down_breakout'] = True

        return df

class LiquidityPlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        try:
            order_book = await exchange.fetch_order_book(symbol, limit=50)
            bid_depth = sum([b[1] for b in order_book['bids'][:10]])
            ask_depth = sum([a[1] for a in order_book['asks'][:10]])
            df['bid_depth'] = bid_depth
            df['ask_depth'] = ask_depth
            df['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-6)
            df['total_depth'] = bid_depth + ask_depth
        except Exception as e:
            logger.error(f"Liquidity fetch error for {symbol}: {str(e)}")
            df['bid_depth'] = df['ask_depth'] = df['depth_imbalance'] = df['total_depth'] = 0
        return df

class VolumeProfilePlugin(IndicatorPlugin):
    async def compute(self, exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        prices = pd.concat([df['close'], df['high'], df['low']])
        volumes = df['volume'].repeat(3)
        hist, edges = np.histogram(prices, bins=50, weights=volumes)
        poc_idx = np.argmax(hist)
        df['vpoc'] = (edges[poc_idx] + edges[poc_idx + 1]) / 2
        vah_idx = poc_idx + int(0.68 * len(hist) / 2)  # Approximate value area
        val_idx = poc_idx - int(0.68 * len(hist) / 2)
        df['vah'] = edges[min(vah_idx, len(edges)-1)]
        df['val'] = edges[max(val_idx, 0)]
        return df

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
    "vwap": VWAPPlugin("vwap"),
    "trendline": TrendLinePlugin("trendline"),
    "liquidity": LiquidityPlugin("liquidity"),
    "volume_profile": VolumeProfilePlugin("volume_profile")
}

async def apply_indicators(exchange: ccxt_async.Exchange, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for plugin_name in CONFIG["indicator_plugins"]:
        if plugin_name in INDICATOR_PLUGINS:
            try:
                df = await INDICATOR_PLUGINS[plugin_name].compute(exchange, symbol, df)
            except Exception as e:
                logger.error(f"Error applying indicator {plugin_name} for {symbol}: {str(e)}")
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['roc'] = pta.roc(df['close'], length=10)
    df['cci'] = pta.cci(df['high'], df['low'], df['close'])
    vol_ma = df['volume'].rolling(20).mean()
    df['is_quality_bar'] = (df['volume'] >= CONFIG["min_volume_multiple"] * vol_ma) & (df['body_size'] >= CONFIG["min_body_ratio"] * df['range_pct'])
    return df.fillna(method='ffill').fillna(0)

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

def get_chart_patterns(df: pd.DataFrame) -> List[str]:
    patterns = []
    tol, lookback = 0.01, 20
    window, lookback_long = 20, 40
    
    close = df['close']
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    valleys = (close.shift(1) > close) & (close.shift(-1) > close)
    peak_idxs = close[peaks].index[-3:]  
    valley_idxs = close[valleys].index[-3:]
    
    if len(peak_idxs) >= 2 and abs(close.loc[peak_idxs[-2]] - close.loc[peak_idxs[-1]]) / close.loc[peak_idxs[-2]] < tol:
        patterns.append("Double Top")
    if len(valley_idxs) >= 2 and abs(close.loc[valley_idxs[-2]] - close.loc[valley_idxs[-1]]) / close.loc[valley_idxs[-2]] < tol:
        patterns.append("Double Bottom")
    
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
    lh = highs.rolling(3, center=True).apply(lambda x: x[1] > x[0] and x[1] > x[2], raw=True).fillna(0)
    p = highs[lh > 0].tail(3)
    if len(p) == 3 and p.iloc[1] > p.iloc[0] and p.iloc[1] > p.iloc[2]:
        patterns.append("Head & Shoulders")
    
    lows = df['low']
    lh = lows.rolling(3, center=True).apply(lambda x: x[1] < x[0] and x[1] < x[2], raw=True).fillna(0)
    t = lows[lh > 0].tail(3)
    if len(t) == 3 and t.iloc[1] < t.iloc[0] and t.iloc[1] < t.iloc[2]:
        patterns.append("Inv Head & Shoulders")
    
    cl = df['close'][-45:]
    mid = 45 // 2
    if cl.iloc[0] > cl.min() and cl.iloc[mid] == cl.min() and cl.iloc[-1] > cl.iloc[mid]:
        patterns.append("Cup & Handle")
    
    close_window = df['close'][-window:]
    if all(close_window.diff()[1:] > 0) and close_window.diff().max() / close_window.diff().min() < 4:
        patterns.append("Rising Wedge")
    if all(close_window.diff()[1:] < 0) and abs(close_window.diff().min() / close_window.diff().max()) < 4:
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
    if all(high.diff()[1:] > 0) and all(low.diff()[1:] < 0):
        patterns.append("Broadening")
    
    # Flag and Pennant
    if len(df) > 10:
        recent_highs = df['high'][-5:]
        recent_lows = df['low'][-5:]
        x = np.arange(len(recent_highs))
        slope_high = np.corrcoef(x, recent_highs)[0,1] * recent_highs.std() / x.std()
        slope_low = np.corrcoef(x, recent_lows)[0,1] * recent_lows.std() / x.std()
        if abs(slope_high - slope_low) < 0.01 and abs(slope_high) > 0:  # Parallel
            patterns.append("Flag")
        if abs(slope_high) > 0 and abs(slope_low) > 0 and slope_high * slope_low < 0:  # Converging
            patterns.append("Pennant")
    
    # Gaps
    if len(df) > 1:
        if df['open'].iloc[-1] > df['high'].iloc[-2]:
            patterns.append("Gap Up")
        if df['open'].iloc[-1] < df['low'].iloc[-2]:
            patterns.append("Gap Down")
    
    # Pipe Bottom
    if len(df) > 2:
        ranges = df['high'] - df['low']
        if ranges.iloc[-2] > ranges.iloc[:-2].max() and ranges.iloc[-1] > ranges.iloc[:-2].max():
            if df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] > (df['high'].iloc[-1] + df['low'].iloc[-1]) / 2:
                patterns.append("Pipe Bottom")
    
    # Narrow Range
    if len(df) > 4:
        ranges = df['high'] - df['low']
        if ranges.iloc[-1] < min(ranges.iloc[-4:-1]):
            patterns.append("Narrow Range")
    
    return patterns

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

def get_adaptive_tf_weights(df: pd.DataFrame) -> Dict:
    volatility = df['atr_pct'].tail(20).mean()
    weights = CONFIG["tf_weights"].copy()
    if volatility > 2.0:  # High vol, favor higher TFs
        weights["1d"] *= 1.5
        weights["4h"] *= 1.2
    elif volatility < 1.0:  # Low vol, favor lower TFs
        weights["1h"] *= 1.5
        weights["2h"] *= 1.2
    return weights