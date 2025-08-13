# scoring.py
from typing import List, Tuple, Dict,Optional
import pandas as pd
from config import CONFIG
from utils import logger
from risk_management import dynamic_risk_management
import time

class ScoringPlugin:
    def __init__(self, name: str):
        self.name = name

    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        raise NotImplementedError

class EMAAlignmentPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['ema20'] > last['ema50'] > last['ema200']:
            score += 20
            patterns.append("EMA Alignment")
        elif direction == "SHORT" and last['ema20'] < last['ema50'] < last['ema200']:
            score += 20
            patterns.append("EMA Alignment")
        return score, patterns

class GoldenCrossPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, golden_cross: str, **context) -> Tuple[int, List[str]]:
        score, patterns = 0, []
        if direction == "LONG" and golden_cross == "golden":
            score += 25
            patterns.append("Golden Cross")
        elif direction == "SHORT" and golden_cross == "death":
            score += 25
            patterns.append("Death Cross")
        return score, patterns

class RSIPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if 30 < last['rsi'] < 70:
            score += 10
            patterns.append("RSI Neutral")
        return score, patterns

class MACDPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last, prev = df.iloc[-1], df.iloc[-2]
        score, patterns = 0, []
        if direction == "LONG" and last['macd'] > last['macds'] and prev['macd'] < prev['macds']:
            score += 15
            patterns.append("MACD Crossover")
        elif direction == "SHORT" and last['macd'] < last['macds'] and prev['macd'] > prev['macds']:
            score += 15
            patterns.append("MACD Crossover")
        return score, patterns

class StochasticPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 80:
            score += 8
            patterns.append("Stochastic Bullish")
        elif direction == "SHORT" and last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > 20:
            score += 8
            patterns.append("Stochastic Bearish")
        return score, patterns

class ADXPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        threshold = context.get("adx_threshold", CONFIG['adx_threshold'])
        if last['adx'] > threshold:
            score += 12
            patterns.append("ADX Strong Trend")
        return score, patterns

class VolumePluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if last['vol_ratio'] > 1.5:
            score += 10
            patterns.append("Volume Spike")
        if last['vol_spike']:
            score += 5
            patterns.append("High Volume")
        return score, patterns

class BollingerBandsPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['bb_position'] < 0.2:
            score += 8
            patterns.append("BB Lower Band")
        elif direction == "SHORT" and last['bb_position'] > 0.8:
            score += 8
            patterns.append("BB Upper Band")
        return score, patterns

class OrderFlowPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['buying_pressure'] > 0.6:
            score += 10
            patterns.append("Buying Pressure")
        elif direction == "SHORT" and last['selling_pressure'] > 0.6:
            score += 10
            patterns.append("Selling Pressure")
        return score, patterns

class ChartPatternsPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, chart_patterns: List[str], **context) -> Tuple[int, List[str]]:
        score, patterns = 0, []
        if direction == "LONG":
            if "Double Bottom" in chart_patterns:
                score += 16
                patterns.append("Double Bottom")
            if "Inv Head & Shoulders" in chart_patterns:
                score += 20
                patterns.append("Inv H&S")
            if "Triple Bottom" in chart_patterns:
                score += 16
                patterns.append("Triple Bottom")
            if "Ascending Triangle" in chart_patterns:
                score += 15
                patterns.append("Ascending Triangle")
            if "Falling Wedge" in chart_patterns:
                score += 15
                patterns.append("Falling Wedge")
            if "Cup & Handle" in chart_patterns:
                score += 18
                patterns.append("Cup & Handle")
            if "Pipe Bottom" in chart_patterns:
                score += 16
                patterns.append("Pipe Bottom")
            if "Gap Up" in chart_patterns:
                score += 10
                patterns.append("Gap Up")
            if "Narrow Range" in chart_patterns:
                score += 12
                patterns.append("Narrow Range")
            if "Flag" in chart_patterns or "Pennant" in chart_patterns:
                score += 14
                patterns.append("Flag/Pennant")
        elif direction == "SHORT":
            if "Double Top" in chart_patterns:
                score += 16
                patterns.append("Double Top")
            if "Head & Shoulders" in chart_patterns:
                score += 20
                patterns.append("H&S")
            if "Triple Top" in chart_patterns:
                score += 16
                patterns.append("Triple Top")
            if "Descending Triangle" in chart_patterns:
                score += 15
                patterns.append("Descending Triangle")
            if "Rising Wedge" in chart_patterns:
                score += 15
                patterns.append("Rising Wedge")
            if "Gap Down" in chart_patterns:
                score += 10
                patterns.append("Gap Down")
            if "Flag" in chart_patterns or "Pennant" in chart_patterns:
                score += 14
                patterns.append("Flag/Pennant")
        return score, patterns

class SMCPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, fvg: List, choch: str, sr_levels: List[float], **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG":
            if choch == "up_bos":
                score += 15
                patterns.append("BOS(up)")
            if fvg and fvg[-1][0] == "bullish":
                score += 12
                patterns.append("Bull FVG")
        elif direction == "SHORT":
            if choch == "down_bos":
                score += 15
                patterns.append("BOS(down)")
            if fvg and fvg[-1][0] == "bearish":
                score += 12
                patterns.append("Bear FVG")
        if any(abs(last['close']-lvl)/last['close']<0.015 for lvl in sr_levels):
            score += 7
            patterns.append("SR Zone")
        return score, patterns

class VWAPPluginScoring(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last.get('vwap') and last['close'] > last['vwap']:
            score += 10
            patterns.append("Above VWAP")
        elif direction == "SHORT" and last.get('vwap') and last['close'] < last['vwap']:
            score += 10
            patterns.append("Below VWAP")
        return score, patterns

class TrendLineBreakoutPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last.get('up_breakout', False):
            score += 15
            patterns.append("Uptrend Breakout")
        elif direction == "SHORT" and last.get('down_breakout', False):
            score += 15
            patterns.append("Downtrend Breakout")
        # Check recent false breakouts
        prev = df.iloc[-10:-1]
        if direction == "LONG" and any(prev.get('false_up_breakout', False)):
            score -= 10
            patterns.append("Recent False Up Breakout")
        elif direction == "SHORT" and any(prev.get('false_down_breakout', False)):
            score -= 10
            patterns.append("Recent False Down Breakout")
        return score, patterns

class LiquidityScoringPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['depth_imbalance'] > 0.2 and last['total_depth'] > 1000:
            score += 15
            patterns.append("Strong Bid Liquidity")
        elif direction == "SHORT" and last['depth_imbalance'] < -0.2 and last['total_depth'] > 1000:
            score += 15
            patterns.append("Strong Ask Liquidity")
        return score, patterns

class VolumeProfileScoringPlugin(ScoringPlugin):
    def score(self, df: pd.DataFrame, direction: str, **context) -> Tuple[int, List[str]]:
        last = df.iloc[-1]
        score, patterns = 0, []
        if direction == "LONG" and last['close'] < last['vpoc'] and last['close'] > last['val']:
            score += 12
            patterns.append("Value Area Bounce")
        elif direction == "SHORT" and last['close'] > last['vpoc'] and last['close'] < last['vah']:
            score += 12
            patterns.append("Value Area Rejection")
        return score, patterns

SCORING_PLUGINS = {
    "ema_alignment": EMAAlignmentPlugin("ema_alignment"),
    "golden_cross": GoldenCrossPlugin("golden_cross"),
    "rsi": RSIPluginScoring("rsi"),
    "macd": MACDPluginScoring("macd"),
    "stoch": StochasticPluginScoring("stoch"),
    "adx": ADXPluginScoring("adx"),
    "volume": VolumePluginScoring("volume"),
    "bbands": BollingerBandsPluginScoring("bbands"),
    "order_flow": OrderFlowPluginScoring("order_flow"),
    "chart_patterns": ChartPatternsPluginScoring("chart_patterns"),
    "smc": SMCPluginScoring("smc"),
    "vwap": VWAPPluginScoring("vwap"),
    "trendline_breakout": TrendLineBreakoutPlugin("trendline_breakout"),
    "liquidity": LiquidityScoringPlugin("liquidity"),
    "volume_profile": VolumeProfileScoringPlugin("volume_profile")
}

def enhanced_signal_score(df: pd.DataFrame, direction: str, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], regime: str, confluence_score: int, golden_cross: str) -> Tuple[int, List[str]]:
    total_score, all_patterns = 0, []
    adx_threshold = CONFIG['adx_threshold']
    if regime == "trending":
        adx_threshold -= 5  # Lower threshold in trending markets
    elif regime == "ranging":
        adx_threshold += 5  # Higher in ranging
    elif regime == "volatile":
        adx_threshold += 3  # Adjust for volatility
    
    context = {
        "chart_patterns": chart_patterns,
        "fvg": fvg,
        "choch": choch,
        "sr_levels": sr_levels,
        "regime": regime,
        "confluence_score": confluence_score,
        "golden_cross": golden_cross,
        "adx_threshold": adx_threshold
    }
    
    for plugin_name in CONFIG["scoring_plugins"]:
        if plugin_name in SCORING_PLUGINS:
            try:
                score, patterns = SCORING_PLUGINS[plugin_name].score(df, direction, **context)
                total_score += score
                all_patterns.extend(patterns)
                logger.info(f"{plugin_name} score: {score}, patterns: {patterns}")
            except Exception as e:
                logger.error(f"Error applying scoring plugin {plugin_name}: {str(e)}")
    
    if regime == "trending":
        total_score *= 1.2
    elif regime == "ranging":
        total_score *= 0.8
    elif regime == "volatile":
        total_score *= 0.9

    if CONFIG['multi_tf_confluence'] and abs(confluence_score) > 1:
        total_score *= 1.15
    else:
        total_score *= 0.9

    return int(total_score), list(set(all_patterns))


def get_enhanced_signal(df: pd.DataFrame, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], regime: str, confluence_score: int, golden_cross: str, symbol: str) -> Optional[Dict]:
    from ml import ml_classifier
    if not df['is_quality_bar'].iloc[-1]:
        logger.info(f"{symbol}: Skipping signal generation due to low-quality bar")
        return None
    
    logger.info(f"Generating signal for {symbol}: Patterns={chart_patterns}, FVG={fvg}, CHOCH={choch}, SR={sr_levels}, Regime={regime}, Confluence={confluence_score}, GoldenCross={golden_cross}")
    for direction in ("LONG", "SHORT"):
        total_score, patterns = enhanced_signal_score(df, direction, chart_patterns, fvg, choch, sr_levels, regime, confluence_score, golden_cross)
        logger.info(f"{symbol}:{direction} Score: {total_score}, Patterns: {patterns}")
        if total_score >= CONFIG['conf_threshold']:
            last = df.iloc[-1]
            entry = last['close']
            if CONFIG['ml_enabled']:
                features = ml_classifier.create_features(df, chart_patterns, fvg, choch, sr_levels, golden_cross, regime)
                ml_confidence, ml_prediction = ml_classifier.predict(features)
                logger.info(f"{direction} signal - ML Confidence: {ml_confidence:.3f}")
                if ml_confidence < CONFIG['ml_threshold']:
                    logger.info(f"{direction} signal filtered out (confidence: {ml_confidence:.3f} < {CONFIG['ml_threshold']})")
                    continue
                final_score = total_score * ml_confidence
            else:
                ml_confidence = 0.5
                final_score = total_score
            sl, tps = dynamic_risk_management(df, entry, direction, regime)
            rr = round(abs(tps[0]-entry)/abs(entry-sl), 2) if abs(entry-sl) > 0 else 0
            # Pattern-based target
            pattern_target = None
            if chart_patterns:
                height = df['close'].max() - df['close'].min()
                if direction == "LONG":
                    pattern_target = entry + height
                else:
                    pattern_target = entry - height
            return {
                "direction": direction, "entry": entry, "sl": sl, "tps": tps,
                "score": int(final_score), "ml_confidence": ml_confidence,
                "traditional_score": total_score, "patterns": patterns,
                "timestamp": time.time(), "rr": rr,
                "regime": regime, "confluence": confluence_score,
                "fvg": str(fvg[-1]) if fvg else "None", "choch": choch,
                "sr": sr_levels, "atr": last.get('atr', 0), "bar_ts": str(df.index[-1]),
                "golden_cross": golden_cross, "pattern_target": pattern_target
            }
        else:
            logger.info(f"{symbol}:{direction} Score {total_score} below threshold {CONFIG['conf_threshold']}")
    return None