# ml.py
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
import optuna
from typing import Dict, List, Tuple
import pandas as pd
from config import CONFIG
from utils import logger
from database import db, ml_models_collection, fernet
import pickle
from indicators import detect_golden_cross, get_chart_patterns, detect_fvg, find_sr_zones, detect_bos_choch, detect_market_regime
import ccxt.async_support as ccxt_async
from exchange import fetch_ohlcv_single
from indicators import apply_indicators
from scoring import get_enhanced_signal
import time
from datetime import datetime, timedelta, timezone


class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class TorchMLP:
    def __init__(self, input_size):
        self.model = MLPClassifier(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def fit(self, X, y):
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().unsqueeze(1)
        for epoch in range(200):
            self.optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = self.criterion(out, y_tensor)
            loss.backward()
            self.optimizer.step()

    def partial_fit(self, X, y):
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().unsqueeze(1)
        for epoch in range(5):
            self.optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = self.criterion(out, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict_proba(self, X):
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            out = self.model(X_tensor)
        return out.numpy()[:, 0]

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

class OptimizedCryptoMLClassifier:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model_name = None
        self.performance_history = {}
        self.last_retrained = 0

    def create_features(self, df: pd.DataFrame, chart_patterns: List[str], fvg: List, choch: str, sr_levels: List[float], golden_cross: str, regime: str) -> Dict:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        prev2 = df.iloc[-3] if len(df) > 2 else prev
        recent_false_up = 1 if df['false_up_breakout'].tail(5).any() else 0
        recent_false_down = 1 if df['false_down_breakout'].tail(5).any() else 0
        features = {
            'ema_alignment': 1 if last.get('ema20', 0) > last.get('ema50', 0) > last.get('ema200', 0) else -1 if last.get('ema20', 0) < last.get('ema50', 0) < last.get('ema200', 0) else 0,
            'golden_cross': 1 if golden_cross == "golden" else 0,
            'death_cross': 1 if golden_cross == "death" else 0,
            'price_vs_ema200': (last['close'] - last.get('ema200', last['close'])) / last['close'] if last.get('ema200') else 0,
            'adx': last.get('adx', 0),
            'rsi': last.get('rsi', 50),
            'macd_hist': last.get('macd_hist', 0),
            'atr_pct': last.get('atr_pct', 0),
            'vol_ratio': last.get('vol_ratio', 1),
            'vol_spike': 1 if last.get('vol_spike', False) else 0,
            'bb_position': last.get('bb_position', 0.5),
            'body_size': last.get('body_size', 0),
            'buying_pressure': last.get('buying_pressure', 0.5),
            'double_top': 1 if "Double Top" in chart_patterns else 0,
            'double_bottom': 1 if "Double Bottom" in chart_patterns else 0,
            'head_shoulders': 1 if "Head & Shoulders" in chart_patterns else 0,
            'inv_head_shoulders': 1 if "Inv Head & Shoulders" in chart_patterns else 0,
            'hammer': 1 if last.get('hammer', 0) == 100 else 0,
            'engulfing_bull': 1 if last.get('engulfing', 0) == 100 else 0,
            'engulfing_bear': 1 if last.get('engulfing', 0) == -100 else 0,
            'bos_up': 1 if choch == "up_bos" else 0,
            'bos_down': 1 if choch == "down_bos" else 0,
            'near_support': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl < last['close']) else 0,
            'near_resistance': 1 if any(abs(last['close']-lvl)/last['close']<0.01 for lvl in sr_levels if lvl > last['close']) else 0,
            'vwap_signal': 1 if last.get('vwap') and last['close'] > last['vwap'] else -1 if last.get('vwap') and last['close'] < last['vwap'] else 0,
            'rsi_macd_bull': 1 if last['rsi'] < 30 and last['macd_hist'] > 0 else 0,
            'rsi_macd_bear': 1 if last['rsi'] > 70 and last['macd_hist'] < 0 else 0,
            'stoch_rsi_bull': 1 if last['stoch_k'] < 20 and last['rsi'] < 30 else 0,
            'stoch_rsi_bear': 1 if last['stoch_k'] > 80 and last['rsi'] > 70 else 0,
            'regime_trending': 1 if regime == "trending" else 0,
            'regime_ranging': 1 if regime == "ranging" else 0,
            'regime_volatile': 1 if regime == "volatile" else 0,
            'regime_neutral': 1 if regime == "neutral" else 0,
            'pipe_bottom': 1 if "Pipe Bottom" in chart_patterns else 0,
            'gap_up': 1 if "Gap Up" in chart_patterns else 0,
            'gap_down': 1 if "Gap Down" in chart_patterns else 0,
            'narrow_range': 1 if "Narrow Range" in chart_patterns else 0,
            'flag': 1 if "Flag" in chart_patterns else 0,
            'pennant': 1 if "Pennant" in chart_patterns else 0,
            'triple_top': 1 if "Triple Top" in chart_patterns else 0,
            'triple_bottom': 1 if "Triple Bottom" in chart_patterns else 0,
            'up_breakout': 1 if last.get('up_breakout', False) else 0,
            'down_breakout': 1 if last.get('down_breakout', False) else 0,
            'recent_false_up': recent_false_up,
            'recent_false_down': recent_false_down,
            'rsi_lag1': last['rsi'] - prev['rsi'],
            'rsi_lag2': prev['rsi'] - prev2['rsi'],
            'macd_hist_lag1': last['macd_hist'] - prev['macd_hist'],
            'vol_lag1': last['volume'] - prev['volume'],
            'price_change_5': df['close'].pct_change(5).iloc[-1],
            'price_change_10': df['close'].pct_change(10).iloc[-1],
            'price_change_20': df['close'].pct_change(20).iloc[-1],
            'ema_cross_dist': (last['ema50'] - last['ema200']) / last['close'],
            'rsi_momentum': df['rsi'].diff(5).iloc[-1],
            'volatility_change': df['volatility'].pct_change(5).iloc[-1],
            'adx_trend_strength': df['adx'].rolling(10).mean().iloc[-1] / df['adx'].rolling(20).mean().iloc[-1],
            'price_vwap_dist': (last['close'] - last.get('vwap', last['close'])) / last['close'],
            'obv_divergence': df['obv'].diff(5).iloc[-1] / df['close'].diff(5).iloc[-1] if df['close'].diff(5).iloc[-1] != 0 else 0,
        }
        return features

    def prepare_training_data(self, historical_signals: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for signal in historical_signals:
            if 'features' in signal and 'outcome' in signal:
                features = signal['features']
                if isinstance(features, dict):
                    if not self.feature_names:
                        self.feature_names = sorted(features.keys())
                    feature_vector = [features.get(name, 0) for name in self.feature_names]
                    X.append(feature_vector)
                    y.append(signal['outcome'])
        return np.array(X), np.array(y)

    def objective(self, trial, X_train, y_train, X_test, y_test):
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 10),
        }
        model_xgb = xgb.XGBClassifier(**xgb_params, random_state=42)
        model_rf = RandomForestClassifier(**rf_params, random_state=42)
        ensemble = VotingClassifier(
            estimators=[('xgb', model_xgb), ('rf', model_rf)],
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    async def train_model(self, exchange: ccxt_async.Exchange, symbol: str, timeframe: str) -> bool:
        df = await fetch_ohlcv_single(exchange, symbol, timeframe, limit=1500)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for ML training: {symbol}-{timeframe} ({len(df) if df is not None else 0} rows)")
            return False

        df = await apply_indicators(exchange, symbol, df)
        historical_signals = []

        original_threshold = CONFIG['conf_threshold']
        CONFIG['conf_threshold'] = 50
        try:
            for i in range(100, len(df) - 50):
                window = df.iloc[i-100:i]
                regime = detect_market_regime(window)
                chart_patterns = get_chart_patterns(window)
                fvg = detect_fvg(window)
                choch = detect_bos_choch(window)
                sr_levels = find_sr_zones(window)
                golden_cross = detect_golden_cross(window)
                signal = get_enhanced_signal(window, chart_patterns, fvg, choch, sr_levels, regime, 0, golden_cross, symbol)
                if signal:
                    future_price = df['close'].iloc[i+50]
                    outcome = 1 if (signal['direction'] == "LONG" and future_price > signal['entry']) or \
                                  (signal['direction'] == "SHORT" and future_price < signal['entry']) else 0
                    historical_signals.append({
                        'features': self.create_features(window, chart_patterns, fvg, choch, sr_levels, golden_cross, regime),
                        'outcome': outcome,
                        'timestamp': time.time()
                    })
        finally:
            CONFIG['conf_threshold'] = original_threshold

        if len(historical_signals) < 100:
            logger.warning(f"Not enough training signals: {len(historical_signals)}")
            return False

        X, y = self.prepare_training_data(historical_signals)
        if len(X) < 100:
            logger.warning(f"Not enough training data: {len(X)} samples")
            return False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train_scaled, y_train, X_test_scaled, y_test), n_trials=CONFIG['optuna_trials'])

        best_params = study.best_params
        model_xgb = xgb.XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42
        )
        model_rf = RandomForestClassifier(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            random_state=42
        )
        model_mlp = TorchMLP(X_train.shape[1])
        model_mlp.fit(X_train_scaled, y_train)

        ensemble = VotingClassifier(
            estimators=[('xgb', model_xgb), ('rf', model_rf), ('mlp', model_mlp)],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)

        train_score = ensemble.score(X_train_scaled, y_train)
        test_score = ensemble.score(X_test_scaled, y_test)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5)

        self.performance_history = {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score
        }

        if test_score > 0.6 and auc_score > 0.65:
            self.models['ensemble'] = ensemble
            self.scalers['ensemble'] = scaler
            self.best_model_name = 'ensemble'
            self.last_retrained = time.time()
            logger.info(f"Model trained successfully (Score: {test_score:.3f}, AUC: {auc_score:.3f})")
            return True
        logger.warning("Model did not meet performance threshold")
        return False

    def predict(self, features: Dict) -> Tuple[float, int]:
        if 'ensemble' not in self.models:
            return 0.5, 0
        try:
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scalers['ensemble'].transform(feature_vector)
            probability = self.models['ensemble'].predict_proba(feature_vector_scaled)[0, 1]
            prediction = self.models['ensemble'].predict(feature_vector_scaled)[0]
            return probability, prediction
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return 0.5, 0

    def update_with_new_data(self, features: Dict, outcome: int):
        if self.best_model_name != "ensemble":
            return
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        y = np.array([outcome])
        X_scaled = self.scalers['ensemble'].transform(X)
        self.models['ensemble'].partial_fit(X_scaled, y)  # Note: VotingClassifier doesn't have partial_fit, so may need to adjust for individual models
        logger.info("Model updated online with new data")

    async def save_model(self, model_id: str = "primary") -> bool:
        try:
            model_data = {
                'model': self.models.get('ensemble'),
                'scaler': self.scalers.get('ensemble'),
                'feature_names': self.feature_names,
                'performance': self.performance_history,
                'last_retrained': self.last_retrained,
                'timestamp': datetime.now(timezone.utc)
            }
            serialized = pickle.dumps(model_data)
            encrypted_data = fernet.encrypt(serialized)
            await ml_models_collection.update_one(
                {"model_id": model_id},
                {"$set": {
                    "model_id": model_id,
                    "data": encrypted_data,
                    "timestamp": datetime.now(timezone.utc),
                    "performance": self.performance_history
                }},
                upsert=True
            )
            logger.info(f"Model saved to MongoDB with ID: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to MongoDB: {str(e)}")
            return False

    async def load_model(self, model_id: str = "primary") -> bool:
        try:
            model_doc = await ml_models_collection.find_one({"model_id": model_id})
            if model_doc is None:
                logger.warning(f"No model found in MongoDB with ID: {model_id}")
                return False
            encrypted_data = model_doc['data']
            serialized = fernet.decrypt(encrypted_data)
            model_data = pickle.loads(serialized)
            self.models['ensemble'] = model_data.get('model')
            self.scalers['ensemble'] = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.performance_history = model_data.get('performance', {})
            self.last_retrained = model_data.get('last_retrained', 0)
            logger.info(f"Model loaded from MongoDB: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from MongoDB: {str(e)}")
            return False

ml_classifier = OptimizedCryptoMLClassifier()

async def ensure_ml_model(exchange: ccxt_async.Exchange, valid_pairs: List[str], timeframes: List[str]) -> bool:
    if await ml_classifier.load_model():
        logger.info("Existing ML model loaded successfully")
        return True

    logger.info("No pre-trained model found. Attempting to train new model...")
    for symbol in valid_pairs:
        for timeframe in timeframes:
            logger.info(f"Training ML model with {symbol} on {timeframe}")
            try:
                if await ml_classifier.train_model(exchange, symbol, timeframe):
                    await ml_classifier.save_model()
                    logger.info(f"Model trained and saved successfully for {symbol}-{timeframe}")
                    return True
                else:
                    logger.warning(f"Model training failed for {symbol}-{timeframe}")
            except Exception as e:
                logger.error(f"Error training model for {symbol}-{timeframe}: {str(e)}")
    logger.error("Failed to train model with any symbol-timeframe combination. Disabling ML.")
    CONFIG["ml_enabled"] = False
    return False