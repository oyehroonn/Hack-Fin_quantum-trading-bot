"""API router for model training, prediction, and management."""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import traceback
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytz
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import (
    ModelListResponse,
    ModelPredictRequest,
    ModelPredictResponse,
    ModelTrainRequest,
    ModelTrainResponse,
    RegimeDetectRequest,
    RegimeDetectResponse,
    SuggestRequest,
    SuggestResponse,
)

router = APIRouter(prefix="/api/models", tags=["models"])

# Lazy-initialised globals
_registry = None
_regime_detector = None


def _get_registry():
    global _registry
    if _registry is None:
        from models.registry import FileModelRegistry
        _registry = FileModelRegistry("models_registry")
    return _registry


def _get_regime_detector():
    global _regime_detector
    if _regime_detector is None:
        from features.regime import StatisticalRegimeDetector
        _regime_detector = StatisticalRegimeDetector()
    return _regime_detector


def _get_model_class(model_type: str):
    """Resolve model type string to class."""
    if model_type in ("xgb_direction", "xgb_classifier"):
        from models.ml.xgb_classifier import XGBDirectionClassifier
        return XGBDirectionClassifier
    elif model_type in ("lgbm_returns", "lgbm_regressor"):
        from models.ml.lgbm_regressor import LGBMReturnRegressor
        return LGBMReturnRegressor
    elif model_type in ("sma_crossover", "sma"):
        from models.baselines.sma_crossover import SMACrossoverModel
        return SMACrossoverModel
    elif model_type in ("mean_reversion", "mr"):
        from models.baselines.mean_reversion import MeanReversionModel
        return MeanReversionModel
    elif model_type in ("breakout",):
        from models.baselines.breakout import BreakoutModel
        return BreakoutModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")


async def _fetch_data(symbol: str, asset_class: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch historical data for training/prediction."""
    utc = pytz.UTC
    end = pd.Timestamp.now(tz=utc)
    start = end - pd.Timedelta(days=days)

    if asset_class == "crypto":
        from data.ingest.binance_public import BinancePublicIngestor
        ingestor = BinancePublicIngestor()
    else:
        from data.ingest.equities_yfinance import EquitiesYFinanceIngestor
        ingestor = EquitiesYFinanceIngestor()

    df = await ingestor.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    return df


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from OHLCV DataFrame."""
    close = df["close"].values
    n = len(close)

    feature_names = [
        "returns_1", "returns_5", "returns_10", "returns_20",
        "sma_ratio", "vol_20", "rsi",
    ]

    features = np.zeros((n, len(feature_names)))

    for i in range(1, n):
        features[i, 0] = close[i] / close[i - 1] - 1 if close[i - 1] > 0 else 0

    for i in range(5, n):
        features[i, 1] = close[i] / close[i - 5] - 1 if close[i - 5] > 0 else 0

    for i in range(10, n):
        features[i, 2] = close[i] / close[i - 10] - 1 if close[i - 10] > 0 else 0

    for i in range(20, n):
        features[i, 3] = close[i] / close[i - 20] - 1 if close[i - 20] > 0 else 0
        sma_10 = np.mean(close[i - 9:i + 1])
        sma_20 = np.mean(close[i - 19:i + 1])
        features[i, 4] = sma_10 / sma_20 if sma_20 > 0 else 1.0

        rets = np.diff(close[i - 19:i + 1]) / close[i - 19:i]
        features[i, 5] = float(np.std(rets) * np.sqrt(252))

    # RSI
    for i in range(15, n):
        deltas = np.diff(close[i - 14:i + 1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g = np.mean(gains)
        avg_l = np.mean(losses)
        if avg_l == 0:
            features[i, 6] = 100.0
        else:
            rs = avg_g / avg_l
            features[i, 6] = 100 - 100 / (1 + rs)

    return features, feature_names


@router.post("/train", response_model=ModelTrainResponse)
async def train_model(req: ModelTrainRequest):
    """Train a trading model on historical data."""
    try:
        # Fetch data
        df = await _fetch_data(req.symbol, req.asset_class, req.timeframe, req.train_days)
        close = df["close"].values

        # Build features
        X, feature_names = _build_features(df)

        # Build labels
        from research.labeling import direction_label, edge_label, triple_barrier_label
        if req.label_type == "direction":
            y = direction_label(close, req.horizon)
        elif req.label_type == "edge":
            y = edge_label(close, req.horizon, req.fee_bps)
        elif req.label_type == "triple_barrier":
            y = triple_barrier_label(close, req.horizon)
        else:
            y = direction_label(close, req.horizon)

        # Build returns for evaluation
        returns = np.zeros(len(close))
        returns[:-req.horizon] = close[req.horizon:] / close[:-req.horizon] - 1

        # Remove NaN rows
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X_valid = X[valid]
        y_valid = y[valid]
        returns_valid = returns[valid]

        if len(X_valid) < 50:
            raise HTTPException(status_code=400, detail="Not enough valid data for training")

        # Train/val split (time-ordered)
        split_idx = int(len(X_valid) * (1 - req.val_pct))
        X_train, X_val = X_valid[:split_idx], X_valid[split_idx:]
        y_train, y_val = y_valid[:split_idx], y_valid[split_idx:]
        ret_val = returns_valid[split_idx:]

        # Create and train model
        model_cls = _get_model_class(req.model_type)
        model = model_cls(model_id=req.model_id, **req.params)
        model.fit(X_train, y_train, feature_names, X_val=X_val, y_val=y_val)

        # Evaluate
        from research.evaluation import compute_model_metrics
        train_preds = model.predict_proba(X_train)
        val_preds = model.predict_proba(X_val)

        train_metrics = compute_model_metrics(
            train_preds, y_train, model_id=req.model_id,
            evaluation_type="in_sample",
        )
        val_metrics = compute_model_metrics(
            val_preds, y_val, returns=ret_val,
            model_id=req.model_id, evaluation_type="oos",
        )

        # Save to registry
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        registry = _get_registry()
        registry.save_model(
            model_id=req.model_id,
            version=version,
            artifact=model,
            metadata={
                "model_type": req.model_type,
                "symbol": req.symbol,
                "asset_class": req.asset_class,
                "timeframe": req.timeframe,
                "label_type": req.label_type,
                "feature_names": feature_names,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            },
        )
        registry.record_metrics(val_metrics)

        # Feature importance
        feat_imp = {}
        if hasattr(model, "get_feature_importance"):
            feat_imp = model.get_feature_importance()

        return ModelTrainResponse(
            success=True,
            model_id=req.model_id,
            version=version,
            train_metrics={
                "accuracy": train_metrics.extra.get("accuracy", 0),
                "sharpe": train_metrics.sharpe,
            },
            val_metrics={
                "sharpe": val_metrics.sharpe,
                "total_return": val_metrics.total_return,
                "win_rate": val_metrics.win_rate,
                "max_drawdown": val_metrics.max_drawdown,
                "num_trades": val_metrics.num_trades,
                "accuracy": val_metrics.extra.get("accuracy", 0),
            },
            feature_importance=feat_imp,
            message=f"Model trained: {len(X_train)} train, {len(X_val)} val samples",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=ModelPredictResponse)
async def predict(req: ModelPredictRequest):
    """Get prediction from a trained model."""
    try:
        registry = _get_registry()
        model = registry.load_model(req.model_id)

        # Fetch recent data
        df = await _fetch_data(req.symbol, req.asset_class, "1d", req.lookback_days)
        X, feature_names = _build_features(df)

        # Use last row
        X_last = X[-1:].copy()
        proba = model.predict_proba(X_last)[0]

        if proba > 0.55:
            signal = "BUY"
        elif proba < 0.45:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = abs(proba - 0.5) * 2

        # Regime detection
        close = df["close"].values
        detector = _get_regime_detector()
        regime_state = detector.detect(close.tolist())

        return ModelPredictResponse(
            success=True,
            symbol=req.symbol,
            signal=signal,
            probability=round(float(proba), 4),
            confidence=round(float(confidence), 4),
            regime=regime_state.regime.value,
            model_id=req.model_id,
            message=f"Prediction: {signal} with {confidence:.1%} confidence",
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found")
    except Exception as e:
        logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=ModelListResponse)
async def list_models(asset_class: Optional[str] = None):
    """List all registered models."""
    registry = _get_registry()
    models = registry.list_models(asset_class)
    return ModelListResponse(models=models)


@router.delete("/{model_id}")
async def delete_model(model_id: str, version: Optional[str] = None):
    """Delete a model (or specific version)."""
    registry = _get_registry()
    registry.delete_model(model_id, version)
    return {"success": True, "message": f"Deleted {model_id}" + (f" v{version}" if version else "")}


@router.post("/regime", response_model=RegimeDetectResponse)
async def detect_regime(req: RegimeDetectRequest):
    """Detect current market regime for a symbol."""
    try:
        df = await _fetch_data(req.symbol, req.asset_class, req.timeframe, req.lookback_days)
        close = df["close"].values

        detector = _get_regime_detector()
        state = detector.detect(close.tolist(), symbol=req.symbol)

        return RegimeDetectResponse(
            success=True,
            symbol=req.symbol,
            regime=state.regime.value,
            confidence=round(float(state.confidence), 4),
            indicators=state.indicators,
        )
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest", response_model=SuggestResponse)
async def smart_suggest(req: SuggestRequest):
    """Get intelligent trade suggestion using ensemble of models + regime detection."""
    try:
        df = await _fetch_data(req.symbol, req.asset_class, "1d", req.lookback_days)
        close = df["close"].values
        X, feature_names = _build_features(df)
        X_last = X[-1:].copy()

        # Detect regime
        detector = _get_regime_detector()
        regime_state = detector.detect(close.tolist(), symbol=req.symbol)

        # Collect signals from requested models
        model_signals: dict[str, dict[str, Any]] = {}
        probas = []

        for model_name in req.models:
            try:
                # Try loading from registry first
                registry = _get_registry()
                model = registry.load_model(model_name)
            except (FileNotFoundError, Exception):
                # Fall back to creating a fresh baseline model
                model_cls = _get_model_class(model_name)
                model = model_cls(model_id=model_name)
                # Baselines don't need training
                if not model.is_fitted:
                    model.fit(X[20:], (close[21:] > close[20:-1]).astype(float), feature_names)

            proba = float(model.predict_proba(X_last)[0])
            probas.append(proba)

            if proba > 0.55:
                sig = "BUY"
            elif proba < 0.45:
                sig = "SELL"
            else:
                sig = "HOLD"

            model_signals[model_name] = {
                "signal": sig,
                "probability": round(proba, 4),
                "confidence": round(abs(proba - 0.5) * 2, 4),
            }

        # Ensemble (simple average)
        avg_proba = float(np.mean(probas)) if probas else 0.5

        if avg_proba > 0.55:
            final_signal = "BUY"
        elif avg_proba < 0.45:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        confidence = abs(avg_proba - 0.5) * 2

        # Risk assessment
        vol_20 = X[-1, 5] if X.shape[1] > 5 else 0.15
        if vol_20 > 0.3:
            risk_level = "HIGH"
        elif vol_20 > 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return SuggestResponse(
            success=True,
            symbol=req.symbol,
            signal=final_signal,
            confidence=round(confidence, 4),
            probability=round(avg_proba, 4),
            regime=regime_state.regime.value,
            model_signals=model_signals,
            risk_level=risk_level,
            recommendation=(
                f"{final_signal} {req.symbol} | "
                f"Confidence: {confidence:.0%} | "
                f"Regime: {regime_state.regime.value} | "
                f"Risk: {risk_level}"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Suggest failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
