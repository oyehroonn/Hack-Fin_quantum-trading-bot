"""API router for research: walk-forward evaluation, tuning, experiments."""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import traceback
from typing import Any

import numpy as np
import pandas as pd
import pytz
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import (
    TuneRequest,
    TuneResponse,
    WalkForwardRequest,
    WalkForwardResponse,
)

router = APIRouter(prefix="/api/research", tags=["research"])


async def _fetch_data(symbol: str, asset_class: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch historical data."""
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
    """Build feature matrix from OHLCV."""
    close = df["close"].values
    n = len(close)

    feature_names = ["returns_1", "returns_5", "returns_10", "returns_20", "sma_ratio", "vol_20", "rsi"]
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
    for i in range(15, n):
        deltas = np.diff(close[i - 14:i + 1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g, avg_l = np.mean(gains), np.mean(losses)
        features[i, 6] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)

    return features, feature_names


def _get_model_class(model_type: str):
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


@router.post("/walk-forward", response_model=WalkForwardResponse)
async def run_walk_forward(req: WalkForwardRequest):
    """Run walk-forward evaluation for a model."""
    try:
        df = await _fetch_data(req.symbol, req.asset_class, req.timeframe, req.total_days)
        close = df["close"].values
        X, feature_names = _build_features(df)

        from research.labeling import direction_label, edge_label
        if req.label_type == "edge":
            y = edge_label(close, 1)
        else:
            y = direction_label(close, 1)

        returns = np.zeros(len(close))
        returns[:-1] = close[1:] / close[:-1] - 1

        # Filter valid
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X_v, y_v, ret_v = X[valid], y[valid], returns[valid]

        if len(X_v) < req.train_days + req.test_days:
            raise HTTPException(status_code=400, detail="Not enough data for walk-forward")

        model_cls = _get_model_class(req.model_type)
        model_params = {"model_id": req.model_id, **req.params}

        from research.evaluation import walk_forward_evaluate, summarise_walk_forward

        folds = walk_forward_evaluate(
            model_cls=model_cls,
            model_params=model_params,
            X=X_v, y=y_v, returns=ret_v,
            train_size=req.train_days,
            test_size=req.test_days,
            step_size=req.step_days,
            feature_names=feature_names,
        )

        summary = summarise_walk_forward(folds)

        fold_results = [
            {
                "fold": i,
                "sharpe": f.sharpe,
                "total_return": f.total_return,
                "max_drawdown": f.max_drawdown,
                "win_rate": f.win_rate,
                "num_trades": f.num_trades,
            }
            for i, f in enumerate(folds)
        ]

        return WalkForwardResponse(
            success=True,
            model_id=req.model_id,
            num_folds=len(folds),
            summary=summary,
            fold_results=fold_results,
            message=f"Walk-forward: {len(folds)} folds, mean Sharpe={summary.get('mean_sharpe', 0):.3f}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Walk-forward failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tune", response_model=TuneResponse)
async def tune_model(req: TuneRequest):
    """Run hyperparameter tuning for a model."""
    try:
        df = await _fetch_data(req.symbol, req.asset_class, req.timeframe, 500)
        close = df["close"].values
        X, feature_names = _build_features(df)

        from research.labeling import edge_label
        y = edge_label(close, 1)
        returns = np.zeros(len(close))
        returns[:-1] = close[1:] / close[:-1] - 1

        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X_v, y_v, ret_v = X[valid], y[valid], returns[valid]

        split = int(len(X_v) * 0.7)
        X_train, X_val = X_v[:split], X_v[split:]
        y_train, y_val = y_v[:split], y_v[split:]
        ret_val = ret_v[split:]

        model_cls = _get_model_class(req.model_type)

        if req.method == "grid":
            from research.tuning import grid_search
            results = grid_search(
                model_cls=model_cls,
                param_grid=req.param_grid,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                returns_val=ret_val,
                feature_names=feature_names,
                metric=req.metric,
            )
        else:
            from research.tuning import random_search
            results = random_search(
                model_cls=model_cls,
                param_distributions=req.param_grid,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                returns_val=ret_val,
                feature_names=feature_names,
                n_iter=req.n_iter,
                metric=req.metric,
            )

        best = results[0] if results else {}

        return TuneResponse(
            success=True,
            best_params=best.get("params", {}),
            best_metric=best.get("metric_value", 0),
            all_results=[
                {
                    "rank": r["rank"],
                    "params": {k: str(v) for k, v in r["params"].items()},
                    "metric_value": r["metric_value"],
                }
                for r in results[:20]
            ],
            message=f"Tuning complete: {len(results)} configs evaluated",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tuning failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(
    tags: str = "",
    status: str = "",
    limit: int = 50,
):
    """List research experiments."""
    from research.experiments import ExperimentTracker
    tracker = ExperimentTracker()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] or None
    return tracker.list_experiments(tags=tag_list, status=status or None, limit=limit)
