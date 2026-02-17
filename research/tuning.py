"""Hyperparameter tuning for trading models.

Provides grid search and random search runners that integrate
with the experiment tracker and walk-forward evaluation.
"""

import itertools
import numpy as np
from typing import Any, Optional

from loguru import logger

from research.evaluation import compute_model_metrics, walk_forward_evaluate, summarise_walk_forward
from research.experiments import ExperimentTracker


def grid_search(
    model_cls: type,
    param_grid: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    returns_val: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    costs_per_trade: float = 0.0015,
    metric: str = "sharpe",
    tracker: Optional[ExperimentTracker] = None,
) -> list[dict[str, Any]]:
    """Grid search over parameter combinations.

    Args:
        model_cls: TradingModel class
        param_grid: {param_name: [value1, value2, ...]}
        X_train, y_train: Training data
        X_val, y_val: Validation data
        returns_val: Validation returns for PnL metrics
        feature_names: Feature names
        costs_per_trade: Trading costs
        metric: Metric to optimise ('sharpe', 'total_return', 'win_rate', etc.)
        tracker: Optional experiment tracker

    Returns:
        Sorted list of {params, metrics, rank}
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(itertools.product(*param_values))

    logger.info(f"Grid search: {len(all_combos)} combinations for {model_cls.__name__}")

    results = []
    for i, combo in enumerate(all_combos):
        params = dict(zip(param_names, combo))

        exp_id = None
        if tracker:
            exp_id = tracker.start_experiment(
                name=f"grid_{model_cls.__name__}",
                config={"model_cls": model_cls.__name__, "params": params},
                tags=["grid_search"],
            )

        try:
            model = model_cls(**params)
            model.fit(X_train, y_train, feature_names, X_val=X_val, y_val=y_val)
            predictions = model.predict_proba(X_val)

            metrics = compute_model_metrics(
                predictions=predictions,
                actuals=y_val,
                returns=returns_val,
                costs_per_trade=costs_per_trade,
                model_id=params.get("model_id", model_cls.__name__),
                model_version=str(i),
                evaluation_type="validation",
            )

            result = {
                "params": params,
                "metric_value": getattr(metrics, metric, 0.0),
                "metrics": metrics,
            }
            results.append(result)

            if tracker and exp_id:
                tracker.log_metrics(exp_id, metrics, phase="val")
                tracker.complete_experiment(exp_id)

            logger.info(f"  [{i + 1}/{len(all_combos)}] {metric}={result['metric_value']:.4f} | {params}")

        except Exception as e:
            logger.warning(f"  [{i + 1}/{len(all_combos)}] Failed: {e}")
            if tracker and exp_id:
                tracker.complete_experiment(exp_id, status="failed", notes=str(e))

    # Sort by metric (descending)
    results.sort(key=lambda x: x["metric_value"], reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    if results:
        best = results[0]
        logger.info(f"Best: {metric}={best['metric_value']:.4f} | {best['params']}")

    return results


def random_search(
    model_cls: type,
    param_distributions: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    returns_val: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    n_iter: int = 20,
    costs_per_trade: float = 0.0015,
    metric: str = "sharpe",
    random_state: int = 42,
    tracker: Optional[ExperimentTracker] = None,
) -> list[dict[str, Any]]:
    """Random search over parameter distributions.

    Args:
        model_cls: TradingModel class
        param_distributions: {param_name: (low, high) or [values]}
        n_iter: Number of random samples
        Other args same as grid_search

    Returns:
        Sorted list of {params, metrics, rank}
    """
    rng = np.random.default_rng(random_state)

    logger.info(f"Random search: {n_iter} iterations for {model_cls.__name__}")

    results = []
    for i in range(n_iter):
        params = {}
        for name, dist in param_distributions.items():
            if isinstance(dist, list):
                params[name] = rng.choice(dist)
            elif isinstance(dist, tuple) and len(dist) == 2:
                low, high = dist
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = int(rng.integers(low, high + 1))
                else:
                    params[name] = float(rng.uniform(low, high))
            else:
                params[name] = dist

        exp_id = None
        if tracker:
            exp_id = tracker.start_experiment(
                name=f"random_{model_cls.__name__}",
                config={"model_cls": model_cls.__name__, "params": {k: str(v) for k, v in params.items()}},
                tags=["random_search"],
            )

        try:
            model = model_cls(**params)
            model.fit(X_train, y_train, feature_names, X_val=X_val, y_val=y_val)
            predictions = model.predict_proba(X_val)

            metrics = compute_model_metrics(
                predictions=predictions,
                actuals=y_val,
                returns=returns_val,
                costs_per_trade=costs_per_trade,
                model_id=params.get("model_id", model_cls.__name__),
                model_version=str(i),
                evaluation_type="validation",
            )

            result = {
                "params": params,
                "metric_value": getattr(metrics, metric, 0.0),
                "metrics": metrics,
            }
            results.append(result)

            if tracker and exp_id:
                tracker.log_metrics(exp_id, metrics, phase="val")
                tracker.complete_experiment(exp_id)

            logger.info(f"  [{i + 1}/{n_iter}] {metric}={result['metric_value']:.4f}")

        except Exception as e:
            logger.warning(f"  [{i + 1}/{n_iter}] Failed: {e}")
            if tracker and exp_id:
                tracker.complete_experiment(exp_id, status="failed", notes=str(e))

    results.sort(key=lambda x: x["metric_value"], reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    return results
