"""Model and strategy evaluation framework.

Computes cost-aware performance metrics, stability tests, and
walk-forward evaluation summaries.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from scipy import stats

from core.types import ModelMetrics


def compute_model_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    returns: Optional[np.ndarray] = None,
    costs_per_trade: float = 0.0015,
    risk_free_rate: float = 0.0,
    model_id: str = "unknown",
    model_version: str = "1.0.0",
    evaluation_type: str = "oos",
) -> ModelMetrics:
    """Compute comprehensive evaluation metrics.

    Args:
        predictions: Model predictions (probabilities or signals)
        actuals: Actual binary outcomes (1=up, 0=down)
        returns: Actual forward returns (if available, for PnL metrics)
        costs_per_trade: Round-trip cost as fraction
        risk_free_rate: Annual risk-free rate
        model_id: Model identifier
        model_version: Model version
        evaluation_type: 'oos', 'in_sample', 'walk_forward'

    Returns:
        ModelMetrics with all computed metrics
    """
    n = len(predictions)
    if n == 0:
        return _empty_metrics(model_id, model_version, evaluation_type)

    # Classification metrics
    pred_labels = (predictions > 0.5).astype(int)
    accuracy = float(np.mean(pred_labels == actuals))
    precision = _safe_divide(np.sum((pred_labels == 1) & (actuals == 1)), np.sum(pred_labels == 1))
    recall = _safe_divide(np.sum((pred_labels == 1) & (actuals == 1)), np.sum(actuals == 1))

    # Trading simulation
    if returns is not None:
        positions = np.where(predictions > 0.5, 1.0, -1.0)
        strategy_returns = positions * returns

        # Subtract costs on position changes
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * costs_per_trade
        net_returns = strategy_returns - costs

        # Metrics from strategy returns
        cumulative = np.cumprod(1 + net_returns)
        total_return = float(cumulative[-1] - 1.0)

        # Annualised metrics (assume daily)
        years = max(n / 252, 1 / 252)
        cagr = float((cumulative[-1]) ** (1 / years) - 1.0) if cumulative[-1] > 0 else 0.0

        daily_std = float(np.std(net_returns))
        annual_vol = daily_std * np.sqrt(252)
        sharpe = float((cagr - risk_free_rate) / annual_vol) if annual_vol > 0 else 0.0

        downside_returns = net_returns[net_returns < 0]
        downside_std = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 1 else 0.001
        sortino = float((cagr - risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(abs(np.min(drawdowns)))

        calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

        # Trade-level metrics
        trade_pnls = net_returns[position_changes > 0]
        num_trades = int(np.sum(position_changes > 0))
        winning_trades = trade_pnls[trade_pnls > 0]
        losing_trades = trade_pnls[trade_pnls < 0]

        win_rate = float(len(winning_trades) / max(num_trades, 1))
        avg_win = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
        avg_loss = float(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.001
        profit_factor = float(abs(np.sum(winning_trades) / np.sum(losing_trades))) if np.sum(losing_trades) != 0 else 0.0

        avg_trade_pnl = float(np.mean(trade_pnls)) if len(trade_pnls) > 0 else 0.0
        turnover = float(np.sum(position_changes) / n)

        # Stability: R² of log equity curve
        log_equity = np.log(np.maximum(cumulative, 1e-10))
        x = np.arange(len(log_equity))
        if len(x) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_equity)
            stability = float(r_value ** 2)
        else:
            stability = 0.0

    else:
        # No returns available — use classification-only metrics
        total_return = 0.0
        sharpe = 0.0
        sortino = 0.0
        max_dd = 0.0
        calmar = 0.0
        win_rate = accuracy
        profit_factor = 0.0
        num_trades = n
        avg_trade_pnl = 0.0
        turnover = 0.0
        stability = 0.0

    return ModelMetrics(
        model_id=model_id,
        model_version=model_version,
        evaluation_type=evaluation_type,
        timestamp=datetime.now(),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        total_return=round(total_return, 6),
        max_drawdown=round(max_dd, 6),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        num_trades=num_trades,
        avg_trade_pnl=round(avg_trade_pnl, 6),
        turnover=round(turnover, 4),
        calmar=round(calmar, 4),
        stability=round(stability, 4),
        extra={
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
    )


def walk_forward_evaluate(
    model_cls: type,
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    returns: Optional[np.ndarray] = None,
    train_size: int = 252,
    test_size: int = 63,
    step_size: int = 21,
    costs_per_trade: float = 0.0015,
    feature_names: Optional[list[str]] = None,
) -> list[ModelMetrics]:
    """Walk-forward evaluation: rolling train→test.

    Args:
        model_cls: TradingModel class (not instance)
        model_params: Parameters to pass to model constructor
        X: Full feature matrix
        y: Full label vector
        returns: Full returns vector (optional)
        train_size: Training window size
        test_size: Test window size
        step_size: Step between windows
        costs_per_trade: Cost per trade
        feature_names: Feature names

    Returns:
        List of ModelMetrics, one per walk-forward fold
    """
    n = len(X)
    results = []
    fold = 0

    start = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        test_returns = returns[train_end:test_end] if returns is not None else None

        # Train fresh model
        model = model_cls(**model_params)
        try:
            model.fit(X_train, y_train, feature_names)
            predictions = model.predict_proba(X_test)
        except Exception as e:
            logger.warning(f"Walk-forward fold {fold} failed: {e}")
            start += step_size
            fold += 1
            continue

        metrics = compute_model_metrics(
            predictions=predictions,
            actuals=y_test,
            returns=test_returns,
            costs_per_trade=costs_per_trade,
            model_id=model_params.get("model_id", "unknown"),
            model_version=model_params.get("version", "1.0.0"),
            evaluation_type="walk_forward",
        )

        results.append(metrics)
        logger.info(
            f"Fold {fold}: Sharpe={metrics.sharpe:.3f}, "
            f"Return={metrics.total_return:.4f}, Trades={metrics.num_trades}"
        )

        start += step_size
        fold += 1

    logger.info(f"Walk-forward: {len(results)} folds completed")
    return results


def summarise_walk_forward(folds: list[ModelMetrics]) -> dict[str, float]:
    """Summarise walk-forward results across folds."""
    if not folds:
        return {}

    return {
        "mean_sharpe": float(np.mean([f.sharpe for f in folds])),
        "std_sharpe": float(np.std([f.sharpe for f in folds])),
        "mean_return": float(np.mean([f.total_return for f in folds])),
        "mean_max_dd": float(np.mean([f.max_drawdown for f in folds])),
        "mean_win_rate": float(np.mean([f.win_rate for f in folds])),
        "mean_profit_factor": float(np.mean([f.profit_factor for f in folds])),
        "mean_stability": float(np.mean([f.stability for f in folds])),
        "total_trades": int(sum(f.num_trades for f in folds)),
        "num_folds": len(folds),
        "pct_positive_folds": float(np.mean([1 if f.total_return > 0 else 0 for f in folds])),
    }


def _safe_divide(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _empty_metrics(model_id: str, version: str, eval_type: str) -> ModelMetrics:
    return ModelMetrics(
        model_id=model_id,
        model_version=version,
        evaluation_type=eval_type,
        timestamp=datetime.now(),
    )
