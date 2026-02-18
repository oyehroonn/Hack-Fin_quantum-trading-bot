"""Statistical analysis tools: confidence intervals, regime detection, rolling metrics."""

import numpy as np
import pandas as pd
from typing import Optional


def rolling_sharpe(returns: np.ndarray, window: int = 30, annualize: int = 365) -> np.ndarray:
    """Rolling Sharpe ratio."""
    r = pd.Series(returns)
    mean = r.rolling(window).mean()
    std = r.rolling(window).std()
    sharpe = (mean / std.replace(0, np.nan)) * np.sqrt(annualize)
    return sharpe.values


def confidence_interval(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Confidence interval for mean return using t-distribution."""
    from scipy import stats
    n = len(returns)
    if n < 3:
        return (0.0, 0.0)
    mean = np.mean(returns)
    sem = stats.sem(returns)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (float(mean - h), float(mean + h))


def detect_regime(prices: np.ndarray, vol_window: int = 20, vol_threshold: float = 0.5) -> str:
    """Simple volatility-based regime detection.

    Returns: 'low_vol', 'normal', or 'high_vol'
    """
    if len(prices) < vol_window + 5:
        return "normal"
    log_returns = np.diff(np.log(prices))
    recent_vol = np.std(log_returns[-vol_window:])
    historical_vol = np.std(log_returns)
    if historical_vol == 0:
        return "normal"
    vol_ratio = recent_vol / historical_vol
    if vol_ratio > (1 + vol_threshold):
        return "high_vol"
    elif vol_ratio < (1 - vol_threshold):
        return "low_vol"
    return "normal"


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute latest RSI value."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def compute_trend_strength(prices: np.ndarray, window: int = 20) -> float:
    """Trend strength: slope of linear regression on recent prices, normalized."""
    if len(prices) < window:
        return 0.0
    y = prices[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / np.mean(y))


def full_statistical_analysis(prices: np.ndarray) -> dict:
    """Comprehensive statistical summary for decision engine."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 30:
        return {"error": "Insufficient data"}

    log_returns = np.diff(np.log(prices))

    ci_low, ci_high = confidence_interval(log_returns)
    regime = detect_regime(prices)
    rsi = compute_rsi(prices)
    trend = compute_trend_strength(prices)

    return {
        "current_price": float(prices[-1]),
        "mean_daily_return": float(np.mean(log_returns)),
        "daily_volatility": float(np.std(log_returns)),
        "annualized_return": float(np.mean(log_returns) * 365),
        "annualized_volatility": float(np.std(log_returns) * np.sqrt(365)),
        "sharpe_ratio": float(np.mean(log_returns) / np.std(log_returns) * np.sqrt(365)) if np.std(log_returns) > 0 else 0,
        "confidence_interval_95": {"low": ci_low, "high": ci_high},
        "regime": regime,
        "rsi": rsi,
        "trend_strength": trend,
        "max_drawdown": float(np.min((prices - np.maximum.accumulate(prices)) / np.maximum.accumulate(prices))),
        "data_points": len(prices),
    }
