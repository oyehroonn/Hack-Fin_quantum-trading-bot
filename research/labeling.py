"""Tradable label generation for ML training.

Moves beyond naive 'price went up' labels to labels that account for
trading costs, minimum edge requirements, and multi-barrier exits.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional

from loguru import logger


def direction_label(
    prices: np.ndarray,
    horizon: int = 1,
    threshold: float = 0.0,
) -> np.ndarray:
    """Binary direction label with optional minimum-move threshold.

    Args:
        prices: Close prices array
        horizon: Forward lookback periods
        threshold: Minimum return to qualify as 'up' (e.g., 0.001 = 0.1%)

    Returns:
        Binary labels: 1 = up (return > threshold), 0 = down/flat
    """
    forward_returns = np.zeros(len(prices))
    forward_returns[:-horizon] = prices[horizon:] / prices[:-horizon] - 1.0
    forward_returns[-horizon:] = np.nan

    labels = np.where(forward_returns > threshold, 1.0, 0.0)
    labels[np.isnan(forward_returns)] = np.nan

    return labels


def edge_label(
    prices: np.ndarray,
    horizon: int = 1,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> np.ndarray:
    """Label: 1 if forward return exceeds round-trip trading costs.

    This is the most realistic label for ML — it asks 'would this trade
    have been profitable after costs?'

    Args:
        prices: Close prices
        horizon: Forward periods
        fee_bps: Round-trip fee in basis points
        slippage_bps: Estimated slippage in basis points

    Returns:
        Binary labels: 1 = edge exists (return > costs), 0 = no edge
    """
    total_cost = (fee_bps + slippage_bps) / 10000.0  # Convert bps to fraction

    forward_returns = np.zeros(len(prices))
    forward_returns[:-horizon] = prices[horizon:] / prices[:-horizon] - 1.0
    forward_returns[-horizon:] = np.nan

    # Long edge: return > cost
    # Short edge: -return > cost (i.e., return < -cost)
    # Combined: |return| > cost
    labels = np.where(np.abs(forward_returns) > total_cost,
                       np.where(forward_returns > 0, 1.0, 0.0),
                       np.nan)  # NaN for ambiguous (within costs)

    # For classification: treat cost-zone as 0 (no trade)
    labels = np.nan_to_num(labels, nan=0.0)
    # Restore trailing NaN
    labels[-horizon:] = np.nan

    return labels


def triple_barrier_label(
    prices: np.ndarray,
    horizon: int = 10,
    take_profit: float = 0.02,
    stop_loss: float = 0.02,
) -> np.ndarray:
    """Triple-barrier labeling method (López de Prado).

    Three barriers:
      - Upper: price hits take_profit → label = 1
      - Lower: price hits stop_loss → label = 0
      - Vertical: time runs out → label based on sign of return

    Args:
        prices: Close prices
        horizon: Maximum holding period (vertical barrier)
        take_profit: Upward barrier as fraction (e.g., 0.02 = 2%)
        stop_loss: Downward barrier as fraction (e.g., 0.02 = 2%)

    Returns:
        Labels: 1 (profit), 0 (loss), NaN (insufficient data)
    """
    n = len(prices)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        entry_price = prices[i]
        upper = entry_price * (1 + take_profit)
        lower = entry_price * (1 - stop_loss)

        end_idx = min(i + horizon, n - 1)
        if end_idx <= i:
            continue

        label = None
        for j in range(i + 1, end_idx + 1):
            if prices[j] >= upper:
                label = 1.0  # Hit take profit
                break
            elif prices[j] <= lower:
                label = 0.0  # Hit stop loss
                break

        if label is None:
            # Vertical barrier: classify by return at horizon end
            final_return = prices[end_idx] / entry_price - 1.0
            label = 1.0 if final_return > 0 else 0.0

        labels[i] = label

    return labels


def volatility_label(
    prices: np.ndarray,
    horizon: int = 5,
    high_vol_quantile: float = 0.75,
) -> np.ndarray:
    """Label: 1 if forward volatility is above historical quantile.

    Useful for training vol-prediction models.

    Args:
        prices: Close prices
        horizon: Forward window for realised vol
        high_vol_quantile: Quantile threshold for 'high vol'

    Returns:
        Binary labels: 1 = high vol period, 0 = normal/low vol
    """
    returns = np.diff(prices) / prices[:-1]
    n = len(returns)

    forward_vols = np.full(n, np.nan)
    for i in range(n - horizon):
        forward_vols[i] = np.std(returns[i:i + horizon])

    threshold = np.nanquantile(forward_vols, high_vol_quantile)
    labels = np.where(forward_vols > threshold, 1.0, 0.0)

    # Pad to match prices length
    labels = np.append(labels, np.nan)

    return labels


def build_labeled_dataset(
    df: pd.DataFrame,
    label_type: str = "edge",
    horizon: int = 1,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    take_profit: float = 0.02,
    stop_loss: float = 0.02,
    threshold: float = 0.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """Build a labeled dataset from OHLCV DataFrame.

    Args:
        df: DataFrame with at least a close/price column
        label_type: One of 'direction', 'edge', 'triple_barrier', 'volatility'
        horizon: Forward lookback periods
        fee_bps: Trading fees in basis points
        slippage_bps: Slippage in basis points
        take_profit: For triple barrier
        stop_loss: For triple barrier
        threshold: For direction label
        price_col: Name of price column

    Returns:
        DataFrame with 'label' column added
    """
    prices = df[price_col].values

    if label_type == "direction":
        labels = direction_label(prices, horizon, threshold)
    elif label_type == "edge":
        labels = edge_label(prices, horizon, fee_bps, slippage_bps)
    elif label_type == "triple_barrier":
        labels = triple_barrier_label(prices, horizon, take_profit, stop_loss)
    elif label_type == "volatility":
        labels = volatility_label(prices, horizon)
    else:
        raise ValueError(f"Unknown label type: {label_type}")

    result = df.copy()
    result["label"] = labels

    valid_count = int(np.sum(~np.isnan(labels)))
    pos_count = int(np.nansum(labels == 1))
    logger.info(
        f"Generated {label_type} labels: {valid_count} valid, "
        f"{pos_count} positive ({pos_count / max(valid_count, 1) * 100:.1f}%)"
    )

    return result
