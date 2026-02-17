"""Microstructure features: spread, order flow, trade imbalance.

These features capture market microstructure signals from
OHLCV data (approximations) or L2 order book data if available.
"""

import numpy as np
import pandas as pd
from typing import Optional


def spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Estimate bid-ask spread from OHLCV using Corwin-Schultz estimator.

    Based on the relationship between daily high-low range and spread.

    Returns:
        Estimated spread as fraction of price
    """
    beta = (np.log(high / low)) ** 2
    beta_sum = beta.rolling(2).sum()

    gamma = (np.log(high.rolling(2).max() / low.rolling(2).min())) ** 2

    alpha = (np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)) / (3 - 2 * np.sqrt(2))
    alpha = alpha.clip(lower=0)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def kyle_lambda(
    price: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Kyle's lambda: price impact per unit of volume.

    Measures how much prices move per unit of trading volume.
    Higher = less liquid, more impact.

    Returns:
        Rolling Kyle's lambda
    """
    returns = price.pct_change()
    signed_volume = volume * np.sign(returns)

    def _regression_slope(y: np.ndarray, x: np.ndarray) -> float:
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 3:
            return np.nan
        y_c, x_c = y[mask], x[mask]
        x_mean = np.mean(x_c)
        y_mean = np.mean(y_c)
        denom = np.sum((x_c - x_mean) ** 2)
        if denom == 0:
            return np.nan
        return float(np.sum((x_c - x_mean) * (y_c - y_mean)) / denom)

    result = pd.Series(index=price.index, dtype=float)
    for i in range(window, len(price)):
        r = returns.iloc[i - window:i].values
        sv = signed_volume.iloc[i - window:i].values
        result.iloc[i] = _regression_slope(r, sv)

    return result


def amihud_illiquidity(
    price: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Amihud illiquidity ratio: |return| / dollar volume.

    Higher = more illiquid.

    Returns:
        Rolling Amihud ratio
    """
    returns = price.pct_change().abs()
    dollar_volume = price * volume

    ratio = returns / dollar_volume.replace(0, np.nan)
    return ratio.rolling(window=window, min_periods=1).mean()


def volume_imbalance(
    close: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Buy/sell volume imbalance proxy.

    Approximates order flow using close position within bar.
    Positive = net buying, negative = net selling.

    Returns:
        Rolling volume imbalance (-1 to 1)
    """
    returns = close.pct_change()
    buy_volume = volume.where(returns > 0, 0)
    sell_volume = volume.where(returns < 0, 0)

    total = buy_volume.rolling(window).sum() + sell_volume.rolling(window).sum()
    imbalance = (buy_volume.rolling(window).sum() - sell_volume.rolling(window).sum()) / total.replace(0, 1)

    return imbalance.clip(-1, 1)


def trade_flow_toxicity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """VPIN-inspired flow toxicity measure.

    Measures the probability of informed trading based on
    volume-synchronised probability of informed trading.

    Returns:
        Flow toxicity score (0 to 1, higher = more toxic/informed)
    """
    returns = close.pct_change()
    buy_vol = volume.where(returns > 0, 0)
    sell_vol = volume.where(returns < 0, 0)

    abs_diff = (buy_vol - sell_vol).abs()
    total_vol = volume.rolling(window).sum()

    toxicity = abs_diff.rolling(window).sum() / total_vol.replace(0, 1)
    return toxicity.clip(0, 1)


def compute_microstructure_features(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute all microstructure features from OHLCV DataFrame.

    Args:
        df: DataFrame with open, high, low, close, volume columns
        window: Rolling window for features

    Returns:
        DataFrame with microstructure feature columns
    """
    features = pd.DataFrame(index=df.index)

    high = df["high"] if "high" in df.columns else df.get("High", pd.Series(dtype=float))
    low = df["low"] if "low" in df.columns else df.get("Low", pd.Series(dtype=float))
    close = df["close"] if "close" in df.columns else df.get("Close", pd.Series(dtype=float))
    volume = df["volume"] if "volume" in df.columns else df.get("Volume", pd.Series(dtype=float))

    if close.empty:
        return features

    features["spread_proxy"] = spread_proxy(high, low, close) if not high.empty else np.nan
    features["kyle_lambda"] = kyle_lambda(close, volume, window)
    features["amihud_illiq"] = amihud_illiquidity(close, volume, window)
    features["volume_imbalance"] = volume_imbalance(close, volume, window)
    features["flow_toxicity"] = trade_flow_toxicity(close, volume, window)

    return features
