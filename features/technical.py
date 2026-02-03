"""Technical indicators for time-series features."""

from typing import Optional

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average.
    
    Args:
        series: Price series
        window: Window size
        
    Returns:
        SMA series
    """
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
    """Exponential Moving Average.
    
    Args:
        series: Price series
        window: Window size (used to compute alpha if not provided)
        alpha: Smoothing factor (0 < alpha <= 1). If None, computed from window.
        
    Returns:
        EMA series
    """
    if alpha is None:
        alpha = 2.0 / (window + 1.0)
    return series.ewm(alpha=alpha, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index.
    
    Args:
        series: Price series
        window: Window size (default 14)
        
    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        DataFrame with columns: macd, signal, histogram
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    })


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.
    
    Args:
        series: Price series
        window: Window size
        num_std: Number of standard deviations
        
    Returns:
        DataFrame with columns: upper, middle, lower, bandwidth
    """
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / middle
    
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
    })


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Window size (default 14)
        
    Returns:
        ATR series
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    
    return atr


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: Optional[int] = None,
) -> pd.Series:
    """Volume Weighted Average Price.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        window: Rolling window (None for cumulative)
        
    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    if window is None:
        # Cumulative VWAP
        vwap = pv.cumsum() / volume.cumsum()
    else:
        # Rolling VWAP
        vwap = pv.rolling(window=window, min_periods=1).sum() / volume.rolling(
            window=window, min_periods=1
        ).sum()
    
    return vwap
