"""Statistical features for time-series."""

from typing import Optional

import numpy as np
import pandas as pd


def returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Simple returns.
    
    Args:
        series: Price series
        periods: Number of periods to look back
        
    Returns:
        Returns series
    """
    return series.pct_change(periods=periods)


def log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns.
    
    Args:
        series: Price series
        periods: Number of periods to look back
        
    Returns:
        Log returns series
    """
    return np.log(series / series.shift(periods=periods))


def rolling_vol(
    series: pd.Series,
    window: int,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling volatility.
    
    Args:
        series: Returns series
        window: Rolling window size
        annualize: Whether to annualize volatility
        periods_per_year: Periods per year for annualization
        
    Returns:
        Rolling volatility series
    """
    vol = series.rolling(window=window, min_periods=1).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score (standardization).
    
    Args:
        series: Series to standardize
        window: Rolling window size
        
    Returns:
        Z-score series
    """
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return (series - mean) / (std + 1e-10)  # Avoid division by zero


def autocorr(series: pd.Series, lag: int = 1, window: Optional[int] = None) -> pd.Series:
    """Autocorrelation.
    
    Args:
        series: Series to compute autocorrelation for
        lag: Lag period
        window: Rolling window (None for full series)
        
    Returns:
        Autocorrelation series
    """
    if window is None:
        # Full series autocorrelation (scalar)
        return series.autocorr(lag=lag)
    else:
        # Rolling autocorrelation
        def _autocorr(x):
            if len(x) < lag + 1:
                return np.nan
            return x.autocorr(lag=lag)
        
        return series.rolling(window=window, min_periods=lag + 1).apply(_autocorr, raw=False)


def rolling_corr(
    series1: pd.Series,
    series2: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling correlation between two series.
    
    Args:
        series1: First series
        series2: Second series
        window: Rolling window size
        
    Returns:
        Rolling correlation series
    """
    return series1.rolling(window=window, min_periods=1).corr(series2)


def rolling_skew(series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness.
    
    Args:
        series: Series to compute skewness for
        window: Rolling window size
        
    Returns:
        Rolling skewness series
    """
    return series.rolling(window=window, min_periods=1).skew()


def rolling_kurtosis(series: pd.Series, window: int) -> pd.Series:
    """Rolling kurtosis.
    
    Args:
        series: Series to compute kurtosis for
        window: Rolling window size
        
    Returns:
        Rolling kurtosis series
    """
    return series.rolling(window=window, min_periods=1).kurtosis()
