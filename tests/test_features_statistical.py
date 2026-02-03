"""Tests for statistical features."""

import numpy as np
import pandas as pd
import pytest

from features.statistical import (
    autocorr,
    log_returns,
    returns,
    rolling_corr,
    rolling_vol,
    zscore,
)


def test_returns():
    """Test simple returns."""
    series = pd.Series([100, 105, 110, 108, 115])
    result = returns(series, periods=1)
    
    assert len(result) == len(series)
    assert np.isnan(result.iloc[0])  # First value is NaN
    assert abs(result.iloc[1] - 0.05) < 0.001  # (105-100)/100 = 0.05


def test_log_returns():
    """Test log returns."""
    series = pd.Series([100, 105, 110])
    result = log_returns(series, periods=1)
    
    assert len(result) == len(series)
    assert np.isnan(result.iloc[0])  # First value is NaN
    expected = np.log(105 / 100)
    assert abs(result.iloc[1] - expected) < 0.001


def test_rolling_vol():
    """Test rolling volatility."""
    # Create a series with known volatility
    np.random.seed(42)
    returns_series = pd.Series(np.random.randn(100) * 0.01)
    result = rolling_vol(returns_series, window=20, annualize=False)
    
    assert len(result) == len(returns_series)
    assert not result.isna().any()
    assert (result >= 0).all()  # Volatility should be non-negative


def test_zscore():
    """Test Z-score."""
    series = pd.Series([100, 102, 98, 105, 103, 101])
    result = zscore(series, window=3)
    
    assert len(result) == len(series)
    assert not result.isna().any()
    # Z-score should have mean ~0 and std ~1 for large windows
    result_large = zscore(series, window=len(series))
    assert abs(result_large.mean()) < 0.1


def test_autocorr():
    """Test autocorrelation."""
    # Create a series with autocorrelation
    np.random.seed(42)
    series = pd.Series(np.random.randn(100))
    result = autocorr(series, lag=1, window=20)
    
    assert len(result) == len(series)
    # First values should be NaN (not enough data)
    assert np.isnan(result.iloc[0])


def test_rolling_corr():
    """Test rolling correlation."""
    np.random.seed(42)
    series1 = pd.Series(np.random.randn(100))
    series2 = series1 + np.random.randn(100) * 0.1  # Correlated
    
    result = rolling_corr(series1, series2, window=20)
    
    assert len(result) == len(series1)
    assert not result.isna().any()
    # Should be positive correlation
    assert result.mean() > 0
