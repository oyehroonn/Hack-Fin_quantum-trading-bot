"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from features.technical import atr, bollinger_bands, ema, macd, rsi, sma, vwap


def test_sma():
    """Test Simple Moving Average."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = sma(series, window=3)
    
    assert len(result) == len(series)
    assert result.iloc[0] == 1.0  # First value
    assert result.iloc[2] == 2.0  # (1+2+3)/3
    assert result.iloc[-1] == 9.0  # (8+9+10)/3


def test_ema():
    """Test Exponential Moving Average."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = ema(series, window=3)
    
    assert len(result) == len(series)
    assert not result.isna().any()
    # EMA should be more responsive to recent values
    assert result.iloc[-1] > sma(series, window=3).iloc[-1]


def test_rsi():
    """Test Relative Strength Index."""
    # Create a series with upward trend
    series = pd.Series([100, 102, 104, 106, 108, 110])
    result = rsi(series, window=3)
    
    assert len(result) == len(series)
    assert not result.isna().any()
    # RSI should be high for upward trend
    assert result.iloc[-1] > 50


def test_macd():
    """Test MACD."""
    series = pd.Series(range(100, 200))
    result = macd(series)
    
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_histogram" in result.columns
    assert len(result) == len(series)


def test_bollinger_bands():
    """Test Bollinger Bands."""
    series = pd.Series(range(100, 200))
    result = bollinger_bands(series, window=20)
    
    assert "bb_upper" in result.columns
    assert "bb_middle" in result.columns
    assert "bb_lower" in result.columns
    assert "bb_bandwidth" in result.columns
    # Upper should be above middle, lower should be below
    assert (result["bb_upper"] > result["bb_middle"]).all()
    assert (result["bb_lower"] < result["bb_middle"]).all()


def test_atr():
    """Test Average True Range."""
    high = pd.Series([110, 112, 115, 113, 118])
    low = pd.Series([100, 102, 105, 103, 108])
    close = pd.Series([105, 107, 110, 111, 115])
    
    result = atr(high, low, close, window=3)
    
    assert len(result) == len(high)
    assert not result.isna().any()
    assert (result > 0).all()  # ATR should always be positive


def test_vwap():
    """Test Volume Weighted Average Price."""
    high = pd.Series([110, 112, 115])
    low = pd.Series([100, 102, 105])
    close = pd.Series([105, 107, 110])
    volume = pd.Series([1000, 2000, 1500])
    
    result = vwap(high, low, close, volume)
    
    assert len(result) == len(high)
    assert not result.isna().any()
    # VWAP should be between low and high
    assert (result >= low).all()
    assert (result <= high).all()
