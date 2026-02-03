"""Tests for feature store."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.feature_store import FeatureStore


@pytest.fixture
def sample_bars():
    """Create sample bar data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    np.random.seed(42)
    
    # Generate realistic price data
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    return pd.DataFrame(
        {
            "open": prices + np.random.randn(100) * 0.1,
            "high": prices + np.abs(np.random.randn(100) * 0.2),
            "low": prices - np.abs(np.random.randn(100) * 0.2),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )


@pytest.fixture
def cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_path = tmp_path / "features_cache"
    cache_path.mkdir()
    yield str(cache_path)
    shutil.rmtree(cache_path, ignore_errors=True)


def test_compute_features_technical(sample_bars, cache_dir):
    """Test computing technical features."""
    store = FeatureStore(cache_dir=cache_dir)
    
    feature_config = {
        "technical": {
            "sma": [10, 20],
            "ema": [10],
            "rsi": [14],
            "bollinger": {"window": 20, "num_std": 2.0},
        }
    }
    
    features = store.compute_features(sample_bars, feature_config, symbol="TEST", timeframe="1D")
    
    assert len(features) == len(sample_bars)
    assert "sma_10" in features.columns
    assert "sma_20" in features.columns
    assert "ema_10" in features.columns
    assert "rsi_14" in features.columns
    assert "bb_upper" in features.columns


def test_compute_features_statistical(sample_bars, cache_dir):
    """Test computing statistical features."""
    store = FeatureStore(cache_dir=cache_dir)
    
    feature_config = {
        "statistical": {
            "returns": [1, 5],
            "log_returns": [1],
            "rolling_vol": [{"window": 20}],
            "zscore": [20],
        }
    }
    
    features = store.compute_features(sample_bars, feature_config, symbol="TEST", timeframe="1D")
    
    assert len(features) == len(sample_bars)
    assert "returns_1" in features.columns
    assert "returns_5" in features.columns
    assert "log_returns_1" in features.columns
    assert "rolling_vol_20" in features.columns
    assert "zscore_20" in features.columns


def test_feature_caching(sample_bars, cache_dir):
    """Test feature caching."""
    store = FeatureStore(cache_dir=cache_dir)
    
    feature_config = {
        "technical": {"sma": [10]},
    }
    
    # First computation
    features1 = store.compute_features(sample_bars, feature_config, symbol="TEST", timeframe="1D", use_cache=True)
    
    # Second computation (should use cache)
    features2 = store.compute_features(sample_bars, feature_config, symbol="TEST", timeframe="1D", use_cache=True)
    
    # Results should be identical
    pd.testing.assert_frame_equal(features1, features2)


def test_no_nans_beyond_warmup(sample_bars, cache_dir):
    """Test that NaNs only exist during warmup period."""
    store = FeatureStore(cache_dir=cache_dir)
    
    feature_config = {
        "technical": {
            "sma": [10, 20],
            "rsi": [14],
            "bollinger": {"window": 20},
        },
        "statistical": {
            "returns": [1],
            "rolling_vol": [{"window": 20}],
        },
    }
    
    features = store.compute_features(sample_bars, feature_config, symbol="TEST", timeframe="1D")
    
    # Find maximum warmup period (largest window)
    max_window = 20
    
    # After warmup, there should be no NaNs
    features_after_warmup = features.iloc[max_window:]
    
    # Check each column
    for col in features_after_warmup.columns:
        nan_count = features_after_warmup[col].isna().sum()
        assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"


def test_multi_index_features(cache_dir):
    """Test features with multi-index DataFrame."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1D")
    symbols = ["AAPL", "MSFT"]
    
    data = []
    for symbol in symbols:
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        for date, price in zip(dates, prices):
            data.append({
                "timestamp": date,
                "symbol": symbol,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000,
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(["timestamp", "symbol"])
    
    store = FeatureStore(cache_dir=cache_dir)
    feature_config = {"technical": {"sma": [10]}}
    
    # Should work with multi-index
    features = store.compute_features(df, feature_config, symbol="AAPL", timeframe="1D")
    
    assert len(features) > 0
    assert "sma_10" in features.columns
