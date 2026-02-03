"""Tests for dataset builder."""

import numpy as np
import pandas as pd
import pytest

from research.dataset import DatasetBuilder, TimeSeriesSplit


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1D")
    np.random.seed(42)
    
    return pd.DataFrame(
        {
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.randn(200),
            "close": 100 + np.cumsum(np.random.randn(200) * 0.1),
        },
        index=dates,
    )


def test_build_windows(sample_features):
    """Test building windows."""
    builder = DatasetBuilder(lookback=10, horizon=1, label_type="returns")
    
    X, y, timestamps = builder.build_windows(sample_features, target_col="close")
    
    assert X.shape[0] == len(y) == len(timestamps)
    assert X.shape[1] == 10  # lookback
    assert X.shape[2] == 3  # n_features (excluding target)
    assert len(y) > 0
    assert not np.isnan(y).any()  # No NaN labels


def test_label_types(sample_features):
    """Test different label types."""
    # Returns
    builder = DatasetBuilder(lookback=10, horizon=1, label_type="returns")
    X, y, _ = builder.build_windows(sample_features, target_col="close")
    assert len(y) > 0
    assert not np.isnan(y).any()
    
    # Direction
    builder = DatasetBuilder(lookback=10, horizon=1, label_type="direction")
    X, y, _ = builder.build_windows(sample_features, target_col="close")
    assert len(y) > 0
    assert not np.isnan(y).any()
    assert np.all((y == 0) | (y == 1))  # Binary


def test_time_series_split():
    """Test time-series split creation."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    timestamps = dates.values
    
    builder = DatasetBuilder(lookback=10, horizon=1)
    split = builder.time_series_split(timestamps, train_ratio=0.7, val_ratio=0.15, purge_gap=5)
    
    assert split.train_start <= split.train_end
    assert split.train_end < split.val_start  # Purge gap
    assert split.val_start <= split.val_end
    assert split.val_end < split.test_start  # Purge gap
    assert split.test_start <= split.test_end


def test_apply_split(sample_features):
    """Test applying split to data."""
    builder = DatasetBuilder(lookback=10, horizon=1)
    
    X, y, timestamps = builder.build_windows(sample_features, target_col="close")
    
    split = builder.time_series_split(timestamps, train_ratio=0.7, val_ratio=0.15, purge_gap=5)
    
    X_train, y_train, X_val, y_val, X_test, y_test = builder.apply_split(X, y, timestamps, split)
    
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_val) + len(X_test) <= len(X)  # Some may be excluded by purge gap


def test_time_ordering_respected(sample_features):
    """Test that train/val/test splits respect time ordering."""
    builder = DatasetBuilder(lookback=10, horizon=1)
    
    X, y, timestamps = builder.build_windows(sample_features, target_col="close")
    
    split = builder.time_series_split(timestamps, train_ratio=0.7, val_ratio=0.15, purge_gap=5)
    X_train, y_train, X_val, y_val, X_test, y_test = builder.apply_split(X, y, timestamps, split)
    
    # Get timestamps for each split
    train_timestamps = timestamps[builder.apply_split(X, y, timestamps, split)[0].shape[0]:]
    # Actually, let's check the split boundaries directly
    timestamps_pd = pd.to_datetime(timestamps)
    
    train_mask = (timestamps_pd >= split.train_start) & (timestamps_pd <= split.train_end)
    val_mask = (timestamps_pd >= split.val_start) & (timestamps_pd <= split.val_end)
    test_mask = (timestamps_pd >= split.test_start) & (timestamps_pd <= split.test_end)
    
    train_times = timestamps_pd[train_mask]
    val_times = timestamps_pd[val_mask]
    test_times = timestamps_pd[test_mask]
    
    # All train times should be before all val times
    if len(train_times) > 0 and len(val_times) > 0:
        assert train_times.max() < val_times.min()
    
    # All val times should be before all test times
    if len(val_times) > 0 and len(test_times) > 0:
        assert val_times.max() < test_times.min()


def test_purge_gap(sample_features):
    """Test that purge gap prevents leakage."""
    builder = DatasetBuilder(lookback=10, horizon=1)
    
    X, y, timestamps = builder.build_windows(sample_features, target_col="close")
    
    # Split with purge gap
    split_with_gap = builder.time_series_split(timestamps, train_ratio=0.7, val_ratio=0.15, purge_gap=10)
    
    # Split without purge gap
    split_no_gap = builder.time_series_split(timestamps, train_ratio=0.7, val_ratio=0.15, purge_gap=0)
    
    # With gap, there should be more separation
    gap_size_with = (split_with_gap.val_start - split_with_gap.train_end).days
    gap_size_without = (split_no_gap.val_start - split_no_gap.train_end).days
    
    assert gap_size_with >= gap_size_without


def test_normalization_fit_on_train_only(sample_features):
    """Test that normalization is fitted on train only."""
    builder = DatasetBuilder(lookback=10, horizon=1)
    
    dataset = builder.build_dataset(
        sample_features,
        target_col="close",
        train_ratio=0.7,
        val_ratio=0.15,
        purge_gap=5,
        normalize=True,
    )
    
    X_train = dataset["X_train"]
    X_val = dataset["X_val"]
    X_test = dataset["X_test"]
    
    # Check that scaler was fitted
    assert builder.scaler is not None
    
    # Train data should be approximately normalized (mean ~0, std ~1)
    # Reshape to check
    n_samples, lookback, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    
    # Mean should be close to 0 (within tolerance)
    means = X_train_flat.mean(axis=0)
    assert np.allclose(means, 0, atol=0.1)
    
    # Std should be close to 1
    stds = X_train_flat.std(axis=0)
    assert np.allclose(stds, 1, atol=0.1)


def test_no_nans_in_dataset(sample_features):
    """Test that dataset has no NaNs."""
    builder = DatasetBuilder(lookback=10, horizon=1)
    
    dataset = builder.build_dataset(
        sample_features,
        target_col="close",
        normalize=True,
    )
    
    # Check for NaNs
    assert not np.isnan(dataset["X_train"]).any()
    assert not np.isnan(dataset["y_train"]).any()
    assert not np.isnan(dataset["X_val"]).any()
    assert not np.isnan(dataset["y_val"]).any()
    assert not np.isnan(dataset["X_test"]).any()
    assert not np.isnan(dataset["y_test"]).any()


def test_multi_index_features():
    """Test dataset building with multi-index features."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    symbols = ["AAPL"]
    
    data = []
    np.random.seed(42)
    for symbol in symbols:
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        for date, price in zip(dates, prices):
            data.append({
                "timestamp": date,
                "symbol": symbol,
                "feature1": np.random.randn(),
                "feature2": np.random.randn(),
                "close": price,
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(["timestamp", "symbol"])
    
    builder = DatasetBuilder(lookback=10, horizon=1)
    dataset = builder.build_dataset(df, target_col="close", normalize=False)
    
    assert len(dataset["X_train"]) > 0
    assert len(dataset["y_train"]) > 0
