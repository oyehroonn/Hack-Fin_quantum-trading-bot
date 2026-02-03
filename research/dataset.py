"""Dataset builder for time-series trading data."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Fallback implementation if sklearn not available
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
            return self
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_


@dataclass
class TimeSeriesSplit:
    """Time-series split configuration."""

    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    val_start: Optional[pd.Timestamp] = None
    val_end: Optional[pd.Timestamp] = None
    test_start: Optional[pd.Timestamp] = None
    test_end: Optional[pd.Timestamp] = None
    purge_gap: int = 0  # Gap between splits to prevent leakage


class DatasetBuilder:
    """Dataset builder for time-series trading."""

    def __init__(
        self,
        lookback: int,
        horizon: int = 1,
        label_type: Literal["returns", "direction", "volatility"] = "returns",
    ) -> None:
        """Initialize dataset builder.

        Args:
            lookback: Number of time steps to look back
            horizon: Number of time steps ahead to predict
            label_type: Type of label to generate
        """
        self.lookback = lookback
        self.horizon = horizon
        self.label_type = label_type
        self.scaler: Optional[StandardScaler] = None

    def build_windows(
        self,
        df_features: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sliding windows from features.

        Args:
            df_features: DataFrame with features
            target_col: Target column name (if None, uses 'close' or first numeric column)

        Returns:
            Tuple of (X, y, timestamps) where:
            - X: Feature windows (n_samples, lookback, n_features)
            - y: Target values (n_samples,)
            - timestamps: Timestamps for each sample
        """
        # Determine target column
        if target_col is None:
            if "close" in df_features.columns:
                target_col = "close"
            else:
                # Use first numeric column
                numeric_cols = df_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found in features")
                target_col = numeric_cols[0]

        if target_col not in df_features.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")

        # Extract features and target
        feature_cols = [col for col in df_features.columns if col != target_col]
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found")

        X_data = df_features[feature_cols].values
        y_data = df_features[target_col].values

        # Generate labels based on label_type
        if self.label_type == "returns":
            # Forward returns
            y_labels = (y_data[self.horizon:] / y_data[:-self.horizon]) - 1.0
            y_labels = np.concatenate([y_labels, np.full(self.horizon, np.nan)])
        elif self.label_type == "direction":
            # Direction (1 for up, 0 for down)
            y_labels = (y_data[self.horizon:] > y_data[:-self.horizon]).astype(float)
            y_labels = np.concatenate([y_labels, np.full(self.horizon, np.nan)])
        elif self.label_type == "volatility":
            # Forward volatility (rolling std of returns)
            returns = np.diff(y_data) / y_data[:-1]
            y_labels = pd.Series(returns).rolling(window=self.horizon).std().values
            y_labels = np.concatenate([np.nan, y_labels])
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        # Build windows
        X_windows = []
        y_windows = []
        timestamps = []

        for i in range(self.lookback, len(X_data) - self.horizon + 1):
            # Feature window
            X_window = X_data[i - self.lookback : i]
            X_windows.append(X_window)

            # Target (at horizon)
            y_val = y_labels[i + self.horizon - 1]
            y_windows.append(y_val)

            # Timestamp
            if isinstance(df_features.index, pd.MultiIndex):
                timestamp = df_features.index[i][0]
            else:
                timestamp = df_features.index[i]
            timestamps.append(timestamp)

        X = np.array(X_windows)
        y = np.array(y_windows)

        # Remove NaN labels
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        timestamps = [ts for ts, valid in zip(timestamps, valid_mask) if valid]

        return X, y, np.array(timestamps)

    def time_series_split(
        self,
        timestamps: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        purge_gap: int = 0,
    ) -> TimeSeriesSplit:
        """Create time-series split with purge gap.

        Args:
            timestamps: Array of timestamps
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            purge_gap: Gap between splits (in time steps)
            test_ratio: Ratio of data for testing (computed from remaining)

        Returns:
            TimeSeriesSplit configuration
        """
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]

        n_total = len(sorted_timestamps)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # Adjust for purge gaps
        train_end_idx = n_train
        val_start_idx = train_end_idx + purge_gap
        val_end_idx = val_start_idx + n_val
        test_start_idx = val_end_idx + purge_gap

        train_start = sorted_timestamps[0]
        train_end = sorted_timestamps[train_end_idx - 1] if train_end_idx > 0 else sorted_timestamps[0]
        val_start = sorted_timestamps[val_start_idx] if val_start_idx < n_total else sorted_timestamps[-1]
        val_end = sorted_timestamps[val_end_idx - 1] if val_end_idx <= n_total else sorted_timestamps[-1]
        test_start = sorted_timestamps[test_start_idx] if test_start_idx < n_total else sorted_timestamps[-1]
        test_end = sorted_timestamps[-1]

        return TimeSeriesSplit(
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            val_start=pd.Timestamp(val_start),
            val_end=pd.Timestamp(val_end),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
            purge_gap=purge_gap,
        )

    def apply_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        split: TimeSeriesSplit,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply time-series split to data.

        Args:
            X: Feature windows
            y: Target values
            timestamps: Timestamps for each sample
            split: Time-series split configuration

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        timestamps_pd = pd.to_datetime(timestamps)

        # Train split
        train_mask = (timestamps_pd >= split.train_start) & (timestamps_pd <= split.train_end)
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Validation split
        val_mask = (timestamps_pd >= split.val_start) & (timestamps_pd <= split.val_end)
        X_val = X[val_mask]
        y_val = y[val_mask]

        # Test split
        test_mask = (timestamps_pd >= split.test_start) & (timestamps_pd <= split.test_end)
        X_test = X[test_mask]
        y_test = y[test_mask]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit_normalizer(self, X_train: np.ndarray) -> None:
        """Fit normalizer on training data only.

        Args:
            X_train: Training feature windows (n_samples, lookback, n_features)
        """
        # Reshape to (n_samples * lookback, n_features) for fitting
        n_samples, lookback, n_features = X_train.shape
        X_flat = X_train.reshape(-1, n_features)

        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_flat)

    def normalize(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Normalize feature windows.

        Args:
            X: Feature windows (n_samples, lookback, n_features)

        Returns:
            Normalized feature windows
        """
        if self.scaler is None:
            raise ValueError("Normalizer not fitted. Call fit_normalizer first.")

        # Reshape for normalization
        n_samples, lookback, n_features = X.shape
        X_flat = X.reshape(-1, n_features)

        # Normalize
        X_norm_flat = self.scaler.transform(X_flat)

        # Reshape back
        X_norm = X_norm_flat.reshape(n_samples, lookback, n_features)

        return X_norm

    def build_dataset(
        self,
        df_features: pd.DataFrame,
        target_col: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        purge_gap: int = 0,
        normalize: bool = True,
    ) -> dict[str, np.ndarray]:
        """Build complete dataset with splits and normalization.

        Args:
            df_features: DataFrame with features
            target_col: Target column name
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            purge_gap: Gap between splits
            normalize: Whether to normalize features

        Returns:
            Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test, split
        """
        # Build windows
        X, y, timestamps = self.build_windows(df_features, target_col=target_col)

        # Create split
        split = self.time_series_split(timestamps, train_ratio=train_ratio, val_ratio=val_ratio, purge_gap=purge_gap)

        # Apply split
        X_train, y_train, X_val, y_val, X_test, y_test = self.apply_split(X, y, timestamps, split)

        # Normalize (fit on train only)
        if normalize:
            self.fit_normalizer(X_train)
            X_train = self.normalize(X_train)
            X_val = self.normalize(X_val)
            X_test = self.normalize(X_test)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "split": split,
        }
