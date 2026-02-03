"""Feature store for computing and caching time-series features."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from features.statistical import (
    autocorr,
    log_returns,
    returns,
    rolling_corr,
    rolling_vol,
    zscore,
)
from features.technical import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    vwap,
)


class FeatureStore:
    """Feature store with computation and caching."""

    def __init__(self, cache_dir: str = "data/features") -> None:
        """Initialize feature store.

        Args:
            cache_dir: Directory for caching features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, feature_config: dict[str, Any]) -> str:
        """Compute hash of feature configuration.

        Args:
            feature_config: Feature configuration dictionary

        Returns:
            Hash string
        """
        config_str = json.dumps(feature_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_cache_path(
        self,
        symbol: str,
        timeframe: str,
        feature_hash: str,
    ) -> Path:
        """Get cache file path.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            feature_hash: Feature configuration hash

        Returns:
            Cache file path
        """
        return self.cache_dir / f"{symbol}_{timeframe}_{feature_hash}.parquet"

    def _compute_technical_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Compute technical indicators.

        Args:
            df: DataFrame with OHLCV data
            config: Technical feature configuration

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=df.index)

        # Ensure we have required columns
        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        close = df["close"]

        # SMA
        if "sma" in config:
            for window in config["sma"]:
                features[f"sma_{window}"] = sma(close, window)

        # EMA
        if "ema" in config:
            for window in config["ema"]:
                features[f"ema_{window}"] = ema(close, window)

        # RSI
        if "rsi" in config:
            for window in config.get("rsi", [14]):
                features[f"rsi_{window}"] = rsi(close, window)

        # MACD
        if "macd" in config:
            macd_config = config["macd"]
            fast = macd_config.get("fast", 12)
            slow = macd_config.get("slow", 26)
            signal = macd_config.get("signal", 9)
            macd_df = macd(close, fast=fast, slow=slow, signal=signal)
            features = pd.concat([features, macd_df], axis=1)

        # Bollinger Bands
        if "bollinger" in config:
            bb_config = config["bollinger"]
            window = bb_config.get("window", 20)
            num_std = bb_config.get("num_std", 2.0)
            bb_df = bollinger_bands(close, window=window, num_std=num_std)
            features = pd.concat([features, bb_df], axis=1)

        # ATR
        if "atr" in config:
            if "high" not in df.columns or "low" not in df.columns:
                raise ValueError("ATR requires 'high' and 'low' columns")
            window = config["atr"].get("window", 14)
            features[f"atr_{window}"] = atr(df["high"], df["low"], close, window=window)

        # VWAP
        if "vwap" in config:
            if "high" not in df.columns or "low" not in df.columns or "volume" not in df.columns:
                raise ValueError("VWAP requires 'high', 'low', and 'volume' columns")
            vwap_config = config["vwap"]
            window = vwap_config.get("window", None)
            features["vwap"] = vwap(df["high"], df["low"], close, df["volume"], window=window)

        return features

    def _compute_statistical_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Compute statistical features.

        Args:
            df: DataFrame with price data
            config: Statistical feature configuration

        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=df.index)

        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        close = df["close"]

        # Returns
        if "returns" in config:
            for periods in config["returns"]:
                features[f"returns_{periods}"] = returns(close, periods=periods)

        # Log returns
        if "log_returns" in config:
            for periods in config["log_returns"]:
                features[f"log_returns_{periods}"] = log_returns(close, periods=periods)

        # Rolling volatility
        if "rolling_vol" in config:
            for vol_config in config["rolling_vol"]:
                window = vol_config.get("window", 20)
                annualize = vol_config.get("annualize", True)
                periods_per_year = vol_config.get("periods_per_year", 252)
                features[f"rolling_vol_{window}"] = rolling_vol(
                    close.pct_change(),
                    window=window,
                    annualize=annualize,
                    periods_per_year=periods_per_year,
                )

        # Z-score
        if "zscore" in config:
            for window in config["zscore"]:
                features[f"zscore_{window}"] = zscore(close, window=window)

        # Autocorrelation
        if "autocorr" in config:
            for ac_config in config["autocorr"]:
                lag = ac_config.get("lag", 1)
                window = ac_config.get("window", None)
                if window:
                    features[f"autocorr_lag{lag}_w{window}"] = autocorr(close, lag=lag, window=window)

        # Rolling correlation (with another series)
        if "rolling_corr" in config:
            for corr_config in config["rolling_corr"]:
                series2_name = corr_config.get("series2")
                window = corr_config.get("window", 20)
                if series2_name and series2_name in df.columns:
                    features[f"rolling_corr_{series2_name}_{window}"] = rolling_corr(
                        close, df[series2_name], window=window
                    )

        return features

    def compute_features(
        self,
        df_bars: pd.DataFrame,
        feature_config: dict[str, Any],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute features from bar data.

        Args:
            df_bars: DataFrame with OHLCV data (multi-index or single index)
            feature_config: Feature configuration dictionary
            symbol: Symbol (for caching, extracted from index if multi-index)
            timeframe: Timeframe (for caching)
            use_cache: Whether to use cache

        Returns:
            DataFrame with computed features
        """
        # Extract symbol and timeframe if not provided
        if symbol is None or timeframe is None:
            if isinstance(df_bars.index, pd.MultiIndex):
                if symbol is None:
                    symbols = df_bars.index.get_level_values(1).unique()
                    if len(symbols) == 1:
                        symbol = str(symbols[0])
                    else:
                        raise ValueError("Multiple symbols found, specify symbol parameter")
                if timeframe is None:
                    timeframe = "unknown"
            else:
                if symbol is None:
                    symbol = "unknown"
                if timeframe is None:
                    timeframe = "unknown"

        # Check cache
        feature_hash = self._compute_hash(feature_config)
        cache_path = self._get_cache_path(symbol, timeframe, feature_hash)

        if use_cache and cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                # Check if cached data covers the requested range
                if isinstance(df_bars.index, pd.MultiIndex):
                    timestamps = df_bars.index.get_level_values(0).unique()
                else:
                    timestamps = df_bars.index
                
                if cached_df.index.min() <= timestamps.min() and cached_df.index.max() >= timestamps.max():
                    # Return cached features for the requested range
                    return cached_df.loc[timestamps]
            except Exception:
                # Cache read failed, recompute
                pass

        # Compute features
        all_features = []

        # Technical features
        if "technical" in feature_config:
            tech_features = self._compute_technical_features(df_bars, feature_config["technical"])
            all_features.append(tech_features)

        # Statistical features
        if "statistical" in feature_config:
            stat_features = self._compute_statistical_features(df_bars, feature_config["statistical"])
            all_features.append(stat_features)

        # Combine all features
        if all_features:
            df_features = pd.concat(all_features, axis=1)
        else:
            df_features = pd.DataFrame(index=df_bars.index)

        # Remove any columns that are all NaN
        df_features = df_features.dropna(axis=1, how="all")

        # Cache results
        if use_cache:
            try:
                df_features.to_parquet(cache_path, compression="snappy")
            except Exception:
                # Cache write failed, continue without caching
                pass

        return df_features
