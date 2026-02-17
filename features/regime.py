"""Market regime detection features.

Classifies markets into trending, mean-reverting, high/low volatility
using statistical tests and rolling indicators.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime
from typing import Optional

from core.interfaces import RegimeDetector
from core.types import Regime, RegimeState


class StatisticalRegimeDetector(RegimeDetector):
    """Detect market regime using statistical indicators.

    Combines:
      - Trend strength via ADX proxy (DM+/DM-/ATR ratio)
      - Volatility regime via realised vol percentile
      - Mean-reversion via variance ratio / Hurst exponent proxy
      - Chop index proxy for range-bound detection
    """

    def __init__(
        self,
        trend_lookback: int = 50,
        vol_lookback: int = 20,
        vol_percentile_low: float = 25.0,
        vol_percentile_high: float = 75.0,
        trend_threshold: float = 0.15,
        hurst_mr_threshold: float = 0.40,
    ) -> None:
        self.trend_lookback = trend_lookback
        self.vol_lookback = vol_lookback
        self.vol_percentile_low = vol_percentile_low
        self.vol_percentile_high = vol_percentile_high
        self.trend_threshold = trend_threshold
        self.hurst_mr_threshold = hurst_mr_threshold

    def detect(
        self,
        prices: list[float],
        volumes: Optional[list[float]] = None,
        timestamp: Optional[datetime] = None,
        symbol: str = "",
    ) -> RegimeState:
        arr = np.array(prices)
        n = len(arr)
        ts = timestamp or datetime.now()

        if n < self.trend_lookback:
            return RegimeState(
                regime=Regime.UNKNOWN, confidence=Decimal("0.1"),
                timestamp=ts, symbol=symbol,
            )

        recent = arr[-self.trend_lookback:]
        indicators = {}

        # 1. Trend strength
        trend_strength, slope = self._trend_strength(recent)
        indicators["trend_strength"] = round(float(trend_strength), 4)
        indicators["slope"] = round(float(slope), 6)

        # 2. Volatility
        vol_window = arr[-self.vol_lookback:]
        rets = np.diff(vol_window) / vol_window[:-1]
        realised_vol = float(np.std(rets) * np.sqrt(252))
        indicators["realised_vol"] = round(realised_vol, 4)

        # Historical vol for percentile ranking
        if n >= self.trend_lookback + self.vol_lookback:
            hist_vols = []
            for i in range(self.vol_lookback, min(n, 500)):
                w = arr[max(0, i - self.vol_lookback):i]
                r = np.diff(w) / w[:-1]
                hist_vols.append(float(np.std(r) * np.sqrt(252)))
            vol_pctile = float(np.searchsorted(sorted(hist_vols), realised_vol) / max(len(hist_vols), 1) * 100)
        else:
            vol_pctile = 50.0
        indicators["vol_percentile"] = round(vol_pctile, 1)

        # 3. Hurst exponent proxy
        hurst = self._variance_ratio_hurst(rets)
        indicators["hurst_proxy"] = round(float(hurst), 4)

        # 4. Chop index proxy
        chop = self._chop_index(recent)
        indicators["chop_index"] = round(float(chop), 4)

        # Classification
        regime, confidence = self._classify(
            trend_strength, slope, realised_vol, vol_pctile, hurst, chop
        )

        return RegimeState(
            regime=regime,
            confidence=Decimal(str(round(confidence, 4))),
            timestamp=ts,
            symbol=symbol,
            indicators=indicators,
        )

    def _classify(
        self,
        trend_strength: float,
        slope: float,
        vol: float,
        vol_pctile: float,
        hurst: float,
        chop: float,
    ) -> tuple[Regime, float]:
        """Classify regime from indicators."""

        # Strong trend overrides everything
        if trend_strength > self.trend_threshold:
            regime = Regime.TRENDING_UP if slope > 0 else Regime.TRENDING_DOWN
            conf = min(0.5 + trend_strength, 0.95)
            return regime, conf

        # Mean-reversion (low Hurst + high chop)
        if hurst < self.hurst_mr_threshold and chop > 0.6:
            return Regime.MEAN_REVERTING, min(0.5 + (0.5 - hurst), 0.90)

        # Volatility extremes
        if vol_pctile > self.vol_percentile_high:
            return Regime.HIGH_VOLATILITY, min(0.5 + (vol_pctile - 75) / 50, 0.90)
        if vol_pctile < self.vol_percentile_low:
            return Regime.LOW_VOLATILITY, min(0.5 + (25 - vol_pctile) / 50, 0.85)

        return Regime.UNKNOWN, 0.3

    @staticmethod
    def _trend_strength(prices: np.ndarray) -> tuple[float, float]:
        """Trend strength via linear regression R² and slope."""
        log_prices = np.log(prices)
        x = np.arange(len(log_prices))
        slope, intercept = np.polyfit(x, log_prices, 1)

        # R² as trend quality
        predicted = slope * x + intercept
        ss_res = np.sum((log_prices - predicted) ** 2)
        ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
        r_squared = 1 - (ss_res / max(ss_tot, 1e-10))

        # Combine slope magnitude and R² for trend strength
        strength = abs(slope) * len(prices) * r_squared
        return float(strength), float(slope)

    @staticmethod
    def _variance_ratio_hurst(returns: np.ndarray, lag: int = 5) -> float:
        """Variance ratio → Hurst proxy. H<0.5 = MR, H>0.5 = trending."""
        if len(returns) < lag * 2:
            return 0.5

        var_1 = np.var(returns)
        if var_1 == 0:
            return 0.5

        lagged = np.array([sum(returns[i:i + lag]) for i in range(0, len(returns) - lag + 1, lag)])
        var_lag = np.var(lagged)

        vr = var_lag / (lag * var_1)
        hurst = 0.5 + (vr - 1.0) * 0.25
        return float(np.clip(hurst, 0.0, 1.0))

    @staticmethod
    def _chop_index(prices: np.ndarray, period: int = 14) -> float:
        """Chop index proxy: high = choppy/ranging, low = trending."""
        if len(prices) < period + 1:
            return 0.5

        recent = prices[-period - 1:]
        atr_sum = 0.0
        for i in range(1, len(recent)):
            tr = max(
                recent[i] - recent[i - 1],
                abs(recent[i] - recent[i - 1]),
                abs(recent[i - 1] - recent[i]),
            )
            atr_sum += tr

        price_range = max(recent) - min(recent)
        if price_range == 0:
            return 1.0

        chop = np.log10(atr_sum / price_range) / np.log10(period)
        return float(np.clip(chop, 0.0, 1.0))


def compute_regime_features(
    df: pd.DataFrame,
    close_col: str = "close",
    vol_lookback: int = 20,
    trend_lookback: int = 50,
) -> pd.DataFrame:
    """Compute regime indicator features for a DataFrame.

    Returns DataFrame with: trend_strength, realised_vol, hurst_proxy, chop_index
    """
    close = df[close_col].values
    n = len(close)

    features = pd.DataFrame(index=df.index)
    features["trend_strength"] = np.nan
    features["realised_vol"] = np.nan
    features["hurst_proxy"] = np.nan
    features["chop_index"] = np.nan

    detector = StatisticalRegimeDetector(
        trend_lookback=trend_lookback, vol_lookback=vol_lookback,
    )

    for i in range(trend_lookback, n):
        prices = close[max(0, i - trend_lookback):i + 1].tolist()
        state = detector.detect(prices)
        features.iloc[i, features.columns.get_loc("trend_strength")] = state.indicators.get("trend_strength", 0)
        features.iloc[i, features.columns.get_loc("realised_vol")] = state.indicators.get("realised_vol", 0)
        features.iloc[i, features.columns.get_loc("hurst_proxy")] = state.indicators.get("hurst_proxy", 0.5)
        features.iloc[i, features.columns.get_loc("chop_index")] = state.indicators.get("chop_index", 0.5)

    return features
