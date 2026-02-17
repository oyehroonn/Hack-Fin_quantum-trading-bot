"""SMA Crossover model wrapped as a TradingModel for the model zoo.

This re-implements the SMA crossover logic from backtest/strategies/sma_crossover.py
as a TradingModel that can be registered, evaluated, and composed.
"""

import numpy as np
from typing import Any, Optional

from models.base import TradingModel


class SMACrossoverModel(TradingModel):
    """SMA crossover model: bullish when fast SMA > slow SMA."""

    model_type = "baseline_sma_crossover"

    def __init__(
        self,
        model_id: str = "sma_crossover",
        version: str = "1.0.0",
        fast_period: int = 10,
        slow_period: int = 30,
        **params: Any,
    ) -> None:
        super().__init__(model_id, version, **params)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.is_fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> "SMACrossoverModel":
        if feature_names:
            self._feature_names = feature_names
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability based on SMA crossover state.

        Expects features: [returns_1, returns_5, returns_10, returns_20, sma_ratio, vol_20, rsi]
        Uses sma_ratio (fast_sma / slow_sma proxy) as the core signal.

        Returns:
            P(up) — above 0.5 when fast > slow (bullish crossover).
        """
        n_samples = X.shape[0]
        probas = np.full(n_samples, 0.5)

        for i in range(n_samples):
            row = X[i]

            sma_ratio = row[4] if X.shape[1] > 4 else 1.0
            vol_20 = row[5] if X.shape[1] > 5 else 0.01

            # Core signal: fast SMA vs slow SMA (encoded as sma_ratio)
            crossover_signal = sma_ratio - 1.0

            # Scale by inverse of volatility (more confident in low-vol)
            if vol_20 > 0:
                scaled_signal = crossover_signal / max(vol_20, 0.005)
            else:
                scaled_signal = crossover_signal * 10

            # Convert to probability
            probas[i] = 0.5 + np.tanh(scaled_signal * 0.5) * 0.35

        return np.clip(probas, 0.01, 0.99)
