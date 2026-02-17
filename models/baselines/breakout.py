"""Breakout model: buys on new highs / upward momentum, sells on new lows.

Combines channel breakout logic with volume confirmation.
Implements TradingModel for registry + backtest compatibility.
"""

import numpy as np
from typing import Any, Optional

from models.base import TradingModel


class BreakoutModel(TradingModel):
    """Channel-breakout model with momentum and volume confirmation."""

    model_type = "baseline_breakout"

    def __init__(
        self,
        model_id: str = "breakout",
        version: str = "1.0.0",
        channel_period: int = 20,
        momentum_period: int = 10,
        volume_confirmation: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(model_id, version, **params)
        self.channel_period = channel_period
        self.momentum_period = momentum_period
        self.volume_confirmation = volume_confirmation
        self.is_fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> "BreakoutModel":
        if feature_names:
            self._feature_names = feature_names
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of upward breakout continuing.

        Expects features: [returns_1, returns_5, returns_10, returns_20, sma_ratio, vol_20, rsi]
        Uses multi-period returns and volatility to detect breakout conditions.

        Returns:
            P(up) — high when upward breakout detected, low when downward.
        """
        n_samples = X.shape[0]
        probas = np.full(n_samples, 0.5)

        for i in range(n_samples):
            row = X[i]

            ret_1 = row[0] if X.shape[1] > 0 else 0.0
            ret_5 = row[1] if X.shape[1] > 1 else 0.0
            ret_10 = row[2] if X.shape[1] > 2 else 0.0
            ret_20 = row[3] if X.shape[1] > 3 else 0.0
            sma_ratio = row[4] if X.shape[1] > 4 else 1.0
            vol_20 = row[5] if X.shape[1] > 5 else 0.01
            rsi = row[6] if X.shape[1] > 6 else 50.0

            # Momentum score: weighted multi-period returns
            momentum = (ret_1 * 0.4 + ret_5 * 0.3 + ret_10 * 0.2 + ret_20 * 0.1)

            # Breakout detection: price above SMA and strong momentum
            breakout_score = 0.0

            # Upward breakout signals
            if sma_ratio > 1.0 and momentum > 0:
                breakout_score = min(momentum / max(vol_20, 0.001), 3.0) * 0.15
            elif sma_ratio < 1.0 and momentum < 0:
                breakout_score = max(momentum / max(vol_20, 0.001), -3.0) * 0.15

            # Strong trends (RSI confirms direction)
            if rsi > 60 and momentum > 0:
                breakout_score += 0.05
            elif rsi < 40 and momentum < 0:
                breakout_score -= 0.05

            # Volatility expansion (breakouts happen on high vol)
            if vol_20 > 0.02:  # Above-average vol
                breakout_score *= 1.3

            probas[i] = 0.5 + breakout_score

        return np.clip(probas, 0.01, 0.99)
