"""Mean-reversion model: buys when price is below lower band, sells above upper.

Uses Bollinger Bands z-score as the primary signal.
Implements TradingModel interface for registry + backtest compatibility.
"""

import numpy as np
from typing import Any, Optional

from models.base import TradingModel


class MeanReversionModel(TradingModel):
    """Mean-reversion model using z-score of price relative to rolling mean."""

    model_type = "baseline_mean_reversion"

    def __init__(
        self,
        model_id: str = "mean_reversion",
        version: str = "1.0.0",
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        **params: Any,
    ) -> None:
        super().__init__(model_id, version, **params)
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.is_fitted = True  # Rule-based, no training needed

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> "MeanReversionModel":
        """No-op for rule-based model. Stores feature names."""
        if feature_names:
            self._feature_names = feature_names
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of upward move using mean-reversion logic.

        Expects X to have columns: [returns_1, returns_5, returns_10, returns_20, sma_ratio, vol_20, rsi]
        Or at minimum: X[:, 4] = sma_ratio (price / SMA)

        Returns:
            P(up) array — high when price is below mean (reversion expected up),
            low when price is above mean (reversion expected down).
        """
        n_samples = X.shape[0]
        probas = np.full(n_samples, 0.5)

        for i in range(n_samples):
            row = X[i]

            # Use sma_ratio if available (col 4), else use returns
            if X.shape[1] > 4:
                sma_ratio = row[4]
                deviation = sma_ratio - 1.0  # How far price is from SMA
            else:
                deviation = row[0] if X.shape[1] > 0 else 0.0

            # Use RSI if available (col 6) for additional confirmation
            rsi = row[6] if X.shape[1] > 6 else 50.0

            # Z-score-like transformation
            zscore = deviation * 100  # Scale

            # Mean reversion: oversold → expect up, overbought → expect down
            if zscore < -self.entry_zscore:
                # Price well below mean → expect reversion up
                probas[i] = min(0.5 + abs(zscore) * 0.05, 0.85)
            elif zscore > self.entry_zscore:
                # Price well above mean → expect reversion down
                probas[i] = max(0.5 - abs(zscore) * 0.05, 0.15)
            elif abs(zscore) < self.exit_zscore:
                # Near mean → neutral
                probas[i] = 0.5
            else:
                # Mild deviation → slight reversion bias
                probas[i] = 0.5 - zscore * 0.02

            # RSI confirmation (oversold < 30 → bullish, overbought > 70 → bearish)
            if rsi < 30:
                probas[i] = min(probas[i] + 0.1, 0.90)
            elif rsi > 70:
                probas[i] = max(probas[i] - 0.1, 0.10)

        return np.clip(probas, 0.01, 0.99)
