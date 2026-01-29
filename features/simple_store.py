"""Simple feature store computing returns and SMA."""

from collections import deque
from decimal import Decimal
from typing import Optional

from core.interfaces import FeatureStore
from core.types import Bar


class SimpleFeatureStore(FeatureStore):
    """Simple feature store that computes returns and SMA."""

    def __init__(self, sma_period: int = 20) -> None:
        """Initialize feature store.

        Args:
            sma_period: Period for Simple Moving Average
        """
        self.sma_period = sma_period
        self._bar_history: dict[str, deque[Bar]] = {}
        self._latest_features: dict[str, dict[str, float]] = {}

    async def compute_features(
        self,
        symbol: str,
        bars: list[Bar],
    ) -> dict[str, float]:
        """Compute features from bars."""
        if symbol not in self._bar_history:
            self._bar_history[symbol] = deque(maxlen=self.sma_period * 2)

        # Add new bars to history
        for bar in bars:
            self._bar_history[symbol].append(bar)

        if len(self._bar_history[symbol]) < 2:
            # Not enough data
            return {}

        # Get latest bar
        latest_bar = self._bar_history[symbol][-1]
        previous_bar = self._bar_history[symbol][-2]

        # Compute returns
        returns = float(
            (latest_bar.close - previous_bar.close) / previous_bar.close
        )

        # Compute SMA
        sma = self._compute_sma(symbol)

        features = {
            "returns": returns,
            "sma": sma,
            "price": float(latest_bar.close),
            "volume": float(latest_bar.volume),
        }

        # Store latest features
        self._latest_features[symbol] = features

        return features

    def _compute_sma(self, symbol: str) -> float:
        """Compute Simple Moving Average."""
        bars = list(self._bar_history[symbol])
        if len(bars) < self.sma_period:
            # Use available bars
            closes = [float(bar.close) for bar in bars]
        else:
            # Use last N bars
            closes = [float(bar.close) for bar in bars[-self.sma_period:]]

        return sum(closes) / len(closes) if closes else 0.0

    async def get_latest_features(
        self,
        symbol: str,
    ) -> dict[str, float]:
        """Get latest computed features for a symbol."""
        return self._latest_features.get(symbol, {})
