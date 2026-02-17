"""Online (streaming) feature store for live trading.

Maintains rolling buffers per symbol and incrementally computes
features on each new bar, avoiding full recomputation.
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from core.types import FeatureVector


class OnlineFeatureStore:
    """Streaming feature computation with rolling buffers.

    Designed for the live execution pipeline:
      bar arrives → update buffers → compute features → return FeatureVector
    """

    def __init__(
        self,
        max_lookback: int = 200,
        sma_periods: Optional[list[int]] = None,
        ema_periods: Optional[list[int]] = None,
        rsi_period: int = 14,
        vol_period: int = 20,
        regime_lookback: int = 50,
    ) -> None:
        self.max_lookback = max_lookback
        self.sma_periods = sma_periods or [10, 20, 50]
        self.ema_periods = ema_periods or [12, 26]
        self.rsi_period = rsi_period
        self.vol_period = vol_period
        self.regime_lookback = regime_lookback

        # Per-symbol buffers
        self._prices: dict[str, deque] = {}
        self._volumes: dict[str, deque] = {}
        self._ema_state: dict[str, dict[int, float]] = {}

    def update(
        self,
        symbol: str,
        close: float,
        volume: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        timestamp: Optional[datetime] = None,
        timeframe: str = "1m",
    ) -> Optional[FeatureVector]:
        """Process a new bar and return updated features.

        Args:
            symbol: Symbol
            close: Close price
            volume: Volume
            high: High price
            low: Low price
            timestamp: Bar timestamp
            timeframe: Bar timeframe

        Returns:
            FeatureVector or None if insufficient data
        """
        ts = timestamp or datetime.now()

        # Initialize buffers
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.max_lookback)
            self._volumes[symbol] = deque(maxlen=self.max_lookback)
            self._ema_state[symbol] = {}

        self._prices[symbol].append(close)
        self._volumes[symbol].append(volume)

        prices = np.array(self._prices[symbol])
        n = len(prices)

        if n < max(self.sma_periods + self.ema_periods + [self.rsi_period, self.vol_period]):
            return None

        features: dict[str, float] = {}

        # Returns
        features["returns_1"] = (prices[-1] / prices[-2] - 1) if n >= 2 else 0
        features["returns_5"] = (prices[-1] / prices[-6] - 1) if n >= 6 else 0
        features["returns_10"] = (prices[-1] / prices[-11] - 1) if n >= 11 else 0
        features["returns_20"] = (prices[-1] / prices[-21] - 1) if n >= 21 else 0

        # SMAs
        for period in self.sma_periods:
            if n >= period:
                features[f"sma_{period}"] = float(np.mean(prices[-period:]))
                features[f"sma_ratio_{period}"] = prices[-1] / features[f"sma_{period}"]

        # EMAs (incremental update)
        for period in self.ema_periods:
            alpha = 2.0 / (period + 1)
            if period not in self._ema_state[symbol]:
                if n >= period:
                    self._ema_state[symbol][period] = float(np.mean(prices[-period:]))
            else:
                prev = self._ema_state[symbol][period]
                self._ema_state[symbol][period] = alpha * close + (1 - alpha) * prev

            if period in self._ema_state[symbol]:
                features[f"ema_{period}"] = self._ema_state[symbol][period]

        # MACD
        if 12 in self._ema_state.get(symbol, {}) and 26 in self._ema_state.get(symbol, {}):
            features["macd"] = self._ema_state[symbol][12] - self._ema_state[symbol][26]

        # RSI
        if n >= self.rsi_period + 1:
            features["rsi"] = self._compute_rsi(prices, self.rsi_period)

        # Volatility
        if n >= self.vol_period + 1:
            rets = np.diff(prices[-self.vol_period - 1:]) / prices[-self.vol_period - 1:-1]
            features["vol_20"] = float(np.std(rets) * np.sqrt(252))
            features["vol_raw"] = float(np.std(rets))

        # Volume features
        vols = np.array(self._volumes[symbol])
        if len(vols) >= 20:
            features["volume_ratio"] = float(vols[-1] / max(np.mean(vols[-20:]), 1))
            features["volume_ma_20"] = float(np.mean(vols[-20:]))

        # Trend strength (simple: R² of recent log-prices)
        if n >= self.regime_lookback:
            recent = prices[-self.regime_lookback:]
            log_p = np.log(recent)
            x = np.arange(len(log_p))
            slope, intercept = np.polyfit(x, log_p, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((log_p - predicted) ** 2)
            ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            features["trend_strength"] = float(abs(slope) * len(recent) * r2)
            features["trend_direction"] = 1.0 if slope > 0 else -1.0

        return FeatureVector(
            symbol=symbol,
            timestamp=ts,
            timeframe=timeframe,
            features=features,
        )

    def get_latest(self, symbol: str) -> Optional[dict[str, float]]:
        """Get the most recently computed features for a symbol."""
        if symbol not in self._prices or len(self._prices[symbol]) < 2:
            return None
        # Recompute from buffer (could cache but buffers are small)
        fv = self.update(
            symbol,
            close=self._prices[symbol][-1],
            volume=self._volumes[symbol][-1] if self._volumes[symbol] else 0,
        )
        return fv.features if fv else None

    def get_price_history(self, symbol: str) -> list[float]:
        """Get price buffer for a symbol."""
        return list(self._prices.get(symbol, []))

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset buffers for a symbol or all symbols."""
        if symbol:
            self._prices.pop(symbol, None)
            self._volumes.pop(symbol, None)
            self._ema_state.pop(symbol, None)
        else:
            self._prices.clear()
            self._volumes.clear()
            self._ema_state.clear()

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
