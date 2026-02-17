"""Base model classes and the Model→Strategy adapter.

This module bridges the gap between:
  - core.interfaces.Model  (predict features → Signal)
  - backtest.strategy.Strategy (on_bar → weights/orders)

ModelStrategy wraps any Model so the backtest engine can run it.
"""

from abc import abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.strategy import WeightBasedStrategy
from core.types import ModelDecision, OrderSide, Signal


class TradingModel:
    """Base class for all trading models in the zoo.

    Subclasses implement fit() and predict_proba() / predict_signal().
    This is a lightweight wrapper that standardises the interface
    while keeping sklearn-style fit/predict semantics.
    """

    model_type: str = "base"

    def __init__(self, model_id: str, version: str = "1.0.0", **params: Any) -> None:
        self.model_id = model_id
        self.version = version
        self.params = params
        self.is_fitted = False
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> "TradingModel":
        """Train the model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional feature names

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of shape (n_samples,) with P(positive class)
        """
        ...

    def predict_signal(
        self,
        X: np.ndarray,
        symbol: str,
        timestamp: datetime,
        threshold_long: float = 0.55,
        threshold_short: float = 0.45,
    ) -> list[ModelDecision]:
        """Convert probabilities to trading decisions.

        Args:
            X: Feature matrix
            symbol: Symbol
            timestamp: Current timestamp
            threshold_long: P(up) above this → BUY
            threshold_short: P(up) below this → SELL

        Returns:
            List of ModelDecision objects
        """
        probas = self.predict_proba(X)
        decisions = []

        for i, p in enumerate(probas):
            if p >= threshold_long:
                side = OrderSide.BUY
                strength = Decimal(str(min((p - 0.5) * 2, 1.0)))
            elif p <= threshold_short:
                side = OrderSide.SELL
                strength = Decimal(str(max((0.5 - p) * -2, -1.0)))
            else:
                continue  # No signal

            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                side=side,
                strength=strength,
                confidence=Decimal(str(abs(p - 0.5) * 2)),
            )
            decisions.append(ModelDecision(
                model_id=self.model_id,
                model_version=self.version,
                symbol=symbol,
                timestamp=timestamp,
                signal=signal,
                probability=Decimal(str(round(p, 6))),
                threshold_used=Decimal(str(threshold_long if side == OrderSide.BUY else threshold_short)),
            ))

        return decisions

    def get_params(self) -> dict[str, Any]:
        """Return model parameters."""
        return {"model_id": self.model_id, "version": self.version, **self.params}

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class ModelStrategy(WeightBasedStrategy):
    """Adapter: wraps a TradingModel so the BacktestEngine can use it.

    The backtest engine calls on_bar(timestamp, bars, portfolio).
    This adapter:
      1. Builds features from bars (using a feature function)
      2. Calls model.predict_signal()
      3. Converts signals to target weights
    """

    def __init__(
        self,
        model: TradingModel,
        feature_fn: Any = None,
        threshold_long: float = 0.55,
        threshold_short: float = 0.45,
        max_weight: float = 1.0,
        vol_target: float = 0.15,
        vol_period: int = 20,
    ) -> None:
        """Initialize model-strategy adapter.

        Args:
            model: Trained TradingModel
            feature_fn: callable(price_history: dict[str, list[float]]) → np.ndarray or None
            threshold_long: Go-long probability threshold
            threshold_short: Go-short probability threshold
            max_weight: Maximum position weight
            vol_target: Annualised vol target for scaling
            vol_period: Lookback for vol calculation
        """
        self.model = model
        self.feature_fn = feature_fn
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.max_weight = max_weight
        self.vol_target = vol_target
        self.vol_period = vol_period
        self.price_history: dict[str, list[float]] = {}

    def on_init(self, symbols: list[str]) -> None:
        for s in symbols:
            self.price_history[s] = []

    def on_bar(
        self,
        timestamp: datetime,
        bars: dict[str, pd.Series],
        portfolio,
    ) -> dict[str, Decimal]:
        weights: dict[str, Decimal] = {}

        for symbol, bar in bars.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            close = float(bar.get("close", bar.get("Close", 0)) if isinstance(bar, pd.Series) else bar.get("close", 0))
            if close == 0:
                continue
            self.price_history[symbol].append(close)

            if len(self.price_history[symbol]) < max(self.vol_period, 30):
                continue

            # Build features
            if self.feature_fn is not None:
                try:
                    X = self.feature_fn(self.price_history)
                    if X is None or len(X) == 0:
                        continue
                except Exception:
                    continue
            else:
                X = self._default_features(symbol)
                if X is None:
                    continue

            # Predict
            try:
                decisions = self.model.predict_signal(
                    X, symbol, timestamp,
                    threshold_long=self.threshold_long,
                    threshold_short=self.threshold_short,
                )
            except Exception as e:
                logger.warning(f"Model prediction failed for {symbol}: {e}")
                continue

            if not decisions:
                continue

            decision = decisions[-1]
            raw_weight = float(decision.signal.strength)

            # Vol scaling
            prices = np.array(self.price_history[symbol][-self.vol_period:])
            if len(prices) >= 2:
                rets = np.diff(prices) / prices[:-1]
                vol = float(np.std(rets) * np.sqrt(252))
                if vol > 0:
                    vol_scale = min(self.vol_target / vol, 2.0)
                    raw_weight *= vol_scale

            weight = max(min(raw_weight, self.max_weight), -self.max_weight)
            weights[symbol] = Decimal(str(round(weight, 6)))

        return weights

    def _default_features(self, symbol: str) -> Optional[np.ndarray]:
        """Build simple features from price history."""
        prices = np.array(self.price_history[symbol])
        if len(prices) < 30:
            return None

        returns_1 = (prices[-1] / prices[-2]) - 1
        returns_5 = (prices[-1] / prices[-6]) - 1 if len(prices) > 5 else 0
        returns_10 = (prices[-1] / prices[-11]) - 1 if len(prices) > 10 else 0
        returns_20 = (prices[-1] / prices[-21]) - 1 if len(prices) > 20 else 0

        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        sma_ratio = sma_10 / sma_20 if sma_20 > 0 else 1.0

        rets = np.diff(prices[-20:]) / prices[-20:-1]
        vol_20 = float(np.std(rets)) if len(rets) > 1 else 0
        rsi = self._compute_rsi(prices[-15:])

        features = np.array([[
            returns_1, returns_5, returns_10, returns_20,
            sma_ratio, vol_20, rsi,
        ]])
        return features

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
