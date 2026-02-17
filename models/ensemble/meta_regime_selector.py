"""Meta-regime selector: picks the best model based on detected market regime.

Different models excel in different market conditions:
  - SMA crossover / breakout → trending markets
  - Mean reversion → mean-reverting / range-bound markets
  - ML models → mixed, but calibrated for recent conditions

This selector detects the current regime and routes to the best model.
"""

import numpy as np
from typing import Any, Optional

from loguru import logger

from core.types import Regime, RegimeState
from models.base import TradingModel


class MetaRegimeSelector(TradingModel):
    """Selects the best model based on detected market regime.

    Maps each Regime → preferred model. Falls back to a default model
    if the regime is UNKNOWN or the preferred model isn't available.
    """

    model_type = "ensemble_regime_selector"

    def __init__(
        self,
        models: dict[str, TradingModel],
        regime_model_map: Optional[dict[str, str]] = None,
        default_model_id: Optional[str] = None,
        model_id: str = "regime_selector",
        version: str = "1.0.0",
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        **params: Any,
    ) -> None:
        """Initialize regime selector.

        Args:
            models: {model_id: TradingModel} — available models
            regime_model_map: {regime_name: model_id} — which model for which regime
            default_model_id: Fallback model ID
            vol_lookback: Lookback for volatility regime detection
            trend_lookback: Lookback for trend regime detection
        """
        super().__init__(model_id, version, **params)
        self.models = models
        self.default_model_id = default_model_id or list(models.keys())[0]
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback

        # Default regime → model mapping
        self.regime_model_map = regime_model_map or {
            Regime.TRENDING_UP.value: "sma_crossover",
            Regime.TRENDING_DOWN.value: "sma_crossover",
            Regime.MEAN_REVERTING.value: "mean_reversion",
            Regime.HIGH_VOLATILITY.value: "breakout",
            Regime.LOW_VOLATILITY.value: "mean_reversion",
            Regime.UNKNOWN.value: self.default_model_id,
        }

        self._current_regime: Optional[RegimeState] = None
        self._price_buffer: list[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "MetaRegimeSelector":
        """Train all constituent models."""
        for mid, model in self.models.items():
            if not model.is_fitted:
                logger.info(f"Training model {mid} for regime selector")
                model.fit(X, y, feature_names, **kwargs)

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.is_fitted = True
        return self

    def detect_regime(self, prices: list[float]) -> RegimeState:
        """Detect market regime from recent prices.

        Uses:
          - Trend: linear regression slope of log-prices
          - Volatility: realised vol vs historical median
          - Mean reversion: Hurst exponent approximation
        """
        from decimal import Decimal
        from datetime import datetime

        prices_arr = np.array(prices)
        n = len(prices_arr)

        if n < self.trend_lookback:
            return RegimeState(
                regime=Regime.UNKNOWN,
                confidence=Decimal("0.1"),
                timestamp=datetime.now(),
                symbol="",
            )

        recent = prices_arr[-self.trend_lookback:]
        vol_window = prices_arr[-self.vol_lookback:]

        # 1. Trend detection via linear regression on log-prices
        log_prices = np.log(recent)
        x = np.arange(len(log_prices))
        slope, intercept = np.polyfit(x, log_prices, 1)
        trend_strength = abs(slope) * len(log_prices)  # Normalised

        # 2. Volatility regime
        returns = np.diff(vol_window) / vol_window[:-1]
        realised_vol = float(np.std(returns) * np.sqrt(252))

        # 3. Hurst exponent proxy (variance ratio test)
        hurst_proxy = self._variance_ratio(returns)

        # Classify
        if trend_strength > 0.15 and slope > 0:
            regime = Regime.TRENDING_UP
            conf = min(trend_strength * 2, 0.95)
        elif trend_strength > 0.15 and slope < 0:
            regime = Regime.TRENDING_DOWN
            conf = min(trend_strength * 2, 0.95)
        elif hurst_proxy < 0.4:
            regime = Regime.MEAN_REVERTING
            conf = min((0.5 - hurst_proxy) * 3, 0.90)
        elif realised_vol > 0.30:
            regime = Regime.HIGH_VOLATILITY
            conf = min(realised_vol, 0.90)
        elif realised_vol < 0.10:
            regime = Regime.LOW_VOLATILITY
            conf = min((0.15 - realised_vol) * 5, 0.85)
        else:
            regime = Regime.UNKNOWN
            conf = 0.3

        self._current_regime = RegimeState(
            regime=regime,
            confidence=Decimal(str(round(conf, 4))),
            timestamp=datetime.now(),
            symbol="",
            indicators={
                "trend_strength": round(float(trend_strength), 4),
                "slope": round(float(slope), 6),
                "realised_vol": round(realised_vol, 4),
                "hurst_proxy": round(float(hurst_proxy), 4),
            },
        )
        return self._current_regime

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Route prediction to the regime-appropriate model."""
        if not self.is_fitted:
            raise RuntimeError("MetaRegimeSelector not fitted")

        # Determine which model to use
        if self._current_regime is not None:
            regime_name = self._current_regime.regime.value
            target_model_id = self.regime_model_map.get(regime_name, self.default_model_id)
        else:
            target_model_id = self.default_model_id

        # Find the model
        model = self.models.get(target_model_id)
        if model is None:
            model = self.models[self.default_model_id]
            logger.warning(f"Model {target_model_id} not found, using {self.default_model_id}")

        return model.predict_proba(X)

    def update_prices(self, price: float) -> None:
        """Feed a new price to update regime detection."""
        self._price_buffer.append(price)
        if len(self._price_buffer) > self.trend_lookback * 2:
            self._price_buffer = self._price_buffer[-self.trend_lookback * 2:]

        if len(self._price_buffer) >= self.trend_lookback:
            self.detect_regime(self._price_buffer)

    @property
    def current_regime(self) -> Optional[RegimeState]:
        return self._current_regime

    @staticmethod
    def _variance_ratio(returns: np.ndarray, lag: int = 5) -> float:
        """Variance ratio test as Hurst exponent proxy.

        VR < 1 → mean reverting (H < 0.5)
        VR ≈ 1 → random walk (H ≈ 0.5)
        VR > 1 → trending (H > 0.5)

        Returns a proxy for Hurst exponent (0..1).
        """
        if len(returns) < lag * 2:
            return 0.5

        # Variance of 1-period returns
        var_1 = np.var(returns)
        if var_1 == 0:
            return 0.5

        # Variance of lag-period returns
        lagged_returns = np.array([
            sum(returns[i:i + lag]) for i in range(0, len(returns) - lag + 1, lag)
        ])
        var_lag = np.var(lagged_returns)

        vr = var_lag / (lag * var_1)

        # Map to Hurst-like (0..1): VR=1 → 0.5, VR<1 → <0.5, VR>1 → >0.5
        hurst = 0.5 + (vr - 1.0) * 0.25
        return float(np.clip(hurst, 0.0, 1.0))
