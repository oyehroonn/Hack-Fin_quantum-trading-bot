"""Dummy model that outputs random but reproducible signals."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import numpy as np

from core.interfaces import Model
from core.types import OrderSide, Signal


class DummyModel(Model):
    """Dummy model that generates random but reproducible signals."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize dummy model.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    async def predict(
        self,
        features: dict[str, float],
        symbol: str,
    ) -> Signal:
        """Generate a signal from features."""
        # Use features to make signal somewhat deterministic
        # but still random
        price = features.get("price", 100.0)
        returns = features.get("returns", 0.0)
        sma = features.get("sma", 100.0)

        # Generate signal strength based on price vs SMA
        # If price > SMA, slightly bullish bias
        # If price < SMA, slightly bearish bias
        price_sma_ratio = price / sma if sma > 0 else 1.0
        base_strength = (price_sma_ratio - 1.0) * 0.5

        # Add random component
        random_component = self._rng.uniform(-0.3, 0.3)

        # Combine to get strength
        strength = np.clip(base_strength + random_component, -1.0, 1.0)

        # Determine side
        side = OrderSide.BUY if strength > 0 else OrderSide.SELL

        # Confidence based on absolute strength
        confidence = abs(strength)

        # Target prices (simple)
        if side == OrderSide.BUY:
            target_price = Decimal(str(price * 1.02))  # 2% target
            stop_loss = Decimal(str(price * 0.98))  # 2% stop
            take_profit = Decimal(str(price * 1.05))  # 5% take profit
        else:
            target_price = Decimal(str(price * 0.98))  # 2% target
            stop_loss = Decimal(str(price * 1.02))  # 2% stop
            take_profit = Decimal(str(price * 0.95))  # 5% take profit

        return Signal(
            symbol=symbol,
            timestamp=datetime.now(),
            side=side,
            strength=Decimal(str(strength)),
            confidence=Decimal(str(confidence)),
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    async def train(
        self,
        features: list[dict[str, float]],
        labels: list[float],
    ) -> None:
        """Train the model (no-op for dummy model)."""
        pass
