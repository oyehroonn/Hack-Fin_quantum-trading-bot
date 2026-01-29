"""Transaction cost and slippage models."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

import numpy as np


class CostModel(ABC):
    """Base class for cost models."""

    @abstractmethod
    def calculate_cost(
        self,
        quantity: Decimal,
        price: Decimal,
        side: str,  # 'BUY' or 'SELL'
    ) -> Decimal:
        """Calculate transaction cost.

        Args:
            quantity: Order quantity
            price: Execution price
            side: Order side ('BUY' or 'SELL')

        Returns:
            Total cost (fees + slippage)
        """
        ...


class FixedFee(CostModel):
    """Fixed fee per transaction."""

    def __init__(self, fee: Decimal = Decimal("1.0")) -> None:
        """Initialize fixed fee model.

        Args:
            fee: Fixed fee amount
        """
        self.fee = fee

    def calculate_cost(
        self,
        quantity: Decimal,
        price: Decimal,
        side: str,
    ) -> Decimal:
        """Calculate fixed fee."""
        return self.fee


class PercentFee(CostModel):
    """Percentage-based fee."""

    def __init__(self, fee_pct: Decimal = Decimal("0.001")) -> None:
        """Initialize percentage fee model.

        Args:
            fee_pct: Fee percentage (e.g., 0.001 for 0.1%)
        """
        self.fee_pct = fee_pct

    def calculate_cost(
        self,
        quantity: Decimal,
        price: Decimal,
        side: str,
    ) -> Decimal:
        """Calculate percentage fee."""
        notional = quantity * price
        return notional * self.fee_pct


class SpreadSlippage(CostModel):
    """Spread-based slippage model."""

    def __init__(self, spread_bps: Decimal = Decimal("5.0")) -> None:
        """Initialize spread slippage model.

        Args:
            spread_bps: Spread in basis points (e.g., 5.0 = 0.05%)
        """
        self.spread_bps = spread_bps

    def calculate_cost(
        self,
        quantity: Decimal,
        price: Decimal,
        side: str,
    ) -> Decimal:
        """Calculate spread slippage cost."""
        notional = quantity * price
        slippage_pct = self.spread_bps / Decimal("10000")
        return notional * slippage_pct


class VolumeSlippage(CostModel):
    """Volume-based slippage model (market impact)."""

    def __init__(
        self,
        impact_factor: Decimal = Decimal("0.1"),
        volume_threshold: Decimal = Decimal("10000"),
    ) -> None:
        """Initialize volume slippage model.

        Args:
            impact_factor: Market impact factor (higher = more slippage)
            volume_threshold: Volume threshold for impact calculation
        """
        self.impact_factor = impact_factor
        self.volume_threshold = volume_threshold

    def calculate_cost(
        self,
        quantity: Decimal,
        price: Decimal,
        side: str,
        volume: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate volume-based slippage.

        Args:
            quantity: Order quantity
            price: Execution price
            side: Order side
            volume: Current bar volume (optional)

        Returns:
            Slippage cost
        """
        notional = quantity * price

        if volume is None or volume == 0:
            # Default slippage if no volume data
            return notional * self.impact_factor * Decimal("0.01")

        # Market impact increases with order size relative to volume
        volume_ratio = abs(quantity) / volume
        impact = self.impact_factor * volume_ratio

        # Cap impact at reasonable level
        impact = min(impact, Decimal("0.05"))  # Max 5% slippage

        return notional * impact


class CompositeCostModel(CostModel):
    """Composite cost model combining multiple models."""

    def __init__(self, *models: CostModel) -> None:
        """Initialize composite cost model.

        Args:
            *models: Cost models to combine
        """
        self.models = models

    def calculate_cost(
        self,
        quantity: Decimal,
        price: Decimal,
        side: str,
        **kwargs,
    ) -> Decimal:
        """Calculate total cost from all models."""
        total_cost = Decimal("0")
        for model in self.models:
            if isinstance(model, VolumeSlippage) and "volume" in kwargs:
                total_cost += model.calculate_cost(quantity, price, side, kwargs["volume"])
            else:
                total_cost += model.calculate_cost(quantity, price, side)
        return total_cost
