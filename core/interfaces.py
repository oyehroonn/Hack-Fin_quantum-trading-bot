"""Interface protocols for the trading system."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from core.types import (
    Bar,
    Tick,
    Order,
    Fill,
    Position,
    PortfolioState,
    Signal,
)


class DataSource(ABC):
    """Interface for data sources."""

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> AsyncIterator[Bar]:
        """Stream bars for a symbol."""
        ...

    @abstractmethod
    async def get_ticks(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> AsyncIterator[Tick]:
        """Stream ticks for a symbol."""
        ...


class FeatureStore(ABC):
    """Interface for feature stores."""

    @abstractmethod
    async def compute_features(
        self,
        symbol: str,
        bars: list[Bar],
    ) -> dict[str, float]:
        """Compute features from bars."""
        ...

    @abstractmethod
    async def get_latest_features(
        self,
        symbol: str,
    ) -> dict[str, float]:
        """Get latest computed features for a symbol."""
        ...


class Model(ABC):
    """Interface for prediction models."""

    @abstractmethod
    async def predict(
        self,
        features: dict[str, float],
        symbol: str,
    ) -> Signal:
        """Generate a signal from features."""
        ...

    @abstractmethod
    async def train(
        self,
        features: list[dict[str, float]],
        labels: list[float],
    ) -> None:
        """Train the model."""
        ...


class Strategy(ABC):
    """Interface for trading strategies."""

    @abstractmethod
    async def generate_signals(
        self,
        portfolio_state: PortfolioState,
        features: dict[str, float],
    ) -> list[Signal]:
        """Generate trading signals based on portfolio state and features."""
        ...


class Broker(ABC):
    """Interface for brokers."""

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit an order and return order ID."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        ...

    @abstractmethod
    async def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        """Get current positions."""
        ...

    @abstractmethod
    async def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        ...


class RiskManager(ABC):
    """Interface for risk managers."""

    @abstractmethod
    async def validate_order(
        self,
        order: Order,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Validate an order against risk limits.
        
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        ...

    @abstractmethod
    async def check_limits(
        self,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Check if portfolio is within risk limits.
        
        Returns:
            Tuple of (is_valid, violation_reason)
        """
        ...
