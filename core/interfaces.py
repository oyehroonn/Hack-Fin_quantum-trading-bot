"""Interface protocols for the trading system.

All business logic depends inward on these ABCs.
Implementations live in adapter layers (data/, models/, execution/, llm/).
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from core.types import (
    Alert,
    Bar,
    Fill,
    LLMAction,
    ModelDecision,
    ModelMetrics,
    Order,
    Position,
    PortfolioState,
    RegimeState,
    Signal,
    StrategyAllocation,
    Tick,
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


# ─── v0.2 additions ───


class ModelRegistry(ABC):
    """Interface for model persistence and champion/challenger management."""

    @abstractmethod
    def save_model(
        self,
        model_id: str,
        version: str,
        artifact: Any,
        metadata: dict[str, Any],
    ) -> None:
        """Save a trained model artifact."""
        ...

    @abstractmethod
    def load_model(self, model_id: str, version: Optional[str] = None) -> Any:
        """Load a model artifact (latest version if not specified)."""
        ...

    @abstractmethod
    def list_models(self, asset_class: Optional[str] = None) -> list[dict[str, Any]]:
        """List registered models with metadata."""
        ...

    @abstractmethod
    def get_champion(self, asset_class: str, timeframe: str) -> Optional[str]:
        """Get champion model_id for a given asset_class/timeframe."""
        ...

    @abstractmethod
    def promote(self, model_id: str, version: str, asset_class: str, timeframe: str) -> None:
        """Promote a model version to champion."""
        ...

    @abstractmethod
    def record_metrics(self, metrics: ModelMetrics) -> None:
        """Record evaluation metrics for a model version."""
        ...


class Evaluator(ABC):
    """Interface for model/strategy evaluation."""

    @abstractmethod
    def evaluate(
        self,
        predictions: list[ModelDecision],
        actuals: list[float],
        costs: Optional[list[float]] = None,
    ) -> ModelMetrics:
        """Evaluate model predictions against actual outcomes."""
        ...


class Allocator(ABC):
    """Interface for multi-strategy portfolio allocation."""

    @abstractmethod
    def allocate(
        self,
        decisions: dict[str, list[ModelDecision]],
        regime: Optional[RegimeState] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> StrategyAllocation:
        """Allocate capital across strategies/models.

        Args:
            decisions: {strategy_id: [ModelDecision per symbol]}
            regime: Current market regime
            portfolio_state: Current portfolio

        Returns:
            Target allocation
        """
        ...


class AlertSink(ABC):
    """Interface for publishing alerts."""

    @abstractmethod
    def publish(self, alert: Alert) -> None:
        """Publish an alert."""
        ...


class LLMClient(ABC):
    """Interface for LLM interactions with strict JSON output."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions
            user_message: User/context message
            response_schema: JSON schema the output must conform to
            temperature: Sampling temperature

        Returns:
            Raw string output (caller validates against schema)
        """
        ...


class RegimeDetector(ABC):
    """Interface for market regime detection."""

    @abstractmethod
    def detect(
        self,
        prices: list[float],
        volumes: Optional[list[float]] = None,
        timestamp: Optional["datetime"] = None,
        symbol: str = "",
    ) -> RegimeState:
        """Detect current market regime from recent data."""
        ...
