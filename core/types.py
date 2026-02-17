"""Core data types for the trading system.

All value objects are frozen dataclasses with validation.
Decimal is used for prices/quantities to avoid floating-point issues.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from enum import Enum


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Bar:
    """OHLCV bar data."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        """Validate bar data."""
        if self.high < self.low:
            raise ValueError("High must be >= low")
        if self.open < self.low or self.open > self.high:
            raise ValueError("Open must be within [low, high]")
        if self.close < self.low or self.close > self.high:
            raise ValueError("Close must be within [low, high]")
        if self.volume < 0:
            raise ValueError("Volume must be >= 0")


@dataclass(frozen=True)
class Tick:
    """Tick data (trade or quote)."""

    symbol: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None


@dataclass(frozen=True)
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self) -> None:
        """Validate order data."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be > 0")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price required for LIMIT orders")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop price required for STOP orders")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Both limit and stop prices required for STOP_LIMIT orders")


@dataclass(frozen=True)
class Fill:
    """Order fill information."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate fill data."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be > 0")
        if self.price <= 0:
            raise ValueError("Price must be > 0")
        if self.fee < 0:
            raise ValueError("Fee must be >= 0")


@dataclass(frozen=True)
class Position:
    """Position representation."""

    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: datetime

    @property
    def market_value(self) -> Decimal:
        """Calculate market value of position."""
        return self.quantity * self.avg_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0


@dataclass(frozen=True)
class PortfolioState:
    """Portfolio state snapshot."""

    timestamp: datetime
    cash: Decimal
    positions: dict[str, Position]
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal

    def __post_init__(self) -> None:
        """Validate portfolio state."""
        if self.cash < 0:
            raise ValueError("Cash cannot be negative")


@dataclass(frozen=True)
class Signal:
    """Trading signal."""

    symbol: str
    timestamp: datetime
    side: OrderSide
    strength: Decimal  # -1 to 1, where 1 is strong buy, -1 is strong sell
    confidence: Decimal  # 0 to 1
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Validate signal data."""
        if not (-1 <= self.strength <= 1):
            raise ValueError("Strength must be between -1 and 1")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


# ─── v0.2 additions: Regime, Model decisions, Allocations, Risk, Alerts, LLM ───


class Regime(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    MEAN_REVERTING = "MEAN_REVERTING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    """Alert categories."""

    RISK_LIMIT = "RISK_LIMIT"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MODEL_DEGRADATION = "MODEL_DEGRADATION"
    DATA_QUALITY = "DATA_QUALITY"
    EXECUTION_ANOMALY = "EXECUTION_ANOMALY"


class LLMActionType(str, Enum):
    """Types of actions the LLM governor can take."""

    SELECT_STRATEGY = "SELECT_STRATEGY"
    ADJUST_RISK = "ADJUST_RISK"
    NO_TRADE = "NO_TRADE"
    EXPLAIN_PERFORMANCE = "EXPLAIN_PERFORMANCE"
    TRIAGE_ANOMALY = "TRIAGE_ANOMALY"


@dataclass(frozen=True)
class FeatureVector:
    """Named feature vector for a single observation."""

    symbol: str
    timestamp: datetime
    timeframe: str
    features: dict[str, float]

    @property
    def names(self) -> list[str]:
        """Feature names in sorted order."""
        return sorted(self.features.keys())

    @property
    def values(self) -> list[float]:
        """Feature values in sorted-name order."""
        return [self.features[k] for k in self.names]


@dataclass(frozen=True)
class RegimeState:
    """Market regime classification result."""

    regime: Regime
    confidence: Decimal
    timestamp: datetime
    symbol: str
    indicators: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0 <= self.confidence <= 1):
            raise ValueError("Regime confidence must be between 0 and 1")


@dataclass(frozen=True)
class ModelDecision:
    """A single model's output for one symbol at one point in time."""

    model_id: str
    model_version: str
    symbol: str
    timestamp: datetime
    signal: Signal
    probability: Optional[Decimal] = None
    threshold_used: Optional[Decimal] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyAllocation:
    """Output of the allocator: per-strategy and per-asset target weights."""

    timestamp: datetime
    strategy_weights: dict[str, Decimal]      # strategy_id → portfolio fraction
    asset_weights: dict[str, Decimal]          # symbol → target weight (-1 to 1)
    regime: Optional[RegimeState] = None
    reason: str = ""

    def __post_init__(self) -> None:
        total = sum(abs(w) for w in self.strategy_weights.values())
        if total > Decimal("1.01"):
            raise ValueError(f"Strategy weights sum {total} exceeds 1.0")


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk limits for the execution layer."""

    max_drawdown: Decimal = Decimal("0.20")
    max_daily_loss: Decimal = Decimal("0.03")
    max_leverage: Decimal = Decimal("2.0")
    max_symbol_exposure: Decimal = Decimal("0.25")
    max_position_size: Decimal = Decimal("10000")
    cooldown_minutes: int = 60


@dataclass(frozen=True)
class Alert:
    """System alert raised by risk, drift, or anomaly detectors."""

    alert_id: str
    severity: AlertSeverity
    alert_type: AlertType
    message: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMAction:
    """Validated output from the LLM governor."""

    action_type: LLMActionType
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: Decimal = Decimal("0.5")
    raw_output: str = ""

    def __post_init__(self) -> None:
        if not (0 <= self.confidence <= 1):
            raise ValueError("LLM confidence must be between 0 and 1")


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for a model (OOS or in-sample)."""

    model_id: str
    model_version: str
    evaluation_type: str  # "oos", "in_sample", "walk_forward"
    timestamp: datetime
    sharpe: float = 0.0
    sortino: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_pnl: float = 0.0
    turnover: float = 0.0
    calmar: float = 0.0
    stability: float = 0.0  # R² of equity curve
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """Result of a research experiment (training + evaluation run)."""

    experiment_id: str
    model_id: str
    model_version: str
    timestamp: datetime
    config: dict[str, Any] = field(default_factory=dict)
    train_metrics: Optional[ModelMetrics] = None
    val_metrics: Optional[ModelMetrics] = None
    test_metrics: Optional[ModelMetrics] = None
    status: str = "completed"  # completed, failed, running
