"""Core data types for the trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
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
