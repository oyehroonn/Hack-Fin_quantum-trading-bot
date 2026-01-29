"""Tests for core types."""

from datetime import datetime
from decimal import Decimal

import pytest

from core.types import (
    Bar,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    PortfolioState,
    Position,
    Signal,
)


def test_bar_creation() -> None:
    """Test bar creation and validation."""
    bar = Bar(
        symbol="AAPL",
        timestamp=datetime.now(),
        open=Decimal("100.0"),
        high=Decimal("105.0"),
        low=Decimal("99.0"),
        close=Decimal("103.0"),
        volume=Decimal("1000"),
    )
    assert bar.symbol == "AAPL"
    assert bar.close == Decimal("103.0")


def test_bar_validation() -> None:
    """Test bar validation."""
    with pytest.raises(ValueError, match="High must be >= low"):
        Bar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=Decimal("100.0"),
            high=Decimal("99.0"),
            low=Decimal("100.0"),
            close=Decimal("100.0"),
            volume=Decimal("1000"),
        )


def test_order_creation() -> None:
    """Test order creation."""
    order = Order(
        order_id="test-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.PENDING


def test_order_validation() -> None:
    """Test order validation."""
    with pytest.raises(ValueError, match="Quantity must be > 0"):
        Order(
            order_id="test-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("0"),
            order_type=OrderType.MARKET,
        )


def test_portfolio_state() -> None:
    """Test portfolio state creation."""
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=Decimal("10"),
            avg_price=Decimal("100.0"),
            unrealized_pnl=Decimal("50.0"),
            realized_pnl=Decimal("20.0"),
            timestamp=datetime.now(),
        )
    }
    state = PortfolioState(
        timestamp=datetime.now(),
        cash=Decimal("50000.0"),
        positions=positions,
        total_value=Decimal("50150.0"),
        unrealized_pnl=Decimal("50.0"),
        realized_pnl=Decimal("20.0"),
    )
    assert state.cash == Decimal("50000.0")
    assert len(state.positions) == 1


def test_signal_creation() -> None:
    """Test signal creation."""
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        side=OrderSide.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
    )
    assert signal.symbol == "AAPL"
    assert signal.strength == Decimal("0.8")


def test_signal_validation() -> None:
    """Test signal validation."""
    with pytest.raises(ValueError, match="Strength must be between -1 and 1"):
        Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            side=OrderSide.BUY,
            strength=Decimal("1.5"),
            confidence=Decimal("0.9"),
        )
