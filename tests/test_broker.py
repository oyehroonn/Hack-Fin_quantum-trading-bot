"""Tests for paper broker fill logic."""

from datetime import datetime
from decimal import Decimal

import pytest

from core.types import Order, OrderSide, OrderType, OrderStatus
from execution.paper_broker import PaperBroker


@pytest.mark.asyncio
async def test_broker_market_order_fill() -> None:
    """Test broker fills market orders."""
    broker = PaperBroker(
        initial_cash=Decimal("100000.0"),
        slippage_bps=5.0,
        commission_bps=1.0,
    )
    broker.set_current_price("AAPL", Decimal("100.0"))

    order = Order(
        order_id="test-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )

    order_id = await broker.submit_order(order)
    assert order_id == "test-1"

    order_status = await broker.get_order_status(order_id)
    assert order_status.status == OrderStatus.FILLED

    fills = await broker.get_fills(order_id)
    assert len(fills) == 1
    assert fills[0].quantity == Decimal("10")
    assert fills[0].price > Decimal("100.0")  # Should have slippage
    assert fills[0].fee > Decimal("0")  # Should have commission


@pytest.mark.asyncio
async def test_broker_insufficient_funds() -> None:
    """Test broker rejects orders with insufficient funds."""
    broker = PaperBroker(
        initial_cash=Decimal("1000.0"),
        slippage_bps=5.0,
        commission_bps=1.0,
    )
    broker.set_current_price("AAPL", Decimal("100.0"))

    # Order that exceeds available cash
    order = Order(
        order_id="test-2",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("20"),  # 20 * 100 = 2000 > 1000
        order_type=OrderType.MARKET,
    )

    order_id = await broker.submit_order(order)
    order_status = await broker.get_order_status(order_id)
    assert order_status.status == OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_broker_position_updates() -> None:
    """Test broker updates positions correctly."""
    broker = PaperBroker(
        initial_cash=Decimal("100000.0"),
        slippage_bps=5.0,
        commission_bps=1.0,
    )
    broker.set_current_price("AAPL", Decimal("100.0"))

    # Buy order
    buy_order = Order(
        order_id="buy-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )
    await broker.submit_order(buy_order)

    positions = await broker.get_positions()
    assert "AAPL" in positions
    assert positions["AAPL"].quantity == Decimal("10")

    # Sell order (partial)
    sell_order = Order(
        order_id="sell-1",
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=Decimal("5"),
        order_type=OrderType.MARKET,
    )
    await broker.set_current_price("AAPL", Decimal("105.0"))
    await broker.submit_order(sell_order)

    positions = await broker.get_positions()
    assert positions["AAPL"].quantity == Decimal("5")  # 10 - 5 = 5


@pytest.mark.asyncio
async def test_broker_portfolio_state() -> None:
    """Test broker portfolio state calculation."""
    broker = PaperBroker(
        initial_cash=Decimal("100000.0"),
        slippage_bps=5.0,
        commission_bps=1.0,
    )
    broker.set_current_price("AAPL", Decimal("100.0"))

    order = Order(
        order_id="test-3",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )
    await broker.submit_order(order)

    portfolio_state = await broker.get_portfolio_state()
    assert portfolio_state.cash < Decimal("100000.0")  # Cash reduced
    assert len(portfolio_state.positions) == 1
    assert portfolio_state.total_value > Decimal("0")


@pytest.mark.asyncio
async def test_broker_slippage_and_fees() -> None:
    """Test broker applies slippage and fees correctly."""
    broker = PaperBroker(
        initial_cash=Decimal("100000.0"),
        slippage_bps=10.0,  # 0.1% slippage
        commission_bps=2.0,  # 0.02% commission
    )
    broker.set_current_price("AAPL", Decimal("100.0"))

    order = Order(
        order_id="test-4",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )
    await broker.submit_order(order)

    fills = await broker.get_fills(order.order_id)
    assert len(fills) == 1
    fill = fills[0]

    # Fill price should be higher than market (slippage for buy)
    assert fill.price > Decimal("100.0")

    # Fee should be calculated
    notional = fill.quantity * fill.price
    expected_fee = notional * Decimal("0.0002")  # 2 bps
    assert abs(fill.fee - expected_fee) < Decimal("0.01")  # Allow small rounding
