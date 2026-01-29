"""Paper trading broker that simulates fills with slippage and fees."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from core.interfaces import Broker
from core.types import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PortfolioState,
)


class PaperBroker(Broker):
    """Paper trading broker with slippage and commission simulation."""

    def __init__(
        self,
        initial_cash: Decimal,
        slippage_bps: float = 5.0,
        commission_bps: float = 1.0,
    ) -> None:
        """Initialize paper broker.

        Args:
            initial_cash: Initial cash balance
            slippage_bps: Slippage in basis points (1 bps = 0.01%)
            commission_bps: Commission in basis points
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.fills: dict[str, list[Fill]] = {}
        self._current_price: dict[str, Decimal] = {}

    def set_current_price(self, symbol: str, price: Decimal) -> None:
        """Set current market price for a symbol."""
        self._current_price[symbol] = price

    async def submit_order(self, order: Order) -> str:
        """Submit an order and return order ID."""
        # Store order
        self.orders[order.order_id] = order

        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            await self._fill_market_order(order)

        return order.order_id

    async def _fill_market_order(self, order: Order) -> None:
        """Fill a market order with slippage and fees."""
        if order.symbol not in self._current_price:
            # No price available, reject order
            updated_order = Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                timestamp=order.timestamp or datetime.now(),
                status=OrderStatus.REJECTED,
            )
            self.orders[order.order_id] = updated_order
            return

        market_price = self._current_price[order.symbol]

        # Apply slippage
        slippage_multiplier = Decimal(str(1 + (self.slippage_bps / 10000)))
        if order.side == OrderSide.BUY:
            fill_price = market_price * slippage_multiplier
        else:
            fill_price = market_price / slippage_multiplier

        # Calculate commission
        notional = order.quantity * fill_price
        commission = notional * Decimal(str(self.commission_bps / 10000))

        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            total_cost = notional + commission
            if total_cost > self.cash:
                # Reject order - insufficient funds
                updated_order = Order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    timestamp=order.timestamp or datetime.now(),
                    status=OrderStatus.REJECTED,
                )
                self.orders[order.order_id] = updated_order
                return

        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=commission,
            timestamp=datetime.now(),
        )

        # Store fill
        if order.order_id not in self.fills:
            self.fills[order.order_id] = []
        self.fills[order.order_id].append(fill)

        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= (notional + commission)
        else:
            self.cash += (notional - commission)

        # Update position
        await self._update_position(order.symbol, fill)

        # Update order status
        updated_order = Order(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            timestamp=order.timestamp or datetime.now(),
            status=OrderStatus.FILLED,
        )
        self.orders[order.order_id] = updated_order

    async def _update_position(self, symbol: str, fill: Fill) -> None:
        """Update position after a fill."""
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=fill.quantity if fill.side == OrderSide.BUY else -fill.quantity,
                avg_price=fill.price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                timestamp=fill.timestamp,
            )
        else:
            # Update existing position
            pos = self.positions[symbol]
            current_price = self._current_price.get(symbol, pos.avg_price)

            if fill.side == OrderSide.BUY:
                new_quantity = pos.quantity + fill.quantity
            else:
                new_quantity = pos.quantity - fill.quantity

            # Calculate new average price
            if new_quantity == 0:
                # Position closed
                new_avg_price = Decimal("0")
            elif (fill.side == OrderSide.BUY and pos.quantity >= 0) or (
                fill.side == OrderSide.SELL and pos.quantity <= 0
            ):
                # Adding to position
                total_cost = (pos.quantity * pos.avg_price) + (fill.quantity * fill.price)
                new_avg_price = total_cost / new_quantity if new_quantity != 0 else Decimal("0")
            else:
                # Reducing or reversing position
                if abs(new_quantity) < abs(pos.quantity):
                    # Reducing position - realize PnL
                    closed_quantity = abs(pos.quantity - new_quantity)
                    if pos.quantity > 0:
                        realized = closed_quantity * (fill.price - pos.avg_price)
                    else:
                        realized = closed_quantity * (pos.avg_price - fill.price)
                    new_avg_price = pos.avg_price
                else:
                    # Reversing position
                    new_avg_price = fill.price

            # Calculate unrealized PnL
            if new_quantity != 0:
                unrealized = new_quantity * (current_price - new_avg_price)
            else:
                unrealized = Decimal("0")

            # Calculate realized PnL
            if new_quantity == 0:
                # Position closed, all PnL is realized
                if pos.quantity > 0:
                    realized = pos.quantity * (fill.price - pos.avg_price)
                else:
                    realized = abs(pos.quantity) * (pos.avg_price - fill.price)
            else:
                realized = pos.realized_pnl

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                avg_price=new_avg_price,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                timestamp=fill.timestamp,
            )

            # Remove position if flat
            if new_quantity == 0:
                del self.positions[symbol]

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False

        updated_order = Order(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            timestamp=order.timestamp or datetime.now(),
            status=OrderStatus.CANCELLED,
        )
        self.orders[order_id] = updated_order
        return True

    async def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.orders[order_id]

    async def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order."""
        return self.fills.get(order_id, [])

    async def get_positions(self) -> dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()

    async def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        # Update unrealized PnL for all positions
        total_unrealized = Decimal("0")
        total_value = self.cash

        for symbol, position in self.positions.items():
            current_price = self._current_price.get(symbol, position.avg_price)
            unrealized = position.quantity * (current_price - position.avg_price)
            total_unrealized += unrealized
            total_value += position.quantity * current_price

        total_realized = sum(pos.realized_pnl for pos in self.positions.values())

        return PortfolioState(
            timestamp=datetime.now(),
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value,
            unrealized_pnl=total_unrealized,
            realized_pnl=total_realized,
        )
