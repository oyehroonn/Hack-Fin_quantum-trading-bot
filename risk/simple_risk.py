"""Simple risk manager enforcing max position and max leverage."""

from decimal import Decimal
from typing import Optional

from core.interfaces import RiskManager
from core.types import Order, OrderSide, PortfolioState


class SimpleRiskManager(RiskManager):
    """Simple risk manager with position and leverage limits."""

    def __init__(
        self,
        max_position_size: Decimal,
        max_leverage: Decimal,
    ) -> None:
        """Initialize risk manager.

        Args:
            max_position_size: Maximum position size in notional value
            max_leverage: Maximum leverage (e.g., 2.0 for 2x)
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage

    async def validate_order(
        self,
        order: Order,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Validate an order against risk limits."""
        # Get current position for symbol
        current_position = portfolio_state.positions.get(order.symbol)

        # Calculate new position size
        if order.side == OrderSide.BUY:
            if current_position:
                new_quantity = current_position.quantity + order.quantity
            else:
                new_quantity = order.quantity
        else:
            if current_position:
                new_quantity = current_position.quantity - order.quantity
            else:
                new_quantity = -order.quantity

        # Estimate fill price (use current position avg_price or a default)
        if current_position:
            estimated_price = current_position.avg_price
        else:
            # Would need market price, but for validation use a reasonable estimate
            # In practice, this would come from market data
            estimated_price = Decimal("100.0")  # Placeholder

        new_position_notional = abs(new_quantity) * estimated_price

        # Check max position size
        if new_position_notional > self.max_position_size:
            return (
                False,
                f"Position size {new_position_notional} exceeds max {self.max_position_size}",
            )

        # Check leverage
        # Leverage = total_position_value / equity
        total_position_value = sum(
            abs(pos.quantity) * pos.avg_price for pos in portfolio_state.positions.values()
        )
        # Add new position
        if order.symbol in portfolio_state.positions:
            # Remove old position value
            old_pos = portfolio_state.positions[order.symbol]
            total_position_value -= abs(old_pos.quantity) * old_pos.avg_price

        total_position_value += new_position_notional
        equity = portfolio_state.total_value

        if equity > 0:
            leverage = total_position_value / equity
            if leverage > self.max_leverage:
                return (
                    False,
                    f"Leverage {leverage:.2f} exceeds max {self.max_leverage}",
                )

        return (True, None)

    async def check_limits(
        self,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Check if portfolio is within risk limits."""
        # Check position sizes
        for symbol, position in portfolio_state.positions.items():
            position_notional = abs(position.quantity) * position.avg_price
            if position_notional > self.max_position_size:
                return (
                    False,
                    f"Position {symbol} size {position_notional} exceeds max {self.max_position_size}",
                )

        # Check leverage
        total_position_value = sum(
            abs(pos.quantity) * pos.avg_price for pos in portfolio_state.positions.values()
        )
        equity = portfolio_state.total_value

        if equity > 0:
            leverage = total_position_value / equity
            if leverage > self.max_leverage:
                return (
                    False,
                    f"Leverage {leverage:.2f} exceeds max {self.max_leverage}",
                )

        return (True, None)
