"""Portfolio rebalancer: convert target allocations into orders.

Compares current positions with target weights and generates
the minimum set of orders to rebalance, respecting constraints.
"""

from decimal import Decimal
from datetime import datetime
from typing import Optional

from loguru import logger

from core.types import Order, OrderSide, OrderType, PortfolioState, StrategyAllocation


class Rebalancer:
    """Convert target allocations into rebalancing orders.

    Handles:
      - Position diff calculation (target vs current)
      - Minimum trade threshold (avoid tiny trades)
      - Maximum single-order size
      - Order generation with proper side/quantity
    """

    def __init__(
        self,
        min_trade_pct: float = 0.01,
        max_trade_pct: float = 0.20,
        min_notional: float = 100.0,
    ) -> None:
        """Initialize rebalancer.

        Args:
            min_trade_pct: Minimum position change to trigger trade (1% default)
            max_trade_pct: Maximum single trade as % of portfolio
            min_notional: Minimum dollar value of a trade
        """
        self.min_trade_pct = min_trade_pct
        self.max_trade_pct = max_trade_pct
        self.min_notional = min_notional

    def generate_orders(
        self,
        allocation: StrategyAllocation,
        portfolio_state: PortfolioState,
        prices: dict[str, Decimal],
    ) -> list[Order]:
        """Generate rebalancing orders from target allocation.

        Args:
            allocation: Target allocation from Allocator
            portfolio_state: Current portfolio state
            prices: Current market prices per symbol

        Returns:
            List of orders to execute
        """
        orders = []
        equity = portfolio_state.equity

        if equity <= 0:
            logger.warning("Portfolio equity is zero, cannot rebalance")
            return orders

        # Current positions as weight of equity
        current_weights: dict[str, Decimal] = {}
        for pos in portfolio_state.positions:
            price = prices.get(pos.symbol, Decimal("0"))
            if price > 0:
                pos_value = pos.quantity * price
                current_weights[pos.symbol] = pos_value / equity

        # Calculate diffs
        all_symbols = set(allocation.asset_weights.keys()) | set(current_weights.keys())

        for symbol in all_symbols:
            target_weight = allocation.asset_weights.get(symbol, Decimal("0"))
            current_weight = current_weights.get(symbol, Decimal("0"))
            price = prices.get(symbol, Decimal("0"))

            if price <= 0:
                continue

            diff_weight = target_weight - current_weight

            # Skip small changes
            if abs(float(diff_weight)) < self.min_trade_pct:
                continue

            # Cap single trade size
            trade_weight = max(min(float(diff_weight), self.max_trade_pct), -self.max_trade_pct)

            # Calculate quantity
            trade_value = abs(Decimal(str(trade_weight))) * equity
            if float(trade_value) < self.min_notional:
                continue

            quantity = trade_value / price

            # Determine side
            side = OrderSide.BUY if trade_weight > 0 else OrderSide.SELL

            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity.quantize(Decimal("0.00000001")),
                timestamp=allocation.timestamp,
            )
            orders.append(order)

            logger.debug(
                f"Rebalance {symbol}: {float(current_weight):.4f} → {float(target_weight):.4f}, "
                f"trade={trade_weight:.4f} ({side.value} {quantity:.6f})"
            )

        # Sort: sells first (free up capital), then buys
        orders.sort(key=lambda o: (0 if o.side == OrderSide.SELL else 1))

        logger.info(
            f"Rebalancer generated {len(orders)} orders "
            f"(reason: {allocation.reason})"
        )
        return orders

    def calculate_turnover(
        self,
        old_allocation: Optional[StrategyAllocation],
        new_allocation: StrategyAllocation,
    ) -> float:
        """Calculate portfolio turnover between two allocations.

        Returns:
            Turnover as sum of absolute weight changes / 2
        """
        if old_allocation is None:
            return 1.0  # Full portfolio construction

        old_weights = old_allocation.asset_weights
        new_weights = new_allocation.asset_weights

        all_symbols = set(old_weights.keys()) | set(new_weights.keys())

        total_change = sum(
            abs(float(new_weights.get(s, Decimal("0")) - old_weights.get(s, Decimal("0"))))
            for s in all_symbols
        )

        return total_change / 2.0  # One-way turnover
