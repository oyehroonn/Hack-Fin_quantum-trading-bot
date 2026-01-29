"""Simulated broker with order handling, fills, fees, slippage."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import numpy as np

from backtest.cost_models import CostModel, CompositeCostModel, FixedFee, PercentFee, SpreadSlippage


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    order_type: str  # 'MARKET' or 'LIMIT'
    limit_price: Optional[Decimal] = None
    timestamp: datetime = datetime.min
    status: str = "PENDING"  # PENDING, FILLED, PARTIALLY_FILLED, REJECTED


@dataclass
class Fill:
    """Fill representation."""

    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    cost: Decimal
    timestamp: datetime


class SimBroker:
    """Simulated broker with realistic order execution."""

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        partial_fill_prob: float = 0.1,
        partial_fill_ratio: float = 0.5,
    ) -> None:
        """Initialize simulated broker.

        Args:
            cost_model: Cost model for fees and slippage
            partial_fill_prob: Probability of partial fill (0-1)
            partial_fill_ratio: Ratio of quantity filled in partial fill (0-1)
        """
        if cost_model is None:
            # Default: 0.1% fee + 5 bps spread
            self.cost_model = CompositeCostModel(
                PercentFee(Decimal("0.001")),
                SpreadSlippage(Decimal("5.0")),
            )
        else:
            self.cost_model = cost_model

        self.partial_fill_prob = partial_fill_prob
        self.partial_fill_ratio = partial_fill_ratio
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self._rng = np.random.default_rng(42)  # Reproducible

    def submit_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "MARKET",
        limit_price: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
    ) -> Order:
        """Submit an order.

        Args:
            order_id: Unique order ID
            symbol: Symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            order_type: Order type ('MARKET' or 'LIMIT')
            limit_price: Limit price (required for LIMIT orders)
            timestamp: Order timestamp

        Returns:
            Order object
        """
        if order_type == "LIMIT" and limit_price is None:
            raise ValueError("Limit price required for LIMIT orders")

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            timestamp=timestamp or datetime.now(),
        )

        self.orders[order_id] = order
        return order

    def process_bar(
        self,
        timestamp: datetime,
        bars: dict[str, dict],  # symbol -> {open, high, low, close, volume}
    ) -> list[Fill]:
        """Process pending orders for a bar.

        Args:
            timestamp: Bar timestamp
            bars: Dictionary of bar data by symbol

        Returns:
            List of fills
        """
        fills = []

        for order in self.orders.values():
            if order.status in ("FILLED", "REJECTED"):
                continue

            if order.symbol not in bars:
                continue

            bar = bars[order.symbol]
            fill = self._try_fill_order(order, timestamp, bar)

            if fill:
                fills.append(fill)
                self.fills.append(fill)

                # Update order status
                if fill.quantity == order.quantity:
                    order.status = "FILLED"
                else:
                    order.status = "PARTIALLY_FILLED"
                    order.quantity -= fill.quantity

        return fills

    def _try_fill_order(
        self,
        order: Order,
        timestamp: datetime,
        bar: dict,
    ) -> Optional[Fill]:
        """Try to fill an order.

        Args:
            order: Order to fill
            timestamp: Bar timestamp
            bar: Bar data {open, high, low, close, volume}

        Returns:
            Fill if order can be filled, None otherwise
        """
        if order.order_type == "MARKET":
            return self._fill_market_order(order, timestamp, bar)
        elif order.order_type == "LIMIT":
            return self._fill_limit_order(order, timestamp, bar)
        return None

    def _fill_market_order(
        self,
        order: Order,
        timestamp: datetime,
        bar: dict,
    ) -> Optional[Fill]:
        """Fill a market order.

        Market orders fill at next bar open with slippage.
        """
        # Market orders fill at bar open
        fill_price = Decimal(str(bar["open"]))

        # Apply slippage based on side
        if order.side == "BUY":
            # Buy at ask (higher)
            slippage = Decimal(str(bar.get("spread", 0) / 2)) if "spread" in bar else Decimal("0")
            fill_price += slippage
        else:
            # Sell at bid (lower)
            slippage = Decimal(str(bar.get("spread", 0) / 2)) if "spread" in bar else Decimal("0")
            fill_price -= slippage

        # Determine fill quantity (may be partial)
        fill_quantity = order.quantity
        if self._rng.random() < self.partial_fill_prob:
            fill_quantity = Decimal(str(float(fill_quantity) * self.partial_fill_ratio))

        # Calculate cost
        volume = Decimal(str(bar.get("volume", 1)))
        cost = self.cost_model.calculate_cost(fill_quantity, fill_price, order.side, volume=volume)

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            cost=cost,
            timestamp=timestamp,
        )

    def _fill_limit_order(
        self,
        order: Order,
        timestamp: datetime,
        bar: dict,
    ) -> Optional[Fill]:
        """Fill a limit order if price touches limit.

        Limit orders fill if:
        - BUY: bar low <= limit_price
        - SELL: bar high >= limit_price
        """
        if order.limit_price is None:
            return None

        limit_price = order.limit_price
        bar_low = Decimal(str(bar["low"]))
        bar_high = Decimal(str(bar["high"]))

        can_fill = False
        if order.side == "BUY" and bar_low <= limit_price:
            can_fill = True
            fill_price = min(limit_price, Decimal(str(bar["open"])))
        elif order.side == "SELL" and bar_high >= limit_price:
            can_fill = True
            fill_price = max(limit_price, Decimal(str(bar["open"])))

        if not can_fill:
            return None

        # Determine fill quantity
        fill_quantity = order.quantity
        if self._rng.random() < self.partial_fill_prob:
            fill_quantity = Decimal(str(float(fill_quantity) * self.partial_fill_ratio))

        # Calculate cost
        volume = Decimal(str(bar.get("volume", 1)))
        cost = self.cost_model.calculate_cost(fill_quantity, fill_price, order.side, volume=volume)

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            cost=cost,
            timestamp=timestamp,
        )

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return [o for o in self.orders.values() if o.status == "PENDING"]
