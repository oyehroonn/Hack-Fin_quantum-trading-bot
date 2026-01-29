"""Portfolio accounting: PnL, positions, equity curve."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """Position accounting."""

    symbol: str
    quantity: Decimal
    avg_price: Decimal
    realized_pnl: Decimal = Decimal("0")
    last_price: Optional[Decimal] = None

    @property
    def market_value(self) -> Decimal:
        """Market value of position."""
        if self.last_price is None:
            return Decimal("0")
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized PnL."""
        if self.last_price is None or self.quantity == 0:
            return Decimal("0")
        return self.quantity * (self.last_price - self.avg_price)

    @property
    def total_pnl(self) -> Decimal:
        """Total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def update_price(self, price: Decimal) -> None:
        """Update last price."""
        self.last_price = price

    def add_fill(
        self,
        quantity: Decimal,
        price: Decimal,
        cost: Decimal,
    ) -> None:
        """Add a fill to position.

        Args:
            quantity: Fill quantity (positive for buy, negative for sell)
            price: Fill price
            cost: Transaction cost
        """
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.avg_price = price
            # Cost reduces realized PnL
            self.realized_pnl = -cost
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to position
            total_cost = (self.quantity * self.avg_price) + (quantity * price)
            self.quantity += quantity
            if self.quantity != 0:
                self.avg_price = total_cost / self.quantity
            self.realized_pnl -= cost
        else:
            # Reducing or reversing position
            # Ensure both are Decimal for min() operation
            qty_abs = abs(self.quantity)
            qty_fill_abs = abs(quantity)
            closed_quantity = min(qty_abs, qty_fill_abs)
            if self.quantity > 0:
                # Closing long position
                pnl = closed_quantity * (price - self.avg_price)
            else:
                # Closing short position
                pnl = closed_quantity * (self.avg_price - price)

            self.realized_pnl += pnl - cost
            self.quantity += quantity

            if self.quantity == 0:
                self.avg_price = Decimal("0")
            else:
                # Update average price for remaining position
                remaining_quantity = abs(self.quantity)
                if remaining_quantity > 0:
                    # For reversed position, new avg price is fill price
                    if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
                        self.avg_price = price


@dataclass
class Trade:
    """Trade record."""

    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Decimal
    cost: Decimal
    pnl: Decimal = Decimal("0")


@dataclass
class Portfolio:
    """Portfolio accounting."""

    initial_cash: Decimal
    cash: Decimal = Decimal("0")
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize portfolio."""
        self.cash = self.initial_cash
        self.equity_curve.append((datetime.min, self.initial_cash))

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update prices for all positions.

        Args:
            prices: Dictionary of symbol -> price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

    def add_fill(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        cost: Decimal,
    ) -> None:
        """Add a fill and update portfolio.

        Args:
            timestamp: Fill timestamp
            symbol: Symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Fill quantity (positive for buy, negative for sell)
            price: Fill price
            cost: Transaction cost
        """
        # Update cash
        notional = quantity * price
        if side == "BUY":
            self.cash -= notional + cost
        else:
            self.cash += notional - cost

        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=Decimal("0"), avg_price=Decimal("0"))

        position = self.positions[symbol]
        old_pnl = position.total_pnl
        position.add_fill(quantity, price, cost)
        new_pnl = position.total_pnl
        trade_pnl = new_pnl - old_pnl

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            cost=cost,
            pnl=trade_pnl,
        )
        self.trades.append(trade)

        # Remove zero positions
        if position.quantity == 0:
            del self.positions[symbol]

    def get_total_value(self, prices: Optional[dict[str, Decimal]] = None) -> Decimal:
        """Get total portfolio value.

        Args:
            prices: Current prices (if None, uses last_price from positions)

        Returns:
            Total portfolio value
        """
        if prices:
            self.update_prices(prices)

        total = self.cash
        for position in self.positions.values():
            total += position.market_value
        return total

    def record_equity(self, timestamp: datetime, prices: Optional[dict[str, Decimal]] = None) -> None:
        """Record equity curve point.

        Args:
            timestamp: Timestamp
            prices: Current prices
        """
        equity = self.get_total_value(prices)
        self.equity_curve.append((timestamp, equity))

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame.

        Returns:
            DataFrame with columns: timestamp, equity
        """
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])

        timestamps, equities = zip(*self.equity_curve)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "equity": equities,
            }
        )

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame.

        Returns:
            DataFrame with trade records
        """
        if not self.trades:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "side", "quantity", "price", "cost", "pnl"]
            )

        return pd.DataFrame(
            {
                "timestamp": [t.timestamp for t in self.trades],
                "symbol": [t.symbol for t in self.trades],
                "side": [t.side for t in self.trades],
                "quantity": [t.quantity for t in self.trades],
                "price": [t.price for t in self.trades],
                "cost": [t.cost for t in self.trades],
                "pnl": [t.pnl for t in self.trades],
            }
        )
