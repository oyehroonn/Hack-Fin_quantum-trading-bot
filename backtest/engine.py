"""Event-driven backtest engine."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd
from loguru import logger

from backtest.accounting import Portfolio
from backtest.broker_sim import SimBroker
from backtest.cost_models import CostModel
from backtest.strategy import Strategy


class BacktestEngine:
    """Event-driven backtest engine."""

    def __init__(
        self,
        initial_cash: Decimal,
        strategy: Strategy,
        cost_model: Optional[CostModel] = None,
        symbols: Optional[list[str]] = None,
    ) -> None:
        """Initialize backtest engine.

        Args:
            initial_cash: Initial cash
            strategy: Strategy to backtest
            cost_model: Cost model for broker
            symbols: List of symbols to trade
        """
        self.initial_cash = initial_cash
        self.strategy = strategy
        self.symbols = symbols or []
        self.broker = SimBroker(cost_model=cost_model)
        self.portfolio = Portfolio(initial_cash=initial_cash)

        # Initialize strategy
        if self.symbols:
            self.strategy.on_init(self.symbols)

    def run(
        self,
        data: pd.DataFrame,  # Multi-index (timestamp, symbol) or single symbol
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Portfolio:
        """Run backtest.

        Args:
            data: DataFrame with bars (multi-index or single symbol)
            start: Start date (optional)
            end: End date (optional)

        Returns:
            Portfolio with results
        """
        # Prepare data
        if isinstance(data.index, pd.MultiIndex):
            # Multi-index (timestamp, symbol)
            data = data.sort_index()
        else:
            # Single symbol - convert to multi-index
            if "symbol" not in data.columns:
                raise ValueError("Single symbol data must have 'symbol' column")
            data = data.set_index(["timestamp", "symbol"])

        # Filter by date range
        if start:
            data = data[data.index.get_level_values(0) >= start]
        if end:
            data = data[data.index.get_level_values(0) <= end]

        # Get unique timestamps
        timestamps = sorted(data.index.get_level_values(0).unique())

        logger.info(f"Running backtest from {timestamps[0]} to {timestamps[-1]} ({len(timestamps)} bars)")

        # Iterate through time
        for timestamp in timestamps:
            # Get bars for this timestamp
            bars_data = data.loc[timestamp]

            # Convert to dictionary format
            bars: dict[str, pd.Series] = {}
            prices: dict[str, Decimal] = {}

            # Handle multi-index DataFrame: data.loc[timestamp] returns DataFrame with symbol as index
            if isinstance(bars_data, pd.DataFrame):
                # Iterate over rows (each row is a symbol)
                for symbol, row in bars_data.iterrows():
                    symbol_str = str(symbol)
                    bars[symbol_str] = row
                    # Access values by column name
                    close_val = row.get("close", row.get("Close", 0))
                    prices[symbol_str] = Decimal(str(close_val))
            elif isinstance(bars_data, pd.Series):
                # Single symbol case - Series with column names as index
                # Get symbol from original data index or use first symbol
                if isinstance(data.index, pd.MultiIndex):
                    # Find symbol for this timestamp
                    matching = data.index[data.index.get_level_values(0) == timestamp]
                    if len(matching) > 0:
                        symbol = str(matching.get_level_values(1)[0])
                    else:
                        symbol = self.symbols[0] if self.symbols else "UNKNOWN"
                else:
                    symbol = self.symbols[0] if self.symbols else "UNKNOWN"
                
                bars[symbol] = bars_data
                close_val = bars_data.get("close", bars_data.get("Close", 0))
                prices[symbol] = Decimal(str(close_val))
            else:
                logger.warning(f"Unexpected bars_data type: {type(bars_data)}")
                continue

            # Update portfolio prices
            self.portfolio.update_prices(prices)

            # Get strategy signals
            signals = self.strategy.on_bar(timestamp, bars, self.portfolio)

            # Process signals
            if isinstance(signals, dict):
                # Weight-based signals
                self._process_weights(timestamp, signals, prices, bars)
            elif isinstance(signals, list):
                # Order-based signals
                self._process_orders(timestamp, signals, bars)

            # Process broker (fill orders)
            bar_dict = {}
            for symbol, bar in bars.items():
                # Handle both Series (with index) and dict-like access
                if isinstance(bar, pd.Series):
                    open_val = bar.get("open") if "open" in bar.index else bar.get("Open", 0)
                    high_val = bar.get("high") if "high" in bar.index else bar.get("High", 0)
                    low_val = bar.get("low") if "low" in bar.index else bar.get("Low", 0)
                    close_val = bar.get("close") if "close" in bar.index else bar.get("Close", 0)
                    volume_val = bar.get("volume") if "volume" in bar.index else bar.get("Volume", 1)
                else:
                    open_val = bar.get("open", bar.get("Open", 0))
                    high_val = bar.get("high", bar.get("High", 0))
                    low_val = bar.get("low", bar.get("Low", 0))
                    close_val = bar.get("close", bar.get("Close", 0))
                    volume_val = bar.get("volume", bar.get("Volume", 1))
                
                bar_dict[symbol] = {
                    "open": float(open_val),
                    "high": float(high_val),
                    "low": float(low_val),
                    "close": float(close_val),
                    "volume": float(volume_val),
                }

            fills = self.broker.process_bar(timestamp, bar_dict)

            # Apply fills to portfolio
            for fill in fills:
                self.portfolio.add_fill(
                    timestamp=fill.timestamp,
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=fill.quantity,
                    price=fill.price,
                    cost=fill.cost,
                )

            # Record equity
            self.portfolio.record_equity(timestamp, prices)

        # Finalize
        self.strategy.on_finish()

        logger.info(f"Backtest complete. Final equity: ${self.portfolio.get_total_value():,.2f}")

        return self.portfolio

    def _process_weights(
        self,
        timestamp: datetime,
        weights: dict[str, Decimal],
        prices: dict[str, Decimal],
        bars: dict[str, pd.Series],
    ) -> None:
        """Process weight-based signals.

        Args:
            timestamp: Current timestamp
            weights: Target weights {symbol: weight}
            prices: Current prices
            bars: Current bars
        """
        total_value = self.portfolio.get_total_value(prices)

        for symbol, target_weight in weights.items():
            if symbol not in prices:
                continue

            # Calculate target value
            target_value = total_value * target_weight

            # Get current position value
            current_value = Decimal("0")
            if symbol in self.portfolio.positions:
                current_value = self.portfolio.positions[symbol].market_value

            # Calculate required trade
            trade_value = target_value - current_value

            if abs(trade_value) < Decimal("10"):  # Minimum trade size
                continue

            # Create order
            price = prices[symbol]
            quantity = trade_value / price

            side = "BUY" if quantity > 0 else "SELL"
            order_id = f"{timestamp}_{symbol}_{side}"

            self.broker.submit_order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=abs(quantity),
                order_type="MARKET",
                timestamp=timestamp,
            )

    def _process_orders(
        self,
        timestamp: datetime,
        orders: list,
        bars: dict[str, pd.Series],
    ) -> None:
        """Process order-based signals.

        Args:
            timestamp: Current timestamp
            orders: List of Order objects
            bars: Current bars
        """
        from backtest.broker_sim import Order

        for order in orders:
            if isinstance(order, Order):
                self.broker.submit_order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    timestamp=timestamp,
                )
