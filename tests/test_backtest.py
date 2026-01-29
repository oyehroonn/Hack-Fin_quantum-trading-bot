"""Tests for backtesting engine."""

from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from backtest.accounting import Portfolio, Position
from backtest.broker_sim import SimBroker, Order
from backtest.cost_models import FixedFee, PercentFee, SpreadSlippage, CompositeCostModel
from backtest.engine import BacktestEngine
from backtest.strategies.sma_crossover import SMACrossoverStrategy


def test_portfolio_position_updates() -> None:
    """Test portfolio position updates."""
    portfolio = Portfolio(initial_cash=Decimal("100000"))

    # Add a buy fill
    portfolio.add_fill(
        timestamp=datetime.now(),
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("10"),
        price=Decimal("100"),
        cost=Decimal("1"),
    )

    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].quantity == Decimal("10")
    assert portfolio.cash < Decimal("100000")


def test_portfolio_pnl_calculation() -> None:
    """Test PnL calculation."""
    portfolio = Portfolio(initial_cash=Decimal("100000"))

    # Buy at 100
    portfolio.add_fill(
        timestamp=datetime.now(),
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("10"),
        price=Decimal("100"),
        cost=Decimal("1"),
    )

    # Update price to 110
    portfolio.update_prices({"AAPL": Decimal("110")})

    position = portfolio.positions["AAPL"]
    assert position.unrealized_pnl == Decimal("100")  # 10 * (110 - 100)
    assert position.realized_pnl == Decimal("-1")  # Cost

    # Sell at 110
    portfolio.add_fill(
        timestamp=datetime.now(),
        symbol="AAPL",
        side="SELL",
        quantity=Decimal("10"),
        price=Decimal("110"),
        cost=Decimal("1"),
    )

    # Position should be closed
    assert "AAPL" not in portfolio.positions
    # Check that we have positive realized PnL from the trade
    trades_df = portfolio.get_trades_df()
    if not trades_df.empty:
        total_pnl = trades_df["pnl"].sum()
        assert total_pnl > Decimal("0")


def test_broker_market_order_fill() -> None:
    """Test broker market order fill."""
    broker = SimBroker(
        cost_model=CompositeCostModel(
            PercentFee(Decimal("0.001")),
            SpreadSlippage(Decimal("5.0")),
        )
    )

    order = broker.submit_order(
        order_id="test-1",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("10"),
        order_type="MARKET",
    )

    # Process bar
    bar = {
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 103.0,
        "volume": 1000.0,
    }

    fills = broker.process_bar(datetime.now(), {"AAPL": bar})

    assert len(fills) == 1
    assert fills[0].quantity > 0
    assert fills[0].cost > 0  # Should have fees


def test_broker_limit_order_fill() -> None:
    """Test broker limit order fill."""
    broker = SimBroker()

    # Buy limit order at 95
    order = broker.submit_order(
        order_id="test-2",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("10"),
        order_type="LIMIT",
        limit_price=Decimal("95"),
    )

    # Bar with low at 94 (should fill)
    bar = {
        "open": 100.0,
        "high": 105.0,
        "low": 94.0,
        "close": 103.0,
        "volume": 1000.0,
    }

    fills = broker.process_bar(datetime.now(), {"AAPL": bar})

    assert len(fills) == 1
    assert fills[0].price <= Decimal("95")

    # Bar with low at 96 (should not fill)
    order2 = broker.submit_order(
        order_id="test-3",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("10"),
        order_type="LIMIT",
        limit_price=Decimal("95"),
    )

    bar2 = {
        "open": 100.0,
        "high": 105.0,
        "low": 96.0,
        "close": 103.0,
        "volume": 1000.0,
    }

    fills2 = broker.process_bar(datetime.now(), {"AAPL": bar2})
    assert len(fills2) == 0  # Should not fill


def test_cost_models() -> None:
    """Test cost models."""
    # Fixed fee
    fixed = FixedFee(Decimal("1.0"))
    assert fixed.calculate_cost(Decimal("10"), Decimal("100"), "BUY") == Decimal("1.0")

    # Percent fee
    pct = PercentFee(Decimal("0.001"))  # 0.1%
    cost = pct.calculate_cost(Decimal("10"), Decimal("100"), "BUY")
    assert cost == Decimal("1.0")  # 10 * 100 * 0.001

    # Spread slippage
    spread = SpreadSlippage(Decimal("5.0"))  # 5 bps
    cost = spread.calculate_cost(Decimal("10"), Decimal("100"), "BUY")
    assert cost > Decimal("0")


def test_backtest_engine_simple() -> None:
    """Test simple backtest run."""
    # Create synthetic data
    dates = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "open": range(100, 100 + len(dates)),
            "high": range(101, 101 + len(dates)),
            "low": range(99, 99 + len(dates)),
            "close": range(100, 100 + len(dates)),
            "volume": [1000] * len(dates),
        }
    )
    data = data.set_index(["timestamp", "symbol"])

    # Simple strategy that does nothing
    class DummyStrategy:
        def on_init(self, symbols):
            pass

        def on_bar(self, timestamp, bars, portfolio):
            return {}

        def on_finish(self):
            pass

    strategy = DummyStrategy()
    engine = BacktestEngine(
        initial_cash=Decimal("100000"),
        strategy=strategy,
        symbols=["AAPL"],
    )

    portfolio = engine.run(data)

    assert portfolio is not None
    assert len(portfolio.equity_curve) > 0


def test_sma_crossover_strategy() -> None:
    """Test SMA crossover strategy."""
    strategy = SMACrossoverStrategy(fast_period=5, slow_period=10)

    # Create synthetic price data
    prices = [100 + i * 0.5 for i in range(20)]  # Upward trend
    bars = {}
    for i, price in enumerate(prices):
        bar = pd.Series(
            {
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000,
            }
        )
        bars["AAPL"] = bar

        strategy.on_init(["AAPL"])
        weights = strategy.on_bar(datetime.now(), bars, None)

        if i >= 10:  # Need enough history
            assert "AAPL" in weights or len(weights) == 0


def test_portfolio_equity_curve() -> None:
    """Test equity curve recording."""
    portfolio = Portfolio(initial_cash=Decimal("100000"))

    # Record equity at different times
    portfolio.record_equity(datetime(2024, 1, 1), {"AAPL": Decimal("100")})
    portfolio.record_equity(datetime(2024, 1, 2), {"AAPL": Decimal("110")})

    equity_df = portfolio.get_equity_curve_df()
    assert len(equity_df) >= 2
    assert "timestamp" in equity_df.columns
    assert "equity" in equity_df.columns
