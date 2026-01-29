#!/usr/bin/env python3
"""Demo runner script that executes a vertical slice end-to-end."""

import asyncio
from decimal import Decimal
from pathlib import Path

from data.dummy_source import DummyDataSource
from execution.paper_broker import PaperBroker
from features.dummy_model import DummyModel
from features.simple_store import SimpleFeatureStore
from infra.config import load_config
from infra.logging import setup_logging, set_correlation_id
from risk.simple_risk import SimpleRiskManager
from core.types import Order, OrderSide, OrderType, OrderStatus

from loguru import logger


async def run_demo(num_steps: int = 1000) -> None:
    """Run the demo trading system."""
    # Setup
    set_correlation_id("demo-run")
    setup_logging(level="INFO")

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    logger.info(f"Loaded config: {config}")

    # Initialize components
    data_source = DummyDataSource(
        symbols=config.symbols,
        initial_price=100.0,
        seed=42,
    )
    feature_store = SimpleFeatureStore(sma_period=20)
    model = DummyModel(seed=42)
    broker = PaperBroker(
        initial_cash=Decimal(str(config.initial_cash)),
        slippage_bps=config.slippage_bps,
        commission_bps=config.commission_bps,
    )
    risk_manager = SimpleRiskManager(
        max_position_size=Decimal(str(config.max_position_size)),
        max_leverage=Decimal(str(config.max_leverage)),
    )

    logger.info("Starting demo run...")

    # Run simulation - process first symbol for simplicity
    # In production, you'd want to process multiple symbols concurrently
    symbol = config.symbols[0]
    step = 0
    
    bar_stream = data_source.get_bars(symbol)
    async for bar in bar_stream:
        if step >= num_steps:
            break

        # Update broker with current price
        broker.set_current_price(symbol, bar.close)

        # Compute features
        features = await feature_store.compute_features(symbol, [bar])
        if not features:
            step += 1
            continue

        # Get portfolio state
        portfolio_state = await broker.get_portfolio_state()

        # Generate signal
        signal = await model.predict(features, symbol)

        # Only trade if confidence is high enough
        if signal.confidence < Decimal("0.3"):
            step += 1
            continue

        # Determine order quantity (simple: fixed size based on confidence)
        base_quantity = Decimal("10")
        quantity = base_quantity * signal.confidence

        # Create order
        order = Order(
            order_id=f"order_{step}_{symbol}",
            symbol=symbol,
            side=signal.side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            timestamp=bar.timestamp,
        )

        # Validate with risk manager
        is_valid, reason = await risk_manager.validate_order(order, portfolio_state)
        if not is_valid:
            logger.warning(f"Order rejected: {reason}")
            step += 1
            continue

        # Submit order
        order_id = await broker.submit_order(order)

        # Check order status
        order_status = await broker.get_order_status(order_id)
        if order_status.status == OrderStatus.FILLED:
            logger.info(
                f"Order filled: {order.side} {order.quantity} {symbol} @ "
                f"{order_status.status}"
            )

        # Check portfolio limits
        portfolio_state = await broker.get_portfolio_state()
        is_valid, reason = await risk_manager.check_limits(portfolio_state)
        if not is_valid:
            logger.warning(f"Portfolio limit violation: {reason}")

        step += 1

    # Print summary
    logger.info("=" * 80)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 80)

    portfolio_state = await broker.get_portfolio_state()
    logger.info(f"Initial Cash: ${config.initial_cash:,.2f}")
    logger.info(f"Final Cash: ${portfolio_state.cash:,.2f}")
    logger.info(f"Total Value: ${portfolio_state.total_value:,.2f}")
    logger.info(f"Unrealized PnL: ${portfolio_state.unrealized_pnl:,.2f}")
    logger.info(f"Realized PnL: ${portfolio_state.realized_pnl:,.2f}")
    logger.info(
        f"Total PnL: ${portfolio_state.unrealized_pnl + portfolio_state.realized_pnl:,.2f}"
    )
    logger.info(
        f"Return: {((portfolio_state.total_value / Decimal(str(config.initial_cash))) - 1) * 100:.2f}%"
    )

    logger.info("\nPositions:")
    for symbol, position in portfolio_state.positions.items():
        logger.info(
            f"  {symbol}: {position.quantity} @ ${position.avg_price:.2f} "
            f"(Unrealized PnL: ${position.unrealized_pnl:.2f})"
        )

    logger.info(f"\nTotal Orders: {len(broker.orders)}")
    logger.info(f"Filled Orders: {sum(1 for o in broker.orders.values() if o.status == OrderStatus.FILLED)}")


if __name__ == "__main__":
    asyncio.run(run_demo(num_steps=1000))
