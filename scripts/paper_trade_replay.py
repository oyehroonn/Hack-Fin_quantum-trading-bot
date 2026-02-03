"""Paper trade replay script - runs for 1 hour of replay."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from core.interfaces import Model
from core.types import OrderSide, OrderType, Signal
from execution.live_data import ReplayDataSource
from execution.orchestrator import ExecutionOrchestrator
from execution.paper_broker import PaperBroker
from features.feature_store import FeatureStore
from infra.journal import Journal
from risk.rules import AdvancedRiskManager, RiskLimits

# Dummy model for demo
class DummyModel(Model):
    async def predict(self, features: dict, symbol: str) -> Signal:
        import random
        from core.types import Signal, OrderSide

        strength = Decimal(str(random.uniform(-0.8, 0.8)))
        side = OrderSide.BUY if strength > 0 else OrderSide.SELL

        return Signal(
            symbol=symbol,
            timestamp=datetime.now(),
            side=side,
            strength=strength,
            confidence=Decimal("0.7"),
        )
    
    async def train(self, features: list[dict], labels: list[float]) -> None:
        """Train the model (not implemented for dummy)."""
        pass


async def main() -> None:
    """Run paper trade replay."""
    from loguru import logger

    # Configuration
    symbol = "AAPL"
    asset_class = "equities"
    timeframe = "1m"
    initial_cash = Decimal("100000")
    speedup_factor = 60.0  # 60x speed (1 hour of data in 1 minute)

    # Calculate date range (1 hour of 1-minute bars)
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=1)

    logger.info(f"Starting paper trade replay for {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Speedup: {speedup_factor}x")

    # Initialize components
    data_source = ReplayDataSource(
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        speedup_factor=speedup_factor,
    )

    feature_store = FeatureStore(cache_dir="data/features")
    model = DummyModel()
    broker = PaperBroker(initial_cash=initial_cash)
    journal = Journal(db_path="data/journal.db")

    # Risk limits
    risk_limits = RiskLimits(
        max_position_size=Decimal("50000"),  # $50k max position
        max_leverage=Decimal("2.0"),  # 2x max leverage
        max_daily_loss=Decimal("0.05"),  # 5% max daily loss
        max_drawdown=Decimal("0.20"),  # 20% max drawdown
        cooldown_after_loss_seconds=300,  # 5 min cooldown
    )

    risk_manager = AdvancedRiskManager(risk_limits, initial_cash)

    # Create orchestrator
    orchestrator = ExecutionOrchestrator(
        data_source=data_source,
        feature_store=feature_store,
        model=model,
        broker=broker,
        risk_manager=risk_manager,
        journal=journal,
        symbols=[symbol],
        update_interval_seconds=1.0,
    )

    # Start stats reporting task
    async def report_stats():
        while orchestrator.is_running:
            await asyncio.sleep(60.0)  # Every minute
            portfolio_state = await broker.get_portfolio_state()
            logger.info(
                f"Portfolio: Cash=${portfolio_state.cash:,.2f}, "
                f"Total=${portfolio_state.total_value:,.2f}, "
                f"PnL=${portfolio_state.realized_pnl + portfolio_state.unrealized_pnl:,.2f}, "
                f"Positions={len(portfolio_state.positions)}"
            )

    # Run orchestrator and stats reporter
    stats_task = asyncio.create_task(report_stats())
    orchestrator_task = asyncio.create_task(orchestrator.run())

    try:
        await orchestrator_task
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        orchestrator.stop()
    finally:
        stats_task.cancel()
        portfolio_state = await broker.get_portfolio_state()
        logger.info(f"\nFinal Portfolio State:")
        logger.info(f"  Cash: ${portfolio_state.cash:,.2f}")
        logger.info(f"  Total Value: ${portfolio_state.total_value:,.2f}")
        logger.info(f"  Realized PnL: ${portfolio_state.realized_pnl:,.2f}")
        logger.info(f"  Unrealized PnL: ${portfolio_state.unrealized_pnl:,.2f}")
        logger.info(f"  Positions: {len(portfolio_state.positions)}")


if __name__ == "__main__":
    asyncio.run(main())
