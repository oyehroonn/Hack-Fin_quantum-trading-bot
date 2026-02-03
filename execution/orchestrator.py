"""Live execution orchestrator with safety rails."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional

from core.interfaces import Broker, DataSource, Model, RiskManager
from core.types import Order, OrderSide, OrderType, PortfolioState, Signal
from execution.live_data import ReplayDataSource
from features.feature_store import FeatureStore
from infra.journal import EventType, Journal
from loguru import logger


class ExecutionOrchestrator:
    """Orchestrates live trading execution with safety rails."""

    def __init__(
        self,
        data_source: DataSource,
        feature_store: FeatureStore,
        model: Model,
        broker: Broker,
        risk_manager: RiskManager,
        journal: Journal,
        symbols: list[str],
        update_interval_seconds: float = 60.0,
    ) -> None:
        """Initialize orchestrator.

        Args:
            data_source: Data source for market data
            feature_store: Feature store for computing features
            model: Prediction model
            broker: Broker for order execution
            risk_manager: Risk manager
            journal: Event journal
            symbols: List of symbols to trade
            update_interval_seconds: How often to check for new data
        """
        self.data_source = data_source
        self.feature_store = feature_store
        self.model = model
        self.broker = broker
        self.risk_manager = risk_manager
        self.journal = journal
        self.symbols = symbols
        self.update_interval = update_interval_seconds
        self.is_running = False
        self._latest_bars: dict[str, list] = {}  # symbol -> list of recent bars

    async def run(self) -> None:
        """Run the orchestrator loop."""
        self.is_running = True
        logger.info("Starting execution orchestrator")

        try:
            async for bar in self.data_source.get_bars(self.symbols[0]):
                if not self.is_running:
                    break

                # Update latest bars
                if bar.symbol not in self._latest_bars:
                    self._latest_bars[bar.symbol] = []
                self._latest_bars[bar.symbol].append(bar)
                # Keep only last 100 bars
                if len(self._latest_bars[bar.symbol]) > 100:
                    self._latest_bars[bar.symbol] = self._latest_bars[bar.symbol][-100:]

                # Process bar
                await self._process_bar(bar)

                # Check risk limits
                portfolio_state = await self.broker.get_portfolio_state()
                is_ok, reason = await self.risk_manager.check_limits(portfolio_state)
                if not is_ok:
                    logger.error(f"Risk limit violation: {reason}")
                    self.journal.log_event(
                        EventType.RISK_VIOLATION,
                        {"reason": reason, "portfolio_state": self._serialize_portfolio_state(portfolio_state)},
                    )
                    # Could trigger circuit breaker here

        except Exception as e:
            logger.exception(f"Error in orchestrator loop: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("Execution orchestrator stopped")

    async def _process_bar(self, bar) -> None:
        """Process a new bar."""
        symbol = bar.symbol

        # Convert bar to DataFrame for feature computation
        import pandas as pd

        bars_df = pd.DataFrame(
            {
                "timestamp": [b.timestamp for b in self._latest_bars[symbol]],
                "open": [float(b.open) for b in self._latest_bars[symbol]],
                "high": [float(b.high) for b in self._latest_bars[symbol]],
                "low": [float(b.low) for b in self._latest_bars[symbol]],
                "close": [float(b.close) for b in self._latest_bars[symbol]],
                "volume": [float(b.volume) for b in self._latest_bars[symbol]],
            }
        )
        bars_df = bars_df.set_index("timestamp")

        # Compute features
        feature_config = {
            "technical": {"sma": [10, 20], "rsi": [14]},
            "statistical": {"returns": [1]},
        }
        features_df = self.feature_store.compute_features(
            bars_df, feature_config, symbol=symbol, timeframe="1m", use_cache=False
        )

        if features_df.empty:
            return

        # Get latest features
        latest_features = features_df.iloc[-1].to_dict()

        # Get model prediction
        signal = await self.model.predict(latest_features, symbol)

        # Log signal
        self.journal.log_event(
            EventType.SIGNAL,
            {
                "symbol": signal.symbol,
                "timestamp": signal.timestamp.isoformat(),
                "side": signal.side.value,
                "strength": str(signal.strength),
                "confidence": str(signal.confidence),
            },
        )

        # Convert signal to order
        order = await self._signal_to_order(signal, bar)

        if order is None:
            return

        # Validate with risk manager
        portfolio_state = await self.broker.get_portfolio_state()
        is_valid, reason = await self.risk_manager.validate_order(order, portfolio_state)

        if not is_valid:
            logger.warning(f"Order rejected by risk manager: {reason}")
            return

        # Submit order
        order_id = await self.broker.submit_order(order)

        # Log order
        self.journal.log_event(
            EventType.ORDER,
            {
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": str(order.quantity),
                "order_type": order.order_type.value,
                "timestamp": order.timestamp.isoformat() if order.timestamp else None,
            },
        )

        # Check for fills (async, non-blocking)
        asyncio.create_task(self._check_fills(order_id))

        # Log portfolio state periodically
        if len(self._latest_bars[symbol]) % 10 == 0:  # Every 10 bars
            await self._log_portfolio_state()

    async def _signal_to_order(self, signal: Signal, bar) -> Optional[Order]:
        """Convert signal to order."""
        import uuid

        # Simple conversion: use signal strength to determine quantity
        # In production, this would be more sophisticated
        base_quantity = Decimal("1.0")
        quantity = base_quantity * abs(signal.strength)

        if signal.strength > 0.5:
            side = OrderSide.BUY
        elif signal.strength < -0.5:
            side = OrderSide.SELL
        else:
            # Signal too weak
            return None

        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            timestamp=datetime.now(),
        )

        return order

    async def _check_fills(self, order_id: str) -> None:
        """Check for order fills."""
        await asyncio.sleep(1.0)  # Wait a bit for fill

        fills = await self.broker.get_fills(order_id)
        for fill in fills:
            self.journal.log_event(
                EventType.FILL,
                {
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": str(fill.quantity),
                    "price": str(fill.price),
                    "fee": str(fill.fee),
                    "timestamp": fill.timestamp.isoformat(),
                },
            )

    async def _log_portfolio_state(self) -> None:
        """Log current portfolio state."""
        portfolio_state = await self.broker.get_portfolio_state()
        self.journal.log_event(
            EventType.PORTFOLIO_STATE,
            self._serialize_portfolio_state(portfolio_state),
        )

    def _serialize_portfolio_state(self, state: PortfolioState) -> dict:
        """Serialize portfolio state for journal."""
        return {
            "timestamp": state.timestamp.isoformat(),
            "cash": str(state.cash),
            "total_value": str(state.total_value),
            "unrealized_pnl": str(state.unrealized_pnl),
            "realized_pnl": str(state.realized_pnl),
            "positions": {
                symbol: {
                    "quantity": str(pos.quantity),
                    "avg_price": str(pos.avg_price),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                    "realized_pnl": str(pos.realized_pnl),
                }
                for symbol, pos in state.positions.items()
            },
        }

    def stop(self) -> None:
        """Stop the orchestrator."""
        self.is_running = False
        logger.info("Stopping execution orchestrator")
