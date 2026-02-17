"""Real-time risk overlay for the execution pipeline.

Sits between the allocator and the broker, enforcing hard limits
and modifying orders before they reach the market.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from core.errors import CircuitBreakerTrippedError, RiskLimitBreachedError
from core.types import Alert, AlertSeverity, AlertType, Order, OrderSide, PortfolioState, RiskLimits


class CircuitBreaker:
    """Trading circuit breaker that halts execution when limits are breached.

    Triggers:
      - Max daily loss exceeded
      - Max drawdown exceeded
      - Rapid loss sequence (cooldown)
      - Manual trip
    """

    def __init__(
        self,
        max_daily_loss: float = 0.03,
        max_drawdown: float = 0.20,
        cooldown_minutes: int = 60,
        max_consecutive_losses: int = 5,
    ) -> None:
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.cooldown_minutes = cooldown_minutes
        self.max_consecutive_losses = max_consecutive_losses

        self._is_tripped = False
        self._trip_time: Optional[datetime] = None
        self._trip_reason: str = ""
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._last_reset_date: Optional[datetime] = None

    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is active (with auto-cooldown reset)."""
        if self._is_tripped and self._trip_time:
            elapsed = datetime.now() - self._trip_time
            if elapsed > timedelta(minutes=self.cooldown_minutes):
                logger.info(f"Circuit breaker cooldown elapsed, resetting")
                self.reset()
                return False
        return self._is_tripped

    def check(self, equity: float, daily_pnl: float) -> Optional[Alert]:
        """Check risk conditions and trip if needed.

        Args:
            equity: Current portfolio equity
            daily_pnl: Today's PnL as fraction of starting equity

        Returns:
            Alert if tripped, None otherwise
        """
        now = datetime.now()

        # Reset daily PnL at day change
        if self._last_reset_date is None or now.date() != self._last_reset_date.date():
            self._daily_pnl = 0.0
            self._last_reset_date = now

        self._daily_pnl = daily_pnl

        # Track peak equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Check daily loss
        if abs(daily_pnl) > self.max_daily_loss and daily_pnl < 0:
            return self._trip(f"Daily loss {daily_pnl:.4f} exceeds limit {self.max_daily_loss}")

        # Check drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self.max_drawdown:
                return self._trip(f"Drawdown {drawdown:.4f} exceeds limit {self.max_drawdown}")

        # Check consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            return self._trip(f"Consecutive losses ({self._consecutive_losses}) hit limit")

        return None

    def record_trade_result(self, pnl: float) -> None:
        """Record trade result for consecutive loss tracking."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def trip(self, reason: str) -> Alert:
        """Manually trip the circuit breaker."""
        return self._trip(reason)

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._is_tripped = False
        self._trip_time = None
        self._trip_reason = ""
        self._consecutive_losses = 0

    def _trip(self, reason: str) -> Alert:
        self._is_tripped = True
        self._trip_time = datetime.now()
        self._trip_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")

        return Alert(
            alert_id=f"cb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.CIRCUIT_BREAKER,
            message=reason,
            timestamp=datetime.now(),
            payload={
                "daily_pnl": self._daily_pnl,
                "peak_equity": self._peak_equity,
                "consecutive_losses": self._consecutive_losses,
            },
        )


class RiskOverlay:
    """Pre-trade risk checks applied to every order before execution.

    Enforces:
      - Max position size per symbol
      - Max portfolio leverage
      - Max single-order size
      - Symbol exposure limits
      - Circuit breaker check
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            max_daily_loss=float(self.limits.max_daily_loss),
            max_drawdown=float(self.limits.max_drawdown),
            cooldown_minutes=self.limits.cooldown_minutes,
        )

    def validate_order(
        self,
        order: Order,
        portfolio: PortfolioState,
        price: Decimal,
    ) -> tuple[bool, str, Optional[Order]]:
        """Validate and potentially modify an order.

        Args:
            order: Proposed order
            portfolio: Current portfolio state
            price: Current market price

        Returns:
            (is_valid, reason, modified_order)
            modified_order may have reduced quantity
        """
        # Circuit breaker check
        if self.circuit_breaker.is_tripped:
            return False, f"Circuit breaker active: {self.circuit_breaker._trip_reason}", None

        # Max position size
        order_value = order.quantity * price
        if order_value > self.limits.max_position_size:
            # Reduce to limit
            new_qty = self.limits.max_position_size / price
            order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=new_qty.quantize(Decimal("0.00000001")),
                timestamp=order.timestamp,
                price=order.price,
            )
            logger.warning(
                f"Order {order.symbol} reduced: "
                f"value {order_value} > limit {self.limits.max_position_size}"
            )

        # Leverage check
        if portfolio.equity > 0:
            total_exposure = sum(
                abs(float(pos.quantity * price))
                for pos in portfolio.positions
            )
            new_exposure = total_exposure + float(order.quantity * price)
            leverage = Decimal(str(new_exposure)) / portfolio.equity

            if leverage > self.limits.max_leverage:
                return False, f"Leverage {leverage:.2f} would exceed limit {self.limits.max_leverage}", None

        # Symbol exposure
        if portfolio.equity > 0:
            symbol_exposure = float(order.quantity * price) / float(portfolio.equity)
            if Decimal(str(symbol_exposure)) > self.limits.max_symbol_exposure:
                # Reduce to limit
                max_value = float(self.limits.max_symbol_exposure) * float(portfolio.equity)
                new_qty = Decimal(str(max_value)) / price
                order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=new_qty.quantize(Decimal("0.00000001")),
                    timestamp=order.timestamp,
                    price=order.price,
                )

        return True, "passed", order

    def check_portfolio(self, portfolio: PortfolioState, daily_pnl: float = 0.0) -> list[Alert]:
        """Check portfolio-level risk and return any alerts."""
        alerts = []

        # Circuit breaker check
        cb_alert = self.circuit_breaker.check(float(portfolio.equity), daily_pnl)
        if cb_alert:
            alerts.append(cb_alert)

        return alerts
