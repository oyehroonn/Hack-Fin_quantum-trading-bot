"""Advanced risk management rules with circuit breakers."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from core.interfaces import RiskManager
from core.types import Order, OrderSide, PortfolioState
from loguru import logger


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_size: Decimal
    max_leverage: Decimal
    max_daily_loss: Decimal  # As fraction of initial equity (e.g., 0.05 = 5%)
    max_drawdown: Decimal  # As fraction of peak equity (e.g., 0.20 = 20%)
    cooldown_after_loss_seconds: int = 3600  # 1 hour default


class MaxPositionSize:
    """Enforce maximum position size per symbol."""

    def __init__(self, max_size: Decimal) -> None:
        """Initialize.

        Args:
            max_size: Maximum position size in notional value
        """
        self.max_size = max_size

    def check(
        self, order: Order, portfolio_state: PortfolioState, current_price: Decimal
    ) -> tuple[bool, Optional[str]]:
        """Check position size limit."""
        current_pos = portfolio_state.positions.get(order.symbol)
        if current_pos:
            current_qty = current_pos.quantity
        else:
            current_qty = Decimal("0")

        if order.side == OrderSide.BUY:
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        new_notional = abs(new_qty) * current_price
        if new_notional > self.max_size:
            return (
                False,
                f"Position size {new_notional} exceeds max {self.max_size}",
            )
        return (True, None)


class MaxLeverage:
    """Enforce maximum leverage."""

    def __init__(self, max_leverage: Decimal) -> None:
        """Initialize.

        Args:
            max_leverage: Maximum leverage (e.g., 2.0 for 2x)
        """
        self.max_leverage = max_leverage

    def check(
        self, order: Order, portfolio_state: PortfolioState, current_price: Decimal
    ) -> tuple[bool, Optional[str]]:
        """Check leverage limit."""
        # Calculate total position value
        total_notional = sum(
            abs(pos.quantity) * pos.avg_price
            for pos in portfolio_state.positions.values()
        )

        # Add new order
        current_pos = portfolio_state.positions.get(order.symbol)
        if current_pos:
            total_notional -= abs(current_pos.quantity) * current_pos.avg_price

        if order.side == OrderSide.BUY:
            new_qty = (current_pos.quantity if current_pos else Decimal("0")) + order.quantity
        else:
            new_qty = (current_pos.quantity if current_pos else Decimal("0")) - order.quantity

        total_notional += abs(new_qty) * current_price

        equity = portfolio_state.total_value
        if equity > 0:
            leverage = total_notional / equity
            if leverage > self.max_leverage:
                return (
                    False,
                    f"Leverage {leverage:.2f} exceeds max {self.max_leverage}",
                )
        return (True, None)


class MaxDailyLoss:
    """Enforce maximum daily loss."""

    def __init__(self, max_loss_pct: Decimal, initial_equity: Decimal) -> None:
        """Initialize.

        Args:
            max_loss_pct: Maximum daily loss as fraction (e.g., 0.05 = 5%)
            initial_equity: Initial equity at start of day
        """
        self.max_loss_pct = max_loss_pct
        self.initial_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.last_reset_date = datetime.now().date()

    def _reset_if_new_day(self) -> None:
        """Reset daily tracking if new day."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_start_equity = self.initial_equity
            self.last_reset_date = today

    def check(self, portfolio_state: PortfolioState) -> tuple[bool, Optional[str]]:
        """Check daily loss limit."""
        self._reset_if_new_day()

        current_equity = portfolio_state.total_value
        daily_loss = self.daily_start_equity - current_equity
        daily_loss_pct = daily_loss / self.daily_start_equity if self.daily_start_equity > 0 else Decimal("0")

        if daily_loss_pct > self.max_loss_pct:
            return (
                False,
                f"Daily loss {daily_loss_pct:.2%} exceeds max {self.max_loss_pct:.2%}",
            )
        return (True, None)


class MaxDrawdown:
    """Enforce maximum drawdown from peak."""

    def __init__(self, max_drawdown_pct: Decimal) -> None:
        """Initialize.

        Args:
            max_drawdown_pct: Maximum drawdown as fraction (e.g., 0.20 = 20%)
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity: Optional[Decimal] = None

    def check(self, portfolio_state: PortfolioState) -> tuple[bool, Optional[str]]:
        """Check drawdown limit."""
        current_equity = portfolio_state.total_value

        # Update peak
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown_pct:
                return (
                    False,
                    f"Drawdown {drawdown:.2%} exceeds max {self.max_drawdown_pct:.2%}",
                )
        return (True, None)


class CooldownAfterLoss:
    """Enforce cooldown period after a loss."""

    def __init__(self, cooldown_seconds: int) -> None:
        """Initialize.

        Args:
            cooldown_seconds: Cooldown period in seconds
        """
        self.cooldown_seconds = cooldown_seconds
        self.last_loss_time: Optional[datetime] = None

    def record_loss(self) -> None:
        """Record a loss event."""
        self.last_loss_time = datetime.now()

    def check(self) -> tuple[bool, Optional[str]]:
        """Check if cooldown period has passed."""
        if self.last_loss_time is None:
            return (True, None)

        elapsed = (datetime.now() - self.last_loss_time).total_seconds()
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            return (
                False,
                f"Cooldown active: {remaining:.0f} seconds remaining",
            )
        return (True, None)


class CircuitBreaker:
    """Circuit breaker that halts trading and optionally liquidates."""

    def __init__(
        self,
        auto_liquidate: bool = False,
        liquidate_on_breach: bool = True,
    ) -> None:
        """Initialize.

        Args:
            auto_liquidate: Whether to automatically liquidate positions
            liquidate_on_breach: Whether to liquidate when circuit breaker trips
        """
        self.auto_liquidate = auto_liquidate
        self.liquidate_on_breach = liquidate_on_breach
        self.is_tripped = False
        self.trip_reason: Optional[str] = None
        self.tripped_at: Optional[datetime] = None

    def trip(self, reason: str) -> None:
        """Trip the circuit breaker."""
        if not self.is_tripped:
            self.is_tripped = True
            self.trip_reason = reason
            self.tripped_at = datetime.now()
            logger.critical(f"Circuit breaker TRIPPED: {reason}")

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.is_tripped = False
        self.trip_reason = None
        self.tripped_at = None
        logger.info("Circuit breaker RESET")

    def check(self) -> tuple[bool, Optional[str]]:
        """Check if circuit breaker is tripped."""
        if self.is_tripped:
            return (False, f"Circuit breaker tripped: {self.trip_reason}")
        return (True, None)


class AdvancedRiskManager(RiskManager):
    """Advanced risk manager with multiple rules and circuit breaker."""

    def __init__(
        self,
        limits: RiskLimits,
        initial_equity: Decimal,
    ) -> None:
        """Initialize.

        Args:
            limits: Risk limit configuration
            initial_equity: Initial equity
        """
        self.limits = limits
        self.initial_equity = initial_equity

        # Initialize rules
        self.max_position = MaxPositionSize(limits.max_position_size)
        self.max_leverage = MaxLeverage(limits.max_leverage)
        self.max_daily_loss = MaxDailyLoss(limits.max_daily_loss, initial_equity)
        self.max_drawdown = MaxDrawdown(limits.max_drawdown)
        self.cooldown = CooldownAfterLoss(limits.cooldown_after_loss_seconds)
        self.circuit_breaker = CircuitBreaker()

    async def validate_order(
        self,
        order: Order,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Validate an order against all risk rules."""
        # Check circuit breaker first
        is_ok, reason = self.circuit_breaker.check()
        if not is_ok:
            return (False, reason)

        # Check cooldown
        is_ok, reason = self.cooldown.check()
        if not is_ok:
            return (False, reason)

        # Get current price (would come from market data in production)
        current_pos = portfolio_state.positions.get(order.symbol)
        current_price = (
            current_pos.avg_price if current_pos else Decimal("100.0")
        )  # Placeholder

        # Check position size
        is_ok, reason = self.max_position.check(order, portfolio_state, current_price)
        if not is_ok:
            return (False, reason)

        # Check leverage
        is_ok, reason = self.max_leverage.check(order, portfolio_state, current_price)
        if not is_ok:
            return (False, reason)

        return (True, None)

    async def check_limits(
        self,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Check if portfolio is within all risk limits."""
        # Check daily loss
        is_ok, reason = self.max_daily_loss.check(portfolio_state)
        if not is_ok:
            self.circuit_breaker.trip(f"Daily loss limit: {reason}")
            return (False, reason)

        # Check drawdown
        is_ok, reason = self.max_drawdown.check(portfolio_state)
        if not is_ok:
            self.circuit_breaker.trip(f"Drawdown limit: {reason}")
            return (False, reason)

        return (True, None)

    def record_loss(self) -> None:
        """Record a loss event (triggers cooldown)."""
        self.cooldown.record_loss()
