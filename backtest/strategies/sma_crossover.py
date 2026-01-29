"""SMA crossover strategy with volatility scaling."""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from backtest.strategy import WeightBasedStrategy


class SMACrossoverStrategy(WeightBasedStrategy):
    """SMA crossover strategy with volatility scaling."""

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        vol_period: int = 20,
        vol_target: float = 0.15,  # 15% annualized volatility target
        max_weight: float = 1.0,
    ) -> None:
        """Initialize SMA crossover strategy.

        Args:
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            vol_period: Volatility lookback period
            vol_target: Target annualized volatility
            max_weight: Maximum position weight per symbol
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_period = vol_period
        self.vol_target = vol_target
        self.max_weight = max_weight

        self.price_history: dict[str, list[float]] = {}
        self.sma_fast: dict[str, float] = {}
        self.sma_slow: dict[str, float] = {}
        self.volatility: dict[str, float] = {}

    def on_init(self, symbols: list[str]) -> None:
        """Initialize price history."""
        for symbol in symbols:
            self.price_history[symbol] = []

    def on_bar(
        self,
        timestamp: datetime,
        bars: dict[str, pd.Series],
        portfolio,
    ) -> dict[str, Decimal]:
        """Generate target weights based on SMA crossover."""
        weights = {}

        for symbol, bar in bars.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            # Get close price
            close = float(bar.get("close", bar.get("Close", 0)))
            if close == 0:
                continue

            # Update price history
            self.price_history[symbol].append(close)

            # Need enough history
            if len(self.price_history[symbol]) < self.slow_period:
                continue

            # Calculate SMAs
            prices = np.array(self.price_history[symbol][-self.slow_period:])
            self.sma_fast[symbol] = np.mean(prices[-self.fast_period:])
            self.sma_slow[symbol] = np.mean(prices)

            # Calculate volatility (annualized)
            if len(prices) >= self.vol_period:
                returns = np.diff(prices[-self.vol_period:]) / prices[-self.vol_period:-1]
                vol = np.std(returns) * np.sqrt(252)  # Annualized (assuming daily)
                self.volatility[symbol] = vol
            else:
                self.volatility[symbol] = self.vol_target  # Default

            # Generate signal
            signal = 0.0
            if self.sma_fast[symbol] > self.sma_slow[symbol]:
                signal = 1.0  # Long signal
            elif self.sma_fast[symbol] < self.sma_slow[symbol]:
                signal = -1.0  # Short signal

            # Volatility scaling
            if self.volatility[symbol] > 0:
                vol_scale = min(self.vol_target / self.volatility[symbol], 2.0)  # Cap at 2x
                signal *= vol_scale

            # Convert to weight
            weight = signal * self.max_weight
            weights[symbol] = Decimal(str(weight))

        return weights
