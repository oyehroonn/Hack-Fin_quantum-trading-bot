"""Strategy interface and base classes."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd


class Strategy(ABC):
    """Base strategy interface."""

    @abstractmethod
    def on_bar(
        self,
        timestamp: datetime,
        bars: dict[str, pd.Series],  # symbol -> bar data
        portfolio: "Portfolio",  # noqa: F821
    ) -> dict[str, Decimal] | list:  # Returns either weights dict or Order list
        """Process a bar and generate signals.

        Args:
            timestamp: Bar timestamp
            bars: Dictionary of bar data by symbol
            portfolio: Current portfolio state

        Returns:
            Either:
            - Dictionary of target weights {symbol: weight} (0-1, sum should be <= 1)
            - List of Order objects for discrete order mode
        """
        ...

    def on_init(self, symbols: list[str]) -> None:
        """Initialize strategy (called once at start).

        Args:
            symbols: List of symbols to trade
        """
        pass

    def on_finish(self) -> None:
        """Called at end of backtest."""
        pass


class OrderBasedStrategy(Strategy):
    """Strategy that generates discrete orders."""

    def on_bar(
        self,
        timestamp: datetime,
        bars: dict[str, pd.Series],
        portfolio,  # Portfolio type
    ) -> list:  # List of Order objects
        """Generate orders (to be implemented by subclass)."""
        return []


class WeightBasedStrategy(Strategy):
    """Strategy that generates target weights."""

    def on_bar(
        self,
        timestamp: datetime,
        bars: dict[str, pd.Series],
        portfolio,  # Portfolio type
    ) -> dict[str, Decimal]:
        """Generate target weights (to be implemented by subclass)."""
        return {}
