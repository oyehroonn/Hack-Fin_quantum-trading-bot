"""Dummy data source that generates synthetic bars."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np

from core.interfaces import DataSource
from core.types import Bar, Tick


class DummyDataSource(DataSource):
    """Generates synthetic OHLCV bars using random walk."""

    def __init__(
        self,
        symbols: list[str],
        initial_price: float = 100.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize dummy data source.

        Args:
            symbols: List of symbols to generate data for
            initial_price: Initial price for all symbols
            seed: Random seed for reproducibility
        """
        self.symbols = symbols
        self.initial_price = initial_price
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_prices: dict[str, Decimal] = {
            symbol: Decimal(str(initial_price)) for symbol in symbols
        }
        self._start_time = datetime.now()

    async def get_bars(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> AsyncIterator[Bar]:
        """Generate synthetic bars for a symbol."""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not in data source")

        current_price = float(self._current_prices[symbol])
        current_time = self._start_time

        while True:
            # Random walk with drift
            drift = 0.0001  # Small upward drift
            volatility = 0.02  # 2% volatility
            change = self._rng.normal(drift, volatility)
            new_price = current_price * (1 + change)

            # Generate OHLC from price movement
            intraday_vol = 0.005  # Intraday volatility
            high_factor = 1 + abs(self._rng.normal(0, intraday_vol))
            low_factor = 1 - abs(self._rng.normal(0, intraday_vol))

            open_price = Decimal(str(current_price))
            close_price = Decimal(str(new_price))
            high_price = Decimal(str(max(current_price, new_price) * high_factor))
            low_price = Decimal(str(min(current_price, new_price) * low_factor))
            volume = Decimal(str(self._rng.integers(1000, 10000)))

            bar = Bar(
                symbol=symbol,
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )

            yield bar

            # Update for next iteration
            current_price = new_price
            self._current_prices[symbol] = close_price
            current_time += timedelta(minutes=1)

            # Small delay to simulate real-time data
            await asyncio.sleep(0.01)

    async def get_ticks(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> AsyncIterator[Tick]:
        """Generate synthetic ticks for a symbol."""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not in data source")

        current_price = float(self._current_prices[symbol])
        current_time = self._start_time

        while True:
            # Small random price movement
            change = self._rng.normal(0, 0.001)
            new_price = current_price * (1 + change)

            tick = Tick(
                symbol=symbol,
                timestamp=current_time,
                price=Decimal(str(new_price)),
                size=Decimal(str(self._rng.integers(100, 1000))),
                bid=Decimal(str(new_price * 0.999)),
                ask=Decimal(str(new_price * 1.001)),
                bid_size=Decimal(str(self._rng.integers(100, 500))),
                ask_size=Decimal(str(self._rng.integers(100, 500))),
            )

            yield tick

            current_price = new_price
            self._current_prices[symbol] = Decimal(str(new_price))
            current_time += timedelta(seconds=1)

            await asyncio.sleep(0.001)
