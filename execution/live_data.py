"""Live data source that replays historical data in real-time."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import AsyncIterator, Optional

import pandas as pd
from loguru import logger

from core.interfaces import DataSource
from core.types import Bar
from data.storage.parquet_store import ParquetStore


class ReplayDataSource(DataSource):
    """Replay historical data as if it were live, with configurable speedup."""

    def __init__(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        speedup_factor: float = 1.0,
        data_path: str = "data/parquet",
    ) -> None:
        """Initialize replay data source.

        Args:
            symbol: Symbol to replay
            asset_class: Asset class
            timeframe: Timeframe
            start_date: Start date for replay
            end_date: End date for replay
            speedup_factor: Speedup factor (1.0 = real-time, 10.0 = 10x speed)
            data_path: Path to parquet data
        """
        self.symbol = symbol
        self.asset_class = asset_class
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.speedup_factor = speedup_factor
        self.data_path = data_path
        self._bars: list[Bar] = []
        self._current_index = 0
        self._replay_start_time: Optional[datetime] = None

    async def _load_data(self) -> None:
        """Load historical data from parquet."""
        try:
            # Load data using ParquetStore
            store = ParquetStore(base_path=self.data_path)
            df = store.read_bars(
                asset_class=self.asset_class,
                symbol=self.symbol,
                timeframe=self.timeframe,
                start=self.start_date,
                end=self.end_date,
            )

            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")

            # Convert to Bar objects
            self._bars = []
            for idx, row in df.iterrows():
                if isinstance(idx, tuple):
                    timestamp = idx[0]
                else:
                    timestamp = idx

                bar = Bar(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row.get("volume", 0))),
                )
                self._bars.append(bar)

            logger.info(f"Loaded {len(self._bars)} bars for {self.symbol}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    async def get_bars(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> AsyncIterator[Bar]:
        """Stream bars for a symbol."""
        if not self._bars:
            await self._load_data()

        if self._replay_start_time is None:
            self._replay_start_time = datetime.now()

        while self._current_index < len(self._bars):
            bar = self._bars[self._current_index]

            # Calculate when this bar should be emitted
            if self._current_index == 0:
                # First bar - emit immediately
                wait_time = 0
            else:
                # Calculate time difference between bars
                prev_bar = self._bars[self._current_index - 1]
                time_diff = bar.timestamp - prev_bar.timestamp
                # Convert to seconds and apply speedup
                wait_time = time_diff.total_seconds() / self.speedup_factor

            # Wait if needed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Check if we've exceeded end_date
            if bar.timestamp > self.end_date:
                break

            self._current_index += 1
            yield bar

    async def get_ticks(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> AsyncIterator:
        """Stream ticks for a symbol (not implemented for replay)."""
        raise NotImplementedError("Tick replay not implemented")

    def reset(self) -> None:
        """Reset replay to beginning."""
        self._current_index = 0
        self._replay_start_time = None

    def get_progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        if not self._bars:
            return 0.0
        return self._current_index / len(self._bars)
