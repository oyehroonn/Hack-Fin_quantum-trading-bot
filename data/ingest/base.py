"""Base ingestor interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class Ingestor(ABC):
    """Base interface for data ingestors."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Args:
            symbol: Symbol to fetch
            timeframe: Timeframe (e.g., '1h', '1d', '1m')
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of bars to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types.

        Args:
            df: Raw DataFrame

        Returns:
            Standardized DataFrame
        """
        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]

        # Common column name mappings
        column_mapping = {
            "time": "timestamp",
            "datetime": "timestamp",
            "date": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vol": "volume",
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Ensure UTC timezone
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Select only required columns
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols].copy()

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df
