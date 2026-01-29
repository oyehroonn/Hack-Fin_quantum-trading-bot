"""YFinance-based equities data ingestor."""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from data.ingest.base import Ingestor


class EquitiesYFinanceIngestor(Ingestor):
    """YFinance-based equities data ingestor."""

    def __init__(self) -> None:
        """Initialize YFinance ingestor."""
        pass

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Symbol (e.g., 'AAPL', 'MSFT')
            timeframe: Timeframe (e.g., '1d', '1h') - yfinance supports: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of bars (ignored, uses start/end)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to yfinance interval
            interval = self._normalize_timeframe(timeframe)

            # Set default dates if not provided
            if start is None:
                start = datetime.now().replace(year=datetime.now().year - 1)
            if end is None:
                end = datetime.now()

            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                prepost=False,
            )

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Rename index to timestamp
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "timestamp"})
            elif df.index.name == "Date":
                df.index.name = "timestamp"
                df = df.reset_index()

            # Select OHLCV columns
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            available_cols = [col for col in ohlcv_cols if col in df.columns]

            if not available_cols:
                logger.warning(f"No OHLCV columns found for {symbol}")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            df = df[["timestamp"] + available_cols].copy()

            # Rename to lowercase
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Standardize
            df = self.standardize_columns(df)

            # Apply limit if specified
            if limit is not None and len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)

            logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} from yfinance: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe to yfinance interval."""
        timeframe = timeframe.lower().strip()

        # yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1wk": "1wk",
            "1mo": "1mo",
            "1m": "1mo",
        }

        return mapping.get(timeframe, "1d")  # Default to daily
