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
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        import asyncio
        
        def _fetch_sync() -> pd.DataFrame:
            """Synchronous fetch function to run in executor."""
            try:
                # Convert timeframe to yfinance interval
                interval = self._normalize_timeframe(timeframe)

                # Set default dates if not provided
                if start is None:
                    start_dt = datetime.now().replace(year=datetime.now().year - 1)
                else:
                    start_dt = start
                    
                if end is None:
                    end_dt = datetime.now()
                else:
                    end_dt = end

                logger.info(f"Fetching {symbol} data from {start_dt} to {end_dt} (interval: {interval})")

                # Fetch data (blocking call) with retry logic
                ticker = yf.Ticker(symbol)
                
                # Calculate period for more reliable fetching
                days_diff = (end_dt - start_dt).days
                if days_diff <= 5:
                    period = "5d"
                elif days_diff <= 30:
                    period = "1mo"
                elif days_diff <= 90:
                    period = "3mo"
                elif days_diff <= 180:
                    period = "6mo"
                elif days_diff <= 365:
                    period = "1y"
                elif days_diff <= 730:
                    period = "2y"
                else:
                    period = "max"
                
                # Try period first (more reliable), then fallback to start/end
                df = pd.DataFrame()
                try:
                    logger.info(f"Trying period={period} for {symbol}")
                    df = ticker.history(
                        period=period,
                        interval=interval,
                        auto_adjust=True,
                        prepost=False,
                    )
                    # Filter by date range if needed
                    if not df.empty and (start_dt or end_dt):
                        if isinstance(df.index, pd.DatetimeIndex):
                            if start_dt:
                                df = df[df.index >= start_dt]
                            if end_dt:
                                df = df[df.index <= end_dt]
                except Exception as e:
                    logger.warning(f"Period method failed for {symbol}: {e}, trying start/end")
                    try:
                        df = ticker.history(
                            start=start_dt,
                            end=end_dt,
                            interval=interval,
                            auto_adjust=True,
                            prepost=False,
                        )
                    except Exception as e2:
                        logger.error(f"Both methods failed for {symbol}: {e2}")
                        raise ValueError(f"Failed to fetch data for {symbol}: {e2}")

                if df.empty:
                    # Try to get info to see if symbol is valid
                    try:
                        info = ticker.info
                        if not info:
                            raise ValueError(f"Symbol {symbol} not found or invalid")
                    except:
                        raise ValueError(f"Symbol {symbol} not found or invalid. No data returned from yfinance.")
                    raise ValueError(f"No data returned from yfinance for {symbol} in date range {start_dt} to {end_dt}")

                logger.info(f"Raw yfinance data shape: {df.shape}, columns: {df.columns.tolist()}")

                # yfinance returns DataFrame with DatetimeIndex
                # Reset index to get Date column
                df = df.reset_index()
                
                # Find date column (could be 'Date' or index name)
                date_col = None
                if "Date" in df.columns:
                    date_col = "Date"
                elif len(df.columns) > 0 and df.index.name == "Date":
                    df = df.reset_index()
                    date_col = "Date"
                else:
                    # Check if first column is datetime
                    for col in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            date_col = col
                            break
                
                if date_col:
                    df = df.rename(columns={date_col: "timestamp"})
                else:
                    # Create timestamp from index if available
                    if isinstance(df.index, pd.DatetimeIndex):
                        df["timestamp"] = df.index
                    else:
                        raise ValueError("Could not find date column in yfinance data")

                # Select OHLCV columns (yfinance uses capitalized names)
                ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
                available_cols = [col for col in ohlcv_cols if col in df.columns]

                if not available_cols:
                    logger.error(f"No OHLCV columns found. Available columns: {df.columns.tolist()}")
                    raise ValueError(f"No OHLCV columns found for {symbol}")

                # Select only needed columns
                df = df[["timestamp"] + available_cols].copy()

                # Rename to lowercase (required by backtest engine)
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })

                # Standardize (handles timezone, types, etc.)
                df = self.standardize_columns(df)

                # Apply limit if specified
                if limit is not None and len(df) > limit:
                    df = df.tail(limit).reset_index(drop=True)

                logger.info(f"Successfully fetched {len(df)} bars for {symbol} ({timeframe})")
                return df

            except Exception as e:
                logger.error(f"Error fetching {symbol} from yfinance: {e}", exc_info=True)
                raise

        # Run blocking yfinance call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(None, _fetch_sync)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            # Return empty DataFrame with correct structure
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
