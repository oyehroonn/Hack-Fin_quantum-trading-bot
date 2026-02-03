"""Binance public API ingestor for crypto (no API key required)."""

from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from data.ingest.base import Ingestor


class BinancePublicIngestor(Ingestor):
    """Binance public API ingestor (no authentication required)."""

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self) -> None:
        """Initialize Binance ingestor."""
        pass

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance public API.

        Args:
            symbol: Symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            timeframe: Timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of bars (default 1000, max 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        import asyncio

        def _fetch_sync() -> pd.DataFrame:
            """Synchronous fetch function."""
            try:
                # Normalize symbol (Binance uses BTCUSDT, not BTC/USDT)
                symbol_clean = symbol.replace("/", "").upper()

                # Normalize timeframe
                interval = self._normalize_timeframe(timeframe)

                # Convert dates to milliseconds
                start_ms = int(start.timestamp() * 1000) if start else None
                end_ms = int(end.timestamp() * 1000) if end else None

                # Set limit (Binance max is 1000)
                fetch_limit = min(limit or 1000, 1000)

                # Build URL
                url = f"{self.BASE_URL}/klines"
                params = {
                    "symbol": symbol_clean,
                    "interval": interval,
                    "limit": fetch_limit,
                }

                if start_ms:
                    params["startTime"] = start_ms
                if end_ms:
                    params["endTime"] = end_ms

                logger.info(f"Fetching {symbol_clean} from Binance: {params}")

                # Make request
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if not data:
                    raise ValueError(f"No data returned from Binance for {symbol_clean}")

                # Convert to DataFrame
                # Binance returns: [timestamp, open, high, low, close, volume, ...]
                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_volume",
                        "trades",
                        "taker_buy_base",
                        "taker_buy_quote",
                        "ignore",
                    ],
                )

                # Convert timestamp from milliseconds to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                # Select only OHLCV columns
                df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

                # Convert to numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Standardize
                df = self.standardize_columns(df)

                logger.info(f"Successfully fetched {len(df)} bars for {symbol_clean} ({timeframe})")
                return df

            except requests.exceptions.RequestException as e:
                logger.error(f"Binance API error for {symbol}: {e}")
                raise ValueError(f"Binance API error: {e}")
            except Exception as e:
                logger.error(f"Error fetching {symbol} from Binance: {e}", exc_info=True)
                raise

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(None, _fetch_sync)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from Binance: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe to Binance interval format.

        Binance intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        timeframe = timeframe.lower().strip()

        mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1wk": "1w",
            "1mo": "1M",
            "1m": "1M",
        }

        return mapping.get(timeframe, "1d")  # Default to daily
