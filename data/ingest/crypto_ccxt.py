"""CCXT-based crypto data ingestor."""

from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger

from data.ingest.base import Ingestor


class CryptoCCXTIngestor(Ingestor):
    """CCXT-based crypto data ingestor."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
    ) -> None:
        """Initialize CCXT ingestor.

        Args:
            exchange_id: Exchange ID (e.g., 'binance', 'coinbase')
            api_key: API key (optional for public data)
            api_secret: API secret (optional for public data)
            sandbox: Use sandbox/testnet
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self._exchange: Optional[ccxt.Exchange] = None

    def _get_exchange(self) -> ccxt.Exchange:
        """Get or create exchange instance."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id, None)
            if exchange_class is None:
                raise ValueError(f"Exchange '{self.exchange_id}' not found in CCXT")

            config = {
                "sandbox": self.sandbox,
                "enableRateLimit": True,
            }

            if self.api_key and self.api_secret:
                config["apiKey"] = self.api_key
                config["secret"] = self.api_secret

            self._exchange = exchange_class(config)

        return self._exchange

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from CCXT exchange.

        Args:
            symbol: Symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d', '1m')
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of bars

        Returns:
            DataFrame with OHLCV data
        """
        try:
            exchange = self._get_exchange()

            # Convert timeframe to CCXT format if needed
            ccxt_timeframe = self._normalize_timeframe(timeframe)

            # Fetch data
            since = int(start.timestamp() * 1000) if start else None
            ohlcv = exchange.fetch_ohlcv(symbol, ccxt_timeframe, since=since, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Convert timestamp from milliseconds to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # Filter by end time if provided
            if end is not None:
                end_ts = pd.Timestamp(end, tz="UTC")
                df = df[df["timestamp"] <= end_ts]

            # Standardize
            df = self.standardize_columns(df)

            logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
            return df

        except ccxt.BaseError as e:
            logger.error(f"CCXT error fetching {symbol}: {e}")
            if "API key" in str(e) or "authentication" in str(e).lower():
                logger.warning("API keys may be missing or invalid - returning empty DataFrame")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            raise
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise

    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe to CCXT format."""
        # CCXT uses formats like '1h', '1d', '1m', '1w'
        # Handle common variations
        timeframe = timeframe.lower().strip()

        # Map common variations
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
        }

        return mapping.get(timeframe, timeframe)
