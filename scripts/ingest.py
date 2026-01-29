#!/usr/bin/env python3
"""CLI script for data ingestion."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from data.ingest.crypto_ccxt import CryptoCCXTIngestor
from data.ingest.equities_yfinance import EquitiesYFinanceIngestor
from data.ingest.forex_oanda import ForexOANDAIngestor
from data.quality.validators import DataValidator
from data.storage.parquet_store import ParquetStore
from infra.logging import setup_logging, set_correlation_id

from loguru import logger


async def ingest_data(
    asset_class: str,
    symbols: list[str],
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    base_path: str = "data/parquet",
) -> None:
    """Ingest data for given parameters.

    Args:
        asset_class: Asset class ('crypto', 'equities', 'forex')
        symbols: List of symbols
        timeframe: Timeframe (e.g., '1h', '1d')
        start: Start timestamp
        end: End timestamp
        base_path: Base path for parquet storage
    """
    # Initialize ingestor based on asset class
    if asset_class == "crypto":
        ingestor = CryptoCCXTIngestor()
    elif asset_class == "equities":
        ingestor = EquitiesYFinanceIngestor()
    elif asset_class == "forex":
        ingestor = ForexOANDAIngestor()
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")

    # Initialize storage and validator
    store = ParquetStore(base_path=base_path)
    validator = DataValidator(raise_on_error=False, strict=False)

    # Ingest each symbol
    for symbol in symbols:
        logger.info(f"Ingesting {asset_class}/{symbol} ({timeframe})")

        try:
            # Fetch data
            df = await ingestor.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )

            if df.empty:
                logger.warning(f"No data fetched for {symbol}")
                continue

            # Validate data
            report = validator.validate(df)
            logger.info(f"Validation for {symbol}:\n{report}")

            if not report.passed:
                logger.warning(f"Validation failed for {symbol}, but continuing...")

            # Store data
            store.write_bars(
                asset_class=asset_class,
                symbol=symbol,
                timeframe=timeframe,
                df=df,
            )

            logger.info(f"Successfully ingested {len(df)} bars for {symbol}")

        except Exception as e:
            logger.error(f"Error ingesting {symbol}: {e}", exc_info=True)
            continue


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest market data")
    parser.add_argument(
        "--asset-class",
        required=True,
        choices=["crypto", "equities", "forex"],
        help="Asset class",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        nargs="+",
        help="Symbols to ingest (e.g., BTC/USDT AAPL MSFT)",
    )
    parser.add_argument(
        "--timeframe",
        required=True,
        help="Timeframe (e.g., 1h, 1d, 1m)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="data/parquet",
        help="Base path for parquet storage",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    # Setup logging
    set_correlation_id("ingest-cli")
    setup_logging(level=args.log_level)

    # Parse dates
    start = None
    if args.start:
        try:
            start = datetime.fromisoformat(args.start)
        except ValueError:
            start = datetime.strptime(args.start, "%Y-%m-%d")

    end = None
    if args.end:
        try:
            end = datetime.fromisoformat(args.end)
        except ValueError:
            end = datetime.strptime(args.end, "%Y-%m-%d")

    # Run ingestion
    asyncio.run(
        ingest_data(
            asset_class=args.asset_class,
            symbols=args.symbols,
            timeframe=args.timeframe,
            start=start,
            end=end,
            base_path=args.base_path,
        )
    )


if __name__ == "__main__":
    main()
