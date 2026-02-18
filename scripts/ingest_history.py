#!/usr/bin/env python3
"""Ingest full crypto history from Binance for ML training.

Usage:
    python3 scripts/ingest_history.py --symbol BTCUSDT --timeframe 1d
    python3 scripts/ingest_history.py --symbol BTCUSDT --timeframe 1h --start 2020-01-01
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest.binance_public import BinancePublicIngestor
from data.storage.parquet_store import ParquetStore
from loguru import logger

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAMES = ["1d", "1h"]


async def ingest_symbol(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch all bars from Binance in batches of 1000."""
    ingestor = BinancePublicIngestor()
    all_dfs = []
    current_start = start
    batch = 0

    while current_start < end:
        try:
            df = await ingestor.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=current_start,
                end=end,
                limit=1000,
            )
            if df.empty:
                break

            all_dfs.append(df)
            last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
            if last_ts.tz is None:
                last_ts = last_ts.tz_localize("UTC")

            if last_ts <= current_start:
                break

            current_start = last_ts + pd.Timedelta(seconds=1)
            batch += 1
            if batch % 10 == 0:
                logger.info(f"  {symbol} {timeframe}: {len(pd.concat(all_dfs))} bars fetched so far (up to {last_ts.date()})")

            await asyncio.sleep(0.2)
        except Exception as e:
            logger.warning(f"Batch error for {symbol} {timeframe}: {e}")
            await asyncio.sleep(2)
            break

    if not all_dfs:
        logger.warning(f"No data fetched for {symbol} {timeframe}")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    logger.info(f"  {symbol} {timeframe}: total {len(combined)} bars ({combined['timestamp'].min()} → {combined['timestamp'].max()})")
    return combined


async def main(symbols: list[str], timeframes: list[str], start_date: str):
    store = ParquetStore("data/parquet")
    utc = pytz.UTC
    end = datetime.now(tz=utc)
    start = pd.Timestamp(start_date, tz=utc) if start_date else pd.Timestamp("2017-01-01", tz=utc)

    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"Ingesting {symbol} {tf} from {start.date()} to {end.date()}")
            df = await ingest_symbol(symbol, tf, start, end)
            if df.empty:
                continue
            store.write_bars(asset_class="crypto", symbol=symbol, timeframe=tf, df=df)
            logger.info(f"Stored {len(df)} bars for {symbol} {tf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest crypto history from Binance")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (default: all major)")
    parser.add_argument("--timeframe", type=str, default=None, help="Single timeframe (default: 1d,1h)")
    parser.add_argument("--start", type=str, default="2017-01-01", help="Start date (default: 2017-01-01)")
    args = parser.parse_args()

    syms = [args.symbol.upper()] if args.symbol else SYMBOLS
    tfs = [args.timeframe] if args.timeframe else TIMEFRAMES

    asyncio.run(main(syms, tfs, args.start))
