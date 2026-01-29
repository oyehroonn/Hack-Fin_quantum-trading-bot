"""Parquet storage with partitioning by timeframe/year/month."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
from loguru import logger


class ParquetStore:
    """Parquet storage with partitioning support."""

    def __init__(self, base_path: str | Path = "data/parquet") -> None:
        """Initialize parquet store.

        Args:
            base_path: Base directory for parquet files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_partition_path(
        self,
        asset_class: str,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
    ) -> Path:
        """Get partition path for given parameters."""
        return (
            self.base_path
            / asset_class
            / symbol
            / timeframe
            / str(year)
            / f"{month:02d}"
        )

    def write_bars(
        self,
        asset_class: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """Write bars to parquet with partitioning.

        Args:
            asset_class: Asset class (e.g., 'crypto', 'equities', 'forex')
            symbol: Symbol (e.g., 'BTC/USDT', 'AAPL')
            timeframe: Timeframe (e.g., '1h', '1d', '1m')
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Validate required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure UTC timezone
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Group by year/month for partitioning
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        # Write each partition
        for (year, month), group_df in df.groupby(["year", "month"]):
            partition_path = self._get_partition_path(
                asset_class, symbol, timeframe, year, month
            )
            partition_path.mkdir(parents=True, exist_ok=True)

            # Remove partition columns before writing
            write_df = group_df.drop(columns=["year", "month"]).copy()

            # Convert to pyarrow table
            table = pa.Table.from_pandas(write_df)

            # Write parquet file
            file_path = partition_path / "data.parquet"

            # If file exists, read and merge
            if file_path.exists():
                existing_table = pq.read_table(file_path)
                existing_df = existing_table.to_pandas()
                # Combine and deduplicate
                combined_df = pd.concat([existing_df, write_df]).drop_duplicates(
                    subset=["timestamp"], keep="last"
                )
                combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
                table = pa.Table.from_pandas(combined_df)

            pq.write_table(
                table,
                file_path,
                compression="snappy",
                use_dictionary=True,
            )

            logger.info(
                f"Wrote {len(write_df)} bars to {file_path} "
                f"({asset_class}/{symbol}/{timeframe}/{year}/{month:02d})"
            )

    def read_bars(
        self,
        asset_class: str,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Read bars from parquet with time filtering.

        Args:
            asset_class: Asset class
            symbol: Symbol
            timeframe: Timeframe
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Returns:
            DataFrame with bars
        """
        # Determine which partitions to read
        if start is None and end is None:
            # Read all partitions
            symbol_path = self.base_path / asset_class / symbol / timeframe
            if not symbol_path.exists():
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Find all parquet files
            parquet_files = list(symbol_path.rglob("data.parquet"))
        else:
            # Read specific partitions
            import pytz
            utc = pytz.UTC
            
            start_dt = start or datetime(2000, 1, 1, tzinfo=utc)
            end_dt = end or datetime.now(utc)

            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=utc)

            # Convert to UTC
            start_dt = start_dt.astimezone(utc)
            end_dt = end_dt.astimezone(utc)

            parquet_files = []
            current = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current <= end_dt:
                partition_path = self._get_partition_path(
                    asset_class, symbol, timeframe, current.year, current.month
                )
                file_path = partition_path / "data.parquet"
                if file_path.exists():
                    parquet_files.append(file_path)
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

        if not parquet_files:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Read all parquet files
        tables = [pq.read_table(f) for f in parquet_files]
        if not tables:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Combine tables
        combined_table = pa.concat_tables(tables)
        df = combined_table.to_pandas()

        # Filter by time range
        if start is not None:
            import pytz
            if start.tzinfo is None:
                start = start.replace(tzinfo=pytz.UTC)
            df = df[df["timestamp"] >= start]
        if end is not None:
            import pytz
            if end.tzinfo is None:
                end = end.replace(tzinfo=pytz.UTC)
            df = df[df["timestamp"] <= end]

        # Sort and deduplicate
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        return df
