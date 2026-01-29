"""DuckDB client for querying parquet datasets."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from loguru import logger


class DuckDBClient:
    """DuckDB client for querying parquet datasets."""

    def __init__(
        self,
        base_path: str | Path = "data/parquet",
        database_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize DuckDB client.

        Args:
            base_path: Base directory for parquet files
            database_path: Path to DuckDB database file (in-memory if None)
        """
        self.base_path = Path(base_path)
        self.conn = duckdb.connect(database_path or ":memory:")
        self._registered_datasets: set[str] = set()

    def register_parquet_dataset(
        self,
        asset_class: str,
        symbol: str,
        timeframe: str,
        alias: Optional[str] = None,
    ) -> None:
        """Register a parquet dataset for querying.

        Args:
            asset_class: Asset class
            symbol: Symbol
            timeframe: Timeframe
            alias: Optional alias for the dataset (default: {asset_class}_{symbol}_{timeframe})
        """
        if alias is None:
            alias = f"{asset_class}_{symbol}_{timeframe}".replace("/", "_").replace(
                "-", "_"
            )

        if alias in self._registered_datasets:
            logger.debug(f"Dataset {alias} already registered")
            return

        # Find all parquet files for this dataset
        dataset_path = self.base_path / asset_class / symbol / timeframe
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return

        parquet_files = list(dataset_path.rglob("data.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {dataset_path}")
            return

        # Register as a view
        # DuckDB can read parquet files directly, but we'll use a glob pattern
        parquet_pattern = str(dataset_path / "**" / "data.parquet")

        query = f"""
        CREATE OR REPLACE VIEW {alias} AS
        SELECT * FROM read_parquet('{parquet_pattern}')
        ORDER BY timestamp
        """

        self.conn.execute(query)
        self._registered_datasets.add(alias)

        logger.info(f"Registered dataset {alias} with {len(parquet_files)} partitions")

    def get_bars(
        self,
        symbols: list[tuple[str, str, str]],  # (asset_class, symbol, timeframe)
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get bars for multiple symbols as a multi-index DataFrame.

        Args:
            symbols: List of (asset_class, symbol, timeframe) tuples
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            timeframe: Optional timeframe filter (if all symbols have same timeframe)

        Returns:
            Multi-index DataFrame with (timestamp, symbol) index
        """
        if not symbols:
            return pd.DataFrame()

        # Register all datasets
        for asset_class, symbol, tf in symbols:
            self.register_parquet_dataset(asset_class, symbol, tf)

        # Build UNION query
        queries = []
        for asset_class, symbol, tf in symbols:
            alias = f"{asset_class}_{symbol}_{tf}".replace("/", "_").replace("-", "_")
            if alias not in self._registered_datasets:
                continue

            query = f"SELECT timestamp, '{symbol}' as symbol, open, high, low, close, volume FROM {alias}"
            if start is not None:
                start_str = start.isoformat() if isinstance(start, datetime) else str(start)
                query += f" WHERE timestamp >= '{start_str}'"
            if end is not None:
                end_str = end.isoformat() if isinstance(end, datetime) else str(end)
                if "WHERE" in query:
                    query += f" AND timestamp <= '{end_str}'"
                else:
                    query += f" WHERE timestamp <= '{end_str}'"

            queries.append(query)

        if not queries:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            )

        # Combine queries
        combined_query = " UNION ALL ".join(queries) + " ORDER BY timestamp, symbol"

        # Execute query
        result = self.conn.execute(combined_query).df()

        if result.empty:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            )

        # Convert timestamp to datetime
        result["timestamp"] = pd.to_datetime(result["timestamp"])

        # Set multi-index
        result = result.set_index(["timestamp", "symbol"])

        return result

    def close(self) -> None:
        """Close DuckDB connection."""
        self.conn.close()

    def __enter__(self) -> "DuckDBClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
