"""Tests for DuckDB client."""

import tempfile
from datetime import datetime

import pandas as pd
import pytest

from data.query.duckdb_client import DuckDBClient
from data.storage.parquet_store import ParquetStore


def test_duckdb_query_single_symbol() -> None:
    """Test DuckDB query for single symbol."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test data
        store = ParquetStore(base_path=tmpdir)

        dates = pd.date_range("2024-01-01", "2024-01-10", freq="1H", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": range(100, 100 + len(dates)),
                "high": range(101, 101 + len(dates)),
                "low": range(99, 99 + len(dates)),
                "close": range(100, 100 + len(dates)),
                "volume": [1000] * len(dates),
            }
        )

        store.write_bars(
            asset_class="crypto",
            symbol="BTC/USDT",
            timeframe="1h",
            df=df,
        )

        # Query with DuckDB
        with DuckDBClient(base_path=tmpdir) as client:
            result = client.get_bars(
                symbols=[("crypto", "BTC/USDT", "1h")],
            )

            assert not result.empty
            assert len(result) == len(df)
            assert "open" in result.columns
            assert "high" in result.columns
            assert "close" in result.columns


def test_duckdb_query_multiple_symbols() -> None:
    """Test DuckDB query for multiple symbols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test data for multiple symbols
        store = ParquetStore(base_path=tmpdir)

        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            dates = pd.date_range("2024-01-01", "2024-01-05", freq="1D", tz="UTC")
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": range(100, 100 + len(dates)),
                    "high": range(101, 101 + len(dates)),
                    "low": range(99, 99 + len(dates)),
                    "close": range(100, 100 + len(dates)),
                    "volume": [1000] * len(dates),
                }
            )

            store.write_bars(
                asset_class="equities",
                symbol=symbol,
                timeframe="1d",
                df=df,
            )

        # Query with DuckDB
        with DuckDBClient(base_path=tmpdir) as client:
            symbol_tuples = [("equities", s, "1d") for s in symbols]
            result = client.get_bars(symbols=symbol_tuples)

            assert not result.empty
            assert isinstance(result.index, pd.MultiIndex)
            assert result.index.names == ["timestamp", "symbol"]

            # Check all symbols are present
            unique_symbols = result.index.get_level_values("symbol").unique()
            assert set(unique_symbols) == set(symbols)


def test_duckdb_query_with_time_filter() -> None:
    """Test DuckDB query with time filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_path=tmpdir)

        dates = pd.date_range("2024-01-01", "2024-01-31", freq="1D", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": range(100, 100 + len(dates)),
                "high": range(101, 101 + len(dates)),
                "low": range(99, 99 + len(dates)),
                "close": range(100, 100 + len(dates)),
                "volume": [1000] * len(dates),
            }
        )

        store.write_bars(
            asset_class="equities",
            symbol="AAPL",
            timeframe="1d",
            df=df,
        )

        # Query with time filter
        with DuckDBClient(base_path=tmpdir) as client:
            start = datetime(2024, 1, 10, tzinfo=pd.Timestamp.now().tz)
            end = datetime(2024, 1, 20, tzinfo=pd.Timestamp.now().tz)

            result = client.get_bars(
                symbols=[("equities", "AAPL", "1d")],
                start=start,
                end=end,
            )

            assert not result.empty
            timestamps = result.index.get_level_values("timestamp")
            assert all(ts >= pd.Timestamp(start) for ts in timestamps)
            assert all(ts <= pd.Timestamp(end) for ts in timestamps)
