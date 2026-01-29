"""Tests for parquet storage."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.storage.parquet_store import ParquetStore


def test_write_and_read_bars() -> None:
    """Test writing and reading bars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_path=tmpdir)

        # Create synthetic data
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="1H", tz="UTC")
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

        # Write bars
        store.write_bars(
            asset_class="crypto",
            symbol="BTC/USDT",
            timeframe="1h",
            df=df,
        )

        # Read all bars
        read_df = store.read_bars(
            asset_class="crypto",
            symbol="BTC/USDT",
            timeframe="1h",
        )

        assert len(read_df) == len(df)
        assert set(read_df.columns) == set(df.columns)
        pd.testing.assert_frame_equal(
            read_df.sort_values("timestamp").reset_index(drop=True),
            df.sort_values("timestamp").reset_index(drop=True),
            check_dtype=False,
        )


def test_read_bars_with_time_filter() -> None:
    """Test reading bars with time filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_path=tmpdir)

        # Create data spanning multiple months
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="1D", tz="UTC")
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

        # Read subset
        start = datetime(2024, 2, 1, tzinfo=pd.Timestamp.now().tz)
        end = datetime(2024, 2, 28, tzinfo=pd.Timestamp.now().tz)

        read_df = store.read_bars(
            asset_class="equities",
            symbol="AAPL",
            timeframe="1d",
            start=start,
            end=end,
        )

        assert len(read_df) > 0
        assert all(read_df["timestamp"] >= pd.Timestamp(start))
        assert all(read_df["timestamp"] <= pd.Timestamp(end))


def test_partition_structure() -> None:
    """Test that partitions are created correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_path=tmpdir)

        # Create data spanning multiple months
        dates = pd.date_range("2024-01-15", "2024-02-15", freq="1D", tz="UTC")
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
            symbol="ETH/USDT",
            timeframe="1d",
            df=df,
        )

        # Check partition structure
        base_path = Path(tmpdir)
        jan_path = base_path / "crypto" / "ETH/USDT" / "1d" / "2024" / "01" / "data.parquet"
        feb_path = base_path / "crypto" / "ETH/USDT" / "1d" / "2024" / "02" / "data.parquet"

        assert jan_path.exists()
        assert feb_path.exists()


def test_write_merge_existing() -> None:
    """Test that writing to existing partition merges data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_path=tmpdir)

        # Write first batch
        dates1 = pd.date_range("2024-01-01", "2024-01-15", freq="1D", tz="UTC")
        df1 = pd.DataFrame(
            {
                "timestamp": dates1,
                "open": range(100, 100 + len(dates1)),
                "high": range(101, 101 + len(dates1)),
                "low": range(99, 99 + len(dates1)),
                "close": range(100, 100 + len(dates1)),
                "volume": [1000] * len(dates1),
            }
        )

        store.write_bars(
            asset_class="equities",
            symbol="MSFT",
            timeframe="1d",
            df=df1,
        )

        # Write overlapping batch
        dates2 = pd.date_range("2024-01-10", "2024-01-20", freq="1D", tz="UTC")
        df2 = pd.DataFrame(
            {
                "timestamp": dates2,
                "open": range(200, 200 + len(dates2)),
                "high": range(201, 201 + len(dates2)),
                "low": range(199, 199 + len(dates2)),
                "close": range(200, 200 + len(dates2)),
                "volume": [2000] * len(dates2),
            }
        )

        store.write_bars(
            asset_class="equities",
            symbol="MSFT",
            timeframe="1d",
            df=df2,
        )

        # Read and verify
        read_df = store.read_bars(
            asset_class="equities",
            symbol="MSFT",
            timeframe="1d",
        )

        # Should have merged data (later writes overwrite duplicates)
        assert len(read_df) >= len(dates1)
        assert len(read_df) <= len(dates1) + len(dates2)
