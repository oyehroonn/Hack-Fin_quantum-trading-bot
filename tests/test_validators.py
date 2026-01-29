"""Tests for data validators."""

import pandas as pd
import pytest

from data.quality.validators import DataValidator, ValidationReport


def test_validate_monotonic_time() -> None:
    """Test monotonic time validation."""
    validator = DataValidator()

    # Valid data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H", tz="UTC"),
            "open": range(10),
            "close": range(10),
        }
    )
    passed, message = validator.validate_monotonic_time(df)
    assert passed is True

    # Invalid data (not monotonic)
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),  # Out of order
            ],
            "open": [1, 2, 3],
            "close": [1, 2, 3],
        }
    )
    passed, message = validator.validate_monotonic_time(df)
    assert passed is False
    assert "monotonic" in message.lower()


def test_validate_no_duplicates() -> None:
    """Test duplicate validation."""
    validator = DataValidator()

    # Valid data (no duplicates)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H", tz="UTC"),
            "open": range(10),
            "close": range(10),
        }
    )
    passed, message = validator.validate_no_duplicates(df, subset=["timestamp"])
    assert passed is True

    # Invalid data (duplicates)
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),  # Duplicate
            ],
            "open": [1, 2],
            "close": [1, 2],
        }
    )
    passed, message = validator.validate_no_duplicates(df, subset=["timestamp"])
    assert passed is False
    assert "duplicate" in message.lower()


def test_validate_missing_ratio() -> None:
    """Test missing data ratio validation."""
    validator = DataValidator()

    # Valid data (low missing ratio)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1H", tz="UTC"),
            "open": range(100),
            "close": range(100),
        }
    )
    passed, message = validator.validate_missing_ratio(df, max_missing_ratio=0.05)
    assert passed is True

    # Invalid data (high missing ratio)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1H", tz="UTC"),
            "open": [None] * 50 + list(range(50)),  # 50% missing
            "close": range(100),
        }
    )
    passed, message = validator.validate_missing_ratio(df, max_missing_ratio=0.05)
    assert passed is False
    assert "missing" in message.lower()


def test_validate_outliers_zscore() -> None:
    """Test outlier validation."""
    validator = DataValidator()

    # Valid data (no outliers)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1H", tz="UTC"),
            "close": range(100, 200),  # Normal range
        }
    )
    passed, message = validator.validate_outliers_zscore(df, zscore_threshold=3.0)
    assert passed is True

    # Data with outliers
    values = list(range(100, 200)) + [1000, 2000]  # Outliers
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=102, freq="1H", tz="UTC"),
            "close": values,
        }
    )
    passed, message = validator.validate_outliers_zscore(df, zscore_threshold=3.0, warn_only=True)
    # Should pass with warn_only=True
    assert passed is True
    assert "outlier" in message.lower()


def test_full_validation() -> None:
    """Test full validation report."""
    validator = DataValidator(raise_on_error=False, strict=False)

    # Valid data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1H", tz="UTC"),
            "open": range(100, 200),
            "high": range(101, 201),
            "low": range(99, 199),
            "close": range(100, 200),
            "volume": [1000] * 100,
        }
    )

    report = validator.validate(df)
    assert isinstance(report, ValidationReport)
    assert report.passed is True
    assert len(report.errors) == 0


def test_validation_raises_on_error() -> None:
    """Test that validation raises on error when configured."""
    validator = DataValidator(raise_on_error=True, strict=True)

    # Invalid data (not monotonic)
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),
            ],
            "open": [1, 2, 3],
            "close": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="Validation failed"):
        validator.validate(df)
