"""Data quality validators."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationReport:
    """Validation report."""

    passed: bool
    checks: dict[str, tuple[bool, str]]  # check_name -> (passed, message)
    errors: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        """String representation."""
        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        for check_name, (passed, message) in self.checks.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {status} {check_name}: {message}")
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


class DataValidator:
    """Data quality validator."""

    def __init__(
        self,
        raise_on_error: bool = False,
        strict: bool = True,
    ) -> None:
        """Initialize validator.

        Args:
            raise_on_error: Raise exception on validation failure
            strict: Strict mode (fail on warnings)
        """
        self.raise_on_error = raise_on_error
        self.strict = strict

    def validate_monotonic_time(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> tuple[bool, str]:
        """Validate that timestamps are monotonic.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            Tuple of (passed, message)
        """
        if timestamp_col not in df.columns:
            return (False, f"Column '{timestamp_col}' not found")

        timestamps = df[timestamp_col]
        is_monotonic = timestamps.is_monotonic_increasing

        if not is_monotonic:
            # Count violations
            diffs = timestamps.diff()
            violations = (diffs <= pd.Timedelta(0)).sum()
            return (
                False,
                f"Timestamps are not monotonic: {violations} violations found",
            )

        return (True, "Timestamps are monotonic")

    def validate_no_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """Validate no duplicate rows.

        Args:
            df: DataFrame to validate
            subset: Columns to check for duplicates (default: all columns)

        Returns:
            Tuple of (passed, message)
        """
        if subset is None:
            subset = list(df.columns)

        duplicates = df.duplicated(subset=subset)
        num_duplicates = duplicates.sum()

        if num_duplicates > 0:
            return (False, f"Found {num_duplicates} duplicate rows")

        return (True, "No duplicates found")

    def validate_missing_ratio(
        self,
        df: pd.DataFrame,
        max_missing_ratio: float = 0.05,
        columns: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """Validate missing data ratio.

        Args:
            df: DataFrame to validate
            max_missing_ratio: Maximum allowed missing ratio (0-1)
            columns: Columns to check (default: all columns)

        Returns:
            Tuple of (passed, message)
        """
        if columns is None:
            columns = list(df.columns)

        messages = []
        all_passed = True

        for col in columns:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / len(df)

            if missing_ratio > max_missing_ratio:
                all_passed = False
                messages.append(
                    f"{col}: {missing_ratio:.2%} missing (max: {max_missing_ratio:.2%})"
                )

        if not all_passed:
            return (False, "; ".join(messages))

        return (True, f"All columns have < {max_missing_ratio:.2%} missing data")

    def validate_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        zscore_threshold: float = 3.0,
        warn_only: bool = True,
    ) -> tuple[bool, str]:
        """Validate outliers using z-score.

        Args:
            df: DataFrame to validate
            columns: Numeric columns to check (default: all numeric columns)
            zscore_threshold: Z-score threshold for outliers
            warn_only: Only warn, don't fail on outliers

        Returns:
            Tuple of (passed, message)
        """
        if columns is None:
            # Get numeric columns
            columns = list(df.select_dtypes(include=[np.number]).columns)

        if not columns:
            return (True, "No numeric columns to validate")

        messages = []
        outliers_found = False

        for col in columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = (z_scores > zscore_threshold).sum()

            if outliers > 0:
                outliers_found = True
                messages.append(f"{col}: {outliers} outliers (z-score > {zscore_threshold})")

        if outliers_found:
            message = "; ".join(messages)
            if warn_only:
                return (True, f"Outliers found (warn only): {message}")
            return (False, f"Outliers found: {message}")

        return (True, "No outliers detected")

    def validate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        check_duplicates: bool = True,
        check_missing: bool = True,
        check_outliers: bool = True,
        max_missing_ratio: float = 0.05,
        outlier_zscore: float = 3.0,
    ) -> ValidationReport:
        """Run all validations.

        Args:
            df: DataFrame to validate
            timestamp_col: Name of timestamp column
            check_duplicates: Check for duplicates
            check_missing: Check missing data
            check_outliers: Check outliers
            max_missing_ratio: Maximum missing ratio
            outlier_zscore: Z-score threshold for outliers

        Returns:
            ValidationReport

        Raises:
            ValueError: If validation fails and raise_on_error is True
        """
        checks: dict[str, tuple[bool, str]] = {}
        errors: list[str] = []
        warnings: list[str] = []

        # Validate monotonic time
        passed, message = self.validate_monotonic_time(df, timestamp_col)
        checks["monotonic_time"] = (passed, message)
        if not passed:
            errors.append(message)

        # Validate duplicates
        if check_duplicates:
            passed, message = self.validate_no_duplicates(df, subset=[timestamp_col])
            checks["no_duplicates"] = (passed, message)
            if not passed:
                errors.append(message)

        # Validate missing data
        if check_missing:
            passed, message = self.validate_missing_ratio(df, max_missing_ratio)
            checks["missing_ratio"] = (passed, message)
            if not passed:
                if self.strict:
                    errors.append(message)
                else:
                    warnings.append(message)

        # Validate outliers
        if check_outliers:
            passed, message = self.validate_outliers_zscore(
                df, zscore_threshold=outlier_zscore, warn_only=True
            )
            checks["outliers"] = (passed, message)
            if not passed and not self.strict:
                warnings.append(message)

        # Determine overall status
        overall_passed = len(errors) == 0 and (not self.strict or len(warnings) == 0)

        report = ValidationReport(
            passed=overall_passed,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

        if not overall_passed and self.raise_on_error:
            raise ValueError(f"Validation failed:\n{report}")

        logger.info(f"Validation {'PASSED' if overall_passed else 'FAILED'}")
        for check_name, (passed, message) in checks.items():
            logger.debug(f"  {check_name}: {message}")

        return report
