"""Walk-forward analysis runner."""

from datetime import datetime
from decimal import Decimal
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from backtest.accounting import Portfolio
from backtest.engine import BacktestEngine
from backtest.report import PerformanceReport
from backtest.strategy import Strategy


class WalkForwardRunner:
    """Walk-forward analysis runner."""

    def __init__(
        self,
        initial_cash: Decimal,
        strategy_factory: Callable[[], Strategy],
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,  # ~3 months
        step_days: int = 21,  # ~1 month step
    ) -> None:
        """Initialize walk-forward runner.

        Args:
            initial_cash: Initial cash for each walk
            strategy_factory: Function that returns a new Strategy instance
            train_period_days: Training period in days
            test_period_days: Test period in days
            step_days: Step size in days
        """
        self.initial_cash = initial_cash
        self.strategy_factory = strategy_factory
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days

    def run(
        self,
        data: pd.DataFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Run walk-forward analysis.

        Args:
            data: DataFrame with bars (multi-index or single symbol)
            start: Start date (optional)
            end: End date (optional)

        Returns:
            DataFrame with walk-forward results
        """
        # Prepare data
        if isinstance(data.index, pd.MultiIndex):
            data = data.sort_index()
        else:
            if "symbol" not in data.columns:
                raise ValueError("Single symbol data must have 'symbol' column")
            data = data.set_index(["timestamp", "symbol"])

        # Filter by date range
        if start:
            data = data[data.index.get_level_values(0) >= start]
        if end:
            data = data[data.index.get_level_values(0) <= end]

        timestamps = sorted(data.index.get_level_values(0).unique())

        if len(timestamps) < self.train_period_days + self.test_period_days:
            raise ValueError("Insufficient data for walk-forward analysis")

        results = []

        # Walk forward
        current_date = timestamps[0]
        end_date = timestamps[-1]

        walk_num = 0
        while current_date < end_date:
            # Calculate train and test periods
            train_start = current_date
            train_end_idx = min(
                len(timestamps) - 1,
                next(
                    (i for i, t in enumerate(timestamps) if t >= train_start),
                    len(timestamps) - 1,
                )
                + self.train_period_days,
            )
            train_end = timestamps[min(train_end_idx, len(timestamps) - 1)]

            test_start = train_end
            test_end_idx = min(
                len(timestamps) - 1,
                next(
                    (i for i, t in enumerate(timestamps) if t >= test_start),
                    len(timestamps) - 1,
                )
                + self.test_period_days,
            )
            test_end = timestamps[min(test_end_idx, len(timestamps) - 1)]

            if test_end <= train_end:
                break

            logger.info(
                f"Walk {walk_num + 1}: Train [{train_start.date()} - {train_end.date()}] "
                f"Test [{test_start.date()} - {test_end.date()}]"
            )

            # Run backtest on test period
            strategy = self.strategy_factory()
            engine = BacktestEngine(
                initial_cash=self.initial_cash,
                strategy=strategy,
            )

            test_data = data[
                (data.index.get_level_values(0) >= test_start)
                & (data.index.get_level_values(0) <= test_end)
            ]

            if test_data.empty:
                logger.warning(f"Empty test data for walk {walk_num + 1}")
                current_date = pd.Timestamp(current_date) + pd.Timedelta(days=self.step_days)
                walk_num += 1
                continue

            portfolio = engine.run(test_data, start=test_start, end=test_end)

            # Calculate metrics
            report = PerformanceReport(portfolio)
            metrics = report.calculate_metrics()

            # Add walk metadata
            metrics["walk_num"] = walk_num
            metrics["train_start"] = train_start.isoformat()
            metrics["train_end"] = train_end.isoformat()
            metrics["test_start"] = test_start.isoformat()
            metrics["test_end"] = test_end.isoformat()

            results.append(metrics)

            # Step forward
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=self.step_days)
            walk_num += 1

        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        logger.info(f"Completed {walk_num} walk-forward iterations")

        return results_df
