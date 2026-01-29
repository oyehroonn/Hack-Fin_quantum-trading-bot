#!/usr/bin/env python3
"""Walk-forward analysis runner script."""

import argparse
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from backtest.strategies.sma_crossover import SMACrossoverStrategy
from backtest.walk_forward import WalkForwardRunner
from infra.logging import setup_logging, set_correlation_id

from loguru import logger


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run walk-forward analysis")
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=252,
        help="Training period in days",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=63,
        help="Test period in days",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=21,
        help="Step size in days",
    )
    parser.add_argument(
        "--fast-period",
        type=int,
        default=10,
        help="Fast SMA period",
    )
    parser.add_argument(
        "--slow-period",
        type=int,
        default=30,
        help="Slow SMA period",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory",
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
    run_id = str(uuid.uuid4())[:8]
    set_correlation_id(f"walk-forward-{run_id}")
    setup_logging(level=args.log_level)

    # Load data
    logger.info(f"Loading data from {args.data_file}")
    data_path = Path(args.data_file)

    if data_path.suffix == ".csv":
        data = pd.read_csv(data_path)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            if "symbol" in data.columns:
                data = data.set_index(["timestamp", "symbol"])
            else:
                data = data.set_index("timestamp")
    elif data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    # Parse dates
    start = None
    if args.start:
        start = datetime.fromisoformat(args.start)

    end = None
    if args.end:
        end = datetime.fromisoformat(args.end)

    # Strategy factory
    def strategy_factory():
        return SMACrossoverStrategy(
            fast_period=args.fast_period,
            slow_period=args.slow_period,
        )

    # Run walk-forward
    runner = WalkForwardRunner(
        initial_cash=Decimal(str(args.initial_cash)),
        strategy_factory=strategy_factory,
        train_period_days=args.train_days,
        test_period_days=args.test_days,
        step_days=args.step_days,
    )

    results_df = runner.run(data, start=start, end=end)

    # Save results
    output_path = Path(args.output_dir) / f"walk_forward_{run_id}"
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")

    # Print summary
    if not results_df.empty:
        logger.info("\nWalk-Forward Summary:")
        logger.info(f"  Number of walks: {len(results_df)}")
        logger.info(f"  Average CAGR: {results_df['cagr'].mean():.2%}")
        logger.info(f"  Average Sharpe: {results_df['sharpe'].mean():.2f}")
        logger.info(f"  Average Max DD: {results_df['max_drawdown'].mean():.2%}")


if __name__ == "__main__":
    main()
