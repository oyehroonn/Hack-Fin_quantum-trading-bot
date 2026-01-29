#!/usr/bin/env python3
"""Backtest runner script."""

import argparse
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.report import PerformanceReport
from backtest.strategies.sma_crossover import SMACrossoverStrategy
from infra.logging import setup_logging, set_correlation_id

from loguru import logger


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run backtest")
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
        "--use-pyfolio",
        action="store_true",
        help="Generate pyfolio tear sheet",
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
    set_correlation_id(f"backtest-{run_id}")
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

    # Create strategy
    strategy = SMACrossoverStrategy(
        fast_period=args.fast_period,
        slow_period=args.slow_period,
    )

    # Get symbols
    if isinstance(data.index, pd.MultiIndex):
        symbols = list(data.index.get_level_values(1).unique())
    else:
        symbols = ["UNKNOWN"]

    # Run backtest
    engine = BacktestEngine(
        initial_cash=Decimal(str(args.initial_cash)),
        strategy=strategy,
        symbols=symbols,
    )

    portfolio = engine.run(data, start=start, end=end)

    # Generate report
    report = PerformanceReport(portfolio)
    metrics = report.calculate_metrics()

    # Print metrics
    logger.info("\nBacktest Results:")
    logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    logger.info(f"  CAGR: {metrics.get('cagr', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {metrics.get('sharpe', 0):.2f}")
    logger.info(f"  Sortino Ratio: {metrics.get('sortino', 0):.2f}")
    logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    logger.info(f"  Number of Trades: {metrics.get('num_trades', 0)}")

    # Save report
    output_path = report.save_report(run_id, args.output_dir, use_pyfolio=args.use_pyfolio)
    logger.info(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
