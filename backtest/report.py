"""Performance reporting and metrics."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from backtest.accounting import Portfolio


class PerformanceReport:
    """Performance report generator."""

    def __init__(self, portfolio: Portfolio) -> None:
        """Initialize performance report.

        Args:
            portfolio: Portfolio with backtest results
        """
        self.portfolio = portfolio

    def calculate_metrics(self) -> dict[str, float]:
        """Calculate performance metrics.

        Returns:
            Dictionary of metrics
        """
        equity_df = self.portfolio.get_equity_curve_df()
        if equity_df.empty or len(equity_df) < 2:
            return {}

        # Calculate returns
        equity_df["returns"] = equity_df["equity"].pct_change()
        equity_df["returns"] = equity_df["returns"].fillna(0)

        # Remove first row (initial equity)
        equity_df = equity_df.iloc[1:].copy()

        if len(equity_df) == 0:
            return {}

        returns = equity_df["returns"].values
        equity = equity_df["equity"].values

        # Time period
        start_date = equity_df["timestamp"].iloc[0]
        end_date = equity_df["timestamp"].iloc[-1]
        years = (end_date - start_date).days / 365.25

        if years == 0:
            years = 1.0

        # Total return
        total_return = (equity[-1] / equity[0]) - 1.0

        # CAGR
        cagr = ((equity[-1] / equity[0]) ** (1 / years)) - 1.0 if years > 0 else 0.0

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = (cagr / volatility) if volatility > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        sortino = (cagr / downside_std) if downside_std > 0 else 0.0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        # Calmar ratio
        calmar = (cagr / max_drawdown) if max_drawdown > 0 else 0.0

        # Win rate and average win/loss
        trades_df = self.portfolio.get_trades_df()
        if not trades_df.empty and "pnl" in trades_df.columns:
            pnl = trades_df["pnl"].values
            winning_trades = pnl[pnl > 0]
            losing_trades = pnl[pnl < 0]

            win_rate = len(winning_trades) / len(pnl) if len(pnl) > 0 else 0.0
            avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
            avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
            avg_win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            avg_win_loss = 0.0

        # Turnover (simplified: sum of absolute trades / average equity)
        if not trades_df.empty and "quantity" in trades_df.columns:
            total_turnover = trades_df["quantity"].abs().sum()
            avg_equity = equity_df["equity"].mean()
            turnover = float(total_turnover / avg_equity) if avg_equity > 0 else 0.0
        else:
            turnover = 0.0

        metrics = {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar": float(calmar),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_win_loss": float(avg_win_loss),
            "turnover": float(turnover),
            "num_trades": len(trades_df) if not trades_df.empty else 0,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "years": float(years),
        }

        return metrics

    def save_report(
        self,
        run_id: str,
        output_dir: str = "runs",
        use_pyfolio: bool = False,
    ) -> Path:
        """Save performance report to files.

        Args:
            run_id: Unique run identifier
            output_dir: Output directory
            use_pyfolio: Whether to generate pyfolio tear sheet

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir) / run_id
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Save metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

        # Save equity curve
        equity_df = self.portfolio.get_equity_curve_df()
        equity_file = output_path / "equity_curve.csv"
        equity_df.to_csv(equity_file, index=False)
        logger.info(f"Saved equity curve to {equity_file}")

        # Save trades
        trades_df = self.portfolio.get_trades_df()
        if not trades_df.empty:
            trades_file = output_path / "trades.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")

        # Try pyfolio integration
        if use_pyfolio:
            try:
                import pyfolio as pf

                # Convert equity curve to returns
                equity_df = self.portfolio.get_equity_curve_df()
                if not equity_df.empty:
                    equity_df = equity_df.set_index("timestamp")
                    equity_df["returns"] = equity_df["equity"].pct_change()

                    # Generate tear sheet
                    tear_sheet_file = output_path / "tear_sheet.html"
                    pf.create_full_tear_sheet(
                        equity_df["returns"],
                        benchmark_rets=None,
                    )
                    logger.info(f"Generated pyfolio tear sheet (saved to {tear_sheet_file})")
            except ImportError:
                logger.warning("pyfolio not installed, skipping tear sheet generation")

        return output_path
