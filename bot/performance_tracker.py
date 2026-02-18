"""Performance tracker: generates KPIs from trade history."""

import numpy as np
from typing import Optional

from loguru import logger


def compute_performance(trades: list[dict], portfolio_value: float = 0, initial_cash: float = 100000) -> dict:
    """Compute performance KPIs from trade history.

    Args:
        trades: List of trade dicts with at least {side, pnl, cost, price, qty, timestamp}
        portfolio_value: Current portfolio value
        initial_cash: Starting cash

    Returns:
        Dict of performance metrics
    """
    sells = [t for t in trades if t.get("side", "").upper() == "SELL" and t.get("pnl") is not None]
    buys = [t for t in trades if t.get("side", "").upper() == "BUY"]

    pnls = [t["pnl"] for t in sells]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]

    total_invested = sum(t.get("cost", 0) for t in buys)
    total_returned = sum(t.get("cost", 0) + (t.get("pnl") or 0) for t in sells)

    total_pnl = sum(pnls) if pnls else 0
    win_rate = len(winning) / len(pnls) if pnls else 0

    if pnls:
        sharpe = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0
    else:
        sharpe = 0

    equity_curve = [initial_cash]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / np.where(running_max > 0, running_max, 1)
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    benchmark_return = 0
    if buys and sells:
        first_buy_price = buys[-1].get("price", 0) if buys else 0
        last_price = sells[0].get("price", 0) if sells else 0
        if first_buy_price > 0:
            benchmark_return = (last_price - first_buy_price) / first_buy_price

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "avg_trade_pnl": float(np.mean(pnls)) if pnls else 0,
        "best_trade_pnl": max(pnls) if pnls else 0,
        "worst_trade_pnl": min(pnls) if pnls else 0,
        "portfolio_value": portfolio_value,
        "benchmark_return": benchmark_return,
        "total_invested": total_invested,
        "total_returned": total_returned,
    }
