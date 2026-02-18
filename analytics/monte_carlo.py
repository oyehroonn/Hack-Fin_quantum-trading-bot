"""Monte Carlo simulation engine for price path forecasting.

Supports Geometric Brownian Motion (GBM) and jump-diffusion models.
Outputs: VaR, CVaR, expected return, confidence intervals, path percentiles.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger


@dataclass
class MonteCarloResult:
    symbol: str
    num_paths: int
    horizon_days: int
    mu: float
    sigma: float
    current_price: float
    expected_price: float
    median_price: float
    expected_return: float
    median_return: float
    var_95: float
    var_99: float
    cvar_95: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    max_drawdown_median: float
    prob_profit: float

    def to_dict(self):
        return asdict(self)


def estimate_params(prices: np.ndarray, annualize: bool = True) -> tuple[float, float]:
    """Estimate drift (mu) and volatility (sigma) from price history."""
    log_returns = np.diff(np.log(prices))
    log_returns = log_returns[np.isfinite(log_returns)]
    if len(log_returns) < 10:
        return 0.0, 0.01

    mu = np.mean(log_returns)
    sigma = np.std(log_returns, ddof=1)
    if annualize:
        mu *= 365
        sigma *= np.sqrt(365)
    return float(mu), float(sigma)


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    num_paths: int = 10000,
    dt: float = 1 / 365,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Geometric Brownian Motion simulation.

    Returns: (num_paths, horizon_days+1) array of price paths.
    """
    rng = np.random.default_rng(seed)
    n_steps = horizon_days
    Z = rng.standard_normal((num_paths, n_steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_increments = drift + diffusion
    log_paths = np.zeros((num_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(log_increments, axis=1)

    return np.exp(log_paths)


def simulate_jump_diffusion(
    s0: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    num_paths: int = 10000,
    jump_intensity: float = 0.1,
    jump_mean: float = -0.02,
    jump_std: float = 0.05,
    dt: float = 1 / 365,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Merton jump-diffusion model."""
    rng = np.random.default_rng(seed)
    n_steps = horizon_days
    Z = rng.standard_normal((num_paths, n_steps))
    J_count = rng.poisson(jump_intensity * dt, (num_paths, n_steps))
    J_size = rng.normal(jump_mean, jump_std, (num_paths, n_steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    jumps = J_count * J_size

    log_increments = drift + diffusion + jumps
    log_paths = np.zeros((num_paths, n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    log_paths[:, 1:] = np.log(s0) + np.cumsum(log_increments, axis=1)

    return np.exp(log_paths)


def compute_max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each path."""
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    return np.min(drawdowns, axis=1)


def run_monte_carlo(
    symbol: str,
    prices: np.ndarray,
    horizon_days: int = 30,
    num_paths: int = 10000,
    model: str = "gbm",
) -> MonteCarloResult:
    """Run full Monte Carlo simulation and return summary statistics."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 20:
        raise ValueError(f"Need at least 20 prices, got {len(prices)}")

    s0 = float(prices[-1])
    mu, sigma = estimate_params(prices)

    logger.info(f"Monte Carlo: {symbol}, S0={s0:.2f}, mu={mu:.4f}, sigma={sigma:.4f}, {num_paths} paths x {horizon_days}d")

    if model == "jump_diffusion":
        paths = simulate_jump_diffusion(s0, mu, sigma, horizon_days, num_paths)
    else:
        paths = simulate_gbm(s0, mu, sigma, horizon_days, num_paths)

    final_prices = paths[:, -1]
    final_returns = (final_prices - s0) / s0

    var_95 = float(np.percentile(final_returns, 5))
    var_99 = float(np.percentile(final_returns, 1))
    tail = final_returns[final_returns <= np.percentile(final_returns, 5)]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    max_dd = compute_max_drawdown(paths)

    return MonteCarloResult(
        symbol=symbol,
        num_paths=num_paths,
        horizon_days=horizon_days,
        mu=mu,
        sigma=sigma,
        current_price=s0,
        expected_price=float(np.mean(final_prices)),
        median_price=float(np.median(final_prices)),
        expected_return=float(np.mean(final_returns)),
        median_return=float(np.median(final_returns)),
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        percentile_5=float(np.percentile(final_prices, 5)),
        percentile_25=float(np.percentile(final_prices, 25)),
        percentile_75=float(np.percentile(final_prices, 75)),
        percentile_95=float(np.percentile(final_prices, 95)),
        max_drawdown_median=float(np.median(max_dd)),
        prob_profit=float(np.mean(final_returns > 0)),
    )
