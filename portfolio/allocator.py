"""Portfolio allocation: multi-strategy and multi-asset capital allocation.

Implements several allocation schemes:
  - Equal weight
  - Risk parity (inverse-variance)
  - Regime-aware (shift weights based on market regime)
  - Kelly criterion (optimal sizing)
"""

import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Optional

from loguru import logger

from core.interfaces import Allocator
from core.types import (
    ModelDecision,
    PortfolioState,
    Regime,
    RegimeState,
    StrategyAllocation,
)


class EqualWeightAllocator(Allocator):
    """Allocate equally across all strategies and symbols."""

    def allocate(
        self,
        decisions: dict[str, list[ModelDecision]],
        regime: Optional[RegimeState] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> StrategyAllocation:
        n_strategies = len(decisions) or 1
        strategy_weight = Decimal(str(round(1.0 / n_strategies, 6)))

        strategy_weights = {sid: strategy_weight for sid in decisions}

        # Combine model signals into asset weights
        asset_weights: dict[str, Decimal] = {}
        for sid, model_decisions in decisions.items():
            for d in model_decisions:
                current = asset_weights.get(d.symbol, Decimal("0"))
                asset_weights[d.symbol] = current + d.signal.strength * strategy_weight

        return StrategyAllocation(
            timestamp=datetime.now(),
            strategy_weights=strategy_weights,
            asset_weights=asset_weights,
            regime=regime,
            reason="equal_weight",
        )


class RiskParityAllocator(Allocator):
    """Allocate inversely proportional to recent realised volatility.

    Lower-vol strategies/assets get higher allocation.
    """

    def __init__(
        self,
        vol_history: Optional[dict[str, list[float]]] = None,
        vol_lookback: int = 20,
        max_strategy_weight: float = 0.5,
    ) -> None:
        self.vol_history = vol_history or {}
        self.vol_lookback = vol_lookback
        self.max_strategy_weight = max_strategy_weight

    def allocate(
        self,
        decisions: dict[str, list[ModelDecision]],
        regime: Optional[RegimeState] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> StrategyAllocation:
        # Estimate vol per strategy (from signal history)
        vols: dict[str, float] = {}
        for sid in decisions:
            hist = self.vol_history.get(sid, [])
            if len(hist) >= 2:
                rets = np.diff(hist) / np.array(hist[:-1])
                vols[sid] = max(float(np.std(rets)), 1e-6)
            else:
                vols[sid] = 0.01  # Default vol

        # Inverse volatility weights
        inv_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())

        strategy_weights: dict[str, Decimal] = {}
        for sid, iv in inv_vols.items():
            w = min(iv / total_inv_vol, self.max_strategy_weight)
            strategy_weights[sid] = Decimal(str(round(w, 6)))

        # Normalise if capped weights reduced total
        total = sum(strategy_weights.values())
        if total > 0:
            strategy_weights = {
                sid: Decimal(str(round(float(w) / float(total), 6)))
                for sid, w in strategy_weights.items()
            }

        # Asset weights
        asset_weights: dict[str, Decimal] = {}
        for sid, model_decisions in decisions.items():
            sw = strategy_weights.get(sid, Decimal("0"))
            for d in model_decisions:
                current = asset_weights.get(d.symbol, Decimal("0"))
                asset_weights[d.symbol] = current + d.signal.strength * sw

        return StrategyAllocation(
            timestamp=datetime.now(),
            strategy_weights=strategy_weights,
            asset_weights=asset_weights,
            regime=regime,
            reason="risk_parity",
        )

    def update_vol(self, strategy_id: str, pnl: float) -> None:
        """Update PnL history for vol estimation."""
        if strategy_id not in self.vol_history:
            self.vol_history[strategy_id] = []
        self.vol_history[strategy_id].append(pnl)
        if len(self.vol_history[strategy_id]) > self.vol_lookback * 2:
            self.vol_history[strategy_id] = self.vol_history[strategy_id][-self.vol_lookback * 2:]


class RegimeAwareAllocator(Allocator):
    """Shift allocation weights based on detected market regime.

    Config maps regime → strategy weight overrides.
    """

    def __init__(
        self,
        regime_weights: Optional[dict[str, dict[str, float]]] = None,
        fallback_allocator: Optional[Allocator] = None,
    ) -> None:
        """Initialize regime-aware allocator.

        Args:
            regime_weights: {regime_name: {strategy_id: weight}}
            fallback_allocator: Allocator to use when regime is UNKNOWN
        """
        self.regime_weights = regime_weights or {
            Regime.TRENDING_UP.value: {"sma_crossover": 0.4, "breakout": 0.4, "mean_reversion": 0.1, "ml": 0.1},
            Regime.TRENDING_DOWN.value: {"sma_crossover": 0.4, "breakout": 0.3, "mean_reversion": 0.1, "ml": 0.2},
            Regime.MEAN_REVERTING.value: {"mean_reversion": 0.5, "ml": 0.3, "sma_crossover": 0.1, "breakout": 0.1},
            Regime.HIGH_VOLATILITY.value: {"breakout": 0.4, "ml": 0.3, "mean_reversion": 0.2, "sma_crossover": 0.1},
            Regime.LOW_VOLATILITY.value: {"mean_reversion": 0.4, "ml": 0.3, "sma_crossover": 0.2, "breakout": 0.1},
        }
        self.fallback = fallback_allocator or EqualWeightAllocator()

    def allocate(
        self,
        decisions: dict[str, list[ModelDecision]],
        regime: Optional[RegimeState] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> StrategyAllocation:
        if regime is None or regime.regime == Regime.UNKNOWN:
            return self.fallback.allocate(decisions, regime, portfolio_state)

        weight_config = self.regime_weights.get(regime.regime.value, {})

        # Apply configured weights, filling missing with equal share
        strategy_weights: dict[str, Decimal] = {}
        assigned_strategies = set()

        for sid in decisions:
            if sid in weight_config:
                strategy_weights[sid] = Decimal(str(round(weight_config[sid], 6)))
                assigned_strategies.add(sid)

        # Unassigned strategies get equal share of remainder
        unassigned = [sid for sid in decisions if sid not in assigned_strategies]
        if unassigned:
            assigned_total = sum(float(w) for w in strategy_weights.values())
            remainder = max(1.0 - assigned_total, 0) / len(unassigned)
            for sid in unassigned:
                strategy_weights[sid] = Decimal(str(round(remainder, 6)))

        # Asset weights
        asset_weights: dict[str, Decimal] = {}
        for sid, model_decisions in decisions.items():
            sw = strategy_weights.get(sid, Decimal("0"))
            for d in model_decisions:
                current = asset_weights.get(d.symbol, Decimal("0"))
                asset_weights[d.symbol] = current + d.signal.strength * sw

        return StrategyAllocation(
            timestamp=datetime.now(),
            strategy_weights=strategy_weights,
            asset_weights=asset_weights,
            regime=regime,
            reason=f"regime_aware:{regime.regime.value}",
        )


class KellyAllocator(Allocator):
    """Kelly criterion-based position sizing.

    Allocates based on edge/odds: f* = (bp - q) / b
    where b=odds, p=win_prob, q=1-p.

    Half-Kelly is used by default for safety.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_position: float = 0.25,
    ) -> None:
        self.fraction = fraction  # Half-Kelly default
        self.max_position = max_position

    def allocate(
        self,
        decisions: dict[str, list[ModelDecision]],
        regime: Optional[RegimeState] = None,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> StrategyAllocation:
        strategy_weights: dict[str, Decimal] = {}
        asset_weights: dict[str, Decimal] = {}

        for sid, model_decisions in decisions.items():
            strategy_weights[sid] = Decimal("1") / Decimal(str(max(len(decisions), 1)))

            for d in model_decisions:
                if d.probability is None:
                    continue

                p = float(d.probability)
                q = 1 - p

                # Assume 1:1 odds (b=1) for simplicity
                b = 1.0
                kelly_fraction = (b * p - q) / b

                # Apply half-Kelly and cap
                position_size = kelly_fraction * self.fraction
                position_size = max(min(position_size, self.max_position), -self.max_position)

                current = asset_weights.get(d.symbol, Decimal("0"))
                asset_weights[d.symbol] = current + Decimal(str(round(position_size, 6)))

        return StrategyAllocation(
            timestamp=datetime.now(),
            strategy_weights=strategy_weights,
            asset_weights=asset_weights,
            regime=regime,
            reason="kelly_criterion",
        )
