"""Decision engine: combines ML signal + Monte Carlo + statistics to decide BUY/SELL/HOLD."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class Decision:
    action: str           # BUY, SELL, HOLD
    confidence: float     # 0-1
    amount_usd: float     # how much to trade
    reasoning: str        # human-readable explanation
    ml_signal: str
    ml_confidence: float
    mc_var: float
    mc_expected_return: float
    regime: str
    rsi: float
    trend_strength: float


class DecisionEngine:
    """Combines multiple signals into a single trading decision."""

    def __init__(
        self,
        trade_amount: float = 100.0,
        buy_confidence_threshold: float = 0.52,
        sell_confidence_threshold: float = 0.48,
        max_var_95: float = -0.25,
        max_portfolio_pct: float = 0.50,
        stop_loss_pct: float = -0.08,
        take_profit_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        rsi_overbought: float = 72.0,
        rsi_oversold: float = 28.0,
    ):
        self.trade_amount = trade_amount
        self.buy_confidence_threshold = buy_confidence_threshold
        self.sell_confidence_threshold = sell_confidence_threshold
        self.max_var_95 = max_var_95
        self.max_portfolio_pct = max_portfolio_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def decide(
        self,
        ml_signal: str,
        ml_confidence: float,
        mc_result: Optional[dict],
        stats: Optional[dict],
        position: Optional[dict],
        cash: float,
        portfolio_value: float,
        recent_trades: list[dict],
    ) -> Decision:
        """
        Make a trading decision.

        Args:
            ml_signal: "BUY", "SELL", or "HOLD" from ML model
            ml_confidence: 0-1 confidence from ML model
            mc_result: Monte Carlo result dict (from MonteCarloResult.to_dict())
            stats: Statistical analysis dict (from full_statistical_analysis())
            position: Current position dict {"qty", "avg_cost"} or None
            cash: Available cash
            portfolio_value: Total portfolio value
            recent_trades: Recent trades list for streak analysis
        """
        reasons = []
        mc_var = mc_result.get("var_95", 0) if mc_result else 0
        mc_expected = mc_result.get("expected_return", 0) if mc_result else 0
        mc_prob_profit = mc_result.get("prob_profit", 0.5) if mc_result else 0.5
        regime = stats.get("regime", "normal") if stats else "normal"
        rsi = stats.get("rsi", 50) if stats else 50
        trend = stats.get("trend_strength", 0) if stats else 0

        consecutive_losses = self._count_consecutive_losses(recent_trades)
        has_position = position is not None and position.get("qty", 0) > 0

        # --- SELL logic (check first) ---
        if has_position:
            avg_cost = position.get("avg_cost", 0)
            current_price = stats.get("current_price", avg_cost) if stats else avg_cost
            if avg_cost > 0:
                position_return = (current_price - avg_cost) / avg_cost
            else:
                position_return = 0

            if position_return <= self.stop_loss_pct:
                reasons.append(f"Stop-loss triggered: position return {position_return:.2%} <= {self.stop_loss_pct:.2%}")
                return self._make_decision("SELL", 0.9, reasons, ml_signal, ml_confidence, mc_var, mc_expected, regime, rsi, trend)

            if position_return >= self.take_profit_pct:
                reasons.append(f"Take-profit triggered: position return {position_return:.2%} >= {self.take_profit_pct:.2%}")
                return self._make_decision("SELL", 0.85, reasons, ml_signal, ml_confidence, mc_var, mc_expected, regime, rsi, trend)

            if ml_signal == "SELL" and ml_confidence >= self.sell_confidence_threshold:
                reasons.append(f"ML signal SELL with confidence {ml_confidence:.2%}")
                return self._make_decision("SELL", ml_confidence, reasons, ml_signal, ml_confidence, mc_var, mc_expected, regime, rsi, trend)

            if rsi >= self.rsi_overbought:
                reasons.append(f"RSI overbought: {rsi:.1f}")
                if ml_signal != "BUY":
                    return self._make_decision("SELL", 0.6, reasons, ml_signal, ml_confidence, mc_var, mc_expected, regime, rsi, trend)

        # --- BUY logic ---
        if not has_position or (has_position and position.get("qty", 0) * stats.get("current_price", 0) < portfolio_value * self.max_portfolio_pct if stats else True):
            buy_ok = True
            buy_conf = ml_confidence

            oversold_opportunity = rsi <= self.rsi_oversold
            positive_mc = mc_prob_profit >= 0.40 or mc_expected > -0.02

            if ml_signal == "BUY":
                reasons.append(f"ML signal BUY with confidence {ml_confidence:.2%}")
            elif oversold_opportunity and positive_mc:
                reasons.append(f"RSI oversold ({rsi:.1f}) + acceptable MC — mean-reversion opportunity")
                buy_conf = 0.55
            else:
                buy_ok = False
                reasons.append(f"ML signal is {ml_signal}, not BUY")

            if ml_signal == "BUY" and ml_confidence < self.buy_confidence_threshold:
                buy_ok = False
                reasons.append(f"ML confidence {ml_confidence:.2%} < threshold {self.buy_confidence_threshold:.2%}")

            if mc_var < self.max_var_95:
                reasons.append(f"Monte Carlo VaR(95%) {mc_var:.2%} (limit {self.max_var_95:.2%})")
                if not oversold_opportunity:
                    buy_ok = False

            if mc_expected < -0.05 and mc_prob_profit < 0.35:
                buy_ok = False
                reasons.append(f"Monte Carlo expected return {mc_expected:.2%}, prob_profit {mc_prob_profit:.2%} — too bearish")

            if consecutive_losses >= self.max_consecutive_losses:
                buy_ok = False
                reasons.append(f"Consecutive losses ({consecutive_losses}) >= max ({self.max_consecutive_losses})")

            if cash < self.trade_amount:
                buy_ok = False
                reasons.append(f"Insufficient cash: ${cash:.2f} < ${self.trade_amount:.2f}")

            if regime == "high_vol":
                buy_conf *= 0.8
                reasons.append("High volatility regime — reduced confidence")

            if rsi <= self.rsi_oversold and ml_signal == "BUY":
                buy_conf = min(buy_conf * 1.1, 1.0)
                reasons.append(f"RSI oversold ({rsi:.1f}) — boost")

            if buy_ok:
                reasons.append(f"BUY signal confirmed: ML={ml_signal}({ml_confidence:.2%}), MC_exp={mc_expected:.2%}, VaR={mc_var:.2%}, regime={regime}")
                return self._make_decision("BUY", buy_conf, reasons, ml_signal, ml_confidence, mc_var, mc_expected, regime, rsi, trend)

        # --- HOLD ---
        reasons.append("No clear BUY or SELL signal — HOLD")
        return self._make_decision("HOLD", 0.5, reasons, ml_signal, ml_confidence, mc_var, mc_expected, regime, rsi, trend)

    def _make_decision(self, action, confidence, reasons, ml_signal, ml_conf, mc_var, mc_exp, regime, rsi, trend) -> Decision:
        amount = self.trade_amount if action == "BUY" else 0
        return Decision(
            action=action,
            confidence=confidence,
            amount_usd=amount,
            reasoning=" | ".join(reasons),
            ml_signal=ml_signal,
            ml_confidence=ml_conf,
            mc_var=mc_var,
            mc_expected_return=mc_exp,
            regime=regime,
            rsi=rsi,
            trend_strength=trend,
        )

    def _count_consecutive_losses(self, trades: list[dict]) -> int:
        count = 0
        for t in trades:
            if t.get("side", "").upper() == "SELL" and (t.get("pnl") or 0) < 0:
                count += 1
            else:
                break
        return count
