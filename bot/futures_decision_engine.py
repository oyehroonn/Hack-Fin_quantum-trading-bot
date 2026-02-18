"""Futures decision engine: LONG/SHORT with dynamic leverage based on risk."""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from loguru import logger


@dataclass
class FuturesDecision:
    action: str           # OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
    direction: str        # LONG, SHORT, or NONE
    leverage: int         # 1-100x
    margin: float         # Collateral amount
    position_size: float  # margin × leverage
    stop_loss: float      # Auto-close price
    take_profit: float    # Target price
    entry_price: float    # Current price for entry
    liquidation_price: float
    confidence: float     # 0-1
    risk_score: float     # 0-1 (lower = safer)
    reasoning: str
    ml_signal: str
    mc_expected_return: float
    mc_var: float
    regime: str


class FuturesDecisionEngine:
    """Decides LONG/SHORT positions with dynamic leverage."""

    def __init__(
        self,
        base_margin: float = 100.0,
        max_leverage: int = 100,
        min_leverage: int = 2,
        max_risk_per_trade: float = 0.05,
        confidence_threshold: float = 0.55,
        stop_loss_pct: float = 0.02,
        take_profit_multiplier: float = 2.0,
    ):
        self.base_margin = base_margin
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_multiplier = take_profit_multiplier

    def calculate_risk_score(
        self,
        mc_var: float,
        volatility: float,
        regime: str,
        confidence: float,
    ) -> float:
        """Calculate risk score 0-1 (lower = safer, can use more leverage)."""
        var_risk = min(abs(mc_var) / 0.20, 1.0)
        vol_risk = min(volatility / 1.0, 1.0)
        regime_risk = {"low_vol": 0.2, "normal": 0.5, "high_vol": 0.8}.get(regime, 0.5)
        confidence_factor = 1 - confidence

        risk = 0.3 * var_risk + 0.25 * vol_risk + 0.25 * regime_risk + 0.2 * confidence_factor
        return min(max(risk, 0), 1)

    def calculate_leverage(self, risk_score: float, regime: str) -> int:
        """Calculate optimal leverage based on risk score."""
        regime_max = {
            "low_vol": self.max_leverage,
            "normal": min(50, self.max_leverage),
            "high_vol": min(20, self.max_leverage),
        }.get(regime, 25)

        if risk_score < 0.25:
            leverage = int(regime_max * 0.8)
        elif risk_score < 0.40:
            leverage = int(regime_max * 0.5)
        elif risk_score < 0.55:
            leverage = int(regime_max * 0.3)
        elif risk_score < 0.70:
            leverage = int(regime_max * 0.15)
        else:
            leverage = self.min_leverage

        return max(self.min_leverage, min(leverage, self.max_leverage))

    def calculate_stops(
        self,
        direction: str,
        entry_price: float,
        leverage: int,
    ) -> tuple[float, float, float]:
        """Calculate stop-loss, take-profit, and liquidation price."""
        effective_sl_pct = self.stop_loss_pct * (20 / leverage)
        tp_pct = effective_sl_pct * self.take_profit_multiplier

        if direction == "LONG":
            stop_loss = entry_price * (1 - effective_sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            liquidation_price = entry_price * (1 - 0.9 / leverage)
        else:
            stop_loss = entry_price * (1 + effective_sl_pct)
            take_profit = entry_price * (1 - tp_pct)
            liquidation_price = entry_price * (1 + 0.9 / leverage)

        return stop_loss, take_profit, liquidation_price

    def decide(
        self,
        ml_signal: str,
        ml_confidence: float,
        mc_result: Optional[dict],
        stats: Optional[dict],
        open_positions: List[dict],
        available_margin: float,
        current_price: float,
    ) -> FuturesDecision:
        """Make a futures trading decision."""
        reasons = []
        mc_var = mc_result.get("var_95", 0) if mc_result else 0
        mc_expected = mc_result.get("expected_return", 0) if mc_result else 0
        mc_prob_profit = mc_result.get("prob_profit", 0.5) if mc_result else 0.5
        regime = stats.get("regime", "normal") if stats else "normal"
        volatility = stats.get("annualized_volatility", 0.5) if stats else 0.5
        rsi = stats.get("rsi", 50) if stats else 50

        has_long = any(p.get("direction") == "LONG" for p in open_positions)
        has_short = any(p.get("direction") == "SHORT" for p in open_positions)

        for pos in open_positions:
            if pos.get("direction") == "LONG":
                unrealized_return = (current_price - pos.get("entry_price", current_price)) / pos.get("entry_price", current_price)
                if unrealized_return >= self.stop_loss_pct * self.take_profit_multiplier:
                    reasons.append(f"LONG position hit take-profit ({unrealized_return:.2%})")
                    return self._make_decision("CLOSE_LONG", "LONG", 0, 0, current_price, 0, 0, 0, 0.9, 0, reasons, ml_signal, mc_expected, mc_var, regime)
                if unrealized_return <= -self.stop_loss_pct * 2:
                    reasons.append(f"LONG position hit stop-loss ({unrealized_return:.2%})")
                    return self._make_decision("CLOSE_LONG", "LONG", 0, 0, current_price, 0, 0, 0, 0.9, 0, reasons, ml_signal, mc_expected, mc_var, regime)

            elif pos.get("direction") == "SHORT":
                unrealized_return = (pos.get("entry_price", current_price) - current_price) / pos.get("entry_price", current_price)
                if unrealized_return >= self.stop_loss_pct * self.take_profit_multiplier:
                    reasons.append(f"SHORT position hit take-profit ({unrealized_return:.2%})")
                    return self._make_decision("CLOSE_SHORT", "SHORT", 0, 0, current_price, 0, 0, 0, 0.9, 0, reasons, ml_signal, mc_expected, mc_var, regime)
                if unrealized_return <= -self.stop_loss_pct * 2:
                    reasons.append(f"SHORT position hit stop-loss ({unrealized_return:.2%})")
                    return self._make_decision("CLOSE_SHORT", "SHORT", 0, 0, current_price, 0, 0, 0, 0.9, 0, reasons, ml_signal, mc_expected, mc_var, regime)

        if has_long and ml_signal == "SELL" and ml_confidence >= 0.60:
            reasons.append(f"Strong SELL signal ({ml_confidence:.0%}) while holding LONG — close position")
            return self._make_decision("CLOSE_LONG", "LONG", 0, 0, current_price, 0, 0, 0, ml_confidence, 0, reasons, ml_signal, mc_expected, mc_var, regime)

        if has_short and ml_signal == "BUY" and ml_confidence >= 0.60:
            reasons.append(f"Strong BUY signal ({ml_confidence:.0%}) while holding SHORT — close position")
            return self._make_decision("CLOSE_SHORT", "SHORT", 0, 0, current_price, 0, 0, 0, ml_confidence, 0, reasons, ml_signal, mc_expected, mc_var, regime)

        if has_long or has_short:
            reasons.append("Already have open position — HOLD")
            return self._make_decision("HOLD", "NONE", 0, 0, current_price, 0, 0, 0, 0.5, 0.5, reasons, ml_signal, mc_expected, mc_var, regime)

        if available_margin < self.base_margin:
            reasons.append(f"Insufficient margin: ${available_margin:.2f} < ${self.base_margin:.2f}")
            return self._make_decision("HOLD", "NONE", 0, 0, current_price, 0, 0, 0, 0.5, 1.0, reasons, ml_signal, mc_expected, mc_var, regime)

        risk_score = self.calculate_risk_score(mc_var, volatility, regime, ml_confidence)
        leverage = self.calculate_leverage(risk_score, regime)
        margin = min(self.base_margin, available_margin * self.max_risk_per_trade)
        position_size = margin * leverage

        if ml_signal == "SELL" and ml_confidence >= self.confidence_threshold:
            direction = "SHORT"
            reasons.append(f"ML signal SELL ({ml_confidence:.0%}) — OPEN SHORT")
            reasons.append(f"MC expected return {mc_expected:.2%}, VaR {mc_var:.2%}")
            reasons.append(f"Risk score {risk_score:.2f} → {leverage}x leverage")

            if mc_expected > 0.02:
                reasons.append(f"Warning: MC expects +{mc_expected:.2%}, but ML says SELL")

        elif ml_signal == "BUY" and ml_confidence >= self.confidence_threshold:
            direction = "LONG"
            reasons.append(f"ML signal BUY ({ml_confidence:.0%}) — OPEN LONG")
            reasons.append(f"MC expected return {mc_expected:.2%}, P(profit) {mc_prob_profit:.0%}")
            reasons.append(f"Risk score {risk_score:.2f} → {leverage}x leverage")

            if mc_expected < -0.02:
                reasons.append(f"Warning: MC expects {mc_expected:.2%}, but ML says BUY")

        elif rsi <= 25:
            direction = "LONG"
            leverage = max(self.min_leverage, leverage // 2)
            reasons.append(f"RSI oversold ({rsi:.1f}) — mean-reversion LONG")
            reasons.append(f"Reduced leverage to {leverage}x due to counter-trend")

        elif rsi >= 75:
            direction = "SHORT"
            leverage = max(self.min_leverage, leverage // 2)
            reasons.append(f"RSI overbought ({rsi:.1f}) — mean-reversion SHORT")
            reasons.append(f"Reduced leverage to {leverage}x due to counter-trend")

        else:
            reasons.append(f"No clear signal: ML={ml_signal} ({ml_confidence:.0%}), RSI={rsi:.1f}")
            return self._make_decision("HOLD", "NONE", 0, 0, current_price, 0, 0, 0, 0.5, risk_score, reasons, ml_signal, mc_expected, mc_var, regime)

        stop_loss, take_profit, liquidation_price = self.calculate_stops(direction, current_price, leverage)
        action = f"OPEN_{direction}"

        return self._make_decision(
            action, direction, leverage, margin, current_price,
            stop_loss, take_profit, liquidation_price,
            ml_confidence, risk_score, reasons, ml_signal, mc_expected, mc_var, regime
        )

    def _make_decision(
        self, action, direction, leverage, margin, entry_price,
        stop_loss, take_profit, liquidation_price, confidence, risk_score,
        reasons, ml_signal, mc_expected, mc_var, regime
    ) -> FuturesDecision:
        return FuturesDecision(
            action=action,
            direction=direction,
            leverage=leverage,
            margin=margin,
            position_size=margin * leverage if leverage > 0 else 0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=entry_price,
            liquidation_price=liquidation_price,
            confidence=confidence,
            risk_score=risk_score,
            reasoning=" | ".join(reasons),
            ml_signal=ml_signal,
            mc_expected_return=mc_expected,
            mc_var=mc_var,
            regime=regime,
        )
