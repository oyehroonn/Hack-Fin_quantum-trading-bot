"""LLM governor/orchestrator for the trading system.

The LLM acts as a high-level decision maker:
  1. Governor: selects strategy based on regime + performance
  2. Sentinel: triages anomalies and risk alerts
  3. Reporter: generates human-readable performance reports
  4. Analyst: extracts features from news/events

All actions are validated and audited.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from loguru import logger

from core.interfaces import LLMClient
from core.types import (
    LLMAction,
    LLMActionType,
    ModelMetrics,
    RegimeState,
)
from llm.client import validate_llm_response
from llm.schemas import (
    ANOMALY_TRIAGE_SCHEMA,
    NEWS_ANALYSIS_SCHEMA,
    PERFORMANCE_REPORT_SCHEMA,
    STRATEGY_SELECTION_SCHEMA,
)


class LLMAuditStore:
    """Audit log for all LLM interactions."""

    def __init__(self, log_file: str = "llm_audit.jsonl") -> None:
        self.log_file = log_file

    def log(
        self,
        action_type: str,
        system_prompt: str,
        user_message: str,
        raw_response: str,
        parsed: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "system_prompt_hash": hash(system_prompt),
            "user_message_length": len(user_message),
            "raw_response_length": len(raw_response),
            "parsed_fields": list(parsed.keys()) if parsed else [],
            "error": error,
        }
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")


class TradingGovernor:
    """LLM-powered governor for high-level trading decisions."""

    def __init__(
        self,
        client: LLMClient,
        audit_store: Optional[LLMAuditStore] = None,
    ) -> None:
        self.client = client
        self.audit = audit_store or LLMAuditStore()

    async def select_strategy(
        self,
        regime: RegimeState,
        model_metrics: dict[str, ModelMetrics],
        available_strategies: list[str],
        portfolio_summary: dict[str, Any],
    ) -> LLMAction:
        """Ask the LLM to select the best strategy for current conditions.

        Args:
            regime: Current market regime
            model_metrics: Recent metrics for each strategy
            available_strategies: List of strategy IDs
            portfolio_summary: Current portfolio state
        """
        system_prompt = (
            "You are a quantitative trading governor. "
            "Your job is to select the optimal trading strategy based on "
            "current market regime, recent model performance, and portfolio state. "
            "You are NOT a predictor — you are an allocator and risk manager."
        )

        metrics_summary = {}
        for sid, m in model_metrics.items():
            metrics_summary[sid] = {
                "sharpe": m.sharpe,
                "total_return": m.total_return,
                "max_drawdown": m.max_drawdown,
                "win_rate": m.win_rate,
            }

        user_message = json.dumps({
            "regime": {
                "type": regime.regime.value,
                "confidence": float(regime.confidence),
                "indicators": regime.indicators,
            },
            "strategy_metrics": metrics_summary,
            "available_strategies": available_strategies,
            "portfolio": portfolio_summary,
        }, indent=2, default=str)

        raw = await self.client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            response_schema=STRATEGY_SELECTION_SCHEMA,
        )

        try:
            parsed = validate_llm_response(raw, STRATEGY_SELECTION_SCHEMA)
            self.audit.log("SELECT_STRATEGY", system_prompt, user_message, raw, parsed)

            action_type = (
                LLMActionType.SELECT_STRATEGY
                if parsed["action"] == "SELECT_STRATEGY"
                else LLMActionType.NO_TRADE
            )

            return LLMAction(
                action_type=action_type,
                timestamp=datetime.now(),
                payload={
                    "strategy_id": parsed.get("strategy_id", "none"),
                    "risk_adjustment": parsed.get("risk_adjustment", {}),
                },
                reasoning=parsed.get("reasoning", ""),
                confidence=Decimal(str(parsed.get("confidence", 0.5))),
                raw_output=raw,
            )

        except Exception as e:
            self.audit.log("SELECT_STRATEGY", system_prompt, user_message, raw, error=str(e))
            logger.error(f"LLM strategy selection failed: {e}")
            return LLMAction(
                action_type=LLMActionType.NO_TRADE,
                timestamp=datetime.now(),
                reasoning=f"LLM error: {e}",
                confidence=Decimal("0.0"),
                raw_output=raw,
            )

    async def triage_anomaly(
        self,
        anomaly_description: str,
        recent_alerts: list[dict[str, Any]],
        portfolio_summary: dict[str, Any],
    ) -> LLMAction:
        """Triage a detected anomaly."""
        system_prompt = (
            "You are a trading risk sentinel. Analyze the anomaly and decide "
            "the appropriate response: IGNORE, MONITOR, REDUCE_EXPOSURE, or HALT_TRADING."
        )

        user_message = json.dumps({
            "anomaly": anomaly_description,
            "recent_alerts": recent_alerts[-5:],
            "portfolio": portfolio_summary,
        }, indent=2, default=str)

        raw = await self.client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            response_schema=ANOMALY_TRIAGE_SCHEMA,
        )

        try:
            parsed = validate_llm_response(raw, ANOMALY_TRIAGE_SCHEMA)
            self.audit.log("TRIAGE_ANOMALY", system_prompt, user_message, raw, parsed)

            return LLMAction(
                action_type=LLMActionType.TRIAGE_ANOMALY,
                timestamp=datetime.now(),
                payload={
                    "severity": parsed.get("severity", "INFO"),
                    "action": parsed.get("action", "MONITOR"),
                    "affected_symbols": parsed.get("affected_symbols", []),
                },
                reasoning=parsed.get("reasoning", ""),
                confidence=Decimal("0.5"),
                raw_output=raw,
            )

        except Exception as e:
            self.audit.log("TRIAGE_ANOMALY", system_prompt, user_message, raw, error=str(e))
            return LLMAction(
                action_type=LLMActionType.TRIAGE_ANOMALY,
                timestamp=datetime.now(),
                payload={"severity": "WARNING", "action": "MONITOR"},
                reasoning=f"Triage error: {e}",
                confidence=Decimal("0.0"),
                raw_output=raw,
            )

    async def generate_report(
        self,
        metrics: dict[str, Any],
        regime: Optional[RegimeState] = None,
        recent_trades: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Generate a human-readable performance report."""
        system_prompt = (
            "You are a quantitative analyst. Generate a clear, actionable "
            "performance report from the provided trading metrics."
        )

        user_message = json.dumps({
            "metrics": metrics,
            "regime": {
                "type": regime.regime.value if regime else "UNKNOWN",
                "confidence": float(regime.confidence) if regime else 0,
            },
            "recent_trades_count": len(recent_trades) if recent_trades else 0,
        }, indent=2, default=str)

        raw = await self.client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            response_schema=PERFORMANCE_REPORT_SCHEMA,
        )

        try:
            parsed = validate_llm_response(raw, PERFORMANCE_REPORT_SCHEMA)
            self.audit.log("REPORT", system_prompt, user_message, raw, parsed)
            return parsed

        except Exception as e:
            self.audit.log("REPORT", system_prompt, user_message, raw, error=str(e))
            return {
                "summary": "Report generation failed",
                "key_metrics_commentary": {},
                "recommendations": [f"Error: {e}"],
            }
