"""Strict JSON schemas for LLM I/O validation.

Every LLM response MUST conform to one of these schemas.
Invalid responses are rejected and logged — never acted upon.
"""

STRATEGY_SELECTION_SCHEMA = {
    "type": "object",
    "required": ["action", "strategy_id", "reasoning", "confidence"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["SELECT_STRATEGY", "NO_TRADE"],
        },
        "strategy_id": {
            "type": "string",
            "description": "The selected strategy ID, or 'none' if NO_TRADE",
        },
        "reasoning": {
            "type": "string",
            "description": "Human-readable explanation for the decision",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "risk_adjustment": {
            "type": "object",
            "properties": {
                "reduce_position_pct": {"type": "number", "minimum": 0, "maximum": 100},
                "add_stop_loss": {"type": "boolean"},
            },
        },
    },
}

ANOMALY_TRIAGE_SCHEMA = {
    "type": "object",
    "required": ["severity", "action", "reasoning"],
    "properties": {
        "severity": {
            "type": "string",
            "enum": ["INFO", "WARNING", "CRITICAL"],
        },
        "action": {
            "type": "string",
            "enum": ["IGNORE", "MONITOR", "REDUCE_EXPOSURE", "HALT_TRADING"],
        },
        "reasoning": {
            "type": "string",
        },
        "affected_symbols": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

PERFORMANCE_REPORT_SCHEMA = {
    "type": "object",
    "required": ["summary", "key_metrics_commentary", "recommendations"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-3 sentence performance summary",
        },
        "key_metrics_commentary": {
            "type": "object",
            "properties": {
                "sharpe": {"type": "string"},
                "drawdown": {"type": "string"},
                "win_rate": {"type": "string"},
                "regime": {"type": "string"},
            },
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "risk_assessment": {
            "type": "string",
        },
    },
}

NEWS_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": ["sentiment", "relevance", "key_events"],
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["BULLISH", "BEARISH", "NEUTRAL", "MIXED"],
        },
        "relevance": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "key_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "impact": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
                    "magnitude": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                },
            },
        },
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}
