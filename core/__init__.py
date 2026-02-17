"""Core types, interfaces, and errors for the trading system (v0.2)."""

from core.types import (
    Alert,
    AlertSeverity,
    AlertType,
    Bar,
    ExperimentResult,
    FeatureVector,
    Fill,
    LLMAction,
    LLMActionType,
    ModelDecision,
    ModelMetrics,
    Order,
    Position,
    PortfolioState,
    Regime,
    RegimeState,
    RiskLimits,
    Signal,
    StrategyAllocation,
    Tick,
)
from core.interfaces import (
    Allocator,
    AlertSink,
    Broker,
    DataSource,
    Evaluator,
    FeatureStore,
    LLMClient,
    Model,
    ModelRegistry,
    RegimeDetector,
    RiskManager,
    Strategy,
)
from core.errors import (
    TradingError,
    DataError,
    ModelError,
    ExecutionError,
    RiskError,
    ConfigError,
    LLMError,
)

__all__ = [
    # Types
    "Alert", "AlertSeverity", "AlertType",
    "Bar", "Tick", "Order", "Fill", "Position", "PortfolioState", "Signal",
    "FeatureVector", "Regime", "RegimeState",
    "ModelDecision", "ModelMetrics", "ExperimentResult",
    "StrategyAllocation", "RiskLimits",
    "LLMAction", "LLMActionType",
    # Interfaces
    "DataSource", "FeatureStore", "Model", "Strategy", "Broker", "RiskManager",
    "ModelRegistry", "Evaluator", "Allocator", "AlertSink", "LLMClient", "RegimeDetector",
    # Errors
    "TradingError", "DataError", "ModelError", "ExecutionError", "RiskError", "ConfigError", "LLMError",
]
