"""Custom error hierarchy for the trading system."""

from typing import Optional


class TradingError(Exception):
    """Base error for all trading system errors."""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        self.code = code or self.__class__.__name__
        super().__init__(message)


# --- Data Errors ---

class DataError(TradingError):
    """Base error for data-related issues."""
    pass


class IngestError(DataError):
    """Error during data ingestion."""
    pass


class ValidationError(DataError):
    """Data validation failure."""
    pass


class StorageError(DataError):
    """Error reading/writing data storage."""
    pass


class DriftError(DataError):
    """Feature or data distribution drift detected."""
    pass


# --- Model Errors ---

class ModelError(TradingError):
    """Base error for model-related issues."""
    pass


class ModelNotFoundError(ModelError):
    """Requested model not found in registry."""
    pass


class ModelTrainingError(ModelError):
    """Error during model training."""
    pass


class ModelPredictionError(ModelError):
    """Error during model prediction."""
    pass


class ModelPromotionError(ModelError):
    """Error during champion/challenger promotion."""
    pass


# --- Execution Errors ---

class ExecutionError(TradingError):
    """Base error for execution-related issues."""
    pass


class OrderRejectedError(ExecutionError):
    """Order rejected by risk manager or broker."""

    def __init__(self, message: str, order_id: Optional[str] = None) -> None:
        self.order_id = order_id
        super().__init__(message, code="ORDER_REJECTED")


class InsufficientFundsError(ExecutionError):
    """Insufficient funds for order."""
    pass


class CircuitBreakerTrippedError(ExecutionError):
    """Circuit breaker is active — all trading halted."""
    pass


# --- Risk Errors ---

class RiskError(TradingError):
    """Base error for risk-related issues."""
    pass


class RiskLimitBreachedError(RiskError):
    """A risk limit has been breached."""

    def __init__(self, message: str, limit_name: str) -> None:
        self.limit_name = limit_name
        super().__init__(message, code="RISK_LIMIT_BREACHED")


# --- Configuration Errors ---

class ConfigError(TradingError):
    """Configuration-related error."""
    pass


# --- LLM Errors ---

class LLMError(TradingError):
    """Base error for LLM-related issues."""
    pass


class LLMParseError(LLMError):
    """LLM returned invalid/unparseable JSON."""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
    pass
