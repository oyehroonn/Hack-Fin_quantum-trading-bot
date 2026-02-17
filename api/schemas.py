"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Any, Optional


# ── Backtest Schemas ──

class BacktestConfig(BaseModel):
    initial_cash: float = 100000.0
    fast_period: int = 10
    slow_period: int = 30
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class BacktestResult(BaseModel):
    success: bool = True
    metrics: dict[str, Any] = {}
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []


# ── Model Schemas ──

class ModelTrainRequest(BaseModel):
    model_type: str = Field(..., description="Model type: xgb_direction, lgbm_returns, sma_crossover, mean_reversion, breakout")
    model_id: str = Field(..., description="Unique model identifier")
    symbol: str = Field(..., description="Symbol to train on")
    asset_class: str = "equities"
    timeframe: str = "1d"
    label_type: str = Field("edge", description="Label type: direction, edge, triple_barrier, volatility")
    horizon: int = Field(1, description="Forward prediction horizon")
    train_days: int = Field(500, description="Days of data for training")
    val_pct: float = Field(0.2, description="Validation set percentage")
    fee_bps: float = Field(10.0, description="Fee in basis points")
    params: dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")


class ModelTrainResponse(BaseModel):
    success: bool
    model_id: str
    version: str
    train_metrics: dict[str, Any] = {}
    val_metrics: dict[str, Any] = {}
    feature_importance: dict[str, float] = {}
    message: str = ""


class ModelPredictRequest(BaseModel):
    model_id: str
    symbol: str
    asset_class: str = "equities"
    timeframe: str = "1d"
    lookback_days: int = 60


class ModelPredictResponse(BaseModel):
    success: bool
    symbol: str
    signal: str  # BUY, SELL, HOLD
    probability: float = 0.5
    confidence: float = 0.0
    regime: Optional[str] = None
    model_id: str = ""
    message: str = ""


class ModelListResponse(BaseModel):
    models: list[dict[str, Any]] = []


# ── Research Schemas ──

class WalkForwardRequest(BaseModel):
    model_type: str
    model_id: str
    symbol: str
    asset_class: str = "equities"
    timeframe: str = "1d"
    label_type: str = "edge"
    train_days: int = 252
    test_days: int = 63
    step_days: int = 21
    total_days: int = 1000
    params: dict[str, Any] = Field(default_factory=dict)


class WalkForwardResponse(BaseModel):
    success: bool
    model_id: str
    num_folds: int = 0
    summary: dict[str, float] = {}
    fold_results: list[dict[str, Any]] = []
    message: str = ""


class TuneRequest(BaseModel):
    model_type: str
    symbol: str
    asset_class: str = "equities"
    timeframe: str = "1d"
    method: str = "random"  # grid or random
    n_iter: int = 20
    param_grid: dict[str, Any] = Field(default_factory=dict)
    metric: str = "sharpe"


class TuneResponse(BaseModel):
    success: bool
    best_params: dict[str, Any] = {}
    best_metric: float = 0.0
    all_results: list[dict[str, Any]] = []
    message: str = ""


# ── Regime Schemas ──

class RegimeDetectRequest(BaseModel):
    symbol: str
    asset_class: str = "equities"
    timeframe: str = "1d"
    lookback_days: int = 100


class RegimeDetectResponse(BaseModel):
    success: bool
    symbol: str
    regime: str
    confidence: float
    indicators: dict[str, float] = {}


# ── Execution Schemas ──

class PaperTradeRequest(BaseModel):
    symbol: str
    model_id: str
    initial_cash: float = 100000.0
    asset_class: str = "equities"
    timeframe: str = "1d"
    duration_days: int = 30


class PaperTradeResponse(BaseModel):
    success: bool
    final_equity: float = 0.0
    total_return: float = 0.0
    num_trades: int = 0
    trades: list[dict[str, Any]] = []
    message: str = ""


# ── Terminal Schemas ──

class TerminalTradeRequest(BaseModel):
    session_id: str = Field(..., description="Trading session ID")
    symbol: str = Field(..., description="Symbol to trade")
    side: str = Field(..., description="BUY or SELL")
    qty: float = Field(..., gt=0, description="Quantity")
    price: float = Field(..., gt=0, description="Price per unit")
    asset_class: str = "equities"


# ── Suggest Schemas ──

class SuggestRequest(BaseModel):
    symbol: str
    asset_class: str = "equities"
    models: list[str] = Field(default_factory=lambda: ["sma_crossover"])
    use_ensemble: bool = False
    use_regime: bool = True
    lookback_days: int = 60


class SuggestResponse(BaseModel):
    success: bool
    symbol: str
    signal: str
    confidence: float = 0.0
    probability: float = 0.5
    regime: Optional[str] = None
    model_signals: dict[str, dict[str, Any]] = {}
    risk_level: str = "MEDIUM"
    recommendation: str = ""
