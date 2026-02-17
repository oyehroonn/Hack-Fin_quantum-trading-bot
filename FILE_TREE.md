# File Tree — Quantum Trading Bot v0.2

```
quantum-trading-bot/
├── config.yaml                    # System configuration (v0.2: models, regime, ensemble, LLM)
├── pyproject.toml                 # Poetry dependencies (v0.2: xgboost, lightgbm, joblib)
│
├── core/                          # Domain types, interfaces, errors
│   ├── __init__.py                # Exports all core types + interfaces + errors
│   ├── types.py                   # Immutable dataclasses (v0.2: Regime, ModelDecision, StrategyAllocation, etc.)
│   ├── interfaces.py              # ABCs (v0.2: ModelRegistry, Evaluator, Allocator, LLMClient, RegimeDetector)
│   └── errors.py                  # [NEW] Custom error hierarchy (TradingError → Data/Model/Execution/Risk/LLM)
│
├── models/                        # [NEW] Model Zoo
│   ├── __init__.py
│   ├── base.py                    # TradingModel base class + ModelStrategy adapter (Model→Backtest bridge)
│   ├── registry.py                # FileModelRegistry: save/load/promote, champion/challenger management
│   ├── baselines/                 # Rule-based models (no training needed)
│   │   ├── sma_crossover.py       # SMA crossover as TradingModel
│   │   ├── mean_reversion.py      # Bollinger/Z-score mean reversion
│   │   └── breakout.py            # Channel breakout with momentum
│   ├── ml/                        # Machine learning models
│   │   ├── xgb_classifier.py      # XGBoost direction classifier (with sklearn fallback)
│   │   ├── lgbm_regressor.py      # LightGBM return regressor (with sklearn fallback)
│   │   └── calibration.py         # Probability calibration (isotonic / Platt / binning)
│   └── ensemble/                  # Ensemble strategies
│       ├── blender.py             # Weighted blending (equal/performance/stacking)
│       └── meta_regime_selector.py # Regime-aware model selection with Hurst/variance-ratio
│
├── research/                      # Research & experimentation
│   ├── __init__.py
│   ├── dataset.py                 # DatasetBuilder (sliding windows, splits, normalisation)
│   ├── labeling.py                # [NEW] Tradable labels: direction, edge, triple-barrier, volatility
│   ├── evaluation.py              # [NEW] Cost-aware metrics, walk-forward eval, stability (R²)
│   ├── experiments.py             # [NEW] Experiment tracker (JSON-based, index, compare)
│   └── tuning.py                  # [NEW] Grid search + random search with tracking
│
├── features/                      # Feature engineering
│   ├── __init__.py
│   ├── technical.py               # SMA, EMA, RSI, MACD, Bollinger, ATR, VWAP
│   ├── statistical.py             # Returns, vol, Z-score, autocorrelation, skew, kurtosis
│   ├── regime.py                  # [NEW] StatisticalRegimeDetector (trend/vol/Hurst/chop)
│   ├── microstructure.py          # [NEW] Spread proxy, Kyle's lambda, Amihud, VPIN, volume imbalance
│   ├── online_store.py            # [NEW] Streaming feature store (rolling buffers, incremental EMA)
│   ├── feature_store.py           # Batch feature store (Parquet cache)
│   ├── simple_store.py            # In-memory feature store for execution
│   └── dummy_model.py             # Demo model
│
├── portfolio/                     # [NEW] Portfolio management
│   ├── __init__.py
│   ├── allocator.py               # EqualWeight, RiskParity, RegimeAware, Kelly allocators
│   └── rebalancer.py              # Target→Orders conversion with min trade thresholds
│
├── backtest/                      # Backtesting engine
│   ├── __init__.py
│   ├── engine.py                  # Event-driven backtest engine
│   ├── strategy.py                # Strategy ABCs (OrderBased, WeightBased)
│   ├── strategies/
│   │   └── sma_crossover.py       # Original SMA crossover strategy
│   ├── broker_sim.py              # Simulated broker (slippage, partial fills)
│   ├── cost_models.py             # Fixed, percent, spread, volume cost models
│   ├── accounting.py              # Position tracking, PnL, equity curve
│   ├── walk_forward.py            # Walk-forward analysis runner
│   └── report.py                  # Performance metrics (Sharpe, Sortino, DD, etc.)
│
├── execution/                     # Live/paper trading
│   ├── __init__.py
│   ├── orchestrator.py            # Execution orchestrator (bar→features→signal→order→fill)
│   ├── paper_broker.py            # Paper trading broker
│   ├── live_data.py               # Replay data source
│   └── risk_overlay.py            # [NEW] Pre-trade risk overlay + CircuitBreaker
│
├── risk/                          # Risk management
│   ├── __init__.py
│   ├── rules.py                   # AdvancedRiskManager (MaxPosition, MaxLeverage, MaxDD, Cooldown)
│   └── simple_risk.py             # SimpleRiskManager
│
├── data/                          # Data layer
│   ├── __init__.py
│   ├── dummy_source.py            # Synthetic data source
│   ├── ingest/
│   │   ├── base.py                # Ingestor ABC
│   │   ├── equities_yfinance.py   # Yahoo Finance ingestor
│   │   ├── binance_public.py      # Binance public API ingestor
│   │   ├── crypto_ccxt.py         # CCXT ingestor
│   │   └── forex_oanda.py         # OANDA ingestor
│   ├── storage/
│   │   ├── parquet_store.py       # Partitioned Parquet storage
│   │   └── manifests.py           # [NEW] Data manifest tracking
│   ├── quality/
│   │   ├── validators.py          # Data quality validators
│   │   └── drift.py               # [NEW] Distribution drift detection (PSI, KS, mean-shift)
│   └── query/
│       └── duckdb_client.py       # DuckDB SQL query client
│
├── llm/                           # [NEW] LLM orchestration
│   ├── __init__.py
│   ├── schemas.py                 # Strict JSON schemas for all LLM I/O
│   ├── client.py                  # OpenAI-compatible client + MockLLMClient + JSON validator
│   └── orchestrator.py            # TradingGovernor (strategy selection, anomaly triage, reports)
│
├── infra/                         # Infrastructure
│   ├── __init__.py
│   ├── config.py                  # Pydantic settings (YAML + env vars)
│   ├── logging.py                 # Loguru structured logging with correlation IDs
│   ├── journal.py                 # SQLite event journal
│   └── scheduler.py               # [NEW] Async task scheduler for periodic operations
│
├── api/                           # FastAPI backend
│   ├── __init__.py
│   ├── main.py                    # App entry point (v0.2: mounts model + research routers)
│   ├── schemas.py                 # [NEW] Pydantic request/response schemas
│   └── routers/                   # [NEW] Decomposed API routers
│       ├── __init__.py
│       ├── models.py              # /api/models/* (train, predict, list, regime, suggest)
│       └── research.py            # /api/research/* (walk-forward, tune, experiments)
│
├── frontend/                      # React UI
│   ├── src/App.jsx
│   └── ...
│
├── scripts/                       # CLI scripts
│   ├── ingest.py
│   ├── run_demo.py
│   ├── paper_trade_replay.py
│   └── run_walk_forward.py
│
├── tests/                         # Test suite
│   ├── test_types.py
│   ├── test_backtest.py
│   └── ...
│
├── SYSTEM_ARCHITECTURE.md         # Full architecture documentation
└── FILE_TREE.md                   # This file
```

## v0.2 New Files Summary

| Directory | New Files | Purpose |
|-----------|-----------|---------|
| `core/` | `errors.py` | Typed error hierarchy |
| `models/` | 10 files | Model zoo (baselines, ML, ensemble, registry) |
| `research/` | 4 files | Labeling, evaluation, experiments, tuning |
| `features/` | 3 files | Regime detection, microstructure, online store |
| `portfolio/` | 2 files | Allocator, rebalancer |
| `execution/` | 1 file | Risk overlay + circuit breaker |
| `llm/` | 4 files | LLM client, schemas, orchestrator, audit |
| `data/` | 2 files | Drift detection, data manifests |
| `infra/` | 1 file | Task scheduler |
| `api/` | 3 files | Schemas + model/research routers |
| **Total** | **30 new files** | |
