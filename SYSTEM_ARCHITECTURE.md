# System Architecture — Quantum Trading Bot

> **Generated**: February 2026  
> **Codebase**: `quantum-trading-bot v0.2.0`  
> **Stack**: Python 3.11+ (FastAPI, Pandas, NumPy, SciPy, XGBoost, LightGBM, sklearn) + React 18 (Vite, Recharts)
> **What's new in v0.2**: Model Zoo (baselines + ML + ensemble), Champion/Challenger registry, Research platform (labeling, evaluation, tuning), Regime detection, Portfolio allocation, LLM orchestration, Drift detection, API router refactor

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Directory Structure](#2-directory-structure)
3. [Layer-by-Layer Breakdown](#3-layer-by-layer-breakdown)
4. [Core Domain Types & Interfaces](#4-core-domain-types--interfaces)
5. [Data Layer](#5-data-layer)
6. [Feature Engineering Layer](#6-feature-engineering-layer)
7. [Backtest Engine](#7-backtest-engine)
8. [Execution Layer (Live / Paper)](#8-execution-layer-live--paper)
9. [Infrastructure Layer](#9-infrastructure-layer)
10. [Research Layer](#10-research-layer)
11. [API Layer (FastAPI)](#11-api-layer-fastapi)
12. [Frontend (React)](#12-frontend-react)
13. [Complete Request Flows](#13-complete-request-flows)
14. [API Endpoint Reference](#14-api-endpoint-reference)
15. [Data Flow Diagrams](#15-data-flow-diagrams)
16. [Configuration & Deployment](#16-configuration--deployment)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                      │
│   Asset class picker · Symbol search · Strategy config · Charts     │
│                     Equity curve · Trades table                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  HTTP (axios)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI + Uvicorn)                   │
│  /api/backtest  /api/backtest/real  /api/backtest/crypto             │
│  /api/backtest/synthetic  /api/backtest/suggest  /api/symbols/search │
└──────┬──────────────┬──────────────┬───────────────┬────────────────┘
       │              │              │               │
       ▼              ▼              ▼               ▼
┌──────────┐  ┌──────────────┐  ┌────────┐  ┌──────────────────┐
│ Backtest │  │ Data Ingest  │  │Features│  │   Execution      │
│  Engine  │  │  (yfinance,  │  │  Store  │  │  (Orchestrator,  │
│          │  │  Binance API)│  │         │  │   Paper Broker)  │
└────┬─────┘  └──────┬───────┘  └────┬───┘  └────────┬─────────┘
     │               │               │               │
     ▼               ▼               ▼               ▼
┌──────────┐  ┌──────────────┐  ┌────────┐  ┌──────────────────┐
│Accounting│  │   Storage    │  │Technical│  │   Risk Manager   │
│Portfolio │  │(Parquet+Duck)│  │  Stats  │  │   (Interfaces)   │
│  Report  │  │  Validators  │  │Indicators│  │                  │
└──────────┘  └──────────────┘  └────────┘  └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │   Infrastructure  │
                            │ Config · Logging  │
                            │ Journal (SQLite)  │
                            └──────────────────┘
```

---

## 2. Directory Structure

```
quantum-trading-bot/
├── api/                          # FastAPI backend
│   ├── main.py                   # All API endpoints (662 lines)
│   ├── requirements.txt          # Python deps for API
│   └── venv/                     # Virtual environment
│
├── core/                         # Domain primitives (types + interfaces)
│   ├── types.py                  # Bar, Tick, Order, Fill, Position, Signal, etc.
│   └── interfaces.py             # ABCs: DataSource, FeatureStore, Model, Strategy, Broker, RiskManager
│
├── data/                         # Data ingestion, storage, quality, querying
│   ├── ingest/
│   │   ├── base.py               # Ingestor ABC
│   │   ├── equities_yfinance.py  # Yahoo Finance ingestor (equities)
│   │   ├── binance_public.py     # Binance public API ingestor (crypto)
│   │   ├── crypto_ccxt.py        # CCXT ingestor (crypto, multi-exchange)
│   │   └── forex_oanda.py        # OANDA ingestor (forex)
│   ├── storage/
│   │   └── parquet_store.py      # Parquet read/write with year/month partitioning
│   ├── quality/
│   │   └── validators.py         # Monotonic time, duplicates, missing, outlier checks
│   └── query/
│       └── duckdb_client.py      # DuckDB analytical queries over parquet
│
├── features/                     # Feature engineering
│   ├── technical.py              # SMA, EMA, RSI, MACD, Bollinger, ATR, VWAP
│   ├── statistical.py            # Returns, log-returns, rolling vol, z-score, autocorr, skew, kurtosis
│   ├── feature_store.py          # Orchestrates feature computation + parquet caching
│   ├── simple_store.py           # Lightweight in-memory feature store (core.interfaces impl)
│   └── dummy_model.py            # Random-signal model for testing (core.interfaces impl)
│
├── backtest/                     # Backtesting engine
│   ├── engine.py                 # Event-driven backtest loop
│   ├── strategy.py               # Strategy ABC (weight-based + order-based)
│   ├── strategies/
│   │   └── sma_crossover.py      # SMA crossover with volatility scaling
│   ├── broker_sim.py             # Simulated broker (market/limit fills, partial fills, fees)
│   ├── accounting.py             # Portfolio, Position, Trade, equity curve
│   ├── cost_models.py            # FixedFee, PercentFee, SpreadSlippage, VolumeSlippage, Composite
│   ├── report.py                 # Metrics (Sharpe, Sortino, Calmar, drawdown, win rate) + pyfolio
│   └── walk_forward.py           # Walk-forward analysis runner
│
├── execution/                    # Live/paper trading
│   ├── orchestrator.py           # Live execution loop with safety rails
│   ├── paper_broker.py           # Paper trading broker (core.interfaces.Broker impl)
│   └── live_data.py              # Replay historical data as live stream
│
├── research/                     # ML dataset building
│   └── dataset.py                # Sliding windows, time-series splits, normalization
│
├── infra/                        # Cross-cutting infrastructure
│   ├── config.py                 # YAML + env var config (Pydantic Settings)
│   ├── logging.py                # Loguru structured logging with correlation IDs
│   └── journal.py                # SQLite event journal (signals, orders, fills, risk)
│
├── scripts/                      # CLI entry points
│   ├── ingest.py                 # Data ingestion CLI
│   ├── run_backtest.py           # Backtest runner CLI
│   ├── run_walk_forward.py       # Walk-forward analysis CLI
│   ├── run_demo.py               # Demo runner
│   └── paper_trade_replay.py     # Paper trade replay script
│
├── frontend/                     # React UI
│   ├── src/
│   │   ├── App.jsx               # Main application component (508 lines)
│   │   ├── App.css               # Styles
│   │   ├── main.jsx              # React entry point
│   │   └── index.css             # Global styles
│   ├── package.json              # React 18, axios, recharts
│   └── vite.config.js            # Vite dev server config
│
├── tests/                        # Test suite
│   ├── test_broker.py
│   ├── test_risk.py
│   ├── test_storage.py
│   ├── test_validators.py
│   ├── test_backtest.py
│   ├── test_dataset.py
│   ├── test_config.py
│   ├── test_features_statistical.py
│   ├── test_features_technical.py
│   ├── test_types.py
│   ├── test_feature_store.py
│   └── test_duckdb.py
│
├── config.yaml                   # System configuration
├── pyproject.toml                # Poetry project definition
├── Makefile                      # Build/run commands
├── start_backend.sh              # Backend startup script
├── start_ui.sh                   # Frontend startup script
└── README.md
```

---

## 3. Layer-by-Layer Breakdown

### Dependency Direction (Clean Architecture)

```
Frontend  →  API  →  Backtest/Execution  →  Core (types, interfaces)
                          ↓                       ↑
                     Data Layer ──────────────────┘
                     Features  ──────────────────┘
                     Infra     ──────────────────┘
```

All business logic depends inward on `core/`. The `core/` package has **zero** external dependencies — it defines only immutable dataclasses and abstract interfaces.

---

## 4. Core Domain Types & Interfaces

### `core/types.py` — Immutable Value Objects

| Type             | Fields                                                                 | Validation Rules                                      |
|------------------|------------------------------------------------------------------------|-------------------------------------------------------|
| `Bar`            | symbol, timestamp, open, high, low, close, volume (all `Decimal`)      | high >= low, open/close within [low, high], vol >= 0  |
| `Tick`           | symbol, timestamp, price, size, bid/ask (optional)                     | —                                                     |
| `Order`          | order_id, symbol, side, quantity, order_type, limit/stop prices        | qty > 0, limit price required for LIMIT orders, etc.  |
| `Fill`           | fill_id, order_id, symbol, side, quantity, price, fee, timestamp       | qty > 0, price > 0, fee >= 0                          |
| `Position`       | symbol, quantity, avg_price, unrealized_pnl, realized_pnl              | Properties: market_value, is_long, is_short, is_flat  |
| `PortfolioState` | timestamp, cash, positions, total_value, unrealized/realized PnL       | cash >= 0                                             |
| `Signal`         | symbol, timestamp, side, strength (-1..1), confidence (0..1), targets  | strength in [-1,1], confidence in [0,1]               |

### Enums

| Enum          | Values                                                     |
|---------------|-----------------------------------------------------------|
| `OrderSide`   | BUY, SELL                                                 |
| `OrderType`   | MARKET, LIMIT, STOP, STOP_LIMIT                           |
| `OrderStatus` | PENDING, SUBMITTED, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED |

### `core/interfaces.py` — Abstract Base Classes

```
DataSource       ──▶  get_bars(symbol) → AsyncIterator[Bar]
                      get_ticks(symbol) → AsyncIterator[Tick]

FeatureStore     ──▶  compute_features(symbol, bars) → dict[str, float]
                      get_latest_features(symbol) → dict[str, float]

Model            ──▶  predict(features, symbol) → Signal
                      train(features, labels) → None

Strategy         ──▶  generate_signals(portfolio_state, features) → list[Signal]

Broker           ──▶  submit_order(order) → str
                      cancel_order(order_id) → bool
                      get_order_status(order_id) → Order
                      get_fills(order_id) → list[Fill]
                      get_positions() → dict[str, Position]
                      get_portfolio_state() → PortfolioState

RiskManager      ──▶  validate_order(order, portfolio) → (bool, reason)
                      check_limits(portfolio) → (bool, reason)
```

---

## 5. Data Layer

### 5.1 Ingestion Pipeline

```
External APIs                    Ingestor (ABC)                  Standardized DataFrame
┌──────────────┐                ┌──────────────┐                ┌────────────────────┐
│ Yahoo Finance │──▶ EquitiesYFinanceIngestor ──▶  fetch_ohlcv()  │ timestamp (UTC)    │
│ Binance API   │──▶ BinancePublicIngestor   ──▶  → DataFrame    │ open, high, low    │
│ CCXT (multi)  │──▶ CryptoCCXTIngestor      ──▶  standardize()  │ close, volume      │
│ OANDA         │──▶ ForexOANDAIngestor      ──▶               │ (float64 types)    │
└──────────────┘                └──────────────┘                └────────────────────┘
```

**Base `Ingestor` Interface** (`data/ingest/base.py`):
- `async fetch_ohlcv(symbol, timeframe, start, end, limit) → pd.DataFrame`
- `standardize_columns(df)` — renames aliases (time→timestamp, o→open, etc.), forces UTC, coerces numerics

**`EquitiesYFinanceIngestor`** (`data/ingest/equities_yfinance.py`):
- Uses `yfinance.Ticker.history()` with retry logic (period-based fallback → start/end fallback)
- Maps timeframes: `1d`, `1h`, `5m`, etc.
- Runs blocking yfinance call in `asyncio.run_in_executor()` so it doesn't block the FastAPI event loop

**`BinancePublicIngestor`** (`data/ingest/binance_public.py`):
- Calls `GET https://api.binance.com/api/v3/klines` — no API key required
- Converts Binance millisecond timestamps to UTC datetime
- Max 1000 bars per request

### 5.2 Storage (`data/storage/parquet_store.py`)

```
data/parquet/
└── {asset_class}/
    └── {symbol}/
        └── {timeframe}/
            └── {year}/
                └── {month:02d}/
                    └── data.parquet     ← Snappy-compressed, dictionary-encoded
```

- **Write**: Groups by year/month → merges with existing partition (deduplicates on timestamp)
- **Read**: Determines which partitions to load based on date range → concatenates → filters → deduplicates
- All timestamps stored in UTC

### 5.3 Quality Validation (`data/quality/validators.py`)

`DataValidator.validate(df)` runs these checks and returns a `ValidationReport`:

| Check               | Logic                                              | Mode        |
|---------------------|----------------------------------------------------|-------------|
| `monotonic_time`    | Timestamps must be strictly increasing              | Error       |
| `no_duplicates`     | No duplicate timestamps                             | Error       |
| `missing_ratio`     | Each column < 5% missing (configurable)             | Error/Warn  |
| `outliers_zscore`   | Z-score outlier detection (threshold=3.0)           | Warn only   |

### 5.4 Analytical Queries (`data/query/duckdb_client.py`)

- Creates DuckDB views over the partitioned parquet files
- Supports multi-symbol queries with UNION ALL
- Used for research/analysis workflows

---

## 6. Feature Engineering Layer

### 6.1 Technical Indicators (`features/technical.py`)

| Function           | Parameters                        | Output               |
|--------------------|-----------------------------------|----------------------|
| `sma(series, w)`   | window                            | Simple Moving Avg    |
| `ema(series, w)`   | window or alpha                   | Exponential MA       |
| `rsi(series, w)`   | window (default 14)               | RSI 0-100            |
| `macd(series)`     | fast=12, slow=26, signal=9        | macd, signal, hist   |
| `bollinger_bands`  | window=20, num_std=2.0            | upper, middle, lower, bandwidth |
| `atr(h, l, c, w)`  | window=14                         | Average True Range   |
| `vwap(h, l, c, v)` | window (None=cumulative)          | Vol-Weighted Avg Price |

### 6.2 Statistical Features (`features/statistical.py`)

| Function           | Parameters                        | Output               |
|--------------------|-----------------------------------|----------------------|
| `returns`          | periods                           | Simple returns       |
| `log_returns`      | periods                           | Log returns          |
| `rolling_vol`      | window, annualize, periods/year   | Rolling volatility   |
| `zscore`           | window                            | Rolling Z-score      |
| `autocorr`         | lag, window                       | Autocorrelation      |
| `rolling_corr`     | series1, series2, window          | Rolling correlation  |
| `rolling_skew`     | window                            | Rolling skewness     |
| `rolling_kurtosis` | window                            | Rolling kurtosis     |

### 6.3 Feature Store (`features/feature_store.py`)

```python
FeatureStore.compute_features(
    df_bars,                    # OHLCV DataFrame
    feature_config={            # Declarative configuration
        "technical": {
            "sma": [10, 20, 50],
            "rsi": [14],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"window": 20, "num_std": 2.0},
        },
        "statistical": {
            "returns": [1, 5],
            "rolling_vol": [{"window": 20, "annualize": True}],
            "zscore": [20],
        },
    },
    symbol="AAPL",
    timeframe="1d",
    use_cache=True,             # Parquet-based caching
)
```

**Caching**: MD5 hash of config → `data/features/{symbol}_{timeframe}_{hash}.parquet`

### 6.4 Simple Feature Store (`features/simple_store.py`)

Implements `core.interfaces.FeatureStore` for the live execution loop:
- In-memory deque of recent bars per symbol
- Computes: returns, SMA, price, volume
- Lightweight — designed for real-time use

### 6.5 Dummy Model (`features/dummy_model.py`)

Implements `core.interfaces.Model`:
- Uses price-vs-SMA ratio + random component to generate signals
- Reproducible via seeded RNG
- Used for paper trading demos/testing

---

## 7. Backtest Engine

### 7.1 Architecture

```
BacktestEngine
    ├── Strategy (ABC)               ← Generates signals per bar
    │   ├── WeightBasedStrategy      ← Returns {symbol: target_weight}
    │   └── OrderBasedStrategy       ← Returns [Order, ...]
    │
    ├── SimBroker                    ← Simulates order execution
    │   └── CostModel (composable)   ← Fees + slippage
    │
    ├── Portfolio (Accounting)       ← Tracks positions, cash, PnL, equity curve
    │
    └── PerformanceReport            ← Computes metrics, saves artifacts
```

### 7.2 Event Loop (`backtest/engine.py`)

```
for each timestamp in data:
    1. Extract bars for this timestamp (per symbol)
    2. Update portfolio prices
    3. Call strategy.on_bar(timestamp, bars, portfolio)
       → Returns either weights dict OR order list
    4. If weights → convert to orders via _process_weights()
       If orders → submit via _process_orders()
    5. broker.process_bar(timestamp, bar_dict)
       → Returns list[Fill]
    6. Apply each fill to portfolio
    7. Record equity curve point

strategy.on_finish()
```

### 7.3 Strategy: SMA Crossover (`backtest/strategies/sma_crossover.py`)

```
Inherits: WeightBasedStrategy → Strategy (ABC)

Logic:
  1. Maintain per-symbol price history
  2. Compute fast SMA and slow SMA
  3. If fast > slow → long signal (+1.0)
     If fast < slow → short signal (-1.0)
  4. Volatility scaling: weight *= min(vol_target / realized_vol, 2.0)
  5. Return {symbol: Decimal(weight)}
```

**Parameters**: fast_period (10), slow_period (30), vol_period (20), vol_target (0.15), max_weight (1.0)

### 7.4 Simulated Broker (`backtest/broker_sim.py`)

**Order Types Supported**: MARKET, LIMIT

**Market Order Fill Logic**:
1. Fill at bar open price
2. Apply spread slippage (if available)
3. Random partial fill (prob=10%, ratio=50%)
4. Calculate cost via composite cost model

**Limit Order Fill Logic**:
1. BUY: fills if bar_low <= limit_price (at min of limit_price, bar_open)
2. SELL: fills if bar_high >= limit_price (at max of limit_price, bar_open)
3. Same partial fill and cost logic

### 7.5 Cost Models (`backtest/cost_models.py`)

```
CompositeCostModel (default)
    ├── PercentFee(0.1%)          ← notional × 0.001
    └── SpreadSlippage(5 bps)     ← notional × 0.0005

Other available models:
    ├── FixedFee($1.00)           ← flat per-trade
    └── VolumeSlippage            ← market impact based on order_size / bar_volume
```

### 7.6 Portfolio Accounting (`backtest/accounting.py`)

**Position tracking**:
- Adding to position → weighted average price
- Reducing position → realizes PnL on closed quantity
- Reversing position → realizes PnL, new avg_price = fill_price
- Zero quantity → position removed

**Equity curve**: `list[tuple[datetime, Decimal]]` — one point per bar

### 7.7 Performance Metrics (`backtest/report.py`)

| Metric         | Formula/Method                                          |
|----------------|---------------------------------------------------------|
| Total Return   | (final_equity / initial_equity) - 1                     |
| CAGR           | (final/initial)^(1/years) - 1                           |
| Volatility     | std(daily_returns) × √252                               |
| Sharpe         | CAGR / Volatility                                       |
| Sortino        | CAGR / Downside_Std                                     |
| Max Drawdown   | max peak-to-trough decline                              |
| Calmar         | CAGR / Max_Drawdown                                     |
| Win Rate       | winning_trades / total_trades                            |
| Avg Win/Loss   | mean(winning_pnl) / mean(losing_pnl)                    |
| Turnover       | sum(abs(trade_qty)) / avg_equity                         |

**Output**: Saves `metrics.json`, `equity_curve.csv`, `trades.csv` to `runs/{run_id}/`

### 7.8 Walk-Forward Analysis (`backtest/walk_forward.py`)

```
┌──────────────────────────────────────────────────────────────────┐
│ Walk 1:  [═══ Train (252d) ═══][── Test (63d) ──]               │
│ Walk 2:       [═══ Train ═════][── Test ──]                     │
│ Walk 3:            [═══ Train ═════][── Test ──]                │
│ ...                                                              │
│ Step: 21 days (configurable)                                     │
└──────────────────────────────────────────────────────────────────┘
```

Each walk: creates fresh strategy → runs backtest on test period → collects metrics → returns DataFrame of all walks.

---

## 8. Execution Layer (Live / Paper)

### 8.1 Execution Orchestrator (`execution/orchestrator.py`)

```
ExecutionOrchestrator
    ├── DataSource      ← Streams bars (live or replay)
    ├── FeatureStore    ← Computes features per bar
    ├── Model           ← Generates signals from features
    ├── Broker          ← Submits orders (paper or live)
    ├── RiskManager     ← Validates orders + checks portfolio limits
    └── Journal         ← Logs all events to SQLite
```

**Main Loop**:
```
async for bar in data_source.get_bars():
    1. Buffer latest 100 bars per symbol
    2. Compute features via FeatureStore
    3. Get signal from Model.predict()
    4. Log signal to Journal
    5. Convert signal to Order (strength > 0.5 → BUY, < -0.5 → SELL)
    6. Validate with RiskManager
    7. Submit to Broker
    8. Log order to Journal
    9. Async check for fills → log to Journal
    10. Periodic portfolio state logging (every 10 bars)
    11. Check risk limits → log violations
```

### 8.2 Paper Broker (`execution/paper_broker.py`)

Implements `core.interfaces.Broker`:
- Market orders fill immediately at current price + slippage
- Manages cash, positions, PnL
- Rejects orders with insufficient funds
- Full order lifecycle: PENDING → FILLED / REJECTED / CANCELLED

### 8.3 Replay Data Source (`execution/live_data.py`)

Implements `core.interfaces.DataSource`:
- Loads historical data from ParquetStore
- Replays bars with configurable speedup factor
- `async get_bars()` yields bars with appropriate time delays

---

## 9. Infrastructure Layer

### 9.1 Configuration (`infra/config.py`)

```yaml
# config.yaml (source of truth)
data_dir: "data/"
symbols: ["AAPL", "MSFT", "GOOGL"]
initial_cash: 100000.0
max_position_size: 10000.0
max_leverage: 2.0
max_drawdown: 0.2               # 20%
stop_loss_pct: 0.02             # 2%
take_profit_pct: 0.05           # 5%
slippage_bps: 5.0
commission_bps: 1.0
log_level: "INFO"
```

**Override chain**: `config.yaml` → `TRADING_*` environment variables → `.env` file

### 9.2 Logging (`infra/logging.py`)

- **Engine**: Loguru
- **Format**: `{timestamp} | {level} | {module}:{function}:{line} | correlation_id={id} | {message}`
- **Correlation IDs**: Context-variable based, thread-safe, propagated across async calls
- **Rotation**: 10 MB per file, 7-day retention
- **Output**: stderr (colorized) + optional file handler

### 9.3 Event Journal (`infra/journal.py`)

**Storage**: SQLite (`data/journal.db`)

**Event Types**:
| Event              | When Logged                         |
|--------------------|-------------------------------------|
| `SIGNAL`           | Model generates a trading signal    |
| `ORDER`            | Order submitted to broker           |
| `FILL`             | Order filled (partial or full)      |
| `PORTFOLIO_STATE`  | Periodic portfolio snapshots        |
| `RISK_VIOLATION`   | Risk limit breached                 |
| `CIRCUIT_BREAKER`  | Emergency stop triggered            |

**Schema**:
```sql
events (
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    data         TEXT NOT NULL,    -- JSON blob
    created_at   TEXT NOT NULL
)
```

Supports schema versioning and migrations.

---

## 10. Research Layer

### `research/dataset.py` — DatasetBuilder

Builds ML-ready datasets from feature DataFrames:

```
DatasetBuilder(lookback=60, horizon=1, label_type="returns")
    │
    ├── build_windows(df_features)
    │   → X: (n_samples, lookback, n_features)  — sliding windows
    │   → y: (n_samples,)                       — forward labels
    │   → timestamps: (n_samples,)
    │
    ├── time_series_split(timestamps, train=0.7, val=0.15)
    │   → TimeSeriesSplit(train_start/end, val_start/end, test_start/end, purge_gap)
    │
    ├── fit_normalizer(X_train)  — StandardScaler on train only
    │
    └── build_dataset(df_features)
        → {X_train, y_train, X_val, y_val, X_test, y_test, split}
```

**Label Types**: `returns` (forward return), `direction` (up=1/down=0), `volatility` (forward vol)

**Purge Gap**: Configurable gap between train/val/test to prevent data leakage.

---

## 11. API Layer (FastAPI)

### `api/main.py`

**Server**: FastAPI + Uvicorn  
**CORS**: Allows `localhost:5173` (Vite) and `localhost:3000`  
**Base URL**: `http://localhost:8000`

---

## 12. Frontend (React)

### Technology

| Library    | Purpose                     |
|------------|-----------------------------|
| React 18   | UI framework                |
| Vite 5     | Dev server + bundler        |
| axios      | HTTP client                 |
| recharts   | Equity curve charting       |

### Component: `App.jsx` (single-page)

**State Management**: `useState` hooks for all state (no Redux/Context needed)

**UI Sections**:
1. **Header**: Title + subtitle
2. **Configuration Form**:
   - Asset class selector (equities / crypto)
   - Synthetic data toggle
   - Symbol input with live search (datalist auto-complete)
   - Period selector (7d → 2y)
   - "Get Trade Suggestion" button
   - Suggestion card (signal, confidence, risk level, recommendation)
   - Initial cash, fast/slow SMA period inputs
   - Start/end date pickers
   - File upload (CSV/Parquet) for custom data
3. **Results Display**:
   - 8-card metrics grid (Total Return, CAGR, Sharpe, Sortino, Max DD, Win Rate, Trades, Turnover)
   - Interactive equity curve chart (Recharts LineChart)
   - Trades table (first 50, with BUY/SELL coloring and PnL)

**API Proxy**: Vite proxies `/api/*` to `http://localhost:8000`

---

## 13. Complete Request Flows

### Flow 1: Real Stock Backtest (e.g., AAPL, 1 year)

```
User clicks "Run Backtest" with symbol=AAPL, useRealData=true
    │
    ▼
Frontend: POST /api/backtest/real?symbol=AAPL&initial_cash=100000&...
    │
    ▼
API (run_real_backtest):
    1. Parse dates → ensure UTC
    2. EquitiesYFinanceIngestor.fetch_ohlcv("AAPL", "1d", start, end)
       └── yfinance.Ticker("AAPL").history(period="1y") [in executor]
       └── standardize_columns() → UTC timestamps, lowercase cols
    3. Validate: required columns present, data not empty
    4. Add "symbol" column, normalize timestamps to UTC
    5. Set multi-index: (timestamp, symbol)
    6. Create SMACrossoverStrategy(fast=10, slow=30)
    7. Create BacktestEngine(initial_cash=100000, strategy, symbols=["AAPL"])
       └── Internally creates SimBroker + Portfolio
    8. engine.run(df, start, end)
       └── For each bar:
           a. strategy.on_bar() → compute SMAs → return weight
           b. Engine converts weight to order → submits to SimBroker
           c. SimBroker fills at next bar open + slippage + fees
           d. Portfolio records fill, updates positions, records equity
    9. PerformanceReport(portfolio).calculate_metrics()
    10. Return JSON: {success, metrics, equity_curve, trades}
    │
    ▼
Frontend:
    - Displays 8 metric cards
    - Renders equity curve chart
    - Renders trades table
```

### Flow 2: Crypto Backtest (e.g., BTCUSDT)

```
Same as Flow 1, except:
    Step 2: BinancePublicIngestor.fetch_ohlcv("BTCUSDT", "1d", start, end)
            └── GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d
```

### Flow 3: Trade Suggestion

```
User clicks "Get Trade Suggestion" for AAPL
    │
    ▼
Frontend: POST /api/backtest/suggest?symbol=AAPL&period_days=30&...
    │
    ▼
API (get_trade_suggestions):
    1. Fetch recent data (30 days)
    2. Run quick backtest
    3. Analyze latest trade or SMA crossover state:
       - fast_sma > slow_sma AND price > fast_sma → BUY
       - fast_sma < slow_sma AND price < fast_sma → SELL
       - Otherwise → HOLD
    4. Compute confidence = f(total_return, win_rate, sharpe)
    5. Assess risk: max_dd > 20% → HIGH, > 10% → MEDIUM, else LOW
    6. Return: {signal, confidence, current_price, risk_level, recommendation, ...}
    │
    ▼
Frontend: Displays suggestion card with color-coded signal
```

### Flow 4: Symbol Search

```
User types "A" in symbol input
    │
    ▼
Frontend: GET /api/symbols/search?query=A&asset_class=equities
    │
    ▼
API (search_symbols):
    - Filters hardcoded popular_stocks list by query
    - For each match: yf.Ticker(sym).info → longName
    - Returns: {symbols: [{symbol, display, name}, ...]}
    │
    ▼
Frontend: Populates <datalist> with options
```

### Flow 5: File Upload Backtest

```
User uploads CSV file → POST /api/backtest (multipart/form-data)
    │
    ▼
API (run_backtest):
    1. Read file bytes
    2. Parse CSV or Parquet
    3. Normalize timestamps to UTC
    4. Set multi-index (timestamp, symbol)
    5. Run backtest engine (same as Flow 1 steps 6-10)
    6. Return results
```

### Flow 6: Synthetic Backtest

```
POST /api/backtest/synthetic with JSON config
    │
    ▼
API (run_synthetic_backtest):
    1. Generate 100 days of linearly increasing AAPL data
    2. Run backtest engine
    3. Return results
```

---

## 14. API Endpoint Reference

### `GET /`
**Description**: Health check  
**Response**: `{"message": "Trading Bot API"}`

---

### `POST /api/backtest`
**Description**: Run backtest with uploaded CSV/Parquet file  
**Content-Type**: `multipart/form-data`

| Parameter     | Type   | Default  | Description           |
|---------------|--------|----------|-----------------------|
| file          | File   | required | CSV or Parquet file   |
| initial_cash  | float  | 100000   | Starting capital      |
| fast_period   | int    | 10       | Fast SMA window       |
| slow_period   | int    | 30       | Slow SMA window       |
| start_date    | string | null     | Filter start (ISO)    |
| end_date      | string | null     | Filter end (ISO)      |

**Response** (200):
```json
{
  "success": true,
  "metrics": {
    "total_return": 0.15,
    "cagr": 0.12,
    "volatility": 0.18,
    "sharpe": 0.67,
    "sortino": 0.85,
    "max_drawdown": 0.08,
    "calmar": 1.5,
    "win_rate": 0.55,
    "avg_win": 150.5,
    "avg_loss": -120.3,
    "avg_win_loss": 1.25,
    "turnover": 2.3,
    "num_trades": 24,
    "start_date": "2024-01-01T00:00:00+00:00",
    "end_date": "2024-12-31T00:00:00+00:00",
    "years": 1.0
  },
  "equity_curve": [
    {"timestamp": "2024-01-01T00:00:00+00:00", "equity": 100000},
    ...
  ],
  "trades": [
    {"timestamp": "...", "symbol": "AAPL", "side": "BUY", "quantity": 10.5, "price": 150.0, "cost": 0.15, "pnl": 0.0},
    ...
  ]
}
```

---

### `POST /api/backtest/synthetic`
**Description**: Run backtest with generated synthetic data  
**Content-Type**: `application/json`

| Parameter     | Type   | Default  | Description       |
|---------------|--------|----------|-------------------|
| initial_cash  | float  | 100000   | Starting capital  |
| fast_period   | int    | 10       | Fast SMA window   |
| slow_period   | int    | 30       | Slow SMA window   |
| start_date    | string | null     | Not used          |
| end_date      | string | null     | Not used          |

**Response**: Same as `/api/backtest`

---

### `POST /api/backtest/real`
**Description**: Run backtest with real stock data from Yahoo Finance  
**Content-Type**: Query parameters

| Parameter     | Type   | Default  | Description              |
|---------------|--------|----------|--------------------------|
| symbol        | string | required | Stock symbol (e.g. AAPL) |
| initial_cash  | float  | 100000   | Starting capital         |
| fast_period   | int    | 10       | Fast SMA window          |
| slow_period   | int    | 30       | Slow SMA window          |
| start_date    | string | null     | ISO date string          |
| end_date      | string | null     | ISO date string          |
| timeframe     | string | "1d"     | Bar timeframe            |

**Response**: Same as `/api/backtest`

---

### `POST /api/backtest/crypto`
**Description**: Run backtest with crypto data from Binance  
**Parameters**: Same as `/api/backtest/real` (symbol = Binance pair like BTCUSDT)  
**Response**: Same as `/api/backtest`

---

### `GET /api/symbols/search`
**Description**: Search for symbols  

| Parameter    | Type   | Default    | Description               |
|------------- |--------|------------|---------------------------|
| query        | string | required   | Search string             |
| asset_class  | string | "equities" | "equities" or "crypto"   |

**Response**:
```json
{
  "symbols": [
    {"symbol": "AAPL", "display": "AAPL - Apple Inc.", "name": "Apple Inc."},
    ...
  ]
}
```

---

### `POST /api/backtest/suggest`
**Description**: Get trade suggestions based on SMA analysis  

| Parameter     | Type   | Default    | Description             |
|---------------|--------|------------|-------------------------|
| symbol        | string | required   | Symbol                  |
| asset_class   | string | "equities" | Asset class             |
| initial_cash  | float  | 10000      | Starting capital        |
| fast_period   | int    | 10         | Fast SMA window         |
| slow_period   | int    | 30         | Slow SMA window         |
| period_days   | int    | 30         | Lookback period (days)  |
| timeframe     | string | "1d"       | Bar timeframe           |

**Response**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "asset_class": "equities",
  "signal": "BUY",
  "confidence": 0.65,
  "current_price": 185.50,
  "expected_monthly_return": 0.012,
  "expected_annual_return": 0.15,
  "risk_level": "LOW",
  "max_drawdown": 0.05,
  "sharpe_ratio": 1.2,
  "win_rate": 0.55,
  "recent_performance": {
    "total_return": 0.08,
    "num_trades": 5
  },
  "recommendation": "BUY AAPL with 65% confidence. Risk level: LOW"
}
```

---

## 15. Data Flow Diagrams

### Backtest Data Flow

```
                ┌──────────────┐
                │  Raw OHLCV   │
                │  (yfinance / │
                │   Binance)   │
                └──────┬───────┘
                       │ standardize_columns()
                       ▼
                ┌──────────────┐
                │  DataFrame   │
                │ timestamp,   │
                │ OHLCV (UTC)  │
                └──────┬───────┘
                       │ set_index([timestamp, symbol])
                       ▼
         ┌─────────────────────────────┐
         │     BacktestEngine.run()    │
         │                             │
         │  ┌───────────────────────┐  │
         │  │ Per Timestamp Loop    │  │
         │  │                       │  │
         │  │  ┌─────────────┐     │  │
         │  │  │  Strategy   │     │  │
         │  │  │  .on_bar()  │     │  │
         │  │  └──────┬──────┘     │  │
         │  │         │ weights    │  │
         │  │         ▼            │  │
         │  │  ┌─────────────┐     │  │
         │  │  │  SimBroker  │     │  │
         │  │  │  .process() │     │  │
         │  │  └──────┬──────┘     │  │
         │  │         │ fills      │  │
         │  │         ▼            │  │
         │  │  ┌─────────────┐     │  │
         │  │  │  Portfolio  │     │  │
         │  │  │  .add_fill()│     │  │
         │  │  │  .record()  │     │  │
         │  │  └─────────────┘     │  │
         │  └───────────────────────┘  │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   PerformanceReport          │
         │   .calculate_metrics()       │
         │                              │
         │   → metrics dict             │
         │   → equity_curve DataFrame   │
         │   → trades DataFrame         │
         └──────────────────────────────┘
```

### Live Execution Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ ReplayData   │     │ FeatureStore │     │ DummyModel   │
│ Source       │────▶│ .compute()   │────▶│ .predict()   │
│ (bars)       │     │              │     │ → Signal     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐              │
                     │ RiskManager  │◀─────────────┘
                     │ .validate()  │
                     └──────┬───────┘
                            │ approved?
                            ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ PaperBroker  │────▶│   Journal    │
                     │ .submit()    │     │ (SQLite log) │
                     └──────────────┘     └──────────────┘
```

---

## 16. Configuration & Deployment

### Starting the Backend

```bash
cd api
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Starting the Frontend

```bash
cd frontend
npm install
npm run dev          # → http://localhost:5173
```

### Vite Proxy Configuration

```js
// frontend/vite.config.js
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

### CLI Scripts

```bash
# Ingest data
python scripts/ingest.py --asset-class equities --symbols AAPL MSFT --timeframe 1d

# Run backtest
python scripts/run_backtest.py --data-file data.csv --fast-period 10 --slow-period 30

# Walk-forward analysis
python scripts/run_walk_forward.py --data-file data.csv
```

### Key Dependencies

| Package           | Version  | Purpose                        |
|-------------------|----------|--------------------------------|
| fastapi           | latest   | API framework                  |
| uvicorn           | latest   | ASGI server                    |
| pandas            | ^2.1     | Data manipulation              |
| numpy             | ^1.26    | Numerical computation          |
| scipy             | ^1.11    | Statistical functions          |
| yfinance          | ^0.2     | Yahoo Finance data             |
| pyarrow           | ^14.0    | Parquet read/write             |
| duckdb            | latest   | Analytical queries             |
| loguru            | ^0.7     | Structured logging             |
| pydantic          | ^2.5     | Data validation                |
| pydantic-settings | ^2.1     | Config management              |
| pyyaml            | ^6.0     | YAML config parsing            |
| ccxt              | ^4.1     | Multi-exchange crypto (unused in API) |
| scikit-learn      | ^1.3     | StandardScaler (dataset)       |
| requests          | ^2.31    | Binance API HTTP calls         |
| pytz              | ^2023.3  | Timezone handling              |
| react             | ^18.2    | Frontend framework             |
| recharts          | ^2.10    | Charting library               |
| axios             | ^1.6     | HTTP client                    |
| vite              | ^5.0     | Frontend bundler               |

---

*End of System Architecture Document*
