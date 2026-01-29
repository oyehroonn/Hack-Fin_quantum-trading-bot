# File Tree

```
quantum-trading-bot/
├── Makefile                    # Build, lint, test, demo commands
├── README.md                   # Project documentation
├── pyproject.toml              # Poetry dependencies and tool configs
├── config.yaml                 # Default configuration
├── .gitignore                  # Git ignore patterns
│
├── core/                       # Core types and interfaces
│   ├── __init__.py
│   ├── types.py                # Bar, Tick, Order, Fill, Position, PortfolioState, Signal
│   └── interfaces.py           # DataSource, FeatureStore, Model, Strategy, Broker, RiskManager
│
├── data/                       # Data sources
│   ├── __init__.py
│   └── dummy_source.py         # DummyDataSource - synthetic bar generation
│
├── features/                   # Feature engineering
│   ├── __init__.py
│   ├── simple_store.py         # SimpleFeatureStore - returns + SMA
│   └── dummy_model.py          # DummyModel - random but reproducible signals
│
├── execution/                  # Order execution
│   ├── __init__.py
│   └── paper_broker.py         # PaperBroker - simulates fills with slippage+fees
│
├── risk/                       # Risk management
│   ├── __init__.py
│   └── simple_risk.py          # SimpleRiskManager - max position + max leverage
│
├── infra/                      # Infrastructure
│   ├── __init__.py
│   ├── config.py               # Config loader (YAML + env overrides)
│   └── logging.py              # Structured logging with correlation IDs
│
├── scripts/                    # Utility scripts
│   └── run_demo.py             # End-to-end demo runner
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_types.py           # Tests for core types
│   ├── test_config.py          # Tests for config loading
│   ├── test_risk.py            # Tests for risk manager
│   └── test_broker.py          # Tests for broker fill logic
│
├── backtest/                   # Backtesting engine (placeholder)
│   └── __init__.py
│
├── portfolio/                  # Portfolio management (placeholder)
│   └── __init__.py
│
├── research/                   # Research notebooks (placeholder)
│   └── __init__.py
│
└── rl/                         # Reinforcement learning (placeholder)
    └── __init__.py
```

## Key Components

### Core Types (`core/types.py`)
- `Bar`: OHLCV bar data with validation
- `Tick`: Tick data (trade/quote)
- `Order`: Order representation with side, type, quantity
- `Fill`: Order fill with price, quantity, fees
- `Position`: Position with quantity, avg_price, PnL
- `PortfolioState`: Portfolio snapshot with cash, positions, PnL
- `Signal`: Trading signal with strength, confidence, targets

### Interfaces (`core/interfaces.py`)
- `DataSource`: Stream bars/ticks
- `FeatureStore`: Compute and retrieve features
- `Model`: Generate predictions/signals
- `Strategy`: Generate trading signals
- `Broker`: Submit orders, get fills, positions
- `RiskManager`: Validate orders, check limits

### Example Vertical Slice
1. **DummyDataSource**: Generates synthetic OHLCV bars using random walk
2. **SimpleFeatureStore**: Computes returns and SMA from bars
3. **DummyModel**: Generates random but reproducible signals based on features
4. **PaperBroker**: Simulates order execution with slippage and commission
5. **SimpleRiskManager**: Enforces max position size and max leverage
6. **run_demo.py**: Orchestrates the full pipeline for 1000 steps

### Configuration
- YAML-based config with environment variable overrides
- Uses `pydantic-settings` for type-safe configuration
- Environment variables prefixed with `TRADING_` override YAML values

### Logging
- Structured logging with `loguru`
- Correlation IDs for request tracing
- Configurable log levels and file output

### Testing
- 3+ basic tests for types and config
- Tests for risk manager limits
- Tests for broker fill logic and position updates
