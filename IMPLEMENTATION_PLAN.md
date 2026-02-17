# Quantum Trading Bot — Full Implementation Plan

> **Goal**: Premium live terminal, persistent trade storage, analytics DB, pattern recognition, Monte Carlo simulations, and continuous ML reinforcement.

---

## Phase 1: Trade Persistence

### 1.1 Backend

| Component | Description |
|-----------|-------------|
| **Trades table** | SQLite `trades` table: `id`, `session_id`, `symbol`, `side`, `qty`, `price`, `cost`, `pnl`, `timestamp`, `asset_class` |
| **Positions table** | `positions`: `session_id`, `symbol`, `qty`, `avg_cost`, `updated_at` |
| **Sessions table** | `sessions`: `id`, `initial_cash`, `current_cash`, `created_at` |
| **API endpoints** | |
| `POST /api/terminal/trade` | Execute BUY/SELL; persist trade + update positions |
| `GET /api/terminal/trades` | List trades (optional: session_id, symbol, limit) |
| `GET /api/terminal/positions` | Current positions for session |
| `GET /api/terminal/portfolio` | Cash, equity, realized P&L |
| `POST /api/terminal/session` | Create/reset session (initial_cash) |

### 1.2 Frontend

- Replace local `terminalState` with API calls
- On load: fetch session (or create), positions, trades
- On BUY/SELL: call `POST /api/terminal/trade`, then refetch
- Optional: `session_id` in URL or localStorage for persistence across refresh

---

## Phase 2: Premium Interactive UI

### 2.1 Candlestick Chart

| Item | Choice |
|------|--------|
| Library | `lightweight-charts` (TradingView) or `react-financial-charts` |
| Data source | `GET /api/market/ohlcv?symbol=X&timeframe=1d&days=60` |
| Updates | Poll every 60s or on symbol change |
| Overlay | Mark BUY/SELL points on chart (markers) |
| Timeframes | 1d, 1h (if data available) |

### 2.2 Live Price Updates

| Option | Implementation |
|--------|----------------|
| **A: Polling** | Poll `/api/backtest/suggest` or new `/api/market/price?symbol=X` every 10–30s |
| **B: WebSocket** | Binance WebSocket for crypto; yfinance has no WS, use polling for equities |
| Choice | Polling for both (simpler); WebSocket for crypto later |

### 2.3 Layout (Binance-style)

- **Left**: Symbol search, asset class, order amount, action buttons, portfolio summary
- **Center**: Candlestick chart (large), position markers
- **Right**: Positions list, recent trades
- **Bottom**: Live price ticker, 24h change

### 2.4 New API

- `GET /api/market/ohlcv` — OHLCV for chart (reuse ingestors)
- `GET /api/market/price` — Latest price (lightweight, for ticker)

---

## Phase 3: Analytics Database

### 3.1 Schema

| Table | Purpose |
|-------|---------|
| `trades` | All executed trades (from Phase 1) |
| `ohlcv` | Synced from ParquetStore or ingested |
| `signals` | Model predictions, regime, labels |
| `patterns` | Detected patterns (breakout, mean_reversion, etc.) |
| `simulations` | Monte Carlo run metadata + summary stats |
| `model_predictions` | Timestamped predictions for backtesting |

### 3.2 Sync Layer

- `sync_parquet_to_db()` — Copy Parquet OHLCV into analytics DB
- `sync_journal_to_trades()` — Ensure Journal FILL events → trades table
- Scheduled job (e.g. daily) or on-demand

### 3.3 Database Choice

- **SQLite** for Phase 1–3 (simplicity, single file)
- **PostgreSQL** or **DuckDB** later for heavy analytics

---

## Phase 4: Monte Carlo & Statistical Analysis

### 4.1 Monte Carlo Engine

| Component | Description |
|-----------|-------------|
| **Path simulation** | Geometric Brownian motion, jump-diffusion |
| **Parameters** | `mu`, `sigma` from historical returns |
| **Runs** | 10,000+ simulations per symbol |
| **Outputs** | VaR, CVaR, drawdown distribution, path percentiles |
| **API** | `POST /api/analytics/monte-carlo` — returns summary stats |

### 4.2 Pattern Recognition

| Pattern | Detection |
|---------|-----------|
| Breakout | Price vs N-day high/low |
| Mean reversion | Z-score, RSI extreme |
| Trend | ADX, slope |
| Volatility regime | Rolling vol percentile |

Store in `patterns` table with `symbol`, `timestamp`, `pattern_type`, `strength`, `metadata`.

### 4.3 Statistical APIs

- `GET /api/analytics/returns?symbol=X&window=20` — Rolling returns
- `GET /api/analytics/volatility?symbol=X` — Realized vol
- `GET /api/analytics/patterns?symbol=X&days=30` — Detected patterns

---

## Phase 5: ML Reinforcement & Continuous Learning

### 5.1 Feedback Loop

1. **Trade outcomes** → labels (win/loss, P&L)
2. **Market data** → features
3. **Periodic retrain** (e.g. weekly) on new data
4. **Champion/challenger** — promote if challenger wins walk-forward

### 5.2 Data Flow

```
Trades DB → Label generation → Training set
     ↑
Market data (OHLCV) → Features → Model prediction
     ↑
New trades (outcomes) → Reinforce
```

### 5.3 Components

- **Label service** — `edge_label`, `triple_barrier` from trades + prices
- **Retrain job** — Scheduled; uses `research/tuning`, `models/registry`
- **Prediction API** — `POST /api/models/predict` (exists) + batch for backtesting

---

## File & Endpoint Summary

| New/Modified | Path | Purpose |
|--------------|------|---------|
| API | `api/routers/terminal.py` | Trades, positions, portfolio, session |
| API | `api/routers/market.py` | OHLCV, price |
| API | `api/routers/analytics.py` | Monte Carlo, patterns |
| DB | `infra/terminal_db.py` | Trades, positions, sessions schema |
| Frontend | `frontend/src/components/CandlestickChart.jsx` | Chart |
| Frontend | `frontend/src/components/TerminalLayout.jsx` | Layout |
| Frontend | `frontend/src/hooks/useLivePrice.js` | Polling |
| Analytics | `analytics/monte_carlo.py` | Monte Carlo engine |
| Analytics | `analytics/patterns.py` | Pattern detection |

---

## Implementation Order

1. **Phase 1** — Terminal DB + API + Frontend integration (trades persist)
2. **Phase 2** — Candlestick chart + live polling + layout
3. **Phase 3** — Analytics DB schema + basic sync
4. **Phase 4** — Monte Carlo + pattern detection APIs
5. **Phase 5** — Retrain pipeline (scaffold)
