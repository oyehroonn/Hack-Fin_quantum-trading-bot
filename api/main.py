import sys
from pathlib import Path

# Add parent directory to Python path so we can import backtest modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from decimal import Decimal
import uuid
import io
import traceback

from backtest.engine import BacktestEngine
from backtest.strategies.sma_crossover import SMACrossoverStrategy
from backtest.report import PerformanceReport

app = FastAPI(title="Trading Bot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BacktestConfig(BaseModel):
    initial_cash: float = 100000.0
    fast_period: int = 10
    slow_period: int = 30
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Trading Bot API"}

@app.post("/api/backtest")
async def run_backtest(
    file: UploadFile = File(...),
    initial_cash: float = 100000.0,
    fast_period: int = 10,
    slow_period: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Run backtest with uploaded data file."""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Parse CSV
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Prepare data
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "symbol" in df.columns:
                df = df.set_index(["timestamp", "symbol"])
            else:
                df = df.set_index("timestamp")
        
        # Parse dates
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else None
        
        # Get symbols
        if isinstance(df.index, pd.MultiIndex):
            symbols = list(df.index.get_level_values(1).unique())
        else:
            symbols = ["UNKNOWN"]
        
        # Create strategy
        strategy = SMACrossoverStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
        )
        
        # Run backtest
        engine = BacktestEngine(
            initial_cash=Decimal(str(initial_cash)),
            strategy=strategy,
            symbols=symbols,
        )
        
        portfolio = engine.run(df, start=start, end=end)
        
        # Generate report
        report = PerformanceReport(portfolio)
        metrics = report.calculate_metrics()
        
        # Get equity curve and trades
        equity_df = portfolio.get_equity_curve_df()
        trades_df = portfolio.get_trades_df()
        
        return {
            "success": True,
            "metrics": metrics,
            "equity_curve": equity_df.to_dict("records") if not equity_df.empty else [],
            "trades": trades_df.to_dict("records") if not trades_df.empty else [],
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/backtest: {error_msg}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

@app.post("/api/backtest/synthetic")
async def run_synthetic_backtest(config: BacktestConfig):
    """Run backtest with synthetic data (for testing)."""
    try:
        # Generate synthetic data
        dates = pd.date_range("2024-01-01", periods=100, freq="1D")
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "open": range(100, 200),
            "high": range(101, 201),
            "low": range(99, 199),
            "close": range(100, 200),
            "volume": [1000] * len(dates),
        })
        df = df.set_index(["timestamp", "symbol"])
        
        # Create strategy
        strategy = SMACrossoverStrategy(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
        )
        
        # Run backtest
        engine = BacktestEngine(
            initial_cash=Decimal(str(config.initial_cash)),
            strategy=strategy,
            symbols=["AAPL"],
        )
        
        portfolio = engine.run(df)
        
        # Generate report
        report = PerformanceReport(portfolio)
        metrics = report.calculate_metrics()
        
        # Get equity curve and trades
        equity_df = portfolio.get_equity_curve_df()
        trades_df = portfolio.get_trades_df()
        
        return {
            "success": True,
            "metrics": metrics,
            "equity_curve": equity_df.to_dict("records") if not equity_df.empty else [],
            "trades": trades_df.to_dict("records") if not trades_df.empty else [],
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/backtest/synthetic: {error_msg}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")
