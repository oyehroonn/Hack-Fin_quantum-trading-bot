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
import pytz

from backtest.engine import BacktestEngine
from backtest.strategies.sma_crossover import SMACrossoverStrategy
from backtest.report import PerformanceReport
from data.ingest.equities_yfinance import EquitiesYFinanceIngestor
from data.ingest.binance_public import BinancePublicIngestor
from loguru import logger

app = FastAPI(
    title="Quantum Trading Bot API",
    version="0.2.0",
    description="State-of-the-art trading bot with ML prediction models, regime detection, and ensemble strategies",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount v0.2 routers ──
try:
    from api.routers.models import router as models_router
    from api.routers.research import router as research_router
    app.include_router(models_router)
    app.include_router(research_router)
    logger.info("v0.2 routers mounted: /api/models, /api/research")
except ImportError as e:
    logger.warning(f"Could not mount v0.2 routers: {e}")

class BacktestConfig(BaseModel):
    initial_cash: float = 100000.0
    fast_period: int = 10
    slow_period: int = 30
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "Quantum Trading Bot API",
        "version": "0.2.0",
        "endpoints": {
            "backtest": "/api/backtest/*",
            "models": "/api/models/*",
            "research": "/api/research/*",
        },
    }

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
        utc = pytz.UTC
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Normalize to UTC
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(utc)
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert(utc)
            if "symbol" in df.columns:
                df = df.set_index(["timestamp", "symbol"])
            else:
                df = df.set_index("timestamp")
        
        # Parse dates and normalize to UTC
        if start_date:
            start = pd.to_datetime(start_date)
            if start.tzinfo is None:
                start = start.tz_localize(utc)
            else:
                start = start.tz_convert(utc)
        else:
            start = None
            
        if end_date:
            end = pd.to_datetime(end_date)
            if end.tzinfo is None:
                end = end.tz_localize(utc)
            else:
                end = end.tz_convert(utc)
        else:
            end = None
        
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
        # Generate synthetic data with UTC timestamps
        utc = pytz.UTC
        dates = pd.date_range("2024-01-01", periods=100, freq="1D", tz=utc)
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

@app.post("/api/backtest/real")
async def run_real_backtest(
    symbol: str,
    initial_cash: float = 100000.0,
    fast_period: int = 10,
    slow_period: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "1d",
):
    """Run backtest with real stock data from Yahoo Finance."""
    try:
        # Parse dates and ensure UTC timezone
        utc = pytz.UTC
        
        if start_date:
            start = pd.to_datetime(start_date)
            if start.tzinfo is None:
                start = start.tz_localize(utc)
            else:
                start = start.tz_convert(utc)
        else:
            start = pd.Timestamp.now(tz=utc) - pd.Timedelta(days=365)
            
        if end_date:
            end = pd.to_datetime(end_date)
            if end.tzinfo is None:
                end = end.tz_localize(utc)
            else:
                end = end.tz_convert(utc)
        else:
            end = pd.Timestamp.now(tz=utc)
        
        # Fetch real data
        ingestor = EquitiesYFinanceIngestor()
        
        # Fetch data (handles async/blocking internally)
        df = await ingestor.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for symbol {symbol}. Please verify the symbol is correct and try a different date range."
            )
        
        # Ensure we have required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=500,
                detail=f"Data format error: missing columns {missing_cols}"
            )
        
        # Add symbol column if not present (needed for multi-index)
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        
        # Ensure timestamp is datetime and UTC
        utc = pytz.UTC
        
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Normalize to UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(utc)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(utc)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Set multi-index (timestamp, symbol) as required by backtest engine
        df = df.set_index(["timestamp", "symbol"])
        
        logger.info(f"Prepared data for backtest: {len(df)} bars, index: {df.index.names}")
        
        # Create strategy
        strategy = SMACrossoverStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
        )
        
        # Run backtest
        engine = BacktestEngine(
            initial_cash=Decimal(str(initial_cash)),
            strategy=strategy,
            symbols=[symbol],
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
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/backtest/real: {error_msg}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

@app.get("/api/symbols/search")
async def search_symbols(query: str, asset_class: str = "equities"):
    """Search for symbols (stocks or crypto).
    
    Args:
        query: Search query (symbol or name)
        asset_class: 'equities' or 'crypto'
    """
    try:
        query_upper = query.upper().strip()
        
        if asset_class == "crypto":
            # Popular crypto pairs on Binance
            popular_crypto = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
                "XRPUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT",
                "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "ETCUSDT",
            ]
            
            # Filter by query
            results = [s for s in popular_crypto if query_upper in s]
            
            # Format for frontend
            return {
                "symbols": [
                    {
                        "symbol": s,
                        "display": f"{s[:-4]}/{s[-4:]}",  # BTC/USDT
                        "name": s[:-4],
                    }
                    for s in results[:20]  # Limit to 20
                ]
            }
        else:
            # For equities, use popular stocks list
            popular_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                "BRK-B", "V", "JNJ", "WMT", "JPM", "MA", "PG", "UNH",
                "HD", "DIS", "BAC", "ADBE", "NFLX", "CRM", "PYPL", "INTC",
                "CMCSA", "PFE", "TMO", "COST", "ABBV", "AVGO", "CSCO",
            ]
            
            # Filter by query
            results = [s for s in popular_stocks if query_upper in s]
            
            # Try to get more info from yfinance
            symbols_with_info = []
            for sym in results[:20]:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(sym)
                    info = ticker.info
                    symbols_with_info.append({
                        "symbol": sym,
                        "display": f"{sym} - {info.get('longName', sym)}",
                        "name": info.get("longName", sym),
                    })
                except:
                    symbols_with_info.append({
                        "symbol": sym,
                        "display": sym,
                        "name": sym,
                    })
            
            return {"symbols": symbols_with_info}
            
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching symbols: {str(e)}")

@app.post("/api/backtest/crypto")
async def run_crypto_backtest(
    symbol: str,
    initial_cash: float = 100000.0,
    fast_period: int = 10,
    slow_period: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "1d",
):
    """Run backtest with crypto data from Binance."""
    try:
        # Parse dates and ensure UTC timezone
        utc = pytz.UTC
        
        if start_date:
            start = pd.to_datetime(start_date)
            if start.tzinfo is None:
                start = start.tz_localize(utc)
            else:
                start = start.tz_convert(utc)
        else:
            start = pd.Timestamp.now(tz=utc) - pd.Timedelta(days=30)
            
        if end_date:
            end = pd.to_datetime(end_date)
            if end.tzinfo is None:
                end = end.tz_localize(utc)
            else:
                end = end.tz_convert(utc)
        else:
            end = pd.Timestamp.now(tz=utc)
        
        # Fetch crypto data
        ingestor = BinancePublicIngestor()
        df = await ingestor.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {symbol}. Please verify the symbol is correct (e.g., BTCUSDT, ETHUSDT)."
            )
        
        # Ensure we have required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=500,
                detail=f"Data format error: missing columns {missing_cols}"
            )
        
        # Add symbol column
        if "symbol" not in df.columns:
            df["symbol"] = symbol.replace("/", "").upper()
        
        # Ensure timestamp is datetime and UTC
        utc = pytz.UTC
        
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Normalize to UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(utc)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(utc)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Set multi-index
        df = df.set_index(["timestamp", "symbol"])
        
        logger.info(f"Prepared crypto data for backtest: {len(df)} bars")
        
        # Create strategy
        strategy = SMACrossoverStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
        )
        
        # Run backtest
        engine = BacktestEngine(
            initial_cash=Decimal(str(initial_cash)),
            strategy=strategy,
            symbols=[symbol.replace("/", "").upper()],
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
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/backtest/crypto: {error_msg}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

@app.post("/api/backtest/suggest")
async def get_trade_suggestions(
    symbol: str,
    asset_class: str = "equities",
    initial_cash: float = 10000.0,
    fast_period: int = 10,
    slow_period: int = 30,
    period_days: int = 30,
    timeframe: str = "1d",
):
    """Get trade suggestions based on backtest analysis.
    
    Returns:
        - Current signal (BUY/SELL/HOLD)
        - Confidence level
        - Expected results based on recent performance
        - Risk assessment
    """
    try:
        from datetime import timedelta
        utc = pytz.UTC
        
        # Calculate date range (ensure UTC)
        end = pd.Timestamp.now(tz=utc)
        start = end - pd.Timedelta(days=period_days)
        
        # Fetch data based on asset class
        if asset_class == "crypto":
            ingestor = BinancePublicIngestor()
            symbol_clean = symbol.replace("/", "").upper()
        else:
            ingestor = EquitiesYFinanceIngestor()
            symbol_clean = symbol.upper()
        
        df = await ingestor.fetch_ohlcv(
            symbol=symbol_clean,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol}"
            )
        
        # Prepare data
        utc = pytz.UTC
        
        if "symbol" not in df.columns:
            df["symbol"] = symbol_clean
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Normalize to UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(utc)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(utc)
            
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.set_index(["timestamp", "symbol"])
        
        # Run quick backtest
        strategy = SMACrossoverStrategy(fast_period=fast_period, slow_period=slow_period)
        engine = BacktestEngine(
            initial_cash=Decimal(str(initial_cash)),
            strategy=strategy,
            symbols=[symbol_clean],
        )
        portfolio = engine.run(df, start=start, end=end)
        
        # Get latest signal
        latest_trades = portfolio.get_trades_df()
        if not latest_trades.empty:
            latest_trade = latest_trades.iloc[-1]
            signal_side = "BUY" if latest_trade["side"] == "BUY" else "SELL"
        else:
            # Analyze current price vs SMAs
            if len(df) >= slow_period:
                close_prices = df["close"].values
                fast_sma = pd.Series(close_prices).rolling(fast_period).mean().iloc[-1]
                slow_sma = pd.Series(close_prices).rolling(slow_period).mean().iloc[-1]
                current_price = close_prices[-1]
                
                if fast_sma > slow_sma and current_price > fast_sma:
                    signal_side = "BUY"
                elif fast_sma < slow_sma and current_price < fast_sma:
                    signal_side = "SELL"
                else:
                    signal_side = "HOLD"
            else:
                signal_side = "HOLD"
        
        # Calculate confidence based on recent performance
        report = PerformanceReport(portfolio)
        metrics = report.calculate_metrics()
        
        # Confidence calculation
        sharpe = metrics.get("sharpe", 0)
        win_rate = metrics.get("win_rate", 0)
        total_return = metrics.get("total_return", 0)
        
        confidence = min(0.95, max(0.1, abs(total_return) * 0.5 + win_rate * 0.3 + min(sharpe / 10, 0.2)))
        
        # Risk assessment
        max_dd = abs(metrics.get("max_drawdown", 0))
        if max_dd > 0.20:
            risk_level = "HIGH"
        elif max_dd > 0.10:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Expected results
        expected_return = total_return * (30 / period_days)  # Annualized estimate
        
        return {
            "success": True,
            "symbol": symbol,
            "asset_class": asset_class,
            "signal": signal_side,
            "confidence": round(confidence, 2),
            "current_price": float(df["close"].iloc[-1]),
            "expected_monthly_return": round(expected_return / 12, 4),
            "expected_annual_return": round(expected_return, 4),
            "risk_level": risk_level,
            "max_drawdown": metrics.get("max_drawdown", 0),
            "sharpe_ratio": metrics.get("sharpe", 0),
            "win_rate": metrics.get("win_rate", 0),
            "recent_performance": {
                "total_return": metrics.get("total_return", 0),
                "num_trades": metrics.get("num_trades", 0),
            },
            "recommendation": f"{signal_side} {symbol} with {confidence:.0%} confidence. Risk level: {risk_level}",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"ERROR in /api/backtest/suggest: {error_msg}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")
