"""Bot API: start/stop bot, view decisions, performance, Monte Carlo results."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import APIRouter, HTTPException
from loguru import logger

from bot.bot_db import BotDB
from bot.autonomous_trader import AutonomousTrader
from bot.performance_tracker import compute_performance
from analytics.monte_carlo import run_monte_carlo
from analytics.statistical import full_statistical_analysis
from infra.terminal_db import TerminalDB

router = APIRouter(prefix="/api/bot", tags=["bot"])

_db: Optional[BotDB] = None
_terminal_db: Optional[TerminalDB] = None
_active_bot: Optional[AutonomousTrader] = None
_bot_task: Optional[asyncio.Task] = None


def _get_db() -> BotDB:
    global _db
    if _db is None:
        _db = BotDB("data/terminal.db")
    return _db


def _get_terminal_db() -> TerminalDB:
    global _terminal_db
    if _terminal_db is None:
        _terminal_db = TerminalDB("data/terminal.db")
    return _terminal_db


@router.post("/start")
async def start_bot(
    session_id: str = "bot-default",
    symbol: str = "BTCUSDT",
    trade_amount: float = 100.0,
    interval: int = 300,
):
    """Start the autonomous trading bot."""
    global _active_bot, _bot_task

    if _active_bot and _active_bot.is_running:
        return {"status": "already_running", "session_id": _active_bot.session_id, "symbol": _active_bot.symbol}

    tdb = _get_terminal_db()
    sess = tdb.get_session(session_id)
    if not sess:
        tdb.create_session(initial_cash=100000.0, session_id=session_id)
        logger.info(f"Created bot session '{session_id}' with $100,000 cash")

    _active_bot = AutonomousTrader(
        session_id=session_id,
        symbol=symbol,
        trade_amount=trade_amount,
        check_interval=interval,
    )
    _bot_task = asyncio.create_task(_active_bot.run())
    logger.info(f"Bot started: {symbol} @ ${trade_amount} every {interval}s")
    return {"status": "started", "session_id": session_id, "symbol": symbol, "trade_amount": trade_amount, "interval": interval}


@router.post("/stop")
async def stop_bot():
    """Stop the autonomous trading bot."""
    global _active_bot, _bot_task
    if _active_bot and _active_bot.is_running:
        _active_bot.stop()
        if _bot_task:
            _bot_task.cancel()
        logger.info("Bot stopped")
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/status")
async def bot_status():
    """Get bot running status."""
    if _active_bot and _active_bot.is_running:
        return {
            "running": True,
            "session_id": _active_bot.session_id,
            "symbol": _active_bot.symbol,
            "trade_amount": _active_bot.trade_amount,
            "trades_executed": _active_bot._trade_count,
        }
    return {"running": False}


@router.get("/decisions")
def get_decisions(session_id: str = "bot-default", limit: int = 100):
    """Get bot decision log."""
    db = _get_db()
    decisions = db.get_decisions(session_id, limit=limit)
    return {"decisions": decisions, "total": len(decisions)}


@router.get("/performance")
def get_performance(session_id: str = "bot-default"):
    """Get latest performance snapshot."""
    db = _get_db()
    perf = db.get_latest_performance(session_id)
    if not perf:
        tdb = _get_terminal_db()
        trades = tdb.get_trades(session_id, limit=10000)
        if trades:
            trade_dicts = [{"side": t.side, "pnl": t.pnl, "cost": t.cost, "price": t.price, "qty": t.qty, "timestamp": t.timestamp} for t in trades]
            portfolio = tdb.get_portfolio(session_id)
            perf = compute_performance(trade_dicts, portfolio_value=portfolio.get("equity", 0), initial_cash=portfolio.get("initial_cash", 100000))
            db.save_performance(session_id, perf)
        else:
            perf = {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
    return perf


@router.get("/performance-history")
def get_performance_history(session_id: str = "bot-default", limit: int = 50):
    """Get performance history over time."""
    db = _get_db()
    return {"snapshots": db.get_performance(session_id, limit=limit)}


@router.post("/monte-carlo")
async def run_mc(symbol: str = "BTCUSDT", horizon: int = 7, paths: int = 10000):
    """Run Monte Carlo simulation on demand."""
    try:
        from data.ingest.binance_public import BinancePublicIngestor
        import pandas as pd
        ingestor = BinancePublicIngestor()
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=365)
        df = await ingestor.fetch_ohlcv(symbol=symbol, timeframe="1d", start=start, end=end)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        prices = df["close"].values.astype(float)
        result = run_monte_carlo(symbol=symbol, prices=prices, horizon_days=horizon, num_paths=paths)
        _get_db().save_simulation(result.to_dict())
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(symbol: str = "BTCUSDT"):
    """Get statistical analysis for a symbol."""
    try:
        from data.ingest.binance_public import BinancePublicIngestor
        import pandas as pd
        ingestor = BinancePublicIngestor()
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=365)
        df = await ingestor.fetch_ohlcv(symbol=symbol, timeframe="1d", start=start, end=end)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        prices = df["close"].values.astype(float)
        return full_statistical_analysis(prices)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _ensure_session(session_id: str, initial_cash: float = 100000.0):
    """Ensure the session exists with proper initial cash."""
    tdb = _get_terminal_db()
    sess = tdb.get_session(session_id)
    if not sess:
        tdb.create_session(initial_cash=initial_cash, session_id=session_id)
        logger.info(f"Created bot session '{session_id}' with ${initial_cash} cash")


@router.post("/run-once")
async def run_once(session_id: str = "bot-default", symbol: str = "BTCUSDT", trade_amount: float = 100.0):
    """Run a single bot decision cycle (for testing)."""
    _ensure_session(session_id, initial_cash=100000.0)
    bot = AutonomousTrader(session_id=session_id, symbol=symbol, trade_amount=trade_amount)
    decision = await bot._run_cycle()
    if decision:
        return {
            "action": decision.action,
            "confidence": decision.confidence,
            "amount_usd": decision.amount_usd,
            "reasoning": decision.reasoning,
            "ml_signal": decision.ml_signal,
            "ml_confidence": decision.ml_confidence,
            "mc_var": decision.mc_var,
            "mc_expected_return": decision.mc_expected_return,
            "regime": decision.regime,
            "rsi": decision.rsi,
            "trend_strength": decision.trend_strength,
        }
    raise HTTPException(status_code=500, detail="Decision cycle failed")


SCAN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]


@router.get("/scan-all")
async def scan_all_cryptos():
    """Scan all major cryptos and return ranked opportunities."""
    from data.ingest.binance_public import BinancePublicIngestor
    import pandas as pd
    import numpy as np

    ingestor = BinancePublicIngestor()
    results = []

    for symbol in SCAN_SYMBOLS:
        try:
            end = pd.Timestamp.now(tz="UTC")
            start = end - pd.Timedelta(days=90)
            df = await ingestor.fetch_ohlcv(symbol=symbol, timeframe="1d", start=start, end=end)
            if df.empty or len(df) < 30:
                continue

            prices = df["close"].values.astype(float)
            current_price = float(prices[-1])

            mc = run_monte_carlo(symbol=symbol, prices=prices, horizon_days=7, num_paths=5000)
            stats = full_statistical_analysis(prices)

            sma_fast = np.mean(prices[-10:])
            sma_slow = np.mean(prices[-30:])
            rsi = stats.get("rsi", 50)
            momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0

            score = 0.5
            if sma_fast > sma_slow:
                score += 0.15
            else:
                score -= 0.15
            if rsi < 30:
                score += 0.15
            elif rsi > 70:
                score -= 0.15
            if momentum > 0.02:
                score += 0.1
            elif momentum < -0.02:
                score -= 0.1
            if mc.prob_profit > 0.5:
                score += 0.1
            score = max(0, min(1, score))

            if score >= 0.55:
                signal = "BUY"
            elif score <= 0.45:
                signal = "SELL"
            else:
                signal = "HOLD"

            results.append({
                "symbol": symbol,
                "price": current_price,
                "signal": signal,
                "confidence": score,
                "expected_return": mc.expected_return,
                "var_95": mc.var_95,
                "prob_profit": mc.prob_profit,
                "regime": stats.get("regime", "normal"),
                "rsi": rsi,
            })
        except Exception as e:
            logger.warning(f"Scan error for {symbol}: {e}")
            continue

    results.sort(key=lambda x: (x["signal"] == "BUY", x["confidence"], x["expected_return"]), reverse=True)
    return {"results": results}
