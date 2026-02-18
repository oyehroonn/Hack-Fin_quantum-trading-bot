"""Futures Bot API: start/stop, positions, trades, performance."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import APIRouter, HTTPException
from loguru import logger

from bot.futures_db import FuturesDB
from bot.futures_trader import FuturesTrader
from bot.futures_decision_engine import FuturesDecisionEngine
from analytics.monte_carlo import run_monte_carlo
from analytics.statistical import full_statistical_analysis

router = APIRouter(prefix="/api/futures", tags=["futures"])

_db: Optional[FuturesDB] = None
_active_trader: Optional[FuturesTrader] = None
_trader_task: Optional[asyncio.Task] = None


def _get_db() -> FuturesDB:
    global _db
    if _db is None:
        _db = FuturesDB("data/terminal.db")
    return _db


@router.post("/start")
async def start_futures_bot(
    session_id: str = "futures-default",
    symbol: str = "BTCUSDT",
    base_margin: float = 100.0,
    max_leverage: int = 50,
    interval: int = 60,
):
    """Start the futures trading bot."""
    global _active_trader, _trader_task

    if _active_trader and _active_trader.is_running:
        return {
            "status": "already_running",
            "session_id": _active_trader.session_id,
            "symbol": _active_trader.symbol,
        }

    db = _get_db()
    session = db.get_session(session_id)
    if not session:
        db.create_session(initial_margin=100000.0, session_id=session_id)
        logger.info(f"Created futures session '{session_id}' with $100,000 margin")

    _active_trader = FuturesTrader(
        session_id=session_id,
        symbol=symbol,
        base_margin=base_margin,
        max_leverage=max_leverage,
        check_interval=interval,
    )
    _trader_task = asyncio.create_task(_active_trader.run())
    logger.info(f"Futures bot started: {symbol} @ ${base_margin} margin, {max_leverage}x max leverage")

    return {
        "status": "started",
        "session_id": session_id,
        "symbol": symbol,
        "base_margin": base_margin,
        "max_leverage": max_leverage,
        "interval": interval,
    }


@router.post("/stop")
async def stop_futures_bot():
    """Stop the futures trading bot."""
    global _active_trader, _trader_task
    if _active_trader and _active_trader.is_running:
        _active_trader.stop()
        if _trader_task:
            _trader_task.cancel()
        return {
            "status": "stopped",
            "trades": _active_trader._trade_count,
            "total_pnl": _active_trader._total_pnl,
        }
    return {"status": "not_running"}


@router.get("/status")
async def futures_status():
    """Get futures bot running status."""
    if _active_trader and _active_trader.is_running:
        return {
            "running": True,
            "session_id": _active_trader.session_id,
            "symbol": _active_trader.symbol,
            "max_leverage": _active_trader.max_leverage,
            "base_margin": _active_trader.base_margin,
            "trades_executed": _active_trader._trade_count,
            "total_pnl": _active_trader._total_pnl,
        }
    return {"running": False}


@router.get("/session")
def get_session(session_id: str = "futures-default"):
    """Get futures session info."""
    db = _get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/positions")
def get_positions(session_id: str = "futures-default", status: str = "OPEN"):
    """Get futures positions."""
    db = _get_db()
    if status == "OPEN":
        positions = db.get_open_positions(session_id)
    else:
        positions = db.get_all_positions(session_id)
    return {"positions": [p.to_dict() for p in positions]}


@router.get("/trades")
def get_trades(session_id: str = "futures-default", limit: int = 100):
    """Get futures trade history."""
    db = _get_db()
    trades = db.get_trades(session_id, limit=limit)
    return {"trades": [{"id": t.id, "symbol": t.symbol, "direction": t.direction, "action": t.action,
                        "price": t.price, "margin": t.margin, "leverage": t.leverage,
                        "position_size": t.position_size, "pnl": t.pnl, "timestamp": t.timestamp}
                       for t in trades]}


@router.post("/run-once")
async def run_once(
    session_id: str = "futures-default",
    symbol: str = "BTCUSDT",
    base_margin: float = 100.0,
    max_leverage: int = 50,
):
    """Run a single futures decision cycle."""
    db = _get_db()
    session = db.get_session(session_id)
    if not session:
        db.create_session(initial_margin=100000.0, session_id=session_id)

    trader = FuturesTrader(
        session_id=session_id,
        symbol=symbol,
        base_margin=base_margin,
        max_leverage=max_leverage,
    )
    decision = await trader._run_cycle()

    if decision:
        return {
            "action": decision.action,
            "direction": decision.direction,
            "leverage": decision.leverage,
            "margin": decision.margin,
            "position_size": decision.position_size,
            "entry_price": decision.entry_price,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "liquidation_price": decision.liquidation_price,
            "confidence": decision.confidence,
            "risk_score": decision.risk_score,
            "reasoning": decision.reasoning,
            "ml_signal": decision.ml_signal,
            "mc_expected_return": decision.mc_expected_return,
            "mc_var": decision.mc_var,
            "regime": decision.regime,
        }
    raise HTTPException(status_code=500, detail="Decision cycle failed")


SCAN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]


@router.get("/scan")
async def scan_futures_opportunities(max_leverage: int = 50):
    """Scan all cryptos for futures trading opportunities."""
    from data.ingest.binance_public import BinancePublicIngestor
    import pandas as pd
    import numpy as np

    ingestor = BinancePublicIngestor()
    engine = FuturesDecisionEngine(max_leverage=max_leverage)
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

            mc = run_monte_carlo(symbol=symbol, prices=prices, horizon_days=3, num_paths=3000)
            stats = full_statistical_analysis(prices)

            sma_fast = np.mean(prices[-10:])
            sma_slow = np.mean(prices[-30:])
            rsi = stats.get("rsi", 50)
            momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0

            score = 0.5
            if sma_fast > sma_slow * 1.01:
                score += 0.15
            elif sma_fast < sma_slow * 0.99:
                score -= 0.15
            if rsi < 30:
                score += 0.12
            elif rsi > 70:
                score -= 0.12
            if momentum > 0.02:
                score += 0.08
            elif momentum < -0.02:
                score -= 0.08

            score = max(0.1, min(0.9, score))

            if score >= 0.58:
                signal = "LONG"
                confidence = score
            elif score <= 0.42:
                signal = "SHORT"
                confidence = 1 - score
            else:
                signal = "NEUTRAL"
                confidence = 0.5

            risk_score = engine.calculate_risk_score(mc.var_95, stats.get("annualized_volatility", 0.5), stats.get("regime", "normal"), confidence)
            leverage = engine.calculate_leverage(risk_score, stats.get("regime", "normal"))

            results.append({
                "symbol": symbol,
                "price": current_price,
                "signal": signal,
                "confidence": confidence,
                "leverage": leverage,
                "risk_score": risk_score,
                "expected_return": mc.expected_return,
                "var_95": mc.var_95,
                "prob_profit": mc.prob_profit,
                "regime": stats.get("regime", "normal"),
                "rsi": rsi,
            })
        except Exception as e:
            logger.warning(f"Futures scan error for {symbol}: {e}")
            continue

    results.sort(key=lambda x: (x["signal"] != "NEUTRAL", x["confidence"]), reverse=True)
    return {"results": results}


@router.post("/close-position/{position_id}")
async def close_position(position_id: str):
    """Manually close a futures position."""
    db = _get_db()

    from data.ingest.binance_public import BinancePublicIngestor
    import pandas as pd

    positions = db.get_open_positions("futures-default")
    pos = next((p for p in positions if p.id == position_id), None)
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")

    ingestor = BinancePublicIngestor()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=1)
    df = await ingestor.fetch_ohlcv(symbol=pos.symbol, timeframe="1m", start=start, end=end, limit=5)
    if df.empty:
        raise HTTPException(status_code=500, detail="Could not get current price")

    current_price = float(df["close"].iloc[-1])
    closed = db.close_position(position_id, current_price, "MANUAL")

    if closed:
        return {
            "status": "closed",
            "position_id": position_id,
            "exit_price": current_price,
            "realized_pnl": closed.realized_pnl,
        }
    raise HTTPException(status_code=500, detail="Failed to close position")
