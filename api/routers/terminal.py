"""Terminal API: trades, positions, portfolio, session persistence."""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import TerminalTradeRequest
from infra.terminal_db import TerminalDB

router = APIRouter(prefix="/api/terminal", tags=["terminal"])

_db: Optional[TerminalDB] = None


def _get_db() -> TerminalDB:
    global _db
    if _db is None:
        _db = TerminalDB("data/terminal.db")
    return _db


@router.post("/session")
def create_session(initial_cash: float = 100000.0):
    """Create a new trading session."""
    db = _get_db()
    session = db.create_session(initial_cash=initial_cash)
    return {"session_id": session.id, "initial_cash": session.initial_cash, "created_at": session.created_at}


@router.get("/session/{session_id}")
def get_session(session_id: str):
    """Get session info."""
    db = _get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session.id, "initial_cash": session.initial_cash, "current_cash": session.current_cash, "created_at": session.created_at}


@router.post("/trade")
def execute_trade(req: TerminalTradeRequest):
    """Execute a BUY or SELL trade."""
    db = _get_db()
    try:
        trade = db.execute_trade(
            session_id=req.session_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            price=req.price,
            asset_class=req.asset_class or "equities",
        )
        return {
            "id": trade.id,
            "session_id": trade.session_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "qty": trade.qty,
            "price": trade.price,
            "cost": trade.cost,
            "pnl": trade.pnl,
            "timestamp": trade.timestamp,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/trades")
def list_trades(session_id: str, symbol: Optional[str] = None, limit: int = 100):
    """List trades for a session."""
    db = _get_db()
    trades = db.get_trades(session_id, symbol=symbol, limit=limit)
    return {
        "trades": [
            {"id": t.id, "symbol": t.symbol, "side": t.side, "qty": t.qty, "price": t.price, "cost": t.cost, "pnl": t.pnl, "timestamp": t.timestamp}
            for t in trades
        ]
    }


@router.get("/positions")
def list_positions(session_id: str):
    """List positions for a session."""
    db = _get_db()
    positions = db.get_positions(session_id)
    return {
        "positions": [
            {"symbol": p.symbol, "qty": p.qty, "avg_cost": p.avg_cost, "updated_at": p.updated_at}
            for p in positions
        ]
    }


@router.get("/all-trades")
def list_all_trades(limit: int = 5000):
    """Export all trades across sessions for analytics/ML training."""
    db = _get_db()
    trades = db.get_all_trades(limit=limit)
    return {
        "trades": [
            {
                "id": t.id,
                "session_id": t.session_id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "price": t.price,
                "cost": t.cost,
                "pnl": t.pnl,
                "timestamp": t.timestamp,
                "asset_class": t.asset_class,
            }
            for t in trades
        ],
        "total": len(trades),
    }


@router.get("/default-session")
def get_default_session(initial_cash: float = 100000.0):
    """Get or create the shared default session. Use when localStorage is empty."""
    db = _get_db()
    DEFAULT_ID = "default"
    sess = db.get_session(DEFAULT_ID)
    if sess:
        return {"session_id": sess.id, "initial_cash": sess.initial_cash, "created_at": sess.created_at}
    session = db.create_session(initial_cash=initial_cash, session_id=DEFAULT_ID)
    return {"session_id": session.id, "initial_cash": session.initial_cash, "created_at": session.created_at}


@router.get("/portfolio")
def get_portfolio(session_id: str, prices: Optional[str] = None):
    """Get portfolio summary. Optional query param prices=SYM1:123.45,SYM2:67.89 for mark prices."""
    db = _get_db()
    price_map = {}
    if prices:
        for part in prices.split(","):
            if ":" in part:
                sym, val = part.split(":", 1)
                try:
                    price_map[sym.strip().upper()] = float(val.strip())
                except ValueError:
                    pass
    portfolio = db.get_portfolio(session_id, prices=price_map if price_map else None)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Session not found")
    return portfolio
