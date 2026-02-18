"""Futures trading database: positions, trades, margin tracking."""

import sqlite3
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path

from loguru import logger


@dataclass
class FuturesPosition:
    id: str
    session_id: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    current_price: float
    margin: float
    leverage: int
    position_size: float  # margin * leverage
    liquidation_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    realized_pnl: float
    status: str  # OPEN, CLOSED, LIQUIDATED
    opened_at: str
    closed_at: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class FuturesTrade:
    id: str
    session_id: str
    position_id: str
    symbol: str
    direction: str
    action: str  # OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT
    price: float
    margin: float
    leverage: int
    position_size: float
    pnl: float
    timestamp: str


class FuturesDB:
    """SQLite storage for simulated futures trading."""

    def __init__(self, db_path: str = "data/terminal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS futures_positions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                margin REAL NOT NULL,
                leverage INTEGER NOT NULL,
                position_size REAL NOT NULL,
                liquidation_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                opened_at TEXT NOT NULL,
                closed_at TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS futures_trades (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                position_id TEXT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                margin REAL NOT NULL,
                leverage INTEGER NOT NULL,
                position_size REAL NOT NULL,
                pnl REAL DEFAULT 0,
                timestamp TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS futures_sessions (
                id TEXT PRIMARY KEY,
                initial_margin REAL NOT NULL,
                current_margin REAL NOT NULL,
                total_pnl REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                liquidations INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_futures_pos_session ON futures_positions(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_futures_pos_status ON futures_positions(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_futures_trades_session ON futures_trades(session_id)")
        conn.commit()
        conn.close()
        logger.info("Futures DB tables initialized")

    def create_session(self, initial_margin: float = 100000.0, session_id: Optional[str] = None) -> dict:
        sid = session_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO futures_sessions 
            (id, initial_margin, current_margin, total_pnl, total_trades, winning_trades, liquidations, created_at)
            VALUES (?, ?, ?, 0, 0, 0, 0, ?)
        """, (sid, initial_margin, initial_margin, now))
        conn.commit()
        conn.close()
        return {"session_id": sid, "initial_margin": initial_margin, "current_margin": initial_margin}

    def get_session(self, session_id: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM futures_sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_session_margin(self, session_id: str, pnl: float, is_win: bool, is_liquidation: bool = False):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE futures_sessions SET 
                current_margin = current_margin + ?,
                total_pnl = total_pnl + ?,
                total_trades = total_trades + 1,
                winning_trades = winning_trades + ?,
                liquidations = liquidations + ?
            WHERE id = ?
        """, (pnl, pnl, 1 if is_win else 0, 1 if is_liquidation else 0, session_id))
        conn.commit()
        conn.close()

    def open_position(
        self,
        session_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        margin: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
    ) -> FuturesPosition:
        pid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        position_size = margin * leverage

        if direction == "LONG":
            liquidation_price = entry_price * (1 - 0.9 / leverage)
        else:
            liquidation_price = entry_price * (1 + 0.9 / leverage)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO futures_positions
            (id, session_id, symbol, direction, entry_price, current_price, margin, leverage,
             position_size, liquidation_price, stop_loss, take_profit, unrealized_pnl, realized_pnl,
             status, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 'OPEN', ?)
        """, (pid, session_id, symbol, direction, entry_price, entry_price, margin, leverage,
              position_size, liquidation_price, stop_loss, take_profit, now))

        cur.execute("""
            INSERT INTO futures_trades
            (id, session_id, position_id, symbol, direction, action, price, margin, leverage, position_size, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (str(uuid.uuid4()), session_id, pid, symbol, direction,
              f"OPEN_{direction}", entry_price, margin, leverage, position_size, now))

        conn.commit()
        conn.close()

        return FuturesPosition(
            id=pid, session_id=session_id, symbol=symbol, direction=direction,
            entry_price=entry_price, current_price=entry_price, margin=margin,
            leverage=leverage, position_size=position_size, liquidation_price=liquidation_price,
            stop_loss=stop_loss, take_profit=take_profit, unrealized_pnl=0, realized_pnl=0,
            status="OPEN", opened_at=now
        )

    def close_position(self, position_id: str, exit_price: float, reason: str = "MANUAL") -> Optional[FuturesPosition]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM futures_positions WHERE id = ? AND status = 'OPEN'", (position_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None

        pos = dict(row)
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        position_size = pos["position_size"]
        margin = pos["margin"]

        if direction == "LONG":
            pnl = (exit_price - entry_price) / entry_price * position_size
        else:
            pnl = (entry_price - exit_price) / entry_price * position_size

        status = "LIQUIDATED" if reason == "LIQUIDATION" else "CLOSED"
        now = datetime.utcnow().isoformat()

        cur.execute("""
            UPDATE futures_positions SET
                current_price = ?, realized_pnl = ?, unrealized_pnl = 0,
                status = ?, closed_at = ?
            WHERE id = ?
        """, (exit_price, pnl, status, now, position_id))

        cur.execute("""
            INSERT INTO futures_trades
            (id, session_id, position_id, symbol, direction, action, price, margin, leverage, position_size, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), pos["session_id"], position_id, pos["symbol"], direction,
              f"CLOSE_{direction}", exit_price, margin, pos["leverage"], position_size, pnl, now))

        conn.commit()
        conn.close()

        self.update_session_margin(pos["session_id"], pnl, pnl > 0, reason == "LIQUIDATION")

        return FuturesPosition(
            id=position_id, session_id=pos["session_id"], symbol=pos["symbol"],
            direction=direction, entry_price=entry_price, current_price=exit_price,
            margin=margin, leverage=pos["leverage"], position_size=position_size,
            liquidation_price=pos["liquidation_price"], stop_loss=pos["stop_loss"],
            take_profit=pos["take_profit"], unrealized_pnl=0, realized_pnl=pnl,
            status=status, opened_at=pos["opened_at"], closed_at=now
        )

    def get_open_positions(self, session_id: str, symbol: Optional[str] = None) -> List[FuturesPosition]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if symbol:
            cur.execute("SELECT * FROM futures_positions WHERE session_id = ? AND symbol = ? AND status = 'OPEN'",
                        (session_id, symbol))
        else:
            cur.execute("SELECT * FROM futures_positions WHERE session_id = ? AND status = 'OPEN'", (session_id,))

        rows = cur.fetchall()
        conn.close()

        return [FuturesPosition(**dict(r)) for r in rows]

    def get_all_positions(self, session_id: str, limit: int = 100) -> List[FuturesPosition]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM futures_positions WHERE session_id = ? ORDER BY opened_at DESC LIMIT ?",
                    (session_id, limit))
        rows = cur.fetchall()
        conn.close()
        return [FuturesPosition(**dict(r)) for r in rows]

    def get_trades(self, session_id: str, limit: int = 100) -> List[FuturesTrade]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM futures_trades WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit))
        rows = cur.fetchall()
        conn.close()
        return [FuturesTrade(**dict(r)) for r in rows]

    def update_position_price(self, position_id: str, current_price: float) -> Optional[float]:
        """Update position's current price and unrealized PnL. Returns unrealized PnL."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM futures_positions WHERE id = ? AND status = 'OPEN'", (position_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None

        pos = dict(row)
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        position_size = pos["position_size"]

        if direction == "LONG":
            unrealized_pnl = (current_price - entry_price) / entry_price * position_size
        else:
            unrealized_pnl = (entry_price - current_price) / entry_price * position_size

        cur.execute("UPDATE futures_positions SET current_price = ?, unrealized_pnl = ? WHERE id = ?",
                    (current_price, unrealized_pnl, position_id))
        conn.commit()
        conn.close()

        return unrealized_pnl
