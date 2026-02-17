"""Terminal persistence: trades, positions, sessions in SQLite."""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


@dataclass
class Trade:
    id: str
    session_id: str
    symbol: str
    side: str
    qty: float
    price: float
    cost: float
    pnl: Optional[float]
    timestamp: str
    asset_class: str = "equities"


@dataclass
class Position:
    session_id: str
    symbol: str
    qty: float
    avg_cost: float
    updated_at: str


@dataclass
class Session:
    id: str
    initial_cash: float
    current_cash: float
    created_at: str


class TerminalDB:
    """SQLite-backed storage for terminal trades, positions, and sessions."""

    def __init__(self, db_path: str = "data/terminal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                initial_cash REAL NOT NULL,
                current_cash REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                cost REAL NOT NULL,
                pnl REAL,
                timestamp TEXT NOT NULL,
                asset_class TEXT DEFAULT 'equities',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_cost REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, symbol),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_positions_session ON positions(session_id)")
        conn.commit()
        conn.close()
        logger.info(f"Terminal DB initialized: {self.db_path}")

    def create_session(self, initial_cash: float = 100000.0, session_id: Optional[str] = None) -> Session:
        sid = session_id if session_id else str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (id, initial_cash, current_cash, created_at) VALUES (?, ?, ?, ?)",
            (sid, initial_cash, initial_cash, now),
        )
        conn.commit()
        conn.close()
        return Session(id=sid, initial_cash=initial_cash, current_cash=initial_cash, created_at=now)

    def get_or_create_session(self, session_id: Optional[str] = None, initial_cash: float = 100000.0) -> Session:
        if session_id:
            s = self.get_session(session_id)
            if s:
                return s
        return self.create_session(initial_cash)

    def get_session(self, session_id: str) -> Optional[Session]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, initial_cash, current_cash, created_at FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return Session(id=row[0], initial_cash=row[1], current_cash=row[2], created_at=row[3])

    def update_session_cash(self, session_id: str, current_cash: float) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE sessions SET current_cash = ? WHERE id = ?", (current_cash, session_id))
        conn.commit()
        conn.close()

    def execute_trade(
        self,
        session_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        asset_class: str = "equities",
    ) -> Trade:
        cost = qty * price
        pnl = None
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        sess = self.get_session(session_id)
        if not sess:
            conn.close()
            raise ValueError(f"Session {session_id} not found")

        if side.upper() == "BUY":
            if cost > sess.current_cash:
                conn.close()
                raise ValueError("Insufficient funds")
            new_cash = sess.current_cash - cost
            pos = self._get_position(cur, session_id, symbol)
            if pos:
                new_qty = pos[2] + qty
                new_avg = (pos[2] * pos[3] + cost) / new_qty
                cur.execute(
                    "UPDATE positions SET qty = ?, avg_cost = ?, updated_at = ? WHERE session_id = ? AND symbol = ?",
                    (new_qty, new_avg, datetime.utcnow().isoformat(), session_id, symbol),
                )
            else:
                cur.execute(
                    "INSERT INTO positions (session_id, symbol, qty, avg_cost, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, symbol, qty, price, datetime.utcnow().isoformat()),
                )
            cur.execute("UPDATE sessions SET current_cash = ? WHERE id = ?", (new_cash, session_id))

        elif side.upper() == "SELL":
            pos = self._get_position(cur, session_id, symbol)
            if not pos or pos[2] < qty:
                conn.close()
                raise ValueError("Insufficient position")
            avg_cost = pos[3]
            pnl = (price - avg_cost) * qty
            proceeds = qty * price
            new_cash = sess.current_cash + proceeds
            new_qty = pos[2] - qty
            if new_qty <= 0:
                cur.execute("DELETE FROM positions WHERE session_id = ? AND symbol = ?", (session_id, symbol))
            else:
                cur.execute(
                    "UPDATE positions SET qty = ?, updated_at = ? WHERE session_id = ? AND symbol = ?",
                    (new_qty, datetime.utcnow().isoformat(), session_id, symbol),
                )
            cur.execute("UPDATE sessions SET current_cash = ? WHERE id = ?", (new_cash, session_id))
        else:
            conn.close()
            raise ValueError(f"Invalid side: {side}")

        trade_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        cur.execute(
            """INSERT INTO trades (id, session_id, symbol, side, qty, price, cost, pnl, timestamp, asset_class)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, session_id, symbol, side.upper(), qty, price, cost, pnl, now, asset_class),
        )
        conn.commit()
        conn.close()
        return Trade(
            id=trade_id,
            session_id=session_id,
            symbol=symbol,
            side=side.upper(),
            qty=qty,
            price=price,
            cost=cost,
            pnl=pnl,
            timestamp=now,
            asset_class=asset_class,
        )

    def _get_position(self, cur: sqlite3.Cursor, session_id: str, symbol: str) -> Optional[tuple]:
        cur.execute("SELECT session_id, symbol, qty, avg_cost, updated_at FROM positions WHERE session_id = ? AND symbol = ?", (session_id, symbol))
        return cur.fetchone()

    def get_positions(self, session_id: str) -> list[Position]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT session_id, symbol, qty, avg_cost, updated_at FROM positions WHERE session_id = ?", (session_id,))
        rows = cur.fetchall()
        conn.close()
        return [Position(session_id=r[0], symbol=r[1], qty=r[2], avg_cost=r[3], updated_at=r[4]) for r in rows]

    def get_all_trades(self, limit: int = 10000) -> list[Trade]:
        """Return all trades across all sessions, for analytics/ML export."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """SELECT id, session_id, symbol, side, qty, price, cost, pnl, timestamp, asset_class
               FROM trades ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            Trade(id=r[0], session_id=r[1], symbol=r[2], side=r[3], qty=r[4], price=r[5], cost=r[6], pnl=r[7], timestamp=r[8], asset_class=r[9] or "equities")
            for r in rows
        ]

    def get_trades(self, session_id: str, symbol: Optional[str] = None, limit: int = 100) -> list[Trade]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if symbol:
            cur.execute(
                """SELECT id, session_id, symbol, side, qty, price, cost, pnl, timestamp, asset_class
                   FROM trades WHERE session_id = ? AND symbol = ? ORDER BY timestamp DESC LIMIT ?""",
                (session_id, symbol, limit),
            )
        else:
            cur.execute(
                """SELECT id, session_id, symbol, side, qty, price, cost, pnl, timestamp, asset_class
                   FROM trades WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?""",
                (session_id, limit),
            )
        rows = cur.fetchall()
        conn.close()
        return [
            Trade(id=r[0], session_id=r[1], symbol=r[2], side=r[3], qty=r[4], price=r[5], cost=r[6], pnl=r[7], timestamp=r[8], asset_class=r[9] or "equities")
            for r in rows
        ]

    def get_portfolio(self, session_id: str, prices: Optional[dict[str, float]] = None) -> dict[str, Any]:
        sess = self.get_session(session_id)
        if not sess:
            return {}
        positions = self.get_positions(session_id)
        equity = sess.current_cash
        positions_list = []
        for p in positions:
            mark = (prices or {}).get(p.symbol, p.avg_cost)
            value = p.qty * mark
            equity += value
            positions_list.append({
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_cost": p.avg_cost,
                "market_value": value,
                "unrealized_pnl": (mark - p.avg_cost) * p.qty,
            })
        realized = sum(
            t.pnl or 0
            for t in self.get_trades(session_id, limit=10000)
            if t.side == "SELL" and t.pnl is not None
        )
        return {
            "session_id": session_id,
            "cash": sess.current_cash,
            "equity": equity,
            "initial_cash": sess.initial_cash,
            "realized_pnl": realized,
            "positions": positions_list,
        }
