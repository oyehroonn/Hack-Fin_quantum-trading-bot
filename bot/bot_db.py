"""Extended DB schema for autonomous bot: decisions, performance, simulations."""

import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class BotDB:
    """SQLite storage for bot decisions, performance snapshots, and simulation runs."""

    def __init__(self, db_path: str = "data/terminal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_decisions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                amount_usd REAL,
                qty REAL,
                price REAL,
                ml_signal TEXT,
                ml_confidence REAL,
                monte_carlo_var REAL,
                monte_carlo_expected_return REAL,
                regime TEXT,
                rsi REAL,
                trend_strength REAL,
                reasoning TEXT,
                trade_id TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_performance (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_trade_pnl REAL,
                best_trade_pnl REAL,
                worst_trade_pnl REAL,
                portfolio_value REAL,
                benchmark_return REAL,
                total_invested REAL,
                total_returned REAL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                num_paths INTEGER,
                horizon_days INTEGER,
                mu REAL,
                sigma REAL,
                current_price REAL,
                expected_return REAL,
                var_95 REAL,
                var_99 REAL,
                cvar_95 REAL,
                prob_profit REAL,
                params TEXT
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_bot_decisions_ts ON bot_decisions(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bot_decisions_session ON bot_decisions(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bot_performance_session ON bot_performance(session_id)")
        conn.commit()
        conn.close()
        logger.info("Bot DB tables initialized")

    def log_decision(self, session_id: str, symbol: str, action: str,
                     amount_usd: float = 0, qty: float = 0, price: float = 0,
                     ml_signal: str = "", ml_confidence: float = 0,
                     mc_var: float = 0, mc_expected_return: float = 0,
                     regime: str = "", rsi: float = 0, trend: float = 0,
                     reasoning: str = "", trade_id: str = "") -> str:
        did = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO bot_decisions
            (id, session_id, timestamp, symbol, action, amount_usd, qty, price,
             ml_signal, ml_confidence, monte_carlo_var, monte_carlo_expected_return,
             regime, rsi, trend_strength, reasoning, trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (did, session_id, now, symbol, action, amount_usd, qty, price,
              ml_signal, ml_confidence, mc_var, mc_expected_return,
              regime, rsi, trend, reasoning, trade_id))
        conn.commit()
        conn.close()
        return did

    def save_performance(self, session_id: str, metrics: dict) -> str:
        pid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO bot_performance
            (id, session_id, timestamp, total_trades, winning_trades, losing_trades,
             win_rate, total_pnl, sharpe_ratio, max_drawdown, avg_trade_pnl,
             best_trade_pnl, worst_trade_pnl, portfolio_value, benchmark_return,
             total_invested, total_returned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (pid, session_id, now,
              metrics.get("total_trades", 0), metrics.get("winning_trades", 0),
              metrics.get("losing_trades", 0), metrics.get("win_rate", 0),
              metrics.get("total_pnl", 0), metrics.get("sharpe_ratio", 0),
              metrics.get("max_drawdown", 0), metrics.get("avg_trade_pnl", 0),
              metrics.get("best_trade_pnl", 0), metrics.get("worst_trade_pnl", 0),
              metrics.get("portfolio_value", 0), metrics.get("benchmark_return", 0),
              metrics.get("total_invested", 0), metrics.get("total_returned", 0)))
        conn.commit()
        conn.close()
        return pid

    def save_simulation(self, result: dict) -> str:
        sid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO simulations
            (id, symbol, timestamp, num_paths, horizon_days, mu, sigma,
             current_price, expected_return, var_95, var_99, cvar_95, prob_profit, params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid, result.get("symbol", ""), now,
              result.get("num_paths", 0), result.get("horizon_days", 0),
              result.get("mu", 0), result.get("sigma", 0),
              result.get("current_price", 0), result.get("expected_return", 0),
              result.get("var_95", 0), result.get("var_99", 0),
              result.get("cvar_95", 0), result.get("prob_profit", 0),
              json.dumps(result)))
        conn.commit()
        conn.close()
        return sid

    def get_decisions(self, session_id: str, limit: int = 100) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM bot_decisions WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_performance(self, session_id: str, limit: int = 50) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM bot_performance WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_latest_performance(self, session_id: str) -> Optional[dict]:
        rows = self.get_performance(session_id, limit=1)
        return rows[0] if rows else None
