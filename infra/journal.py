"""Event journal for logging trading events."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class EventType(str, Enum):
    """Event type enumeration."""

    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO_STATE = "PORTFOLIO_STATE"
    RISK_VIOLATION = "RISK_VIOLATION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


@dataclass
class JournalEvent:
    """Journal event record."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    data: dict[str, Any]


class Journal:
    """SQLite-based event journal."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "data/journal.db") -> None:
        """Initialize journal.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        # Schema version table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """
        )

        # Check current version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        current_version = result[0] if result[0] else 0

        if current_version < self.SCHEMA_VERSION:
            self._migrate(cursor, current_version, self.SCHEMA_VERSION)
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, datetime.now().isoformat()),
            )

        conn.commit()
        conn.close()

    def _migrate(self, cursor: sqlite3.Cursor, from_version: int, to_version: int) -> None:
        """Run schema migrations."""
        logger.info(f"Migrating schema from {from_version} to {to_version}")

        # Future migrations would go here
        # if from_version < 2:
        #     cursor.execute("ALTER TABLE events ADD COLUMN ...")

    def log_event(
        self,
        event_type: EventType,
        data: dict[str, Any],
        event_id: Optional[str] = None,
    ) -> str:
        """Log an event.

        Args:
            event_type: Type of event
            data: Event data (will be JSON serialized)
            event_id: Optional event ID (auto-generated if not provided)

        Returns:
            Event ID
        """
        import uuid

        if event_id is None:
            event_id = str(uuid.uuid4())

        # Convert Decimal to string for JSON serialization
        def decimal_serializer(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        serialized_data = json.dumps(data, default=decimal_serializer)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO events (event_id, event_type, timestamp, data, created_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                event_id,
                event_type.value,
                datetime.now().isoformat(),
                serialized_data,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

        return event_id

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[JournalEvent]:
        """Get events from journal.

        Args:
            event_type: Filter by event type
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of events

        Returns:
            List of journal events
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT event_id, event_type, timestamp, data FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            event_id, event_type_str, timestamp_str, data_str = row
            data = json.loads(data_str)
            events.append(
                JournalEvent(
                    event_id=event_id,
                    event_type=EventType(event_type_str),
                    timestamp=datetime.fromisoformat(timestamp_str),
                    data=data,
                )
            )

        conn.close()
        return events
