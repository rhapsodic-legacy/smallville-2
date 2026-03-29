"""
SQLite structured storage for NPC knowledge.

Stores hard facts as subject-predicate-object triples, relationship data,
events, and goals. Provides query interface for structured questions
like "who owes me gold?" or "who is allied with whom?"
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default DB path — can be overridden for testing
DEFAULT_DB_PATH = Path("data/memory.db")


@dataclass
class Fact:
    """A subject-predicate-object triple representing a known fact."""
    fact_id: int = 0
    npc_id: str = ""
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    confidence: float = 1.0
    source: str = ""          # "observation", "conversation", "reflection"
    created_at: float = 0.0   # game minutes
    updated_at: float = 0.0

    def to_natural(self) -> str:
        """Convert to natural language for LLM prompts."""
        return f"{self.subject} {self.predicate} {self.obj}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "npc_id": self.npc_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
        }


@dataclass
class GoalRecord:
    """A persisted goal for an NPC."""
    goal_id: int = 0
    npc_id: str = ""
    description: str = ""
    importance: float = 0.5
    status: str = "active"    # "active", "completed", "abandoned"
    created_at: float = 0.0
    deadline: float = 0.0     # 0 = no deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "npc_id": self.npc_id,
            "description": self.description,
            "importance": self.importance,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class EventRecord:
    """A recorded event with participants and outcome."""
    event_id: int = 0
    event_type: str = ""       # "conversation", "trade", "conflict", "observation"
    participants: str = ""     # comma-separated NPC IDs
    description: str = ""
    location_x: int = 0
    location_z: int = 0
    game_time: float = 0.0
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "participants": self.participants.split(",") if self.participants else [],
            "description": self.description,
            "location": {"x": self.location_x, "z": self.location_z},
            "game_time": self.game_time,
            "importance": self.importance,
        }


class StructuredMemory:
    """
    SQLite-backed structured storage for NPC knowledge.

    Stores facts (SPO triples), goals, and events.
    Thread-safe via check_same_thread=False — writes are serialised by SQLite.
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None

    def initialise(self) -> None:
        """Create database and tables."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("Structured memory initialised: %s", self.db_path)

    def _create_tables(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_id      TEXT NOT NULL,
                subject     TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                object      TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0,
                source      TEXT DEFAULT 'observation',
                created_at  REAL DEFAULT 0,
                updated_at  REAL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_npc
            ON facts (npc_id)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_subject
            ON facts (subject)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                goal_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_id      TEXT NOT NULL,
                description TEXT NOT NULL,
                importance  REAL DEFAULT 0.5,
                status      TEXT DEFAULT 'active',
                created_at  REAL DEFAULT 0,
                deadline    REAL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_npc
            ON goals (npc_id, status)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type   TEXT NOT NULL,
                participants TEXT DEFAULT '',
                description  TEXT NOT NULL,
                location_x   INTEGER DEFAULT 0,
                location_z   INTEGER DEFAULT 0,
                game_time    REAL DEFAULT 0,
                importance   REAL DEFAULT 0.5
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_time
            ON events (game_time)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events (event_type)
        """)

        c.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StructuredMemory not initialised — call initialise()")
        return self._conn

    # ---------- Facts CRUD ----------

    def add_fact(
        self,
        npc_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "observation",
        game_time: float = 0.0,
    ) -> int:
        """
        Store a fact. If an identical SPO triple already exists for the NPC,
        update its confidence and timestamp instead.
        """
        existing = self.conn.execute(
            "SELECT fact_id FROM facts "
            "WHERE npc_id=? AND subject=? AND predicate=? AND object=?",
            (npc_id, subject, predicate, obj),
        ).fetchone()

        if existing:
            self.conn.execute(
                "UPDATE facts SET confidence=?, updated_at=? WHERE fact_id=?",
                (confidence, game_time, existing["fact_id"]),
            )
            self.conn.commit()
            return existing["fact_id"]

        cursor = self.conn.execute(
            "INSERT INTO facts (npc_id, subject, predicate, object, confidence, "
            "source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (npc_id, subject, predicate, obj, confidence, source,
             game_time, game_time),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_facts(
        self,
        npc_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """Query facts known by an NPC, optionally filtered."""
        query = "SELECT * FROM facts WHERE npc_id=?"
        params: list[Any] = [npc_id]

        if subject:
            query += " AND subject=?"
            params.append(subject)
        if predicate:
            query += " AND predicate=?"
            params.append(predicate)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_facts_about(self, npc_id: str, about: str, limit: int = 20) -> list[Fact]:
        """Get all facts where the subject or object matches `about`."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE npc_id=? AND (subject=? OR object=?) "
            "ORDER BY updated_at DESC LIMIT ?",
            (npc_id, about, about, limit),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def remove_fact(self, fact_id: int) -> None:
        self.conn.execute("DELETE FROM facts WHERE fact_id=?", (fact_id,))
        self.conn.commit()

    # ---------- Goals CRUD ----------

    def add_goal(
        self,
        npc_id: str,
        description: str,
        importance: float = 0.5,
        game_time: float = 0.0,
        deadline: float = 0.0,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO goals (npc_id, description, importance, status, "
            "created_at, deadline) VALUES (?, ?, ?, 'active', ?, ?)",
            (npc_id, description, importance, game_time, deadline),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_active_goals(self, npc_id: str) -> list[GoalRecord]:
        rows = self.conn.execute(
            "SELECT * FROM goals WHERE npc_id=? AND status='active' "
            "ORDER BY importance DESC",
            (npc_id,),
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def update_goal_status(self, goal_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE goals SET status=? WHERE goal_id=?", (status, goal_id),
        )
        self.conn.commit()

    # ---------- Events CRUD ----------

    def record_event(
        self,
        event_type: str,
        description: str,
        participants: list[str] | None = None,
        location_x: int = 0,
        location_z: int = 0,
        game_time: float = 0.0,
        importance: float = 0.5,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO events (event_type, participants, description, "
            "location_x, location_z, game_time, importance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_type, ",".join(participants or []), description,
             location_x, location_z, game_time, importance),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_events(
        self,
        event_type: str | None = None,
        participant: str | None = None,
        since: float = 0.0,
        limit: int = 50,
    ) -> list[EventRecord]:
        """Query events, optionally by type, participant, and time range."""
        query = "SELECT * FROM events WHERE game_time >= ?"
        params: list[Any] = [since]

        if event_type:
            query += " AND event_type=?"
            params.append(event_type)
        if participant:
            query += " AND participants LIKE ?"
            params.append(f"%{participant}%")

        query += " ORDER BY game_time DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_recent_events(self, limit: int = 20) -> list[EventRecord]:
        """Get the most recent events regardless of type."""
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY game_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ---------- Stats (for UI inspector) ----------

    def get_stats(self) -> dict[str, Any]:
        """Summary stats for the memory inspector."""
        facts_count = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        goals_count = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE status='active'"
        ).fetchone()[0]
        events_count = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        npc_count = self.conn.execute(
            "SELECT COUNT(DISTINCT npc_id) FROM facts"
        ).fetchone()[0]

        return {
            "total_facts": facts_count,
            "active_goals": goals_count,
            "total_events": events_count,
            "npcs_with_facts": npc_count,
        }

    # ---------- Row mapping ----------

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            fact_id=row["fact_id"],
            npc_id=row["npc_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            obj=row["object"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> GoalRecord:
        return GoalRecord(
            goal_id=row["goal_id"],
            npc_id=row["npc_id"],
            description=row["description"],
            importance=row["importance"],
            status=row["status"],
            created_at=row["created_at"],
            deadline=row["deadline"],
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            event_type=row["event_type"],
            participants=row["participants"],
            description=row["description"],
            location_x=row["location_x"],
            location_z=row["location_z"],
            game_time=row["game_time"],
            importance=row["importance"],
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
