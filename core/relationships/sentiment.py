"""
Sentiment dimensions — per-pair relationship tracking.

Each NPC pair has directional sentiment across five dimensions:
trust, fear, respect, affection, debt. Storage is sparse — only
non-default (non-zero) relationships are persisted in SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default DB path — can be overridden for testing
DEFAULT_DB_PATH = Path("data/relationships.db")

# Dimension names used throughout the system
DIMENSIONS = ("trust", "fear", "respect", "affection", "debt", "resonance")

# Absolute caps to prevent runaway values
DIMENSION_MIN = -100.0
DIMENSION_MAX = 100.0

# Thresholds for qualitative labels in prompts
THRESHOLDS = {
    "strong_negative": -50,
    "negative": -20,
    "neutral_low": -5,
    "neutral_high": 5,
    "positive": 20,
    "strong_positive": 50,
}


@dataclass
class Sentiment:
    """Directional sentiment from one NPC towards another."""
    npc_from: str
    npc_to: str
    trust: float = 0.0
    fear: float = 0.0
    respect: float = 0.0
    affection: float = 0.0
    debt: float = 0.0
    resonance: float = 0.0  # kinship from shared interests/occupation
    updated_at: float = 0.0

    def get(self, dimension: str) -> float:
        """Get a dimension by name."""
        return getattr(self, dimension, 0.0)

    def set(self, dimension: str, value: float) -> None:
        """Set a dimension by name, clamping to bounds."""
        clamped = max(DIMENSION_MIN, min(DIMENSION_MAX, value))
        setattr(self, dimension, clamped)

    def is_default(self) -> bool:
        """True if all dimensions are zero (can be pruned)."""
        return all(abs(self.get(d)) < 0.01 for d in DIMENSIONS)

    def overall_disposition(self) -> float:
        """Single number summarising how this NPC feels about the other.
        Positive = friendly, negative = hostile."""
        return (
            self.trust * 0.25
            + self.respect * 0.20
            + self.affection * 0.30
            - self.fear * 0.15
            + self.debt * 0.10
        )

    def to_description(self) -> str:
        """Human-readable summary for LLM prompts."""
        parts = []
        for dim in DIMENSIONS:
            val = self.get(dim)
            label = _value_to_label(dim, val)
            if label:
                parts.append(label)
        if not parts:
            return "neutral acquaintance"
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.npc_from,
            "to": self.npc_to,
            "trust": self.trust,
            "fear": self.fear,
            "respect": self.respect,
            "affection": self.affection,
            "debt": self.debt,
            "disposition": round(self.overall_disposition(), 1),
            "description": self.to_description(),
            "updated_at": self.updated_at,
        }


def _value_to_label(dimension: str, value: float) -> str:
    """Convert a numeric sentiment value to a natural-language label."""
    if abs(value) < THRESHOLDS["neutral_high"]:
        return ""

    labels = {
        "trust": {True: "trusts them", False: "distrusts them"},
        "fear": {True: "fears them", False: "feels safe around them"},
        "respect": {True: "respects them", False: "looks down on them"},
        "affection": {True: "feels warmly towards them", False: "dislikes them"},
        "debt": {True: "owes them a favour", False: "feels owed by them"},
    }

    is_positive = value > 0
    base = labels.get(dimension, {True: f"high {dimension}", False: f"low {dimension}"})
    desc = base[is_positive]

    if abs(value) >= THRESHOLDS["strong_positive"]:
        desc = f"strongly {desc}"
    return desc


class SentimentTracker:
    """
    SQLite-backed sparse storage for NPC-pair sentiment.

    Only non-default relationships are stored. Directional — A's sentiment
    towards B is independent of B's sentiment towards A.
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
        logger.info("Sentiment tracker initialised: %s", self.db_path)

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiments (
                npc_from    TEXT NOT NULL,
                npc_to      TEXT NOT NULL,
                trust       REAL DEFAULT 0,
                fear        REAL DEFAULT 0,
                respect     REAL DEFAULT 0,
                affection   REAL DEFAULT 0,
                debt        REAL DEFAULT 0,
                resonance   REAL DEFAULT 0,
                updated_at  REAL DEFAULT 0,
                PRIMARY KEY (npc_from, npc_to)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_from
            ON sentiments (npc_from)
        """)
        self.conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SentimentTracker not initialised — call initialise()"
            )
        return self._conn

    def get(self, npc_from: str, npc_to: str) -> Sentiment:
        """Get sentiment from one NPC to another. Returns default if none stored."""
        row = self.conn.execute(
            "SELECT * FROM sentiments WHERE npc_from=? AND npc_to=?",
            (npc_from, npc_to),
        ).fetchone()

        if row is None:
            return Sentiment(npc_from=npc_from, npc_to=npc_to)
        return self._row_to_sentiment(row)

    def set(self, sentiment: Sentiment, game_time: float = 0.0) -> None:
        """Store or update a sentiment record. Prunes if all dimensions are default."""
        sentiment.updated_at = game_time

        if sentiment.is_default():
            self.conn.execute(
                "DELETE FROM sentiments WHERE npc_from=? AND npc_to=?",
                (sentiment.npc_from, sentiment.npc_to),
            )
            self.conn.commit()
            return

        self.conn.execute(
            "INSERT INTO sentiments (npc_from, npc_to, trust, fear, respect, "
            "affection, debt, resonance, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(npc_from, npc_to) DO UPDATE SET "
            "trust=excluded.trust, fear=excluded.fear, respect=excluded.respect, "
            "affection=excluded.affection, debt=excluded.debt, "
            "resonance=excluded.resonance, updated_at=excluded.updated_at",
            (sentiment.npc_from, sentiment.npc_to, sentiment.trust,
             sentiment.fear, sentiment.respect, sentiment.affection,
             sentiment.debt, sentiment.resonance, sentiment.updated_at),
        )
        self.conn.commit()

    def modify(
        self,
        npc_from: str,
        npc_to: str,
        dimension: str,
        delta: float,
        game_time: float = 0.0,
    ) -> Sentiment:
        """
        Modify a single dimension by delta. Returns the updated sentiment.

        This is the primary interface for the event impact system.
        """
        if dimension not in DIMENSIONS:
            logger.warning("Unknown sentiment dimension: %s", dimension)
            return self.get(npc_from, npc_to)

        sentiment = self.get(npc_from, npc_to)
        current = sentiment.get(dimension)
        sentiment.set(dimension, current + delta)
        self.set(sentiment, game_time)
        return sentiment

    def modify_mutual(
        self,
        npc_a: str,
        npc_b: str,
        dimension: str,
        delta: float,
        game_time: float = 0.0,
    ) -> tuple[Sentiment, Sentiment]:
        """Modify the same dimension for both directions of a pair."""
        sa = self.modify(npc_a, npc_b, dimension, delta, game_time)
        sb = self.modify(npc_b, npc_a, dimension, delta, game_time)
        return sa, sb

    def get_all_for(self, npc_id: str) -> list[Sentiment]:
        """Get all relationships where this NPC is the source."""
        rows = self.conn.execute(
            "SELECT * FROM sentiments WHERE npc_from=? ORDER BY updated_at DESC",
            (npc_id,),
        ).fetchall()
        return [self._row_to_sentiment(r) for r in rows]

    def get_all_towards(self, npc_id: str) -> list[Sentiment]:
        """Get all relationships where this NPC is the target."""
        rows = self.conn.execute(
            "SELECT * FROM sentiments WHERE npc_to=? ORDER BY updated_at DESC",
            (npc_id,),
        ).fetchall()
        return [self._row_to_sentiment(r) for r in rows]

    def get_strongest_relationships(
        self, npc_id: str, limit: int = 5,
    ) -> list[Sentiment]:
        """Get the NPC's most intense relationships (by absolute disposition)."""
        all_rels = self.get_all_for(npc_id)
        all_rels.sort(key=lambda s: abs(s.overall_disposition()), reverse=True)
        return all_rels[:limit]

    def decay_all(
        self,
        elapsed_game_minutes: float,
        rate_per_day: float = 0.02,
    ) -> int:
        """Drift all non-zero sentiment dimensions toward zero.

        Called once per game-day. Each dimension loses
        ``abs(value) * rate_per_day`` per day, so strong feelings
        decay faster in absolute terms but the *proportion* is constant.
        At 2 % per day a value of 50 takes ~35 days to halve — slow enough
        that active relationships easily outpace decay, but neglected ones
        fade over a few in-game weeks.

        Returns the number of relationships updated.
        """
        rows = self.conn.execute(
            "SELECT * FROM sentiments",
        ).fetchall()
        updated = 0
        for row in rows:
            s = self._row_to_sentiment(row)
            changed = False
            for dim in DIMENSIONS:
                val = s.get(dim)
                if abs(val) < 0.01:
                    continue
                # Shrink toward zero
                reduction = abs(val) * rate_per_day
                if val > 0:
                    s.set(dim, val - reduction)
                else:
                    s.set(dim, val + reduction)
                changed = True
            if changed:
                self.set(s, s.updated_at)
                updated += 1
        return updated

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for UI inspector."""
        count = self.conn.execute(
            "SELECT COUNT(*) FROM sentiments"
        ).fetchone()[0]
        npc_count = self.conn.execute(
            "SELECT COUNT(DISTINCT npc_from) FROM sentiments"
        ).fetchone()[0]
        return {
            "total_relationships": count,
            "npcs_with_relationships": npc_count,
        }

    @staticmethod
    def _row_to_sentiment(row: sqlite3.Row) -> Sentiment:
        return Sentiment(
            npc_from=row["npc_from"],
            npc_to=row["npc_to"],
            trust=row["trust"],
            fear=row["fear"],
            respect=row["respect"],
            affection=row["affection"],
            debt=row["debt"],
            resonance=row["resonance"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
