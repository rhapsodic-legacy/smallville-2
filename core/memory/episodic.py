"""
ChromaDB episodic memory storage.

Stores NPC observations and experiences as embeddings for semantic retrieval.
Retrieval scoring combines recency, importance, and relevance (cosine similarity)
following the Stanford Generative Agents approach.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """A single episodic memory entry."""
    memory_id: str = ""
    npc_id: str = ""
    description: str = ""
    category: str = ""         # "observation", "conversation", "reflection", "event"
    importance: float = 0.5    # 0.0–1.0 poignancy score
    game_time: float = 0.0     # game minutes when formed
    location_x: int = 0
    location_z: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "npc_id": self.npc_id,
            "description": self.description,
            "category": self.category,
            "importance": self.importance,
            "game_time": self.game_time,
            "location": {"x": self.location_x, "z": self.location_z},
        }


@dataclass
class RetrievalResult:
    """A memory with its composite retrieval score."""
    memory: EpisodicMemory
    relevance_score: float = 0.0   # cosine similarity
    recency_score: float = 0.0     # exponential decay
    importance_score: float = 0.0  # raw importance
    composite_score: float = 0.0   # weighted combination

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "scores": {
                "relevance": round(self.relevance_score, 3),
                "recency": round(self.recency_score, 3),
                "importance": round(self.importance_score, 3),
                "composite": round(self.composite_score, 3),
            },
        }


# Retrieval weights (Stanford paper defaults)
RECENCY_WEIGHT = 1.0
IMPORTANCE_WEIGHT = 1.0
RELEVANCE_WEIGHT = 1.0

# Decay factor for recency scoring (higher = faster decay)
RECENCY_DECAY = 0.995


class EpisodicStore:
    """
    ChromaDB-backed episodic memory for NPC experiences.

    Each memory is embedded and stored with metadata for composite retrieval.
    Falls back to in-memory storage if ChromaDB is unavailable.
    """

    def __init__(self, persist_directory: str | None = None):
        self._persist_dir = persist_directory
        self._client = None
        self._collection = None
        self._fallback_mode = False
        self._fallback_memories: dict[str, EpisodicMemory] = {}
        self._counter = 0

    def initialise(self) -> None:
        """Set up ChromaDB client and collection."""
        try:
            import chromadb
            if self._persist_dir:
                self._client = chromadb.PersistentClient(
                    path=self._persist_dir,
                )
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name="npc_episodic_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Episodic memory initialised (ChromaDB, %d existing memories)",
                self._collection.count(),
            )
        except Exception as e:
            logger.warning(
                "ChromaDB unavailable (%s) — using in-memory fallback", e,
            )
            self._fallback_mode = True

    def _next_id(self, npc_id: str) -> str:
        self._counter += 1
        return f"{npc_id}_mem_{self._counter}"

    # ---------- Storage ----------

    def add_memory(
        self,
        npc_id: str,
        description: str,
        category: str = "observation",
        importance: float = 0.5,
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new episodic memory. Returns the memory ID."""
        memory_id = self._next_id(npc_id)
        metadata = {
            "npc_id": npc_id,
            "category": category,
            "importance": importance,
            "game_time": game_time,
            "location_x": location_x,
            "location_z": location_z,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        if self._fallback_mode:
            self._fallback_memories[memory_id] = EpisodicMemory(
                memory_id=memory_id,
                npc_id=npc_id,
                description=description,
                category=category,
                importance=importance,
                game_time=game_time,
                location_x=location_x,
                location_z=location_z,
                metadata=metadata,
            )
        else:
            self._collection.add(
                ids=[memory_id],
                documents=[description],
                metadatas=[metadata],
            )

        return memory_id

    # ---------- Retrieval ----------

    def retrieve(
        self,
        npc_id: str,
        query: str,
        current_game_time: float = 0.0,
        limit: int = 10,
        category: str | None = None,
        recency_weight: float = RECENCY_WEIGHT,
        importance_weight: float = IMPORTANCE_WEIGHT,
        relevance_weight: float = RELEVANCE_WEIGHT,
    ) -> list[RetrievalResult]:
        """
        Retrieve memories by composite scoring.

        Score = recency_weight * recency + importance_weight * importance
                + relevance_weight * relevance

        Where recency uses exponential decay from the Stanford paper.
        """
        if self._fallback_mode:
            return self._fallback_retrieve(
                npc_id, query, current_game_time, limit, category,
                recency_weight, importance_weight, relevance_weight,
            )

        where_filter = {"npc_id": npc_id}
        if category:
            where_filter = {
                "$and": [
                    {"npc_id": npc_id},
                    {"category": category},
                ],
            }

        # Fetch more than needed — we'll re-rank with composite scoring
        fetch_limit = min(limit * 3, 100)

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=fetch_limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        # Build RetrievalResult list with composite scoring
        scored: list[RetrievalResult] = []
        for i, mem_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0.0

            memory = EpisodicMemory(
                memory_id=mem_id,
                npc_id=meta.get("npc_id", npc_id),
                description=doc,
                category=meta.get("category", ""),
                importance=meta.get("importance", 0.5),
                game_time=meta.get("game_time", 0.0),
                location_x=int(meta.get("location_x", 0)),
                location_z=int(meta.get("location_z", 0)),
            )

            relevance = max(0.0, 1.0 - distance)  # cosine distance → similarity
            recency = self._recency_score(memory.game_time, current_game_time)
            imp = memory.importance

            composite = (
                recency_weight * recency
                + importance_weight * imp
                + relevance_weight * relevance
            )

            scored.append(RetrievalResult(
                memory=memory,
                relevance_score=relevance,
                recency_score=recency,
                importance_score=imp,
                composite_score=composite,
            ))

        scored.sort(key=lambda r: r.composite_score, reverse=True)
        return scored[:limit]

    def get_recent(
        self,
        npc_id: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[EpisodicMemory]:
        """Get most recent memories by game time (no semantic query needed)."""
        if self._fallback_mode:
            mems = [
                m for m in self._fallback_memories.values()
                if m.npc_id == npc_id
                and (category is None or m.category == category)
            ]
            mems.sort(key=lambda m: m.game_time, reverse=True)
            return mems[:limit]

        where_filter = {"npc_id": npc_id}
        if category:
            where_filter = {
                "$and": [
                    {"npc_id": npc_id},
                    {"category": category},
                ],
            }

        try:
            results = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        if not results["ids"]:
            return []

        memories = []
        for i, mem_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            memories.append(EpisodicMemory(
                memory_id=mem_id,
                npc_id=meta.get("npc_id", npc_id),
                description=results["documents"][i],
                category=meta.get("category", ""),
                importance=meta.get("importance", 0.5),
                game_time=meta.get("game_time", 0.0),
                location_x=int(meta.get("location_x", 0)),
                location_z=int(meta.get("location_z", 0)),
            ))

        memories.sort(key=lambda m: m.game_time, reverse=True)
        return memories[:limit]

    def count(self, npc_id: str | None = None) -> int:
        """Count memories, optionally for a specific NPC."""
        if self._fallback_mode:
            if npc_id:
                return sum(
                    1 for m in self._fallback_memories.values()
                    if m.npc_id == npc_id
                )
            return len(self._fallback_memories)

        if npc_id:
            try:
                result = self._collection.get(
                    where={"npc_id": npc_id},
                    include=[],
                )
                return len(result["ids"])
            except Exception:
                return 0
        return self._collection.count()

    # ---------- Importance accumulator (for reflection triggers) ----------

    def importance_since(
        self,
        npc_id: str,
        since_game_time: float,
    ) -> float:
        """Sum importance of memories formed since a given time."""
        if self._fallback_mode:
            return sum(
                m.importance for m in self._fallback_memories.values()
                if m.npc_id == npc_id and m.game_time >= since_game_time
            )

        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"npc_id": npc_id},
                        {"game_time": {"$gte": since_game_time}},
                    ],
                },
                include=["metadatas"],
            )
            return sum(
                m.get("importance", 0.0) for m in results.get("metadatas", [])
            )
        except Exception:
            return 0.0

    # ---------- Stats (for UI inspector) ----------

    def get_stats(self) -> dict[str, Any]:
        if self._fallback_mode:
            total = len(self._fallback_memories)
            by_category: dict[str, int] = {}
            for m in self._fallback_memories.values():
                by_category[m.category] = by_category.get(m.category, 0) + 1
            return {
                "total_memories": total,
                "by_category": by_category,
                "backend": "in-memory fallback",
            }

        total = self._collection.count()
        return {
            "total_memories": total,
            "backend": "chromadb",
        }

    # ---------- Internals ----------

    @staticmethod
    def _recency_score(memory_time: float, current_time: float) -> float:
        """Exponential decay based on game-time difference."""
        if current_time <= memory_time:
            return 1.0
        hours_elapsed = (current_time - memory_time) / 60.0
        return math.pow(RECENCY_DECAY, hours_elapsed)

    def _fallback_retrieve(
        self,
        npc_id: str,
        query: str,
        current_game_time: float,
        limit: int,
        category: str | None,
        recency_weight: float,
        importance_weight: float,
        relevance_weight: float,
    ) -> list[RetrievalResult]:
        """Simple keyword-based retrieval for fallback mode."""
        query_words = set(query.lower().split())
        candidates = [
            m for m in self._fallback_memories.values()
            if m.npc_id == npc_id
            and (category is None or m.category == category)
        ]

        scored: list[RetrievalResult] = []
        for mem in candidates:
            mem_words = set(mem.description.lower().split())
            overlap = len(query_words & mem_words)
            relevance = overlap / max(len(query_words), 1)
            recency = self._recency_score(mem.game_time, current_game_time)
            imp = mem.importance

            composite = (
                recency_weight * recency
                + importance_weight * imp
                + relevance_weight * relevance
            )

            scored.append(RetrievalResult(
                memory=mem,
                relevance_score=relevance,
                recency_score=recency,
                importance_score=imp,
                composite_score=composite,
            ))

        scored.sort(key=lambda r: r.composite_score, reverse=True)
        return scored[:limit]
