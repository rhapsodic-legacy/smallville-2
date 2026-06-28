"""
Episodic memory storage — simple in-memory text store.

NPC observations and experiences are plain text. They are stored in a
dict keyed by memory id, with a per-NPC tag index, and read back by
recency / keyword / tag. No vector database, no embeddings, no
compaction-tombstone hiding.

(ChromaDB was removed 2026-06 after a silent, undiagnosable at-scale
retrieval failure dropped a third of the town's memories. For storing
and reading plain text it was the wrong tool: reading a list cannot
silently lose a third of the NPCs. Semantic retrieval can be re-added
deliberately if a real need for it ever appears.)
"""

from __future__ import annotations

import logging
import math
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
    # Phase K tags — surgical pointers into a per-NPC tag index so
    # specific details (the accusation about bread, the commitment
    # to help Dara) stay findable by tag.
    tags: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "npc_id": self.npc_id,
            "description": self.description,
            "category": self.category,
            "importance": self.importance,
            "game_time": self.game_time,
            "location": {"x": self.location_x, "z": self.location_z},
            "tags": sorted(self.tags),
        }


# Valid tag characters. Tags are short, lowercase, alnum + limited
# punctuation so a downstream search-index or grep works cleanly.
_TAG_PATTERN = None  # initialised lazily to avoid import-time re.compile cost


def normalise_tag(raw: str) -> str:
    """Canonicalise a tag string.

    Lowercases, strips, replaces whitespace with underscores, and
    strips characters outside `[a-z0-9_:-]`. Returns the empty string
    when the input cleans to nothing so callers can filter.
    """
    global _TAG_PATTERN
    if _TAG_PATTERN is None:
        import re
        _TAG_PATTERN = re.compile(r"[^a-z0-9_:-]+")
    if not raw:
        return ""
    stripped = raw.strip().lower()
    stripped = stripped.replace(" ", "_")
    cleaned = _TAG_PATTERN.sub("", stripped)
    return cleaned


def normalise_tags(tags: Any) -> set[str]:
    """Accept a set/list/tuple/str of tag candidates; return canonical set."""
    if not tags:
        return set()
    if isinstance(tags, str):
        raw = tags.split()
    else:
        raw = list(tags)
    return {t for t in (normalise_tag(x) for x in raw) if t}


# Internal delimiter for serialising tags into the scalar `tags`
# metadata field. Space is safe because normalise_tag strips whitespace.
_TAGS_METADATA_DELIM = " "


@dataclass
class RetrievalResult:
    """A memory with its composite retrieval score."""
    memory: EpisodicMemory
    relevance_score: float = 0.0   # keyword overlap
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
    """In-memory episodic memory for NPC experiences.

    Plain-text memories in a dict keyed by memory id, plus a per-NPC
    tag index for tag lookups. Reads return everything stored for an
    NPC (filtered only by recency / category / tag the caller asks
    for) — nothing is hidden. Reading is a list comprehension; it
    cannot silently lose memories.

    `persist_directory` / `fallback_only` are accepted for backward
    compatibility and ignored — there is only the one simple store now.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        *,
        fallback_only: bool = False,
    ):
        self._persist_dir = persist_directory
        self._memories: dict[str, EpisodicMemory] = {}
        self._counter = 0
        # Per-NPC, per-tag -> set of memory_ids, for O(1) tag lookup.
        self._tag_index: dict[str, dict[str, set[str]]] = {}

    def initialise(self) -> None:
        logger.info("Episodic memory initialised (in-memory text store)")

    def _next_id(self, npc_id: str) -> str:
        self._counter += 1
        return f"{npc_id}_mem_{self._counter}"

    def _parse_tags_from_metadata(self, raw: Any) -> set[str]:
        """Turn the scalar-encoded `tags` metadata field back into a set."""
        if not raw:
            return set()
        if isinstance(raw, (list, tuple, set)):
            return {str(t) for t in raw if t}
        return {t for t in str(raw).split(_TAGS_METADATA_DELIM) if t}

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
        tags: Any = None,
    ) -> str:
        """Store a new episodic memory. Returns the memory id."""
        memory_id = self._next_id(npc_id)
        tag_set = normalise_tags(tags)
        metadata = {
            "npc_id": npc_id,
            "category": category,
            "importance": importance,
            "game_time": game_time,
            "location_x": location_x,
            "location_z": location_z,
            "tags": _TAGS_METADATA_DELIM.join(sorted(tag_set)),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        self._memories[memory_id] = EpisodicMemory(
            memory_id=memory_id,
            npc_id=npc_id,
            description=description,
            category=category,
            importance=importance,
            game_time=game_time,
            location_x=location_x,
            location_z=location_z,
            metadata=metadata,
            tags=set(tag_set),
        )
        if tag_set:
            bucket = self._tag_index.setdefault(npc_id, {})
            for tag in tag_set:
                bucket.setdefault(tag, set()).add(memory_id)
        return memory_id

    def _for_npc(self, npc_id: str) -> list[EpisodicMemory]:
        return [m for m in self._memories.values() if m.npc_id == npc_id]

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
        include_compacted: bool = False,
    ) -> list[RetrievalResult]:
        """Rank an NPC's memories by recency + importance + keyword
        overlap with the query. Enough to feed recent, loosely-relevant
        context back into a prompt — no embeddings."""
        query_words = set(query.lower().split())
        scored: list[RetrievalResult] = []
        for mem in self._for_npc(npc_id):
            if category is not None and mem.category != category:
                continue
            mem_words = set(mem.description.lower().split())
            overlap = len(query_words & mem_words)
            relevance = overlap / max(len(query_words), 1)
            recency = self._recency_score(mem.game_time, current_game_time)
            composite = (
                recency_weight * recency
                + importance_weight * mem.importance
                + relevance_weight * relevance
            )
            scored.append(RetrievalResult(
                memory=mem,
                relevance_score=relevance,
                recency_score=recency,
                importance_score=mem.importance,
                composite_score=composite,
            ))
        scored.sort(key=lambda r: r.composite_score, reverse=True)
        return scored[:limit]

    def get_recent(
        self,
        npc_id: str,
        limit: int = 10,
        category: str | None = None,
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        """Most recent memories by game time (newest first)."""
        mems = [
            m for m in self._for_npc(npc_id)
            if category is None or m.category == category
        ]
        mems.sort(key=lambda m: m.game_time, reverse=True)
        return mems[:limit]

    def get_by_id(self, memory_id: str) -> EpisodicMemory | None:
        return self._memories.get(memory_id)

    def get_raw_by_id(self, memory_id: str) -> EpisodicMemory | None:
        return self.get_by_id(memory_id)

    def get_compacted_sources(
        self, memory_id: str,
    ) -> list[EpisodicMemory]:
        """Walk from a summary memory to the originals it absorbed via
        the `compacted_from` metadata pointer. Returns [] if absent."""
        summary = self.get_by_id(memory_id)
        if summary is None:
            return []
        raw = (summary.metadata or {}).get("compacted_from", "")
        if not raw:
            return []
        ids = raw.split(_TAGS_METADATA_DELIM) if isinstance(raw, str) else raw
        sources: list[EpisodicMemory] = []
        for sid in ids:
            if not sid:
                continue
            mem = self.get_by_id(sid)
            if mem is not None:
                sources.append(mem)
        return sources

    def retrieve_by_tags(
        self,
        npc_id: str,
        tags: Any,
        limit: int = 10,
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        """Return memories for this NPC carrying ANY of the given tags,
        newest first."""
        tag_set = normalise_tags(tags)
        if not tag_set:
            return []
        bucket = self._tag_index.get(npc_id, {})
        hits: set[str] = set()
        for tag in tag_set:
            hits.update(bucket.get(tag, ()))
        mems = [m for m in (self.get_by_id(mid) for mid in hits) if m]
        mems.sort(key=lambda m: m.game_time, reverse=True)
        return mems[:limit]

    # ---------- Metadata patch / delete ----------

    def update_metadata(
        self, memory_id: str, updates: dict[str, Any],
    ) -> bool:
        """Patch a stored memory's metadata (e.g. flip `unresolved` to
        False once a matter has been aired). Returns True on success."""
        if not updates:
            return False
        mem = self._memories.get(memory_id)
        if mem is None:
            return False
        mem.metadata.update(updates)
        if "tags" in updates:
            new_tags = self._parse_tags_from_metadata(updates["tags"])
            self._reindex_after_tag_change(
                mem.npc_id, memory_id, old_tags=mem.tags, new_tags=new_tags,
            )
            mem.tags = new_tags
        return True

    def _reindex_after_tag_change(
        self,
        npc_id: str,
        memory_id: str,
        *,
        old_tags: set[str],
        new_tags: set[str],
    ) -> None:
        """Patch the per-NPC tag index after a tag set changes."""
        if not npc_id:
            return
        bucket = self._tag_index.setdefault(npc_id, {})
        for tag in old_tags - new_tags:
            entry = bucket.get(tag)
            if entry is not None:
                entry.discard(memory_id)
                if not entry:
                    bucket.pop(tag, None)
        for tag in new_tags - old_tags:
            bucket.setdefault(tag, set()).add(memory_id)

    def delete_by_metadata(self, key: str, value: Any) -> int:
        """Delete every memory whose metadata[key] == value. Used by
        conversation consolidation to sweep per-turn entries once the
        summary is written. Returns the number removed."""
        to_remove = [
            (mid, mem) for mid, mem in self._memories.items()
            if mem.metadata.get(key) == value
        ]
        for mid, mem in to_remove:
            self._remove_from_tag_index(mem.npc_id, mid, mem.tags)
            del self._memories[mid]
        return len(to_remove)

    def _remove_from_tag_index(
        self, npc_id: str, memory_id: str, tags: set[str],
    ) -> None:
        """Drop a memory from every tag bucket it sits in."""
        if not npc_id or not tags:
            return
        bucket = self._tag_index.get(npc_id)
        if not bucket:
            return
        for tag in tags:
            entry = bucket.get(tag)
            if entry is not None:
                entry.discard(memory_id)
                if not entry:
                    bucket.pop(tag, None)
        if not bucket:
            self._tag_index.pop(npc_id, None)

    # ---------- Counts / windows / stats ----------

    def count(self, npc_id: str | None = None) -> int:
        """Count memories, optionally for a specific NPC."""
        if npc_id:
            return sum(1 for m in self._memories.values() if m.npc_id == npc_id)
        return len(self._memories)

    def get_memories_in_window(
        self,
        npc_id: str,
        start_game_time: float,
        end_game_time: float,
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        """Every memory whose `game_time` lies in [start, end), oldest
        first. Used by day summarisation to read a day's bucket."""
        mems = [
            m for m in self._for_npc(npc_id)
            if start_game_time <= m.game_time < end_game_time
        ]
        mems.sort(key=lambda m: m.game_time)
        return mems

    def importance_since(
        self, npc_id: str, since_game_time: float,
    ) -> float:
        """Sum importance of memories formed since a given time."""
        return sum(
            m.importance for m in self._memories.values()
            if m.npc_id == npc_id and m.game_time >= since_game_time
        )

    def get_stats(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        for m in self._memories.values():
            by_category[m.category] = by_category.get(m.category, 0) + 1
        return {
            "total_memories": len(self._memories),
            "by_category": by_category,
            "backend": "in-memory text store",
        }

    # ---------- Internals ----------

    @staticmethod
    def _recency_score(memory_time: float, current_time: float) -> float:
        """Exponential decay based on game-time difference."""
        if current_time <= memory_time:
            return 1.0
        hours_elapsed = (current_time - memory_time) / 60.0
        return math.pow(RECENCY_DECAY, hours_elapsed)
