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
    # Phase K tags — surgical pointers into a per-NPC tag index so
    # specific details (the accusation about bread, the commitment
    # to help Dara) remain findable even after compaction collapses
    # raw turn memories.
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


# Internal delimiter for serialising tags into ChromaDB metadata
# (which only supports scalar values). Space is safe because
# normalise_tag strips whitespace from individual tags.
_TAGS_METADATA_DELIM = " "


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

    def __init__(
        self,
        persist_directory: str | None = None,
        *,
        fallback_only: bool = False,
    ):
        self._persist_dir = persist_directory
        self._client = None
        self._collection = None
        self._fallback_mode = fallback_only
        self._fallback_memories: dict[str, EpisodicMemory] = {}
        self._counter = 0

        # Phase K tag index: per-NPC, per-tag → set of memory_ids.
        # Rebuilt lazily by `initialise` when running against a
        # persistent ChromaDB that already has memories; otherwise
        # updated incrementally by `add_memory`. Lookup is O(1) on
        # the tag level.
        self._tag_index: dict[str, dict[str, set[str]]] = {}

    def initialise(self) -> None:
        """Set up ChromaDB client and collection."""
        if self._fallback_mode:
            logger.info("Episodic memory initialised (in-memory fallback, ChromaDB skipped)")
            return
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
            # Phase K — rebuild the in-memory tag index from persisted
            # metadata. For a session-scoped chroma client this is a
            # no-op; for a PersistentClient it restores the index so
            # tag retrieval works immediately.
            self._rebuild_tag_index_from_collection()
        except Exception as e:
            logger.warning(
                "ChromaDB unavailable (%s) — using in-memory fallback", e,
            )
            self._fallback_mode = True

    def _rebuild_tag_index_from_collection(self) -> None:
        """Populate `_tag_index` from whatever's already in ChromaDB.

        Cheap on a fresh in-memory store (the collection is empty).
        Runs once at startup; thereafter `add_memory` keeps the
        index in sync. `delete_by_metadata` and `update_metadata`
        also maintain it.
        """
        if self._fallback_mode:
            return
        try:
            results = self._collection.get(include=["metadatas"])
        except Exception as e:
            logger.warning("Tag index rebuild failed: %s", e)
            return
        ids = results.get("ids") or []
        if not ids:
            return
        metas = results.get("metadatas") or []
        for mid, meta in zip(ids, metas):
            if not meta:
                continue
            npc_id = meta.get("npc_id", "")
            if not npc_id:
                continue
            tag_set = self._parse_tags_from_metadata(meta.get("tags"))
            if not tag_set:
                continue
            npc_bucket = self._tag_index.setdefault(npc_id, {})
            for tag in tag_set:
                npc_bucket.setdefault(tag, set()).add(mid)

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
        tags: Any = None,
    ) -> str:
        """Store a new episodic memory. Returns the memory ID.

        `tags` (Phase K) can be a set / list / tuple / space-
        separated string of tag candidates. Normalisation is applied
        via `normalise_tags`. Tags land in ChromaDB metadata as a
        single space-separated string (ChromaDB metadata is scalar-
        only) and are mirrored in the in-memory tag index for O(1)
        lookup by tag.
        """
        memory_id = self._next_id(npc_id)
        tag_set = normalise_tags(tags)
        tags_metadata = _TAGS_METADATA_DELIM.join(sorted(tag_set))
        metadata = {
            "npc_id": npc_id,
            "category": category,
            "importance": importance,
            "game_time": game_time,
            "location_x": location_x,
            "location_z": location_z,
            "tags": tags_metadata,
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
                tags=set(tag_set),
            )
        else:
            self._collection.add(
                ids=[memory_id],
                documents=[description],
                metadatas=[metadata],
            )

        # Mirror into the tag index so retrieve_by_tags is cheap.
        if tag_set:
            npc_bucket = self._tag_index.setdefault(npc_id, {})
            for tag in tag_set:
                npc_bucket.setdefault(tag, set()).add(memory_id)

        return memory_id

    def _parse_tags_from_metadata(self, raw: Any) -> set[str]:
        """Turn the scalar-encoded `tags` metadata field back into a set."""
        if not raw:
            return set()
        if isinstance(raw, (list, tuple, set)):
            return {str(t) for t in raw if t}
        return {t for t in str(raw).split(_TAGS_METADATA_DELIM) if t}

    # Phase H.3 — the tombstone marker. `compact_day` patches it onto
    # every raw memory it compacts. Default retrieval paths filter
    # these out so a day's raw observations stop surfacing once the
    # day_summary exists. Passing `include_compacted=True` lifts the
    # filter — diagnostics (memory panel, provenance inspector) use
    # that. When H.4 lands, week rollup will patch `compacted_into`
    # onto the day_summaries it rolls up, and this single filter
    # will demote them automatically in favour of the week_summary.
    @staticmethod
    def _is_tombstoned(meta: Any) -> bool:
        if not meta:
            return False
        return bool(meta.get("compacted_into"))

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
        """
        Retrieve memories by composite scoring.

        Score = recency_weight * recency + importance_weight * importance
                + relevance_weight * relevance

        Where recency uses exponential decay from the Stanford paper.
        Phase H.3: tombstoned memories (those carrying a
        `compacted_into` pointer because a day_summary absorbed them)
        are hidden by default. Pass `include_compacted=True` to see
        them — used by diagnostics / provenance inspection.
        """
        if self._fallback_mode:
            return self._fallback_retrieve(
                npc_id, query, current_game_time, limit, category,
                recency_weight, importance_weight, relevance_weight,
                include_compacted=include_compacted,
            )

        where_filter = {"npc_id": npc_id}
        if category:
            where_filter = {
                "$and": [
                    {"npc_id": npc_id},
                    {"category": category},
                ],
            }

        # Fetch more than needed — we'll re-rank with composite
        # scoring and tombstone filtering may drop some candidates.
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
            if not include_compacted and self._is_tombstoned(meta):
                continue
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
                tags=self._parse_tags_from_metadata(meta.get("tags")),
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
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        """Get most recent memories by game time (no semantic query needed).

        Phase H.3: tombstoned memories are hidden by default — pass
        `include_compacted=True` to see provenance.
        """
        if self._fallback_mode:
            mems = [
                m for m in self._fallback_memories.values()
                if m.npc_id == npc_id
                and (category is None or m.category == category)
                and (include_compacted or not self._is_tombstoned(m.metadata))
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
            meta = dict(results["metadatas"][i] or {})
            if not include_compacted and self._is_tombstoned(meta):
                continue
            memories.append(EpisodicMemory(
                memory_id=mem_id,
                npc_id=meta.get("npc_id", npc_id),
                description=results["documents"][i],
                category=meta.get("category", ""),
                importance=meta.get("importance", 0.5),
                game_time=meta.get("game_time", 0.0),
                location_x=int(meta.get("location_x", 0)),
                location_z=int(meta.get("location_z", 0)),
                metadata=meta,
                tags=self._parse_tags_from_metadata(meta.get("tags")),
            ))

        memories.sort(key=lambda m: m.game_time, reverse=True)
        return memories[:limit]

    def get_raw_by_id(self, memory_id: str) -> EpisodicMemory | None:
        """H.5 alias — fetch a memory regardless of tombstone status.

        Semantically identical to `get_by_id`; exists so diagnostic
        callers (memory panel, provenance inspector) signal intent.
        Default retrieval paths hide compacted originals; this one
        never does.
        """
        return self.get_by_id(memory_id)

    def get_compacted_sources(
        self, memory_id: str,
    ) -> list[EpisodicMemory]:
        """Walk from a summary memory to the originals it absorbed.

        Reads `compacted_from` (space-delimited id list) off the
        summary's metadata and resolves each. Missing ids silently
        drop out — that's expected for tests that stub raw ids. A
        non-summary memory (or one without `compacted_from`) returns
        an empty list.
        """
        summary = self.get_by_id(memory_id)
        if summary is None:
            return []
        raw = (summary.metadata or {}).get("compacted_from", "")
        if not raw:
            return []
        # `compacted_from` uses the same space-delimited encoding as
        # tag metadata — `_TAGS_METADATA_DELIM` is " ".
        ids = raw.split(_TAGS_METADATA_DELIM) if isinstance(raw, str) else raw
        sources: list[EpisodicMemory] = []
        for sid in ids:
            if not sid:
                continue
            mem = self.get_by_id(sid)
            if mem is not None:
                sources.append(mem)
        return sources

    def get_by_id(self, memory_id: str) -> EpisodicMemory | None:
        """Fetch a single memory by id. Used by Phase K tag retrieval
        to turn index hits into full memory objects.
        """
        if self._fallback_mode:
            return self._fallback_memories.get(memory_id)
        try:
            results = self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("get_by_id failed for %s: %s", memory_id, e)
            return None
        ids = results.get("ids") or []
        if not ids:
            return None
        meta = dict(results["metadatas"][0] or {})
        doc = results["documents"][0] if results.get("documents") else ""
        return EpisodicMemory(
            memory_id=ids[0],
            npc_id=meta.get("npc_id", ""),
            description=doc,
            category=meta.get("category", ""),
            importance=meta.get("importance", 0.5),
            game_time=meta.get("game_time", 0.0),
            location_x=int(meta.get("location_x", 0)),
            location_z=int(meta.get("location_z", 0)),
            metadata=meta,
            tags=self._parse_tags_from_metadata(meta.get("tags")),
        )

    def retrieve_by_tags(
        self,
        npc_id: str,
        tags: Any,
        limit: int = 10,
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        """Phase K.3 — return memories for this NPC carrying ANY of
        the given tags, newest first.

        Lookup is O(t + k) where t is the number of tags probed and
        k is the number of matches. Tags are normalised first so
        callers can pass raw candidate strings without fear.
        Phase H.3: tombstoned memories (none expected today — tagged
        memories bypass compaction — but the filter keeps this path
        consistent with `retrieve` if rollup semantics ever change).
        """
        tag_set = normalise_tags(tags)
        if not tag_set:
            return []
        # Index path (fast) — always correct when maintained by
        # add_memory. Rebuilt on `initialise` for chroma-backed stores.
        bucket = self._tag_index.get(npc_id, {})
        hits: set[str] = set()
        for tag in tag_set:
            hits.update(bucket.get(tag, ()))
        if not hits:
            return []
        mems: list[EpisodicMemory] = []
        for mid in hits:
            m = self.get_by_id(mid)
            if m is None:
                continue
            if not include_compacted and self._is_tombstoned(m.metadata):
                continue
            mems.append(m)
        mems.sort(key=lambda m: m.game_time, reverse=True)
        return mems[:limit]

    def update_metadata(
        self, memory_id: str, updates: dict[str, Any],
    ) -> bool:
        """Patch a stored memory's metadata.

        Used by Phase C to flip `unresolved: False` on outcome
        memories once the relevant matter has been aired in a
        subsequent conversation. Returns True on success.
        """
        if not updates:
            return False

        if self._fallback_mode:
            mem = self._fallback_memories.get(memory_id)
            if mem is None:
                return False
            mem.metadata.update(updates)
            # Mirror a tag change into the fallback dataclass + index.
            if "tags" in updates:
                self._reindex_after_tag_change(
                    mem.npc_id, memory_id,
                    old_tags=mem.tags,
                    new_tags=self._parse_tags_from_metadata(updates["tags"]),
                )
                mem.tags = self._parse_tags_from_metadata(updates["tags"])
            return True

        try:
            existing = self._collection.get(
                ids=[memory_id], include=["metadatas"],
            )
        except Exception as e:
            logger.warning("update_metadata lookup failed: %s", e)
            return False
        if not existing.get("ids"):
            return False

        old_meta = dict(existing["metadatas"][0] or {})
        merged = dict(old_meta)
        merged.update(updates)
        try:
            self._collection.update(ids=[memory_id], metadatas=[merged])
        except Exception as e:
            logger.warning("update_metadata write failed: %s", e)
            return False
        # Keep the tag index in sync when a metadata patch touches tags.
        if "tags" in updates:
            self._reindex_after_tag_change(
                old_meta.get("npc_id", ""),
                memory_id,
                old_tags=self._parse_tags_from_metadata(old_meta.get("tags")),
                new_tags=self._parse_tags_from_metadata(updates["tags"]),
            )
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
        """Delete every memory whose metadata[key] == value.

        Used by conversation consolidation to sweep per-turn entries
        once the final summary memory has been written. Returns the
        number of memories removed.
        """
        if self._fallback_mode:
            to_remove = [
                (mid, mem) for mid, mem in self._fallback_memories.items()
                if mem.metadata.get(key) == value
            ]
            for mid, mem in to_remove:
                self._remove_from_tag_index(mem.npc_id, mid, mem.tags)
                del self._fallback_memories[mid]
            return len(to_remove)

        try:
            results = self._collection.get(
                where={key: value},
                include=["metadatas"],
            )
        except Exception as e:
            logger.warning("ChromaDB delete_by_metadata lookup failed: %s", e)
            return 0

        ids = results.get("ids") or []
        if not ids:
            return 0
        metas = results.get("metadatas") or []
        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            logger.warning("ChromaDB delete_by_metadata failed: %s", e)
            return 0
        for mid, meta in zip(ids, metas):
            if not meta:
                continue
            self._remove_from_tag_index(
                meta.get("npc_id", ""),
                mid,
                self._parse_tags_from_metadata(meta.get("tags")),
            )
        return len(ids)

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

    # ---------- Windowed fetch (for compaction) ----------

    def get_memories_in_window(
        self,
        npc_id: str,
        start_game_time: float,
        end_game_time: float,
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        """Return every memory whose `game_time` lies in [start, end).

        Used by Phase H compaction to pull a full day's worth of
        memories so the summariser can see both the untagged chatter
        it will collapse and the tagged matters it must leave alone.
        Inclusive of start, exclusive of end — so sequential day
        windows never double-count a boundary memory.
        Phase H.3: tombstoned memories (already compacted into a
        day_summary) are hidden by default; re-running compaction
        on an already-compacted day therefore returns no compactable
        candidates and is a cheap no-op. Pass `include_compacted=
        True` from diagnostics to see provenance.
        """
        if self._fallback_mode:
            mems = [
                m for m in self._fallback_memories.values()
                if m.npc_id == npc_id
                and start_game_time <= m.game_time < end_game_time
                and (include_compacted or not self._is_tombstoned(m.metadata))
            ]
            mems.sort(key=lambda m: m.game_time)
            return mems

        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"npc_id": npc_id},
                        {"game_time": {"$gte": start_game_time}},
                        {"game_time": {"$lt": end_game_time}},
                    ],
                },
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("get_memories_in_window failed: %s", e)
            return []

        ids = results.get("ids") or []
        if not ids:
            return []

        memories: list[EpisodicMemory] = []
        metas = results.get("metadatas") or []
        docs = results.get("documents") or []
        for i, mid in enumerate(ids):
            meta = dict(metas[i] or {})
            if not include_compacted and self._is_tombstoned(meta):
                continue
            doc = docs[i] if i < len(docs) else ""
            memories.append(EpisodicMemory(
                memory_id=mid,
                npc_id=meta.get("npc_id", npc_id),
                description=doc,
                category=meta.get("category", ""),
                importance=meta.get("importance", 0.5),
                game_time=meta.get("game_time", 0.0),
                location_x=int(meta.get("location_x", 0)),
                location_z=int(meta.get("location_z", 0)),
                metadata=meta,
                tags=self._parse_tags_from_metadata(meta.get("tags")),
            ))
        memories.sort(key=lambda m: m.game_time)
        return memories

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
        include_compacted: bool = False,
    ) -> list[RetrievalResult]:
        """Simple keyword-based retrieval for fallback mode."""
        query_words = set(query.lower().split())
        candidates = [
            m for m in self._fallback_memories.values()
            if m.npc_id == npc_id
            and (category is None or m.category == category)
            and (include_compacted or not self._is_tombstoned(m.metadata))
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
