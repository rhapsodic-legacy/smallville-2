"""
Memory manager — unified interface for all NPC memory operations.

Combines structured storage (SQLite), episodic memory (ChromaDB),
and spatial memory into a single API. Handles the memory formation
pipeline: observe → score importance → store. Provides tier-aware
retrieval for the cognition system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from core.memory.structured import StructuredMemory, Fact, EventRecord
from core.memory.episodic import (
    EpisodicStore, EpisodicMemory, RetrievalResult,
)
from core.memory.spatial import SpatialMemory

if TYPE_CHECKING:
    from core.npc.llm_client import LLMProvider
    from core.relationships.sentiment import SentimentTracker
    from core.relationships.structures import FactionManager

logger = logging.getLogger(__name__)

# Importance threshold that triggers a reflection
REFLECTION_IMPORTANCE_THRESHOLD = 150.0

# How many memories to include in context for different tiers
TIER_CONTEXT_LIMITS = {
    1: 10,  # Full LLM — rich memory context
    2: 5,   # Simplified — moderate context
    3: 2,   # State machine — minimal structured facts only
    4: 0,   # Frozen — no retrieval
}


@dataclass
class MemoryContext:
    """Package of retrieved memories for an NPC's cognition cycle."""
    episodic: list[RetrievalResult]
    facts: list[Fact]
    spatial_summary: str

    def to_prompt_text(self) -> str:
        """Format memories for inclusion in an LLM prompt."""
        parts = []

        if self.facts:
            fact_lines = [f.to_natural() for f in self.facts[:10]]
            parts.append("Known facts:\n" + "\n".join(f"- {fl}" for fl in fact_lines))

        if self.episodic:
            mem_lines = [r.memory.description for r in self.episodic]
            parts.append(
                "Relevant memories:\n" + "\n".join(f"- {ml}" for ml in mem_lines)
            )

        if self.spatial_summary:
            parts.append(self.spatial_summary)

        return "\n\n".join(parts) if parts else "No relevant memories."

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodic": [r.to_dict() for r in self.episodic],
            "facts": [f.to_dict() for f in self.facts],
            "spatial_summary": self.spatial_summary,
        }


class MemoryManager:
    """
    Unified memory interface for all NPC memory operations.

    Owns the three memory subsystems and provides:
    - Memory formation (observe/score/store)
    - Unified retrieval (combines all stores)
    - Importance tracking for reflection triggers
    - Conversation recording
    """

    def __init__(
        self,
        structured: StructuredMemory | None = None,
        episodic: EpisodicStore | None = None,
        spatial: SpatialMemory | None = None,
        llm: LLMProvider | None = None,
        sentiment: SentimentTracker | None = None,
        factions: FactionManager | None = None,
    ):
        self.structured = structured or StructuredMemory()
        self.episodic = episodic or EpisodicStore()
        self.spatial = spatial or SpatialMemory()
        self.llm = llm
        self.sentiment: SentimentTracker | None = sentiment
        self.factions: FactionManager | None = factions

        # Track last reflection time per NPC for importance accumulator
        self._last_reflection_time: dict[str, float] = {}

    def initialise(self) -> None:
        """Initialise all memory subsystems."""
        self.structured.initialise()
        self.episodic.initialise()
        logger.info("Memory manager initialised (all subsystems)")

    # ---------- Memory formation ----------

    async def record_observation(
        self,
        npc_id: str,
        description: str,
        category: str = "observation",
        importance: float = 0.5,
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
        tile_sector: str = "",
        tile_arena: str = "",
    ) -> str:
        """
        Full memory formation pipeline for a single observation.

        1. Score importance (use provided or LLM-scored)
        2. Store in episodic memory
        3. Update spatial memory
        4. Extract and store structured facts if detectable
        """
        # Store episodic
        memory_id = self.episodic.add_memory(
            npc_id=npc_id,
            description=description,
            category=category,
            importance=importance,
            game_time=game_time,
            location_x=location_x,
            location_z=location_z,
        )

        # Update spatial memory
        if tile_sector:
            self.spatial.update_from_perception(
                npc_id=npc_id,
                sector=tile_sector,
                arena=tile_arena,
                note=description,
                game_time=game_time,
            )

        # Try to extract structured facts from the observation
        self._extract_facts(npc_id, description, game_time)

        return memory_id

    async def record_conversation(
        self,
        npc_a_id: str,
        npc_b_id: str,
        npc_a_name: str,
        npc_b_name: str,
        exchanges: list[dict[str, str]],
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
    ) -> None:
        """
        Record a completed conversation into memory for both participants.

        Stores as episodic memory + structured event + relationship fact.
        """
        exchange_text = " | ".join(
            f"{e.get('speaker', '?')}: {e.get('message', '')}"
            for e in exchanges
        )
        summary = f"Conversation between {npc_a_name} and {npc_b_name}: {exchange_text}"

        # Record event
        self.structured.record_event(
            event_type="conversation",
            description=summary[:500],
            participants=[npc_a_id, npc_b_id],
            location_x=location_x,
            location_z=location_z,
            game_time=game_time,
            importance=0.5,
        )

        # Store episodic memory for both participants
        for npc_id, other_name in [
            (npc_a_id, npc_b_name), (npc_b_id, npc_a_name),
        ]:
            self.episodic.add_memory(
                npc_id=npc_id,
                description=f"Had a conversation with {other_name}. {exchange_text}",
                category="conversation",
                importance=0.6,
                game_time=game_time,
                location_x=location_x,
                location_z=location_z,
            )

            # Store relationship fact
            self.structured.add_fact(
                npc_id=npc_id,
                subject=npc_id,
                predicate="spoke_with",
                obj=other_name,
                confidence=1.0,
                source="conversation",
                game_time=game_time,
            )

    async def record_reflection(
        self,
        npc_id: str,
        insight: str,
        game_time: float = 0.0,
    ) -> str:
        """Store a reflection as a high-importance episodic memory."""
        memory_id = self.episodic.add_memory(
            npc_id=npc_id,
            description=f"Reflection: {insight}",
            category="reflection",
            importance=0.8,
            game_time=game_time,
        )
        self._last_reflection_time[npc_id] = game_time
        return memory_id

    # ---------- Retrieval ----------

    def retrieve_context(
        self,
        npc_id: str,
        query: str,
        cognition_tier: int = 1,
        current_game_time: float = 0.0,
    ) -> MemoryContext:
        """
        Retrieve a memory context package for an NPC's cognition cycle.

        Amount and type of retrieval depends on cognition tier.
        """
        limit = TIER_CONTEXT_LIMITS.get(cognition_tier, 0)
        if limit == 0:
            return MemoryContext(episodic=[], facts=[], spatial_summary="")

        # Episodic retrieval (tier 1-2 only — tier 3 skips embedding search)
        episodic_results: list[RetrievalResult] = []
        if cognition_tier <= 2:
            episodic_results = self.episodic.retrieve(
                npc_id=npc_id,
                query=query,
                current_game_time=current_game_time,
                limit=limit,
            )

        # Structured facts (all active tiers)
        facts = self.structured.get_facts(npc_id=npc_id, limit=limit)

        # Spatial summary (tier 1-2 only)
        spatial_summary = ""
        if cognition_tier <= 2:
            spatial_summary = self.spatial.get_world_summary(npc_id)

        return MemoryContext(
            episodic=episodic_results,
            facts=facts,
            spatial_summary=spatial_summary,
        )

    def get_relationship_context(
        self,
        npc_id: str,
        other_name: str,
        other_id: str = "",
    ) -> str:
        """
        Get what an NPC knows about another NPC.
        Combines structured facts, sentiment, and faction context.
        Used to enrich conversation and planning prompts.
        """
        parts: list[str] = []

        # Structured facts
        facts = self.structured.get_facts_about(npc_id, other_name, limit=10)
        if facts:
            lines = [f.to_natural() for f in facts]
            parts.append("What you know about them: " + "; ".join(lines))

        # Sentiment dimensions
        if self.sentiment and other_id:
            sent = self.sentiment.get(npc_id, other_id)
            desc = sent.to_description()
            if desc != "neutral acquaintance":
                parts.append(f"Your feelings towards them: {desc}")

        # Faction context
        if self.factions and other_id:
            if self.factions.same_faction(npc_id, other_id):
                faction = self.factions.get_npc_faction(npc_id)
                parts.append(
                    f"You are both members of {faction.name}."
                    if faction else "You are in the same faction."
                )
            elif self.factions.are_allies(npc_id, other_id):
                parts.append("Your factions are allied.")
            elif self.factions.are_rivals(npc_id, other_id):
                parts.append("Your factions are rivals.")

        if not parts:
            return "You know them as a fellow townsperson."
        return " ".join(parts)

    # ---------- Importance accumulator ----------

    def should_reflect(
        self,
        npc_id: str,
        current_game_time: float,
    ) -> bool:
        """Check if accumulated importance warrants a reflection."""
        last_time = self._last_reflection_time.get(npc_id, 0.0)
        total = self.episodic.importance_since(npc_id, last_time)
        return total >= REFLECTION_IMPORTANCE_THRESHOLD

    def get_focal_points(
        self,
        npc_id: str,
        limit: int = 3,
    ) -> list[str]:
        """
        Get the most important recent memories as focal points
        for reflection generation.
        """
        last_time = self._last_reflection_time.get(npc_id, 0.0)
        recent = self.episodic.get_recent(npc_id, limit=20)

        # Filter to since last reflection and sort by importance
        since = [m for m in recent if m.game_time >= last_time]
        since.sort(key=lambda m: m.importance, reverse=True)

        return [m.description for m in since[:limit]]

    # ---------- Fact extraction ----------

    def _extract_facts(
        self,
        npc_id: str,
        description: str,
        game_time: float,
    ) -> None:
        """
        Simple heuristic extraction of structured facts from observations.

        Detects patterns like "X is a Y", "X is doing Y".
        Full LLM-based extraction happens during reflection.
        """
        desc_lower = description.lower()

        # "Name the occupation is doing something"
        if " the " in desc_lower and " is " in desc_lower:
            parts = description.split(" the ", 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                rest = parts[1]
                if " is " in rest:
                    occ_and_action = rest.split(" is ", 1)
                    if len(occ_and_action) == 2:
                        occupation = occ_and_action[0].strip()
                        self.structured.add_fact(
                            npc_id=npc_id,
                            subject=subject,
                            predicate="is_a",
                            obj=occupation,
                            source="observation",
                            game_time=game_time,
                        )

    # ---------- Stats and inspector ----------

    def get_stats(self) -> dict[str, Any]:
        """Combined stats from all memory subsystems."""
        return {
            "structured": self.structured.get_stats(),
            "episodic": self.episodic.get_stats(),
            "spatial": self.spatial.get_stats(),
        }

    def get_npc_memory_summary(self, npc_id: str) -> dict[str, Any]:
        """Full memory dump for a specific NPC (for inspector)."""
        facts = self.structured.get_facts(npc_id, limit=100)
        goals = self.structured.get_active_goals(npc_id)
        recent_episodic = self.episodic.get_recent(npc_id, limit=20)
        spatial_tree = self.spatial.get_tree(npc_id)

        return {
            "npc_id": npc_id,
            "facts": [f.to_dict() for f in facts],
            "goals": [g.to_dict() for g in goals],
            "recent_memories": [m.to_dict() for m in recent_episodic],
            "spatial": spatial_tree,
            "episodic_count": self.episodic.count(npc_id),
            "last_reflection": self._last_reflection_time.get(npc_id, 0.0),
        }

    def get_recent_activity(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get recent memory activity across all NPCs (for inspector feed)."""
        events = self.structured.get_recent_events(limit=limit)
        return [e.to_dict() for e in events]

    def close(self) -> None:
        """Clean up resources."""
        self.structured.close()
