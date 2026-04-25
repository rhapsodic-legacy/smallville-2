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
    EpisodicStore, EpisodicMemory, RetrievalResult, normalise_tags,
)
from core.memory.spatial import SpatialMemory

if TYPE_CHECKING:
    from core.npc.llm_client import LLMProvider
    from core.relationships.sentiment import SentimentTracker
    from core.relationships.structures import FactionManager

logger = logging.getLogger(__name__)

# Importance threshold that triggers a reflection.
# Stanford used ~100. With observations at 0.5 importance each,
# this triggers after ~160 observations (~80 perception cycles).
REFLECTION_IMPORTANCE_THRESHOLD = 80.0

# How many memories to include in context for different tiers
TIER_CONTEXT_LIMITS = {
    1: 10,  # Full LLM — rich memory context
    2: 5,   # Simplified — moderate context
    3: 2,   # State machine — minimal structured facts only
    4: 0,   # Frozen — no retrieval
}


def _other_name(participants: dict[str, str], this_id: str) -> str:
    """Return the name of the other participant, or 'them' if unknown."""
    for npc_id, name in participants.items():
        if npc_id != this_id:
            return name
    return "them"


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

        # Phase A.5 — rolling buffer of recently-formed notable memories.
        # Drained each tick by the broadcast layer to animate a sparkle
        # over an NPC's head when they form a memory the player should
        # know about. Only entries at or above the notable-importance
        # threshold are queued, so idle perception spam ("so-and-so is
        # walking nearby") doesn't flood the HUD.
        self._memory_events: list[dict[str, Any]] = []

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

    async def store_perception(
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
        mentioned_npc_id: str = "",
    ) -> str:
        """Store a perception with relationship-boosted importance.

        If the perception mentions an NPC the observer has strong
        feelings about, importance is boosted so that reflections
        and replanning are triggered by socially meaningful events.
        """
        boosted = importance

        # Boost if the perceived NPC has a strong relationship
        if mentioned_npc_id and self.sentiment is not None:
            sent = self.sentiment.get(npc_id, mentioned_npc_id)
            disposition = abs(sent.overall_disposition())
            # Strong relationship (disposition ≥30) → up to +0.2 boost
            if disposition >= 10:
                boosted = min(1.0, importance + min(0.2, disposition / 100 * 0.3))

        return await self.record_observation(
            npc_id=npc_id,
            description=description,
            category=category,
            importance=boosted,
            game_time=game_time,
            location_x=location_x,
            location_z=location_z,
            tile_sector=tile_sector,
            tile_arena=tile_arena,
        )

    # ---------- Per-turn conversation persistence ----------
    #
    # A turn is one player/NPC utterance plus its reply. Persisting
    # mid-conversation means the listener's memory layer sees the
    # exchange right away — important for long player chats that
    # don't finish until the window closes, and for NPC↔NPC dialogues
    # we'd like to reflect on partial state.
    #
    # Turn memories are tagged with category "conversation_turn" and
    # the originating conversation id so the final consolidation pass
    # (on conversation close) can dedupe them.

    # Memories at or above this importance fire a `memory_formed`
    # event into the broadcast queue. Keep it above normal perception
    # noise (0.5) but below reflections (0.8).
    MEMORY_EVENT_THRESHOLD = 0.6

    def _emit_memory_event(
        self, npc_id: str, importance: float,
        category: str, summary: str,
    ) -> None:
        """Queue a `memory_formed` event if the memory is notable.

        Kept as a private sink so persistence helpers don't need to
        know about the broadcast layer. Drained by
        `drain_memory_events()` once per tick.
        """
        if importance < self.MEMORY_EVENT_THRESHOLD:
            return
        self._memory_events.append({
            "npc_id": npc_id,
            "importance": round(importance, 2),
            "category": category,
            "summary": summary[:140],
        })

    def drain_memory_events(self) -> list[dict[str, Any]]:
        """Return and clear the queued memory_formed events."""
        events = self._memory_events
        self._memory_events = []
        return events

    TURN_MEMORY_CATEGORY = "conversation_turn"
    TURN_MEMORY_DEFAULT_IMPORTANCE = 0.45
    TURN_MEMORY_HIGH_KEYWORDS = (
        "accuse", "accusation", "liar", "lie", "lying",
        "hoard", "steal", "thief", "betray", "enemy",
        "kill", "fight", "attack", "promise", "swear",
        "love", "hate", "furious", "desperate", "emergency",
    )
    TURN_MEMORY_HIGH_IMPORTANCE = 0.7

    def _score_turn_importance(self, text: str) -> float:
        """Quick heuristic for how weighty a single exchange is.

        Long, emotionally-loaded turns rate higher — those are the
        ones the player expects the NPC to remember. Cheap: keyword
        hit + length proxy. Downstream phases (B) will replace this
        with structured outcome extraction.
        """
        if not text:
            return 0.0
        lowered = text.lower()
        if any(kw in lowered for kw in self.TURN_MEMORY_HIGH_KEYWORDS):
            return self.TURN_MEMORY_HIGH_IMPORTANCE
        # Mild importance bump for long turns (500+ chars ≈ paragraph).
        if len(text) >= 500:
            return min(1.0, self.TURN_MEMORY_DEFAULT_IMPORTANCE + 0.1)
        return self.TURN_MEMORY_DEFAULT_IMPORTANCE

    async def persist_conversation_turn(
        self,
        conversation_id: str,
        npc_a_id: str,
        npc_b_id: str,
        npc_a_name: str,
        npc_b_name: str,
        exchange: dict[str, str],
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
    ) -> list[str]:
        """Write a single exchange pair into both participants' memory.

        Returns the episodic memory ids produced (one per participant)
        so callers — typically the server's tick broadcaster — can
        surface them via the `memory_formed` notification channel.

        Shape of `exchange`: {"speaker": name, "message": text}. A
        turn consists of just that one utterance; the conversation's
        natural alternation means each call captures half of a turn,
        and a back-and-forth pair produces two persistence calls.
        """
        speaker = exchange.get("speaker", "?")
        message = exchange.get("message", "")
        if not message.strip():
            return []

        importance = self._score_turn_importance(message)
        description = f"{speaker} said: \"{message}\""

        memory_ids: list[str] = []
        for npc_id in (npc_a_id, npc_b_id):
            memory_id = self.episodic.add_memory(
                npc_id=npc_id,
                description=description,
                category=self.TURN_MEMORY_CATEGORY,
                importance=importance,
                game_time=game_time,
                location_x=location_x,
                location_z=location_z,
                extra_metadata={
                    "conversation_id": conversation_id,
                    "speaker": speaker,
                    "partner_a": npc_a_name,
                    "partner_b": npc_b_name,
                },
            )
            memory_ids.append(memory_id)
            self._emit_memory_event(
                npc_id, importance, self.TURN_MEMORY_CATEGORY, description,
            )
        return memory_ids

    async def persist_new_exchanges(
        self,
        conv: Any,
        npc_a: Any,
        npc_b: Any,
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
    ) -> list[str]:
        """Walk a Conversation's unpersisted tail and record each exchange.

        Idempotent — uses `conv.persisted_exchange_count` as a cursor
        so repeated calls on a growing conversation only add memories
        for genuinely new utterances. Works for both player↔NPC and
        NPC↔NPC paths. `conv` must expose `conv_id`, `exchanges`, and
        `persisted_exchange_count`.
        """
        if not getattr(conv, "exchanges", None):
            return []
        new_ids: list[str] = []
        cursor = getattr(conv, "persisted_exchange_count", 0)
        while cursor < len(conv.exchanges):
            ex = conv.exchanges[cursor]
            try:
                ids = await self.persist_conversation_turn(
                    conversation_id=conv.conv_id,
                    npc_a_id=npc_a.npc_id,
                    npc_b_id=npc_b.npc_id,
                    npc_a_name=npc_a.name,
                    npc_b_name=npc_b.name,
                    exchange={
                        "speaker": ex.speaker_name,
                        "message": ex.message,
                    },
                    game_time=game_time,
                    location_x=location_x,
                    location_z=location_z,
                )
                new_ids.extend(ids)
            except Exception:
                logger.exception(
                    "Per-turn persistence failed (conv=%s)", conv.conv_id,
                )
            cursor += 1
        conv.persisted_exchange_count = cursor
        return new_ids

    def consolidate_conversation_turns(
        self, conversation_id: str,
    ) -> int:
        """Remove the per-turn episodic entries for a conversation.

        Called after a conversation closes and `record_conversation`
        has written the consolidated summary. Returns the number of
        turn memories removed so callers can log the churn.
        """
        removed = self.episodic.delete_by_metadata(
            "conversation_id", conversation_id,
        )
        if removed:
            logger.info(
                "Consolidated %d turn memories for conversation %s",
                removed, conversation_id,
            )
        return removed

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
        Extracts structured facts from conversation content (LLM or heuristic).
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
            convo_description = (
                f"Had a conversation with {other_name}. {exchange_text}"
            )
            self.episodic.add_memory(
                npc_id=npc_id,
                description=convo_description,
                category="conversation",
                importance=0.6,
                game_time=game_time,
                location_x=location_x,
                location_z=location_z,
            )
            self._emit_memory_event(
                npc_id, 0.6, "conversation", convo_description,
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

        # Extract structured facts from conversation content
        await self._extract_conversation_facts(
            npc_a_id, npc_b_id, npc_a_name, npc_b_name,
            exchanges, game_time,
        )

    # ---------- Phase K: tag derivation helpers ----------
    #
    # Tags are the "pointers" the Phase K roadmap names — small,
    # normalised strings that let an NPC's thinking layer probe the
    # tag index in microseconds. We derive them at memory-creation
    # time from Phase B outcomes and town-agenda shapes, so the
    # cost is paid once (on persistence) and retrieval is free.

    _TAG_STOPWORDS: set[str] = {
        "about", "after", "again", "against", "always", "before",
        "being", "between", "could", "doing", "every", "first",
        "going", "might", "never", "other", "should", "since",
        "their", "there", "these", "thing", "those", "under",
        "until", "where", "which", "while", "would",
    }
    _TAG_MAX_FROM_TEXT: int = 3

    def _extract_topic_tags(self, text: str) -> set[str]:
        """Pull low-noise topic tags from a free-text claim/subject.

        Picks the longest tokens (length > 4) as topic-markers,
        stopword-filtered. Keeps at most `_TAG_MAX_FROM_TEXT` so a
        verbose claim doesn't flood the index.
        """
        if not text:
            return set()
        import re
        tokens = [
            t.lower() for t in re.findall(r"[A-Za-z]+", text)
            if len(t) > 4 and t.lower() not in self._TAG_STOPWORDS
        ]
        tokens.sort(key=len, reverse=True)
        return normalise_tags(tokens[: self._TAG_MAX_FROM_TEXT])

    def tags_for_commitment(self, c: Any) -> set[str]:
        base = {"outcome:commitment"}
        base.update(self._extract_topic_tags(getattr(c, "subject", "")))
        about = getattr(c, "about", "") or ""
        if about:
            base.add(f"about:{normalise_tags(about).pop() if normalise_tags(about) else ''}")
            base.discard("about:")
            base.update(self._extract_topic_tags(about))
        return {t for t in base if t}

    def tags_for_accusation(self, a: Any) -> set[str]:
        base = {"outcome:accusation"}
        accused = normalise_tags(getattr(a, "accused", ""))
        accuser = normalise_tags(getattr(a, "accuser", ""))
        for t in accused:
            base.add(f"accused:{t}")
            base.add(t)
        for t in accuser:
            base.add(f"accuser:{t}")
        base.update(self._extract_topic_tags(getattr(a, "claim", "")))
        return base

    def tags_for_relayed_claim(self, r: Any) -> set[str]:
        base = {"outcome:relayed_claim"}
        subject = normalise_tags(getattr(r, "subject", ""))
        cited = normalise_tags(getattr(r, "cited_source", ""))
        relayed = normalise_tags(getattr(r, "relayed_by", ""))
        for t in subject:
            base.add(f"subject:{t}")
            base.add(t)
        for t in cited:
            base.add(f"cited:{t}")
        for t in relayed:
            base.add(f"from:{t}")
        base.update(self._extract_topic_tags(getattr(r, "claim", "")))
        return base

    def tags_for_town_event(
        self, goal_id: str, category: str,
    ) -> set[str]:
        base: set[str] = set()
        goal_tag = normalise_tags(goal_id)
        for t in goal_tag:
            base.add(f"agenda:{t}")
            base.add(t)
        if category:
            base.add(f"category:{normalise_tags(category).pop() if normalise_tags(category) else category.lower()}")
        return {t for t in base if t}

    def retrieve_by_tags(
        self,
        npc_id: str,
        tags: Any,
        limit: int = 10,
    ) -> list[EpisodicMemory]:
        """Passthrough to `EpisodicStore.retrieve_by_tags`."""
        return self.episodic.retrieve_by_tags(npc_id, tags, limit=limit)

    # Phase K.6 — how much the composite retrieval score gets bumped
    # for a tag-index hit. High enough to lift a buried-but-tagged
    # memory above recent-but-untagged chatter; bounded so a single
    # tag-match doesn't swamp a dozen genuinely relevant recent
    # observations.
    TAG_RETRIEVAL_BOOST: float = 0.5

    def retrieve_with_tag_boost(
        self,
        npc_id: str,
        query: str,
        context_tags: Any,
        current_game_time: float = 0.0,
        limit: int = 10,
        category: str | None = None,
    ) -> list[RetrievalResult]:
        """Composite retrieval that lifts tag-matched memories.

        Runs the normal semantic retrieval, then probes the tag index
        for `context_tags`. Any memory present in both sets gets a
        fixed importance bonus; tag-only hits get injected at the
        bottom with the boost applied so they can't be *missed* when
        relevant.
        """
        base = self.episodic.retrieve(
            npc_id=npc_id,
            query=query,
            current_game_time=current_game_time,
            limit=limit,
            category=category,
        )
        tag_hits = self.episodic.retrieve_by_tags(
            npc_id, context_tags, limit=limit,
        )
        if not tag_hits:
            return base

        boosted_ids = {m.memory_id for m in tag_hits}
        for result in base:
            if result.memory.memory_id in boosted_ids:
                result.composite_score += self.TAG_RETRIEVAL_BOOST

        # Inject any tag hits that weren't in the semantic result.
        base_ids = {r.memory.memory_id for r in base}
        extras = [m for m in tag_hits if m.memory_id not in base_ids]
        for mem in extras:
            base.append(RetrievalResult(
                memory=mem,
                relevance_score=0.0,
                recency_score=0.0,
                importance_score=mem.importance,
                composite_score=mem.importance + self.TAG_RETRIEVAL_BOOST,
            ))

        base.sort(key=lambda r: r.composite_score, reverse=True)
        return base[:limit]

    def infer_tags_from_context(
        self,
        npc: Any,
        partner_id: str = "",
        partner_name: str = "",
        active_agenda_titles: Any = None,
        recent_text: str = "",
    ) -> set[str]:
        """Phase K.4 — derive the tag vector to probe for "right now".

        Unions several lightweight signals so retrieval stays cheap:
        - The partner's name (lets Petra-about-Bran memories surface
          whenever Bran is nearby).
        - Active town agenda titles (lets agenda-tagged memories
          surface when the current goal matches).
        - Every self_concept key on this NPC (lets identity-linked
          memories surface).
        - Noun-ish tokens from any recent free-text the caller wants
          to feed in (e.g. the most recent conversation line).

        Microsecond operation — pure set-building, no I/O.
        """
        tag_set: set[str] = set()
        if partner_name:
            tag_set.update(normalise_tags(partner_name))
        if partner_id:
            tag_set.update(normalise_tags(partner_id))
        if active_agenda_titles:
            for title in active_agenda_titles:
                for t in normalise_tags(title):
                    tag_set.add(t)
                    tag_set.add(f"agenda:{t}")
        self_concept = getattr(npc, "self_concept", None) or {}
        for key in self_concept:
            tag_set.update(normalise_tags(key.replace(":", "_")))
        if recent_text:
            tag_set.update(self._extract_topic_tags(recent_text))
        return {t for t in tag_set if t}

    def store_conversation_outcomes(
        self,
        outcome: Any,
        participants: dict[str, str],
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
    ) -> list[str]:
        """Persist a Phase B `ConversationOutcome` across both participants.

        `participants` maps npc_id -> name. Commitments land only on
        the speaker ("I have committed to ..."). Accusations land on
        accuser, accused, and any other participant who witnessed the
        conversation. Relayed claims produce first-class records on
        both participants so downstream retrieval (Phase C) can
        surface them by subject or by cited source.

        Returns the flat list of memory ids written so callers can
        include them in logging / diagnostics.
        """
        if outcome is None or outcome.is_empty():
            return []

        name_to_id: dict[str, str] = {
            name.lower(): npc_id for npc_id, name in participants.items()
        }

        def resolve_id(name: str) -> str:
            return name_to_id.get((name or "").lower(), "")

        written: list[str] = []

        commitment_tags = {}  # cache per-commitment tags
        # --- Commitments: land on the speaker at high importance. ---
        for c in outcome.commitments:
            speaker_id = resolve_id(c.speaker)
            if not speaker_id:
                continue
            desc = f"I promised to {c.subject.strip().rstrip('.')}."
            if c.about:
                desc += f" ({c.about})"
            tags = self.tags_for_commitment(c)
            commitment_tags[id(c)] = tags
            mid = self.episodic.add_memory(
                npc_id=speaker_id,
                description=desc,
                category="commitment",
                importance=0.75,
                game_time=game_time,
                location_x=location_x,
                location_z=location_z,
                extra_metadata={
                    "outcome_kind": "commitment",
                    "source_speaker": c.speaker,
                    "unresolved": True,
                },
                tags=tags,
            )
            written.append(mid)
            self._emit_memory_event(
                speaker_id, 0.75, "commitment", desc,
            )

        # --- Accusations: land on accuser, accused, and witnesses. ---
        for a in outcome.accusations:
            accuser_id = resolve_id(a.accuser)
            accused_id = resolve_id(a.accused)
            if not a.claim:
                continue
            self_line = (
                f"I accused {a.accused or 'them'} of {a.claim.strip().rstrip('.')}."
            )
            target_line = (
                f"{a.accuser or 'Someone'} accused me of "
                f"{a.claim.strip().rstrip('.')}."
            )
            witness_line = (
                f"{a.accuser or 'Someone'} accused "
                f"{a.accused or 'someone'} of {a.claim.strip().rstrip('.')}."
            )
            tags = self.tags_for_accusation(a)
            for npc_id in participants:
                if npc_id == accuser_id:
                    desc = self_line
                elif npc_id == accused_id:
                    desc = target_line
                else:
                    desc = witness_line
                mid = self.episodic.add_memory(
                    npc_id=npc_id,
                    description=desc,
                    category="accusation",
                    importance=0.8,
                    game_time=game_time,
                    location_x=location_x,
                    location_z=location_z,
                    extra_metadata={
                        "outcome_kind": "accusation",
                        "accuser": a.accuser,
                        "accused": a.accused,
                        "unresolved": True,
                    },
                    tags=tags,
                )
                written.append(mid)
                self._emit_memory_event(
                    npc_id, 0.8, "accusation", desc,
                )

        # --- Relayed claims: both participants get a first-class
        # record. The listener's entry flags `cited_source` so Phase C
        # retrieval can surface it when they next encounter that
        # source; the speaker's entry keeps the chain visible from
        # their side too (useful when the speaker meets the subject).
        for r in outcome.relayed_claims:
            relayed_by_id = resolve_id(r.relayed_by)
            if not r.claim or not r.cited_source:
                continue
            summary_subject = r.subject or "them"
            base = (
                f"{r.relayed_by or 'Someone'} told me that "
                f"{r.cited_source} said {summary_subject} "
                f"{r.claim.strip().rstrip('.')}."
            )
            speaker_line = (
                f"I told {_other_name(participants, relayed_by_id)} "
                f"that {r.cited_source} said {summary_subject} "
                f"{r.claim.strip().rstrip('.')}."
            )
            tags = self.tags_for_relayed_claim(r)
            for npc_id in participants:
                desc = speaker_line if npc_id == relayed_by_id else base
                mid = self.episodic.add_memory(
                    npc_id=npc_id,
                    description=desc,
                    category="relayed_claim",
                    importance=0.75,
                    game_time=game_time,
                    location_x=location_x,
                    location_z=location_z,
                    extra_metadata={
                        "outcome_kind": "relayed_claim",
                        "subject": r.subject,
                        "claim": r.claim,
                        "cited_source": r.cited_source,
                        "relayed_by": r.relayed_by,
                        "unresolved": True,
                    },
                    tags=tags,
                )
                written.append(mid)
                self._emit_memory_event(
                    npc_id, 0.75, "relayed_claim", desc,
                )

        return written

    # ---------- Phase C: unresolved matter retrieval & resolution ----------
    #
    # Outcome memories (commitments / accusations / relayed claims)
    # carry an `unresolved=True` metadata flag from Phase B. Phase C
    # surfaces the ones relevant to the NPC's current conversation
    # partner so the prompt can nudge them to bring it up, then marks
    # the original records resolved once the topic is aired.

    _UNRESOLVED_CATEGORIES: tuple[str, ...] = (
        "commitment", "accusation", "relayed_claim",
    )
    _MATTER_FETCH_LIMIT = 40

    # Phase I.3 — stagnation escalation. Every bedtime review bumps
    # `stagnation_days` on a stalled commitment's metadata; the
    # retrieval ranker adds a capped per-day boost so stale matters
    # rise into the prompt's top-N. Tuned for 60+ day sims: the
    # linear growth saturates at day 15 so old commitments plateau
    # rather than permanently drown out everything else. After the
    # cap, recency (game_time) breaks ties between saturated items,
    # naturally favouring the more-recent-but-also-stalled entry
    # over a 30-day-old relic.
    STAGNATION_BOOST_PER_DAY: float = 0.04
    STAGNATION_BOOST_CAP: int = 15

    def retrieve_unresolved_matters(
        self,
        npc_id: str,
        partner_id: str = "",
        partner_name: str = "",
        limit: int = 3,
    ) -> list[Any]:
        """Return `partner`-relevant open matters the NPC hasn't aired.

        A matter is relevant when the partner is named as the
        accuser, accused, cited_source, subject, or the partner's
        name appears in the memory description (last-resort match
        for commitments whose `about` field is empty).

        Matters are sorted by composite score desc, then recency
        desc. Composite score = `importance + stagnation_boost` per
        `_stagnation_boost` — so a commitment that's been stalling
        for weeks rises above fresh commitments, capped at
        `STAGNATION_BOOST_CAP` days to prevent ancient baggage from
        drowning out merely-moderately-stalled newer items.
        `limit` keeps the prompt block short — planner can raise it.
        """
        partner_key = (partner_name or "").lower().strip()
        matters: list[Any] = []

        for category in self._UNRESOLVED_CATEGORIES:
            mems = self.episodic.get_recent(
                npc_id, limit=self._MATTER_FETCH_LIMIT, category=category,
            )
            for mem in mems:
                meta = getattr(mem, "metadata", None) or {}
                if not meta.get("unresolved"):
                    continue
                if not self._matter_names_partner(
                    mem, meta, partner_id, partner_key,
                ):
                    continue
                matters.append(mem)

        matters.sort(
            key=lambda m: (
                m.importance + self._stagnation_boost(m),
                m.game_time,
            ),
            reverse=True,
        )
        return matters[:limit]

    @classmethod
    def _stagnation_boost(cls, mem: Any) -> float:
        """Return the capped retrieval boost for a commitment's
        `stagnation_days` counter. Non-commitments (accusations,
        relayed_claims) return 0 — they don't accumulate stagnation.

        Cap and per-day weight are class constants so sub-projects
        can tune the ramp by subclassing without editing core.
        """
        if getattr(mem, "category", None) != "commitment":
            return 0.0
        meta = getattr(mem, "metadata", None) or {}
        try:
            days = int(meta.get("stagnation_days", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        if days <= 0:
            return 0.0
        effective = min(days, cls.STAGNATION_BOOST_CAP)
        return cls.STAGNATION_BOOST_PER_DAY * effective

    @staticmethod
    def _matter_names_partner(
        mem: Any, meta: dict[str, Any],
        partner_id: str, partner_key: str,
    ) -> bool:
        """Test whether a stored outcome names this conversation partner.

        Deliberately permissive — a commitment that mentions the
        partner only in its free-text description should still
        surface, otherwise "I promised to speak with Petra tomorrow"
        stays invisible when Petra is the one we're talking to.
        """
        if not (partner_id or partner_key):
            return False

        for field in ("accused", "accuser", "cited_source",
                      "subject", "relayed_by"):
            val = (meta.get(field) or "").strip().lower()
            if val and partner_key and val == partner_key:
                return True

        if partner_key and partner_key in (mem.description or "").lower():
            return True
        return False

    def format_unresolved_matters(
        self, matters: list[Any], partner_name: str,
    ) -> str:
        """Render matters as a one-liner for prompt injection.

        Returns the empty string for no matters so the prompt stays
        clean. Matches the shape of `TownAgenda.summary_for_prompt`.
        """
        if not matters:
            return ""
        phrases: list[str] = []
        for mem in matters:
            desc = (mem.description or "").strip().rstrip(".")
            if desc:
                phrases.append(desc)
        if not phrases:
            return ""
        return (
            f"Matters you want to raise with {partner_name}: "
            + "; ".join(phrases) + "."
        )

    def resolve_matters_from_transcript(
        self,
        npc_id: str,
        partner_id: str,
        partner_name: str,
        transcript_text: str,
    ) -> int:
        """Flip `unresolved` → False on matters aired in this chat.

        A matter is resolved when:
        - The conversation was between the holder and the named
          party (accused / cited_source / subject / accuser).
        - And any distinctive keyword from the original claim
          appears in the transcript (>4-char word match).

        This is intentionally conservative: a chance meeting where
        neither party raises the subject does NOT close the matter.
        Returns the number of matters resolved so callers can log it.
        """
        if not transcript_text:
            return 0
        lowered = transcript_text.lower()
        partner_key = (partner_name or "").lower().strip()

        # Only consider matters naming the partner to begin with.
        candidates = self.retrieve_unresolved_matters(
            npc_id=npc_id,
            partner_id=partner_id,
            partner_name=partner_name,
            limit=self._MATTER_FETCH_LIMIT,
        )

        resolved = 0
        for mem in candidates:
            meta = getattr(mem, "metadata", None) or {}
            claim_text = (
                meta.get("claim") or meta.get("subject") or mem.description
            )
            if not self._transcript_airs_claim(
                claim_text, lowered, partner_key,
            ):
                continue
            if self.episodic.update_metadata(
                mem.memory_id, {"unresolved": False},
            ):
                resolved += 1
        if resolved:
            logger.info(
                "RESOLVED %d matter(s) for %s ↔ %s",
                resolved, npc_id, partner_name,
            )
        return resolved

    @staticmethod
    def _transcript_airs_claim(
        claim_text: str, transcript_lower: str, partner_key: str,
    ) -> bool:
        """Heuristic: was the claim discussed?

        Require at least one distinctive token from the claim text
        (length > 4, not a stopword) to appear in the transcript. The
        partner's own name is filtered — we need to see the *topic*
        come up, not just their greeting.
        """
        import re

        stop = {
            "about", "after", "again", "against", "always",
            "because", "before", "being", "between", "could",
            "doing", "every", "first", "going", "hello",
            "might", "never", "other", "quick", "shall",
            "should", "since", "their", "there", "these",
            "thing", "those", "under", "until", "where",
            "which", "while", "would", "years",
        }
        tokens = {
            t for t in re.findall(r"[a-z]+", (claim_text or "").lower())
            if len(t) > 4 and t not in stop and t != partner_key
        }
        if not tokens:
            return False
        return any(t in transcript_lower for t in tokens)

    def record_town_event_memory(
        self,
        npc_id: str,
        description: str,
        category: str,
        importance: float,
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
        goal_id: str = "",
    ) -> str:
        """Persist a town-agenda-sourced memory for a single NPC.

        Thin wrapper over `EpisodicStore.add_memory` that also fires
        a `memory_formed` broadcast event when the importance is
        notable. Used by Phase F to seed NPCs with awareness of
        proposed / joined / completed / expired town goals.

        Kept synchronous because every agenda listener is synchronous
        — writing through `record_observation` would force the
        listener callers to become coroutines.
        """
        memory_id = self.episodic.add_memory(
            npc_id=npc_id,
            description=description,
            category=category,
            importance=importance,
            game_time=game_time,
            location_x=location_x,
            location_z=location_z,
            extra_metadata={"town_goal_id": goal_id} if goal_id else None,
            tags=self.tags_for_town_event(goal_id, category),
        )
        self._emit_memory_event(npc_id, importance, category, description)
        return memory_id

    # ---------- Phase H: day-level compaction ----------

    async def compact_day(
        self,
        npc_id: str,
        game_day: int,
        *,
        npc: Any = None,
        llm: Any = None,
    ) -> str | None:
        """Collapse `game_day`'s untagged memories into a day_summary.

        Tagged memories (Phase K outcomes, town events, notes) skip
        the pass unchanged. Compacted originals are tombstoned with
        a `compacted_into` metadata pointer rather than deleted, so
        provenance stays queryable for diagnostics. Returns the id of
        the summary memory, or None when there's nothing to compact.

        Delegates to `core.memory.compaction.compact_day` to keep the
        summariser logic out of this already-oversized module.
        """
        from core.memory import compaction
        return await compaction.compact_day(
            self, npc_id, game_day, npc=npc, llm=llm,
        )

    async def compact_week(
        self,
        npc_id: str,
        week_number: int,
        *,
        npc: Any = None,
        llm: Any = None,
    ) -> str | None:
        """Collapse a week's day_summaries into a week_summary.

        The week window is `[week * 7, (week + 1) * 7)` game-days.
        Operates only on `day_summary` memories — raws remain either
        part of their own day's summary (already tombstoned) or
        tagged and anchored (left alone). Returns the new summary
        id, or None when no day_summary exists in the week window.
        """
        from core.memory import compaction
        return await compaction.compact_week(
            self, npc_id, week_number, npc=npc, llm=llm,
        )

    # ---------- Phase I.1: bedtime self-review ----------

    async def daily_self_review(
        self,
        npc_id: str,
        game_day: int,
        *,
        npc: Any = None,
        llm: Any = None,
    ) -> Any:
        """Run the bedtime self-review for `npc` on `game_day`.

        Delegates to `core.memory.self_review.daily_self_review` to
        keep this already-oversized module from growing further.
        Returns a `SelfReviewResult` (or None when nothing to review).
        """
        from core.memory import self_review
        return await self_review.daily_self_review(
            self, npc_id, game_day, npc=npc, llm=llm,
        )

    def retrieve_self_commitments(
        self, npc_id: str, limit: int = 6,
    ) -> list[Any]:
        """Return the NPC's own open commitments, newest first.

        Convenience passthrough for callers that want the unresolved
        self-commitment list without running the full review (e.g.
        diagnostic panels, tests, or tools that want to show "what
        does this NPC still owe themselves?"). Mirrors the filter
        `self_review._unresolved_self_commitments` uses.
        """
        from core.memory import self_review
        return self_review._unresolved_self_commitments(
            self, npc_id, limit,
        )

    # ---------- Phase H.5: provenance-chain access ----------

    def get_raw_by_id(self, memory_id: str) -> EpisodicMemory | None:
        """Diagnostic passthrough — fetch a memory ignoring tombstones.

        `retrieve` / `get_recent` / `retrieve_by_tags` hide compacted
        memories by default. Callers that want the raw original (the
        memory panel walking a provenance chain, an inspector tool,
        an audit) go through this method to signal intent.
        """
        return self.episodic.get_raw_by_id(memory_id)

    def get_compacted_sources(
        self, memory_id: str,
    ) -> list[EpisodicMemory]:
        """Return the originals a summary absorbed.

        Given a `day_summary` id, returns the raw memories
        compacted into it. Given a `week_summary` id, returns the
        day_summaries it rolled up. Chain further by calling this
        recursively on each result. Non-summary ids return [].
        """
        return self.episodic.get_compacted_sources(memory_id)

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

    # ---------- Intent recording ----------

    async def record_intent(
        self,
        npc_id: str,
        description: str,
        destination: tuple[int, int] | None = None,
        schedule_slot: str = "",
        subtasks: list[str] | None = None,
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
    ) -> str:
        """
        Record what an NPC is TRYING to do — their intent.

        Unlike observations (reactive — what they saw/heard), intents
        capture the NPC's goal: where they're heading, why, and what
        subtasks they plan to execute. This makes it possible to
        diagnose movement bugs from logs alone.
        """
        parts = [description]
        if destination:
            parts.append(f"destination=({destination[0]}, {destination[1]})")
        if schedule_slot:
            parts.append(f"slot={schedule_slot}")
        if subtasks:
            parts.append(f"plan: {'; '.join(subtasks[:4])}")

        intent_text = f"Intent: {' | '.join(parts)}"

        memory_id = self.episodic.add_memory(
            npc_id=npc_id,
            description=intent_text,
            category="intent",
            importance=0.3,  # Low — diagnostic, not narratively important
            game_time=game_time,
            location_x=location_x,
            location_z=location_z,
        )

        logger.info(
            "NPC %s intent: %s → (%s, %s) [%s]",
            npc_id, description,
            destination[0] if destination else "?",
            destination[1] if destination else "?",
            schedule_slot,
        )

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

    # ---------- Conversation fact extraction ----------

    async def _extract_conversation_facts(
        self,
        npc_a_id: str,
        npc_b_id: str,
        npc_a_name: str,
        npc_b_name: str,
        exchanges: list[dict[str, str]],
        game_time: float,
    ) -> list[tuple[str, str, str]]:
        """Extract structured facts from conversation content.

        Uses LLM if available, otherwise falls back to keyword heuristics.
        Facts are stored for both participants (both heard the same thing).
        Returns the list of (subject, predicate, object) triples extracted.
        """
        conversation_text = "\n".join(
            f"{e.get('speaker', '?')}: {e.get('message', '')}"
            for e in exchanges
        )

        facts: list[tuple[str, str, str]] = []

        # Try LLM extraction, fall back to heuristic
        if self.llm is not None:
            try:
                from core.npc.llm_client import format_prompt
                prompt = format_prompt(
                    "conversation_extract_facts",
                    npc_a_name=npc_a_name,
                    npc_b_name=npc_b_name,
                    conversation=conversation_text,
                )
                response = await self.llm.complete(
                    system="You extract factual information from NPC conversations.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
                    purpose="reflection",
                )
                facts = _parse_fact_triples(response.strip())
            except Exception as e:
                logger.warning("LLM conversation fact extraction failed: %s", e)

        # Always run heuristic to catch what LLM missed
        heuristic_facts = _extract_facts_heuristic(exchanges)
        seen = set(facts)
        for hf in heuristic_facts:
            if hf not in seen:
                facts.append(hf)
                seen.add(hf)

        # Store facts for both participants
        for subject, predicate, obj in facts:
            for npc_id in (npc_a_id, npc_b_id):
                self.structured.add_fact(
                    npc_id=npc_id,
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    confidence=0.8,
                    source="conversation",
                    game_time=game_time,
                )
            # Also store as episodic memory (higher importance for personal facts)
            fact_desc = f"Learnt from conversation: {subject} {predicate.replace('_', ' ')} {obj}"
            importance = 0.7 if predicate in (
                "is_hungry", "is_sick", "is_injured", "needs_help",
                "is_angry", "is_sad", "is_afraid",
            ) else 0.5
            for npc_id in (npc_a_id, npc_b_id):
                self.episodic.add_memory(
                    npc_id=npc_id,
                    description=fact_desc,
                    category="conversation_fact",
                    importance=importance,
                    game_time=game_time,
                )

        return facts

    # ---------- Stats and inspector ----------

    def get_stats(self) -> dict[str, Any]:
        """Combined stats from all memory subsystems."""
        return {
            "structured": self.structured.get_stats(),
            "episodic": self.episodic.get_stats(),
            "spatial": self.spatial.get_stats(),
        }

    def get_npc_memory_summary(
        self,
        npc_id: str,
        limit: int = 20,
        include_compacted: bool = False,
    ) -> dict[str, Any]:
        """Full memory dump for a specific NPC (for inspector).

        `limit=0` returns every memory for the NPC (no cap) so the
        external memory-dump script can see complete history without
        the 20-item default truncation. `include_compacted=True`
        surfaces tombstoned raw memories alongside their summaries
        — useful for auditing what the compactor absorbed.
        """
        facts = self.structured.get_facts(npc_id, limit=100)
        goals = self.structured.get_active_goals(npc_id)
        effective_limit = limit if limit and limit > 0 else 10_000
        recent_episodic = self.episodic.get_recent(
            npc_id, limit=effective_limit,
            include_compacted=include_compacted,
        )
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


# ---------- Module-level helpers for conversation fact extraction ----------

# Keywords that signal a personal state worth extracting as a fact
_STATE_KEYWORDS: dict[str, str] = {
    "hungry": "is_hungry",
    "starving": "is_hungry",
    "tired": "is_tired",
    "exhausted": "is_tired",
    "sick": "is_sick",
    "ill": "is_sick",
    "injured": "is_injured",
    "hurt": "is_injured",
    "angry": "is_angry",
    "furious": "is_angry",
    "sad": "is_sad",
    "afraid": "is_afraid",
    "scared": "is_afraid",
    "worried": "is_worried",
    "lonely": "is_lonely",
    "happy": "is_happy",
    "broke": "has_no_gold",
    "rich": "is_wealthy",
}

# Phrases that signal needs or intentions
_NEED_PATTERNS: list[tuple[str, str]] = [
    ("need help", "needs_help"),
    ("needs help", "needs_help"),
    ("looking for work", "seeking_work"),
    ("looking for a job", "seeking_work"),
    ("want to trade", "wants_to_trade"),
    ("want to buy", "wants_to_buy"),
    ("want to sell", "wants_to_sell"),
    ("planning to leave", "planning_to_leave"),
    ("going to leave", "planning_to_leave"),
]


def _parse_fact_triples(response: str) -> list[tuple[str, str, str]]:
    """Parse LLM response into (subject, predicate, object) triples."""
    if "NO_FACTS" in response:
        return []

    facts: list[tuple[str, str, str]] = []
    for line in response.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and all(parts[:3]):
            subject = parts[0]
            predicate = parts[1].replace(" ", "_").lower()
            obj = parts[2]
            facts.append((subject, predicate, obj))
    return facts


def _extract_facts_heuristic(
    exchanges: list[dict[str, str]],
) -> list[tuple[str, str, str]]:
    """Keyword-based fallback for extracting facts from conversation text."""
    facts: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for exchange in exchanges:
        speaker = exchange.get("speaker", "Someone")
        message = exchange.get("message", "").lower()

        # Check personal state keywords ("I'm hungry", "I am tired")
        for keyword, predicate in _STATE_KEYWORDS.items():
            if keyword in message and ("i'm" in message or "i am" in message
                                        or "i feel" in message):
                triple = (speaker, predicate, "true")
                if triple not in seen:
                    seen.add(triple)
                    facts.append(triple)

        # Check need/intention patterns
        for pattern, predicate in _NEED_PATTERNS:
            if pattern in message:
                triple = (speaker, predicate, "true")
                if triple not in seen:
                    seen.add(triple)
                    facts.append(triple)

    return facts
