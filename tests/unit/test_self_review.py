"""
Phase I.1 + I.2 — bedtime commitment self-review.

Covers the `core.memory.self_review` module in isolation (no
NPCManager): response parser, heuristic fallback, happy path with a
stub LLM, tag inheritance from source commitments, action-intent
extraction, day_summary threading, memory-metadata shape, and the
preserved-category invariant that keeps review memories alive
through the next day's compaction pass.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.memory import self_review
from core.memory.compaction import PRESERVED_CATEGORIES
from core.memory.episodic import EpisodicMemory, EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import (
    GoalProgress, REVIEW_CATEGORY, REVIEW_IMPORTANCE,
    SelfReviewResult, VALID_STATUSES, daily_self_review,
    _fallback_review, _parse_review_response,
    _unresolved_self_commitments, _latest_day_summary,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import LLMProvider
from core.npc.models import NPC, PersonalityTraits
from core.time_system.clock import MINUTES_PER_DAY


# ---------- Helpers ----------


def _mgr() -> MemoryManager:
    mgr = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mgr.initialise()
    return mgr


def _npc(
    name: str = "Seren", occupation: str = "baker",
    long_term_goals: list[str] | None = None,
    cognition_tier: int = 1,
) -> NPC:
    # Default to tier 1 so LLM-driven paths (notably
    # `classify_insight`) actually run when a test wires an LLM. The
    # fallback-only tests don't care about the tier.
    return NPC(
        npc_id=name.lower(),
        name=name,
        age=30,
        occupation=occupation,
        backstory="",
        personality=PersonalityTraits(),
        long_term_goals=long_term_goals or [],
        cognition_tier=cognition_tier,
    )


def _seed_commitment(
    mgr: MemoryManager, npc_id: str, subject: str,
    *, unresolved: bool = True, tags: set[str] | None = None,
    game_time: float = 60.0,
) -> str:
    """Drop a Phase B commitment memory onto the NPC's own store.

    Mirrors what `persist_outcomes` would write: category=commitment,
    importance=0.75, metadata.unresolved, tags the caller provides.
    """
    return mgr.episodic.add_memory(
        npc_id=npc_id,
        description=f"I promised to {subject}.",
        category="commitment",
        importance=0.75,
        game_time=game_time,
        extra_metadata={
            "outcome_kind": "commitment",
            "source_speaker": npc_id,
            "unresolved": unresolved,
        },
        tags=tags or set(),
    )


def _seed_day_summary(
    mgr: MemoryManager, npc_id: str, day: int, text: str = "A quiet day.",
) -> str:
    return mgr.episodic.add_memory(
        npc_id=npc_id,
        description=text,
        category="day_summary",
        importance=0.6,
        game_time=(day + 1) * MINUTES_PER_DAY - 1.0,
        extra_metadata={"day": day},
    )


class _StubLLM(LLMProvider):
    """Minimal LLM stub: returns a fixed response per purpose.

    Records every call so tests can assert the prompt shape.
    """

    def __init__(self, response_by_purpose: dict[str, str]):
        super().__init__()
        self._responses = response_by_purpose
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, system: str, messages: list[dict[str, str]],
        max_tokens: int = 300, temperature: float = 0.7,
        purpose: str = "general", **kwargs: Any,
    ) -> str:
        self.calls.append({
            "system": system, "messages": messages, "purpose": purpose,
        })
        if purpose in self._responses:
            return self._responses[purpose]
        # Default: return NO_ACTION for classify_insight so we don't
        # accidentally build action intents in tests that didn't arm one.
        if purpose == "reflection":
            return "NO_ACTION"
        return ""


# ---------- Response parser ----------


class TestParseReviewResponse:
    def test_full_block_parses(self):
        text = (
            "SUMMARY: It was a hard day but I kept my head.\n"
            "GOAL: deliver bread to Bran\n"
            "STATUS: stalled\n"
            "NOTE: Bran was out when I called.\n"
            "GOAL: finish roof repair\n"
            "STATUS: moving\n"
            "NOTE: nailed in half the shingles.\n"
            "NEXT: try Bran's house first thing tomorrow.\n"
        )
        per_goal, summary, next_line = _parse_review_response(text, [], [])
        assert summary == "It was a hard day but I kept my head."
        assert len(per_goal) == 2
        assert per_goal[0].goal_text == "deliver bread to Bran"
        assert per_goal[0].status == "stalled"
        assert per_goal[0].note == "Bran was out when I called."
        assert per_goal[1].status == "moving"
        assert next_line == "try Bran's house first thing tomorrow."

    def test_invalid_status_falls_back_to_stalled(self):
        text = (
            "GOAL: something\n"
            "STATUS: dithering\n"  # not in VALID_STATUSES
            "NOTE: .\n"
        )
        per_goal, _, _ = _parse_review_response(text, [], [])
        assert per_goal[0].status == "stalled"

    def test_missing_status_defaults_to_stalled(self):
        text = "GOAL: something\nNOTE: forgot.\n"
        per_goal, _, _ = _parse_review_response(text, [], [])
        assert len(per_goal) == 1
        assert per_goal[0].status == "stalled"

    def test_no_goal_blocks_synthesises_stalled_from_sources(self):
        """If the LLM dropped the structured block entirely, every
        known commitment/long-term goal still gets a stalled entry
        so downstream code always sees per-goal data."""
        commitments = [
            EpisodicMemory(
                memory_id="m1",
                description="I promised to mend the fence.",
                category="commitment",
            ),
        ]
        long_term = ["finish the novel"]
        per_goal, _, _ = _parse_review_response(
            "SUMMARY: tired.\nNEXT: NO_ACTION", commitments, long_term,
        )
        assert len(per_goal) == 2
        assert all(g.status == "stalled" for g in per_goal)

    def test_case_insensitive_keys(self):
        text = "summary: ok.\ngoal: a\nstatus: done\nnote: .\n"
        per_goal, summary, _ = _parse_review_response(text, [], [])
        assert summary == "ok."
        assert per_goal[0].status == "done"

    def test_valid_statuses_covered(self):
        assert {"moving", "stalled", "abandoned", "done"} == VALID_STATUSES


# ---------- Heuristic fallback ----------


class TestFallbackReview:
    def test_empty_inputs_yields_nothing_moved_line(self):
        per_goal, summary, next_line = _fallback_review(
            "Seren", 3, [], [],
        )
        assert per_goal == []
        assert "Day 3" in summary
        assert next_line == ""

    def test_commitments_marked_stalled(self):
        commitments = [
            EpisodicMemory(
                memory_id=f"m{i}",
                description=f"I promised to do thing {i}.",
                category="commitment",
            )
            for i in range(2)
        ]
        per_goal, summary, _ = _fallback_review(
            "Seren", 0, commitments, [],
        )
        assert len(per_goal) == 2
        assert all(g.status == "stalled" for g in per_goal)

    def test_long_term_goals_marked_stalled(self):
        per_goal, _, _ = _fallback_review(
            "Seren", 0, [], ["finish the novel", "befriend the baker"],
        )
        assert len(per_goal) == 2
        assert all(g.status == "stalled" for g in per_goal)


# ---------- Commitment lookup ----------


class TestUnresolvedSelfCommitments:
    def test_returns_only_unresolved(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "bake for market", unresolved=True)
        _seed_commitment(mgr, "seren", "old resolved", unresolved=False)
        hits = _unresolved_self_commitments(mgr, "seren", limit=6)
        subjects = [m.description for m in hits]
        assert any("market" in s for s in subjects)
        assert not any("old resolved" in s for s in subjects)

    def test_scoped_per_npc(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "mine")
        _seed_commitment(mgr, "bran", "yours")
        seren_hits = _unresolved_self_commitments(mgr, "seren", limit=6)
        assert len(seren_hits) == 1
        assert seren_hits[0].description == "I promised to mine."

    def test_limit_respected(self):
        mgr = _mgr()
        for i in range(10):
            _seed_commitment(
                mgr, "seren", f"task {i}",
                game_time=60.0 + i,  # stagger so recency order is stable
            )
        hits = _unresolved_self_commitments(mgr, "seren", limit=3)
        assert len(hits) == 3

    def test_exposed_via_memory_manager(self):
        """`MemoryManager.retrieve_self_commitments` is the public
        passthrough diagnostic panels are expected to use."""
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "bake for market")
        hits = mgr.retrieve_self_commitments("seren")
        assert len(hits) == 1


# ---------- Day-summary lookup ----------


class TestLatestDaySummary:
    def test_returns_the_days_summary(self):
        mgr = _mgr()
        _seed_day_summary(mgr, "seren", day=0, text="short day.")
        hit = _latest_day_summary(mgr, "seren", 0)
        assert hit is not None
        assert hit.description == "short day."

    def test_returns_none_when_absent(self):
        mgr = _mgr()
        assert _latest_day_summary(mgr, "seren", 0) is None

    def test_ignores_other_days(self):
        mgr = _mgr()
        _seed_day_summary(mgr, "seren", day=0, text="old.")
        _seed_day_summary(mgr, "seren", day=1, text="new.")
        hit = _latest_day_summary(mgr, "seren", 1)
        assert hit is not None
        assert hit.description == "new."


# ---------- End-to-end review ----------


class TestDailySelfReviewFallback:
    def test_returns_none_on_fully_blank_slate(self):
        mgr = _mgr()
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc()),
        )
        assert result is None

    def test_writes_memory_without_llm(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "bake for the market")
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=None),
        )
        assert isinstance(result, SelfReviewResult)
        assert result.memory_id
        stored = mgr.episodic.get_by_id(result.memory_id)
        assert stored is not None
        assert stored.category == REVIEW_CATEGORY
        assert stored.importance == pytest.approx(REVIEW_IMPORTANCE)
        assert "[stalled]" in stored.description

    def test_long_term_goals_appear(self):
        mgr = _mgr()
        npc = _npc(long_term_goals=["finish the novel"])
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=npc, llm=None),
        )
        assert result is not None
        assert any(
            "novel" in g.goal_text.lower() for g in result.per_goal
        )


class TestDailySelfReviewLLM:
    def test_happy_path_parses_structured_response(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "deliver bread to Bran")
        llm = _StubLLM({
            "self_review": (
                "SUMMARY: Bread didn't reach Bran today.\n"
                "GOAL: deliver bread to Bran\n"
                "STATUS: stalled\n"
                "NOTE: he was out when I called.\n"
                "NEXT: NO_ACTION\n"
            ),
        })
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=llm),
        )
        assert result is not None
        assert result.summary_text.startswith("Bread didn't")
        assert result.per_goal[0].status == "stalled"
        assert result.action_intent is None
        # Exactly one self_review call — not multiple retries.
        assert sum(1 for c in llm.calls if c["purpose"] == "self_review") == 1

    def test_day_summary_threaded_into_prompt(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "bake bread")
        _seed_day_summary(
            mgr, "seren", day=0,
            text="I burned the loaves and went to bed early.",
        )
        llm = _StubLLM({"self_review": "SUMMARY: rough.\nNEXT: NO_ACTION"})
        asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=llm),
        )
        prompt_text = llm.calls[0]["messages"][0]["content"]
        assert "burned the loaves" in prompt_text

    def test_empty_llm_response_falls_back(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "bake bread")
        llm = _StubLLM({"self_review": ""})
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=llm),
        )
        # Fallback path populates a stalled entry per commitment.
        assert result is not None
        assert any(g.status == "stalled" for g in result.per_goal)

    def test_action_intent_extracted(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "deliver bread to Bran")
        llm = _StubLLM({
            "self_review": (
                "SUMMARY: bread still in the larder.\n"
                "GOAL: deliver bread to Bran\n"
                "STATUS: stalled\n"
                "NOTE: he was out when I called.\n"
                "NEXT: visit Bran's house first thing to deliver bread.\n"
            ),
            "reflection": (
                "ACTION: visit Bran's house to deliver bread\n"
                "LOCATION: bran's house\n"
                "DURATION: 30\n"
            ),
        })
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=llm),
        )
        assert result is not None
        assert result.action_intent is not None
        assert "bread" in result.action_intent.activity.lower()


class TestReviewMemoryShape:
    def test_tags_union_from_source_commitments(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "deliver bread", tags={"bread"})
        _seed_commitment(
            mgr, "seren", "see the healer", tags={"healer", "bran"},
        )
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=None),
        )
        assert result is not None
        assert set(result.kept_tags) == {"bread", "healer", "bran"}
        stored = mgr.episodic.get_by_id(result.memory_id)
        assert stored is not None
        assert {"bread", "healer", "bran"} <= stored.tags

    def test_metadata_records_source_count_and_day(self):
        mgr = _mgr()
        for i in range(3):
            _seed_commitment(mgr, "seren", f"task {i}", game_time=60.0 + i)
        result = asyncio.run(
            daily_self_review(mgr, "seren", 4, npc=_npc(), llm=None),
        )
        assert result is not None
        stored = mgr.episodic.get_by_id(result.memory_id)
        assert stored is not None
        assert stored.metadata["day"] == 4
        assert stored.metadata["source_count"] == 3
        # status_counts is space-delimited key=value pairs; stalled
        # should appear because fallback marks everything stalled.
        assert "stalled=3" in stored.metadata["status_counts"]

    def test_review_time_at_days_last_second(self):
        mgr = _mgr()
        _seed_commitment(mgr, "seren", "do a thing")
        result = asyncio.run(
            daily_self_review(mgr, "seren", 2, npc=_npc(), llm=None),
        )
        assert result is not None
        stored = mgr.episodic.get_by_id(result.memory_id)
        expected = (2 + 1) * MINUTES_PER_DAY - 1.0
        assert stored.game_time == pytest.approx(expected)


# ---------- Phase-H preservation contract ----------


class TestCommitmentReviewSurvivesCompaction:
    def test_is_preserved_category(self):
        """Compaction MUST leave commitment_review memories alone;
        otherwise the bedtime review would be gone by the next day."""
        assert REVIEW_CATEGORY in PRESERVED_CATEGORIES

    def test_review_not_compacted_by_next_day(self):
        """Seed a review on day 0, run day-0 compaction, confirm the
        review still exists and is NOT tombstoned."""
        from core.memory.compaction import compact_day

        mgr = _mgr()
        _seed_commitment(mgr, "seren", "bake bread")
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=None),
        )
        assert result is not None
        # Run day-0 compaction; the review lives inside day 0's window.
        asyncio.run(compact_day(mgr, "seren", 0, npc=_npc()))
        still_there = mgr.episodic.get_by_id(result.memory_id)
        assert still_there is not None
        # Not tombstoned into the day_summary.
        assert not (still_there.metadata or {}).get("compacted_into")
