"""
Phase I.3 — stagnation escalation on unresolved commitments.

Covers two paired pieces:
- `core.memory.self_review._apply_stagnation_updates` — the counter
  update that runs inside `daily_self_review` per bedtime.
- `core.memory.manager.MemoryManager.retrieve_unresolved_matters` /
  `._stagnation_boost` — the composite-score ranking that lifts
  stale commitments above fresh ones.

The counter itself is unbounded (so Phase I.5 can read it), but
the retrieval boost saturates at `STAGNATION_BOOST_CAP` days. Both
those invariants are asserted here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.memory import self_review
from core.memory.episodic import EpisodicMemory, EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import (
    GoalProgress, STAGNATION_METADATA_KEY, _apply_stagnation_updates,
    daily_self_review,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
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


def _npc(name: str = "Seren") -> NPC:
    return NPC(
        npc_id=name.lower(), name=name, age=30, occupation="baker",
        backstory="", personality=PersonalityTraits(),
        cognition_tier=1,
    )


def _seed_commitment(
    mgr: MemoryManager, npc_id: str, subject: str,
    *, partner_name: str = "", game_time: float = 60.0,
    stagnation_days: int | None = None,
    importance: float = 0.75,
) -> str:
    """Drop a Phase B commitment onto the NPC's own store.

    `partner_name`, when provided, is injected into the description
    so `_matter_names_partner`'s last-resort substring check picks
    it up. Stagnation_days can be pre-seeded to simulate a matter
    that already has a history of stalling.
    """
    desc = f"I promised to {subject}"
    if partner_name:
        desc += f" for {partner_name}"
    desc += "."
    extra: dict[str, Any] = {
        "outcome_kind": "commitment",
        "source_speaker": npc_id,
        "unresolved": True,
    }
    if stagnation_days is not None:
        extra[STAGNATION_METADATA_KEY] = stagnation_days
    return mgr.episodic.add_memory(
        npc_id=npc_id,
        description=desc,
        category="commitment",
        importance=importance,
        game_time=game_time,
        extra_metadata=extra,
    )


# ---------- Counter updates ----------


class TestApplyStagnationUpdates:
    def test_stalled_verdict_increments(self):
        mgr = _mgr()
        cid = _seed_commitment(mgr, "seren", "bake bread")
        commitment = mgr.episodic.get_by_id(cid)
        assert commitment is not None

        updates = _apply_stagnation_updates(
            mgr, [commitment],
            [GoalProgress(goal_text="bake bread", status="stalled")],
        )
        assert updates == {cid: 1}

        reloaded = mgr.episodic.get_by_id(cid)
        assert reloaded.metadata[STAGNATION_METADATA_KEY] == 1

    def test_moving_verdict_resets_to_zero(self):
        mgr = _mgr()
        cid = _seed_commitment(
            mgr, "seren", "bake bread", stagnation_days=7,
        )
        commitment = mgr.episodic.get_by_id(cid)

        updates = _apply_stagnation_updates(
            mgr, [commitment],
            [GoalProgress(goal_text="bake bread", status="moving")],
        )
        assert updates == {cid: 0}

        reloaded = mgr.episodic.get_by_id(cid)
        assert reloaded.metadata[STAGNATION_METADATA_KEY] == 0

    def test_done_verdict_resets_to_zero(self):
        mgr = _mgr()
        cid = _seed_commitment(
            mgr, "seren", "bake bread", stagnation_days=4,
        )
        commitment = mgr.episodic.get_by_id(cid)

        updates = _apply_stagnation_updates(
            mgr, [commitment],
            [GoalProgress(goal_text="bake bread", status="done")],
        )
        assert updates == {cid: 0}

    def test_abandoned_verdict_freezes_counter(self):
        """Abandoned is the conscious drop — the counter stops
        moving so I.5 can read the terminal value."""
        mgr = _mgr()
        cid = _seed_commitment(
            mgr, "seren", "bake bread", stagnation_days=6,
        )
        commitment = mgr.episodic.get_by_id(cid)

        updates = _apply_stagnation_updates(
            mgr, [commitment],
            [GoalProgress(goal_text="bake bread", status="abandoned")],
        )
        # No write emitted; reload confirms counter unchanged.
        assert updates == {}
        reloaded = mgr.episodic.get_by_id(cid)
        assert reloaded.metadata[STAGNATION_METADATA_KEY] == 6

    def test_counter_grows_unbounded(self):
        """I.5 needs the raw counter to keep growing past the
        retrieval cap (15 days) so it can trigger identity deltas
        after prolonged stagnation."""
        mgr = _mgr()
        cid = _seed_commitment(mgr, "seren", "bake bread")
        commitment = mgr.episodic.get_by_id(cid)

        for _ in range(30):
            commitment = mgr.episodic.get_by_id(cid)
            _apply_stagnation_updates(
                mgr, [commitment],
                [GoalProgress(goal_text="bake bread", status="stalled")],
            )

        reloaded = mgr.episodic.get_by_id(cid)
        assert reloaded.metadata[STAGNATION_METADATA_KEY] == 30

    def test_per_commitment_independent(self):
        """Two commitments on the same NPC track independently —
        one can stall while the other moves."""
        mgr = _mgr()
        c1 = _seed_commitment(mgr, "seren", "bake bread")
        c2 = _seed_commitment(mgr, "seren", "mend fence", game_time=120.0)
        commitments = [
            mgr.episodic.get_by_id(c1),
            mgr.episodic.get_by_id(c2),
        ]
        _apply_stagnation_updates(
            mgr, commitments,
            [
                GoalProgress(goal_text="bake bread", status="stalled"),
                GoalProgress(goal_text="mend fence", status="moving"),
            ],
        )
        # c1 incremented; c2 stays at 0 (missing key == 0 functionally).
        assert mgr.episodic.get_by_id(c1).metadata[
            STAGNATION_METADATA_KEY
        ] == 1
        assert mgr.episodic.get_by_id(c2).metadata.get(
            STAGNATION_METADATA_KEY, 0,
        ) == 0

    def test_unmentioned_commitment_treated_as_stalled(self):
        """If the LLM returned fewer per_goal entries than there are
        commitments, the remainder default to stalled — an unspoken
        goal is, functionally, a stagnating one."""
        mgr = _mgr()
        c1 = _seed_commitment(mgr, "seren", "bake bread")
        c2 = _seed_commitment(mgr, "seren", "mend fence", game_time=120.0)
        commitments = [
            mgr.episodic.get_by_id(c1),
            mgr.episodic.get_by_id(c2),
        ]
        _apply_stagnation_updates(
            mgr, commitments,
            [GoalProgress(goal_text="bake bread", status="moving")],
            # c2 has no per_goal entry — treated as stalled.
        )
        # c1 stays at 0 (moving verdict on a zero counter is a no-op
        # write — the key is absent, which is functionally 0).
        assert mgr.episodic.get_by_id(c1).metadata.get(
            STAGNATION_METADATA_KEY, 0,
        ) == 0
        assert mgr.episodic.get_by_id(c2).metadata[
            STAGNATION_METADATA_KEY
        ] == 1

    def test_trailing_per_goal_entries_ignored(self):
        """Extras in per_goal (long-term goals, hallucinations) don't
        crash the positional match and don't affect counters."""
        mgr = _mgr()
        c1 = _seed_commitment(mgr, "seren", "bake bread")
        commitments = [mgr.episodic.get_by_id(c1)]
        _apply_stagnation_updates(
            mgr, commitments,
            [
                GoalProgress(goal_text="bake bread", status="stalled"),
                GoalProgress(goal_text="finish novel", status="moving"),
                GoalProgress(goal_text="befriend baker", status="done"),
            ],
        )
        assert mgr.episodic.get_by_id(c1).metadata[
            STAGNATION_METADATA_KEY
        ] == 1

    def test_empty_commitment_list_noop(self):
        mgr = _mgr()
        assert _apply_stagnation_updates(mgr, [], []) == {}


# ---------- Retrieval boost ----------


class TestStagnationBoost:
    def test_zero_days_no_boost(self):
        mem = EpisodicMemory(category="commitment", metadata={})
        assert MemoryManager._stagnation_boost(mem) == 0.0

    def test_linear_below_cap(self):
        mem = EpisodicMemory(
            category="commitment",
            metadata={STAGNATION_METADATA_KEY: 5},
        )
        expected = 5 * MemoryManager.STAGNATION_BOOST_PER_DAY
        assert MemoryManager._stagnation_boost(mem) == pytest.approx(expected)

    def test_cap_saturates(self):
        """Day 15 and day 60 produce the same boost."""
        mem_15 = EpisodicMemory(
            category="commitment",
            metadata={STAGNATION_METADATA_KEY: 15},
        )
        mem_60 = EpisodicMemory(
            category="commitment",
            metadata={STAGNATION_METADATA_KEY: 60},
        )
        expected = (
            MemoryManager.STAGNATION_BOOST_CAP
            * MemoryManager.STAGNATION_BOOST_PER_DAY
        )
        assert MemoryManager._stagnation_boost(mem_15) == pytest.approx(expected)
        assert MemoryManager._stagnation_boost(mem_60) == pytest.approx(expected)

    def test_non_commitment_gets_no_boost(self):
        """Accusations and relayed_claims don't accumulate stagnation."""
        mem = EpisodicMemory(
            category="accusation",
            metadata={STAGNATION_METADATA_KEY: 10},
        )
        assert MemoryManager._stagnation_boost(mem) == 0.0

    def test_malformed_value_returns_zero(self):
        mem = EpisodicMemory(
            category="commitment",
            metadata={STAGNATION_METADATA_KEY: "seven"},
        )
        assert MemoryManager._stagnation_boost(mem) == 0.0

    def test_missing_metadata_returns_zero(self):
        mem = EpisodicMemory(category="commitment", metadata={})
        assert MemoryManager._stagnation_boost(mem) == 0.0


# ---------- Retrieval ranking end-to-end ----------


class TestRetrieveUnresolvedMattersRanking:
    def test_stale_commitment_outranks_fresh_at_equal_importance(self):
        """A 5-day stalled commitment at importance 0.75 should rank
        above a fresh commitment at importance 0.75."""
        mgr = _mgr()
        fresh_id = _seed_commitment(
            mgr, "seren", "mend fence",
            partner_name="Petra",
            game_time=500.0,
        )
        stale_id = _seed_commitment(
            mgr, "seren", "deliver bread",
            partner_name="Petra",
            game_time=60.0,
            stagnation_days=5,
        )
        matters = mgr.retrieve_unresolved_matters(
            "seren", partner_name="Petra", limit=5,
        )
        ordered_ids = [m.memory_id for m in matters]
        assert ordered_ids.index(stale_id) < ordered_ids.index(fresh_id)

    def test_fresh_critical_beats_short_stalled(self):
        """A fresh accusation at importance 0.95 should still rank
        above a 3-day stalled commitment at importance 0.75 —
        recent critical news isn't eclipsed by minor stagnation."""
        mgr = _mgr()
        accusation_id = mgr.episodic.add_memory(
            npc_id="seren",
            description="Petra accused me of hoarding bread.",
            category="accusation",
            importance=0.95,
            game_time=600.0,
            extra_metadata={
                "outcome_kind": "accusation",
                "accuser": "Petra",
                "accused": "seren",
                "claim": "hoarding bread",
                "unresolved": True,
            },
        )
        stalled_id = _seed_commitment(
            mgr, "seren", "mend the fence",
            partner_name="Petra",
            game_time=60.0,
            stagnation_days=3,
        )
        matters = mgr.retrieve_unresolved_matters(
            "seren", partner_name="Petra", limit=5,
        )
        ordered_ids = [m.memory_id for m in matters]
        assert ordered_ids.index(accusation_id) < ordered_ids.index(stalled_id)

    def test_15_day_stalled_beats_fresh_critical(self):
        """After 15 days stalled, a base-0.75 commitment should
        outrank even a fresh critical accusation at 0.95 — the
        bedtime ritual has become the dominant signal."""
        mgr = _mgr()
        accusation_id = mgr.episodic.add_memory(
            npc_id="seren",
            description="Petra accused me of hoarding bread.",
            category="accusation",
            importance=0.95,
            game_time=600.0,
            extra_metadata={
                "outcome_kind": "accusation",
                "accuser": "Petra",
                "accused": "seren",
                "claim": "hoarding bread",
                "unresolved": True,
            },
        )
        long_stalled_id = _seed_commitment(
            mgr, "seren", "mend the fence",
            partner_name="Petra",
            game_time=60.0,
            stagnation_days=15,
        )
        matters = mgr.retrieve_unresolved_matters(
            "seren", partner_name="Petra", limit=5,
        )
        ordered_ids = [m.memory_id for m in matters]
        assert ordered_ids.index(long_stalled_id) < ordered_ids.index(
            accusation_id
        )

    def test_saturated_items_ordered_by_recency(self):
        """Two commitments both past the cap should resolve the tie
        by recency — the more recently raised one comes first."""
        mgr = _mgr()
        older_id = _seed_commitment(
            mgr, "seren", "fix the roof",
            partner_name="Petra",
            game_time=60.0,
            stagnation_days=40,  # well past cap
        )
        newer_id = _seed_commitment(
            mgr, "seren", "call on the healer",
            partner_name="Petra",
            game_time=200.0,
            stagnation_days=20,  # past cap
        )
        matters = mgr.retrieve_unresolved_matters(
            "seren", partner_name="Petra", limit=5,
        )
        ordered_ids = [m.memory_id for m in matters]
        assert ordered_ids.index(newer_id) < ordered_ids.index(older_id)

    def test_resolved_commitment_drops_out(self):
        """`unresolved=False` still filters the matter regardless of
        how high its stagnation counter ran."""
        mgr = _mgr()
        resolved = _seed_commitment(
            mgr, "seren", "deliver bread",
            partner_name="Petra",
            game_time=60.0,
            stagnation_days=30,
        )
        mgr.episodic.update_metadata(resolved, {"unresolved": False})
        matters = mgr.retrieve_unresolved_matters(
            "seren", partner_name="Petra", limit=5,
        )
        assert all(m.memory_id != resolved for m in matters)


# ---------- End-to-end through daily_self_review ----------


class TestDailySelfReviewUpdatesStagnation:
    def test_fallback_review_increments_stalled_commitment(self):
        """No LLM → fallback marks everything stalled → the
        commitment's stagnation_days should go up by 1."""
        mgr = _mgr()
        cid = _seed_commitment(mgr, "seren", "bake for the market")
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=None),
        )
        assert result is not None
        reloaded = mgr.episodic.get_by_id(cid)
        assert reloaded.metadata[STAGNATION_METADATA_KEY] == 1

    def test_consecutive_reviews_compound(self):
        """Running the review three days in a row should produce
        stagnation_days == 3 on an always-stalled commitment."""
        mgr = _mgr()
        cid = _seed_commitment(mgr, "seren", "bake bread")
        for day in range(3):
            asyncio.run(
                daily_self_review(mgr, "seren", day, npc=_npc(), llm=None),
            )
        assert mgr.episodic.get_by_id(cid).metadata[
            STAGNATION_METADATA_KEY
        ] == 3

    def test_review_snapshot_metadata_records_new_values(self):
        """The review memory's `stagnation_snapshot` field should
        carry the post-update counters so diagnostics can read the
        bedtime state without chasing provenance."""
        mgr = _mgr()
        cid = _seed_commitment(
            mgr, "seren", "bake bread", stagnation_days=4,
        )
        result = asyncio.run(
            daily_self_review(mgr, "seren", 0, npc=_npc(), llm=None),
        )
        assert result is not None
        review = mgr.episodic.get_by_id(result.memory_id)
        snapshot = review.metadata.get("stagnation_snapshot", "")
        # Expect "commitment_id=5" (the post-increment value).
        assert f"{cid}=5" in snapshot
