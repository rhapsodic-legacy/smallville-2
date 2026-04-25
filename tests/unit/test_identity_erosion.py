"""
Phase I.5 — soft identity erosion on long-stagnated commitments.

When a commitment's `stagnation_days` counter crosses
`STAGNATION_IDENTITY_THRESHOLD` (20) for the first time, the NPC
internalises the failure: a matching self_concept key (e.g.
`builder_of:bridge` for a commitment about the bridge) drops by
`IDENTITY_DELTA`, or if no subject match exists, `unreliable:self`
is introduced/strengthened by the same magnitude. Fires exactly
once per commitment via the `identity_eroded` metadata flag.

Covers:
- Subject-match token scanning (tokenisation, min-length gate,
  underscore-split of multi-word targets).
- First-crossing fires; repeat days are no-ops.
- Fallback to `unreliable:self` when no subject match exists, with
  positive delta so a fresh belief accumulates via repeated failures.
- Reflection memory is written, tagged with source tags, and
  metadata names the source commitment.
- Below-threshold stagnations don't fire.
- Review memory surfaces an `identity_erosions` metadata field.
- End-to-end through `daily_self_review` across 21 days of
  continuous stalling.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.memory import self_review
from core.memory.episodic import EpisodicMemory, EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import (
    IDENTITY_DELTA, IDENTITY_ERODED_FLAG, IDENTITY_FALLBACK_KEY,
    STAGNATION_IDENTITY_THRESHOLD, STAGNATION_METADATA_KEY,
    GoalProgress, IdentityErosionEvent, SelfReviewResult,
    _apply_identity_erosion, _apply_stagnation_updates,
    _find_matching_self_concept_key, _tokenise_subject,
    daily_self_review,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.models import NPC, PersonalityTraits


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
    name: str = "Seren", self_concept: dict[str, float] | None = None,
) -> NPC:
    return NPC(
        npc_id=name.lower(), name=name, age=30, occupation="baker",
        backstory="", personality=PersonalityTraits(),
        self_concept=dict(self_concept or {}),
        cognition_tier=1,
    )


def _seed_commitment(
    mgr: MemoryManager, npc_id: str, subject: str,
    *, stagnation_days: int | None = None,
    identity_eroded: bool = False,
    tags: set[str] | None = None,
    game_time: float = 60.0,
) -> str:
    extra: dict[str, Any] = {
        "outcome_kind": "commitment",
        "source_speaker": npc_id,
        "unresolved": True,
    }
    if stagnation_days is not None:
        extra[STAGNATION_METADATA_KEY] = stagnation_days
    if identity_eroded:
        extra[IDENTITY_ERODED_FLAG] = True
    return mgr.episodic.add_memory(
        npc_id=npc_id,
        description=f"I promised to {subject}.",
        category="commitment",
        importance=0.75,
        game_time=game_time,
        tags=tags or set(),
        extra_metadata=extra,
    )


# ---------- Tokenisation ----------


class TestTokeniseSubject:
    def test_drops_short_words(self):
        tokens = _tokenise_subject("go to the bridge")
        assert "go" not in tokens  # 2 chars
        assert "to" not in tokens
        assert "the" not in tokens
        assert "bridge" in tokens

    def test_dedupes_case_insensitive(self):
        tokens = _tokenise_subject("Bridge bridge BRIDGE")
        assert tokens.count("bridge") == 1

    def test_handles_punctuation(self):
        tokens = _tokenise_subject("repair the bridge, again!")
        assert "bridge" in tokens
        assert "again" in tokens
        assert "," not in "".join(tokens)

    def test_preserves_insertion_order(self):
        tokens = _tokenise_subject("bridge roof bread")
        assert tokens == ["bridge", "roof", "bread"]

    def test_empty_text_returns_empty(self):
        assert _tokenise_subject("") == []
        assert _tokenise_subject("   ") == []


# ---------- Subject-match key derivation ----------


class TestFindMatchingSelfConceptKey:
    def test_returns_matching_key(self):
        npc = _npc(self_concept={"helped:bridge": 0.8})
        key = _find_matching_self_concept_key(npc, "I promised to repair the bridge.")
        assert key == "helped:bridge"

    def test_returns_none_without_match(self):
        npc = _npc(self_concept={"role:baker": 0.8})
        key = _find_matching_self_concept_key(npc, "I promised to repair the bridge.")
        assert key is None

    def test_prefers_highest_confidence(self):
        npc = _npc(self_concept={
            "helped:bridge": 0.4,
            "built:bridge": 0.9,
        })
        key = _find_matching_self_concept_key(npc, "repair the bridge")
        assert key == "built:bridge"

    def test_underscore_split_targets(self):
        """A key like `helped:south_field` should match a commitment
        that mentions `south` or `field`."""
        npc = _npc(self_concept={"helped:south_field": 0.8})
        key = _find_matching_self_concept_key(npc, "check the south crops")
        assert key == "helped:south_field"

    def test_empty_self_concept_returns_none(self):
        npc = _npc()
        assert _find_matching_self_concept_key(npc, "anything") is None

    def test_empty_description_returns_none(self):
        npc = _npc(self_concept={"helped:bridge": 0.8})
        assert _find_matching_self_concept_key(npc, "") is None


# ---------- Erosion firing ----------


class TestApplyIdentityErosion:
    def test_below_threshold_no_event(self):
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(mgr, npc.npc_id, "repair the bridge")
        commitment = mgr.episodic.get_by_id(cid)
        # Updates dict mimics what `_apply_stagnation_updates`
        # would have written — one day shy of threshold.
        updates = {cid: STAGNATION_IDENTITY_THRESHOLD - 1}
        events = asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment], updates, game_time=1000.0,
        ))
        assert events == []
        # self_concept untouched.
        assert npc.self_concept["helped:bridge"] == 0.8

    def test_crossing_threshold_fires_once(self):
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(mgr, npc.npc_id, "repair the bridge")
        commitment = mgr.episodic.get_by_id(cid)
        updates = {cid: STAGNATION_IDENTITY_THRESHOLD}
        events = asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment], updates, game_time=1000.0,
        ))
        assert len(events) == 1
        evt = events[0]
        assert evt.commitment_id == cid
        assert evt.self_concept_key == "helped:bridge"
        assert evt.delta == pytest.approx(-IDENTITY_DELTA)
        assert npc.self_concept["helped:bridge"] == pytest.approx(0.7)

    def test_second_call_is_noop_via_flag(self):
        """After the first fire, the commitment carries
        `identity_eroded=True`; a second invocation at an even-
        higher day count must not emit another delta."""
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(mgr, npc.npc_id, "repair the bridge")

        # First fire.
        commitment = mgr.episodic.get_by_id(cid)
        asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment],
            {cid: STAGNATION_IDENTITY_THRESHOLD}, game_time=1000.0,
        ))
        first_confidence = npc.self_concept["helped:bridge"]

        # Second fire (should no-op).
        commitment = mgr.episodic.get_by_id(cid)
        events = asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment],
            {cid: STAGNATION_IDENTITY_THRESHOLD + 5},
            game_time=2000.0,
        ))
        assert events == []
        assert npc.self_concept["helped:bridge"] == first_confidence

    def test_fallback_to_unreliable_self(self):
        """No self_concept key matches 'repair the bridge' → fallback
        introduces `unreliable:self` at +IDENTITY_DELTA."""
        mgr = _mgr()
        npc = _npc(self_concept={"role:baker": 0.8})
        cid = _seed_commitment(mgr, npc.npc_id, "repair the bridge")
        commitment = mgr.episodic.get_by_id(cid)

        events = asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment],
            {cid: STAGNATION_IDENTITY_THRESHOLD}, game_time=1000.0,
        ))
        assert len(events) == 1
        assert events[0].self_concept_key == IDENTITY_FALLBACK_KEY
        assert events[0].delta == pytest.approx(+IDENTITY_DELTA)
        # Fresh key introduced at 0.1 (floor is 0.05).
        assert npc.self_concept[IDENTITY_FALLBACK_KEY] == pytest.approx(0.1)

    def test_repeated_fallback_accumulates(self):
        """Three independent commitments that all hit the threshold
        without subject match strengthen `unreliable:self` progressively."""
        mgr = _mgr()
        npc = _npc(self_concept={"role:baker": 0.8})

        for i in range(3):
            cid = _seed_commitment(
                mgr, npc.npc_id, f"do task {i}",
                game_time=60.0 + i,
            )
            commitment = mgr.episodic.get_by_id(cid)
            asyncio.run(_apply_identity_erosion(
                mgr, npc, [commitment],
                {cid: STAGNATION_IDENTITY_THRESHOLD + i},
                game_time=1000.0 + i,
            ))

        assert npc.self_concept[IDENTITY_FALLBACK_KEY] == pytest.approx(0.3)

    def test_missing_from_updates_is_skipped(self):
        """A commitment not present in stagnation_updates (moving or
        abandoned verdict that left the counter unchanged) doesn't
        get a crossing check."""
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(
            mgr, npc.npc_id, "repair the bridge", stagnation_days=50,
        )
        commitment = mgr.episodic.get_by_id(cid)
        events = asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment], {}, game_time=1000.0,
        ))
        assert events == []

    def test_no_npc_is_noop(self):
        """When the caller can't provide an NPC dataclass (some
        tests/tools), the helper short-circuits without error."""
        mgr = _mgr()
        cid = _seed_commitment(mgr, "seren", "anything")
        commitment = mgr.episodic.get_by_id(cid)
        events = asyncio.run(_apply_identity_erosion(
            mgr, None, [commitment],
            {cid: STAGNATION_IDENTITY_THRESHOLD}, game_time=1000.0,
        ))
        assert events == []

    def test_reflection_memory_written_with_source_tags(self):
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(
            mgr, npc.npc_id, "repair the bridge",
            tags={"bridge", "town"},
        )
        commitment = mgr.episodic.get_by_id(cid)

        events = asyncio.run(_apply_identity_erosion(
            mgr, npc, [commitment],
            {cid: STAGNATION_IDENTITY_THRESHOLD}, game_time=1000.0,
        ))
        assert events[0].reflection_memory_id
        ref = mgr.episodic.get_by_id(events[0].reflection_memory_id)
        assert ref is not None
        assert ref.category == "reflection"
        assert "bridge" in ref.tags
        assert "town" in ref.tags
        # Metadata preserves the provenance pointer.
        assert ref.metadata.get("source_commitment_id") == cid
        assert ref.metadata.get("self_concept_key") == "helped:bridge"


# ---------- End-to-end through daily_self_review ----------


class TestDailySelfReviewIdentityErosion:
    def test_21_days_stalling_triggers_erosion(self):
        """Drive the full `daily_self_review` function across 21
        fallback reviews. At day 20 the counter crosses the
        threshold and identity erosion fires; subsequent days
        don't re-fire."""
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(
            mgr, npc.npc_id, "repair the bridge", tags={"bridge"},
        )
        erosion_day: int | None = None
        erosion_count = 0
        for day in range(21):
            result = asyncio.run(
                daily_self_review(mgr, npc.npc_id, day, npc=npc, llm=None),
            )
            if result and result.identity_erosions:
                erosion_count += 1
                if erosion_day is None:
                    erosion_day = day

        # Crossed at day 19 (0-indexed): stagnation_days 20 after
        # the 20th review. Assertion is inclusive of off-by-one
        # variance in the daily loop.
        assert erosion_day is not None
        assert erosion_day >= STAGNATION_IDENTITY_THRESHOLD - 1
        assert erosion_count == 1
        assert npc.self_concept["helped:bridge"] == pytest.approx(0.7)

    def test_review_metadata_records_erosion(self):
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        cid = _seed_commitment(
            mgr, npc.npc_id, "repair the bridge",
            stagnation_days=STAGNATION_IDENTITY_THRESHOLD - 1,
        )
        result = asyncio.run(
            daily_self_review(mgr, npc.npc_id, 5, npc=npc, llm=None),
        )
        assert result is not None
        assert len(result.identity_erosions) == 1
        review = mgr.episodic.get_by_id(result.memory_id)
        assert cid in (review.metadata.get("identity_erosions") or "")
        assert "helped:bridge" in (
            review.metadata.get("identity_erosions") or ""
        )

    def test_prior_flag_prevents_fresh_fire(self):
        """A commitment already flagged `identity_eroded=True`
        doesn't re-fire even when the counter re-crosses after being
        loaded into a new run."""
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        _seed_commitment(
            mgr, npc.npc_id, "repair the bridge",
            stagnation_days=STAGNATION_IDENTITY_THRESHOLD - 1,
            identity_eroded=True,
        )
        result = asyncio.run(
            daily_self_review(mgr, npc.npc_id, 0, npc=npc, llm=None),
        )
        assert result is not None
        assert result.identity_erosions == []
        assert npc.self_concept["helped:bridge"] == 0.8
