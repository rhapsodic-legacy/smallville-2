"""
Phase I.6 — multi-day identity arc showcase.

Two sim tests braided around the I.3/I.5/I.4 feedback loop:

Test A ties vocalness and erosion together: across 22+ bedtime
reviews, a stalled commitment's composite retrieval score (the proxy
for "vocalness" — where it lands in the NPC's prompt context when
talking to the relevant partner) climbs day-over-day, saturates at
the I.3 retrieval cap, and then on the first day the counter
crosses the I.5 identity threshold, one erosion event fires and the
matching self_concept belief drops by exactly one delta. No
re-fires on subsequent days.

Test B covers the stagnated-then-abandoned path: once a commitment
has already erosion-fired past threshold, a later `abandoned`
verdict freezes the counter and does NOT trigger a second erosion.
The `identity_eroded` flag on the source commitment is the only
gate — abandonment doesn't bypass it.

Scope note (strict interpretation, Phase I.6):
  A commitment that's *abandoned before reaching the stagnation
  threshold* (say, dropped on day 5) currently incurs zero identity
  cost — the counter never reached 20 and I.5 never fires. That's
  a known narrative hole; see MEMORY_V2_ROADMAP.md Tuning watchlist
  for future-scope discussion (would live in a Phase I.7 or a
  pre-req to Phase J, not here). This test suite only asserts
  strict-interpretation behaviour: abandonment alone is not a
  separate identity signal today.

Completion path is covered end-to-end by
`tests/simulation/test_identity_reinforcement.py` (Phase I.4).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import (
    IDENTITY_DELTA, IDENTITY_ERODED_FLAG,
    STAGNATION_IDENTITY_THRESHOLD, STAGNATION_METADATA_KEY,
    daily_self_review,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import LLMProvider
from core.npc.models import NPC, PersonalityTraits


# ---------- Fixtures ----------


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


def _seed_bridge_commitment(
    mgr: MemoryManager, npc_id: str,
    *, stagnation_days: int | None = None,
    identity_eroded: bool = False,
    game_time: float = 60.0,
) -> str:
    """Seed a partner-named commitment that `retrieve_unresolved_matters`
    will surface when partner_name='petra'."""
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
        description="I promised Petra I would repair the bridge.",
        category="commitment",
        importance=0.75,
        game_time=game_time,
        tags={"bridge"},
        extra_metadata=extra,
    )


def _composite_score(mem: Any) -> float:
    """Replicate the retrieve_unresolved_matters sort key for a single
    memory — importance plus the I.3 stagnation boost."""
    return mem.importance + MemoryManager._stagnation_boost(mem)


# ---------- Test A — vocalness ramp + erosion ----------


@pytest.mark.timeout(120)
def test_stalled_commitment_ramps_in_retrieval_then_erodes() -> None:
    """Drive the full daily_self_review across 23 fallback reviews and
    confirm the I.3 composite score ramps to the cap, plateaus, and
    exactly one I.5 erosion event fires when the counter first
    crosses STAGNATION_IDENTITY_THRESHOLD."""
    async def _run():
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        commitment_id = _seed_bridge_commitment(mgr, npc.npc_id)

        scores: list[float] = []
        erosion_events = []
        for day in range(23):
            result = await daily_self_review(
                mgr, npc.npc_id, day, npc=npc, llm=None,
            )
            if result and result.identity_erosions:
                erosion_events.extend(result.identity_erosions)

            # Proxy (c) for "vocalness" — the composite retrieval
            # score a partner-relevant lookup would rank this
            # commitment by. Same sort key `retrieve_unresolved_
            # matters` uses internally.
            matters = mgr.retrieve_unresolved_matters(
                npc.npc_id, partner_name="petra", limit=5,
            )
            bridge_matter = next(
                (m for m in matters if "bridge" in (m.description or "").lower()),
                None,
            )
            assert bridge_matter is not None, (
                f"day {day}: bridge commitment missing from retrieval "
                f"(got {[m.description for m in matters]})"
            )
            scores.append(_composite_score(bridge_matter))

        # --- Ramp: composite score climbs strictly through the cap.
        # Day N's score reflects counter N+1 (review increments the
        # counter first, retrieval reads after). So scores[0] = 0.79,
        # scores[14] = 0.75 + 15*0.04 = 1.35 (exactly at cap).
        for i in range(1, 15):
            assert scores[i] > scores[i - 1], (
                f"day {i}: expected score to climb from {scores[i - 1]}, "
                f"got {scores[i]}"
            )

        # --- Plateau: once counter passes STAGNATION_BOOST_CAP (15),
        # the boost is clamped and score flatlines.
        cap_score = 0.75 + 15 * MemoryManager.STAGNATION_BOOST_PER_DAY
        assert scores[14] == pytest.approx(cap_score)
        for i in range(15, 23):
            assert scores[i] == pytest.approx(cap_score), (
                f"day {i}: expected plateau at {cap_score}, got {scores[i]}"
            )

        # --- Exactly one erosion, fired at the threshold crossing.
        assert len(erosion_events) == 1, (
            f"expected one erosion event, got {len(erosion_events)}"
        )
        evt = erosion_events[0]
        assert evt.commitment_id == commitment_id
        assert evt.self_concept_key == "helped:bridge"
        assert evt.delta == pytest.approx(-IDENTITY_DELTA)

        # --- Belief dropped by exactly one delta.
        assert npc.self_concept["helped:bridge"] == pytest.approx(0.7)

        # --- Provenance: source commitment carries the flag; a
        # reflection memory describing the erosion exists and is
        # tagged with the source commitment's `bridge` tag.
        stalled = mgr.episodic.get_by_id(commitment_id)
        assert (stalled.metadata or {}).get(IDENTITY_ERODED_FLAG) is True

        reflections = [
            m for m in mgr.episodic.get_recent(npc.npc_id, limit=200)
            if m.category == "reflection"
            and (m.metadata or {}).get("outcome_kind") == "identity_erosion"
        ]
        assert reflections, "no identity_erosion reflection written"
        assert "bridge" in reflections[0].tags

    asyncio.run(_run())


# ---------- Test B — stagnated then abandoned, strict path ----------


class _AbandonedVerdictProvider(LLMProvider):
    """Minimal stub: every self_review call returns a block-format
    response that marks the first commitment as `abandoned`.

    Only used to drive the one bedtime pass in Test B that needs an
    abandonment verdict. Every other LLM purpose raises — this
    provider is narrowly scoped and shouldn't be reused."""

    async def complete(
        self, system: str, messages: list[dict[str, str]],
        max_tokens: int = 300, temperature: float = 0.7,
        purpose: str = "general", npc_id: str | None = None,
    ) -> str:
        assert purpose == "self_review", (
            f"_AbandonedVerdictProvider called with unexpected purpose "
            f"{purpose!r}"
        )
        return (
            "SUMMARY: I have stopped pretending I'll see this through.\n"
            "GOAL: repair the bridge\n"
            "STATUS: abandoned\n"
            "NOTE: I'm letting it go.\n"
            "NEXT: NO_ACTION\n"
        )


@pytest.mark.timeout(60)
def test_stagnated_then_abandoned_fires_exactly_once() -> None:
    """Commitment stagnates past threshold → I.5 fires on the crossing
    day. On a later day the NPC abandons it → counter freezes, no
    second erosion. Subsequent stalled days also don't re-fire
    (identity_eroded flag is the sole gate)."""
    async def _run():
        mgr = _mgr()
        npc = _npc(self_concept={"helped:bridge": 0.8})
        # Pre-seed the counter at 19 so the very first stalled review
        # crosses the threshold. Keeps the test short without
        # sacrificing the "stagnated first, then abandoned" shape.
        commitment_id = _seed_bridge_commitment(
            mgr, npc.npc_id,
            stagnation_days=STAGNATION_IDENTITY_THRESHOLD - 1,
        )

        # Day 0 — fallback (stalled) review. Counter 19 → 20 → crosses
        # threshold → erosion fires.
        r0 = await daily_self_review(mgr, npc.npc_id, 0, npc=npc, llm=None)
        assert r0 is not None
        assert len(r0.identity_erosions) == 1

        belief_after_erosion = npc.self_concept["helped:bridge"]
        assert belief_after_erosion == pytest.approx(0.7)

        stalled = mgr.episodic.get_by_id(commitment_id)
        assert (stalled.metadata or {}).get(IDENTITY_ERODED_FLAG) is True
        counter_after_erosion = int(
            stalled.metadata.get(STAGNATION_METADATA_KEY, 0),
        )
        assert counter_after_erosion == STAGNATION_IDENTITY_THRESHOLD

        # Day 1 — NPC abandons. Counter must freeze (not increment),
        # and the existing flag must prevent a second erosion event.
        r1 = await daily_self_review(
            mgr, npc.npc_id, 1, npc=npc, llm=_AbandonedVerdictProvider(),
        )
        assert r1 is not None
        assert r1.identity_erosions == [], (
            "abandonment re-fired erosion despite identity_eroded flag"
        )

        stalled = mgr.episodic.get_by_id(commitment_id)
        counter_after_abandon = int(
            (stalled.metadata or {}).get(STAGNATION_METADATA_KEY, 0),
        )
        assert counter_after_abandon == counter_after_erosion, (
            f"abandonment should freeze counter at {counter_after_erosion}, "
            f"got {counter_after_abandon}"
        )
        assert npc.self_concept["helped:bridge"] == pytest.approx(
            belief_after_erosion,
        ), "self_concept moved on abandonment day — expected no delta"

        # Days 2-5 — subsequent stalled reviews must still not re-fire.
        # The flag is the single gate; the counter continuing to
        # climb under stalled verdicts is fine as long as identity
        # stays untouched.
        for day in range(2, 6):
            r = await daily_self_review(
                mgr, npc.npc_id, day, npc=npc, llm=None,
            )
            assert r is not None
            assert r.identity_erosions == [], (
                f"day {day}: erosion re-fired after identity_eroded flag set"
            )

        assert npc.self_concept["helped:bridge"] == pytest.approx(
            belief_after_erosion,
        )

    asyncio.run(_run())
