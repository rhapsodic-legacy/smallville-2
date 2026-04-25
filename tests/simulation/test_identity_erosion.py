"""
Phase I.5 — end-to-end: a persistently stalled commitment eventually
erodes the NPC's self_concept.

Drives the full NPCManager through a ~22 game-day sim with the
bedtime review forced to DETERMINISTIC (heuristic fallback marks
everything stalled). Asserts:

1. The NPC's subject-matched self_concept key loses exactly one
   `IDENTITY_DELTA` worth of confidence (a single crossing event,
   not repeated deltas per subsequent day).
2. A `reflection` memory describing the erosion exists, tagged
   with the source commitment's tags so Phase K retrieval surfaces
   it alongside the original commitment.
3. The source commitment carries `identity_eroded=True` so a later
   run through the same sim won't double-charge.
"""

from __future__ import annotations

import asyncio
import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import (
    IDENTITY_DELTA, IDENTITY_ERODED_FLAG, STAGNATION_METADATA_KEY,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.router import CognitionRouter
from core.npc.cognition.router.policy import (
    CognitionPolicy, ROUTE_DETERMINISTIC,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world


def _make_sim(seed: int = 1901) -> tuple[NPCManager, GameClock]:
    config = WorldConfig(population=3, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
        llm=llm,
    )
    memory.initialise()
    policy = CognitionPolicy()
    policy.set_mode("self_review", ROUTE_DETERMINISTIC)
    router = CognitionRouter(policy=policy)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed,
        memory=memory, router=router,
    )
    mgr.spawn_population(3)
    return mgr, GameClock()


async def _run_days(
    mgr: NPCManager, clock: GameClock, num_days: int,
) -> None:
    real_delta = 8.0
    game_minutes_per_tick = 9.6
    total = num_days * MINUTES_PER_DAY
    ticks = int(total / game_minutes_per_tick) + 1
    for _ in range(ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)


@pytest.mark.timeout(300)
def test_long_stalled_commitment_erodes_matching_identity() -> None:
    async def _run():
        mgr, clock = _make_sim()
        npc = mgr.npcs[0]
        # Seed a strongly-held identity keyed on "bridge" and the
        # matching unresolved commitment.
        npc.self_concept["helped:bridge"] = 0.8
        stalled_id = mgr.memory.episodic.add_memory(
            npc_id=npc.npc_id,
            description="I promised to repair the bridge.",
            category="commitment",
            importance=0.75,
            game_time=120.0,
            tags={"bridge"},
            extra_metadata={
                "outcome_kind": "commitment",
                "source_speaker": npc.npc_id,
                "unresolved": True,
            },
        )

        # ~22 days gives 21 bedtime reviews; crossing fires on
        # review 20 (counter reaches STAGNATION_IDENTITY_THRESHOLD).
        await _run_days(mgr, clock, num_days=22)

        # Confidence dropped by exactly one delta. Floating-point
        # tolerance because the Big-5 drift system writes to this
        # dict each tick via personality decay — we assert against
        # the baseline minus IDENTITY_DELTA, not exact equality.
        remaining = npc.self_concept.get("helped:bridge", 0.0)
        assert remaining == pytest.approx(0.8 - IDENTITY_DELTA, abs=0.02), (
            f"expected ~{0.8 - IDENTITY_DELTA}, got {remaining}"
        )

        # Commitment marked eroded so repeat runs are no-ops.
        stalled = mgr.memory.episodic.get_by_id(stalled_id)
        assert stalled is not None
        assert (stalled.metadata or {}).get(IDENTITY_ERODED_FLAG) is True
        # And the counter itself climbed well past the threshold.
        days = int(stalled.metadata.get(STAGNATION_METADATA_KEY, 0))
        assert days >= 20

        # A reflection memory describing the erosion exists,
        # tagged back to the source commitment's `bridge` tag so
        # Phase K retrieval picks it up.
        reflections = [
            m for m in mgr.memory.episodic.get_recent(
                npc.npc_id, limit=200,
            )
            if m.category == "reflection"
            and (m.metadata or {}).get("outcome_kind") == "identity_erosion"
        ]
        assert reflections, (
            "no identity_erosion reflection memory written"
        )
        assert reflections[0].metadata.get("source_commitment_id") == stalled_id
        assert "bridge" in reflections[0].tags

    asyncio.run(_run())


@pytest.mark.timeout(300)
def test_erosion_fires_at_most_once_per_commitment() -> None:
    """Run far past the threshold (35 days) and assert only ONE
    erosion reflection exists for the commitment — the identity_eroded
    flag must prevent day-by-day re-firing."""
    async def _run():
        mgr, clock = _make_sim(seed=1903)
        npc = mgr.npcs[0]
        npc.self_concept["helped:bridge"] = 0.8
        stalled_id = mgr.memory.episodic.add_memory(
            npc_id=npc.npc_id,
            description="I promised to repair the bridge.",
            category="commitment",
            importance=0.75,
            game_time=120.0,
            tags={"bridge"},
            extra_metadata={
                "outcome_kind": "commitment",
                "source_speaker": npc.npc_id,
                "unresolved": True,
            },
        )

        await _run_days(mgr, clock, num_days=35)

        reflections = [
            m for m in mgr.memory.episodic.get_recent(
                npc.npc_id, limit=500,
            )
            if m.category == "reflection"
            and (m.metadata or {}).get("outcome_kind") == "identity_erosion"
            and (m.metadata or {}).get("source_commitment_id") == stalled_id
        ]
        assert len(reflections) == 1, (
            f"expected exactly 1 erosion reflection, got {len(reflections)}"
        )

    asyncio.run(_run())
