"""
Phase I.3 — end-to-end: a commitment that never moves becomes
progressively more dominant in `retrieve_unresolved_matters`.

Drives the full NPCManager through a multi-day sim with the
bedtime self-review forced to the DETERMINISTIC router verdict
(so every review marks everything stalled — mirrors the worst-case
"nothing moved" scenario). Asserts:

1. The commitment's `stagnation_days` counter grows each game day.
2. After ~5 bedtime reviews the stalled commitment sits above a
   fresh commitment at the same base importance.
3. The counter keeps growing past the retrieval cap; I.5 reads the
   raw value, so it must not saturate on the write side.
"""

from __future__ import annotations

import asyncio
import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import STAGNATION_METADATA_KEY
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


def _make_sim(seed: int = 1301) -> tuple[NPCManager, GameClock]:
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
    # Force bedtime review to the deterministic fallback so every
    # review ends in `stalled` verdicts across the board. The LLM
    # path (where a response might flip one to `moving`) is covered
    # by `test_self_review.py`; this test is specifically about the
    # stagnation ramp under steady stalling.
    policy = CognitionPolicy()
    policy.set_mode("self_review", ROUTE_DETERMINISTIC)
    router = CognitionRouter(policy=policy)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed,
        memory=memory, router=router,
    )
    mgr.spawn_population(3)
    return mgr, GameClock()


def _seed_unresolved_commitment(
    mgr: NPCManager, npc_id: str, partner_name: str,
    subject: str, game_time: float = 120.0,
) -> str:
    return mgr.memory.episodic.add_memory(
        npc_id=npc_id,
        description=f"I promised to {subject} for {partner_name}.",
        category="commitment",
        importance=0.75,
        game_time=game_time,
        extra_metadata={
            "outcome_kind": "commitment",
            "source_speaker": npc_id,
            "unresolved": True,
        },
    )


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
def test_stagnation_ramps_and_reorders_matters() -> None:
    async def _run():
        mgr, clock = _make_sim()
        npc = mgr.npcs[0]
        stalled_id = _seed_unresolved_commitment(
            mgr, npc.npc_id, "Petra", "deliver bread",
            game_time=120.0,
        )

        # Advance ~6 game-days so bedtime fires for days 0..4.
        await _run_days(mgr, clock, num_days=6)

        # --- Counter climbed ---
        stalled = mgr.memory.episodic.get_by_id(stalled_id)
        assert stalled is not None
        days = int(stalled.metadata.get(STAGNATION_METADATA_KEY, 0))
        assert days >= 4, (
            f"stagnation_days={days} after ~6 days of stalling; "
            "expected at least 4 bedtime increments"
        )

        # --- Add a FRESH commitment on the same partner at equal
        # base importance — the stale one should rank above it. ---
        fresh_id = _seed_unresolved_commitment(
            mgr, npc.npc_id, "Petra", "fetch water",
            game_time=mgr._current_minutes,
        )
        matters = mgr.memory.retrieve_unresolved_matters(
            npc.npc_id, partner_name="Petra", limit=5,
        )
        ordered_ids = [m.memory_id for m in matters]
        assert stalled_id in ordered_ids
        assert fresh_id in ordered_ids
        assert ordered_ids.index(stalled_id) < ordered_ids.index(fresh_id), (
            "stale commitment did NOT outrank fresh commitment in "
            "retrieve_unresolved_matters — ranking boost not applied"
        )

    asyncio.run(_run())


@pytest.mark.timeout(300)
def test_counter_grows_past_retrieval_cap() -> None:
    """I.5 reads the raw counter to trigger soft identity deltas —
    it must NOT saturate at `STAGNATION_BOOST_CAP` on the write side,
    only on the retrieval-boost side."""
    async def _run():
        mgr, clock = _make_sim(seed=1303)
        npc = mgr.npcs[0]
        stalled_id = _seed_unresolved_commitment(
            mgr, npc.npc_id, "Petra", "deliver bread",
        )

        # Run well past the cap (~18 days) to verify the counter
        # continues to climb beyond STAGNATION_BOOST_CAP (15).
        await _run_days(mgr, clock, num_days=18)

        stalled = mgr.memory.episodic.get_by_id(stalled_id)
        assert stalled is not None
        days = int(stalled.metadata.get(STAGNATION_METADATA_KEY, 0))
        cap = MemoryManager.STAGNATION_BOOST_CAP
        assert days > cap, (
            f"stagnation_days={days} capped at {cap} on write side — "
            "I.5 needs the raw counter to keep growing"
        )

    asyncio.run(_run())
