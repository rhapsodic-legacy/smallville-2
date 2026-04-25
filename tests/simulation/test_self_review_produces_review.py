"""
Phase I.1/I.2 — end-to-end: the bedtime self-review fires on day
rollover and produces a `commitment_review` memory naming the
NPC's open commitment.

Parallel to `test_compaction_preserves_tags.py`: rather than testing
the module in isolation, this drives the full NPCManager through a
multi-day sim and asserts the new tick-time wiring actually runs.

Scenario:
1. Seed NPC[0] on day 0 with an unresolved self-commitment
   ("I promised to check the south field.") carrying a
   Phase K tag `field`.
2. Run the sim through ~3 game-days so day 0's rollover tick fires.
3. Assert:
   - `_last_self_reviewed_day[npc] >= 0` — the cursor advanced.
   - A `commitment_review` memory exists for the NPC naming the
     "field" commitment in its description.
   - The review inherits the source commitment's tags (Phase K
     anchoring) so it surfaces via `retrieve_by_tags(["field"])`.
   - The review is NOT tombstoned by the next day's compaction
     (PRESERVED_CATEGORIES contract).
"""

from __future__ import annotations

import asyncio
import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.npc.cognition.router import CognitionRouter
from core.npc.cognition.router.policy import (
    CognitionPolicy, ROUTE_DETERMINISTIC,
)
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world


def _make_sim(seed: int = 811) -> tuple[NPCManager, GameClock]:
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
    # Force self_review to DETERMINISTIC so the mock LLM isn't on
    # the critical path — the heuristic fallback still produces a
    # `commitment_review` with the expected tag anchoring, which is
    # what this test cares about. The LLM voice path is covered by
    # the unit tests.
    policy = CognitionPolicy()
    policy.set_mode("self_review", ROUTE_DETERMINISTIC)
    router = CognitionRouter(policy=policy)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed,
        memory=memory, router=router,
    )
    mgr.spawn_population(3)
    return mgr, GameClock()


def _seed_self_commitment(mgr: NPCManager, npc_id: str) -> str:
    """Drop a Phase B-shaped unresolved commitment on day 0."""
    return mgr.memory.episodic.add_memory(
        npc_id=npc_id,
        description="I promised to check the south field.",
        category="commitment",
        importance=0.75,
        game_time=120.0,
        tags={"field"},
        extra_metadata={
            "outcome_kind": "commitment",
            "source_speaker": npc_id,
            "unresolved": True,
        },
    )


async def _run_through_day(
    mgr: NPCManager, clock: GameClock, num_days: int,
) -> None:
    """Advance the sim so at least `num_days` rollovers fire."""
    real_delta = 8.0
    game_minutes_per_tick = 9.6
    total_game_minutes = num_days * MINUTES_PER_DAY
    num_ticks = int(total_game_minutes / game_minutes_per_tick) + 1
    for _ in range(num_ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)


@pytest.mark.timeout(300)
def test_self_review_writes_commitment_review_at_day_rollover() -> None:
    async def _run():
        mgr, clock = _make_sim()
        npc = mgr.npcs[0]
        commitment_id = _seed_self_commitment(mgr, npc.npc_id)

        await _run_through_day(mgr, clock, num_days=3)

        # Cursor advanced at least through day 0.
        assert mgr._last_self_reviewed_day.get(npc.npc_id, -1) >= 0, (
            "self-review did not fire — cursor not advanced"
        )

        # A commitment_review memory exists for day 0 naming the
        # open commitment.
        reviews = [
            m for m in mgr.memory.episodic.get_recent(
                npc.npc_id, limit=100,
            )
            if m.category == "commitment_review"
        ]
        assert reviews, (
            "no commitment_review memory produced during multi-day "
            "sim — day-rollover wiring broken"
        )
        # At least one review's metadata points at day 0 and cites
        # the seeded commitment as a source.
        day0 = [r for r in reviews if (r.metadata or {}).get("day") == 0]
        assert day0, "no commitment_review metadata says day == 0"
        source_ids = set(
            (day0[0].metadata.get("source_ids") or "").split()
        )
        assert commitment_id in source_ids, (
            "commitment_review's source_ids did not include the "
            "seeded unresolved commitment"
        )

        # Phase K anchoring: the review inherits the `field` tag
        # and surfaces via tag-based retrieval.
        tag_hits = mgr.memory.episodic.retrieve_by_tags(
            npc.npc_id, ["field"],
        )
        assert any(
            m.memory_id == day0[0].memory_id for m in tag_hits
        ), (
            "commitment_review not discoverable via retrieve_by_tags "
            "— Phase K anchoring broken"
        )

        # Preservation contract: the review is NOT tombstoned by
        # the day-1/day-2 compaction passes.
        review_raw = mgr.memory.episodic.get_raw_by_id(day0[0].memory_id)
        assert review_raw is not None
        assert not (review_raw.metadata or {}).get("compacted_into"), (
            "commitment_review was tombstoned by compaction — "
            "PRESERVED_CATEGORIES contract violated"
        )

    asyncio.run(_run())
