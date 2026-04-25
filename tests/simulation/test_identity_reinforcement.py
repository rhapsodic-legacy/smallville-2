"""
Phase I.4 — end-to-end: a completed town goal reinforces the
self_concept of everyone who contributed, and leaves bystanders alone.

Uses the real NPCManager + TownAgenda wiring (no listener stubbing);
contributions are recorded through the public `record_contribution`
path so the completion listener fires naturally. The sim runs long
enough (~3 game days) to show the reinforcement survives subsequent
ticks and the bedtime compaction/self-review pass — i.e. reflection
is written, identity persists, and no accidental re-fire occurs.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.self_review import REINFORCEMENT_DELTA
from core.npc.cognition.router import CognitionRouter
from core.npc.cognition.router.policy import (
    CognitionPolicy, ROUTE_DETERMINISTIC,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world
from core.world.town_agenda import create_goal_from_template


def _make_sim(seed: int = 2411) -> tuple[NPCManager, GameClock]:
    config = WorldConfig(population=5, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
        llm=llm,
    )
    memory.initialise()
    # Keep bedtime work deterministic so the sim doesn't depend on
    # LLM stubs — mirrors the I.5 sim's approach.
    policy = CognitionPolicy()
    policy.set_mode("self_review", ROUTE_DETERMINISTIC)
    router = CognitionRouter(policy=policy)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed,
        memory=memory, router=router,
    )
    mgr.spawn_population(5)
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
def test_goal_completion_reinforces_contributor_identity() -> None:
    async def _run():
        mgr, clock = _make_sim()

        # Pick two contributors and one bystander. The remaining
        # NPCs fill out the population so the tick loop has a
        # realistic world to tick against.
        contributors = mgr.npcs[:2]
        bystander = mgr.npcs[2]
        contributor_ids = [n.npc_id for n in contributors]

        # Propose a repair_bridge goal. required_contributions=4, so
        # we record 4 contributions to complete it — the extras are
        # drawn from the remaining population.
        goal = create_goal_from_template("repair_bridge", current_day=0)
        mgr.town_agenda.propose(goal, current_day=0)

        all_contributors = contributor_ids + [
            mgr.npcs[3].npc_id, mgr.npcs[4].npc_id,
        ]
        # Fire the last contribution last so we exercise the
        # PROPOSED → ACTIVE → COMPLETED transition naturally. The
        # listener runs inside record_contribution when the final
        # count transitions to COMPLETED.
        for npc_id in all_contributors[:-1]:
            mgr.town_agenda.record_contribution(
                goal.goal_id, npc_id, current_day=0,
            )
        completed = mgr.town_agenda.record_contribution(
            goal.goal_id, all_contributors[-1], current_day=0,
        )
        assert completed, "test setup failure: goal did not complete"

        # Let the sim run a few days so the bedtime pass fires and
        # we can confirm the reinforced belief survives it.
        await _run_days(mgr, clock, num_days=3)

        # Every contributor carries the reinforced belief. Floating-
        # point tolerance matches the I.5 sim's convention (Big-5
        # drift etc. also mutate adjacent state).
        for npc in contributors:
            remaining = npc.self_concept.get("built:bridge", 0.0)
            assert remaining == pytest.approx(REINFORCEMENT_DELTA, abs=0.02), (
                f"expected ~{REINFORCEMENT_DELTA} for {npc.name}, got {remaining}"
            )

        # Bystander didn't contribute → no identity bump on the
        # goal's key. (Other self_concept changes from normal sim
        # activity are fine; we only assert on the specific key.)
        assert "built:bridge" not in bystander.self_concept

        # Reflection memory written for at least one contributor,
        # tagged with the goal_id + `town_agenda` so Phase K
        # retrieval can surface it.
        reflections = [
            m for m in mgr.memory.episodic.get_recent(
                contributors[0].npc_id, limit=200,
            )
            if m.category == "reflection"
            and (m.metadata or {}).get("outcome_kind")
            == "identity_reinforcement"
        ]
        assert reflections, (
            "no identity_reinforcement reflection memory written"
        )
        assert reflections[0].metadata.get("source_goal_id") == goal.goal_id
        assert "town_agenda" in reflections[0].tags
        assert goal.goal_id in reflections[0].tags

    asyncio.run(_run())
