"""Phase 0 acceptance tests for the foundation rebuild
(FOUNDATION_REBUILD_ROADMAP.md).

These encode the TARGET behaviour and are expected to FAIL until the
rebuild lands — they are marked xfail(strict=True) so the suite stays
green now AND so the marker must be removed in the phase that fixes the
behaviour (an xpass under strict=True is a failure, forcing the flip).

1. test_town_goal_completes_organically — a proposed town goal must be
   able to COMPLETE through the organic path (NPCs take it on, perform
   it, contributions are credited). Today every goal expires 0/N
   because mid-day replanning wipes the injected goal entry before the
   NPC reaches it. Flips GREEN in Phase 4.

2. test_schedule_stays_bounded — an NPC's daily_schedule must stay
   under a hard cap. Today replanning leaks ~+2 entries per replan and
   schedules bloat to 18+ in a day. Flips GREEN in Phase 3.

Both use MockProvider: the bugs are provider-independent scheduling
plumbing, so a deterministic sim reproduces them in seconds.
"""

from __future__ import annotations

import asyncio

import pytest

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
from core.world.town_agenda import create_goal_from_template, GoalStatus


# A daily schedule should never exceed a small bounded number of entries.
# Occupation templates have ~6-8 entries; a sound plan stays in that range.
SCHEDULE_CAP = 12


def _make_sim(pop: int = 6, seed: int = 42) -> tuple[NPCManager, GameClock]:
    config = WorldConfig(population=pop, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
        llm=llm,
    )
    memory.initialise()
    # Keep bedtime self-review deterministic; leave daily_schedule on its
    # default route so replanning still exercises the real path.
    policy = CognitionPolicy()
    policy.set_mode("self_review", ROUTE_DETERMINISTIC)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed,
        memory=memory, router=CognitionRouter(policy=policy),
    )
    mgr.spawn_population(pop)
    return mgr, GameClock()


async def _run_days(mgr, clock, num_days, *, on_tick=None):
    real_delta = 8.0
    game_minutes_per_tick = 9.6
    ticks = int(num_days * MINUTES_PER_DAY / game_minutes_per_tick) + 1
    for _ in range(ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)
        if on_tick is not None:
            on_tick()


@pytest.mark.timeout(300)
def test_town_goal_completes_organically() -> None:
    # GREEN as of Phase 3.5 + Phase 4: with a faithful schedule (parser
    # parses real durations; replan keeps the in-progress entry) NPCs reach
    # and perform the goal, and bedtime-safe commitment crediting completes
    # it organically — instead of every cycle expiring at 0 contributions.
    async def _run():
        mgr, clock = _make_sim(pop=6)

        # Seed willing contributors so participation is not the variable
        # under test — we are testing the credit *path*, not the gate.
        goal = create_goal_from_template("repair_bridge", current_day=0)
        for npc in mgr.npcs[:5]:
            npc.self_concept[f"supports:{goal.goal_id}"] = 0.9
        mgr.town_agenda.propose(goal, current_day=0)

        # Run long enough for several propose/deadline cycles.
        for _ in range(4):
            await _run_days(mgr, clock, num_days=2)
            g = mgr.town_agenda.get("repair_bridge")
            if g and g.status == GoalStatus.COMPLETED:
                break
            # Re-propose if the cycle expired, so we get repeated chances.
            if g is None or g.status == GoalStatus.EXPIRED:
                fresh = create_goal_from_template(
                    "repair_bridge", current_day=clock.day,
                )
                if fresh:
                    mgr.town_agenda.propose(fresh, current_day=clock.day)

        completed = [
            g for g in mgr.town_agenda._goals.values()
            if g.goal_id == "repair_bridge"
            and g.status == GoalStatus.COMPLETED
        ]
        assert completed, (
            "no repair_bridge cycle ever completed organically — "
            "contributions were never credited"
        )

    asyncio.run(_run())


@pytest.mark.timeout(300)
def test_schedule_stays_bounded() -> None:
    # GREEN as of Phase 3: replan re-derives without growing the schedule,
    # so daily_schedule stays bounded (~7) instead of bloating to 20+.
    async def _run():
        mgr, clock = _make_sim(pop=6)
        max_len = {"v": 0}

        def _watch():
            for n in mgr.npcs:
                max_len["v"] = max(max_len["v"], len(n.daily_schedule))

        await _run_days(mgr, clock, num_days=3, on_tick=_watch)

        assert max_len["v"] <= SCHEDULE_CAP, (
            f"daily_schedule bloated to {max_len['v']} entries "
            f"(cap {SCHEDULE_CAP})"
        )

    asyncio.run(_run())
