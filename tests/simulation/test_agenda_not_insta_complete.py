"""
Regression: town-agenda goals must not propose + complete in the
same tick.

Observed bug (Dara's memory, Day 83 00:01 and Day 78 00:00):

  TOWN_AGENDA  Day 84 00:01  Town proposed "Repair the old bridge"
  TOWN_EVENT   Day 84 00:01  Completed by Kira, Seren, Jasper, Aelric

Proposal and completion arrived in the same game-second, with
exactly the number of personality-matching NPCs listed as
"contributors" though they had never visited the bridge or spent
any time on the activity.

Root cause: `_inject_goal_entry` eagerly called
`record_contribution` at schedule-injection time, once per matching
NPC. Goals with `required_contributions == N` therefore completed
the same tick they were proposed. The fix moves crediting to
`_advance_npc_action` (when the NPC actually finishes the entry)
and makes `TownGoal.record_contribution` dedup per NPC.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world
from core.world.town_agenda import GoalStatus, TownGoal


def _mgr() -> NPCManager:
    config = WorldConfig(population=6, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=MockProvider(), seed=42,
        memory=memory,
    )
    mgr.spawn_population(6)
    return mgr


class TestInjectionDoesNotContribute:
    """`_inject_goal_entry` must NOT credit the NPC at injection
    time — only when `_advance_npc_action` finishes the entry."""

    def test_injection_alone_does_not_advance_progress(self):
        mgr = _mgr()
        # Hand-craft a goal requiring 3 contributions. Propose it
        # into the agenda ourselves (bypass the overseer timing).
        goal = TownGoal(
            goal_id="test_goal",
            title="Test task",
            description="Do a thing",
            activity_text="do the thing",
            location_hint="town_square",
            duration_minutes=60,
            required_contributions=3,
            deadline_day=99,
            personality_bias={},
            created_day=0,
        )
        mgr.town_agenda.propose(goal, current_day=0)

        # Give every NPC a minimal schedule so injection has a slot
        # to commandeer.
        from core.npc.models import ScheduleEntry
        for npc in mgr.npcs:
            npc.daily_schedule = [
                ScheduleEntry(
                    slot="afternoon", activity="idle",
                    location="home", priority=1,
                    duration_minutes=600,
                ),
            ]
            npc.schedule_index = 0
            mgr._inject_goal_entry(npc, current_day=0)

        # After injection for 6 NPCs: progress must still be 0
        # and the goal must not be completed.
        refreshed = mgr.town_agenda._goals["test_goal"]
        assert refreshed.progress == 0, (
            f"Injection must not advance progress. "
            f"Got progress={refreshed.progress}, "
            f"contributors={refreshed.contributors}"
        )
        assert refreshed.status != GoalStatus.COMPLETED


class TestRecordContributionDedup:
    """TownGoal.record_contribution must dedup per NPC."""

    def test_double_contribution_by_same_npc_noop(self):
        goal = TownGoal(
            goal_id="g",
            title="T",
            description="",
            activity_text="a",
            location_hint="",
            duration_minutes=60,
            required_contributions=3,
            deadline_day=5,
            personality_bias={},
            created_day=0,
        )
        assert not goal.record_contribution("alice")
        assert goal.progress == 1
        # Second call by alice: no-op.
        assert not goal.record_contribution("alice")
        assert goal.progress == 1
        assert goal.contributors == {"alice"}

    def test_dedup_prevents_premature_completion(self):
        goal = TownGoal(
            goal_id="g",
            title="T",
            description="",
            activity_text="a",
            location_hint="",
            duration_minutes=60,
            required_contributions=2,
            deadline_day=5,
            personality_bias={},
            created_day=0,
        )
        # Alice tries to contribute twice.
        goal.record_contribution("alice")
        goal.record_contribution("alice")
        # Should NOT be complete (only one real contributor).
        assert goal.status != GoalStatus.COMPLETED
        # Bob contributes → now complete.
        assert goal.record_contribution("bob")
        assert goal.status == GoalStatus.COMPLETED


class TestContributionOnActionFinish:
    """`_advance_npc_action` credits the finishing entry's goal."""

    def test_finishing_goal_entry_records_contribution(self):
        async def _run():
            mgr = _mgr()
            goal = TownGoal(
                goal_id="bridge",
                title="Repair the bridge",
                description="",
                activity_text="help repair the bridge",
                location_hint="town_square",
                duration_minutes=60,
                required_contributions=3,
                deadline_day=99,
                personality_bias={},
                created_day=0,
            )
            mgr.town_agenda.propose(goal, current_day=0)

            npc = mgr.npcs[0]
            from core.npc.models import ScheduleEntry
            entry = ScheduleEntry(
                slot="afternoon", activity="help repair the bridge",
                location="town_square", priority=8,
                duration_minutes=60,
            )
            setattr(entry, "town_goal_id", "bridge")
            # Next entry to advance into (so we don't hit the
            # schedule-regeneration branch).
            tail = ScheduleEntry(
                slot="evening", activity="rest",
                location="home", priority=1, duration_minutes=60,
            )
            npc.daily_schedule = [entry, tail]
            npc.schedule_index = 0
            npc.schedule_day = 0

            # Patch _dispatch_to_entry to a no-op so the test stays
            # focused on the contribution bookkeeping, not movement.
            async def _noop(*_args, **_kwargs): return None
            with patch.object(mgr, "_dispatch_to_entry", _noop):
                await mgr._advance_npc_action(
                    npc, current_minutes=60.0, current_slot="afternoon",
                )

            refreshed = mgr.town_agenda._goals["bridge"]
            assert npc.npc_id in refreshed.contributors
            assert refreshed.progress == 1

        asyncio.run(_run())


class TestNoInstaCompleteEndToEnd:
    """With a fresh goal proposed AT the propose listener, and a
    full schedule generation + inject pass for every NPC, progress
    stays at 0 until an NPC actually finishes the goal entry."""

    def test_propose_then_inject_produces_zero_progress(self):
        mgr = _mgr()
        # Direct propose, as if the overseer did it.
        goal = TownGoal(
            goal_id="festival",
            title="Harvest festival",
            description="",
            activity_text="help with the festival",
            location_hint="town_square",
            duration_minutes=60,
            required_contributions=3,
            deadline_day=99,
            personality_bias={},
            created_day=0,
        )
        mgr.town_agenda.propose(goal, current_day=0)

        # Emulate the step-2b loop in `cognition_tick`: inject goal
        # entries for every NPC whose schedule matches.
        from core.npc.models import ScheduleEntry
        for npc in mgr.npcs:
            npc.daily_schedule = [
                ScheduleEntry(
                    slot="afternoon", activity="idle",
                    location="home", priority=1,
                    duration_minutes=600,
                ),
            ]
            npc.schedule_index = 0
            mgr._inject_goal_entry(npc, current_day=0)

        # 6 NPCs injected, goal still not progressed.
        refreshed = mgr.town_agenda._goals["festival"]
        assert refreshed.progress == 0
        assert refreshed.status in (
            GoalStatus.PROPOSED, GoalStatus.ACTIVE,
        )
        # No one is a contributor yet.
        assert refreshed.contributors == set()
