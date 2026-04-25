"""Tests for mid-day replanning (Phase B Step 5).

Verifies that Tier 1-2 NPCs can re-evaluate their schedule mid-day
based on recent perceptions and reflections, and that the schedule
is correctly modified when the LLM returns new entries.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from core.npc.models import NPC, PersonalityTraits, ScheduleEntry
from core.npc.cognition.plan import (
    should_replan, replan_schedule, REPLAN_INTERVALS, _parse_llm_schedule,
)


def _make_npc(
    npc_id: str = "alice_0",
    name: str = "Alice",
    occupation: str = "blacksmith",
    tier: int = 1,
) -> NPC:
    """Create a minimal NPC for testing."""
    npc = NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory=f"{name} is a {occupation}.",
        occupation=occupation,
        x=5.0, z=5.0,
        home_x=5, home_z=5,
        cognition_tier=tier,
    )
    return npc


def _make_schedule() -> list[ScheduleEntry]:
    """Create a typical daily schedule."""
    return [
        ScheduleEntry("early_morning", "eat breakfast", "home", 3, duration_minutes=60),
        ScheduleEntry("morning", "work at forge", "work", 7, duration_minutes=270),
        ScheduleEntry("afternoon", "eat lunch", "tavern", 4, duration_minutes=60),
        ScheduleEntry("afternoon", "work at forge", "work", 7, duration_minutes=240),
        ScheduleEntry("evening", "socialise", "tavern", 4, duration_minutes=240),
        ScheduleEntry("night", "sleep", "home", 9, duration_minutes=540),
    ]


class TestShouldReplan:
    def test_tier1_due_for_replan(self):
        """Tier 1 NPC past interval should need replan."""
        npc = _make_npc(tier=1)
        npc.daily_schedule = _make_schedule()
        npc.last_replan_minutes = 100.0
        assert should_replan(npc, 161.0) is True  # 61 min elapsed, interval=60

    def test_tier1_not_due(self):
        """Tier 1 NPC within interval should not replan."""
        npc = _make_npc(tier=1)
        npc.daily_schedule = _make_schedule()
        npc.last_replan_minutes = 100.0
        assert should_replan(npc, 150.0) is False  # 50 min elapsed

    def test_tier2_longer_interval(self):
        """Tier 2 should have 120-minute interval."""
        npc = _make_npc(tier=2)
        npc.daily_schedule = _make_schedule()
        npc.last_replan_minutes = 100.0
        assert should_replan(npc, 210.0) is False  # 110 min < 120
        assert should_replan(npc, 221.0) is True   # 121 min >= 120

    def test_tier3_never_replans(self):
        """Tier 3 NPCs should never replan."""
        npc = _make_npc(tier=3)
        npc.daily_schedule = _make_schedule()
        npc.last_replan_minutes = 0.0
        assert should_replan(npc, 9999.0) is False

    def test_tier4_never_replans(self):
        """Tier 4 (frozen) NPCs should never replan."""
        npc = _make_npc(tier=4)
        assert should_replan(npc, 9999.0) is False

    def test_custom_schedule_skips_replan(self):
        """NPCs with custom schedules should not replan."""
        npc = _make_npc(tier=1)
        npc.has_custom_schedule = True
        npc.daily_schedule = _make_schedule()
        assert should_replan(npc, 9999.0) is False

    def test_first_tick_not_due(self):
        """At game start (0 minutes), NPC should not replan immediately."""
        npc = _make_npc(tier=1)
        npc.daily_schedule = _make_schedule()
        npc.last_replan_minutes = 0.0
        assert should_replan(npc, 30.0) is False


class TestReplanSchedule:
    def test_no_change_response(self):
        """LLM returning NO_CHANGE should not modify the schedule."""
        async def _run():
            npc = _make_npc(tier=1)
            npc.daily_schedule = _make_schedule()
            npc.schedule_index = 1
            original_len = len(npc.daily_schedule)

            llm = AsyncMock()
            llm.complete = AsyncMock(return_value="NO_CHANGE")

            changed = await replan_schedule(npc, llm, 200.0)
            assert changed is False
            assert len(npc.daily_schedule) == original_len
            assert npc.last_replan_minutes == 200.0
        asyncio.new_event_loop().run_until_complete(_run())

    def test_new_schedule_replaces_remaining(self):
        """Valid LLM response should replace remaining entries."""
        async def _run():
            npc = _make_npc(tier=1)
            npc.daily_schedule = _make_schedule()
            npc.schedule_index = 2  # Currently at lunch (index 2)

            llm = AsyncMock()
            llm.complete = AsyncMock(return_value=(
                "13:00-14:00 — bring lunch to Bob, bridge\n"
                "14:00-17:00 — work at the forge, work\n"
                "17:00-21:00 — socialise at tavern, tavern\n"
                "21:00-06:00 — sleep at home, home"
            ))

            changed = await replan_schedule(npc, llm, 780.0)
            assert changed is True
            # Original schedule had 6 entries. Index 2 means 2 completed.
            # The remaining entries should be replaced by the 4 new ones.
            assert len(npc.daily_schedule) >= 4
            # The first 2 entries (completed) should be preserved
            assert npc.daily_schedule[0].activity == "eat breakfast"
            assert npc.daily_schedule[1].activity == "work at forge"
            assert npc.last_replan_minutes == 780.0
        asyncio.new_event_loop().run_until_complete(_run())

    def test_llm_failure_returns_false(self):
        """LLM exception should not crash, just return False."""
        async def _run():
            npc = _make_npc(tier=1)
            npc.daily_schedule = _make_schedule()
            npc.schedule_index = 1

            llm = AsyncMock()
            llm.complete = AsyncMock(side_effect=Exception("API timeout"))

            changed = await replan_schedule(npc, llm, 200.0)
            assert changed is False
            assert npc.last_replan_minutes == 200.0
        asyncio.new_event_loop().run_until_complete(_run())

    def test_empty_schedule_returns_false(self):
        """NPC with no schedule should not replan."""
        async def _run():
            npc = _make_npc(tier=1)
            npc.daily_schedule = []

            llm = AsyncMock()
            changed = await replan_schedule(npc, llm, 200.0)
            assert changed is False
        asyncio.new_event_loop().run_until_complete(_run())

    def test_replan_uses_context(self):
        """Replan should pass perceptions and reflections to LLM."""
        async def _run():
            npc = _make_npc(tier=1)
            npc.daily_schedule = _make_schedule()
            npc.schedule_index = 1
            npc.long_term_goals = ["become master blacksmith"]

            llm = AsyncMock()
            llm.complete = AsyncMock(return_value="NO_CHANGE")

            await replan_schedule(
                npc, llm, 200.0,
                recent_perceptions=["Bob looks hungry at the bridge"],
                recent_reflections=["I should help Bob"],
                relationship_summary="Bob is your close friend.",
            )

            # Verify the LLM was called with context
            call_args = llm.complete.call_args
            prompt = call_args.kwargs.get("messages", call_args[1].get(
                "messages", [{}]))[0].get("content", "")
            assert "Bob looks hungry" in prompt or "help Bob" in prompt
        asyncio.new_event_loop().run_until_complete(_run())


class TestReplanIntervals:
    def test_tier1_interval(self):
        assert REPLAN_INTERVALS[1] == 60.0

    def test_tier2_interval(self):
        assert REPLAN_INTERVALS[2] == 120.0

    def test_tier3_no_interval(self):
        assert REPLAN_INTERVALS[3] is None

    def test_tier4_no_interval(self):
        assert REPLAN_INTERVALS[4] is None
