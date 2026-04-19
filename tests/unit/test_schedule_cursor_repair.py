"""Regression: the cognition tick repairs invalid schedule cursors.

The observed bug (day-8 run): NPCs ended up with
`daily_schedule=[], schedule_index=1, action_start_minutes > 0`.
In that state `_advance_npc_action` never runs (step 3 bails on
empty schedule) and `needs_new_schedule` may or may not refire
depending on `schedule_day`, leaving the NPC frozen at home.

These tests drive only `_normalise_schedule_cursors` — the safety
net — and verify it heals each of the three bad-state variants
without disturbing healthy NPCs.
"""

from __future__ import annotations

import pytest

from core.npc.models import ScheduleEntry
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


@pytest.fixture
def mgr() -> NPCManager:
    config = WorldConfig(population=4, terrain="riverside", seed=7)
    grid, buildings = generate_world(config)
    m = NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=7)
    m.spawn_population(4)
    return m


def test_empty_schedule_with_nonzero_index_is_repaired(mgr: NPCManager) -> None:
    npc = mgr.npcs[0]
    npc.daily_schedule = []
    npc.schedule_index = 1
    npc.action_start_minutes = 12407.0

    mgr._normalise_schedule_cursors()

    assert npc.schedule_index == 0
    assert npc.action_start_minutes == 0.0


def test_cursor_off_end_of_nonempty_schedule_is_wrapped(
    mgr: NPCManager,
) -> None:
    npc = mgr.npcs[0]
    npc.daily_schedule = [
        ScheduleEntry(slot="morning", activity="work",
                      location="work", duration_minutes=60),
    ]
    npc.schedule_index = 5  # off the end
    npc.action_start_minutes = 500.0

    mgr._normalise_schedule_cursors()

    assert npc.schedule_index == 0
    assert npc.action_start_minutes == 0.0


def test_healthy_schedule_is_untouched(mgr: NPCManager) -> None:
    npc = mgr.npcs[0]
    npc.daily_schedule = [
        ScheduleEntry(slot="morning", activity="work",
                      location="work", duration_minutes=60),
        ScheduleEntry(slot="evening", activity="rest",
                      location="home", duration_minutes=120),
    ]
    npc.schedule_index = 1
    npc.action_start_minutes = 300.0

    mgr._normalise_schedule_cursors()

    assert npc.schedule_index == 1
    assert npc.action_start_minutes == 300.0


def test_empty_schedule_with_clean_cursor_is_untouched(
    mgr: NPCManager,
) -> None:
    """Don't noise-log a freshly-regenerating NPC."""
    npc = mgr.npcs[0]
    npc.daily_schedule = []
    npc.schedule_index = 0
    npc.action_start_minutes = 0.0

    mgr._normalise_schedule_cursors()

    assert npc.schedule_index == 0
    assert npc.action_start_minutes == 0.0
