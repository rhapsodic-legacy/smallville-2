"""
Position-tracking oscillation test.

Runs a headless simulation across a FULL DAY, recording every NPC's
(x, z) position at every tick. Detects:

1. Position oscillation: NPC bouncing between 2-3 tiles repeatedly
2. Lockstep synchronisation: multiple NPCs moving in unison
3. Perpetual walking: NPC never settling down at a location

This test catches the actual bug the user sees — NPCs physically
jittering back and forth on screen at any time of day.
"""

from __future__ import annotations

import asyncio
import pytest
from collections import Counter

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.time_system.clock import GameClock
from core.world.generator import WorldConfig, generate_world


@pytest.fixture
def world():
    config = WorldConfig(population=6, terrain="riverside", seed=99)
    grid, buildings = generate_world(config)
    return grid, buildings


@pytest.fixture
def manager(world):
    grid, buildings = world
    llm = MockProvider()
    memory = MemoryManager(llm=llm)
    mgr = NPCManager(
        grid=grid,
        buildings=buildings,
        llm=llm,
        seed=99,
        memory=memory,
    )
    mgr.spawn_population(6)
    return mgr


def _run_full_day(manager: NPCManager, total_ticks: int = 1200):
    """Run simulation for a full day, recording position timelines.

    Returns dict mapping npc_id -> list of (tick, x, z, activity, slot).
    """
    clock = GameClock()
    timelines: dict[str, list[tuple]] = {
        npc.npc_id: [] for npc in manager.npcs
    }

    async def _run():
        for tick in range(total_ticks):
            clock.tick(1.0)
            await manager.tick(clock, 1.0)

            slot = clock.schedule_slot.value
            for npc in manager.npcs:
                timelines[npc.npc_id].append((
                    tick, npc.tile_x, npc.tile_z,
                    npc.activity.value, slot,
                ))

    asyncio.get_event_loop().run_until_complete(_run())
    return timelines


def _detect_oscillation(positions: list[tuple[int, int]], window: int = 20) -> int:
    """Count direction reversals in a sliding window.

    An oscillation is: NPC moves A→B→A or A→B→A→B. We detect this by
    counting how many times the NPC reverses direction in a window.
    A reversal is moving to a tile, then back to the previous tile.
    """
    max_reversals = 0
    for start in range(0, len(positions) - window):
        chunk = positions[start:start + window]
        reversals = 0
        for i in range(2, len(chunk)):
            if chunk[i] == chunk[i - 2] and chunk[i] != chunk[i - 1]:
                reversals += 1
        max_reversals = max(max_reversals, reversals)
    return max_reversals


class TestPositionOscillation:
    """Detect NPCs bouncing back and forth at any time of day."""

    def test_no_oscillation_full_day(self, manager):
        """No NPC should oscillate (bounce A→B→A) more than 3 times
        in any 20-tick window across the entire day."""
        timelines = _run_full_day(manager)

        violations = []
        for npc_id, entries in timelines.items():
            positions = [(x, z) for _, x, z, _, _ in entries]
            worst = _detect_oscillation(positions)
            if worst > 3:
                # Find the slot where the worst oscillation happens
                npc = next(n for n in manager.npcs if n.npc_id == npc_id)
                violations.append(
                    f"{npc.name} ({npc_id}): {worst} reversals in 20-tick window"
                )

        assert len(violations) == 0, (
            f"{len(violations)} NPCs oscillating:\n"
            + "\n".join(violations)
        )

    def test_no_perpetual_walking(self, manager):
        """NPCs should not spend more than 50% of any slot walking.

        Stanford model: walk to destination, do action, stay put.
        If an NPC walks for 50%+ of a slot, something is wrong.
        """
        timelines = _run_full_day(manager)

        violations = []
        for npc_id, entries in timelines.items():
            # Group by slot
            slot_ticks: dict[str, list[str]] = {}
            for _, _, _, activity, slot in entries:
                slot_ticks.setdefault(slot, []).append(activity)

            npc = next(n for n in manager.npcs if n.npc_id == npc_id)
            for slot, activities in slot_ticks.items():
                total = len(activities)
                if total < 10:
                    continue
                walking = sum(1 for a in activities if a == "walking")
                pct = walking / total
                if pct > 0.5:
                    violations.append(
                        f"{npc.name} slot={slot}: "
                        f"{walking}/{total} ticks walking ({pct:.0%})"
                    )

        assert len(violations) == 0, (
            f"{len(violations)} NPCs walking excessively:\n"
            + "\n".join(violations)
        )


class TestLockstepMovement:
    """Detect multiple NPCs moving in synchronised lockstep."""

    def test_no_lockstep_movement(self, manager):
        """At most 2 NPCs should start walking on the exact same tick.

        Staggered departures should prevent lockstep.
        """
        timelines = _run_full_day(manager)

        # Find ticks where NPCs transition TO walking
        walk_start_ticks: list[int] = []
        for npc_id, entries in timelines.items():
            prev_act = None
            for tick, _, _, activity, _ in entries:
                if activity == "walking" and prev_act != "walking":
                    walk_start_ticks.append(tick)
                prev_act = activity

        # Count how many NPCs start walking on the same tick
        tick_counts = Counter(walk_start_ticks)
        sync_ticks = [
            (tick, count) for tick, count in tick_counts.items()
            if count >= 3
        ]

        assert len(sync_ticks) <= 2, (
            f"{len(sync_ticks)} ticks had 3+ NPCs starting to walk simultaneously: "
            + ", ".join(f"tick {t}: {c} NPCs" for t, c in sync_ticks[:5])
        )


class TestPositionStability:
    """NPCs should spend most of their time stationary."""

    def test_position_changes_reasonable(self, manager):
        """Each NPC should change position fewer than 200 times in a day.

        A day is ~1200 ticks. Walking somewhere takes ~20-40 ticks.
        With 5-6 slot transitions, max ~300 walking ticks. Excessive
        position changes indicate the dispatch loop bug.
        """
        timelines = _run_full_day(manager)

        for npc_id, entries in timelines.items():
            positions = [(x, z) for _, x, z, _, _ in entries]
            changes = sum(
                1 for a, b in zip(positions[:-1], positions[1:])
                if a != b
            )
            npc = next(n for n in manager.npcs if n.npc_id == npc_id)
            assert changes < 200, (
                f"{npc.name} changed position {changes} times in "
                f"{len(positions)} ticks — excessive movement"
            )
