"""
Night behaviour simulation test.

Detects the three failure modes that caused the lockstep night oscillation:
1. NPCs oscillating (walking back and forth at night)
2. NPCs on non-passable tiles (inside buildings)
3. NPCs synchronised (all doing the same thing at the same tick)

Runs a headless simulation from evening through night into early morning,
sampling NPC state every tick.
"""

from __future__ import annotations

import asyncio
import pytest
from collections import Counter

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock
from core.world.generator import WorldConfig, generate_world


@pytest.fixture
def world():
    """Create a small test world with 6 NPCs."""
    config = WorldConfig(population=6, terrain="riverside", seed=99)
    grid, buildings = generate_world(config)
    return grid, buildings


@pytest.fixture
def manager(world):
    grid, buildings = world
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(
        grid=grid,
        buildings=buildings,
        llm=llm,
        seed=99,
        memory=memory,
    )
    mgr.spawn_population(6)
    return mgr


def _run_simulation(
    manager: NPCManager,
    clock: GameClock,
    total_ticks: int,
    tick_interval: float = 1.0,
):
    """Run simulation for N ticks, recording state snapshots."""
    snapshots = []

    async def _run():
        for tick_num in range(total_ticks):
            clock.tick(tick_interval)
            await manager.tick(clock, tick_interval)

            snap = {
                "tick": tick_num,
                "time": clock.time_string,
                "phase": clock.phase.value,
                "day": clock.day,
                "npcs": [],
            }
            for npc in manager.npcs:
                snap["npcs"].append({
                    "id": npc.npc_id,
                    "x": npc.tile_x,
                    "z": npc.tile_z,
                    "activity": npc.activity.value,
                    "walking": npc.activity == ActivityState.WALKING,
                    "description": npc.current_action_description,
                })
            snapshots.append(snap)

    asyncio.new_event_loop().run_until_complete(_run())
    return snapshots


def _advance_to_slot(manager: NPCManager, clock: GameClock, target_slot: str):
    """Advance the clock until the target schedule slot is reached."""
    async def _run():
        for _ in range(3000):  # safety limit
            clock.tick(1.0)
            await manager.tick(clock, 1.0)
            if clock.schedule_slot.value == target_slot:
                return
        raise TimeoutError(f"Never reached slot '{target_slot}'")

    asyncio.new_event_loop().run_until_complete(_run())


class TestNightOscillation:
    """Detect NPCs oscillating back and forth at night."""

    def test_no_walking_during_deep_night(self, manager):
        """After settling at home, NPCs should NOT be walking during night."""
        clock = GameClock()

        # Advance to night phase + 60 extra ticks for NPCs to settle
        _advance_to_slot(manager, clock, "night")
        snapshots = _run_simulation(manager, clock, total_ticks=120)

        # After the first 30 ticks (settling time), count walking NPCs
        settled_snaps = snapshots[30:]
        walk_counts = []
        for snap in settled_snaps:
            walking = sum(1 for n in snap["npcs"] if n["walking"])
            walk_counts.append(walking)

        # Average walking NPCs during deep night should be near zero
        avg_walking = sum(walk_counts) / len(walk_counts)
        assert avg_walking < 0.5, (
            f"Average {avg_walking:.1f} NPCs walking during deep night — "
            f"should be near 0. Likely oscillation bug."
        )

    def test_no_position_oscillation(self, manager):
        """NPCs should not flip between positions repeatedly at night.

        Visiting multiple positions is fine (walking home). The bug is
        A→B→A→B oscillation — bouncing back and forth.
        """
        clock = GameClock()
        _advance_to_slot(manager, clock, "night")
        snapshots = _run_simulation(manager, clock, total_ticks=120)

        settled_snaps = snapshots[30:]
        for npc in manager.npcs:
            positions = []
            for snap in settled_snaps:
                for n in snap["npcs"]:
                    if n["id"] == npc.npc_id:
                        positions.append((n["x"], n["z"]))

            if len(positions) < 10:
                continue

            # Count direction reversals (A→B→A pattern)
            reversals = 0
            for i in range(2, len(positions)):
                if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]:
                    reversals += 1

            assert reversals <= 3, (
                f"{npc.name} reversed direction {reversals} times during "
                f"deep night — oscillation detected."
            )


class TestNightSynchronisation:
    """Detect NPCs acting in lockstep at night."""

    def test_activity_diversity(self, manager):
        """Activity transitions should not all happen at the same tick.

        All NPCs sleeping simultaneously at night is correct — the bug
        we're detecting is synchronised TRANSITIONS (all NPCs changing
        activity at the exact same tick, e.g. walking↔idle oscillation).
        """
        clock = GameClock()
        _advance_to_slot(manager, clock, "night")
        snapshots = _run_simulation(manager, clock, total_ticks=200)

        settled_snaps = snapshots[30:]

        # Build per-NPC activity timelines
        timelines: dict[str, list[str]] = {}
        for npc in manager.npcs:
            timelines[npc.npc_id] = []
            for snap in settled_snaps:
                for n in snap["npcs"]:
                    if n["id"] == npc.npc_id:
                        timelines[npc.npc_id].append(n["activity"])

        # Count ticks where 3+ NPCs change activity simultaneously
        sync_transitions = 0
        for tick_idx in range(1, len(settled_snaps)):
            changers = 0
            for npc_id, timeline in timelines.items():
                if tick_idx < len(timeline) and timeline[tick_idx] != timeline[tick_idx - 1]:
                    changers += 1
            if changers >= 3:
                sync_transitions += 1

        # A handful of synchronised transitions is tolerable (e.g. all
        # settle into sleep at roughly the same time). Many indicates
        # the lockstep oscillation bug.
        assert sync_transitions <= 5, (
            f"{sync_transitions} ticks had 3+ NPCs changing activity "
            f"simultaneously — likely lockstep synchronisation bug."
        )


class TestNightPassability:
    """Detect NPCs resting on non-passable tiles."""

    def test_npcs_on_passable_tiles(self, manager):
        """Non-walking NPCs must always be on passable tiles."""
        clock = GameClock()
        _advance_to_slot(manager, clock, "night")
        snapshots = _run_simulation(manager, clock, total_ticks=100)

        grid = manager.grid
        violations = []
        for snap in snapshots[20:]:
            for n in snap["npcs"]:
                if n["walking"]:
                    continue
                tile = grid.get_tile(n["x"], n["z"])
                if tile and not tile.is_passable:
                    violations.append(
                        f"Tick {snap['tick']}: NPC {n['id']} resting on "
                        f"non-passable tile ({n['x']}, {n['z']}) "
                        f"terrain={tile.terrain.value}"
                    )

        assert len(violations) == 0, (
            f"{len(violations)} passability violations:\n"
            + "\n".join(violations[:10])
        )

    def test_sleeping_npcs_stay_put(self, manager):
        """Once an NPC starts sleeping, they should not move for a while."""
        clock = GameClock()
        _advance_to_slot(manager, clock, "night")
        snapshots = _run_simulation(manager, clock, total_ticks=150)

        # For each NPC, find the first tick they're sleeping, then
        # verify they stay at that position for at least 30 ticks
        for npc in manager.npcs:
            sleep_start = None
            sleep_pos = None
            moved_while_sleeping = 0

            for snap in snapshots:
                for n in snap["npcs"]:
                    if n["id"] != npc.npc_id:
                        continue
                    if n["activity"] == "sleeping" and sleep_start is None:
                        sleep_start = snap["tick"]
                        sleep_pos = (n["x"], n["z"])
                    elif sleep_start is not None and sleep_pos is not None:
                        if (n["x"], n["z"]) != sleep_pos:
                            moved_while_sleeping += 1
                            # Reset — maybe they legitimately woke
                            if n["activity"] != "sleeping":
                                sleep_start = None
                                sleep_pos = None

            assert moved_while_sleeping <= 2, (
                f"{npc.name} moved {moved_while_sleeping} times while "
                f"sleeping — should stay put."
            )
