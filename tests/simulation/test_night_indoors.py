"""
Night indoors test — sleeping NPCs must be inside buildings.

At midnight, all NPCs should be sleeping inside their home building
footprint, not standing around outside. Advances to night slot,
lets NPCs settle, then verifies every sleeping NPC is on an interior
or door tile within a building.
"""

from __future__ import annotations

import asyncio
import pytest

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock
from core.world.generator import (
    WorldConfig,
    generate_world,
    PlacedBuilding,
    _building_interior_tiles,
)


@pytest.fixture
def world():
    config = WorldConfig(population=10, terrain="riverside", seed=42)
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
        seed=42,
        memory=memory,
    )
    mgr.spawn_population(10)
    return mgr


def _advance_to_slot(manager: NPCManager, clock: GameClock, target_slot: str):
    """Advance the clock until the target schedule slot is reached."""
    async def _run():
        for _ in range(3000):
            clock.tick(1.0)
            await manager.tick(clock, 1.0)
            if clock.schedule_slot.value == target_slot:
                return
        raise TimeoutError(f"Never reached slot '{target_slot}'")
    asyncio.new_event_loop().run_until_complete(_run())


def _run_ticks(manager: NPCManager, clock: GameClock, n: int):
    """Run n ticks."""
    async def _run():
        for _ in range(n):
            clock.tick(1.0)
            await manager.tick(clock, 1.0)
    asyncio.new_event_loop().run_until_complete(_run())


def _get_all_interior_tiles(buildings: list[PlacedBuilding]) -> set[tuple[int, int]]:
    """Union of all building interior + door tiles."""
    tiles = set()
    for b in buildings:
        interior = _building_interior_tiles(
            b.x, b.z, b.width, b.height, b.door_x, b.door_z,
        )
        interior.add((b.door_x, b.door_z))
        tiles |= interior
    return tiles


class TestSleepingIndoors:
    """All sleeping NPCs should be inside a building at night."""

    def test_sleeping_npcs_inside_at_night(self, manager):
        """After settling into night, sleeping NPCs must be on building
        interior tiles or door tiles — not outside on grass/path."""
        clock = GameClock()

        # Advance to night slot, then let NPCs settle for 60 ticks
        _advance_to_slot(manager, clock, "night")
        _run_ticks(manager, clock, 60)

        grid = manager.grid
        all_interior = _get_all_interior_tiles(manager.buildings)

        outside_sleepers = []
        sleeping_count = 0

        for npc in manager.npcs:
            if npc.activity != ActivityState.SLEEPING:
                continue
            sleeping_count += 1

            pos = (npc.tile_x, npc.tile_z)
            tile = grid.get_tile(pos[0], pos[1])

            if pos not in all_interior:
                outside_sleepers.append(
                    f"{npc.name}: at ({pos[0]},{pos[1]}) "
                    f"terrain={tile.terrain.value if tile else '?'} "
                    f"interior={tile.interior if tile else '?'} "
                    f"home=({npc.home_x},{npc.home_z})"
                )

        # Most NPCs should be sleeping by now
        assert sleeping_count >= len(manager.npcs) - 2, (
            f"Only {sleeping_count}/{len(manager.npcs)} NPCs are sleeping "
            f"after settling into night — expected nearly all."
        )

        assert len(outside_sleepers) == 0, (
            f"{len(outside_sleepers)} NPCs sleeping outside:\n"
            + "\n".join(outside_sleepers)
        )

    def test_npcs_stay_sleeping_through_night(self, manager):
        """NPCs should remain sleeping throughout the night slot,
        not revert to idle 'finishing up' partway through."""
        clock = GameClock()
        _advance_to_slot(manager, clock, "night")
        _run_ticks(manager, clock, 30)  # initial settling

        # Sample 200 ticks through the rest of the night
        idle_at_night = []

        async def _sample():
            for tick in range(200):
                clock.tick(1.0)
                await manager.tick(clock, 1.0)

                # Stop sampling if we've left the night slot
                if clock.schedule_slot.value != "night":
                    break

                for npc in manager.npcs:
                    if (npc.activity == ActivityState.IDLE
                            and npc.current_action_description == "finishing up"):
                        idle_at_night.append(
                            f"Tick {tick} ({clock.time_string}): "
                            f"{npc.name} idle 'finishing up' at "
                            f"({npc.tile_x},{npc.tile_z})"
                        )

        asyncio.new_event_loop().run_until_complete(_sample())

        # Deduplicate by NPC
        unique_npcs = set()
        for v in idle_at_night:
            name = v.split(": ", 1)[1].split(" idle")[0]
            unique_npcs.add(name)

        assert len(unique_npcs) == 0, (
            f"{len(unique_npcs)} NPCs reverted to idle during night "
            f"({len(idle_at_night)} total instances):\n"
            + "\n".join(idle_at_night[:20])
        )

    def test_sleeping_npcs_at_home(self, manager):
        """Sleeping NPCs should be near their assigned home building."""
        clock = GameClock()
        _advance_to_slot(manager, clock, "night")
        _run_ticks(manager, clock, 60)

        homes = [b for b in manager.buildings if b.building_type == "home"]

        wrong_building = []
        for npc in manager.npcs:
            if npc.activity != ActivityState.SLEEPING:
                continue

            # Find the NPC's assigned home building
            home_building = None
            for h in homes:
                interior = _building_interior_tiles(
                    h.x, h.z, h.width, h.height, h.door_x, h.door_z,
                )
                interior.add((h.door_x, h.door_z))
                if (npc.home_x, npc.home_z) in interior:
                    home_building = h
                    break

            if home_building is None:
                continue

            # Check NPC is within their home building footprint or
            # within 1 tile of the door (just settled)
            in_home = (
                home_building.x <= npc.tile_x < home_building.x + home_building.width
                and home_building.z <= npc.tile_z < home_building.z + home_building.height
            )
            near_door = (
                abs(npc.tile_x - home_building.door_x) <= 1
                and abs(npc.tile_z - home_building.door_z) <= 1
            )

            if not in_home and not near_door:
                wrong_building.append(
                    f"{npc.name}: at ({npc.tile_x},{npc.tile_z}) "
                    f"home=({npc.home_x},{npc.home_z}) "
                    f"home_building={home_building.name} "
                    f"({home_building.x},{home_building.z})"
                )

        assert len(wrong_building) == 0, (
            f"{len(wrong_building)} NPCs sleeping away from home:\n"
            + "\n".join(wrong_building)
        )
