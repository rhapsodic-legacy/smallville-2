"""
Emergency movement overrides.

Standalone functions extracted from NPCManager for force-navigating
NPCs during emergencies (kaiju, fire, rally, etc.).
"""

from __future__ import annotations

import random
from typing import Callable

from core.npc.models import NPC
from core.npc.cognition.execute import navigate_to
from core.npc.cognition.converse import end_conversation
from core.world.spatial_awareness import get_occupied_tiles, find_rest_tile


def spread_destination(
    npc: NPC,
    target_x: int,
    target_z: int,
    grid,
    npcs: list[NPC],
) -> tuple[int, int]:
    """Find nearest passable unoccupied tile near the target.

    Thin wrapper around the spatial awareness module, kept here so
    emergency functions can call it without a manager reference.
    """
    occupied = get_occupied_tiles(npcs)
    return find_rest_tile(
        target_x, target_z, grid, occupied,
        exclude_npc_id=npc.npc_id, npcs=npcs,
    )


def flee_destination(
    npc: NPC,
    danger_x: int,
    danger_z: int,
    grid,
    npcs: list[NPC],
    rng: random.Random,
) -> tuple[int, int]:
    """Calculate a destination away from the danger point.

    Moves in the opposite direction, scaled by a safe distance.
    """
    dx = npc.x - danger_x
    dz = npc.z - danger_z
    dist = abs(dx) + abs(dz)

    if dist == 0:
        # On top of the danger — pick a random direction
        dx = rng.choice([-1, 1])
        dz = rng.choice([-1, 1])
        dist = 2

    # Flee 15 tiles away from danger
    scale = 15 / max(dist, 1)
    flee_x = npc.x + int(dx * scale)
    flee_z = npc.z + int(dz * scale)

    # Clamp to grid bounds
    min_x, min_z, max_x, max_z = grid.bounds
    flee_x = max(min_x, min(flee_x, max_x))
    flee_z = max(min_z, min(flee_z, max_z))

    return spread_destination(npc, flee_x, flee_z, grid, npcs)


def force_navigate_all(
    npcs: list[NPC],
    grid,
    pending_departures: dict,
    rng: random.Random,
    get_npc: Callable[[str], NPC | None],
    target_x: int,
    target_z: int,
    description: str = "emergency movement",
    flee_from: bool = False,
    filter_fn: Callable[[NPC], bool] | None = None,
) -> int:
    """Force all (or filtered) NPCs to navigate immediately.

    Bypasses stagger. Use for emergencies (kaiju, fire, rally).
    If flee_from=True, NPCs run AWAY from the target coords.
    Returns number of NPCs that started navigating.
    """
    count = 0

    for npc in npcs:
        if npc.cognition_tier >= 4:
            continue
        if filter_fn and not filter_fn(npc):
            continue

        # Cancel any pending staggered departure
        pending_departures.pop(npc.npc_id, None)

        # End conversations immediately
        if npc.conversation_partner:
            other = get_npc(npc.conversation_partner)
            if other:
                end_conversation(npc, other)

        if flee_from:
            dest = flee_destination(npc, target_x, target_z, grid, npcs, rng)
        else:
            dest = spread_destination(npc, target_x, target_z, grid, npcs)

        if navigate_to(npc, grid, dest[0], dest[1]):
            npc.current_action_description = description
            count += 1

    return count


def force_navigate_npc(
    npc_id: str,
    npcs: list[NPC],
    grid,
    pending_departures: dict,
    get_npc: Callable[[str], NPC | None],
    target_x: int,
    target_z: int,
    description: str = "",
) -> bool:
    """Force a single NPC to navigate immediately, bypassing stagger."""
    npc = get_npc(npc_id)
    if not npc or npc.cognition_tier >= 4:
        return False

    pending_departures.pop(npc_id, None)
    tx, tz = spread_destination(npc, target_x, target_z, grid, npcs)
    if navigate_to(npc, grid, tx, tz):
        if description:
            npc.current_action_description = description
        return True
    return False
