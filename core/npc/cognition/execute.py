"""
Execution module.

Translates NPC plans into concrete actions: pathfinding, movement,
activity state changes, and object interactions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.npc.models import ActivityState, Direction
from core.world.pathfinding import find_path

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid
    from core.world.generator import PlacedBuilding

logger = logging.getLogger(__name__)

# Minimum distance between two resting NPCs (0 = same tile forbidden)
MIN_SEPARATION = 1

# Seconds of real time before we consider an NPC stuck
STUCK_TIMEOUT = 5.0


def execute_tick(
    npc: NPC,
    grid: Grid,
    buildings: list[PlacedBuilding],
    current_slot: str,
    real_delta: float,
    all_npcs: list[NPC] | None = None,
) -> None:
    """
    Run one execution tick for an NPC.

    Handles movement along paths, activity transitions, and
    need-based overrides (sleep when exhausted, eat when starving).
    all_npcs is passed so arrival can check for occupied tiles.
    """
    # Need-based overrides
    override = _check_need_overrides(npc, current_slot, buildings)
    if override:
        _set_destination(npc, grid, override[0], override[1])
        npc.current_action_description = override[2]

    # If we have a path, follow it — but detect stuck NPCs
    if npc.current_path and npc.path_index < len(npc.current_path):
        old_index = npc.path_index
        _follow_path(npc, real_delta)

        # Track stuck time: if path_index hasn't advanced, accumulate
        if npc.path_index == old_index and npc.path_index == npc._last_path_index:
            npc._stuck_time += real_delta
        else:
            npc._stuck_time = 0.0
            npc._last_path_index = npc.path_index

        # Abandon path if stuck too long
        if npc._stuck_time >= STUCK_TIMEOUT:
            logger.debug(
                "%s stuck for %.1fs — abandoning path at (%d,%d)",
                npc.name, npc._stuck_time, npc.tile_x, npc.tile_z,
            )
            npc.current_path = []
            npc.path_index = 0
            npc._move_progress = 0.0
            npc._stuck_time = 0.0
            npc.x = float(npc.tile_x)
            npc.z = float(npc.tile_z)
            npc.activity = ActivityState.IDLE
            npc.current_action_description = "gave up walking — path blocked"
        return

    # Path complete or no path — update activity based on schedule
    if npc.activity == ActivityState.WALKING:
        _arrive_at_destination(npc, grid, current_slot, all_npcs or [])
        return

    # Tick sub-task timer (only when not walking)
    _tick_subtask(npc, real_delta, current_slot)


def _check_need_overrides(
    npc: NPC,
    current_slot: str,
    buildings: list[PlacedBuilding],
) -> tuple[int, int, str] | None:
    """
    Check if critical needs should override the current plan.

    Returns (target_x, target_z, description) or None.
    """
    # Exhausted → go home to sleep
    if npc.energy < 0.1 and npc.activity != ActivityState.SLEEPING:
        return (npc.home_x, npc.home_z, "going home to rest — exhausted")

    # Starving → find tavern
    if npc.hunger > 0.9 and npc.activity != ActivityState.EATING:
        for b in buildings:
            if b.building_type == "tavern":
                return (b.door_x, b.door_z, "going to the tavern — starving")

    return None


_ACTIVITY_STATE_MAP = {
    "idle": ActivityState.IDLE,
    "working": ActivityState.WORKING,
    "eating": ActivityState.EATING,
    "sleeping": ActivityState.SLEEPING,
    "talking": ActivityState.TALKING,
    "gathering": ActivityState.GATHERING,
}


def _tick_subtask(npc: NPC, real_delta: float, current_slot: str) -> None:
    """Advance the sub-task timer. When a sub-task expires, pop the next one."""
    if not npc.current_subtask:
        # No sub-task active — try to start one from the queue
        if npc.subtask_queue:
            _start_next_subtask(npc)
        return

    # Convert real_delta to approximate game minutes for sub-task timing.
    # Default time scale: 1 real second ~= 1.2 game minutes (20-min day).
    game_minutes = real_delta * 1.2
    npc.subtask_time_remaining -= game_minutes

    if npc.subtask_time_remaining <= 0:
        # Current sub-task complete
        if npc.subtask_queue:
            _start_next_subtask(npc)
        else:
            # All sub-tasks for this slot done — wait for next decomposition
            npc.current_subtask = None
            npc.subtask_time_remaining = 0.0
            npc.current_action_description = "finishing up"
            npc.activity = ActivityState.IDLE


def _start_next_subtask(npc: NPC) -> None:
    """Pop the next sub-task from the queue and activate it."""
    task = npc.subtask_queue.pop(0)
    npc.current_subtask = task
    npc.subtask_time_remaining = task.duration_minutes
    npc.current_action_description = task.description
    npc.activity = _ACTIVITY_STATE_MAP.get(task.activity_state, ActivityState.IDLE)


def set_activity_for_location(
    npc: NPC,
    current_slot: str,
) -> None:
    """Set activity from the current sub-task, or fall back to location heuristic.

    If the NPC has an active sub-task, use its activity state and description.
    Otherwise, infer from proximity to home/work and the time slot.
    """
    # Sub-task system takes priority when active
    if npc.current_subtask:
        npc.activity = _ACTIVITY_STATE_MAP.get(
            npc.current_subtask.activity_state, ActivityState.IDLE,
        )
        npc.current_action_description = npc.current_subtask.description
        return

    # Start a queued sub-task if available
    if npc.subtask_queue:
        _start_next_subtask(npc)
        return

    # Legacy fallback: location-based heuristic
    if npc.distance_to(npc.home_x, npc.home_z) <= 1:
        if current_slot == "night":
            npc.activity = ActivityState.SLEEPING
            npc.current_action_description = "sleeping at home"
        elif current_slot == "early_morning":
            npc.activity = ActivityState.EATING
            npc.current_action_description = "having breakfast"
        else:
            npc.activity = ActivityState.IDLE
            npc.current_action_description = "resting at home"

    elif npc.distance_to(npc.work_x, npc.work_z) <= 1:
        npc.activity = ActivityState.WORKING
        npc.current_action_description = f"working as {npc.occupation}"

    else:
        entry = npc.get_current_schedule_entry(current_slot)
        if entry:
            npc.current_action_description = entry.activity
            kw = entry.activity.lower()
            if "eat" in kw or "lunch" in kw:
                npc.activity = ActivityState.EATING
            elif "sleep" in kw or "rest" in kw:
                npc.activity = ActivityState.SLEEPING
            elif "socialise" in kw or "tavern" in kw:
                npc.activity = ActivityState.TALKING
            else:
                npc.activity = ActivityState.IDLE
        else:
            npc.activity = ActivityState.IDLE
            npc.current_action_description = "idle"


def navigate_to(npc: NPC, grid: Grid, target_x: int, target_z: int) -> bool:
    """
    Set up a path for the NPC to follow to the target.

    Returns True if a path was found, False otherwise.
    """
    return _set_destination(npc, grid, target_x, target_z)


def _set_destination(
    npc: NPC,
    grid: Grid,
    target_x: int,
    target_z: int,
    description: str = "",
) -> bool:
    """Calculate path and start the NPC walking."""
    if npc.is_at(target_x, target_z):
        return True

    path = find_path(grid, npc.tile_x, npc.tile_z, target_x, target_z)
    if path is None:
        logger.debug(
            "%s could not find path from (%d,%d) to (%d,%d)",
            npc.name, npc.tile_x, npc.tile_z, target_x, target_z,
        )
        return False

    npc.current_path = path
    npc.path_index = 0
    npc.activity = ActivityState.WALKING
    if description:
        npc.current_action_description = description

    return True


def _follow_path(npc: NPC, real_delta: float) -> None:
    """
    Advance the NPC along their current path using discrete tile positions.

    Records tiles traversed this tick in npc._tick_trail so the client
    can walk through them smoothly (Stanford approach — no full path sent).
    """
    npc._tick_trail = []

    if not npc.current_path or npc.path_index >= len(npc.current_path):
        return

    progress = npc._move_progress + npc.move_speed * real_delta

    while npc.path_index < len(npc.current_path):
        next_x, next_z = npc.current_path[npc.path_index]
        dx = abs(next_x - npc.tile_x)
        dz = abs(next_z - npc.tile_z)
        step_cost = 1.414 if (dx != 0 and dz != 0) else 1.0
        if progress < step_cost:
            break
        npc.direction = _direction_towards(npc.tile_x, npc.tile_z, next_x, next_z)
        npc.x = float(next_x)
        npc.z = float(next_z)
        npc._tick_trail.append((next_x, next_z))
        npc.path_index += 1
        progress -= step_cost

    # Stay snapped to the last reached waypoint — no fractional positions.
    npc._move_progress = progress

    # Check if path is complete
    if npc.path_index >= len(npc.current_path):
        npc.current_path = []
        npc.path_index = 0
        npc._move_progress = 0.0


def _arrive_at_destination(
    npc: NPC,
    grid: Grid,
    current_slot: str,
    all_npcs: list[NPC],
) -> None:
    """Handle arrival at the path destination.

    Before settling, checks if the arrival tile is occupied by another
    resting NPC. If so, nudges to the nearest free passable tile.
    """
    from core.world.spatial_awareness import get_occupied_tiles, find_rest_tile

    npc.current_path = []
    npc.path_index = 0
    npc._move_progress = 0.0
    # Snap to grid on arrival
    npc.x = float(npc.tile_x)
    npc.z = float(npc.tile_z)

    # Check if this tile is already occupied by a resting NPC.
    # Do NOT discard our own position — we're still WALKING so
    # get_occupied_tiles already excludes us. Another resting NPC
    # on this tile must remain visible in the occupied set.
    occupied = get_occupied_tiles(all_npcs)
    if (npc.tile_x, npc.tile_z) in occupied:
        new_x, new_z = find_rest_tile(
            npc.tile_x, npc.tile_z, grid, occupied,
            exclude_npc_id=npc.npc_id, npcs=all_npcs,
        )
        npc.x = float(new_x)
        npc.z = float(new_z)

    set_activity_for_location(npc, current_slot)


def _direction_towards(from_x: int, from_z: int, to_x: int, to_z: int) -> Direction:
    """Determine facing direction for movement (including diagonal steps)."""
    dx = to_x - from_x
    dz = to_z - from_z
    if dx == 0 and dz == 0:
        return Direction.SOUTH
    # Favour the axis with greater displacement
    if abs(dx) >= abs(dz):
        return Direction.EAST if dx > 0 else Direction.WEST
    return Direction.NORTH if dz > 0 else Direction.SOUTH
