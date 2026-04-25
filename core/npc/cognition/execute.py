"""
Execution module — Stanford model.

Two states only:
  1. WALKING: following a committed path to a destination. Nothing interrupts.
  2. DOING: performing the current action at the current location until the
     subtask timer expires. Then pop next subtask. When queue is empty,
     stay idle at current location until the next slot transition.

No per-tick need overrides. No re-dispatching. No refill loops.
The manager handles slot transitions; execute just follows through.
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

# Seconds of real time before we consider an NPC stuck
STUCK_TIMEOUT = 5.0

# Tiles claimed by NPCs arriving in the current movement_tick.
# Prevents same-tick arrivals from landing on the same tile.
# Cleared each tick by clear_arrival_claims().
_arrived_this_tick: set[tuple[int, int]] = set()


def clear_arrival_claims() -> None:
    """Reset the per-tick arrival claims. Call once per movement_tick."""
    _arrived_this_tick.clear()


_ACTIVITY_STATE_MAP = {
    "idle": ActivityState.IDLE,
    "working": ActivityState.WORKING,
    "eating": ActivityState.EATING,
    "sleeping": ActivityState.SLEEPING,
    "talking": ActivityState.TALKING,
    "gathering": ActivityState.GATHERING,
}


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

    Stanford model — two states:
      - Path exists → follow it (WALKING). Nothing interrupts.
      - No path → tick subtask timer (DOING). Stay put.
    """
    # ── State 1: WALKING — follow committed path ──
    if npc.current_path and npc.path_index < len(npc.current_path):
        old_index = npc.path_index
        _follow_path(npc, real_delta)

        # Stuck detection
        if npc.path_index == old_index and npc.path_index == npc._last_path_index:
            npc._stuck_time += real_delta
        else:
            npc._stuck_time = 0.0
            npc._last_path_index = npc.path_index

        if npc._stuck_time >= STUCK_TIMEOUT:
            logger.debug(
                "%s stuck for %.1fs — abandoning path at (%d,%d)",
                npc.name, npc._stuck_time, npc.tile_x, npc.tile_z,
            )
            _clear_path(npc)
            npc.activity = ActivityState.IDLE
            npc.current_action_description = "stopped — path blocked"
        return

    # Path just completed — handle arrival
    if npc.activity == ActivityState.WALKING:
        _arrive(npc, grid, current_slot, all_npcs or [])
        return

    # ── State 2: DOING — tick subtask timer, stay put ──
    _tick_subtask(npc, real_delta, current_slot)


def _tick_subtask(npc: NPC, real_delta: float, current_slot: str = "") -> None:
    """Advance the subtask timer. When done, pop next. When empty, stay idle."""
    if not npc.current_subtask:
        if npc.subtask_queue:
            _start_next_subtask(npc)
        return

    # 1 real second ≈ 1.2 game minutes (20-min day cycle)
    game_minutes = real_delta * 1.2
    npc.subtask_time_remaining -= game_minutes

    if npc.subtask_time_remaining <= 0:
        if npc.subtask_queue:
            _start_next_subtask(npc)
        else:
            # All subtasks done — stay put until action duration expires
            # and the manager advances to the next schedule entry.
            npc.current_subtask = None
            npc.subtask_time_remaining = 0.0

            # Infer activity from current schedule entry description
            if npc.daily_schedule and npc.schedule_index < len(npc.daily_schedule):
                entry = npc.daily_schedule[npc.schedule_index]
                desc = entry.activity.lower()
                if "sleep" in desc:
                    npc.activity = ActivityState.SLEEPING
                    npc.current_action_description = "sleeping at home"
                elif "eat" in desc or "breakfast" in desc or "lunch" in desc:
                    npc.activity = ActivityState.EATING
                    npc.current_action_description = entry.activity
                elif "work" in desc or "forge" in desc or "tend" in desc or "serve" in desc:
                    npc.activity = ActivityState.WORKING
                    npc.current_action_description = entry.activity
                else:
                    npc.activity = ActivityState.IDLE
                    npc.current_action_description = "finishing up"
            else:
                npc.current_action_description = "finishing up"
                npc.activity = ActivityState.IDLE


def _start_next_subtask(npc: NPC) -> None:
    """Pop the next subtask from the queue and activate it."""
    task = npc.subtask_queue.pop(0)
    npc.current_subtask = task
    npc.subtask_time_remaining = task.duration_minutes
    npc.current_action_description = task.description
    npc.activity = _ACTIVITY_STATE_MAP.get(task.activity_state, ActivityState.IDLE)


def set_activity_for_location(
    npc: NPC,
    current_slot: str,
) -> None:
    """Set activity from current subtask, or fall back to location heuristic.

    Called once on arrival. After this, _tick_subtask handles transitions.
    """
    if npc.current_subtask:
        npc.activity = _ACTIVITY_STATE_MAP.get(
            npc.current_subtask.activity_state, ActivityState.IDLE,
        )
        npc.current_action_description = npc.current_subtask.description
        return

    if npc.subtask_queue:
        _start_next_subtask(npc)
        return

    # Fallback: location-based heuristic
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
        npc.activity = ActivityState.IDLE
        npc.current_action_description = "idle"


def navigate_to(npc: NPC, grid: Grid, target_x: int, target_z: int) -> bool:
    """Set up a path for the NPC to follow. Returns True if path found."""
    if npc.is_at(target_x, target_z):
        return True

    path = find_path(grid, npc.tile_x, npc.tile_z, target_x, target_z)
    if path is None:
        return False

    npc.current_path = path
    npc.path_index = 0
    npc._move_progress = 0.0
    npc._stuck_time = 0.0
    npc.activity = ActivityState.WALKING
    return True


def _clear_path(npc: NPC) -> None:
    """Reset all path state."""
    npc.current_path = []
    npc.path_index = 0
    npc._move_progress = 0.0
    npc._stuck_time = 0.0
    npc.x = float(npc.tile_x)
    npc.z = float(npc.tile_z)


def _follow_path(npc: NPC, real_delta: float) -> None:
    """
    Advance the NPC along their committed path — one tile per tick.

    Stanford model: each step, each NPC moves at most 1 tile.
    Progress accumulates via move_speed * real_delta. When progress >= 1.0,
    the NPC advances one tile and the remainder carries over. This means
    faster NPCs (higher move_speed) cross tiles sooner, creating natural
    desynchronisation without multi-tile jumps.

    Records the single tile moved in npc._tick_trail for client animation.
    """
    npc._tick_trail = []

    if not npc.current_path or npc.path_index >= len(npc.current_path):
        return

    npc._move_progress += npc.move_speed * real_delta

    # At most 1 tile per tick — Stanford approach
    if npc.path_index < len(npc.current_path):
        next_x, next_z = npc.current_path[npc.path_index]
        dx = abs(next_x - npc.tile_x)
        dz = abs(next_z - npc.tile_z)
        step_cost = 1.414 if (dx != 0 and dz != 0) else 1.0

        if npc._move_progress >= step_cost:
            npc.direction = _direction_towards(
                npc.tile_x, npc.tile_z, next_x, next_z,
            )
            npc.x = float(next_x)
            npc.z = float(next_z)
            npc._tick_trail.append((next_x, next_z))
            npc.path_index += 1
            npc._move_progress -= step_cost

    # Path complete — clear it so arrival is detected next tick
    if npc.path_index >= len(npc.current_path):
        npc.current_path = []
        npc.path_index = 0
        npc._move_progress = 0.0


def _arrive(
    npc: NPC,
    grid: Grid,
    current_slot: str,
    all_npcs: list[NPC],
) -> None:
    """Handle arrival at destination. Nudge if tile occupied, then settle.

    When nudged to a free tile, persist that position on the current
    schedule entry so future re-dispatches (post-conversation, etc.)
    don't send the NPC back to the original occupied tile.

    Uses a module-level set (_arrived_this_tick) so that when multiple
    NPCs arrive in the same movement_tick loop, later arrivals see
    tiles claimed by earlier arrivals. The set is cleared each tick
    by calling clear_arrival_claims().
    """
    from core.world.spatial_awareness import get_occupied_tiles, find_rest_tile

    _clear_path(npc)

    # Build occupied set: resting NPCs + tiles claimed by earlier
    # arrivals in this same tick
    occupied = get_occupied_tiles(all_npcs) | _arrived_this_tick

    if (npc.tile_x, npc.tile_z) in occupied:
        new_x, new_z = find_rest_tile(
            npc.tile_x, npc.tile_z, grid, occupied,
            exclude_npc_id=npc.npc_id, npcs=all_npcs,
        )
        if (new_x, new_z) != (npc.tile_x, npc.tile_z):
            npc._tick_trail = [(new_x, new_z)]
            npc.x = float(new_x)
            npc.z = float(new_z)

    # Record this tile so later arrivals in the same tick see it
    _arrived_this_tick.add((npc.tile_x, npc.tile_z))

    # Persist final position on current schedule entry so re-dispatches
    # (post-conversation, advance) use the resolved tile, not the original.
    if npc.daily_schedule and npc.schedule_index < len(npc.daily_schedule):
        entry = npc.daily_schedule[npc.schedule_index]
        entry.target_x = npc.tile_x
        entry.target_z = npc.tile_z

    set_activity_for_location(npc, current_slot)
    logger.info(
        "ARRIVE %s: at (%d,%d) → %s [%s]",
        npc.name, npc.tile_x, npc.tile_z,
        npc.current_action_description, npc.activity.value,
    )


def _direction_towards(from_x: int, from_z: int, to_x: int, to_z: int) -> Direction:
    """Determine facing direction for movement."""
    dx = to_x - from_x
    dz = to_z - from_z
    if dx == 0 and dz == 0:
        return Direction.SOUTH
    if abs(dx) >= abs(dz):
        return Direction.EAST if dx > 0 else Direction.WEST
    return Direction.NORTH if dz > 0 else Direction.SOUTH
