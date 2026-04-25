"""
Spatial awareness — the foundational layer for NPC positioning.

Answers the question: "given the world and all NPCs, where should
this NPC come to rest?" NPCs may pass through each other freely
while walking, but must NEVER share a tile when at rest.

All rest-placement and conversation-positioning calls go through
this module. It is the single source of truth for safe tile selection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid

logger = logging.getLogger(__name__)

# Maximum search radius when looking for a free tile
MAX_SEARCH_RADIUS = 8


def _is_walking(npc: NPC) -> bool:
    """Check if NPC is walking. Lazy import to avoid circular dependency."""
    from core.npc.models import ActivityState
    return npc.activity == ActivityState.WALKING


def get_occupied_tiles(npcs: list[NPC]) -> set[tuple[int, int]]:
    """
    Return the set of tiles occupied by resting (non-walking) NPCs.

    Walking NPCs are excluded — they are transient and may pass
    through any tile freely.
    """
    return {
        (npc.tile_x, npc.tile_z)
        for npc in npcs
        if not _is_walking(npc)
    }


def find_rest_tile(
    target_x: int,
    target_z: int,
    grid: Grid,
    occupied: set[tuple[int, int]],
    exclude_npc_id: str = "",
    npcs: list[NPC] | None = None,
) -> tuple[int, int]:
    """
    Find the nearest passable, unoccupied tile to the target.

    If the target itself is free, returns it directly.
    Otherwise spirals outward up to MAX_SEARCH_RADIUS.
    Falls back to the original target if nothing is found
    (shouldn't happen on a reasonably sized map).

    If exclude_npc_id is set and npcs is provided, that NPC's
    current position is excluded from the occupied set (useful
    when re-seating an NPC that is already counted as occupying
    a tile).
    """
    if npcs and exclude_npc_id:
        occupied = occupied.copy()
        # Only discard the excluded NPC's position if no other resting
        # NPC is also on that tile
        for npc in npcs:
            if npc.npc_id == exclude_npc_id and not _is_walking(npc):
                others_on_tile = any(
                    o.npc_id != exclude_npc_id
                    and not _is_walking(o)
                    and o.tile_x == npc.tile_x and o.tile_z == npc.tile_z
                    for o in npcs
                )
                if not others_on_tile:
                    occupied.discard((npc.tile_x, npc.tile_z))

    # Check the target first
    tile = grid.get_tile(target_x, target_z)
    if tile and tile.is_passable and (target_x, target_z) not in occupied:
        return (target_x, target_z)

    # If target is inside a building, prefer tiles in the same building
    # before spiralling outward to the exterior.
    target_arena = tile.arena if tile and tile.interior else ""
    if target_arena:
        for radius in range(1, MAX_SEARCH_RADIUS + 1):
            best = _best_in_ring(
                target_x, target_z, radius, grid, occupied,
                require_arena=target_arena,
            )
            if best:
                return best

        # No free interior tiles — exit through the door tile.
        # Find the door: the arena-tagged tile on the building perimeter
        # that is passable but NOT interior.
        for radius in range(1, MAX_SEARCH_RADIUS + 1):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dz) != radius:
                        continue
                    tx, tz = target_x + dx, target_z + dz
                    t = grid.get_tile(tx, tz)
                    if (t and t.is_passable and not t.interior
                            and t.arena == target_arena
                            and (tx, tz) not in occupied):
                        return (tx, tz)

        # Door occupied too — find tile just outside the door
        for radius in range(1, MAX_SEARCH_RADIUS + 1):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dz) != radius:
                        continue
                    tx, tz = target_x + dx, target_z + dz
                    t = grid.get_tile(tx, tz)
                    if (t and t.is_passable and not t.interior
                            and t.arena == target_arena):
                        # Find free tile adjacent to the door (outside)
                        for ddx, ddz in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                            ox, oz = tx + ddx, tz + ddz
                            ot = grid.get_tile(ox, oz)
                            if (ot and ot.is_passable and not ot.interior
                                    and (ox, oz) not in occupied):
                                return (ox, oz)

    # Spiral outward ring by ring (no arena restriction)
    for radius in range(1, MAX_SEARCH_RADIUS + 1):
        best = _best_in_ring(target_x, target_z, radius, grid, occupied)
        if best:
            return best

    # Exhausted search — return original target as last resort
    logger.warning(
        "No free rest tile found near (%d, %d) within radius %d",
        target_x, target_z, MAX_SEARCH_RADIUS,
    )
    return (target_x, target_z)


def find_conversation_positions(
    npc_a: NPC,
    npc_b: NPC,
    grid: Grid,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Find two adjacent, unoccupied tiles where two NPCs can converse.

    Strategy: pick the midpoint between the two NPCs, then find two
    adjacent free tiles near that midpoint. If the NPCs are already
    adjacent, keep them where they are.
    """
    dist = abs(npc_a.x - npc_b.x) + abs(npc_a.z - npc_b.z)

    # Already adjacent (distance 1) — just confirm tiles are free
    if dist == 1:
        occ_excl = occupied - {(npc_a.x, npc_a.z), (npc_b.x, npc_b.z)}
        a_ok = (npc_a.x, npc_a.z) not in occ_excl
        b_ok = (npc_b.x, npc_b.z) not in occ_excl
        if a_ok and b_ok:
            return ((npc_a.x, npc_a.z), (npc_b.x, npc_b.z))

    # Already on same tile (shouldn't happen, but handle it)
    if dist == 0:
        # Move B to an adjacent free tile
        occ_excl = occupied - {(npc_a.x, npc_a.z), (npc_b.x, npc_b.z)}
        pos_b = _find_adjacent_free(npc_a.x, npc_a.z, grid, occ_excl)
        if pos_b:
            return ((npc_a.x, npc_a.z), pos_b)

    # General case: find the midpoint, then find a pair of adjacent
    # free tiles near it
    mid_x = (npc_a.x + npc_b.x) // 2
    mid_z = (npc_a.z + npc_b.z) // 2

    # Exclude both NPCs' current positions from occupied
    # (they're about to move)
    occ_excl = occupied - {(npc_a.x, npc_a.z), (npc_b.x, npc_b.z)}

    pair = _find_adjacent_pair(mid_x, mid_z, grid, occ_excl)
    if pair:
        return pair

    # Fallback: move B towards A
    pos_a = find_rest_tile(
        npc_a.x, npc_a.z, grid, occ_excl,
    )
    occ_with_a = occ_excl | {pos_a}
    pos_b = _find_adjacent_free(pos_a[0], pos_a[1], grid, occ_with_a)
    if pos_b:
        return (pos_a, pos_b)

    # Last resort: keep current positions
    return ((npc_a.tile_x, npc_a.tile_z), (npc_b.tile_x, npc_b.tile_z))


def resolve_overlaps(
    npcs: list[NPC],
    grid: Grid,
) -> int:
    """
    Safety net: scan all resting NPCs and nudge any that share a tile.

    Iterates up to 3 passes so that nudging NPC A onto NPC B's tile
    is caught and resolved in the same call. Nudged NPCs get trail
    data so the client animates the separation. Returns total NPCs moved.

    The player NPC is never nudged: the player's position is
    input-authoritative and silently teleporting the avatar because
    they stood on top of another NPC would feel like the game fighting
    the player. NPCs around the player are the ones that move aside.
    """
    total_moved = 0

    for _pass in range(3):
        # Rebuild occupied set fresh each pass to see current positions
        occupied = get_occupied_tiles(npcs)

        # Build map of tile → list of resting NPCs on that tile.
        # The player is kept OUT of the candidate list so they never
        # get moved by overlap resolution — but their tile is still in
        # `occupied`, which causes NPCs to be nudged away from the player.
        tile_npcs: dict[tuple[int, int], list[NPC]] = {}
        for npc in npcs:
            if _is_walking(npc):
                continue
            if getattr(npc, "npc_id", "") == "player":
                continue
            pos = (npc.tile_x, npc.tile_z)
            tile_npcs.setdefault(pos, []).append(npc)

        moved_this_pass = 0
        for pos, stacked in tile_npcs.items():
            if len(stacked) <= 1:
                continue

            # First NPC stays, others get nudged
            for extra_npc in stacked[1:]:
                # Exclude this NPC's current position so it can be moved
                search_occupied = occupied - {(extra_npc.tile_x, extra_npc.tile_z)}
                # But keep the position of the NPC that's staying
                search_occupied.add(pos)

                # Outdoor NPCs search wider (user-requested behaviour)
                tile = grid.get_tile(pos[0], pos[1])
                is_indoor = tile and tile.interior

                new_pos = find_rest_tile(
                    pos[0], pos[1], grid, search_occupied,
                )
                if new_pos != pos:
                    # Set trail so client animates the nudge
                    if not hasattr(extra_npc, '_tick_trail'):
                        extra_npc._tick_trail = []
                    extra_npc._tick_trail.append(
                        (new_pos[0], new_pos[1])
                    )
                    extra_npc.x = float(new_pos[0])
                    extra_npc.z = float(new_pos[1])
                    occupied.add(new_pos)
                    moved_this_pass += 1

                    # Persist the nudged position onto the current
                    # schedule entry so re-dispatches don't send
                    # the NPC back to the original occupied tile.
                    if (extra_npc.daily_schedule
                            and extra_npc.schedule_index
                            < len(extra_npc.daily_schedule)):
                        entry = extra_npc.daily_schedule[
                            extra_npc.schedule_index
                        ]
                        entry.target_x = new_pos[0]
                        entry.target_z = new_pos[1]

                    logger.debug(
                        "Nudged %s from (%d,%d) to (%d,%d) (pass %d)",
                        extra_npc.name, pos[0], pos[1],
                        new_pos[0], new_pos[1], _pass + 1,
                    )

        total_moved += moved_this_pass
        if moved_this_pass == 0:
            break  # Stable — no overlaps remain

    return total_moved


# ---------- Internal helpers ----------


def _best_in_ring(
    cx: int, cz: int, radius: int,
    grid: Grid,
    occupied: set[tuple[int, int]],
    require_arena: str = "",
) -> tuple[int, int] | None:
    """
    Find the best free passable tile on the ring at `radius`
    from (cx, cz). Prefers tiles closest to the centre of the ring
    (i.e. directly N/S/E/W before diagonals).

    If require_arena is set, only considers tiles in that arena
    (keeps NPCs inside the same building).
    """
    candidates: list[tuple[int, int, int]] = []  # (x, z, manhattan_from_target)

    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            # Only check the perimeter of the ring
            if abs(dx) != radius and abs(dz) != radius:
                continue
            tx, tz = cx + dx, cz + dz
            tile = grid.get_tile(tx, tz)
            if tile and tile.is_passable and (tx, tz) not in occupied:
                if require_arena and tile.arena != require_arena:
                    continue
                dist = abs(dx) + abs(dz)
                candidates.append((tx, tz, dist))

    if not candidates:
        return None

    # Sort by distance from target (prefer cardinal directions)
    candidates.sort(key=lambda c: c[2])
    return (candidates[0][0], candidates[0][1])


def _find_adjacent_free(
    x: int, z: int,
    grid: Grid,
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the nearest passable, unoccupied tile adjacent to (x, z)."""
    for tile in grid.get_passable_neighbours(x, z):
        if (tile.x, tile.z) not in occupied:
            return (tile.x, tile.z)
    return None


def _find_adjacent_pair(
    cx: int, cz: int,
    grid: Grid,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Find a pair of adjacent, passable, unoccupied tiles near (cx, cz).

    Searches outward from the midpoint. Returns (pos_a, pos_b) or None.
    """
    for radius in range(0, MAX_SEARCH_RADIUS):
        tiles = grid.tiles_in_radius(cx, cz, radius)
        for tile_a in tiles:
            if not tile_a.is_passable:
                continue
            if (tile_a.x, tile_a.z) in occupied:
                continue
            # Look for an adjacent free tile
            for tile_b in grid.get_passable_neighbours(tile_a.x, tile_a.z):
                if (tile_b.x, tile_b.z) not in occupied:
                    return ((tile_a.x, tile_a.z), (tile_b.x, tile_b.z))
    return None
