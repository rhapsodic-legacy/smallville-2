"""
A* pathfinding on the spatial grid.

Finds shortest walkable path between two tile coordinates.
Supports 4-directional and 8-directional movement.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.world.grid import Grid


def _heuristic(x1: int, z1: int, x2: int, z2: int, diagonal: bool) -> float:
    """Estimated distance between two points."""
    dx = abs(x1 - x2)
    dz = abs(z1 - z2)
    if diagonal:
        # Chebyshev distance
        return max(dx, dz)
    # Manhattan distance
    return dx + dz


def _movement_cost(dx: int, dz: int) -> float:
    """Cost to move one step. Diagonal is ~1.41."""
    if dx != 0 and dz != 0:
        return 1.414
    return 1.0


def find_path(
    grid: Grid,
    start_x: int,
    start_z: int,
    goal_x: int,
    goal_z: int,
    diagonal: bool = True,
    max_steps: int = 500,
) -> list[tuple[int, int]] | None:
    """
    A* pathfinding from (start_x, start_z) to (goal_x, goal_z).

    Returns a list of (x, z) coordinates from start to goal (inclusive),
    or None if no path exists within max_steps iterations.

    If the goal tile is not passable (e.g. inside a building), the path
    terminates at the nearest passable neighbour instead — NPCs stop at
    the door rather than walking through walls.
    """
    if not grid.in_bounds(start_x, start_z) or not grid.in_bounds(goal_x, goal_z):
        return None

    # If goal is non-passable, redirect to nearest passable neighbour
    goal_x, goal_z = _resolve_passable_goal(grid, goal_x, goal_z)

    start = (start_x, start_z)
    goal = (goal_x, goal_z)

    if start == goal:
        return [start]

    goal_tile = grid.get_tile(goal_x, goal_z)
    if goal_tile is None:
        return None

    # If goal is inside a building, allow routing through that building's
    # interior tiles. NPCs must not shortcut through other buildings.
    # Also allow routing through the start building so NPCs can leave.
    goal_arena = goal_tile.arena if goal_tile.interior else ""
    start_tile = grid.get_tile(start_x, start_z)
    start_arena = start_tile.arena if (start_tile and start_tile.interior) else ""

    # Open set: (f_score, counter, (x, z))
    counter = 0
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    steps = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if diagonal:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set and steps < max_steps:
        steps += 1
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            return _reconstruct(came_from, current)

        cx, cz = current

        for dx, dz in directions:
            nx, nz = cx + dx, cz + dz
            neighbour = (nx, nz)

            tile = grid.get_tile(nx, nz)
            if tile is None or not tile.is_passable:
                continue
            # Interior tiles block routing — no shortcuts through buildings.
            # Exception: tiles in the same building as the goal or start.
            if tile.interior and not (
                (goal_arena and tile.arena == goal_arena)
                or (start_arena and tile.arena == start_arena)
            ):
                continue

            # Diagonal corner-cutting prevention: both adjacent cardinal
            # tiles must be passable AND non-wall so NPCs never clip
            # building corners or door-adjacent walls.
            if dx != 0 and dz != 0:
                t_horiz = grid.get_tile(cx + dx, cz)
                t_vert = grid.get_tile(cx, cz + dz)
                if (not t_horiz or not t_horiz.is_passable
                        or not t_vert or not t_vert.is_passable):
                    continue

            cost = _movement_cost(dx, dz)
            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f = tentative_g + _heuristic(nx, nz, goal_x, goal_z, diagonal)
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbour))

    return None  # No path found


def _resolve_passable_goal(
    grid: Grid, goal_x: int, goal_z: int,
) -> tuple[int, int]:
    """
    If the goal tile is non-passable, find the nearest passable tile.

    Searches outward in rings up to radius 5. This ensures NPCs
    stop at doors rather than pathing into building interiors.
    """
    tile = grid.get_tile(goal_x, goal_z)
    if tile is not None and tile.is_passable:
        return (goal_x, goal_z)

    # Spiral outward to find a passable tile
    for radius in range(1, 6):
        best: tuple[int, int] | None = None
        best_dist = float("inf")
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dz) != radius:
                    continue  # perimeter only
                nx, nz = goal_x + dx, goal_z + dz
                t = grid.get_tile(nx, nz)
                if t is not None and t.is_passable:
                    dist = abs(dx) + abs(dz)
                    if dist < best_dist:
                        best = (nx, nz)
                        best_dist = dist
        if best is not None:
            return best

    return (goal_x, goal_z)  # fallback — shouldn't happen on a normal map


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    """Reconstruct path by walking backwards through came_from."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def smooth_path(
    grid: Grid,
    path: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Line-of-sight path smoothing (string-pulling).

    Removes unnecessary waypoints so NPCs walk in natural straight lines
    instead of grid-locked staircases. Walks through the path and skips
    intermediate waypoints whenever there's a clear line of sight.
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    current = 0

    while current < len(path) - 1:
        # Look as far ahead as possible with clear line of sight
        farthest = current + 1
        for candidate in range(len(path) - 1, current + 1, -1):
            if _has_line_of_sight(grid, path[current], path[candidate]):
                farthest = candidate
                break
        smoothed.append(path[farthest])
        current = farthest

    return smoothed


def _has_line_of_sight(
    grid: Grid,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    """
    Check if there's a clear walkable line between two tiles.

    Uses Bresenham's line algorithm to check every tile along the line,
    plus adjacent tiles to ensure NPCs won't clip building corners
    during sub-tile interpolation.
    """
    x0, z0 = start
    x1, z1 = end
    dx = abs(x1 - x0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sz = 1 if z1 > z0 else -1
    err = dx - dz

    while True:
        # Check the tile itself
        tile = grid.get_tile(x0, z0)
        if tile is None or not tile.is_passable:
            return False

        # Check tiles perpendicular to movement for corner safety.
        # Without this, smoothed diagonal paths can clip building corners
        # during float interpolation.
        if x0 != x1 and z0 != z1:
            # Moving diagonally — check both adjacent tiles
            t1 = grid.get_tile(x0 + sx, z0)
            t2 = grid.get_tile(x0, z0 + sz)
            # If both adjacent tiles are blocked, the diagonal is not safe
            if (t1 is None or not t1.is_passable) and (t2 is None or not t2.is_passable):
                return False

        if x0 == x1 and z0 == z1:
            break
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x0 += sx
        if e2 < dx:
            err += dx
            z0 += sz

    return True


def path_length(path: list[tuple[int, int]] | None) -> float:
    """Calculate the total length of a path. Returns inf if path is None."""
    if path is None:
        return float("inf")
    total = 0.0
    for i in range(1, len(path)):
        dx = abs(path[i][0] - path[i - 1][0])
        dz = abs(path[i][1] - path[i - 1][1])
        total += _movement_cost(dx, dz)
    return total
