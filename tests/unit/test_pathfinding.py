"""Tests for A* pathfinding."""

from core.world.grid import Grid, Terrain
from core.world.pathfinding import find_path, path_length, _resolve_passable_goal


class TestPathfinding:
    def test_straight_line_path(self):
        grid = Grid(20, 20)
        path = find_path(grid, 0, 0, 5, 0)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (5, 0)
        assert len(path) == 6

    def test_path_around_obstacle(self):
        grid = Grid(20, 20)
        # Wall blocking direct horizontal path
        for z in range(-3, 4):
            grid.set_terrain(3, z, Terrain.WATER)
        path = find_path(grid, 0, 0, 6, 0)
        assert path is not None
        assert path[-1] == (6, 0)
        # Path should go around the wall
        assert len(path) > 7

    def test_no_path_when_fully_blocked(self):
        grid = Grid(10, 10)
        # Surround the goal with water
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue
                grid.set_terrain(3 + dx, 3 + dz, Terrain.WATER)
        path = find_path(grid, 0, 0, 3, 3)
        assert path is None

    def test_same_start_and_goal(self):
        grid = Grid(10, 10)
        path = find_path(grid, 0, 0, 0, 0)
        assert path == [(0, 0)]

    def test_diagonal_path(self):
        grid = Grid(20, 20)
        path = find_path(grid, 0, 0, 3, 3, diagonal=True)
        assert path is not None
        assert len(path) == 4  # diagonal moves

    def test_out_of_bounds_returns_none(self):
        grid = Grid(10, 10)
        path = find_path(grid, 0, 0, 100, 100)
        assert path is None

    def test_path_length_calculation(self):
        grid = Grid(20, 20)
        path = find_path(grid, 0, 0, 5, 0)
        assert path_length(path) == 5.0

    def test_path_length_none(self):
        assert path_length(None) == float("inf")


class TestNonPassableGoalRedirect:
    """NPCs must stop at doors, not walk into building interiors."""

    def test_goal_inside_building_redirects_to_door(self):
        grid = Grid(20, 20)
        # Create a 3x3 non-passable building block
        for bx in range(4, 7):
            for bz in range(4, 7):
                tile = grid.get_tile(bx, bz)
                tile.walkable = False
        # Make the door passable (tile just south of the building)
        door = grid.get_tile(5, 3)
        assert door.is_passable  # grass, walkable by default

        # Path to centre of building should redirect to nearest passable
        path = find_path(grid, 0, 0, 5, 5)
        assert path is not None
        last = path[-1]
        last_tile = grid.get_tile(last[0], last[1])
        assert last_tile.is_passable, "Path must end on passable tile"

    def test_passable_goal_unchanged(self):
        grid = Grid(20, 20)
        result = _resolve_passable_goal(grid, 5, 5)
        assert result == (5, 5)

    def test_non_passable_goal_finds_neighbour(self):
        grid = Grid(20, 20)
        grid.set_terrain(5, 5, Terrain.WATER)
        result = _resolve_passable_goal(grid, 5, 5)
        # Should be adjacent to (5, 5), distance 1
        assert abs(result[0] - 5) + abs(result[1] - 5) == 1
        tile = grid.get_tile(result[0], result[1])
        assert tile.is_passable

    def test_path_never_ends_on_non_passable(self):
        """Regression: old code allowed A* to reach non-passable goals."""
        grid = Grid(20, 20)
        # Single non-passable tile
        grid.set_terrain(3, 0, Terrain.WATER)
        path = find_path(grid, 0, 0, 3, 0)
        assert path is not None
        last_tile = grid.get_tile(path[-1][0], path[-1][1])
        assert last_tile.is_passable


class TestPathfindingPerformance:
    def test_long_path_with_higher_step_limit(self):
        grid = Grid(50, 50)
        # Long path needs more steps
        path = find_path(grid, -20, -20, 20, 20, max_steps=5000)
        assert path is not None
