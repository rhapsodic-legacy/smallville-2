"""Tests for the spatial grid system."""

import pytest
from core.world.grid import Grid, Terrain, Tile, WorldObject


class TestGridCreation:
    def test_creates_correct_number_of_tiles(self):
        grid = Grid(10, 10)
        count = sum(1 for _ in grid)
        assert count == 100

    def test_origin_is_near_centre(self):
        grid = Grid(20, 20)
        tile = grid.get_tile(0, 0)
        assert tile is not None
        assert tile.x == 0
        assert tile.z == 0

    def test_bounds_are_symmetric(self):
        grid = Grid(20, 20)
        min_x, min_z, max_x, max_z = grid.bounds
        assert min_x == -10
        assert min_z == -10
        assert max_x == 9
        assert max_z == 9

    def test_default_terrain_is_grass(self):
        grid = Grid(5, 5)
        tile = grid.get_tile(0, 0)
        assert tile.terrain == Terrain.GRASS

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            Grid(0, 10)
        with pytest.raises(ValueError):
            Grid(10, -1)

    def test_out_of_bounds_returns_none(self):
        grid = Grid(10, 10)
        assert grid.get_tile(100, 100) is None


class TestTile:
    def test_water_is_not_passable(self):
        tile = Tile(x=0, z=0, terrain=Terrain.WATER)
        assert not tile.is_passable

    def test_grass_is_passable(self):
        tile = Tile(x=0, z=0, terrain=Terrain.GRASS)
        assert tile.is_passable

    def test_non_walkable_object_blocks_tile(self):
        tile = Tile(x=0, z=0, terrain=Terrain.GRASS)
        tile.objects.append(WorldObject("wall", "building", "Wall", walkable=False))
        assert not tile.is_passable

    def test_address_with_sector_and_arena(self):
        tile = Tile(x=0, z=0, sector="market", arena="blacksmith_1")
        assert tile.address == "smallville:market:blacksmith_1"

    def test_address_without_arena(self):
        tile = Tile(x=0, z=0, sector="residential")
        assert tile.address == "smallville:residential"


class TestGridQueries:
    def test_neighbours_cardinal(self):
        grid = Grid(10, 10)
        neighbours = grid.get_neighbours(0, 0, diagonal=False)
        assert len(neighbours) == 4

    def test_neighbours_diagonal(self):
        grid = Grid(10, 10)
        neighbours = grid.get_neighbours(0, 0, diagonal=True)
        assert len(neighbours) == 8

    def test_corner_has_fewer_neighbours(self):
        grid = Grid(10, 10)
        min_x, min_z, _, _ = grid.bounds
        neighbours = grid.get_neighbours(min_x, min_z, diagonal=False)
        assert len(neighbours) == 2

    def test_tiles_in_radius(self):
        grid = Grid(20, 20)
        tiles = grid.tiles_in_radius(0, 0, 2)
        # Manhattan distance ≤ 2: 1 + 4 + 8 = 13
        assert len(tiles) == 13

    def test_tiles_in_rect(self):
        grid = Grid(20, 20)
        tiles = grid.tiles_in_rect(-1, -1, 1, 1)
        assert len(tiles) == 9

    def test_set_terrain(self):
        grid = Grid(10, 10)
        grid.set_terrain(0, 0, Terrain.WATER)
        tile = grid.get_tile(0, 0)
        assert tile.terrain == Terrain.WATER
        assert not tile.walkable

    def test_find_tiles_by_sector(self):
        grid = Grid(10, 10)
        grid.set_sector(0, 0, "market")
        grid.set_sector(1, 0, "market")
        found = grid.find_tiles_by_sector("market")
        assert len(found) == 2


class TestGridSerialisation:
    def test_to_dict_has_required_keys(self):
        grid = Grid(5, 5)
        data = grid.to_dict()
        assert "width" in data
        assert "height" in data
        assert "tiles" in data
        assert len(data["tiles"]) == 25

    def test_tile_dict_has_terrain(self):
        grid = Grid(5, 5)
        data = grid.to_dict()
        tile = data["tiles"][0]
        assert "terrain" in tile
        assert tile["terrain"] == "grass"
