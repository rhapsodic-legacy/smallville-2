"""Tests for the procedural town generator."""

from core.world.grid import Terrain
from core.world.generator import WorldConfig, TownGenerator, generate_world


class TestWorldConfig:
    def test_defaults(self):
        config = WorldConfig()
        assert config.population == 10
        assert config.terrain == "riverside"
        assert config.grid_width == 60


class TestTownGenerator:
    def test_generates_grid(self):
        config = WorldConfig(population=5, seed=42)
        grid, buildings = generate_world(config)
        assert grid.width == 60
        assert grid.height == 60

    def test_places_required_buildings(self):
        config = WorldConfig(population=5, seed=42)
        _, buildings = generate_world(config)
        types = {b.building_type for b in buildings}
        assert "tavern" in types
        assert "blacksmith" in types

    def test_places_homes_for_population(self):
        config = WorldConfig(population=8, seed=42)
        _, buildings = generate_world(config)
        homes = [b for b in buildings if b.building_type == "home"]
        assert len(homes) >= 1

    def test_riverside_has_water(self):
        config = WorldConfig(terrain="riverside", seed=42)
        grid, _ = generate_world(config)
        water_tiles = grid.find_tiles_by_terrain(Terrain.WATER)
        assert len(water_tiles) > 0

    def test_has_roads(self):
        config = WorldConfig(population=5, seed=42)
        grid, _ = generate_world(config)
        road_tiles = grid.find_tiles_by_terrain(Terrain.ROAD)
        assert len(road_tiles) > 0

    def test_has_resource_nodes(self):
        config = WorldConfig(population=5, seed=42)
        grid, _ = generate_world(config)
        resource_count = sum(
            1 for tile in grid
            for obj in tile.objects
            if obj.object_type == "resource"
        )
        assert resource_count > 0

    def test_deterministic_with_seed(self):
        config = WorldConfig(population=5, seed=123)
        grid1, b1 = generate_world(config)
        grid2, b2 = generate_world(config)
        assert len(b1) == len(b2)
        for a, b in zip(b1, b2):
            assert a.x == b.x
            assert a.z == b.z

    def test_town_centre_is_developed(self):
        config = WorldConfig(seed=42)
        grid, _ = generate_world(config)
        centre = grid.get_tile(0, 0)
        # Centre gets roads laid over it, so it's either dirt or road
        assert centre.terrain in (Terrain.DIRT, Terrain.ROAD)

    def test_serialisation_roundtrip(self):
        config = WorldConfig(population=3, seed=42, grid_width=20, grid_height=20)
        grid, _ = generate_world(config)
        data = grid.to_dict()
        assert data["width"] == 20
        assert len(data["tiles"]) == 400

    def test_forest_edge_terrain(self):
        config = WorldConfig(terrain="forest_edge", seed=42)
        grid, _ = generate_world(config)
        forest_tiles = grid.find_tiles_by_terrain(Terrain.FOREST)
        assert len(forest_tiles) > 0

    def test_church_with_large_population(self):
        config = WorldConfig(population=12, seed=42)
        _, buildings = generate_world(config)
        types = {b.building_type for b in buildings}
        assert "church" in types
