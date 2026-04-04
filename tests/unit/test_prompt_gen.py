"""
Tests for the town generation prompt system.

Covers heuristic parsing, feature placement, LLM parser validation,
and end-to-end orchestration.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from core.world.generator import TownGenerator, WorldConfig, generate_world
from core.world.grid import Terrain
from core.world.prompt_gen import GeneratedWorldSpec, TownPromptGenerator
from core.world.prompt_gen.features import (
    MOOD_PRESETS,
    TerrainFeature,
    TownMood,
    place_features,
)
from core.world.prompt_gen.heuristic import parse_description
from core.world.prompt_gen.parser import (
    ParsedTownDescription,
    _extract_json,
    _validate_and_build,
)


# ---------- Heuristic parser tests ----------

class TestHeuristicParser:
    """Tests for keyword-based fallback parsing."""

    def test_riverside_terrain(self):
        result = parse_description("a town by the river")
        assert result.terrain == "riverside"

    def test_forest_terrain(self):
        result = parse_description("settlement near the dark forest")
        assert result.terrain == "forest_edge"

    def test_hillside_terrain(self):
        result = parse_description("a mountain village")
        assert result.terrain == "hillside"

    def test_plains_default(self):
        result = parse_description("a simple village")
        assert result.terrain == "plains"

    def test_farming_economy(self):
        result = parse_description("an agricultural farming community")
        assert result.economy == "farming"

    def test_trading_economy(self):
        result = parse_description("a bustling merchant trading post")
        assert result.economy == "trading"

    def test_mining_economy(self):
        result = parse_description("an iron mining settlement")
        assert result.economy == "mining"

    def test_ruler_detection(self):
        result = parse_description("a town ruled by a stern lord")
        assert result.has_ruler is True

    def test_no_ruler(self):
        result = parse_description("a peaceful hamlet")
        assert result.has_ruler is False

    def test_small_population(self):
        result = parse_description("a tiny hamlet")
        assert result.population_hint == 4

    def test_large_population(self):
        result = parse_description("a large sprawling town")
        assert result.population_hint is not None
        assert result.population_hint >= 14

    def test_cozy_mood(self):
        result = parse_description("a cozy little village")
        assert result.mood is not None
        assert result.mood.descriptor == "cozy"

    def test_fortified_mood(self):
        result = parse_description("a fortified stronghold")
        assert result.mood is not None
        assert result.mood.descriptor == "fortified"

    def test_bridge_feature(self):
        result = parse_description("riverside town with two bridges")
        assert result.terrain == "riverside"
        assert any(f.type == "bridge" for f in result.features)
        bridge = next(f for f in result.features if f.type == "bridge")
        assert bridge.count == 2

    def test_bridge_only_with_river(self):
        """Bridges should not appear without riverside terrain."""
        result = parse_description("a plains town with a bridge")
        assert not any(f.type == "bridge" for f in result.features)

    def test_multiple_features(self):
        result = parse_description(
            "riverside town with ruins and a garden and watchtower"
        )
        types = {f.type for f in result.features}
        assert "ruins" in types
        assert "garden" in types
        assert "watchtower" in types

    def test_wall_feature(self):
        result = parse_description("town with defensive walls")
        assert any(f.type == "wall" for f in result.features)

    def test_count_word_three(self):
        result = parse_description("riverside town with three bridges")
        bridge = next(f for f in result.features if f.type == "bridge")
        assert bridge.count == 3

    def test_full_description(self):
        """Test a complex, realistic description."""
        result = parse_description(
            "a cozy riverside farming village with two bridges and a garden"
        )
        assert result.terrain == "riverside"
        assert result.economy == "farming"
        assert result.mood is not None
        assert result.mood.descriptor == "cozy"
        types = {f.type for f in result.features}
        assert "bridge" in types
        assert "garden" in types


# ---------- LLM parser validation tests ----------

class TestParserValidation:
    """Tests for LLM output validation and JSON extraction."""

    def test_extract_json_plain(self):
        raw = '{"terrain": "riverside"}'
        assert _extract_json(raw) == '{"terrain": "riverside"}'

    def test_extract_json_code_block(self):
        raw = '```json\n{"terrain": "riverside"}\n```'
        assert json.loads(_extract_json(raw))["terrain"] == "riverside"

    def test_extract_json_with_text(self):
        raw = 'Here is the output:\n{"terrain": "plains"}\nEnd.'
        assert json.loads(_extract_json(raw))["terrain"] == "plains"

    def test_extract_json_no_json(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")

    def test_validate_good_data(self):
        data = {
            "terrain": "riverside",
            "economy": "farming",
            "population_hint": 8,
            "has_ruler": False,
            "mood": "cozy",
            "features": [
                {"type": "bridge", "count": 2, "placement": "auto"},
            ],
            "name_suggestion": "Willowbrook",
        }
        result = _validate_and_build(data)
        assert result.terrain == "riverside"
        assert result.economy == "farming"
        assert result.population_hint == 8
        assert result.mood.descriptor == "cozy"
        assert len(result.features) == 1
        assert result.features[0].count == 2
        assert result.name_suggestion == "Willowbrook"

    def test_validate_invalid_terrain_defaults(self):
        result = _validate_and_build({"terrain": "volcanic"})
        assert result.terrain == "plains"

    def test_validate_invalid_economy_defaults(self):
        result = _validate_and_build({"economy": "piracy"})
        assert result.economy == "mixed"

    def test_validate_population_clamped(self):
        result = _validate_and_build({"population_hint": 100})
        assert result.population_hint == 20

    def test_validate_population_min(self):
        result = _validate_and_build({"population_hint": 1})
        assert result.population_hint == 4

    def test_validate_invalid_feature_skipped(self):
        result = _validate_and_build({
            "features": [{"type": "volcano", "count": 1}],
        })
        assert len(result.features) == 0

    def test_validate_bridge_without_river_skipped(self):
        result = _validate_and_build({
            "terrain": "plains",
            "features": [{"type": "bridge", "count": 1}],
        })
        assert len(result.features) == 0

    def test_validate_feature_count_clamped(self):
        result = _validate_and_build({
            "terrain": "riverside",
            "features": [{"type": "bridge", "count": 10}],
        })
        assert result.features[0].count == 4


# ---------- Feature placement tests ----------

class TestFeaturePlacement:
    """Tests for terrain feature placement on the grid."""

    def _make_riverside_grid(self, seed=42):
        config = WorldConfig(terrain="riverside", seed=seed, population=6)
        gen = TownGenerator(config)
        gen.generate()
        return gen.grid

    def _make_plains_grid(self, seed=42):
        config = WorldConfig(terrain="plains", seed=seed, population=6)
        gen = TownGenerator(config)
        gen.generate()
        return gen.grid

    def test_bridge_placement(self):
        grid = self._make_riverside_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="bridge", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1
        assert records[0]["type"] == "bridge"

        # Verify the bridge tile is walkable
        bx, bz = records[0]["x"], records[0]["z"]
        tile = grid.get_tile(bx, bz)
        assert tile is not None
        assert tile.is_passable

    def test_two_bridges(self):
        grid = self._make_riverside_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="bridge", count=2)]
        records = place_features(grid, features, rng)
        # Should place at least 1 (2 if river has enough candidate rows)
        assert len(records) >= 1
        assert all(r["type"] == "bridge" for r in records)

    def test_pond_placement(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="pond", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1
        assert records[0]["type"] == "pond"

        # Verify water tiles exist around the pond location
        px, pz = records[0]["x"], records[0]["z"]
        tile = grid.get_tile(px, pz)
        assert tile is not None
        assert tile.terrain == Terrain.WATER

    def test_wall_placement(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="wall", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1
        assert records[0]["type"] == "wall"
        assert records[0]["tiles"] > 0

    def test_garden_placement(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="garden", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1
        assert records[0]["type"] == "garden"

    def test_ruins_placement(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="ruins", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1

    def test_watchtower_placement(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="watchtower", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1

    def test_well_placement(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="well", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 1

    def test_bridge_no_water(self):
        """Bridge placement should fail gracefully on plains (no river)."""
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="bridge", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 0

    def test_unknown_feature_ignored(self):
        grid = self._make_plains_grid()
        import random
        rng = random.Random(42)
        features = [TerrainFeature(type="volcano", count=1)]
        records = place_features(grid, features, rng)
        assert len(records) == 0

    def test_generate_world_with_features(self):
        """Test that generate_world() accepts features."""
        features = [TerrainFeature(type="well", count=1)]
        grid, buildings = generate_world(
            WorldConfig(seed=42, population=6),
            features=features,
        )
        # Well should be on the grid somewhere
        found = False
        for tile in grid:
            for obj in tile.objects:
                if obj.metadata.get("feature") == "well":
                    found = True
                    break
        assert found


# ---------- Orchestrator tests ----------

class TestTownPromptGenerator:
    """Tests for the TownPromptGenerator orchestrator."""

    def test_heuristic_mode(self):
        """Without LLM, should use heuristic fallback."""
        gen = TownPromptGenerator(seed=42)
        spec = asyncio.get_event_loop().run_until_complete(
            gen.generate_config("a cozy riverside town with two bridges")
        )
        assert isinstance(spec, GeneratedWorldSpec)
        assert spec.config.terrain == "riverside"
        assert spec.config.economy == "mixed"
        # Cozy mood reduces population
        assert spec.config.population < 10
        assert any(f.type == "bridge" for f in spec.features)

    def test_fortified_implies_ruler(self):
        gen = TownPromptGenerator(seed=42)
        spec = asyncio.get_event_loop().run_until_complete(
            gen.generate_config("a fortified mining town with walls")
        )
        assert spec.config.has_ruler is True
        assert any(f.type == "wall" for f in spec.features)

    def test_spec_to_dict(self):
        gen = TownPromptGenerator(seed=42)
        spec = asyncio.get_event_loop().run_until_complete(
            gen.generate_config("a small plains village")
        )
        d = spec.to_dict()
        assert "config" in d
        assert "features" in d
        assert "town_name" in d
        assert "parsed" in d

    def test_custom_grid_size(self):
        gen = TownPromptGenerator(seed=42, grid_width=80, grid_height=80)
        spec = asyncio.get_event_loop().run_until_complete(
            gen.generate_config("a large trading hub on the plains")
        )
        assert spec.config.grid_width == 80
        assert spec.config.grid_height == 80

    def test_mood_economy_hint(self):
        """Rustic mood should suggest farming economy."""
        gen = TownPromptGenerator(seed=42)
        spec = asyncio.get_event_loop().run_until_complete(
            gen.generate_config("a rustic hamlet in the hills")
        )
        assert spec.config.economy == "farming"


# ---------- Data model tests ----------

class TestDataModels:
    """Tests for serialisation of features and moods."""

    def test_terrain_feature_roundtrip(self):
        feat = TerrainFeature(type="bridge", count=2, placement="auto")
        d = feat.to_dict()
        feat2 = TerrainFeature.from_dict(d)
        assert feat2.type == "bridge"
        assert feat2.count == 2

    def test_mood_roundtrip(self):
        mood = TownMood(
            descriptor="cozy",
            population_modifier=0.7,
            building_bias={"tavern": 1},
        )
        d = mood.to_dict()
        mood2 = TownMood.from_dict(d)
        assert mood2.descriptor == "cozy"
        assert mood2.population_modifier == 0.7

    def test_parsed_description_to_dict(self):
        parsed = ParsedTownDescription(
            terrain="riverside",
            economy="mixed",
            features=[TerrainFeature(type="bridge", count=1)],
            name_suggestion="Willowbrook",
        )
        d = parsed.to_dict()
        assert d["terrain"] == "riverside"
        assert len(d["features"]) == 1
