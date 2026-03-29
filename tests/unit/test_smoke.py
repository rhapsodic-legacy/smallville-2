"""Smoke tests — verify basic imports and project structure."""

import pytest


def test_core_imports():
    """Verify the core package is importable."""
    import core
    assert core.__version__ == "0.1.0"


def test_submodule_imports():
    """Verify all core submodules are importable."""
    import core.world
    import core.npc
    import core.npc.cognition
    import core.memory
    import core.relationships
    import core.events
    import core.economy
    import core.evolution
    import core.time_system
    import core.player


def test_sample_npc_data_fixture(sample_npc_data):
    """Verify the NPC fixture has required fields."""
    required_fields = [
        "npc_id", "name", "age", "personality_traits", "occupation",
        "location", "health", "energy", "gold", "inventory", "skills",
    ]
    for field in required_fields:
        assert field in sample_npc_data, f"Missing field: {field}"


def test_sample_world_config_fixture(sample_world_config):
    """Verify the world config fixture has required fields."""
    assert sample_world_config["population"] > 0
    assert sample_world_config["terrain"] in ["riverside", "plains", "forest_clearing", "hilltop"]
