"""
Tests for Phase 6.1: Resource System.

Covers resource types, node templates, resource nodes, gathering mechanics,
skill scaling, regeneration, depletion, and ResourceManager operations.
"""

import pytest

from core.economy.resources import (
    ResourceType,
    ResourceNode,
    ResourceManager,
    GatheringSession,
    NodeTemplate,
    NODE_TEMPLATES,
    RESOURCE_NAME_MAP,
)
from core.npc.models import NPC, PersonalityTraits, ActivityState
from core.world.grid import Grid, WorldObject
from core.world.generator import TownGenerator, WorldConfig


# ---------- Helpers ----------


def _make_npc(
    npc_id: str = "npc_1",
    name: str = "Aldric",
    x: int = 0,
    z: int = 0,
    skills: dict | None = None,
) -> NPC:
    """Create a minimal NPC for resource tests."""
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory="A test NPC.",
        occupation="labourer",
        x=x,
        z=z,
        skills=skills if skills is not None else {"gathering": 0.5, "farming": 0.3, "mining": 0.4},
    )


def _make_node(
    node_id: str = "oak_tree_1",
    resource_name: str = "oak_tree",
    x: int = 0,
    z: int = 0,
    current_amount: int | None = None,
) -> ResourceNode:
    """Create a ResourceNode from a template."""
    tmpl = NODE_TEMPLATES[resource_name]
    return ResourceNode(
        node_id=node_id,
        resource_name=resource_name,
        resource_type=tmpl.resource_type,
        x=x,
        z=z,
        capacity=tmpl.capacity,
        current_amount=current_amount if current_amount is not None else tmpl.capacity,
        gather_time=tmpl.gather_time,
        base_yield=tmpl.base_yield,
        required_skill=tmpl.required_skill,
        min_skill_level=tmpl.min_skill_level,
        regen_per_day=tmpl.regen_per_day,
    )


def _make_grid_with_resources() -> Grid:
    """Create a small grid with a few resource objects placed on it."""
    grid = Grid(20, 20)
    # Place an oak tree at (5, 5)
    grid.place_object(5, 5, WorldObject(
        object_id="oak_tree_1",
        object_type="resource",
        name="Oak Tree",
        walkable=True,
        metadata={"resource": "oak_tree", "yield": 4},
    ))
    # Place an iron deposit at (-3, -3)
    grid.place_object(-3, -3, WorldObject(
        object_id="iron_deposit_1",
        object_type="resource",
        name="Iron Deposit",
        walkable=True,
        metadata={"resource": "iron_deposit", "yield": 2},
    ))
    # Place a berry bush at (2, 2)
    grid.place_object(2, 2, WorldObject(
        object_id="berry_bush_1",
        object_type="resource",
        name="Berry Bush",
        walkable=True,
        metadata={"resource": "berry_bush", "yield": 3},
    ))
    # Place a non-resource object (building) — should be ignored
    grid.place_object(0, 0, WorldObject(
        object_id="tavern_1",
        object_type="building",
        name="Tavern",
        walkable=False,
    ))
    return grid


# ---------- ResourceType ----------


class TestResourceType:
    """Enum coverage."""

    def test_all_types_have_values(self):
        expected = {"wood", "stone", "iron", "gold_ore", "food", "wheat", "berries"}
        actual = {rt.value for rt in ResourceType}
        assert actual == expected

    def test_resource_name_map_covers_generator_resources(self):
        for name, (rtype, display) in RESOURCE_NAME_MAP.items():
            assert isinstance(rtype, ResourceType)
            assert isinstance(display, str)


# ---------- NodeTemplate ----------


class TestNodeTemplates:
    """Verify built-in templates are sane."""

    def test_all_mapped_resources_have_templates(self):
        for name in RESOURCE_NAME_MAP:
            assert name in NODE_TEMPLATES, f"Missing template for {name}"

    def test_template_values_positive(self):
        for name, tmpl in NODE_TEMPLATES.items():
            assert tmpl.capacity > 0, f"{name} capacity must be positive"
            assert tmpl.gather_time > 0, f"{name} gather_time must be positive"
            assert tmpl.base_yield > 0, f"{name} base_yield must be positive"
            assert tmpl.regen_per_day >= 0, f"{name} regen must be non-negative"
            assert 0.0 <= tmpl.min_skill_level <= 1.0


# ---------- ResourceNode ----------


class TestResourceNode:
    """Unit tests for the ResourceNode dataclass."""

    def test_fullness(self):
        node = _make_node(current_amount=10)
        assert node.fullness == 10 / 20  # oak_tree capacity = 20

    def test_fullness_at_capacity(self):
        node = _make_node()
        assert node.fullness == 1.0

    def test_fullness_empty(self):
        node = _make_node(current_amount=0)
        assert node.fullness == 0.0
        assert node.is_depleted

    def test_can_gather_success(self):
        node = _make_node()
        npc = _make_npc(skills={"gathering": 0.5})
        ok, reason = node.can_gather(npc)
        assert ok
        assert reason == "ok"

    def test_can_gather_depleted(self):
        node = _make_node(current_amount=0)
        npc = _make_npc()
        ok, reason = node.can_gather(npc)
        assert not ok
        assert reason == "depleted"

    def test_can_gather_skill_too_low(self):
        node = _make_node(resource_name="iron_deposit")
        npc = _make_npc(skills={"mining": 0.1})  # needs 0.2
        ok, reason = node.can_gather(npc)
        assert not ok
        assert "mining" in reason

    def test_can_gather_no_skill_at_all(self):
        node = _make_node(resource_name="iron_deposit")
        npc = _make_npc(skills={})  # no mining skill
        ok, reason = node.can_gather(npc)
        assert not ok

    def test_can_gather_zero_min_skill(self):
        """Oak trees require gathering >= 0.0, so even 0.0 skill works."""
        node = _make_node()
        npc = _make_npc(skills={})  # 0.0 gathering (missing key)
        ok, _ = node.can_gather(npc)
        assert ok

    def test_calculate_yield_scales_with_skill(self):
        node = _make_node()  # base_yield = 3
        low = _make_npc(skills={"gathering": 0.0})
        high = _make_npc(skills={"gathering": 1.0})
        yield_low = node.calculate_yield(low)
        yield_high = node.calculate_yield(high)
        assert yield_high > yield_low

    def test_calculate_yield_minimum_one(self):
        node = _make_node()
        npc = _make_npc(skills={"gathering": 0.0})
        assert node.calculate_yield(npc) >= 1

    def test_calculate_yield_capped_at_current(self):
        node = _make_node(current_amount=1)
        npc = _make_npc(skills={"gathering": 1.0})
        assert node.calculate_yield(npc) == 1

    def test_extract(self):
        node = _make_node(current_amount=10)
        taken = node.extract(3)
        assert taken == 3
        assert node.current_amount == 7

    def test_extract_more_than_available(self):
        node = _make_node(current_amount=2)
        taken = node.extract(5)
        assert taken == 2
        assert node.current_amount == 0
        assert node.is_depleted

    def test_regenerate(self):
        node = _make_node(current_amount=0)
        # Simulate a full game day (1440 minutes)
        node.regenerate(1440.0)
        assert node.current_amount > 0
        assert node.current_amount <= node.regen_per_day

    def test_regenerate_does_not_exceed_capacity(self):
        node = _make_node(current_amount=19)  # capacity = 20
        # Regenerate a huge amount of time
        node.regenerate(14400.0)  # 10 days
        assert node.current_amount == node.capacity

    def test_regenerate_no_op_at_full(self):
        node = _make_node()  # already at capacity
        node.regenerate(1440.0)
        assert node.current_amount == node.capacity

    def test_to_dict(self):
        node = _make_node()
        d = node.to_dict()
        assert d["node_id"] == "oak_tree_1"
        assert d["resource_type"] == "wood"
        assert d["capacity"] == 20
        assert d["is_depleted"] is False


# ---------- ResourceManager — Initialisation ----------


class TestResourceManagerInit:
    """Tests for grid scanning and node creation."""

    def test_initialise_from_grid(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        count = mgr.initialise_from_grid(grid)
        assert count == 3  # oak, iron, berry — tavern is a building

    def test_ignores_non_resource_objects(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        mgr.initialise_from_grid(grid)
        assert mgr.get_node_at(0, 0) is None  # tavern tile

    def test_node_positions_correct(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        mgr.initialise_from_grid(grid)
        oak = mgr.get_node_at(5, 5)
        assert oak is not None
        assert oak.resource_type == ResourceType.WOOD

    def test_uses_generator_yield(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        mgr.initialise_from_grid(grid)
        oak = mgr.get_node_at(5, 5)
        assert oak.base_yield == 4  # from metadata, not template default

    def test_initialise_from_generated_world(self):
        """Integration: scan a fully generated world."""
        config = WorldConfig(population=5, economy="mixed", seed=42)
        gen = TownGenerator(config)
        grid = gen.generate()
        mgr = ResourceManager()
        count = mgr.initialise_from_grid(grid)
        assert count > 0


# ---------- ResourceManager — Queries ----------


class TestResourceManagerQueries:

    def setup_method(self):
        self.grid = _make_grid_with_resources()
        self.mgr = ResourceManager()
        self.mgr.initialise_from_grid(self.grid)

    def test_get_node_by_id(self):
        node = self.mgr.get_node("oak_tree_1")
        assert node is not None
        assert node.resource_name == "oak_tree"

    def test_get_node_missing(self):
        assert self.mgr.get_node("nonexistent") is None

    def test_get_all_nodes(self):
        assert len(self.mgr.get_all_nodes()) == 3

    def test_get_nodes_by_type(self):
        iron_nodes = self.mgr.get_nodes_by_type(ResourceType.IRON)
        assert len(iron_nodes) == 1
        assert iron_nodes[0].resource_name == "iron_deposit"

    def test_get_nearest_node(self):
        # From origin, berry bush at (2,2) is closest
        nearest = self.mgr.get_nearest_node(0, 0)
        assert nearest is not None
        assert nearest.node_id == "berry_bush_1"

    def test_get_nearest_node_by_type(self):
        nearest = self.mgr.get_nearest_node(0, 0, resource_type=ResourceType.WOOD)
        assert nearest is not None
        assert nearest.node_id == "oak_tree_1"

    def test_get_nearest_node_skips_depleted(self):
        berry = self.mgr.get_node("berry_bush_1")
        berry.current_amount = 0
        nearest = self.mgr.get_nearest_node(0, 0, only_available=True)
        assert nearest.node_id != "berry_bush_1"

    def test_get_nearest_node_includes_depleted_when_asked(self):
        berry = self.mgr.get_node("berry_bush_1")
        berry.current_amount = 0
        nearest = self.mgr.get_nearest_node(0, 0, only_available=False)
        assert nearest.node_id == "berry_bush_1"


# ---------- ResourceManager — Gathering ----------


class TestGathering:

    def setup_method(self):
        self.grid = _make_grid_with_resources()
        self.mgr = ResourceManager()
        self.mgr.initialise_from_grid(self.grid)
        self.npc = _make_npc(x=5, z=5, skills={"gathering": 0.5})
        self.node = self.mgr.get_node("oak_tree_1")

    def test_start_gathering_success(self):
        ok, msg = self.mgr.start_gathering(self.npc, self.node, 100.0)
        assert ok
        assert msg == "ok"
        assert self.mgr.is_gathering(self.npc.npc_id)

    def test_start_gathering_too_far(self):
        self.npc.x = 0
        self.npc.z = 0
        ok, msg = self.mgr.start_gathering(self.npc, self.node, 100.0)
        assert not ok
        assert msg == "too far from node"

    def test_start_gathering_adjacent_tile(self):
        """NPC one tile away (adjacent) should be allowed."""
        self.npc.x = 5
        self.npc.z = 4  # one tile north
        ok, msg = self.mgr.start_gathering(self.npc, self.node, 100.0)
        assert ok

    def test_start_gathering_already_gathering(self):
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        ok, msg = self.mgr.start_gathering(self.npc, self.node, 101.0)
        assert not ok
        assert msg == "already gathering"

    def test_start_gathering_depleted(self):
        self.node.current_amount = 0
        ok, msg = self.mgr.start_gathering(self.npc, self.node, 100.0)
        assert not ok
        assert msg == "depleted"

    def test_complete_gathering_success(self):
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        # Advance past gather_time
        done, result = self.mgr.complete_gathering(
            self.npc, 100.0 + self.node.gather_time,
        )
        assert done
        assert result["resource_type"] == "wood"
        assert result["amount"] > 0
        assert self.npc.inventory.get("wood", 0) > 0
        assert not self.mgr.is_gathering(self.npc.npc_id)

    def test_complete_gathering_too_early(self):
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        done, result = self.mgr.complete_gathering(self.npc, 105.0)
        assert not done
        assert result["reason"] == "not finished"

    def test_complete_gathering_no_session(self):
        done, result = self.mgr.complete_gathering(self.npc, 200.0)
        assert not done
        assert result["reason"] == "no active session"

    def test_complete_gathering_node_depleted_during(self):
        """Another NPC depletes the node while we're gathering."""
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        self.node.current_amount = 0  # someone else took everything
        done, result = self.mgr.complete_gathering(
            self.npc, 100.0 + self.node.gather_time,
        )
        assert not done
        assert result["reason"] == "node depleted during gathering"

    def test_cancel_gathering(self):
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        assert self.mgr.cancel_gathering(self.npc.npc_id)
        assert not self.mgr.is_gathering(self.npc.npc_id)

    def test_cancel_gathering_not_active(self):
        assert not self.mgr.cancel_gathering("nobody")

    def test_get_session(self):
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        session = self.mgr.get_session(self.npc.npc_id)
        assert session is not None
        assert session.node_id == "oak_tree_1"
        assert session.completes_at == 100.0 + self.node.gather_time

    def test_inventory_accumulates(self):
        """Multiple gathers add to inventory."""
        self.npc.inventory["wood"] = 5
        self.mgr.start_gathering(self.npc, self.node, 100.0)
        self.mgr.complete_gathering(self.npc, 200.0)
        assert self.npc.inventory["wood"] > 5


# ---------- ResourceManager — Tick & Regen ----------


class TestResourceManagerTick:

    def test_tick_regenerates_nodes(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        mgr.initialise_from_grid(grid)
        oak = mgr.get_node("oak_tree_1")
        oak.current_amount = 0
        # Tick for a full game day
        mgr.tick(1440.0)
        assert oak.current_amount > 0


# ---------- ResourceManager — Serialisation ----------


class TestResourceManagerState:

    def test_get_state(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        mgr.initialise_from_grid(grid)
        state = mgr.get_state()
        assert len(state["nodes"]) == 3
        assert state["active_sessions"] == 0

    def test_get_stats(self):
        grid = _make_grid_with_resources()
        mgr = ResourceManager()
        mgr.initialise_from_grid(grid)
        stats = mgr.get_stats()
        assert stats["total_nodes"] == 3
        assert stats["depleted_nodes"] == 0
        assert "wood" in stats["by_type"]
