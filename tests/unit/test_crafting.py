"""
Tests for Phase 6.4: Crafting System.

Covers recipes, quality determination, crafting lifecycle,
skill gain, cancellation with refunds, extensibility API
(add/remove/replace recipes, custom quality functions, tags,
bulk loading from dicts).
"""

import pytest

from core.economy.crafting import (
    ItemQuality,
    Recipe,
    CraftingSession,
    CraftingManager,
    DEFAULT_RECIPES,
    QUALITY_YIELD_MULTIPLIER,
    QUALITY_SUFFIX,
    default_quality_function,
)
from core.npc.models import NPC, PersonalityTraits


# ---------- Helpers ----------


def _make_npc(
    npc_id: str = "npc_1",
    name: str = "Aldric",
    skills: dict | None = None,
    inventory: dict | None = None,
) -> NPC:
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory="A test NPC.",
        occupation="blacksmith",
        skills=skills if skills is not None else {"smithing": 0.7, "crafting": 0.5},
        inventory=inventory if inventory is not None else {},
    )


def _simple_recipe(**overrides) -> Recipe:
    defaults = dict(
        recipe_id="test_item",
        name="Test Item",
        result_item="test_output",
        result_quantity=1,
        ingredients={"wood": 2},
        required_skill="crafting",
        min_skill_level=0.0,
        craft_time=10.0,
        skill_gain=0.01,
        tags={"test"},
    )
    defaults.update(overrides)
    return Recipe(**defaults)


# ---------- ItemQuality ----------


class TestItemQuality:

    def test_default_quality_low_skill(self):
        assert default_quality_function(0.1) == ItemQuality.POOR

    def test_default_quality_mid_skill(self):
        assert default_quality_function(0.4) == ItemQuality.STANDARD

    def test_default_quality_high_skill(self):
        assert default_quality_function(0.7) == ItemQuality.FINE

    def test_default_quality_master_skill(self):
        assert default_quality_function(0.9) == ItemQuality.MASTERWORK

    def test_boundary_030(self):
        assert default_quality_function(0.3) == ItemQuality.STANDARD

    def test_boundary_060(self):
        assert default_quality_function(0.6) == ItemQuality.FINE

    def test_boundary_080(self):
        assert default_quality_function(0.8) == ItemQuality.MASTERWORK

    def test_yield_multiplier_masterwork(self):
        assert QUALITY_YIELD_MULTIPLIER[ItemQuality.MASTERWORK] == 2

    def test_suffix_standard_empty(self):
        assert QUALITY_SUFFIX[ItemQuality.STANDARD] == ""

    def test_suffix_masterwork(self):
        assert "masterwork" in QUALITY_SUFFIX[ItemQuality.MASTERWORK]


# ---------- Recipe ----------


class TestRecipe:

    def test_to_dict(self):
        r = _simple_recipe()
        d = r.to_dict()
        assert d["recipe_id"] == "test_item"
        assert d["ingredients"] == {"wood": 2}
        assert "test" in d["tags"]

    def test_from_dict(self):
        data = {
            "recipe_id": "magic_ring",
            "name": "Magic Ring",
            "result_item": "ring",
            "ingredients": {"gold_ore": 3},
            "required_skill": "enchanting",
            "min_skill_level": 0.5,
            "craft_time": 60.0,
            "tags": ["magic", "jewellery"],
            "metadata": {"lore": "An ancient design"},
        }
        r = Recipe.from_dict(data)
        assert r.recipe_id == "magic_ring"
        assert r.ingredients == {"gold_ore": 3}
        assert "magic" in r.tags
        assert r.metadata["lore"] == "An ancient design"

    def test_from_dict_defaults(self):
        data = {"recipe_id": "x", "name": "X", "result_item": "x"}
        r = Recipe.from_dict(data)
        assert r.result_quantity == 1
        assert r.min_skill_level == 0.0
        assert r.tags == set()

    def test_default_recipes_exist(self):
        ids = {r.recipe_id for r in DEFAULT_RECIPES}
        assert "plank" in ids
        assert "steel_sword" in ids
        assert "bread" in ids
        assert "furniture" in ids

    def test_round_trip(self):
        r = _simple_recipe(tags={"alpha", "beta"})
        d = r.to_dict()
        r2 = Recipe.from_dict(d)
        assert r2.recipe_id == r.recipe_id
        assert r2.ingredients == r.ingredients
        assert r2.tags == r.tags


# ---------- CraftingManager — Basics ----------


class TestCraftingManagerBasics:

    def test_default_recipes_loaded(self):
        mgr = CraftingManager()
        assert len(mgr.get_all_recipes()) == len(DEFAULT_RECIPES)

    def test_custom_recipes(self):
        r = _simple_recipe()
        mgr = CraftingManager(recipes=[r])
        assert len(mgr.get_all_recipes()) == 1

    def test_get_recipe(self):
        mgr = CraftingManager()
        assert mgr.get_recipe("plank") is not None
        assert mgr.get_recipe("nonexistent") is None


# ---------- CraftingManager — Extensibility ----------


class TestExtensibility:

    def test_add_recipe(self):
        mgr = CraftingManager(recipes=[])
        r = _simple_recipe(recipe_id="new_thing")
        mgr.add_recipe(r)
        assert mgr.get_recipe("new_thing") is not None

    def test_add_recipe_overwrites(self):
        mgr = CraftingManager(recipes=[])
        r1 = _simple_recipe(recipe_id="x", name="Old")
        r2 = _simple_recipe(recipe_id="x", name="New")
        mgr.add_recipe(r1)
        mgr.add_recipe(r2)
        assert mgr.get_recipe("x").name == "New"

    def test_remove_recipe(self):
        mgr = CraftingManager(recipes=[_simple_recipe()])
        assert mgr.remove_recipe("test_item")
        assert mgr.get_recipe("test_item") is None

    def test_remove_nonexistent(self):
        mgr = CraftingManager(recipes=[])
        assert not mgr.remove_recipe("nope")

    def test_replace_recipe(self):
        mgr = CraftingManager(recipes=[_simple_recipe()])
        new = _simple_recipe(name="Upgraded Item")
        assert mgr.replace_recipe("test_item", new)
        assert mgr.get_recipe("test_item").name == "Upgraded Item"

    def test_replace_nonexistent(self):
        mgr = CraftingManager(recipes=[])
        assert not mgr.replace_recipe("nope", _simple_recipe())

    def test_replace_with_different_id(self):
        """Replace old ID with recipe that has a new ID."""
        mgr = CraftingManager(recipes=[_simple_recipe(recipe_id="old")])
        new = _simple_recipe(recipe_id="new")
        assert mgr.replace_recipe("old", new)
        assert mgr.get_recipe("old") is None
        assert mgr.get_recipe("new") is not None

    def test_add_recipes_from_dicts(self):
        mgr = CraftingManager(recipes=[])
        dicts = [
            {"recipe_id": "a", "name": "A", "result_item": "a_out",
             "ingredients": {"wood": 1}, "tags": ["basic"]},
            {"recipe_id": "b", "name": "B", "result_item": "b_out",
             "ingredients": {"stone": 2}},
        ]
        count = mgr.add_recipes_from_dicts(dicts)
        assert count == 2
        assert mgr.get_recipe("a") is not None
        assert "basic" in mgr.get_recipe("a").tags

    def test_get_recipes_by_tag(self):
        mgr = CraftingManager()
        smithing = mgr.get_recipes_by_tag("smithing")
        assert all("smithing" in r.tags for r in smithing)
        assert len(smithing) >= 2  # iron_ingot, steel_sword, iron_tools

    def test_get_recipes_by_skill(self):
        mgr = CraftingManager()
        cooking = mgr.get_recipes_by_skill("cooking")
        assert all(r.required_skill == "cooking" for r in cooking)
        assert len(cooking) >= 2  # bread, berry_jam, flour

    def test_custom_quality_function_global(self):
        """Replace the global quality function."""
        always_masterwork = lambda skill: ItemQuality.MASTERWORK
        mgr = CraftingManager(quality_fn=always_masterwork)
        npc = _make_npc(skills={"crafting": 0.1})
        recipe = _simple_recipe()
        assert mgr.determine_quality(npc, recipe) == ItemQuality.MASTERWORK

    def test_set_quality_function(self):
        mgr = CraftingManager()
        always_poor = lambda skill: ItemQuality.POOR
        mgr.set_quality_function(always_poor)
        npc = _make_npc(skills={"crafting": 1.0})
        assert mgr.determine_quality(npc, _simple_recipe()) == ItemQuality.POOR

    def test_per_recipe_quality_function(self):
        """Per-recipe quality_fn takes precedence over global."""
        always_fine = lambda skill: ItemQuality.FINE
        recipe = _simple_recipe(quality_fn=always_fine)
        mgr = CraftingManager(recipes=[recipe])
        npc = _make_npc(skills={"crafting": 0.0})
        assert mgr.determine_quality(npc, recipe) == ItemQuality.FINE

    def test_recipe_metadata(self):
        r = _simple_recipe(metadata={"sell_bonus": 1.5, "lore": "Ancient technique"})
        assert r.metadata["sell_bonus"] == 1.5


# ---------- CraftingManager — Can Craft ----------


class TestCanCraft:

    def test_can_craft_success(self):
        mgr = CraftingManager(recipes=[_simple_recipe()])
        npc = _make_npc(skills={"crafting": 0.5}, inventory={"wood": 10})
        ok, msg = mgr.can_craft(npc, mgr.get_recipe("test_item"))
        assert ok

    def test_can_craft_no_skill(self):
        recipe = _simple_recipe(min_skill_level=0.5)
        mgr = CraftingManager(recipes=[recipe])
        npc = _make_npc(skills={"crafting": 0.1}, inventory={"wood": 10})
        ok, msg = mgr.can_craft(npc, recipe)
        assert not ok
        assert "crafting" in msg

    def test_can_craft_no_ingredients(self):
        mgr = CraftingManager(recipes=[_simple_recipe()])
        npc = _make_npc(inventory={})
        ok, msg = mgr.can_craft(npc, mgr.get_recipe("test_item"))
        assert not ok
        assert "wood" in msg

    def test_can_craft_insufficient_ingredients(self):
        mgr = CraftingManager(recipes=[_simple_recipe()])
        npc = _make_npc(inventory={"wood": 1})  # needs 2
        ok, msg = mgr.can_craft(npc, mgr.get_recipe("test_item"))
        assert not ok

    def test_get_available_recipes(self):
        r1 = _simple_recipe(recipe_id="easy", min_skill_level=0.0, ingredients={"wood": 1})
        r2 = _simple_recipe(recipe_id="hard", min_skill_level=0.9, ingredients={"wood": 1})
        mgr = CraftingManager(recipes=[r1, r2])
        npc = _make_npc(skills={"crafting": 0.5}, inventory={"wood": 5})
        available = mgr.get_available_recipes(npc)
        ids = [r.recipe_id for r in available]
        assert "easy" in ids
        assert "hard" not in ids


# ---------- CraftingManager — Lifecycle ----------


class TestCraftingLifecycle:

    def setup_method(self):
        self.recipe = _simple_recipe(craft_time=10.0, skill_gain=0.02)
        self.mgr = CraftingManager(recipes=[self.recipe])
        self.npc = _make_npc(skills={"crafting": 0.5}, inventory={"wood": 10})

    def test_start_crafting(self):
        ok, msg = self.mgr.start_crafting(self.npc, "test_item", 100.0)
        assert ok
        assert msg == "ok"
        assert self.mgr.is_crafting(self.npc.npc_id)

    def test_start_deducts_ingredients(self):
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        assert self.npc.inventory["wood"] == 8  # 10 - 2

    def test_start_unknown_recipe(self):
        ok, msg = self.mgr.start_crafting(self.npc, "unicorn", 100.0)
        assert not ok
        assert "unknown" in msg

    def test_start_already_crafting(self):
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        ok, msg = self.mgr.start_crafting(self.npc, "test_item", 101.0)
        assert not ok
        assert "already crafting" in msg

    def test_complete_crafting(self):
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        done, result = self.mgr.complete_crafting(self.npc, 110.0)
        assert done
        assert result["item"] == "test_output"
        assert result["quantity"] >= 1
        assert not self.mgr.is_crafting(self.npc.npc_id)

    def test_complete_too_early(self):
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        done, result = self.mgr.complete_crafting(self.npc, 105.0)
        assert not done
        assert "not finished" in result["reason"]

    def test_complete_no_session(self):
        done, result = self.mgr.complete_crafting(self.npc, 200.0)
        assert not done

    def test_complete_adds_to_inventory(self):
        self.npc.inventory["test_output"] = 3
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        self.mgr.complete_crafting(self.npc, 110.0)
        assert self.npc.inventory["test_output"] > 3

    def test_complete_applies_skill_gain(self):
        old_skill = self.npc.skills["crafting"]
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        self.mgr.complete_crafting(self.npc, 110.0)
        assert self.npc.skills["crafting"] == pytest.approx(old_skill + 0.02)

    def test_skill_gain_capped_at_one(self):
        self.npc.skills["crafting"] = 0.999
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        self.mgr.complete_crafting(self.npc, 110.0)
        assert self.npc.skills["crafting"] == 1.0

    def test_get_session(self):
        self.mgr.start_crafting(self.npc, "test_item", 100.0)
        session = self.mgr.get_session(self.npc.npc_id)
        assert session is not None
        assert session.recipe.recipe_id == "test_item"
        assert session.completes_at == 110.0


# ---------- Quality and Yield ----------


class TestQualityOutcomes:

    def test_poor_quality_standard_yield(self):
        recipe = _simple_recipe(result_quantity=2)
        mgr = CraftingManager(
            recipes=[recipe],
            quality_fn=lambda s: ItemQuality.POOR,
        )
        npc = _make_npc(skills={"crafting": 0.1}, inventory={"wood": 10})
        mgr.start_crafting(npc, "test_item", 0.0)
        _, result = mgr.complete_crafting(npc, 100.0)
        assert result["quantity"] == 2  # multiplier = 1
        assert "(poor)" in result["item"]

    def test_masterwork_double_yield(self):
        recipe = _simple_recipe(result_quantity=2)
        mgr = CraftingManager(
            recipes=[recipe],
            quality_fn=lambda s: ItemQuality.MASTERWORK,
        )
        npc = _make_npc(skills={"crafting": 1.0}, inventory={"wood": 10})
        mgr.start_crafting(npc, "test_item", 0.0)
        _, result = mgr.complete_crafting(npc, 100.0)
        assert result["quantity"] == 4  # 2 * 2
        assert "(masterwork)" in result["item"]

    def test_standard_no_suffix(self):
        recipe = _simple_recipe()
        mgr = CraftingManager(
            recipes=[recipe],
            quality_fn=lambda s: ItemQuality.STANDARD,
        )
        npc = _make_npc(inventory={"wood": 10})
        mgr.start_crafting(npc, "test_item", 0.0)
        _, result = mgr.complete_crafting(npc, 100.0)
        assert result["item"] == "test_output"  # no suffix

    def test_fine_suffix(self):
        recipe = _simple_recipe()
        mgr = CraftingManager(
            recipes=[recipe],
            quality_fn=lambda s: ItemQuality.FINE,
        )
        npc = _make_npc(inventory={"wood": 10})
        mgr.start_crafting(npc, "test_item", 0.0)
        _, result = mgr.complete_crafting(npc, 100.0)
        assert "(fine)" in result["item"]


# ---------- Cancellation ----------


class TestCancellation:

    def test_cancel_refunds_ingredients(self):
        mgr = CraftingManager(recipes=[_simple_recipe()])
        npc = _make_npc(inventory={"wood": 10})
        mgr.start_crafting(npc, "test_item", 100.0)
        assert npc.inventory["wood"] == 8
        ok, msg = mgr.cancel_crafting(npc)
        assert ok
        assert npc.inventory["wood"] == 10  # refunded
        assert not mgr.is_crafting(npc.npc_id)

    def test_cancel_no_session(self):
        mgr = CraftingManager()
        npc = _make_npc()
        ok, msg = mgr.cancel_crafting(npc)
        assert not ok


# ---------- State ----------


class TestCraftingState:

    def test_get_state(self):
        mgr = CraftingManager()
        state = mgr.get_state()
        assert state["recipes"] == len(DEFAULT_RECIPES)
        assert state["active_sessions"] == 0

    def test_get_stats(self):
        mgr = CraftingManager()
        stats = mgr.get_stats()
        assert stats["recipe_count"] == len(DEFAULT_RECIPES)
        assert "smithing" in stats["skills_used"]
        assert "basic" in stats["tags_in_use"]
