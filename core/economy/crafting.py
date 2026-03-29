"""
Crafting system — recipes, skill-gated production, quality outcomes.

NPCs transform raw resources into finished goods.  Recipes define
ingredients, skill requirements, and craft time.  Quality is derived
from skill level and can be modified by pluggable quality functions.

Designed for extensibility:
  - Recipes and quality functions are data, not code.
  - AI Game Studio can inject/remove/replace recipes and quality
    modifiers at runtime via the CraftingManager API.
  - Recipe tags allow filtering by theme, tier, or category.
  - Custom quality functions let game designers reshape the
    skill-to-quality curve per world or per recipe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC

logger = logging.getLogger(__name__)


# ---------- Item Quality ----------

class ItemQuality(Enum):
    """Quality tiers for crafted items."""
    POOR = "poor"
    STANDARD = "standard"
    FINE = "fine"
    MASTERWORK = "masterwork"


# Maps quality to a quantity multiplier (masterwork yields double)
QUALITY_YIELD_MULTIPLIER: dict[ItemQuality, int] = {
    ItemQuality.POOR: 1,
    ItemQuality.STANDARD: 1,
    ItemQuality.FINE: 1,
    ItemQuality.MASTERWORK: 2,
}

# Quality label appended to item name for display (empty = no suffix)
QUALITY_SUFFIX: dict[ItemQuality, str] = {
    ItemQuality.POOR: " (poor)",
    ItemQuality.STANDARD: "",
    ItemQuality.FINE: " (fine)",
    ItemQuality.MASTERWORK: " (masterwork)",
}


# ---------- Default quality function ----------

QualityFunction = Callable[[float], ItemQuality]


def default_quality_function(skill_level: float) -> ItemQuality:
    """Map skill level (0.0–1.0) to a quality tier."""
    if skill_level < 0.3:
        return ItemQuality.POOR
    if skill_level < 0.6:
        return ItemQuality.STANDARD
    if skill_level < 0.8:
        return ItemQuality.FINE
    return ItemQuality.MASTERWORK


# ---------- Recipe ----------

@dataclass
class Recipe:
    """
    A crafting recipe — the blueprint for producing an item.

    Extensibility fields:
      tags:     Arbitrary labels for filtering (e.g. "weapon", "tier_2",
                "elven"). AI Game Studio uses these to scope recipe
                availability per world theme.
      metadata: Open dict for game-designer data (lore, unlock conditions,
                sell_price_bonus, etc.).
      quality_fn: Optional per-recipe quality function. Falls back to
                  the manager's default when None.
    """
    recipe_id: str
    name: str
    result_item: str
    result_quantity: int = 1

    ingredients: dict[str, int] = field(default_factory=dict)
    required_skill: str = "crafting"
    min_skill_level: float = 0.0
    craft_time: float = 10.0      # game minutes
    skill_gain: float = 0.01      # added to NPC skill on completion

    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_fn: QualityFunction | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "name": self.name,
            "result_item": self.result_item,
            "result_quantity": self.result_quantity,
            "ingredients": dict(self.ingredients),
            "required_skill": self.required_skill,
            "min_skill_level": self.min_skill_level,
            "craft_time": self.craft_time,
            "skill_gain": self.skill_gain,
            "tags": sorted(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        """Deserialise a recipe from a plain dict (e.g. JSON config)."""
        return cls(
            recipe_id=data["recipe_id"],
            name=data["name"],
            result_item=data["result_item"],
            result_quantity=data.get("result_quantity", 1),
            ingredients=dict(data.get("ingredients", {})),
            required_skill=data.get("required_skill", "crafting"),
            min_skill_level=data.get("min_skill_level", 0.0),
            craft_time=data.get("craft_time", 10.0),
            skill_gain=data.get("skill_gain", 0.01),
            tags=set(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )


# ---------- Default Recipes ----------

DEFAULT_RECIPES: list[Recipe] = [
    Recipe(
        recipe_id="plank", name="Plank",
        result_item="plank", result_quantity=2,
        ingredients={"wood": 1},
        required_skill="crafting", min_skill_level=0.0,
        craft_time=5.0, skill_gain=0.005,
        tags={"basic", "material"},
    ),
    Recipe(
        recipe_id="bread", name="Bread",
        result_item="food", result_quantity=2,
        ingredients={"wheat": 2},
        required_skill="cooking", min_skill_level=0.1,
        craft_time=10.0, skill_gain=0.01,
        tags={"basic", "food"},
    ),
    Recipe(
        recipe_id="berry_jam", name="Berry Jam",
        result_item="food", result_quantity=1,
        ingredients={"berries": 3},
        required_skill="cooking", min_skill_level=0.0,
        craft_time=8.0, skill_gain=0.01,
        tags={"basic", "food"},
    ),
    Recipe(
        recipe_id="iron_ingot", name="Iron Ingot",
        result_item="iron_ingot", result_quantity=1,
        ingredients={"iron": 2},
        required_skill="smithing", min_skill_level=0.2,
        craft_time=15.0, skill_gain=0.01,
        tags={"basic", "material", "smithing"},
    ),
    Recipe(
        recipe_id="stone_block", name="Stone Block",
        result_item="stone_block", result_quantity=1,
        ingredients={"stone": 2},
        required_skill="crafting", min_skill_level=0.1,
        craft_time=12.0, skill_gain=0.01,
        tags={"basic", "material"},
    ),
    Recipe(
        recipe_id="steel_sword", name="Steel Sword",
        result_item="steel_sword", result_quantity=1,
        ingredients={"iron_ingot": 2, "wood": 1},
        required_skill="smithing", min_skill_level=0.5,
        craft_time=30.0, skill_gain=0.02,
        tags={"weapon", "smithing", "advanced"},
    ),
    Recipe(
        recipe_id="iron_tools", name="Iron Tools",
        result_item="iron_tools", result_quantity=1,
        ingredients={"iron_ingot": 1, "wood": 1},
        required_skill="smithing", min_skill_level=0.3,
        craft_time=20.0, skill_gain=0.015,
        tags={"tool", "smithing"},
    ),
    Recipe(
        recipe_id="healing_salve", name="Healing Salve",
        result_item="healing_salve", result_quantity=1,
        ingredients={"berries": 2},
        required_skill="medicine", min_skill_level=0.2,
        craft_time=10.0, skill_gain=0.01,
        tags={"medicine", "consumable"},
    ),
    Recipe(
        recipe_id="flour", name="Flour",
        result_item="flour", result_quantity=2,
        ingredients={"wheat": 3},
        required_skill="cooking", min_skill_level=0.0,
        craft_time=8.0, skill_gain=0.005,
        tags={"basic", "food", "material"},
    ),
    Recipe(
        recipe_id="furniture", name="Furniture",
        result_item="furniture", result_quantity=1,
        ingredients={"plank": 4},
        required_skill="crafting", min_skill_level=0.3,
        craft_time=25.0, skill_gain=0.02,
        tags={"advanced", "construction"},
    ),
]


# ---------- Crafting Session ----------

@dataclass
class CraftingSession:
    """Tracks an in-progress crafting action."""
    npc_id: str
    recipe: Recipe
    started_at: float        # game minutes (absolute)
    quality: ItemQuality = ItemQuality.STANDARD
    ingredients_consumed: dict[str, int] = field(default_factory=dict)

    @property
    def completes_at(self) -> float:
        return self.started_at + self.recipe.craft_time


# ---------- Crafting Manager ----------

class CraftingManager:
    """
    Manages recipes, crafting sessions, and quality determination.

    Extensibility API for AI Game Studio:
      add_recipe / remove_recipe / replace_recipe — modify recipe pool
      add_recipes_from_dicts — bulk load from JSON/config
      set_quality_function — replace the global quality curve
      get_recipes_by_tag — filter recipes for themed worlds
    """

    def __init__(
        self,
        recipes: list[Recipe] | None = None,
        quality_fn: QualityFunction | None = None,
    ):
        self._recipes: dict[str, Recipe] = {}
        self._sessions: dict[str, CraftingSession] = {}  # npc_id → session
        self._quality_fn: QualityFunction = quality_fn or default_quality_function

        for recipe in (recipes if recipes is not None else DEFAULT_RECIPES):
            self._recipes[recipe.recipe_id] = recipe

    # ---------- Recipe Management ----------

    def add_recipe(self, recipe: Recipe) -> None:
        """Register a new recipe (overwrites if recipe_id exists)."""
        self._recipes[recipe.recipe_id] = recipe

    def remove_recipe(self, recipe_id: str) -> bool:
        """Remove a recipe by ID. Returns True if it existed."""
        if recipe_id in self._recipes:
            del self._recipes[recipe_id]
            return True
        return False

    def replace_recipe(self, recipe_id: str, recipe: Recipe) -> bool:
        """Replace an existing recipe. Returns False if original didn't exist."""
        if recipe_id not in self._recipes:
            return False
        if recipe.recipe_id != recipe_id:
            del self._recipes[recipe_id]
        self._recipes[recipe.recipe_id] = recipe
        return True

    def add_recipes_from_dicts(self, recipe_dicts: list[dict[str, Any]]) -> int:
        """Bulk-load recipes from plain dicts (e.g. JSON config). Returns count added."""
        count = 0
        for data in recipe_dicts:
            recipe = Recipe.from_dict(data)
            self._recipes[recipe.recipe_id] = recipe
            count += 1
        return count

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        return self._recipes.get(recipe_id)

    def get_all_recipes(self) -> list[Recipe]:
        return list(self._recipes.values())

    def get_recipes_by_tag(self, tag: str) -> list[Recipe]:
        """Return all recipes that have a specific tag."""
        return [r for r in self._recipes.values() if tag in r.tags]

    def get_recipes_by_skill(self, skill: str) -> list[Recipe]:
        """Return all recipes that use a specific skill."""
        return [r for r in self._recipes.values() if r.required_skill == skill]

    # ---------- Quality Function ----------

    def set_quality_function(self, fn: QualityFunction) -> None:
        """Replace the global skill-to-quality mapping."""
        self._quality_fn = fn

    def determine_quality(self, npc: NPC, recipe: Recipe) -> ItemQuality:
        """Determine quality for a craft attempt by this NPC."""
        skill_level = npc.skills.get(recipe.required_skill, 0.0)
        fn = recipe.quality_fn or self._quality_fn
        return fn(skill_level)

    # ---------- Crafting Checks ----------

    def can_craft(self, npc: NPC, recipe: Recipe) -> tuple[bool, str]:
        """Check whether an NPC can craft a recipe right now."""
        if npc.npc_id in self._sessions:
            return False, "already crafting"

        skill_level = npc.skills.get(recipe.required_skill, 0.0)
        if skill_level < recipe.min_skill_level:
            return False, (
                f"requires {recipe.required_skill} >= {recipe.min_skill_level:.1f}"
            )

        for item, qty in recipe.ingredients.items():
            if npc.inventory.get(item, 0) < qty:
                return False, f"needs {qty} {item}"

        return True, "ok"

    def get_available_recipes(self, npc: NPC) -> list[Recipe]:
        """Return all recipes this NPC can currently craft."""
        return [r for r in self._recipes.values() if self.can_craft(npc, r)[0]]

    # ---------- Crafting Lifecycle ----------

    def start_crafting(
        self,
        npc: NPC,
        recipe_id: str,
        game_time: float,
    ) -> tuple[bool, str]:
        """
        Begin crafting. Validates skill and ingredients, deducts
        ingredients, and starts a timed session.
        """
        recipe = self._recipes.get(recipe_id)
        if recipe is None:
            return False, f"unknown recipe: {recipe_id}"

        can, reason = self.can_craft(npc, recipe)
        if not can:
            return False, reason

        # Deduct ingredients
        consumed: dict[str, int] = {}
        for item, qty in recipe.ingredients.items():
            npc.inventory[item] = npc.inventory.get(item, 0) - qty
            consumed[item] = qty

        quality = self.determine_quality(npc, recipe)

        session = CraftingSession(
            npc_id=npc.npc_id,
            recipe=recipe,
            started_at=game_time,
            quality=quality,
            ingredients_consumed=consumed,
        )
        self._sessions[npc.npc_id] = session

        logger.debug(
            "%s started crafting %s (quality: %s)",
            npc.name, recipe.name, quality.value,
        )
        return True, "ok"

    def complete_crafting(
        self,
        npc: NPC,
        game_time: float,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Complete a crafting session if enough time has passed.

        Adds the result item(s) to NPC inventory, applies skill gain.
        Returns (completed, result_dict).
        """
        session = self._sessions.get(npc.npc_id)
        if session is None:
            return False, {"reason": "no active session"}

        if game_time < session.completes_at:
            return False, {
                "reason": "not finished",
                "remaining": session.completes_at - game_time,
            }

        recipe = session.recipe
        quality = session.quality

        # Calculate yield
        base_qty = recipe.result_quantity
        multiplier = QUALITY_YIELD_MULTIPLIER.get(quality, 1)
        final_qty = base_qty * multiplier

        # Determine inventory key — suffix only for fine/masterwork
        suffix = QUALITY_SUFFIX.get(quality, "")
        item_key = recipe.result_item + suffix if suffix else recipe.result_item

        npc.inventory[item_key] = npc.inventory.get(item_key, 0) + final_qty

        # Skill gain
        old_skill = npc.skills.get(recipe.required_skill, 0.0)
        new_skill = min(1.0, old_skill + recipe.skill_gain)
        npc.skills[recipe.required_skill] = new_skill

        del self._sessions[npc.npc_id]

        result = {
            "item": item_key,
            "quantity": final_qty,
            "quality": quality.value,
            "skill_gained": round(recipe.skill_gain, 4),
            "new_skill_level": round(new_skill, 4),
        }

        logger.debug(
            "%s crafted %d × %s (%s)",
            npc.name, final_qty, item_key, quality.value,
        )
        return True, result

    def cancel_crafting(self, npc: NPC) -> tuple[bool, str]:
        """
        Cancel an in-progress craft and refund ingredients.

        Returns (cancelled, message).
        """
        session = self._sessions.get(npc.npc_id)
        if session is None:
            return False, "no active session"

        # Refund ingredients
        for item, qty in session.ingredients_consumed.items():
            npc.inventory[item] = npc.inventory.get(item, 0) + qty

        del self._sessions[npc.npc_id]
        return True, "ok"

    def get_session(self, npc_id: str) -> CraftingSession | None:
        return self._sessions.get(npc_id)

    def is_crafting(self, npc_id: str) -> bool:
        return npc_id in self._sessions

    # ---------- State ----------

    def get_state(self) -> dict[str, Any]:
        return {
            "recipes": len(self._recipes),
            "active_sessions": len(self._sessions),
        }

    def get_stats(self) -> dict[str, Any]:
        tags: set[str] = set()
        for r in self._recipes.values():
            tags.update(r.tags)
        return {
            "recipe_count": len(self._recipes),
            "active_sessions": len(self._sessions),
            "tags_in_use": sorted(tags),
            "skills_used": sorted({r.required_skill for r in self._recipes.values()}),
        }
