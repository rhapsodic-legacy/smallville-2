"""
Action catalogue for the deterministic planner.

Defines what NPCs can do and provides a registry that is fully
extensible at runtime — the AI Game Studio can add, remove, and
replace actions without touching this file.

Each ActionDef describes:
- What the action is (type, display name)
- When it's available (preconditions)
- How to score its utility (need_weights, personality_weights, time_weights)
- Which execution rule set handles it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.npc.cognition.planner.context import PlannerContext

logger = logging.getLogger(__name__)


# ---------- Action types ----------

class ActionType(Enum):
    """Built-in action types. Custom actions use string keys instead."""
    EAT = "eat"
    SLEEP = "sleep"
    WORK = "work"
    GATHER = "gather"
    TRADE = "trade"
    CRAFT = "craft"
    SOCIALISE = "socialise"
    WANDER = "wander"
    FLEE = "flee"
    REST = "rest"
    PRAY = "pray"
    PATROL = "patrol"
    CONSTRUCT = "construct"


# ---------- Action definition ----------

@dataclass
class ActionDef:
    """
    Definition of a single action an NPC can take.

    All scoring weights are optional — missing keys are treated as 0.0.
    The precondition callable is the first gate: if it returns False,
    the action won't even be scored.

    Attributes:
        action_id:          Unique string key (e.g. "eat", "custom_dance")
        display_name:       Human-readable name for UI / logs
        need_weights:       Maps need name -> weight. Needs are 0-1 floats on the NPC.
                            Higher need value * higher weight = higher utility.
                            e.g. {"hunger": 2.0} means hunger drives eating.
        personality_weights: Maps Big Five trait -> weight modifier.
                            e.g. {"extraversion": 0.5} boosts score for extroverts.
        time_weights:       Maps schedule slot -> multiplier.
                            e.g. {"night": 2.0} makes sleep score high at night.
        base_utility:       Constant added to the score before weighting.
        precondition:       Optional callable(npc, context) -> bool.
                            If provided and returns False, action is skipped.
        target_selector:    Optional callable(npc, context) -> (x, z) | None.
                            Picks where the NPC goes for this action.
        tags:               Arbitrary tags for filtering (e.g. "economy", "combat")
        metadata:           Open dict for AI Game Studio extensions.
    """
    action_id: str
    display_name: str
    need_weights: dict[str, float] = field(default_factory=dict)
    personality_weights: dict[str, float] = field(default_factory=dict)
    time_weights: dict[str, float] = field(default_factory=dict)
    base_utility: float = 0.0
    precondition: Callable[[Any, Any], bool] | None = None
    target_selector: Callable[[Any, Any], tuple[int, int] | None] | None = None
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------- Action registry ----------

class ActionRegistry:
    """
    Runtime-extensible catalogue of available actions.

    The registry ships with sensible defaults but every action can
    be added, removed, replaced, or queried by tag. This is the
    primary extension point for the AI Game Studio.
    """

    def __init__(self) -> None:
        self._actions: dict[str, ActionDef] = {}

    def register(self, action: ActionDef) -> None:
        """Add or overwrite an action definition."""
        self._actions[action.action_id] = action

    def remove(self, action_id: str) -> ActionDef | None:
        """Remove and return an action, or None if not found."""
        return self._actions.pop(action_id, None)

    def replace(self, action_id: str, action: ActionDef) -> None:
        """Replace an action. Raises KeyError if it doesn't exist."""
        if action_id not in self._actions:
            raise KeyError(f"Action '{action_id}' not registered")
        self._actions[action_id] = action

    def get(self, action_id: str) -> ActionDef | None:
        return self._actions.get(action_id)

    def all(self) -> list[ActionDef]:
        return list(self._actions.values())

    def by_tag(self, tag: str) -> list[ActionDef]:
        return [a for a in self._actions.values() if tag in a.tags]

    def ids(self) -> set[str]:
        return set(self._actions.keys())

    def __len__(self) -> int:
        return len(self._actions)

    def __contains__(self, action_id: str) -> bool:
        return action_id in self._actions


# ---------- Default preconditions ----------

def _can_eat(npc: Any, ctx: Any) -> bool:
    """Can eat if hungry and a tavern/home exists."""
    return npc.hunger > 0.2

def _can_sleep(npc: Any, ctx: Any) -> bool:
    return npc.energy < 0.7

def _can_gather(npc: Any, ctx: Any) -> bool:
    """Can gather if resource nodes exist nearby."""
    return len(ctx.nearby_resource_nodes) > 0

def _can_trade(npc: Any, ctx: Any) -> bool:
    """Can trade if there's a market and NPC has inventory or gold."""
    return ctx.has_market and (npc.gold > 5 or len(npc.inventory) > 0)

def _can_craft(npc: Any, ctx: Any) -> bool:
    """Can craft if NPC has materials and recipes are available."""
    return len(npc.inventory) > 0 and len(ctx.available_recipes) > 0

def _can_socialise(npc: Any, ctx: Any) -> bool:
    """Can socialise if other NPCs are nearby."""
    return len(ctx.nearby_npcs) > 0

def _can_flee(npc: Any, ctx: Any) -> bool:
    """Flee only if there's an active threat."""
    return ctx.threat_level > 0.0

def _can_patrol(npc: Any, ctx: Any) -> bool:
    return npc.occupation == "guard"

def _can_pray(npc: Any, ctx: Any) -> bool:
    return ctx.has_church


def _can_construct(npc: Any, ctx: Any) -> bool:
    """Can construct if there are active construction sites."""
    return len(getattr(ctx, "construction_sites", [])) > 0


def _select_construct_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    """Pick the nearest construction site that still needs work."""
    sites = getattr(ctx, "construction_sites", [])
    if not sites:
        return None
    # Prefer sites that need resources the NPC has, otherwise pick nearest
    best = min(sites, key=lambda s: s["distance"])
    return (best["x"], best["z"])


# ---------- Default target selectors ----------

def _select_eat_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    """Go to tavern if available, otherwise home."""
    if ctx.tavern_door:
        return ctx.tavern_door
    return (npc.home_x, npc.home_z)

def _select_sleep_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    return (npc.home_x, npc.home_z)

def _select_work_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    return (npc.work_x, npc.work_z)

def _select_gather_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    """Pick the nearest resource node with capacity."""
    if not ctx.nearby_resource_nodes:
        return None
    best = min(ctx.nearby_resource_nodes, key=lambda n: n["distance"])
    return (best["x"], best["z"])

def _select_socialise_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    if ctx.tavern_door:
        return ctx.tavern_door
    if ctx.nearby_npcs:
        other = ctx.nearby_npcs[0]
        return (other["x"], other["z"])
    return None

def _select_wander_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    return ctx.random_passable_tile

def _select_pray_target(npc: Any, ctx: Any) -> tuple[int, int] | None:
    return ctx.church_door


# ---------- Default actions ----------

def build_default_registry() -> ActionRegistry:
    """Create the standard action registry with all built-in actions."""
    registry = ActionRegistry()

    registry.register(ActionDef(
        action_id="eat",
        display_name="Eat",
        need_weights={"hunger": 4.0},
        time_weights={"early_morning": 1.5, "morning": 0.5, "evening": 1.3},
        base_utility=0.1,
        precondition=_can_eat,
        target_selector=_select_eat_target,
        tags={"survival", "basic"},
    ))

    registry.register(ActionDef(
        action_id="sleep",
        display_name="Sleep",
        need_weights={"energy_deficit": 4.0},
        time_weights={"night": 3.0, "evening": 0.5},
        base_utility=0.05,
        precondition=_can_sleep,
        target_selector=_select_sleep_target,
        tags={"survival", "basic"},
    ))

    registry.register(ActionDef(
        action_id="work",
        display_name="Work",
        personality_weights={"conscientiousness": 0.8},
        time_weights={"morning": 2.0, "afternoon": 1.8},
        base_utility=1.0,
        target_selector=_select_work_target,
        tags={"economy", "basic"},
    ))

    registry.register(ActionDef(
        action_id="gather",
        display_name="Gather resources",
        personality_weights={"conscientiousness": 0.4},
        time_weights={"morning": 1.5, "afternoon": 1.3},
        base_utility=0.6,
        precondition=_can_gather,
        target_selector=_select_gather_target,
        tags={"economy", "outdoor"},
    ))

    registry.register(ActionDef(
        action_id="trade",
        display_name="Trade at market",
        personality_weights={"extraversion": 0.3, "agreeableness": 0.2},
        time_weights={"morning": 1.5, "afternoon": 1.8},
        base_utility=0.5,
        precondition=_can_trade,
        tags={"economy", "social"},
    ))

    registry.register(ActionDef(
        action_id="craft",
        display_name="Craft items",
        personality_weights={"conscientiousness": 0.5, "openness": 0.3},
        time_weights={"morning": 1.3, "afternoon": 1.5},
        base_utility=0.5,
        precondition=_can_craft,
        target_selector=_select_work_target,
        tags={"economy"},
    ))

    registry.register(ActionDef(
        action_id="socialise",
        display_name="Socialise",
        personality_weights={"extraversion": 1.2, "agreeableness": 0.5},
        time_weights={"evening": 2.0, "afternoon": 1.0},
        base_utility=0.4,
        precondition=_can_socialise,
        target_selector=_select_socialise_target,
        tags={"social", "basic"},
    ))

    registry.register(ActionDef(
        action_id="wander",
        display_name="Wander around town",
        personality_weights={"openness": 0.6},
        base_utility=0.2,
        target_selector=_select_wander_target,
        tags={"basic"},
    ))

    registry.register(ActionDef(
        action_id="flee",
        display_name="Flee from danger",
        base_utility=0.0,  # Only scores high via threat_level need
        need_weights={"threat": 5.0},
        precondition=_can_flee,
        tags={"survival", "combat"},
    ))

    registry.register(ActionDef(
        action_id="rest",
        display_name="Rest at home",
        need_weights={"energy_deficit": 1.0},
        time_weights={"night": 1.5},
        base_utility=0.3,
        target_selector=_select_sleep_target,
        tags={"basic"},
    ))

    registry.register(ActionDef(
        action_id="patrol",
        display_name="Patrol the town",
        personality_weights={"conscientiousness": 0.5},
        time_weights={"morning": 1.5, "night": 2.0},
        base_utility=0.8,
        precondition=_can_patrol,
        tags={"guard", "duty"},
    ))

    registry.register(ActionDef(
        action_id="pray",
        display_name="Pray at the church",
        personality_weights={"neuroticism": 0.3},
        time_weights={"early_morning": 1.5, "evening": 1.0},
        base_utility=0.2,
        precondition=_can_pray,
        target_selector=_select_pray_target,
        tags={"social"},
    ))

    registry.register(ActionDef(
        action_id="construct",
        display_name="Contribute to construction",
        personality_weights={"conscientiousness": 0.8, "agreeableness": 0.4},
        time_weights={"morning": 1.5, "afternoon": 1.8},
        base_utility=1.2,
        precondition=_can_construct,
        target_selector=_select_construct_target,
        tags={"economy", "community"},
    ))

    return registry
