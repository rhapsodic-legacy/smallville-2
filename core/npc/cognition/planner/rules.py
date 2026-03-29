"""
Execution rules — Total War-style behaviour logic.

Once the utility scorer picks an action, the rule set determines
HOW to execute it in 3D spacetime. Rules handle:

- Target selection refinement (which node, which NPC, which building)
- Spatial behaviour (approach, flee, patrol routes, group movement)
- State transitions (walking → gathering → walking → crafting)
- Action descriptions for the UI

Each action type has a RuleSet. Rule sets are registered in a
RuleRegistry that is fully extensible at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC, ScheduleEntry
    from core.npc.cognition.planner.context import PlannerContext
    from core.npc.cognition.planner.utility import ScoredAction

logger = logging.getLogger(__name__)


# ---------- Planned action output ----------

@dataclass
class PlannedAction:
    """
    The final output of the deterministic planner.

    This is what gets fed back into the execution system —
    same interface as LLM-generated plans, so the rest of the
    engine doesn't care which system produced it.
    """
    action_id: str
    description: str
    target_x: int
    target_z: int
    activity_state: str = "idle"   # maps to ActivityState value
    utility_score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_schedule_entry(self, slot: str) -> Any:
        """Convert to a ScheduleEntry for compatibility with the plan system."""
        from core.npc.models import ScheduleEntry
        return ScheduleEntry(
            slot=slot,
            activity=self.description,
            location="planner",
            priority=max(1, min(10, int(self.utility_score * 2))),
            target_x=self.target_x,
            target_z=self.target_z,
        )


# ---------- Rule set ----------

# A rule function: (npc, context, scored_action) -> PlannedAction | None
RuleFunction = Callable[[Any, Any, Any], PlannedAction | None]


@dataclass
class RuleSet:
    """
    Execution rules for a single action type.

    Rules are tried in order. The first to return a PlannedAction wins.
    If all return None, the action is abandoned and the planner moves
    to the next-best scored action.

    This composability allows complex behaviour: e.g. "gather" might
    try rule_gather_nearest first, then rule_gather_alternate, then
    rule_gather_give_up.
    """
    action_id: str
    rules: list[RuleFunction] = field(default_factory=list)

    def execute(
        self,
        npc: Any,
        ctx: Any,
        scored: Any,
    ) -> PlannedAction | None:
        """Try rules in order, return first successful result."""
        for rule_fn in self.rules:
            try:
                result = rule_fn(npc, ctx, scored)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(
                    "Rule failed for %s/%s: %s",
                    npc.name if hasattr(npc, "name") else "?",
                    self.action_id, e,
                )
        return None


# ---------- Rule registry ----------

class RuleRegistry:
    """
    Maps action IDs to their execution rule sets.

    Extensible: add/replace/remove rule sets at runtime. If no
    rule set exists for an action, a generic fallback is used.
    """

    def __init__(self) -> None:
        self._rule_sets: dict[str, RuleSet] = {}
        self._fallback: RuleFunction = _rule_generic_fallback

    def register(self, rule_set: RuleSet) -> None:
        self._rule_sets[rule_set.action_id] = rule_set

    def remove(self, action_id: str) -> RuleSet | None:
        return self._rule_sets.pop(action_id, None)

    def get(self, action_id: str) -> RuleSet | None:
        return self._rule_sets.get(action_id)

    def set_fallback(self, fn: RuleFunction) -> None:
        """Replace the generic fallback rule."""
        self._fallback = fn

    def execute(
        self,
        action_id: str,
        npc: Any,
        ctx: Any,
        scored: Any,
    ) -> PlannedAction | None:
        """Execute the rule set for an action, falling back to generic."""
        rule_set = self._rule_sets.get(action_id)
        if rule_set:
            result = rule_set.execute(npc, ctx, scored)
            if result is not None:
                return result

        # Fallback
        return self._fallback(npc, ctx, scored)


# ---------- Built-in rules ----------

def _rule_generic_fallback(
    npc: Any, ctx: Any, scored: Any,
) -> PlannedAction | None:
    """Generic: go to the scored target, or stay put."""
    target = scored.target
    if target is None:
        target = (npc.x, npc.z)
    return PlannedAction(
        action_id=scored.action_id,
        description=scored.display_name,
        target_x=target[0],
        target_z=target[1],
        activity_state="idle",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_eat(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    target = scored.target or (npc.home_x, npc.home_z)
    return PlannedAction(
        action_id="eat",
        description="going to eat",
        target_x=target[0],
        target_z=target[1],
        activity_state="eating",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_sleep(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    return PlannedAction(
        action_id="sleep",
        description="going home to sleep",
        target_x=npc.home_x,
        target_z=npc.home_z,
        activity_state="sleeping",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_work(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    desc = f"heading to work as {npc.occupation}"
    return PlannedAction(
        action_id="work",
        description=desc,
        target_x=npc.work_x,
        target_z=npc.work_z,
        activity_state="working",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_gather_nearest(
    npc: Any, ctx: Any, scored: Any,
) -> PlannedAction | None:
    """Gather from the nearest accessible resource node."""
    if not ctx.nearby_resource_nodes:
        return None
    node = ctx.nearby_resource_nodes[0]
    resource_type = node.get("resource_type", "resources")
    return PlannedAction(
        action_id="gather",
        description=f"gathering {resource_type}",
        target_x=node["x"],
        target_z=node["z"],
        activity_state="gathering",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
        metadata={"resource_type": resource_type},
    )


def _rule_socialise(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    """Head towards a social destination (tavern or nearest NPC)."""
    target = scored.target
    if target is None:
        if ctx.tavern_door:
            target = ctx.tavern_door
        elif ctx.nearby_npcs:
            other = ctx.nearby_npcs[0]
            target = (other["x"], other["z"])
        else:
            return None
    return PlannedAction(
        action_id="socialise",
        description="going to socialise",
        target_x=target[0],
        target_z=target[1],
        activity_state="talking",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_wander(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    target = scored.target or ctx.random_passable_tile
    if target is None:
        target = (npc.home_x, npc.home_z)
    return PlannedAction(
        action_id="wander",
        description="wandering around town",
        target_x=target[0],
        target_z=target[1],
        activity_state="idle",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_flee(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    """Flee away from the threat source."""
    if ctx.threat_level <= 0:
        return None
    # Flee direction: away from threat
    dx = npc.x - ctx.threat_x
    dz = npc.z - ctx.threat_z
    dist = abs(dx) + abs(dz)
    if dist == 0:
        dx, dz = 1, 1
        dist = 2
    scale = 15 / max(dist, 1)
    flee_x = npc.x + int(dx * scale)
    flee_z = npc.z + int(dz * scale)
    return PlannedAction(
        action_id="flee",
        description="fleeing from danger!",
        target_x=flee_x,
        target_z=flee_z,
        activity_state="walking",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_rest(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    return PlannedAction(
        action_id="rest",
        description="resting at home",
        target_x=npc.home_x,
        target_z=npc.home_z,
        activity_state="idle",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_patrol(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    """Guards patrol towards a wandering destination."""
    target = ctx.random_passable_tile or (npc.work_x, npc.work_z)
    return PlannedAction(
        action_id="patrol",
        description="patrolling the town",
        target_x=target[0],
        target_z=target[1],
        activity_state="walking",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_craft(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    return PlannedAction(
        action_id="craft",
        description=f"crafting at the workshop",
        target_x=npc.work_x,
        target_z=npc.work_z,
        activity_state="working",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_trade(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    target = scored.target
    if target is None and ctx.has_market:
        # Find market stall door
        target = (npc.work_x, npc.work_z)
    if target is None:
        return None
    return PlannedAction(
        action_id="trade",
        description="heading to the market to trade",
        target_x=target[0],
        target_z=target[1],
        activity_state="working",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_pray(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    if not ctx.church_door:
        return None
    return PlannedAction(
        action_id="pray",
        description="going to the church to pray",
        target_x=ctx.church_door[0],
        target_z=ctx.church_door[1],
        activity_state="idle",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
    )


def _rule_construct(npc: Any, ctx: Any, scored: Any) -> PlannedAction | None:
    """Head to the nearest construction site and contribute."""
    sites = getattr(ctx, "construction_sites", [])
    if not sites:
        return None
    best = min(sites, key=lambda s: s["distance"])
    # Target the approach tile (south of site, like a door)
    tx = best["x"] + best.get("access_dx", 0)
    tz = best["z"] + best.get("access_dz", 0)
    return PlannedAction(
        action_id="construct",
        description=f"contributing to construction at ({best['x']}, {best['z']})",
        target_x=tx,
        target_z=tz,
        activity_state="working",
        utility_score=scored.total_score,
        breakdown=scored.breakdown,
        metadata={"site_id": best.get("site_id", "")},
    )


# ---------- Default rule registry ----------

def build_default_rules() -> RuleRegistry:
    """Create the standard rule registry with all built-in rules."""
    registry = RuleRegistry()

    registry.register(RuleSet("eat", [_rule_eat]))
    registry.register(RuleSet("sleep", [_rule_sleep]))
    registry.register(RuleSet("work", [_rule_work]))
    registry.register(RuleSet("gather", [_rule_gather_nearest]))
    registry.register(RuleSet("socialise", [_rule_socialise]))
    registry.register(RuleSet("wander", [_rule_wander]))
    registry.register(RuleSet("flee", [_rule_flee]))
    registry.register(RuleSet("rest", [_rule_rest]))
    registry.register(RuleSet("patrol", [_rule_patrol]))
    registry.register(RuleSet("craft", [_rule_craft]))
    registry.register(RuleSet("trade", [_rule_trade]))
    registry.register(RuleSet("pray", [_rule_pray]))
    registry.register(RuleSet("construct", [_rule_construct]))

    return registry
