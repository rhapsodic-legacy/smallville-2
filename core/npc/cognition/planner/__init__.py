"""
Deterministic planner — Sims-style utility + Total War-style rules.

Public API for the planner system. Creates a DeterministicPlanner
with default or custom components, and exposes plan_action() as the
main entry point.

The planner is fully modular — every component (action registry,
utility scorer, rule registry, context builder) can be swapped out
independently at runtime. This is the primary extension point for
the AI Game Studio's cognition tuning.

Usage:
    planner = DeterministicPlanner(grid, buildings)
    action = planner.plan_action(npc, all_npcs, current_slot)
    # action is a PlannedAction with target coords, description, etc.

Custom components:
    planner = DeterministicPlanner(
        grid, buildings,
        action_registry=my_registry,
        scorer=my_scorer,
        rule_registry=my_rules,
    )
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from core.npc.cognition.planner.actions import (
    ActionType,
    ActionDef,
    ActionRegistry,
    build_default_registry,
)
from core.npc.cognition.planner.context import (
    PlannerContext,
    ContextBuilder,
)
from core.npc.cognition.planner.utility import (
    ScoredAction,
    UtilityScorer,
    exponential_curve,
    linear_curve,
    step_curve,
)
from core.npc.cognition.planner.rules import (
    PlannedAction,
    RuleSet,
    RuleRegistry,
    build_default_rules,
)

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid
    from core.world.generator import PlacedBuilding

logger = logging.getLogger(__name__)


class DeterministicPlanner:
    """
    The main deterministic planner. Orchestrates:
    1. Context building (world snapshot)
    2. Utility scoring (rank all actions)
    3. Rule execution (translate winning action into spatial behaviour)

    Every component is independently swappable:
    - action_registry: what actions exist
    - scorer: how actions are scored
    - rule_registry: how actions are executed
    - context_builder: how world state is gathered
    """

    def __init__(
        self,
        grid: Grid,
        buildings: list[PlacedBuilding],
        action_registry: ActionRegistry | None = None,
        scorer: UtilityScorer | None = None,
        rule_registry: RuleRegistry | None = None,
        context_builder: ContextBuilder | None = None,
        seed: int | None = None,
    ) -> None:
        self.grid = grid
        self.buildings = buildings
        self.actions = action_registry or build_default_registry()
        self.scorer = scorer or UtilityScorer()
        self.rules = rule_registry or build_default_rules()
        self.context_builder = context_builder or ContextBuilder(
            grid, buildings, seed=seed,
        )

    def plan_action(
        self,
        npc: NPC,
        all_npcs: list[NPC],
        current_slot: str,
        game_minutes: float = 0.0,
        current_day: int = 0,
        resource_nodes: list[dict[str, Any]] | None = None,
        available_recipes: list[str] | None = None,
        construction_sites: list[dict[str, Any]] | None = None,
        threat_level: float = 0.0,
        threat_x: int = 0,
        threat_z: int = 0,
    ) -> PlannedAction | None:
        """
        Plan the best action for an NPC right now.

        Returns a PlannedAction with target coordinates and description,
        or None if no action is viable (extremely unlikely with defaults).
        """
        # 1. Build context
        ctx = self.context_builder.build(
            npc=npc,
            all_npcs=all_npcs,
            current_slot=current_slot,
            game_minutes=game_minutes,
            current_day=current_day,
            resource_nodes=resource_nodes,
            available_recipes=available_recipes,
            construction_sites=construction_sites,
            threat_level=threat_level,
            threat_x=threat_x,
            threat_z=threat_z,
        )

        # 2. Score all actions
        scored = self.scorer.evaluate_all(
            npc, ctx, self.actions.all(),
        )

        if not scored:
            logger.debug("%s: no viable actions", npc.name)
            return None

        # 3. Try rule execution in score order (fall through if rules fail)
        for candidate in scored:
            result = self.rules.execute(
                candidate.action_id, npc, ctx, candidate,
            )
            if result is not None:
                logger.debug(
                    "%s planner: %s (score=%.2f)",
                    npc.name, result.action_id, result.utility_score,
                )
                return result

        # All rules failed — shouldn't happen with fallback, but be safe
        logger.warning("%s: all planner rules failed", npc.name)
        return None

    def score_all(
        self,
        npc: NPC,
        all_npcs: list[NPC],
        current_slot: str,
        **kwargs: Any,
    ) -> list[ScoredAction]:
        """
        Score all actions without executing rules.

        Useful for debugging, UI displays, and the cognition router's
        importance estimation.
        """
        ctx = self.context_builder.build(
            npc=npc,
            all_npcs=all_npcs,
            current_slot=current_slot,
            **kwargs,
        )
        return self.scorer.evaluate_all(npc, ctx, self.actions.all())


__all__ = [
    "DeterministicPlanner",
    "ActionType",
    "ActionDef",
    "ActionRegistry",
    "build_default_registry",
    "PlannerContext",
    "ContextBuilder",
    "ScoredAction",
    "UtilityScorer",
    "exponential_curve",
    "linear_curve",
    "step_curve",
    "PlannedAction",
    "RuleSet",
    "RuleRegistry",
    "build_default_rules",
]
