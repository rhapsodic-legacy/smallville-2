"""
Sims-style utility scoring system.

Evaluates every available action for an NPC and ranks them by
utility. The highest-scoring action wins. Scoring combines:

1. Need urgency (exponential curves — hunger at 0.9 is way more
   urgent than at 0.5)
2. Personality modifiers (an extrovert scores socialising higher)
3. Time-of-day modifiers (sleep scores high at night)
4. Action base utility (work has inherent value)

The scorer is pluggable — custom need curves, personality
multipliers, and scoring functions can be injected at runtime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from core.npc.cognition.goal_mapper import (
    GOAL_ACTION_BONUS, GOAL_TAG_BONUS,
    aggregate_boost_actions, aggregate_boost_tags,
)

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.npc.cognition.planner.actions import ActionDef
    from core.npc.cognition.planner.context import PlannerContext

logger = logging.getLogger(__name__)


# ---------- Scored result ----------

@dataclass
class ScoredAction:
    """An action with its computed utility score and breakdown."""
    action_id: str
    display_name: str
    total_score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    target: tuple[int, int] | None = None


# ---------- Need curves ----------

def exponential_curve(value: float, steepness: float = 3.0) -> float:
    """
    Maps a need value (0-1) to urgency (0-1) with exponential growth.

    At low values, urgency is near zero. Past ~0.6, urgency climbs
    steeply. At 1.0, urgency is 1.0. Steepness controls the curve
    shape — higher = sharper transition.

    This mirrors the Sims' need decay: you don't care about food
    at 20% hungry, but at 90% it dominates everything.
    """
    clamped = max(0.0, min(1.0, value))
    return (math.exp(steepness * clamped) - 1) / (math.exp(steepness) - 1)


def linear_curve(value: float, **_kwargs: Any) -> float:
    """Simple linear mapping — urgency equals value directly."""
    return max(0.0, min(1.0, value))


def step_curve(value: float, threshold: float = 0.5, **_kwargs: Any) -> float:
    """Binary: 0 below threshold, 1 at or above."""
    return 1.0 if value >= threshold else 0.0


# Type for pluggable need curve functions
NeedCurve = Callable[..., float]


# ---------- Need extraction ----------

def extract_needs(npc: NPC, ctx: PlannerContext) -> dict[str, float]:
    """
    Extract the current need values for scoring.

    Returns a dict of need_name -> value (0-1 where 1 = maximum need).
    This is pluggable — subclass UtilityScorer and override to add
    custom needs (e.g. "boredom", "loyalty", "fear").
    """
    return {
        "hunger": npc.hunger,
        "energy_deficit": 1.0 - npc.energy,
        "threat": ctx.threat_level,
        # Social need: introverts need less, extroverts need more
        # Rises over time without conversation (simplified here)
        "social": max(0.0, npc.personality.extraversion - 0.2),
    }


# ---------- Utility scorer ----------

class UtilityScorer:
    """
    Scores all available actions for an NPC.

    Fully pluggable:
    - Replace need_curve to change urgency shapes
    - Replace need_extractor to add custom needs
    - Replace personality_multiplier to change trait effects
    - Override score_action() for completely custom logic per action
    """

    def __init__(
        self,
        need_curve: NeedCurve | None = None,
        need_extractor: Callable[[Any, Any], dict[str, float]] | None = None,
        personality_multiplier: float = 1.0,
        time_multiplier: float = 1.0,
    ) -> None:
        self.need_curve = need_curve or exponential_curve
        self.need_extractor = need_extractor or extract_needs
        self.personality_multiplier = personality_multiplier
        self.time_multiplier = time_multiplier

        # Per-action scoring overrides: action_id -> custom scorer
        self._custom_scorers: dict[
            str, Callable[[Any, Any, Any], float]
        ] = {}

    def set_custom_scorer(
        self,
        action_id: str,
        scorer: Callable[[Any, Any, Any], float],
    ) -> None:
        """Register a custom scoring function for a specific action."""
        self._custom_scorers[action_id] = scorer

    def remove_custom_scorer(self, action_id: str) -> None:
        self._custom_scorers.pop(action_id, None)

    def evaluate_all(
        self,
        npc: NPC,
        ctx: PlannerContext,
        actions: list[ActionDef],
    ) -> list[ScoredAction]:
        """
        Score all available actions and return sorted by utility (highest first).

        Actions whose precondition fails are excluded entirely.
        """
        needs = self.need_extractor(npc, ctx)
        # Aggregate goal affinity once per pass — it's identical for
        # every action scored on this NPC's tick.
        boost_actions = aggregate_boost_actions(npc)
        boost_tags = aggregate_boost_tags(npc)
        results: list[ScoredAction] = []

        for action in actions:
            # Gate: precondition check
            if action.precondition is not None:
                try:
                    if not action.precondition(npc, ctx):
                        continue
                except Exception:
                    continue  # broken precondition = skip

            score, breakdown = self.score_action(
                npc, ctx, action, needs,
                boost_actions=boost_actions,
                boost_tags=boost_tags,
            )

            # Resolve target
            target = None
            if action.target_selector:
                try:
                    target = action.target_selector(npc, ctx)
                except Exception:
                    pass

            results.append(ScoredAction(
                action_id=action.action_id,
                display_name=action.display_name,
                total_score=score,
                breakdown=breakdown,
                target=target,
            ))

        results.sort(key=lambda s: s.total_score, reverse=True)
        return results

    def score_action(
        self,
        npc: NPC,
        ctx: PlannerContext,
        action: ActionDef,
        needs: dict[str, float],
        boost_actions: set[str] | None = None,
        boost_tags: set[str] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute the utility score for a single action.

        `boost_actions` and `boost_tags` come from the NPC's active
        derived goals (see `goal_mapper`). Passed by `evaluate_all`
        so they're computed once per evaluation pass. If omitted
        (e.g. a caller scoring a single action in isolation), they
        are resolved from the NPC on demand.

        Returns (total_score, breakdown_dict).
        """
        # Check for per-action custom scorer
        if action.action_id in self._custom_scorers:
            custom_score = self._custom_scorers[action.action_id](
                npc, ctx, action,
            )
            return (custom_score, {"custom": custom_score})

        if boost_actions is None:
            boost_actions = aggregate_boost_actions(npc)
        if boost_tags is None:
            boost_tags = aggregate_boost_tags(npc)

        breakdown: dict[str, float] = {}

        # 1. Base utility
        base = action.base_utility
        breakdown["base"] = base

        # 2. Need-weighted utility (Sims-style curves)
        need_score = 0.0
        for need_name, weight in action.need_weights.items():
            raw_value = needs.get(need_name, 0.0)
            curved = self.need_curve(raw_value)
            contribution = curved * weight
            need_score += contribution
            breakdown[f"need_{need_name}"] = round(contribution, 3)
        breakdown["needs_total"] = round(need_score, 3)

        # 3. Personality modifier
        personality_score = 0.0
        for trait_name, weight in action.personality_weights.items():
            trait_value = getattr(npc.personality, trait_name, 0.5)
            contribution = trait_value * weight * self.personality_multiplier
            personality_score += contribution
            breakdown[f"personality_{trait_name}"] = round(contribution, 3)
        breakdown["personality_total"] = round(personality_score, 3)

        # 4. Time-of-day modifier
        time_score = action.time_weights.get(ctx.current_slot, 0.0)
        time_score *= self.time_multiplier
        breakdown["time"] = round(time_score, 3)

        # 5. Goal affinity — derived goals bias the NPC toward
        # action_ids they explicitly recommend, and more weakly toward
        # actions that share a tag with the goal's themes.
        goal_bonus = 0.0
        if action.action_id in boost_actions:
            goal_bonus += GOAL_ACTION_BONUS
        if boost_tags and action.tags & boost_tags:
            goal_bonus += GOAL_TAG_BONUS
        breakdown["goal_affinity"] = round(goal_bonus, 3)

        total = base + need_score + personality_score + time_score + goal_bonus
        breakdown["total"] = round(total, 3)
        return (total, breakdown)
