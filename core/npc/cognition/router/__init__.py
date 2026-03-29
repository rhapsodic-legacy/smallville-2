"""
Cognition router — smart dispatcher between LLM and deterministic.

The router sits between "NPC needs to think" and "which system
handles it." For each decision, it evaluates:

1. The policy (user-configured per-decision-type rules)
2. The budget (remaining tokens, throughput capacity)
3. Scene pressure (how many NPCs need decisions right now)
4. NPC priority (story-critical characters always get LLM)
5. Decision importance (novel situations score higher)

The result is a RouteDecision: either LLM or DETERMINISTIC, with
a reason string for debugging and the cognition guide UI.

Every component is independently swappable — the router, budget,
policy, and importance scorer can all be replaced at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

from core.npc.cognition.router.budget import TokenBudget, BudgetSnapshot
from core.npc.cognition.router.policy import (
    CognitionPolicy,
    AutoConfig,
    ROUTE_LLM,
    ROUTE_DETERMINISTIC,
    ROUTE_AUTO,
    VALID_MODES,
    policy_all_llm,
    policy_all_deterministic,
    policy_conversations_only,
    policy_local_llm,
)

if TYPE_CHECKING:
    from core.npc.models import NPC

logger = logging.getLogger(__name__)


# ---------- Route decision ----------

class Route(Enum):
    LLM = "llm"
    DETERMINISTIC = "deterministic"


@dataclass
class RouteDecision:
    """The router's verdict for a single decision."""
    route: Route
    reason: str
    auto_score: float = 0.0  # only set when mode was "auto"
    npc_id: str = ""
    decision_type: str = ""


# ---------- Importance scorer ----------

def default_importance_scorer(
    npc: Any,
    decision_type: str,
    focus_x: int = 0,
    focus_z: int = 0,
    novelty: float = 0.5,
) -> float:
    """
    Score how important/interesting this decision is.

    Higher scores push "auto" decisions towards LLM.
    Returns 0.0–1.0.

    Override this to implement custom importance logic
    (e.g. faction leaders score higher, quest-relevant NPCs, etc.)
    """
    score = 0.0

    # Base importance by decision type
    type_importance = {
        "conversation": 0.7,
        "reflection": 0.6,
        "daily_schedule": 0.4,
        "reaction": 0.5,
        "trade_evaluation": 0.3,
        "craft_choice": 0.2,
        "flee": 0.1,  # must be fast, deterministic preferred
        "gather_choice": 0.1,
        "work_choice": 0.1,
    }
    score += type_importance.get(decision_type, 0.3)

    # Proximity to focus point (player/camera)
    if hasattr(npc, "distance_to"):
        dist = npc.distance_to(focus_x, focus_z)
        proximity = max(0.0, 1.0 - dist / 30.0)  # 0 at 30+ tiles
        score += proximity * 0.3

    # Novelty factor (passed in by caller)
    score += novelty * 0.2

    return min(1.0, score)


# Type for pluggable importance scorers
ImportanceScorer = Callable[..., float]


# ---------- Cognition router ----------

class CognitionRouter:
    """
    Smart dispatcher between LLM and deterministic cognition.

    The router is the central coordination point for the hybrid
    intelligence system. It decides, for each NPC decision, whether
    to spend an LLM call or use the deterministic planner.

    All components are independently replaceable:
    - policy: the user's routing configuration
    - budget: token tracking and limits
    - importance_scorer: how decisions are ranked
    """

    def __init__(
        self,
        policy: CognitionPolicy | None = None,
        budget: TokenBudget | None = None,
        importance_scorer: ImportanceScorer | None = None,
    ) -> None:
        self.policy = policy or CognitionPolicy()
        self.budget = budget or TokenBudget(
            daily_limit=self.policy.token_budget_daily,
            reserve_fraction=self.policy.reserve_fraction,
            max_calls_per_minute=self.policy.max_calls_per_minute,
            max_concurrent=self.policy.max_concurrent_llm_calls,
        )
        self.importance_scorer = importance_scorer or default_importance_scorer

        # Scene pressure tracking
        self._pending_decisions: int = 0

        # Statistics
        self._stats_llm: int = 0
        self._stats_deterministic: int = 0
        self._stats_by_type: dict[str, dict[str, int]] = {}

    # ---------- Main routing API ----------

    def route(
        self,
        npc: NPC,
        decision_type: str,
        focus_x: int = 0,
        focus_z: int = 0,
        novelty: float = 0.5,
        estimated_tokens: int = 200,
    ) -> RouteDecision:
        """
        Decide whether a specific NPC decision should use LLM or deterministic.

        This is the main entry point called by the cognition system
        before each decision.
        """
        mode = self.policy.get_mode(decision_type)

        # Priority NPC override: always LLM regardless of mode
        if self.policy.is_priority_npc(npc.npc_id):
            if self.budget.can_spend(estimated_tokens, is_priority=True):
                decision = RouteDecision(
                    Route.LLM, "priority NPC",
                    npc_id=npc.npc_id, decision_type=decision_type,
                )
                self._record(decision)
                return decision

        # Fixed modes
        if mode == ROUTE_LLM:
            if self.budget.can_spend(
                estimated_tokens,
                is_priority=self.policy.is_priority_decision(decision_type),
            ) and self.budget.can_call():
                decision = RouteDecision(
                    Route.LLM, "policy: llm",
                    npc_id=npc.npc_id, decision_type=decision_type,
                )
            else:
                decision = RouteDecision(
                    Route.DETERMINISTIC, "policy: llm but budget exhausted",
                    npc_id=npc.npc_id, decision_type=decision_type,
                )
            self._record(decision)
            return decision

        if mode == ROUTE_DETERMINISTIC:
            decision = RouteDecision(
                Route.DETERMINISTIC, "policy: deterministic",
                npc_id=npc.npc_id, decision_type=decision_type,
            )
            self._record(decision)
            return decision

        # Auto mode — the smart path
        return self._route_auto(
            npc, decision_type, focus_x, focus_z,
            novelty, estimated_tokens,
        )

    def route_batch(
        self,
        decisions: list[tuple[Any, str]],
        focus_x: int = 0,
        focus_z: int = 0,
    ) -> list[RouteDecision]:
        """
        Route multiple decisions at once with scene pressure awareness.

        Pass a list of (npc, decision_type) tuples. The router considers
        the batch size when computing scene pressure, which may cause
        more decisions to go deterministic.
        """
        self._pending_decisions = len(decisions)
        results = []
        for npc, decision_type in decisions:
            results.append(self.route(
                npc, decision_type, focus_x, focus_z,
            ))
        self._pending_decisions = 0
        return results

    # ---------- Auto routing ----------

    def _route_auto(
        self,
        npc: NPC,
        decision_type: str,
        focus_x: int,
        focus_z: int,
        novelty: float,
        estimated_tokens: int,
    ) -> RouteDecision:
        """Smart routing: compute an importance score and compare to threshold."""
        cfg = self.policy.auto_config

        # Scene pressure check: too many decisions at once → all deterministic
        if self._pending_decisions > self.policy.auto_downgrade_threshold:
            # Exception: priority NPCs still get LLM
            if not self.policy.is_priority_npc(npc.npc_id):
                decision = RouteDecision(
                    Route.DETERMINISTIC,
                    f"scene pressure ({self._pending_decisions} pending)",
                    npc_id=npc.npc_id, decision_type=decision_type,
                )
                self._record(decision)
                return decision

        # Budget check
        is_priority = self.policy.is_priority_decision(decision_type)
        if not self.budget.can_spend(estimated_tokens, is_priority):
            decision = RouteDecision(
                Route.DETERMINISTIC, "budget exhausted",
                npc_id=npc.npc_id, decision_type=decision_type,
            )
            self._record(decision)
            return decision

        if not self.budget.can_call():
            decision = RouteDecision(
                Route.DETERMINISTIC, "throughput limit",
                npc_id=npc.npc_id, decision_type=decision_type,
            )
            self._record(decision)
            return decision

        # Compute importance score
        raw_score = self.importance_scorer(
            npc, decision_type, focus_x, focus_z, novelty,
        )

        # Apply auto_config weights
        score = raw_score
        score *= cfg.importance_weight

        # Budget pressure reduces score (conservative as budget depletes)
        budget_factor = 1.0 - self.budget.budget_pressure
        score *= max(0.1, budget_factor)

        # Scene pressure reduces score
        if self._pending_decisions > 1 and cfg.scene_pressure_divisor > 0:
            score /= (1 + self._pending_decisions / cfg.scene_pressure_divisor)

        # Route based on threshold
        if score >= cfg.llm_threshold:
            decision = RouteDecision(
                Route.LLM,
                f"auto: score {score:.2f} >= {cfg.llm_threshold}",
                auto_score=score,
                npc_id=npc.npc_id, decision_type=decision_type,
            )
        else:
            decision = RouteDecision(
                Route.DETERMINISTIC,
                f"auto: score {score:.2f} < {cfg.llm_threshold}",
                auto_score=score,
                npc_id=npc.npc_id, decision_type=decision_type,
            )

        self._record(decision)
        return decision

    # ---------- Budget passthrough ----------

    def record_llm_spend(self, tokens: int, purpose: str = "general") -> None:
        """Record tokens spent after an LLM call completes."""
        self.budget.record_spend(tokens, purpose)

    def begin_llm_call(self) -> None:
        self.budget.begin_call()

    def end_llm_call(self) -> None:
        self.budget.end_call()

    # ---------- Runtime configuration ----------

    def set_policy(self, policy: CognitionPolicy) -> None:
        """Hot-swap the routing policy."""
        self.policy = policy
        # Rebuild budget if limits changed
        self.budget = TokenBudget(
            daily_limit=policy.token_budget_daily,
            reserve_fraction=policy.reserve_fraction,
            max_calls_per_minute=policy.max_calls_per_minute,
            max_concurrent=policy.max_concurrent_llm_calls,
        )
        logger.info("Cognition policy updated")

    def set_importance_scorer(self, scorer: ImportanceScorer) -> None:
        """Hot-swap the importance scoring function."""
        self.importance_scorer = scorer

    def add_priority_npc(self, npc_id: str) -> None:
        self.policy.priority_npcs.add(npc_id)

    def remove_priority_npc(self, npc_id: str) -> None:
        self.policy.priority_npcs.discard(npc_id)

    def set_route(self, decision_type: str, mode: str) -> None:
        """Change routing for a single decision type at runtime."""
        self.policy.set_mode(decision_type, mode)

    # ---------- Statistics ----------

    def _record(self, decision: RouteDecision) -> None:
        """Track routing statistics."""
        if decision.route == Route.LLM:
            self._stats_llm += 1
        else:
            self._stats_deterministic += 1

        key = decision.decision_type
        if key not in self._stats_by_type:
            self._stats_by_type[key] = {"llm": 0, "deterministic": 0}
        self._stats_by_type[key][decision.route.value] += 1

    def get_stats(self) -> dict[str, Any]:
        total = self._stats_llm + self._stats_deterministic
        return {
            "total_decisions": total,
            "llm_decisions": self._stats_llm,
            "deterministic_decisions": self._stats_deterministic,
            "llm_ratio": (
                round(self._stats_llm / total, 3) if total > 0 else 0.0
            ),
            "by_type": dict(self._stats_by_type),
            "budget": self.budget.get_stats(),
            "policy": self.policy.to_dict(),
        }


__all__ = [
    "CognitionRouter",
    "Route",
    "RouteDecision",
    "TokenBudget",
    "BudgetSnapshot",
    "CognitionPolicy",
    "AutoConfig",
    "default_importance_scorer",
    "ROUTE_LLM",
    "ROUTE_DETERMINISTIC",
    "ROUTE_AUTO",
    "policy_all_llm",
    "policy_all_deterministic",
    "policy_conversations_only",
    "policy_local_llm",
]
