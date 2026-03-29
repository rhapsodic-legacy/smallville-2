"""
Cognition policy — user-configurable routing rules.

The policy defines HOW the router decides between LLM and
deterministic for each decision type. It is the primary
configuration surface for users and the AI Game Studio.

Three routing modes per decision type:
- "llm":           Always use LLM (burns tokens)
- "deterministic": Always use planner (free)
- "auto":          Router decides based on budget + importance + scene

Policies are plain data — serialisable, diffable, and swappable
at runtime without restarting the simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Valid routing modes
ROUTE_LLM = "llm"
ROUTE_DETERMINISTIC = "deterministic"
ROUTE_AUTO = "auto"
VALID_MODES = {ROUTE_LLM, ROUTE_DETERMINISTIC, ROUTE_AUTO}

# Known decision types (extensible — unknown types default to policy fallback)
DECISION_TYPES = {
    "daily_schedule",
    "reaction",
    "conversation",
    "trade_evaluation",
    "craft_choice",
    "flee",
    "reflection",
    "gather_choice",
    "work_choice",
}


@dataclass
class AutoConfig:
    """
    Configuration for "auto" routing mode.

    Controls the threshold and multipliers the router uses when
    deciding whether a specific decision is worth an LLM call.
    """
    # Score threshold: decisions scoring above this use LLM
    llm_threshold: float = 0.5

    # Multipliers applied to the auto-routing score
    novelty_weight: float = 1.0      # novel situations boost LLM chance
    proximity_weight: float = 1.0    # near-player decisions boost LLM
    importance_weight: float = 1.0   # high-importance decisions boost LLM

    # Scene pressure: when many NPCs need decisions, downgrade to deterministic
    scene_pressure_divisor: float = 1.0  # score /= (1 + pending / divisor)


@dataclass
class CognitionPolicy:
    """
    Full routing policy — the user's configuration for the cognition router.

    Attributes:
        routing:        Per-decision-type routing mode.
        default_mode:   Fallback for decision types not in routing dict.
        auto_config:    Settings for "auto" routing decisions.
        token_budget_daily: Daily token limit (0 = unlimited).
        reserve_fraction:   Fraction of budget reserved for priority decisions.
        auto_downgrade_threshold: If more than this many NPCs need decisions
                                  in one tick, force all "auto" to deterministic.
        priority_npcs:  NPC IDs that always get LLM (story-critical characters).
        priority_decisions: Decision types that count as "priority" for budget reserve.
        max_concurrent_llm_calls: Throughput cap (for local LLMs).
        max_calls_per_minute: Rate limit for API providers.
    """
    routing: dict[str, str] = field(default_factory=lambda: {
        "daily_schedule": ROUTE_AUTO,
        "reaction": ROUTE_AUTO,
        "conversation": ROUTE_LLM,
        "trade_evaluation": ROUTE_DETERMINISTIC,
        "craft_choice": ROUTE_DETERMINISTIC,
        "flee": ROUTE_DETERMINISTIC,
        "reflection": ROUTE_AUTO,
        "gather_choice": ROUTE_DETERMINISTIC,
        "work_choice": ROUTE_DETERMINISTIC,
    })

    default_mode: str = ROUTE_AUTO
    auto_config: AutoConfig = field(default_factory=AutoConfig)

    # Budget
    token_budget_daily: int = 500_000
    reserve_fraction: float = 0.2

    # Scene pressure
    auto_downgrade_threshold: int = 15

    # Priority overrides
    priority_npcs: set[str] = field(default_factory=set)
    priority_decisions: set[str] = field(default_factory=lambda: {
        "conversation", "reflection",
    })

    # Throughput
    max_concurrent_llm_calls: int = 10
    max_calls_per_minute: int = 50

    def get_mode(self, decision_type: str) -> str:
        """Get the routing mode for a decision type."""
        return self.routing.get(decision_type, self.default_mode)

    def set_mode(self, decision_type: str, mode: str) -> None:
        """Set the routing mode for a decision type."""
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {VALID_MODES}"
            )
        self.routing[decision_type] = mode

    def is_priority_npc(self, npc_id: str) -> bool:
        return npc_id in self.priority_npcs

    def is_priority_decision(self, decision_type: str) -> bool:
        return decision_type in self.priority_decisions

    # ---------- Serialisation ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing": dict(self.routing),
            "default_mode": self.default_mode,
            "auto_config": {
                "llm_threshold": self.auto_config.llm_threshold,
                "novelty_weight": self.auto_config.novelty_weight,
                "proximity_weight": self.auto_config.proximity_weight,
                "importance_weight": self.auto_config.importance_weight,
                "scene_pressure_divisor": self.auto_config.scene_pressure_divisor,
            },
            "token_budget_daily": self.token_budget_daily,
            "reserve_fraction": self.reserve_fraction,
            "auto_downgrade_threshold": self.auto_downgrade_threshold,
            "priority_npcs": list(self.priority_npcs),
            "priority_decisions": list(self.priority_decisions),
            "max_concurrent_llm_calls": self.max_concurrent_llm_calls,
            "max_calls_per_minute": self.max_calls_per_minute,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CognitionPolicy:
        """Reconstruct a policy from a serialised dict."""
        auto_data = data.get("auto_config", {})
        return cls(
            routing=data.get("routing", {}),
            default_mode=data.get("default_mode", ROUTE_AUTO),
            auto_config=AutoConfig(
                llm_threshold=auto_data.get("llm_threshold", 0.5),
                novelty_weight=auto_data.get("novelty_weight", 1.0),
                proximity_weight=auto_data.get("proximity_weight", 1.0),
                importance_weight=auto_data.get("importance_weight", 1.0),
                scene_pressure_divisor=auto_data.get(
                    "scene_pressure_divisor", 1.0,
                ),
            ),
            token_budget_daily=data.get("token_budget_daily", 500_000),
            reserve_fraction=data.get("reserve_fraction", 0.2),
            auto_downgrade_threshold=data.get("auto_downgrade_threshold", 15),
            priority_npcs=set(data.get("priority_npcs", [])),
            priority_decisions=set(data.get("priority_decisions", [
                "conversation", "reflection",
            ])),
            max_concurrent_llm_calls=data.get("max_concurrent_llm_calls", 10),
            max_calls_per_minute=data.get("max_calls_per_minute", 50),
        )


# ---------- Preset policies ----------

def policy_all_llm() -> CognitionPolicy:
    """Every decision uses LLM. Maximum quality, maximum cost."""
    policy = CognitionPolicy()
    for key in policy.routing:
        policy.routing[key] = ROUTE_LLM
    policy.default_mode = ROUTE_LLM
    return policy


def policy_all_deterministic() -> CognitionPolicy:
    """Every decision is deterministic. Zero cost, no LLM needed."""
    policy = CognitionPolicy()
    for key in policy.routing:
        policy.routing[key] = ROUTE_DETERMINISTIC
    policy.default_mode = ROUTE_DETERMINISTIC
    policy.token_budget_daily = 0
    return policy


def policy_conversations_only() -> CognitionPolicy:
    """Only conversations use LLM. Good balance for limited budgets."""
    policy = CognitionPolicy()
    for key in policy.routing:
        policy.routing[key] = ROUTE_DETERMINISTIC
    policy.routing["conversation"] = ROUTE_LLM
    policy.default_mode = ROUTE_DETERMINISTIC
    return policy


def policy_local_llm(max_concurrent: int = 2) -> CognitionPolicy:
    """Optimised for local LLMs with limited throughput."""
    policy = CognitionPolicy()
    policy.token_budget_daily = 0  # unlimited tokens
    policy.max_concurrent_llm_calls = max_concurrent
    policy.max_calls_per_minute = max_concurrent * 10
    policy.auto_downgrade_threshold = 5
    return policy
