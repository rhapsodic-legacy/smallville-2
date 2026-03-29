"""
Event impact system — data-driven rules mapping events to effects.

Three impact modes:
  1. Hard-coded: deterministic effects (trade → +5 trust)
  2. Conditional: effects gated by state checks (proposal IF would_accept)
  3. Boolean/narrative: set world or NPC flags (war = True)

Rules are data, not code — configurable for the AI Game Studio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.relationships.sentiment import SentimentTracker

logger = logging.getLogger(__name__)


@dataclass
class EventEffect:
    """A single effect produced by an event rule."""
    effect_type: str         # "modify_sentiment", "modify_global", "set_flag"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.effect_type, **self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventEffect:
        effect_type = data.pop("type", data.pop("effect_type", "unknown"))
        return cls(effect_type=effect_type, params=data)


@dataclass
class EventCondition:
    """A condition that must be met for effects to apply."""
    check_type: str          # "sentiment_above", "sentiment_below", "flag_set", "flag_not_set"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.check_type, **self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventCondition:
        check_type = data.pop("type", data.pop("check_type", "always"))
        return cls(check_type=check_type, params=data)


@dataclass
class EventRule:
    """
    A rule mapping an event type to effects.

    Scope: "individual" (affects participants), "world" (affects all NPCs).
    """
    event_type: str
    effects: list[EventEffect] = field(default_factory=list)
    conditions: list[EventCondition] = field(default_factory=list)
    scope: str = "individual"   # "individual" or "world"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "effects": [e.to_dict() for e in self.effects],
            "conditions": [c.to_dict() for c in self.conditions],
            "scope": self.scope,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventRule:
        return cls(
            event_type=data["event_type"],
            effects=[EventEffect.from_dict(dict(e)) for e in data.get("effects", [])],
            conditions=[EventCondition.from_dict(dict(c)) for c in data.get("conditions", [])],
            scope=data.get("scope", "individual"),
            description=data.get("description", ""),
        )


@dataclass
class GameEvent:
    """An event occurrence to be processed by the impact system."""
    event_type: str
    participants: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    game_time: float = 0.0
    location_x: int = 0
    location_z: int = 0


# Built-in rules for common events
DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "event_type": "conversation",
        "effects": [
            {"type": "modify_sentiment", "dimension": "trust", "delta": 2},
            {"type": "modify_sentiment", "dimension": "affection", "delta": 1},
        ],
        "scope": "individual",
        "description": "Casual conversations build mild trust and affection.",
    },
    {
        "event_type": "trade_completed",
        "effects": [
            {"type": "modify_sentiment", "dimension": "trust", "delta": 5},
            {"type": "modify_sentiment", "dimension": "respect", "delta": 3},
        ],
        "scope": "individual",
        "description": "Successful trade builds trust and mutual respect.",
    },
    {
        "event_type": "trade_refused",
        "effects": [
            {"type": "modify_sentiment", "dimension": "trust", "delta": -3},
            {"type": "modify_sentiment", "dimension": "respect", "delta": -2},
        ],
        "scope": "individual",
        "description": "Refused trade erodes trust slightly.",
    },
    {
        "event_type": "gift_given",
        "effects": [
            {"type": "modify_sentiment", "dimension": "affection", "delta": 10},
            {"type": "modify_sentiment", "dimension": "debt", "delta": 5},
        ],
        "scope": "individual",
        "description": "Receiving a gift increases affection and sense of debt.",
    },
    {
        "event_type": "insult",
        "effects": [
            {"type": "modify_sentiment", "dimension": "affection", "delta": -15},
            {"type": "modify_sentiment", "dimension": "respect", "delta": -10},
        ],
        "scope": "individual",
        "description": "Insults damage affection and respect.",
    },
    {
        "event_type": "helped_in_need",
        "effects": [
            {"type": "modify_sentiment", "dimension": "trust", "delta": 15},
            {"type": "modify_sentiment", "dimension": "affection", "delta": 10},
            {"type": "modify_sentiment", "dimension": "debt", "delta": 10},
        ],
        "scope": "individual",
        "description": "Help during a crisis creates strong bonds.",
    },
    {
        "event_type": "betrayal",
        "effects": [
            {"type": "modify_sentiment", "dimension": "trust", "delta": -40},
            {"type": "modify_sentiment", "dimension": "fear", "delta": 10},
            {"type": "modify_sentiment", "dimension": "affection", "delta": -30},
        ],
        "scope": "individual",
        "description": "Betrayal devastates trust and affection.",
    },
    {
        "event_type": "threat",
        "effects": [
            {"type": "modify_sentiment", "dimension": "fear", "delta": 20},
            {"type": "modify_sentiment", "dimension": "trust", "delta": -10},
        ],
        "scope": "individual",
        "description": "Threats increase fear and reduce trust.",
    },
    {
        "event_type": "proposal_accepted",
        "effects": [
            {"type": "modify_sentiment", "dimension": "affection", "delta": 50},
            {"type": "modify_sentiment", "dimension": "trust", "delta": 30},
            {"type": "set_flag", "flag": "engaged", "value": True},
        ],
        "conditions": [
            {"type": "sentiment_above", "dimension": "affection", "threshold": 30},
        ],
        "scope": "individual",
        "description": "Accepted proposal requires mutual affection.",
    },
    {
        "event_type": "war_declared",
        "effects": [
            {"type": "modify_global", "param": "aggression_modifier", "delta": 30},
            {"type": "set_flag", "flag": "war", "value": True},
        ],
        "scope": "world",
        "description": "War increases global aggression and sets war flag.",
    },
    {
        "event_type": "festival",
        "effects": [
            {"type": "modify_global", "param": "morale_modifier", "delta": 10},
            {"type": "set_flag", "flag": "festival_active", "value": True},
        ],
        "scope": "world",
        "description": "Festivals boost morale across the population.",
    },
    {
        "event_type": "gathering_complete",
        "effects": [
            {"type": "set_npc_flag", "flag": "last_gathered", "value": True},
        ],
        "scope": "individual",
        "description": "Resource gathering completed — records activity for memory.",
    },
    {
        "event_type": "construction_complete",
        "effects": [
            {"type": "modify_global", "param": "morale_modifier", "delta": 5},
            {"type": "set_npc_flag", "flag": "built_something", "value": True},
        ],
        "scope": "individual",
        "description": "Building completed — contributors gain pride, town morale rises.",
    },
]


class EventImpactSystem:
    """
    Data-driven event processing engine.

    Matches incoming events against rules, checks conditions,
    and applies effects to the sentiment tracker and world state.
    """

    def __init__(
        self,
        sentiment_tracker: SentimentTracker | None = None,
    ):
        self.sentiment = sentiment_tracker
        self._rules: dict[str, list[EventRule]] = {}
        self._world_flags: dict[str, Any] = {}
        self._world_params: dict[str, float] = {}
        self._npc_flags: dict[str, dict[str, Any]] = {}
        # Custom condition evaluators (extensible for AI Game Studio)
        self._custom_checks: dict[str, Callable] = {}

    def initialise(self, rules: list[dict[str, Any]] | None = None) -> None:
        """Load event rules. Uses defaults if none provided."""
        rule_defs = rules if rules is not None else DEFAULT_RULES
        self._rules.clear()
        for rule_data in rule_defs:
            rule = EventRule.from_dict(dict(rule_data))
            self._rules.setdefault(rule.event_type, []).append(rule)
        logger.info(
            "Event impact system initialised with %d rules across %d event types",
            sum(len(v) for v in self._rules.values()),
            len(self._rules),
        )

    def add_rule(self, rule: EventRule | dict[str, Any]) -> None:
        """Add a rule at runtime (for AI Game Studio injection)."""
        if isinstance(rule, dict):
            rule = EventRule.from_dict(rule)
        self._rules.setdefault(rule.event_type, []).append(rule)

    def process_event(self, event: GameEvent) -> list[dict[str, Any]]:
        """
        Process a game event through the rules engine.

        Returns a list of applied effects for logging/UI.
        """
        rules = self._rules.get(event.event_type, [])
        if not rules:
            return []

        applied: list[dict[str, Any]] = []

        for rule in rules:
            # Check conditions
            if not self._check_conditions(rule, event):
                continue

            # Apply effects
            for effect in rule.effects:
                result = self._apply_effect(effect, event, rule.scope)
                if result:
                    applied.append(result)

        if applied:
            logger.debug(
                "Event '%s' triggered %d effects",
                event.event_type, len(applied),
            )
        return applied

    def _check_conditions(self, rule: EventRule, event: GameEvent) -> bool:
        """Evaluate all conditions for a rule. All must pass."""
        for cond in rule.conditions:
            if not self._evaluate_condition(cond, event):
                return False
        return True

    def _evaluate_condition(
        self, condition: EventCondition, event: GameEvent,
    ) -> bool:
        """Evaluate a single condition."""
        ct = condition.check_type

        if ct == "always":
            return True

        if ct == "sentiment_above" and self.sentiment:
            if len(event.participants) < 2:
                return False
            dim = condition.params.get("dimension", "trust")
            threshold = condition.params.get("threshold", 0)
            sent = self.sentiment.get(event.participants[0], event.participants[1])
            return sent.get(dim) >= threshold

        if ct == "sentiment_below" and self.sentiment:
            if len(event.participants) < 2:
                return False
            dim = condition.params.get("dimension", "trust")
            threshold = condition.params.get("threshold", 0)
            sent = self.sentiment.get(event.participants[0], event.participants[1])
            return sent.get(dim) < threshold

        if ct == "flag_set":
            flag = condition.params.get("flag", "")
            return self._world_flags.get(flag, False) is True

        if ct == "flag_not_set":
            flag = condition.params.get("flag", "")
            return self._world_flags.get(flag, False) is not True

        if ct == "npc_flag_set":
            flag = condition.params.get("flag", "")
            if not event.participants:
                return False
            npc_id = event.participants[0]
            return self._npc_flags.get(npc_id, {}).get(flag, False) is True

        # Check custom evaluators
        if ct in self._custom_checks:
            return self._custom_checks[ct](condition, event)

        logger.warning("Unknown condition type: %s", ct)
        return True  # unknown conditions pass by default

    def _apply_effect(
        self,
        effect: EventEffect,
        event: GameEvent,
        scope: str,
    ) -> dict[str, Any] | None:
        """Apply a single effect and return a log entry."""
        et = effect.effect_type

        if et == "modify_sentiment" and self.sentiment:
            if len(event.participants) < 2:
                return None
            dim = effect.params.get("dimension", "trust")
            delta = effect.params.get("delta", 0)
            mutual = effect.params.get("mutual", True)

            if mutual:
                self.sentiment.modify_mutual(
                    event.participants[0], event.participants[1],
                    dim, delta, event.game_time,
                )
            else:
                self.sentiment.modify(
                    event.participants[0], event.participants[1],
                    dim, delta, event.game_time,
                )
            return {
                "effect": "modify_sentiment",
                "dimension": dim,
                "delta": delta,
                "participants": event.participants[:2],
                "mutual": mutual,
            }

        if et == "modify_global":
            param = effect.params.get("param", "")
            delta = effect.params.get("delta", 0)
            current = self._world_params.get(param, 0.0)
            self._world_params[param] = current + delta
            return {
                "effect": "modify_global",
                "param": param,
                "new_value": self._world_params[param],
            }

        if et == "set_flag":
            flag = effect.params.get("flag", "")
            value = effect.params.get("value", True)

            if scope == "world":
                self._world_flags[flag] = value
                return {"effect": "set_flag", "scope": "world",
                        "flag": flag, "value": value}
            else:
                # Set flag on each participant
                for pid in event.participants:
                    self._npc_flags.setdefault(pid, {})[flag] = value
                return {"effect": "set_flag", "scope": "individual",
                        "flag": flag, "value": value,
                        "participants": event.participants}

        if et == "set_npc_flag":
            flag = effect.params.get("flag", "")
            value = effect.params.get("value", True)
            for pid in event.participants:
                self._npc_flags.setdefault(pid, {})[flag] = value
            return {"effect": "set_npc_flag", "flag": flag, "value": value,
                    "participants": event.participants}

        logger.warning("Unknown effect type: %s", et)
        return None

    # ---------- World state queries ----------

    def get_world_flag(self, flag: str, default: Any = None) -> Any:
        return self._world_flags.get(flag, default)

    def set_world_flag(self, flag: str, value: Any) -> None:
        self._world_flags[flag] = value

    def get_world_param(self, param: str, default: float = 0.0) -> float:
        return self._world_params.get(param, default)

    def get_npc_flag(
        self, npc_id: str, flag: str, default: Any = None,
    ) -> Any:
        return self._npc_flags.get(npc_id, {}).get(flag, default)

    def get_npc_flags(self, npc_id: str) -> dict[str, Any]:
        return dict(self._npc_flags.get(npc_id, {}))

    # ---------- Inspection ----------

    def get_rules(self) -> list[dict[str, Any]]:
        """All loaded rules (for API / inspector)."""
        result = []
        for rules_list in self._rules.values():
            for rule in rules_list:
                result.append(rule.to_dict())
        return result

    def get_world_state(self) -> dict[str, Any]:
        return {
            "flags": dict(self._world_flags),
            "params": dict(self._world_params),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "rule_count": sum(len(v) for v in self._rules.values()),
            "event_types": list(self._rules.keys()),
            "world_flags": len(self._world_flags),
            "npc_flags": sum(len(v) for v in self._npc_flags.values()),
        }
