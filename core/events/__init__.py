"""Events module — event impact system with hard-coded, conditional, and boolean triggers."""

from core.events.impact import (
    EventImpactSystem,
    EventRule,
    EventEffect,
    EventCondition,
    GameEvent,
    DEFAULT_RULES,
)

__all__ = [
    "EventImpactSystem",
    "EventRule",
    "EventEffect",
    "EventCondition",
    "GameEvent",
    "DEFAULT_RULES",
]
