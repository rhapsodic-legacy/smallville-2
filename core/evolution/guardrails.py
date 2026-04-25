"""
Behavioural guardrails — prevent degenerate strategies and maintain
narrative consistency.

Three layers:
  1. Hard limits: absolute bounds on NPC parameters (health, gold, etc.)
  2. Rate limiters: cap how fast interventions can change things
  3. Narrative rules: pluggable rule system for world-specific constraints

All rules are modular — the AI Game Studio can register custom rules
without modifying this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.evolution.overseer import Intervention

logger = logging.getLogger(__name__)


# ---------- Rule results ----------

@dataclass
class RuleViolation:
    """A single guardrail violation."""
    rule_name: str
    severity: str  # "block", "warn", "adjust"
    message: str
    suggested_fix: dict[str, Any] | None = None


@dataclass
class GuardrailResult:
    """Result of checking an intervention or NPC state against guardrails."""
    allowed: bool
    violations: list[RuleViolation] = field(default_factory=list)
    adjusted_parameters: dict[str, Any] | None = None

    def add_violation(self, violation: RuleViolation) -> None:
        self.violations.append(violation)
        if violation.severity == "block":
            self.allowed = False


# ---------- Hard limits ----------

# Absolute parameter bounds for NPCs
NPC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "health": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "hunger": (0.0, 1.0),
    "gold": (0.0, 10_000.0),
    "conversation_cooldown": (10.0, 300.0),
}

# Maximum interventions per evaluation cycle
MAX_INTERVENTIONS_PER_CYCLE = 5

# Maximum active policies per NPC
MAX_POLICIES_PER_NPC = 3

# Maximum active prompt modifiers per NPC
MAX_MODIFIERS_PER_NPC = 5


def check_param_bounds(npc: NPC) -> list[RuleViolation]:
    """Check that NPC parameters are within hard limits."""
    violations: list[RuleViolation] = []

    for param, (lo, hi) in NPC_PARAM_BOUNDS.items():
        if not hasattr(npc, param):
            continue
        val = getattr(npc, param)
        if not isinstance(val, (int, float)):
            continue
        if val < lo:
            violations.append(RuleViolation(
                rule_name=f"param_below_min_{param}",
                severity="adjust",
                message=f"{npc.name}.{param}={val:.2f} below minimum {lo}",
                suggested_fix={param: lo},
            ))
        elif val > hi:
            violations.append(RuleViolation(
                rule_name=f"param_above_max_{param}",
                severity="adjust",
                message=f"{npc.name}.{param}={val:.2f} above maximum {hi}",
                suggested_fix={param: hi},
            ))

    return violations


def clamp_npc_params(npc: NPC) -> int:
    """Clamp NPC parameters to hard limits. Returns count of clamped values."""
    clamped = 0
    for param, (lo, hi) in NPC_PARAM_BOUNDS.items():
        if not hasattr(npc, param):
            continue
        val = getattr(npc, param)
        if not isinstance(val, (int, float)):
            continue
        if val < lo:
            setattr(npc, param, lo)
            clamped += 1
        elif val > hi:
            setattr(npc, param, hi)
            clamped += 1
    return clamped


# ---------- Rate limiters ----------

@dataclass
class RateLimiter:
    """Tracks intervention frequency to prevent rapid-fire changes."""
    max_per_day: int = 10
    _counts: dict[int, int] = field(default_factory=dict)

    def check(self, game_day: int) -> bool:
        """Returns True if another intervention is allowed today."""
        return self._counts.get(game_day, 0) < self.max_per_day

    def record(self, game_day: int) -> None:
        """Record that an intervention was applied."""
        self._counts[game_day] = self._counts.get(game_day, 0) + 1
        # Prune old days (keep last 7)
        old_days = [d for d in self._counts if d < game_day - 7]
        for d in old_days:
            del self._counts[d]

    def get_remaining(self, game_day: int) -> int:
        """How many interventions remain for today."""
        return max(0, self.max_per_day - self._counts.get(game_day, 0))


# ---------- Narrative rules (pluggable) ----------

# Rule function signature: (intervention, npcs) -> RuleViolation | None
NarrativeRule = Callable[["Intervention", list["NPC"]], RuleViolation | None]


def _rule_no_mass_policy_spam(
    intervention: Intervention, npcs: list[NPC],
) -> RuleViolation | None:
    """Block population-wide policy injection if population is small."""
    if (
        intervention.intervention_type == "policy_inject"
        and intervention.target == "population"
        and len(npcs) <= 2
    ):
        return RuleViolation(
            rule_name="no_mass_policy_small_pop",
            severity="block",
            message="Population-wide policy blocked: only "
                    f"{len(npcs)} NPCs (minimum 3 required)",
        )
    return None


def _rule_no_conflicting_policies(
    intervention: Intervention, npcs: list[NPC],
) -> RuleViolation | None:
    """Warn if injecting hermit policy on an NPC that also has politician."""
    if intervention.intervention_type != "policy_inject":
        return None

    conflicts = {
        "hermit": "politician",
        "politician": "hermit",
        "survival_mode": "merchant",
    }
    conflict = conflicts.get(intervention.action)
    if conflict is None:
        return None

    return RuleViolation(
        rule_name="conflicting_policy_warning",
        severity="warn",
        message=f"Policy '{intervention.action}' may conflict with "
                f"'{conflict}' if both active on same NPC",
    )


def _rule_no_negative_cooldown(
    intervention: Intervention, npcs: list[NPC],
) -> RuleViolation | None:
    """Block parameter tunes that would push conversation cooldown negative."""
    if intervention.intervention_type != "parameter_tune":
        return None
    boost = intervention.parameters.get("conversation_chance_boost", 0)
    if boost > 0.5:
        return RuleViolation(
            rule_name="excessive_conversation_boost",
            severity="adjust",
            message=f"Conversation boost {boost} too large (max 0.5)",
            suggested_fix={"conversation_chance_boost": 0.5},
        )
    return None


# Built-in narrative rules
_BUILTIN_RULES: list[NarrativeRule] = [
    _rule_no_mass_policy_spam,
    _rule_no_conflicting_policies,
    _rule_no_negative_cooldown,
]


# ---------- Guardrail engine ----------

class GuardrailEngine:
    """Central guardrail system that checks interventions and NPC state.

    Supports pluggable narrative rules — consumers can register custom
    rules via add_rule() without modifying the engine.
    """

    def __init__(self) -> None:
        self._rules: list[NarrativeRule] = list(_BUILTIN_RULES)
        self._rate_limiter = RateLimiter()
        self._violation_log: list[dict] = []

    def add_rule(self, rule: NarrativeRule) -> None:
        """Register a custom narrative rule."""
        self._rules.append(rule)

    def remove_rule(self, rule: NarrativeRule) -> bool:
        """Remove a previously registered rule. Returns True if found."""
        try:
            self._rules.remove(rule)
            return True
        except ValueError:
            return False

    def check_intervention(
        self,
        intervention: Intervention,
        npcs: list[NPC],
        game_day: int,
    ) -> GuardrailResult:
        """Check whether an intervention is allowed.

        Runs rate limiting, then all narrative rules.
        Returns a GuardrailResult indicating whether to proceed.
        """
        result = GuardrailResult(allowed=True)

        # Rate limit check
        if not self._rate_limiter.check(game_day):
            result.add_violation(RuleViolation(
                rule_name="rate_limit_exceeded",
                severity="block",
                message=f"Rate limit reached for day {game_day} "
                        f"({self._rate_limiter.max_per_day} max)",
            ))

        # Run all narrative rules
        for rule in self._rules:
            try:
                violation = rule(intervention, npcs)
                if violation is not None:
                    result.add_violation(violation)
            except Exception as e:
                logger.warning("Guardrail rule %s failed: %s", rule.__name__, e)

        # Log violations
        for v in result.violations:
            self._violation_log.append({
                "rule": v.rule_name,
                "severity": v.severity,
                "message": v.message,
                "day": game_day,
            })

        return result

    def record_applied(self, game_day: int) -> None:
        """Record that an intervention was successfully applied."""
        self._rate_limiter.record(game_day)

    def check_population_health(self, npcs: list[NPC]) -> list[RuleViolation]:
        """Check all NPCs for parameter bound violations."""
        all_violations: list[RuleViolation] = []
        for npc in npcs:
            all_violations.extend(check_param_bounds(npc))
        return all_violations

    def enforce_bounds(self, npcs: list[NPC]) -> int:
        """Clamp all NPC parameters to hard limits. Returns total clamped."""
        total = 0
        for npc in npcs:
            total += clamp_npc_params(npc)
        if total:
            logger.debug("Clamped %d NPC parameters to bounds", total)
        return total

    def filter_interventions(
        self,
        interventions: list[Intervention],
        npcs: list[NPC],
        game_day: int,
    ) -> list[Intervention]:
        """Filter a list of interventions, removing blocked ones.

        Also enforces MAX_INTERVENTIONS_PER_CYCLE cap.
        Adjusts parameters where suggested fixes exist.
        """
        allowed: list[Intervention] = []

        for intervention in interventions[:MAX_INTERVENTIONS_PER_CYCLE]:
            result = self.check_intervention(intervention, npcs, game_day)

            if not result.allowed:
                logger.info(
                    "Guardrail blocked intervention: %s — %s",
                    intervention.action,
                    "; ".join(v.message for v in result.violations),
                )
                continue

            # Apply suggested parameter adjustments from warn/adjust violations
            for v in result.violations:
                if v.severity == "adjust" and v.suggested_fix:
                    intervention.parameters.update(v.suggested_fix)

            allowed.append(intervention)

        dropped = len(interventions) - len(allowed)
        if dropped > 0:
            logger.info(
                "Guardrails: %d of %d interventions filtered",
                dropped, len(interventions),
            )

        return allowed

    def get_violation_log(self, limit: int = 50) -> list[dict]:
        """Return recent violations for debugging."""
        return self._violation_log[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "total_rules": len(self._rules),
            "total_violations": len(self._violation_log),
            "rate_limiter_max_per_day": self._rate_limiter.max_per_day,
        }
