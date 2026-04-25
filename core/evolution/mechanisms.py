"""
Evolution mechanisms — how the overseer's interventions are applied.

Three mechanism types:
  1. Parameter tuning: adjust NPC-level parameters (conversation chance,
     schedule variety, gathering priority, risk tolerance)
  2. Policy injection: inject behavioural policy templates that override
     or augment normal planning (merchant strategy, survival mode, etc.)
  3. Prompt modifiers: inject text into NPC LLM prompts to nudge behaviour
     without changing parameters (motivation boosts, warnings, directives)

All mechanisms are reversible — policies expire, parameters can be reset,
and prompt modifiers have TTLs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.evolution.overseer import Intervention

logger = logging.getLogger(__name__)


# ---------- Policy templates ----------

@dataclass
class PolicyTemplate:
    """A behavioural policy that modifies NPC decision-making."""
    name: str
    description: str
    parameter_overrides: dict[str, Any] = field(default_factory=dict)
    schedule_biases: dict[str, float] = field(default_factory=dict)
    duration_days: int = 3  # auto-expire after N game-days
    applied_day: int = 0

    def is_expired(self, current_day: int) -> bool:
        return (current_day - self.applied_day) >= self.duration_days

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameter_overrides": self.parameter_overrides,
            "schedule_biases": self.schedule_biases,
            "duration_days": self.duration_days,
            "applied_day": self.applied_day,
        }


# Built-in policy templates
POLICY_TEMPLATES: dict[str, PolicyTemplate] = {
    "merchant": PolicyTemplate(
        name="merchant",
        description="Focus on trading and gold accumulation",
        parameter_overrides={"trade_priority": 8, "gather_priority": 6},
        schedule_biases={"trade": 0.3, "work": 0.2, "socialise": 0.2},
    ),
    "hermit": PolicyTemplate(
        name="hermit",
        description="Withdraw from social activity, focus on self-sufficiency",
        parameter_overrides={"conversation_chance": -0.2, "gather_priority": 8},
        schedule_biases={"gather": 0.4, "work": 0.3, "rest": 0.2},
    ),
    "politician": PolicyTemplate(
        name="politician",
        description="Maximise social connections and influence",
        parameter_overrides={"conversation_chance": 0.3, "conversation_cooldown": -20},
        schedule_biases={"socialise": 0.4, "work": 0.1, "trade": 0.2},
    ),
    "survival_mode": PolicyTemplate(
        name="survival_mode",
        description="Emergency survival focus — eat, rest, gather essentials",
        parameter_overrides={"eat_priority": 9, "rest_priority": 8, "gather_priority": 7},
        schedule_biases={"eat": 0.3, "rest": 0.2, "gather": 0.3},
        duration_days=2,
    ),
}


# ---------- Prompt modifiers ----------

@dataclass
class PromptModifier:
    """Text injected into an NPC's LLM prompts to nudge behaviour."""
    text: str
    source: str = "overseer"  # who created it
    ttl_days: int = 2         # auto-expire
    applied_day: int = 0

    def is_expired(self, current_day: int) -> bool:
        return (current_day - self.applied_day) >= self.ttl_days

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "ttl_days": self.ttl_days,
            "applied_day": self.applied_day,
        }


# ---------- Mechanism engine ----------

class MechanismEngine:
    """Applies and manages overseer interventions on the NPC population.

    Tracks active policies and prompt modifiers per NPC. Provides
    get_active_modifiers() for the cognition system to query when
    building LLM prompts.
    """

    def __init__(self):
        # npc_id → list of active policies
        self._active_policies: dict[str, list[PolicyTemplate]] = {}
        # npc_id → list of active prompt modifiers
        self._prompt_modifiers: dict[str, list[PromptModifier]] = {}
        # Population-wide parameter adjustments (additive)
        self._population_params: dict[str, float] = {}
        # Log of applied interventions
        self._applied_log: list[dict] = []

    def apply_intervention(
        self,
        intervention: Intervention,
        npcs: list[NPC],
        current_day: int,
    ) -> bool:
        """Apply a single intervention. Returns True if applied."""
        itype = intervention.intervention_type

        if itype == "parameter_tune":
            return self._apply_parameter_tune(intervention, npcs)
        elif itype == "policy_inject":
            return self._apply_policy(intervention, npcs, current_day)
        elif itype == "prompt_modifier":
            return self._apply_prompt_modifier(intervention, npcs, current_day)
        else:
            logger.warning("Unknown intervention type: %s", itype)
            return False

    def _apply_parameter_tune(
        self, intervention: Intervention, npcs: list[NPC],
    ) -> bool:
        """Adjust NPC or population parameters."""
        action = intervention.action
        params = intervention.parameters

        if intervention.target == "population":
            # Store population-wide adjustment
            for key, value in params.items():
                self._population_params[key] = (
                    self._population_params.get(key, 0.0) + value
                )

            # Apply specific known actions
            if action == "boost_conversation_chance":
                boost = params.get("conversation_chance_boost", 0.1)
                for npc in npcs:
                    npc.conversation_cooldown = max(
                        20, npc.conversation_cooldown - boost * 100,
                    )

            elif action == "increase_schedule_variety":
                # Flag for the planner to add more variety next regeneration
                self._population_params["variety_boost"] = params.get(
                    "variety_boost", 0.2,
                )

            elif action == "boost_gathering_priority":
                self._population_params["gathering_priority_boost"] = params.get(
                    "gathering_priority_boost", 2,
                )

        else:
            # Target-specific — find the NPC
            target_npc = None
            for npc in npcs:
                if npc.name == intervention.target or npc.npc_id == intervention.target:
                    target_npc = npc
                    break
            if not target_npc:
                return False

            for key, value in params.items():
                if hasattr(target_npc, key):
                    current = getattr(target_npc, key)
                    if isinstance(current, (int, float)):
                        setattr(target_npc, key, current + value)

        self._applied_log.append({
            "type": "parameter_tune",
            "action": action,
            "target": intervention.target,
        })
        logger.info(
            "Applied parameter tune: %s on %s",
            action, intervention.target,
        )
        return True

    def _apply_policy(
        self, intervention: Intervention, npcs: list[NPC], current_day: int,
    ) -> bool:
        """Inject a policy template."""
        action = intervention.action
        template = POLICY_TEMPLATES.get(action)
        if template is None:
            logger.warning("Unknown policy template: %s", action)
            return False

        # Clone the template with the current day
        policy = PolicyTemplate(
            name=template.name,
            description=template.description,
            parameter_overrides=dict(template.parameter_overrides),
            schedule_biases=dict(template.schedule_biases),
            duration_days=template.duration_days,
            applied_day=current_day,
        )

        if intervention.target == "population":
            for npc in npcs:
                self._active_policies.setdefault(npc.npc_id, []).append(policy)
        else:
            for npc in npcs:
                if npc.name == intervention.target or npc.npc_id == intervention.target:
                    self._active_policies.setdefault(npc.npc_id, []).append(policy)
                    break

        self._applied_log.append({
            "type": "policy_inject",
            "policy": action,
            "target": intervention.target,
        })
        logger.info("Injected policy '%s' on %s", action, intervention.target)
        return True

    def _apply_prompt_modifier(
        self, intervention: Intervention, npcs: list[NPC], current_day: int,
    ) -> bool:
        """Add a prompt modifier to an NPC."""
        modifier_text = intervention.parameters.get("modifier", intervention.action)
        modifier = PromptModifier(
            text=modifier_text,
            source="overseer",
            ttl_days=intervention.parameters.get("ttl_days", 2),
            applied_day=current_day,
        )

        if intervention.target == "population":
            for npc in npcs:
                self._prompt_modifiers.setdefault(npc.npc_id, []).append(modifier)
        else:
            for npc in npcs:
                if npc.name == intervention.target or npc.npc_id == intervention.target:
                    self._prompt_modifiers.setdefault(npc.npc_id, []).append(modifier)
                    break

        self._applied_log.append({
            "type": "prompt_modifier",
            "target": intervention.target,
            "text": modifier_text[:50],
        })
        logger.info(
            "Added prompt modifier for %s: %s",
            intervention.target, modifier_text[:50],
        )
        return True

    # ---------- Query API ----------

    def get_active_modifiers(self, npc_id: str) -> list[str]:
        """Get all active prompt modifier texts for an NPC.

        Called by the cognition system when building LLM prompts.
        """
        modifiers = self._prompt_modifiers.get(npc_id, [])
        return [m.text for m in modifiers if not m.is_expired(0)]

    def get_active_policies(self, npc_id: str) -> list[PolicyTemplate]:
        """Get all active policies for an NPC."""
        return [p for p in self._active_policies.get(npc_id, [])]

    def get_population_params(self) -> dict[str, float]:
        """Get current population-wide parameter adjustments."""
        return dict(self._population_params)

    # ---------- Maintenance ----------

    def expire_old(self, current_day: int) -> int:
        """Remove expired policies and modifiers. Returns count removed."""
        removed = 0

        for npc_id in list(self._active_policies):
            before = len(self._active_policies[npc_id])
            self._active_policies[npc_id] = [
                p for p in self._active_policies[npc_id]
                if not p.is_expired(current_day)
            ]
            removed += before - len(self._active_policies[npc_id])
            if not self._active_policies[npc_id]:
                del self._active_policies[npc_id]

        for npc_id in list(self._prompt_modifiers):
            before = len(self._prompt_modifiers[npc_id])
            self._prompt_modifiers[npc_id] = [
                m for m in self._prompt_modifiers[npc_id]
                if not m.is_expired(current_day)
            ]
            removed += before - len(self._prompt_modifiers[npc_id])
            if not self._prompt_modifiers[npc_id]:
                del self._prompt_modifiers[npc_id]

        if removed:
            logger.debug("Expired %d policies/modifiers on day %d", removed, current_day)
        return removed

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for the mechanism engine."""
        return {
            "active_policies": sum(
                len(v) for v in self._active_policies.values()
            ),
            "active_modifiers": sum(
                len(v) for v in self._prompt_modifiers.values()
            ),
            "population_params": dict(self._population_params),
            "total_applied": len(self._applied_log),
        }
