"""
Overseer agent — periodic evaluation and policy injection.

Runs on a configurable cycle (default: every game-day). Each cycle:
  1. Evaluate population fitness
  2. Detect stagnation, imbalance, or runaway behaviours
  3. Generate interventions (via LLM or heuristic)
  4. Apply interventions through the mechanisms module

Uses Claude Opus for strategy analysis (when LLM available),
falls back to rule-based heuristics otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from core.evolution.fitness import (
    FitnessConfig, FitnessScore, PopulationMetrics,
    evaluate_population, THEME_CONFIGS,
)

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.npc.llm_client import LLMProvider
    from core.memory.manager import MemoryManager
    from core.relationships.sentiment import SentimentTracker

logger = logging.getLogger(__name__)


# ---------- Intervention model ----------

@dataclass
class Intervention:
    """A specific action the overseer wants to take."""
    intervention_type: str  # "parameter_tune", "policy_inject", "prompt_modifier"
    target: str             # "population", npc_id, or faction_id
    action: str             # what to do
    reason: str             # why (for logging/debugging)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.intervention_type,
            "target": self.target,
            "action": self.action,
            "reason": self.reason,
            "parameters": self.parameters,
        }


@dataclass
class EvaluationReport:
    """Result of a single overseer evaluation cycle."""
    game_day: int
    metrics: PopulationMetrics
    scores: list[FitnessScore]
    triggers: list[str]
    interventions: list[Intervention]

    def to_dict(self) -> dict:
        return {
            "game_day": self.game_day,
            "metrics": self.metrics.to_dict(),
            "triggers": self.triggers,
            "interventions": [i.to_dict() for i in self.interventions],
            "npc_scores": [s.to_dict() for s in self.scores],
        }


# ---------- Trigger detection ----------

# Thresholds for heuristic trigger detection
STAGNATION_THRESHOLD = 0.02   # fitness change < 2% over eval window
IMBALANCE_THRESHOLD = 0.4     # max-min fitness gap > 0.4
LOW_SOCIAL_THRESHOLD = 0.25   # mean social score below this
LOW_PROSPERITY_THRESHOLD = 0.2
HIGH_STRUGGLE_RATIO = 0.4     # >40% of NPCs struggling


def detect_triggers(
    metrics: PopulationMetrics,
    previous_metrics: PopulationMetrics | None = None,
) -> list[str]:
    """Detect conditions that warrant intervention."""
    triggers: list[str] = []

    # Stagnation: fitness hasn't changed since last evaluation
    if previous_metrics is not None:
        delta = abs(metrics.mean_fitness - previous_metrics.mean_fitness)
        if delta < STAGNATION_THRESHOLD:
            triggers.append("stagnation")

    # Imbalance: large gap between best and worst NPC
    if metrics.population_size > 1:
        gap = metrics.max_fitness - metrics.min_fitness
        if gap > IMBALANCE_THRESHOLD:
            triggers.append("imbalance")

    # Low social cohesion
    if metrics.mean_social < LOW_SOCIAL_THRESHOLD:
        triggers.append("low_social")

    # Low prosperity
    if metrics.mean_prosperity < LOW_PROSPERITY_THRESHOLD:
        triggers.append("low_prosperity")

    # Too many struggling NPCs
    if metrics.population_size > 0:
        struggle_ratio = len(metrics.struggling_npcs) / metrics.population_size
        if struggle_ratio > HIGH_STRUGGLE_RATIO:
            triggers.append("mass_struggle")

    return triggers


# ---------- LLM strategy analysis ----------

STRATEGY_PROMPT = (
    "You are an overseer managing a medieval village simulation.\n\n"
    "Population: {population_size} NPCs\n"
    "Day: {game_day}\n\n"
    "Fitness metrics:\n"
    "  Mean fitness: {mean_fitness:.2f}\n"
    "  Survival: {mean_survival:.2f}\n"
    "  Prosperity: {mean_prosperity:.2f}\n"
    "  Social: {mean_social:.2f}\n"
    "  Goals: {mean_goals:.2f}\n"
    "  Engagement: {mean_engagement:.2f}\n\n"
    "Struggling NPCs: {struggling}\n"
    "Thriving NPCs: {thriving}\n\n"
    "Triggered conditions: {triggers}\n\n"
    "Suggest 1-3 specific interventions to improve the village.\n"
    "For each, provide:\n"
    "TYPE: parameter_tune | policy_inject | prompt_modifier\n"
    "TARGET: population | <npc_name>\n"
    "ACTION: <what to change>\n"
    "REASON: <why>\n"
)


async def _llm_strategy(
    metrics: PopulationMetrics,
    triggers: list[str],
    game_day: int,
    llm: LLMProvider,
) -> list[Intervention]:
    """Ask the LLM for strategic interventions."""
    try:
        prompt = STRATEGY_PROMPT.format(
            population_size=metrics.population_size,
            game_day=game_day,
            mean_fitness=metrics.mean_fitness,
            mean_survival=metrics.mean_survival,
            mean_prosperity=metrics.mean_prosperity,
            mean_social=metrics.mean_social,
            mean_goals=metrics.mean_goals,
            mean_engagement=metrics.mean_engagement,
            struggling=", ".join(metrics.struggling_npcs[:5]) or "none",
            thriving=", ".join(metrics.thriving_npcs[:5]) or "none",
            triggers=", ".join(triggers) or "none",
        )

        response = await llm.complete(
            system="You are an AI overseer for a village simulation.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.6,
            purpose="overseer",
        )

        return _parse_interventions(response)

    except Exception as e:
        logger.warning("LLM strategy analysis failed: %s", e)
        return []


def _parse_interventions(response: str) -> list[Intervention]:
    """Parse LLM response into Intervention objects."""
    interventions: list[Intervention] = []
    current: dict[str, str] = {}

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            if current.get("action"):
                interventions.append(Intervention(
                    intervention_type=current.get("type", "parameter_tune"),
                    target=current.get("target", "population"),
                    action=current.get("action", ""),
                    reason=current.get("reason", ""),
                ))
                current = {}
            continue

        upper = line.upper()
        if upper.startswith("TYPE:"):
            current["type"] = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("TARGET:"):
            current["target"] = line.split(":", 1)[1].strip()
        elif upper.startswith("ACTION:"):
            current["action"] = line.split(":", 1)[1].strip()
        elif upper.startswith("REASON:"):
            current["reason"] = line.split(":", 1)[1].strip()

    # Capture last intervention
    if current.get("action"):
        interventions.append(Intervention(
            intervention_type=current.get("type", "parameter_tune"),
            target=current.get("target", "population"),
            action=current.get("action", ""),
            reason=current.get("reason", ""),
        ))

    return interventions


# ---------- Heuristic interventions ----------

def _heuristic_interventions(
    triggers: list[str],
    metrics: PopulationMetrics,
    scores: list[FitnessScore],
) -> list[Intervention]:
    """Rule-based fallback when LLM is unavailable."""
    interventions: list[Intervention] = []

    if "stagnation" in triggers:
        interventions.append(Intervention(
            intervention_type="parameter_tune",
            target="population",
            action="increase_schedule_variety",
            reason="Population fitness stagnated — injecting schedule variation",
            parameters={"variety_boost": 0.2},
        ))

    if "low_social" in triggers:
        interventions.append(Intervention(
            intervention_type="parameter_tune",
            target="population",
            action="boost_conversation_chance",
            reason="Low social cohesion — increasing conversation probability",
            parameters={"conversation_chance_boost": 0.15},
        ))

    if "low_prosperity" in triggers:
        interventions.append(Intervention(
            intervention_type="parameter_tune",
            target="population",
            action="boost_gathering_priority",
            reason="Low prosperity — raising resource gathering priority",
            parameters={"gathering_priority_boost": 2},
        ))

    if "imbalance" in triggers:
        # Help the weakest NPCs
        for name in metrics.struggling_npcs[:3]:
            interventions.append(Intervention(
                intervention_type="prompt_modifier",
                target=name,
                action="add_motivation_boost",
                reason=f"{name} is struggling — injecting motivational prompt modifier",
                parameters={"modifier": "You feel a renewed sense of purpose."},
            ))

    if "mass_struggle" in triggers:
        interventions.append(Intervention(
            intervention_type="policy_inject",
            target="population",
            action="survival_mode",
            reason="Too many NPCs struggling — switching to survival focus",
            parameters={"fitness_config": "warzone"},
        ))

    return interventions


# ---------- Overseer class ----------

class Overseer:
    """
    Periodic population evaluator and policy injector.

    Call evaluate() once per game-day (or on demand). The overseer
    scores the population, detects issues, generates interventions,
    and returns an EvaluationReport. The caller (NPCManager) is
    responsible for actually applying the interventions.
    """

    def __init__(
        self,
        fitness_config: FitnessConfig | None = None,
        llm: LLMProvider | None = None,
        theme: str = "default",
    ):
        self.config = fitness_config or THEME_CONFIGS.get(theme, FitnessConfig())
        self.llm = llm
        self._previous_metrics: PopulationMetrics | None = None
        self._history: list[EvaluationReport] = []

    async def evaluate(
        self,
        npcs: list[NPC],
        game_day: int,
        sentiment: SentimentTracker | None = None,
        memory: MemoryManager | None = None,
    ) -> EvaluationReport:
        """Run a full evaluation cycle.

        1. Score all NPCs
        2. Detect triggers
        3. Generate interventions (LLM or heuristic)
        4. Store report in history
        """
        scores, metrics = evaluate_population(
            npcs, self.config, sentiment, memory,
        )

        triggers = detect_triggers(metrics, self._previous_metrics)

        # Generate interventions
        interventions: list[Intervention] = []
        if triggers:
            if self.llm is not None:
                interventions = await _llm_strategy(
                    metrics, triggers, game_day, self.llm,
                )
            # Always layer heuristic on top (LLM may miss obvious fixes)
            heuristic = _heuristic_interventions(triggers, metrics, scores)
            # Deduplicate by action
            seen_actions = {i.action for i in interventions}
            for h in heuristic:
                if h.action not in seen_actions:
                    interventions.append(h)

        report = EvaluationReport(
            game_day=game_day,
            metrics=metrics,
            scores=scores,
            triggers=triggers,
            interventions=interventions,
        )

        self._previous_metrics = metrics
        self._history.append(report)

        logger.info(
            "Overseer eval day %d: fitness=%.2f, triggers=%s, interventions=%d",
            game_day, metrics.mean_fitness, triggers, len(interventions),
        )

        return report

    def get_history(self, limit: int = 10) -> list[dict]:
        """Return recent evaluation reports."""
        return [r.to_dict() for r in self._history[-limit:]]

    def set_theme(self, theme: str) -> None:
        """Switch fitness weights to a different theme."""
        self.config = THEME_CONFIGS.get(theme, self.config)
        logger.info("Overseer theme set to '%s'", theme)
