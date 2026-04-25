"""Evolution module — overseer agent, fitness functions, policy injection."""

from core.evolution.fitness import (
    FitnessConfig,
    FitnessScore,
    PopulationMetrics,
    evaluate_npc,
    evaluate_population,
    THEME_CONFIGS,
)
from core.evolution.overseer import (
    Intervention,
    EvaluationReport,
    Overseer,
)
from core.evolution.mechanisms import (
    PolicyTemplate,
    PromptModifier,
    MechanismEngine,
    POLICY_TEMPLATES,
)
from core.evolution.guardrails import (
    GuardrailEngine,
    GuardrailResult,
    RuleViolation,
)

__all__ = [
    # Fitness
    "FitnessConfig",
    "FitnessScore",
    "PopulationMetrics",
    "evaluate_npc",
    "evaluate_population",
    "THEME_CONFIGS",
    # Overseer
    "Intervention",
    "EvaluationReport",
    "Overseer",
    # Mechanisms
    "PolicyTemplate",
    "PromptModifier",
    "MechanismEngine",
    "POLICY_TEMPLATES",
    # Guardrails
    "GuardrailEngine",
    "GuardrailResult",
    "RuleViolation",
]
