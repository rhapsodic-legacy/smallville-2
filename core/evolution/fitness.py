"""
Fitness functions for population evaluation.

Multi-objective scoring across five dimensions:
  - Survival: health, energy, hunger satisfaction
  - Prosperity: gold, resources, skill development
  - Social: relationship count, sentiment quality, conversation frequency
  - Goals: progress toward stated long-term goals
  - Engagement: schedule variety, activity diversity, movement patterns

Each dimension returns 0.0–1.0. A FitnessConfig sets per-dimension weights
so different world themes (farming village vs trading hub vs warzone) can
prioritise different behaviours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.memory.manager import MemoryManager
    from core.relationships.sentiment import SentimentTracker

logger = logging.getLogger(__name__)


# ---------- Configuration ----------

@dataclass
class FitnessConfig:
    """Weights for each fitness dimension (must sum to ~1.0)."""
    survival: float = 0.20
    prosperity: float = 0.20
    social: float = 0.25
    goals: float = 0.15
    engagement: float = 0.20

    def to_dict(self) -> dict[str, float]:
        return {
            "survival": self.survival,
            "prosperity": self.prosperity,
            "social": self.social,
            "goals": self.goals,
            "engagement": self.engagement,
        }


# Preset configs for different world themes
THEME_CONFIGS: dict[str, FitnessConfig] = {
    "default": FitnessConfig(),
    "farming": FitnessConfig(
        survival=0.25, prosperity=0.25, social=0.15, goals=0.15, engagement=0.20,
    ),
    "trading": FitnessConfig(
        survival=0.15, prosperity=0.30, social=0.25, goals=0.15, engagement=0.15,
    ),
    "warzone": FitnessConfig(
        survival=0.35, prosperity=0.10, social=0.20, goals=0.20, engagement=0.15,
    ),
    "social": FitnessConfig(
        survival=0.10, prosperity=0.10, social=0.40, goals=0.15, engagement=0.25,
    ),
}


# ---------- Per-NPC dimension scorers ----------

def score_survival(npc: NPC) -> float:
    """Score an NPC's physical well-being (0–1).

    Combines health, energy, and hunger (inverted — 0 hunger is ideal).
    """
    hunger_score = 1.0 - npc.hunger  # 0 hunger → 1.0
    return (npc.health * 0.4 + npc.energy * 0.3 + hunger_score * 0.3)


def score_prosperity(npc: NPC) -> float:
    """Score an NPC's economic success (0–1).

    Gold: logarithmic (diminishing returns past ~100).
    Skills: average skill level.
    Inventory: has at least some resources.
    """
    import math

    # Gold: 0 gold → 0.0, 50 gold → 0.5, 100+ gold → ~0.8
    gold_score = min(1.0, math.log1p(npc.gold) / math.log1p(100))

    # Skills: average of all skill values (each 0–1)
    skill_vals = list(npc.skills.values())
    skill_score = sum(skill_vals) / max(len(skill_vals), 1)

    # Inventory: 1.0 if they have any resources, scaled by variety
    inv_types = len([v for v in npc.inventory.values() if v > 0])
    inv_score = min(1.0, inv_types / 3.0)

    return gold_score * 0.5 + skill_score * 0.3 + inv_score * 0.2


def score_social(
    npc: NPC,
    sentiment: SentimentTracker | None = None,
) -> float:
    """Score an NPC's social integration (0–1).

    Relationship count, average disposition, and conversation frequency.
    """
    if sentiment is None:
        return 0.5  # neutral when no sentiment data

    relationships = sentiment.get_all_for(npc.npc_id)

    # Relationship count: 0 → 0.0, 3+ → 1.0
    rel_count = len(relationships)
    count_score = min(1.0, rel_count / 3.0)

    # Average disposition: normalised to 0–1 range (disposition is -100 to +100)
    if relationships:
        avg_disp = sum(s.overall_disposition() for s in relationships) / rel_count
        disp_score = (avg_disp + 100) / 200  # maps [-100, +100] → [0, 1]
    else:
        disp_score = 0.5

    # Conversation recency: recent conversation → higher score
    # conversation_cooldown is 60 min, so talking within last 120 min is good
    convo_recency = 0.0
    if npc.last_conversation_time > 0:
        convo_recency = 1.0  # had at least one conversation

    return count_score * 0.4 + disp_score * 0.4 + convo_recency * 0.2


def score_goals(
    npc: NPC,
    memory: MemoryManager | None = None,
) -> float:
    """Score an NPC's progress toward goals (0–1).

    Having goals at all is worth something. Active goals in structured
    memory with substeps completed are worth more.
    """
    # Has goals defined
    has_goals = 1.0 if npc.long_term_goals else 0.0

    # Goal progress from structured memory
    goal_progress = 0.0
    if memory is not None:
        goals = memory.structured.get_active_goals(npc.npc_id)
        if goals:
            completed = sum(1 for g in goals if g.status == "completed")
            goal_progress = completed / len(goals)

    return has_goals * 0.3 + goal_progress * 0.7


def score_engagement(npc: NPC) -> float:
    """Score how actively the NPC engages with the world (0–1).

    Activity diversity: not stuck in one state all day.
    Schedule variety: has multiple different activities planned.
    Movement: not frozen in place.
    """
    # Schedule variety: unique activities in daily schedule
    if npc.daily_schedule:
        activities = {e.activity.lower() for e in npc.daily_schedule}
        locations = {e.location for e in npc.daily_schedule}
        variety_score = min(1.0, len(activities) / 5.0)
        location_score = min(1.0, len(locations) / 3.0)
    else:
        variety_score = 0.0
        location_score = 0.0

    # Is doing something (not idle)
    from core.npc.models import ActivityState
    active_score = 0.0 if npc.activity == ActivityState.IDLE else 1.0

    return variety_score * 0.4 + location_score * 0.3 + active_score * 0.3


# ---------- Composite scoring ----------

@dataclass
class FitnessScore:
    """Complete fitness evaluation for an NPC."""
    npc_id: str
    npc_name: str
    survival: float = 0.0
    prosperity: float = 0.0
    social: float = 0.0
    goals: float = 0.0
    engagement: float = 0.0
    weighted_total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "npc_id": self.npc_id,
            "npc_name": self.npc_name,
            "survival": round(self.survival, 3),
            "prosperity": round(self.prosperity, 3),
            "social": round(self.social, 3),
            "goals": round(self.goals, 3),
            "engagement": round(self.engagement, 3),
            "weighted_total": round(self.weighted_total, 3),
        }


def evaluate_npc(
    npc: NPC,
    config: FitnessConfig | None = None,
    sentiment: SentimentTracker | None = None,
    memory: MemoryManager | None = None,
) -> FitnessScore:
    """Evaluate a single NPC across all fitness dimensions."""
    cfg = config or FitnessConfig()

    s = FitnessScore(npc_id=npc.npc_id, npc_name=npc.name)
    s.survival = score_survival(npc)
    s.prosperity = score_prosperity(npc)
    s.social = score_social(npc, sentiment)
    s.goals = score_goals(npc, memory)
    s.engagement = score_engagement(npc)
    s.weighted_total = (
        s.survival * cfg.survival
        + s.prosperity * cfg.prosperity
        + s.social * cfg.social
        + s.goals * cfg.goals
        + s.engagement * cfg.engagement
    )
    return s


# ---------- Population-level metrics ----------

@dataclass
class PopulationMetrics:
    """Aggregated fitness across the entire population."""
    population_size: int = 0
    mean_fitness: float = 0.0
    min_fitness: float = 0.0
    max_fitness: float = 0.0
    mean_survival: float = 0.0
    mean_prosperity: float = 0.0
    mean_social: float = 0.0
    mean_goals: float = 0.0
    mean_engagement: float = 0.0
    struggling_npcs: list[str] = field(default_factory=list)
    thriving_npcs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "population_size": self.population_size,
            "mean_fitness": round(self.mean_fitness, 3),
            "min_fitness": round(self.min_fitness, 3),
            "max_fitness": round(self.max_fitness, 3),
            "mean_survival": round(self.mean_survival, 3),
            "mean_prosperity": round(self.mean_prosperity, 3),
            "mean_social": round(self.mean_social, 3),
            "mean_goals": round(self.mean_goals, 3),
            "mean_engagement": round(self.mean_engagement, 3),
            "struggling_npcs": self.struggling_npcs,
            "thriving_npcs": self.thriving_npcs,
        }


def evaluate_population(
    npcs: list[NPC],
    config: FitnessConfig | None = None,
    sentiment: SentimentTracker | None = None,
    memory: MemoryManager | None = None,
    struggling_threshold: float = 0.3,
    thriving_threshold: float = 0.7,
) -> tuple[list[FitnessScore], PopulationMetrics]:
    """Evaluate the entire population and compute aggregate metrics.

    Returns (individual_scores, population_metrics).
    """
    if not npcs:
        return [], PopulationMetrics()

    scores = [evaluate_npc(npc, config, sentiment, memory) for npc in npcs]

    totals = [s.weighted_total for s in scores]
    metrics = PopulationMetrics(
        population_size=len(npcs),
        mean_fitness=sum(totals) / len(totals),
        min_fitness=min(totals),
        max_fitness=max(totals),
        mean_survival=sum(s.survival for s in scores) / len(scores),
        mean_prosperity=sum(s.prosperity for s in scores) / len(scores),
        mean_social=sum(s.social for s in scores) / len(scores),
        mean_goals=sum(s.goals for s in scores) / len(scores),
        mean_engagement=sum(s.engagement for s in scores) / len(scores),
        struggling_npcs=[
            s.npc_name for s in scores if s.weighted_total < struggling_threshold
        ],
        thriving_npcs=[
            s.npc_name for s in scores if s.weighted_total > thriving_threshold
        ],
    )

    return scores, metrics
