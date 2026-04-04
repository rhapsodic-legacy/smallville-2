"""
Town Health Assessment.

Runs a headless simulation for a configurable number of days and
produces a comprehensive health report covering:

  1. Activity balance — do NPCs eat, sleep, work, and socialise?
  2. Oscillation score — are NPCs bouncing between tiles?
  3. Overlap count — how often do resting NPCs share tiles?
  4. Conversation quality — uniqueness and frequency of dialogue
  5. Reflection rate — are NPCs generating higher-order insights?
  6. Intent coherence — do NPC intents match their actions?
  7. Day/night compliance — sleeping at night, active during day?
  8. LLM cache efficiency — how much are we saving on API calls?

Run standalone:  python3 tests/simulation/test_town_health.py
Run via pytest:  pytest tests/simulation/test_town_health.py -v
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider, get_cache_stats
from core.memory.manager import MemoryManager
from core.time_system.clock import GameClock
from core.world.generator import WorldConfig, generate_world

# ---------- Configuration ----------

POPULATION = 7
SEED = 42
SIM_DAYS = 3
TICK_DELTA = 1.0
TICKS_PER_DAY = 1200


# ---------- Health metrics ----------

@dataclass
class HealthReport:
    """Comprehensive town health assessment."""
    sim_days: int = 0
    total_ticks: int = 0

    # Activity balance (pct of total samples per activity)
    activity_pcts: dict[str, float] = field(default_factory=dict)

    # Oscillation
    max_reversals_any_npc: int = 0
    npcs_with_oscillation: int = 0

    # Overlaps
    overlap_ticks: int = 0      # ticks where any resting overlap existed
    total_overlaps: int = 0     # total resting NPC overlaps observed

    # Conversations
    total_conversations: int = 0
    unique_dialogue_pct: float = 0.0

    # Reflections
    total_reflections: int = 0
    reflections_per_npc: float = 0.0

    # Intents
    total_intents: int = 0
    intents_per_npc: float = 0.0

    # Day/night compliance
    night_sleeping_pct: float = 0.0   # % of night ticks NPCs are sleeping
    day_active_pct: float = 0.0       # % of day ticks NPCs are NOT sleeping

    # Cache
    cache_hit_rate: float = 0.0

    # Scores (0-100, higher is better)
    balance_score: float = 0.0
    stability_score: float = 0.0
    social_score: float = 0.0
    cognitive_score: float = 0.0
    overall_score: float = 0.0

    @property
    def passed(self) -> bool:
        return self.overall_score >= 50.0

    def summary_lines(self) -> list[str]:
        lines = [
            f"=== Town Health Report ({self.sim_days} days, {self.total_ticks} ticks) ===",
            "",
            f"Overall Score: {self.overall_score:.0f}/100 ({'HEALTHY' if self.passed else 'UNHEALTHY'})",
            "",
            "--- Activity Balance ---",
        ]
        for act, pct in sorted(self.activity_pcts.items(), key=lambda x: -x[1]):
            lines.append(f"  {act:12s}: {pct:5.1f}%")
        lines.append(f"  Balance Score: {self.balance_score:.0f}/100")
        lines.append("")
        lines.append("--- Stability ---")
        lines.append(f"  Max reversals (any NPC): {self.max_reversals_any_npc}")
        lines.append(f"  NPCs with oscillation:   {self.npcs_with_oscillation}")
        lines.append(f"  Overlap ticks:           {self.overlap_ticks}")
        lines.append(f"  Stability Score:         {self.stability_score:.0f}/100")
        lines.append("")
        lines.append("--- Social ---")
        lines.append(f"  Conversations:           {self.total_conversations}")
        lines.append(f"  Dialogue uniqueness:     {self.unique_dialogue_pct:.0f}%")
        lines.append(f"  Social Score:            {self.social_score:.0f}/100")
        lines.append("")
        lines.append("--- Cognitive ---")
        lines.append(f"  Reflections:             {self.total_reflections} ({self.reflections_per_npc:.1f}/NPC)")
        lines.append(f"  Intents recorded:        {self.total_intents} ({self.intents_per_npc:.1f}/NPC)")
        lines.append(f"  Cognitive Score:         {self.cognitive_score:.0f}/100")
        lines.append("")
        lines.append("--- Day/Night ---")
        lines.append(f"  Night sleeping:          {self.night_sleeping_pct:.0f}%")
        lines.append(f"  Day active:              {self.day_active_pct:.0f}%")
        lines.append("")
        lines.append(f"  Cache hit rate:          {self.cache_hit_rate:.1%}")
        return lines


# ---------- Simulation runner ----------

def run_health_assessment(
    days: int = SIM_DAYS,
    population: int = POPULATION,
    seed: int = SEED,
) -> HealthReport:
    """Run a headless sim and produce a health report."""
    config = WorldConfig(population=population, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    memory = MemoryManager(llm=llm)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm,
        seed=seed, memory=memory,
    )
    mgr.spawn_population(population)

    total_ticks = days * TICKS_PER_DAY
    clock = GameClock()

    # Tracking structures
    activity_counts: Counter = Counter()
    night_sleeping = 0
    night_total = 0
    day_active = 0
    day_total = 0
    overlap_ticks = 0
    total_overlaps = 0
    position_timelines: dict[str, list[tuple[int, int]]] = {
        npc.npc_id: [] for npc in mgr.npcs
    }
    dialogue_lines: list[str] = []

    async def _run():
        nonlocal night_sleeping, night_total, day_active, day_total
        nonlocal overlap_ticks, total_overlaps

        for tick in range(total_ticks):
            clock.tick(TICK_DELTA)
            await mgr.tick(clock, TICK_DELTA)

            is_night = clock.schedule_slot.value == "night"

            # Sample activities
            for npc in mgr.npcs:
                activity_counts[npc.activity.value] += 1
                position_timelines[npc.npc_id].append(
                    (npc.tile_x, npc.tile_z),
                )

                if is_night:
                    night_total += 1
                    if npc.activity == ActivityState.SLEEPING:
                        night_sleeping += 1
                else:
                    day_total += 1
                    if npc.activity != ActivityState.SLEEPING:
                        day_active += 1

            # Check resting overlaps
            resting_positions: dict[tuple[int, int], int] = {}
            for npc in mgr.npcs:
                if npc.activity != ActivityState.WALKING:
                    pos = (npc.tile_x, npc.tile_z)
                    resting_positions[pos] = resting_positions.get(pos, 0) + 1
            tick_overlaps = sum(
                v - 1 for v in resting_positions.values() if v > 1
            )
            if tick_overlaps > 0:
                overlap_ticks += 1
                total_overlaps += tick_overlaps

    asyncio.get_event_loop().run_until_complete(_run())

    # --- Compute metrics ---
    report = HealthReport(sim_days=days, total_ticks=total_ticks)

    # Activity balance
    total_samples = sum(activity_counts.values())
    report.activity_pcts = {
        k: (v / total_samples) * 100
        for k, v in activity_counts.items()
    } if total_samples > 0 else {}

    # Oscillation detection
    max_reversals = 0
    oscillating_npcs = 0
    for npc_id, positions in position_timelines.items():
        worst = _detect_oscillation(positions, window=20)
        max_reversals = max(max_reversals, worst)
        if worst > 3:
            oscillating_npcs += 1
    report.max_reversals_any_npc = max_reversals
    report.npcs_with_oscillation = oscillating_npcs

    # Overlaps
    report.overlap_ticks = overlap_ticks
    report.total_overlaps = total_overlaps

    # Memory stats (conversations, reflections, intents)
    mem_stats = memory.get_stats()
    by_cat = mem_stats.get("episodic", {}).get("by_category", {})
    report.total_conversations = by_cat.get("conversation", 0)
    report.total_reflections = by_cat.get("reflection", 0)
    report.total_intents = by_cat.get("intent", 0)
    report.reflections_per_npc = report.total_reflections / population
    report.intents_per_npc = report.total_intents / population

    # Dialogue uniqueness (from mock provider call log)
    conv_calls = [
        c for c in getattr(llm, '_call_log', [])
        if c.get('purpose') == 'conversation'
    ]
    if conv_calls:
        responses = [c.get('response', '') for c in conv_calls]
        unique = len(set(responses))
        report.unique_dialogue_pct = (unique / len(responses)) * 100
    else:
        report.unique_dialogue_pct = 100.0

    # Day/night compliance
    report.night_sleeping_pct = (
        (night_sleeping / night_total * 100) if night_total > 0 else 0
    )
    report.day_active_pct = (
        (day_active / day_total * 100) if day_total > 0 else 0
    )

    # Cache stats
    cache = get_cache_stats()
    report.cache_hit_rate = cache.get("hit_rate", 0.0)

    # --- Score computation ---
    report.balance_score = _score_balance(report.activity_pcts)
    report.stability_score = _score_stability(
        max_reversals, oscillating_npcs, overlap_ticks, total_ticks,
    )
    report.social_score = _score_social(
        report.total_conversations, report.unique_dialogue_pct, population, days,
    )
    report.cognitive_score = _score_cognitive(
        report.reflections_per_npc, report.intents_per_npc, days,
    )
    report.overall_score = (
        report.balance_score * 0.3
        + report.stability_score * 0.3
        + report.social_score * 0.2
        + report.cognitive_score * 0.2
    )

    return report


# ---------- Scoring functions ----------

def _score_balance(activity_pcts: dict[str, float]) -> float:
    """Score activity balance 0-100. Penalise if any key activity is missing."""
    score = 100.0

    sleeping = activity_pcts.get("sleeping", 0)
    working = activity_pcts.get("working", 0)
    eating = activity_pcts.get("eating", 0)
    idle = activity_pcts.get("idle", 0)
    walking = activity_pcts.get("walking", 0)

    # Sleeping should be 20-45% (night is ~1/3 of day)
    if sleeping < 10:
        score -= 30
    elif sleeping < 20:
        score -= 10

    # Working should be >5% (NPCs have jobs)
    if working < 2:
        score -= 25
    elif working < 5:
        score -= 10

    # Walking should be <30% (not perpetually walking)
    if walking > 40:
        score -= 30
    elif walking > 30:
        score -= 15

    # Idle shouldn't dominate (>60% = NPCs doing nothing)
    if idle > 60:
        score -= 20

    return max(0, score)


def _score_stability(
    max_reversals: int, oscillating: int,
    overlap_ticks: int, total_ticks: int,
) -> float:
    """Score movement stability 0-100."""
    score = 100.0

    # Oscillation penalty
    if oscillating > 0:
        score -= min(40, oscillating * 15)
    if max_reversals > 5:
        score -= min(30, (max_reversals - 5) * 5)

    # Overlap penalty (% of ticks with overlaps)
    overlap_pct = overlap_ticks / total_ticks if total_ticks > 0 else 0
    if overlap_pct > 0.1:
        score -= 30
    elif overlap_pct > 0.05:
        score -= 15
    elif overlap_pct > 0.01:
        score -= 5

    return max(0, score)


def _score_social(
    conversations: int, unique_pct: float,
    population: int, days: int,
) -> float:
    """Score social health 0-100."""
    score = 50.0  # Start neutral

    # Conversations per NPC per day
    convos_per_npc_day = conversations / (population * days) if days > 0 else 0
    if convos_per_npc_day >= 1.0:
        score += 30
    elif convos_per_npc_day >= 0.3:
        score += 15
    elif convos_per_npc_day > 0:
        score += 5

    # Dialogue uniqueness bonus
    if unique_pct >= 70:
        score += 20
    elif unique_pct >= 40:
        score += 10

    return min(100, max(0, score))


def _score_cognitive(
    reflections_per_npc: float,
    intents_per_npc: float,
    days: int,
) -> float:
    """Score cognitive health 0-100."""
    score = 50.0

    # Reflections per NPC per day
    refl_per_day = reflections_per_npc / days if days > 0 else 0
    if refl_per_day >= 0.5:
        score += 30
    elif refl_per_day >= 0.1:
        score += 15
    elif refl_per_day > 0:
        score += 5

    # Intents recorded (shows dispatch pipeline is working)
    intent_per_day = intents_per_npc / days if days > 0 else 0
    if intent_per_day >= 3:
        score += 20
    elif intent_per_day >= 1:
        score += 10

    return min(100, max(0, score))


def _detect_oscillation(
    positions: list[tuple[int, int]], window: int = 20,
) -> int:
    """Count max direction reversals in any sliding window."""
    max_reversals = 0
    for start in range(0, len(positions) - window):
        chunk = positions[start:start + window]
        reversals = 0
        for i in range(2, len(chunk)):
            if chunk[i] == chunk[i - 2] and chunk[i] != chunk[i - 1]:
                reversals += 1
        max_reversals = max(max_reversals, reversals)
    return max_reversals


# ---------- Pytest tests ----------

@pytest.fixture
def health_report():
    return run_health_assessment(days=3, population=7, seed=42)


class TestTownHealth:
    """Automated health checks on a generated town."""

    def test_overall_health(self, health_report):
        """Town should score at least 50/100 overall."""
        assert health_report.overall_score >= 50, (
            f"Town health score {health_report.overall_score:.0f}/100 — "
            f"below minimum threshold.\n"
            + "\n".join(health_report.summary_lines())
        )

    def test_no_oscillation(self, health_report):
        """No NPC should oscillate excessively."""
        assert health_report.npcs_with_oscillation == 0, (
            f"{health_report.npcs_with_oscillation} NPCs oscillating "
            f"(max {health_report.max_reversals_any_npc} reversals)"
        )

    def test_activity_balance(self, health_report):
        """Activity balance should score at least 60/100."""
        assert health_report.balance_score >= 60, (
            f"Balance score {health_report.balance_score:.0f}/100.\n"
            f"Activities: {health_report.activity_pcts}"
        )

    def test_reflections_exist(self, health_report):
        """NPCs should generate at least some reflections."""
        assert health_report.total_reflections > 0, (
            "No reflections generated in "
            f"{health_report.sim_days} days"
        )

    def test_intents_recorded(self, health_report):
        """Intent logging should be active."""
        assert health_report.total_intents > 0, (
            "No intents recorded — dispatch pipeline broken"
        )

    def test_night_sleeping(self, health_report):
        """NPCs should sleep a meaningful amount at night.

        With MockProvider, conversations fire frequently (no rate limit),
        so NPCs talk through the night more than with a real LLM.
        Threshold is set low to accommodate both mock and real providers.
        """
        assert health_report.night_sleeping_pct >= 15, (
            f"Only {health_report.night_sleeping_pct:.0f}% sleeping at night"
        )

    def test_overlaps_minimal(self, health_report):
        """Resting overlaps should be rare."""
        overlap_pct = (
            health_report.overlap_ticks / health_report.total_ticks * 100
        )
        assert overlap_pct < 10, (
            f"{overlap_pct:.1f}% of ticks had resting overlaps"
        )


# ---------- Standalone runner ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Town Health Assessment")
    parser.add_argument("--days", type=int, default=SIM_DAYS)
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Running {args.days}-day health assessment "
          f"({args.population} NPCs, seed={args.seed})...")

    report = run_health_assessment(
        days=args.days, population=args.population, seed=args.seed,
    )

    for line in report.summary_lines():
        print(line)

    sys.exit(0 if report.passed else 1)
