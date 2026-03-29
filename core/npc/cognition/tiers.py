"""
Tiered cognition system.

Assigns NPCs to cognitive tiers based on distance from the player
and relevance factors. Each tier defines which parts of the cognition
pipeline run and how often.

Tier 1: Full LLM cycle (perceive, plan, reflect, execute) — nearby, relevant
Tier 2: Simplified LLM (less frequent, shorter prompts) — medium distance
Tier 3: Statistical state machine (no LLM calls) — far away
Tier 4: Frozen (no updates) — very far or irrelevant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC


# ---------- Tier configuration ----------

@dataclass(frozen=True)
class TierConfig:
    """Configuration for a cognition tier."""
    tier: int
    perception_interval: float   # game minutes between perception cycles
    plan_interval: float         # game minutes between planning cycles
    uses_llm: bool               # whether this tier calls the LLM
    description: str


TIER_CONFIGS: dict[int, TierConfig] = {
    1: TierConfig(
        tier=1,
        perception_interval=2.0,
        plan_interval=15.0,
        uses_llm=True,
        description="Full LLM cognition — perceive, plan, reflect, execute",
    ),
    2: TierConfig(
        tier=2,
        perception_interval=10.0,
        plan_interval=60.0,
        uses_llm=True,
        description="Simplified LLM — less frequent, shorter prompts",
    ),
    3: TierConfig(
        tier=3,
        perception_interval=30.0,
        plan_interval=0.0,  # no LLM planning, uses state machine
        uses_llm=False,
        description="State machine — follows schedule, no LLM calls",
    ),
    4: TierConfig(
        tier=4,
        perception_interval=0.0,
        plan_interval=0.0,
        uses_llm=False,
        description="Frozen — no updates until relevant",
    ),
}


# ---------- Distance thresholds ----------

# Manhattan distance from camera/player focus point to NPC
TIER_1_RADIUS = 10   # tiles
TIER_2_RADIUS = 20   # tiles
TIER_3_RADIUS = 35   # tiles
# Beyond tier 3 radius → tier 4


def assign_tier(
    npc: NPC,
    focus_x: int,
    focus_z: int,
    relevance_boost: int = 0,
) -> int:
    """
    Determine the cognition tier for an NPC.

    Args:
        npc: The NPC to evaluate.
        focus_x, focus_z: Camera/player focus position.
        relevance_boost: Extra relevance (e.g. in conversation with player = -1 tier).

    Returns:
        Tier number (1-4).
    """
    distance = npc.distance_to(focus_x, focus_z)

    if distance <= TIER_1_RADIUS:
        tier = 1
    elif distance <= TIER_2_RADIUS:
        tier = 2
    elif distance <= TIER_3_RADIUS:
        tier = 3
    else:
        tier = 4

    # Apply relevance boost (lower tier = better)
    tier = max(1, tier - relevance_boost)

    # NPCs in conversation always get at least tier 2
    if npc.conversation_partner is not None:
        tier = min(tier, 2)

    return tier


def should_perceive(npc: NPC, current_game_minutes: float) -> bool:
    """Check if enough time has passed for this NPC's tier to perceive."""
    config = TIER_CONFIGS.get(npc.cognition_tier)
    if config is None or config.perception_interval <= 0:
        return False
    elapsed = current_game_minutes - npc.last_perception_tick
    return elapsed >= config.perception_interval


def should_plan(npc: NPC, current_game_minutes: float) -> bool:
    """Check if enough time has passed for this NPC's tier to plan."""
    config = TIER_CONFIGS.get(npc.cognition_tier)
    if config is None or config.plan_interval <= 0:
        return False
    elapsed = current_game_minutes - npc.last_plan_tick
    return elapsed >= config.plan_interval


def get_tier_config(tier: int) -> TierConfig:
    """Get configuration for a tier, defaulting to tier 4 for unknowns."""
    return TIER_CONFIGS.get(tier, TIER_CONFIGS[4])


def update_all_tiers(
    npcs: list[NPC],
    focus_x: int,
    focus_z: int,
) -> dict[int, list[str]]:
    """
    Reassign tiers for all NPCs based on current focus point.

    Returns a dict mapping tier -> list of npc_ids for logging.
    """
    tier_groups: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    for npc in npcs:
        new_tier = assign_tier(npc, focus_x, focus_z)
        npc.cognition_tier = new_tier
        tier_groups[new_tier].append(npc.npc_id)

    return tier_groups
