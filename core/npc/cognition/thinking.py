"""
Thinking-level profiles for LLM cognition.

Controls how deeply the LLM deliberates per decision and how large
its token budget is. Intended to be swappable per-NPC so that in a
large town, a few "hero" NPCs can run with DEEP thinking while the
background population runs on FAST (or pure deterministic via the
router) and the sim stays performant.

Three presets cover the common cases:

  FAST      — No thinking mode. Small token budget. Near-instant
              responses. Good for big towns, low-end hardware, or
              when the user prioritises throughput over quality.
  BALANCED  — Thinking mode on for high-value decisions only
              (daily plans, reflections, overseer evaluations).
              Default.
  DEEP      — Thinking mode on for nearly everything, including
              conversations and reactions. Larger budget. Slow but
              highest-quality NPC behaviour. Good for story beats or
              "hero" characters the player focuses on.

Layer structure (intentionally expandable):

  1. Provider-level default profile — set by the user's global
     thinking toggle. Applies to every LLM call that doesn't specify
     otherwise.
  2. Per-NPC overrides — `LLMProvider.set_npc_profile(npc_id, profile)`
     lets specific NPCs opt into a different profile. Hero NPCs DEEP,
     crowd extras FAST.
  3. Per-call hint — callers can pass `npc_id=` to `complete()` and
     the provider resolves the profile automatically.

The profile does NOT decide LLM vs deterministic — that's the
router's job (CognitionPolicy). ThinkingProfile only shapes WHAT the
LLM does when it's already been chosen. Combining the two gives the
full spread: deterministic → FAST → BALANCED → DEEP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ThinkingLevel(str, Enum):
    """Three coarse quality tiers exposed to users."""
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


@dataclass(frozen=True)
class ThinkingProfile:
    """
    Bundle of settings that controls a single LLM call's depth.

    Attributes:
        level: The coarse label this profile corresponds to.
        thinking_purposes: Decision purposes for which thinking mode
            should be enabled. Providers that support thinking (e.g.
            Gemma via Ollama) honour this. Providers that don't
            (MockProvider, ClaudeProvider today) ignore it.
        thinking_budget: Max output tokens when thinking mode is on.
            Thinking + answer both draw from this pool, so bigger =
            more deliberation.
        quick_budget: Max output tokens when thinking mode is off.
            Kept small because a direct response shouldn't need much.
        temperature_multiplier: Applied to the caller's requested
            temperature. 1.0 = no change; higher values make outputs
            more varied (handy for DEEP to explore ideas).
    """
    level: ThinkingLevel = ThinkingLevel.BALANCED
    thinking_purposes: frozenset[str] = field(default_factory=frozenset)
    thinking_budget: int = 500
    quick_budget: int = 150
    temperature_multiplier: float = 1.0

    def should_think(self, purpose: str) -> bool:
        """Should the LLM use thinking mode for this purpose?"""
        return purpose in self.thinking_purposes

    def budget_for(self, purpose: str, requested: int) -> int:
        """Resolve the effective max_tokens for a call at this profile."""
        if purpose in self.thinking_purposes:
            return max(requested, self.thinking_budget)
        # Quick-response purposes get a small cap; honour smaller
        # caller requests but never exceed the profile's cap.
        return min(requested, self.quick_budget)


# ---------- Presets ----------
#
# The purpose sets below are deliberately conservative:
# - FAST never thinks.
# - BALANCED thinks on big-picture decisions only (daily plans,
#   reflections, overseer evaluation).
# - DEEP adds thinking for conversational and replanning moments too.

FAST: ThinkingProfile = ThinkingProfile(
    level=ThinkingLevel.FAST,
    thinking_purposes=frozenset(),
    thinking_budget=200,
    quick_budget=120,
    temperature_multiplier=0.9,
)

BALANCED: ThinkingProfile = ThinkingProfile(
    level=ThinkingLevel.BALANCED,
    thinking_purposes=frozenset({
        "daily_plan",
        "reflection",
        "overseer",
    }),
    thinking_budget=500,
    quick_budget=150,
    temperature_multiplier=1.0,
)

DEEP: ThinkingProfile = ThinkingProfile(
    level=ThinkingLevel.DEEP,
    thinking_purposes=frozenset({
        "daily_plan",
        "reflection",
        "overseer",
        "conversation_initiate",
        "conversation",
        "replan",
        "reaction",
    }),
    thinking_budget=1024,
    quick_budget=250,
    temperature_multiplier=1.1,
)


PRESETS: dict[ThinkingLevel, ThinkingProfile] = {
    ThinkingLevel.FAST: FAST,
    ThinkingLevel.BALANCED: BALANCED,
    ThinkingLevel.DEEP: DEEP,
}


def profile_for_level(level: str | ThinkingLevel) -> ThinkingProfile:
    """Look up a preset by level name or enum. Defaults to BALANCED on unknown input."""
    if isinstance(level, str):
        try:
            level = ThinkingLevel(level.lower())
        except ValueError:
            return BALANCED
    return PRESETS.get(level, BALANCED)
