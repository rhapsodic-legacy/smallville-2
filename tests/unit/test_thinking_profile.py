"""
Tests for the ThinkingProfile layer.

Covers the three presets, purpose → thinking/budget resolution, and
the per-NPC override path that lets large towns mix qualities.
"""

import pytest

from core.npc.cognition.thinking import (
    FAST, BALANCED, DEEP, PRESETS,
    ThinkingLevel, ThinkingProfile, profile_for_level,
)
from core.npc.llm_client import MockProvider


class TestPresets:

    def test_fast_never_thinks(self):
        assert FAST.level == ThinkingLevel.FAST
        assert FAST.thinking_purposes == frozenset()
        assert FAST.should_think("daily_plan") is False
        assert FAST.should_think("conversation") is False
        assert FAST.should_think("reflection") is False

    def test_balanced_thinks_on_big_decisions(self):
        assert BALANCED.should_think("daily_plan") is True
        assert BALANCED.should_think("reflection") is True
        assert BALANCED.should_think("overseer") is True
        # Quick-response purposes do NOT think in balanced mode.
        assert BALANCED.should_think("conversation") is False
        assert BALANCED.should_think("reaction") is False

    def test_deep_thinks_on_conversation_too(self):
        assert DEEP.should_think("daily_plan") is True
        assert DEEP.should_think("conversation") is True
        assert DEEP.should_think("reaction") is True
        assert DEEP.should_think("replan") is True

    def test_budget_scales_with_level(self):
        # Thinking budget: DEEP > BALANCED > FAST.
        assert DEEP.thinking_budget > BALANCED.thinking_budget
        assert BALANCED.thinking_budget > FAST.thinking_budget

    def test_budget_for_thinking_purpose(self):
        # Thinking purposes get AT LEAST the profile's thinking budget.
        assert BALANCED.budget_for("daily_plan", 100) == BALANCED.thinking_budget
        # But caller requesting more gets more.
        assert BALANCED.budget_for("daily_plan", 2000) == 2000

    def test_budget_for_quick_purpose(self):
        # Quick purposes get AT MOST the profile's quick cap.
        assert BALANCED.budget_for("conversation", 5000) == BALANCED.quick_budget
        # But caller requesting less gets less.
        assert BALANCED.budget_for("conversation", 50) == 50


class TestProfileLookup:

    def test_lookup_by_string(self):
        assert profile_for_level("fast") is FAST
        assert profile_for_level("balanced") is BALANCED
        assert profile_for_level("deep") is DEEP

    def test_lookup_case_insensitive(self):
        assert profile_for_level("FAST") is FAST
        assert profile_for_level("Deep") is DEEP

    def test_lookup_by_enum(self):
        assert profile_for_level(ThinkingLevel.DEEP) is DEEP

    def test_unknown_falls_back_to_balanced(self):
        assert profile_for_level("nonsense") is BALANCED
        assert profile_for_level("") is BALANCED


class TestProviderIntegration:
    """Verify the LLMProvider base carries profile state and per-NPC overrides."""

    def test_default_profile_is_balanced(self):
        p = MockProvider()
        assert p.profile is BALANCED

    def test_set_profile(self):
        p = MockProvider()
        p.set_profile(FAST)
        assert p.profile is FAST

    def test_per_npc_override(self):
        p = MockProvider()
        p.set_profile(FAST)
        p.set_npc_profile("hero_1", DEEP)

        # Resolution: hero gets DEEP, everyone else inherits FAST.
        assert p._resolve_profile("hero_1") is DEEP
        assert p._resolve_profile("crowd_2") is FAST
        assert p._resolve_profile(None) is FAST

    def test_clear_npc_override(self):
        p = MockProvider()
        p.set_npc_profile("hero_1", DEEP)
        p.clear_npc_profile("hero_1")
        assert p._resolve_profile("hero_1") is p.profile

    def test_large_town_mixed_profile(self):
        """Realistic scenario: 3 heroes on DEEP, default FAST for the
        rest. The same provider serves both tiers."""
        p = MockProvider()
        p.set_profile(FAST)
        for hero in ("alice_0", "bob_1", "carol_2"):
            p.set_npc_profile(hero, DEEP)

        # Heroes think.
        for hero in ("alice_0", "bob_1", "carol_2"):
            assert p._resolve_profile(hero).should_think("conversation")

        # Crowd doesn't.
        for crowd in ("extra_7", "extra_8", "extra_9"):
            assert not p._resolve_profile(crowd).should_think("conversation")
