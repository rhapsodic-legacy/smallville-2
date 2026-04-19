"""Tests for Big-5 personality drift and spawn-baseline decay."""

import pytest

from core.memory.reflection import (
    classify_emotional_valence,
    apply_personality_drift,
)
from core.npc.models import NPC, PersonalityTraits


def _make_npc(
    traits: dict[str, float] | None = None,
    baseline: dict[str, float] | None = None,
) -> NPC:
    personality = PersonalityTraits(**(traits or {}))
    baseline_personality = PersonalityTraits(**(baseline or traits or {}))
    return NPC(
        npc_id="drift_test",
        name="Testy",
        age=30,
        personality=personality,
        spawn_baseline=baseline_personality,
        backstory="drifter",
        occupation="farmer",
    )


class TestPersonalityTraitsMutation:
    def test_mutate_clamps_upper(self):
        p = PersonalityTraits(openness=0.95)
        p.mutate("openness", 0.2)
        assert p.openness == 1.0

    def test_mutate_clamps_lower(self):
        p = PersonalityTraits(neuroticism=0.05)
        p.mutate("neuroticism", -0.5)
        assert p.neuroticism == 0.0

    def test_mutate_ignores_unknown(self):
        p = PersonalityTraits()
        p.mutate("bogus", 0.5)
        # No error; unknown trait is silently dropped
        assert not hasattr(p, "bogus")

    def test_apply_deltas_sums(self):
        p = PersonalityTraits(openness=0.4, agreeableness=0.4)
        p.apply_deltas({"openness": 0.1, "agreeableness": -0.1})
        assert p.openness == pytest.approx(0.5)
        assert p.agreeableness == pytest.approx(0.3)

    def test_copy_is_independent(self):
        p = PersonalityTraits(openness=0.3)
        dup = p.copy()
        dup.openness = 0.9
        assert p.openness == 0.3

    def test_nudge_toward_pulls_back(self):
        # Drifted: openness 0.8, baseline 0.5 → 50% rate → 0.65
        p = PersonalityTraits(openness=0.8)
        baseline = PersonalityTraits(openness=0.5)
        p.nudge_toward(baseline, rate=0.5)
        assert p.openness == pytest.approx(0.65)

    def test_nudge_toward_rate_zero_is_noop(self):
        p = PersonalityTraits(openness=0.8)
        baseline = PersonalityTraits(openness=0.5)
        p.nudge_toward(baseline, rate=0.0)
        assert p.openness == pytest.approx(0.8)


class TestEmotionalValence:
    def test_positive_warmth_increases_agreeableness(self):
        d = classify_emotional_valence(
            "I am so grateful for my friend's kindness today.",
            importance=0.8,
        )
        assert d.get("agreeableness", 0.0) > 0
        # Joy reduces neuroticism
        assert d.get("neuroticism", 0.0) < 0

    def test_fear_increases_neuroticism(self):
        d = classify_emotional_valence(
            "I am terrified of what might happen tomorrow.",
            importance=0.8,
        )
        assert d.get("neuroticism", 0.0) > 0

    def test_empty_text_returns_empty_dict(self):
        assert classify_emotional_valence("", importance=0.5) == {}

    def test_neutral_text_returns_empty_dict(self):
        assert classify_emotional_valence(
            "The weather is clear today.", importance=0.5,
        ) == {}

    def test_importance_scales_magnitude(self):
        low = classify_emotional_valence("I am afraid.", importance=0.1)
        high = classify_emotional_valence("I am afraid.", importance=1.0)
        # Both have neuroticism; high should be larger in magnitude
        assert abs(high["neuroticism"]) > abs(low["neuroticism"])

    def test_drift_bounded_per_event(self):
        # A single maximum-importance reflection should not move any
        # trait by more than ~0.05 — the spec says "0.01–0.03 per event".
        # We allow a small buffer for rules that stack on the same trait.
        d = classify_emotional_valence(
            "I am so happy, grateful, and proud of what we accomplished.",
            importance=1.0,
        )
        for trait, delta in d.items():
            assert abs(delta) < 0.08, f"{trait} delta {delta} too large"


class TestApplyPersonalityDrift:
    def test_drift_nudges_personality_in_place(self):
        npc = _make_npc({"agreeableness": 0.5, "neuroticism": 0.5})
        delta = apply_personality_drift(
            npc, "I am so grateful to have friends.", importance=0.8,
        )
        assert delta  # non-empty
        assert npc.personality.agreeableness > 0.5
        assert npc.personality.neuroticism < 0.5

    def test_neutral_text_does_not_drift(self):
        npc = _make_npc({"agreeableness": 0.5, "neuroticism": 0.5})
        apply_personality_drift(npc, "The bridge is made of stone.")
        assert npc.personality.agreeableness == 0.5
        assert npc.personality.neuroticism == 0.5

    def test_cumulative_drift_stays_in_bounds(self):
        # 1000 fear-inducing reflections must not push neuroticism > 1.
        npc = _make_npc({"neuroticism": 0.5})
        for _ in range(1000):
            apply_personality_drift(
                npc, "I am afraid and anxious.", importance=1.0,
            )
        assert 0.0 <= npc.personality.neuroticism <= 1.0


class TestSpawnBaselineDecay:
    """The manager decays personality back toward spawn_baseline daily.

    We test the NPC-side primitive (nudge_toward) plus a direct call
    to the manager's helper via a minimal fixture.
    """

    def test_decay_pulls_toward_baseline(self):
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=42,
        )
        mgr.spawn_population(2)

        npc = mgr.npcs[0]
        assert npc.spawn_baseline is not None
        baseline_openness = npc.spawn_baseline.openness

        # Manually drift
        npc.personality.openness = min(1.0, baseline_openness + 0.2)
        drifted = npc.personality.openness

        mgr._decay_personalities()
        assert drifted > npc.personality.openness >= baseline_openness, (
            "decay should have pulled openness back toward baseline "
            f"(drifted={drifted}, now={npc.personality.openness}, "
            f"baseline={baseline_openness})"
        )

    def test_decay_preserves_baseline(self):
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=7)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=7,
        )
        mgr.spawn_population(2)
        npc = mgr.npcs[0]
        before = npc.spawn_baseline.to_dict()

        # Drift then decay a bunch
        for _ in range(10):
            npc.personality.mutate("openness", 0.05)
            mgr._decay_personalities()

        # Spawn baseline must never change — it's the anchor.
        assert npc.spawn_baseline.to_dict() == before

    def test_spawn_baseline_set_on_spawn(self):
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=3, terrain="riverside", seed=1)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=1,
        )
        mgr.spawn_population(3)

        for npc in mgr.npcs:
            assert npc.spawn_baseline is not None
            assert npc.spawn_baseline.to_dict() == npc.personality.to_dict()
