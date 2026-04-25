"""Tests for the evolution layer: fitness, overseer, mechanisms, guardrails."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.npc.models import NPC, PersonalityTraits, ScheduleEntry, ActivityState
from core.evolution.fitness import (
    FitnessConfig, FitnessScore, PopulationMetrics,
    score_survival, score_prosperity, score_social, score_goals,
    score_engagement, evaluate_npc, evaluate_population, THEME_CONFIGS,
)
from core.evolution.overseer import (
    Intervention, EvaluationReport, Overseer,
    detect_triggers, _heuristic_interventions, _parse_interventions,
)
from core.evolution.mechanisms import (
    PolicyTemplate, PromptModifier, MechanismEngine, POLICY_TEMPLATES,
)
from core.evolution.guardrails import (
    GuardrailEngine, GuardrailResult, RuleViolation,
    check_param_bounds, clamp_npc_params, RateLimiter,
    MAX_INTERVENTIONS_PER_CYCLE, MAX_POLICIES_PER_NPC,
    NPC_PARAM_BOUNDS,
)


# ---------- Fixtures ----------

def _make_npc(
    npc_id: str = "npc_1",
    name: str = "Alice",
    health: float = 0.8,
    energy: float = 0.7,
    hunger: float = 0.2,
    gold: int = 50,
    skills: dict | None = None,
    inventory: dict | None = None,
    activity: ActivityState = ActivityState.WORKING,
    schedule: list | None = None,
    goals: list | None = None,
    conversation_cooldown: float = 60,
    last_conversation_time: float = 100,
) -> NPC:
    """Helper to create test NPCs with sensible defaults."""
    npc = NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory="Test NPC",
        occupation="farmer",
    )
    npc.health = health
    npc.energy = energy
    npc.hunger = hunger
    npc.gold = gold
    npc.skills = skills or {"farming": 0.6, "cooking": 0.4}
    npc.inventory = inventory or {"food": 3, "wood": 1}
    npc.activity = activity
    npc.long_term_goals = goals if goals is not None else ["become a master farmer"]
    npc.conversation_cooldown = conversation_cooldown
    npc.last_conversation_time = last_conversation_time
    if schedule is not None:
        npc.daily_schedule = schedule
    else:
        npc.daily_schedule = [
            ScheduleEntry(slot="morning", activity="farm", location="field"),
            ScheduleEntry(slot="afternoon", activity="trade", location="market"),
            ScheduleEntry(slot="evening", activity="eat", location="tavern"),
            ScheduleEntry(slot="night", activity="sleep", location="home"),
        ]
    return npc


def _make_intervention(
    itype: str = "parameter_tune",
    target: str = "population",
    action: str = "boost_conversation_chance",
    reason: str = "test",
    params: dict | None = None,
) -> Intervention:
    return Intervention(
        intervention_type=itype,
        target=target,
        action=action,
        reason=reason,
        parameters=params or {},
    )


# ====================================================================
# FITNESS TESTS
# ====================================================================

class TestScoreSurvival:
    def test_perfect_health(self):
        npc = _make_npc(health=1.0, energy=1.0, hunger=0.0)
        assert score_survival(npc) == pytest.approx(1.0)

    def test_worst_state(self):
        npc = _make_npc(health=0.0, energy=0.0, hunger=1.0)
        assert score_survival(npc) == pytest.approx(0.0)

    def test_mixed_state(self):
        npc = _make_npc(health=0.5, energy=0.5, hunger=0.5)
        score = score_survival(npc)
        assert 0.3 < score < 0.7


class TestScoreProsperity:
    def test_wealthy_npc(self):
        npc = _make_npc(gold=200, skills={"a": 0.9, "b": 0.8}, inventory={"x": 5, "y": 3, "z": 1})
        score = score_prosperity(npc)
        assert score > 0.7

    def test_poor_npc(self):
        npc = _make_npc(gold=0, skills={"a": 0.1}, inventory={})
        score = score_prosperity(npc)
        assert score < 0.3

    def test_gold_diminishing_returns(self):
        npc_50 = _make_npc(gold=50)
        npc_500 = _make_npc(gold=500)
        score_50 = score_prosperity(npc_50)
        score_500 = score_prosperity(npc_500)
        # 10x more gold should not give 10x more score
        assert score_500 < score_50 * 3


class TestScoreSocial:
    def test_no_sentiment_tracker(self):
        npc = _make_npc()
        assert score_social(npc, None) == 0.5

    def test_with_relationships(self):
        npc = _make_npc()
        tracker = MagicMock()
        rel = MagicMock()
        rel.overall_disposition.return_value = 50.0
        tracker.get_all_for.return_value = [rel, rel, rel]
        score = score_social(npc, tracker)
        assert score > 0.5


class TestScoreGoals:
    def test_no_goals(self):
        npc = _make_npc(goals=[])
        assert score_goals(npc, None) == 0.0

    def test_has_goals_no_memory(self):
        npc = _make_npc(goals=["become rich"])
        score = score_goals(npc, None)
        assert score == pytest.approx(0.3)  # has_goals * 0.3 + 0

    def test_with_completed_goals(self):
        npc = _make_npc(goals=["become rich"])
        memory = MagicMock()
        goal = MagicMock()
        goal.status = "completed"
        memory.structured.get_active_goals.return_value = [goal]
        score = score_goals(npc, memory)
        assert score == pytest.approx(1.0)  # 0.3 + 0.7


class TestScoreEngagement:
    def test_active_varied_schedule(self):
        npc = _make_npc(activity=ActivityState.WORKING)
        # Default schedule has 4 unique activities + 4 unique locations
        score = score_engagement(npc)
        assert score > 0.5

    def test_idle_no_schedule(self):
        npc = _make_npc(activity=ActivityState.IDLE, schedule=[])
        score = score_engagement(npc)
        assert score == pytest.approx(0.0)


class TestEvaluateNpc:
    def test_default_config(self):
        npc = _make_npc()
        score = evaluate_npc(npc)
        assert 0.0 <= score.weighted_total <= 1.0
        assert score.npc_id == "npc_1"
        assert score.npc_name == "Alice"

    def test_custom_config(self):
        npc = _make_npc()
        # All weight on survival
        cfg = FitnessConfig(survival=1.0, prosperity=0, social=0, goals=0, engagement=0)
        score = evaluate_npc(npc, cfg)
        assert score.weighted_total == pytest.approx(score.survival)


class TestEvaluatePopulation:
    def test_empty_population(self):
        scores, metrics = evaluate_population([])
        assert scores == []
        assert metrics.population_size == 0

    def test_multiple_npcs(self):
        npcs = [
            _make_npc("a", "Alice", health=1.0, gold=100),
            _make_npc("b", "Bob", health=0.2, gold=0),
        ]
        scores, metrics = evaluate_population(npcs)
        assert metrics.population_size == 2
        assert metrics.min_fitness <= metrics.mean_fitness <= metrics.max_fitness

    def test_struggling_and_thriving(self):
        npcs = [
            _make_npc("a", "Alice", health=1.0, energy=1.0, hunger=0.0, gold=200),
            _make_npc("b", "Bob", health=0.1, energy=0.1, hunger=0.9, gold=0),
        ]
        scores, metrics = evaluate_population(
            npcs, struggling_threshold=0.35, thriving_threshold=0.5,
        )
        assert "Alice" in metrics.thriving_npcs or "Bob" in metrics.struggling_npcs


class TestThemeConfigs:
    def test_all_themes_exist(self):
        for theme in ["default", "farming", "trading", "warzone", "social"]:
            assert theme in THEME_CONFIGS

    def test_weights_roughly_sum_to_one(self):
        for name, cfg in THEME_CONFIGS.items():
            total = cfg.survival + cfg.prosperity + cfg.social + cfg.goals + cfg.engagement
            assert total == pytest.approx(1.0, abs=0.01), f"{name} weights sum to {total}"


# ====================================================================
# OVERSEER TESTS
# ====================================================================

class TestDetectTriggers:
    def test_stagnation(self):
        prev = PopulationMetrics(mean_fitness=0.5, population_size=5)
        curr = PopulationMetrics(mean_fitness=0.505, population_size=5)
        triggers = detect_triggers(curr, prev)
        assert "stagnation" in triggers

    def test_no_stagnation_with_change(self):
        prev = PopulationMetrics(mean_fitness=0.5, population_size=5)
        curr = PopulationMetrics(mean_fitness=0.6, population_size=5)
        triggers = detect_triggers(curr, prev)
        assert "stagnation" not in triggers

    def test_imbalance(self):
        metrics = PopulationMetrics(
            min_fitness=0.1, max_fitness=0.9, population_size=5,
        )
        triggers = detect_triggers(metrics)
        assert "imbalance" in triggers

    def test_low_social(self):
        metrics = PopulationMetrics(mean_social=0.1, population_size=5)
        triggers = detect_triggers(metrics)
        assert "low_social" in triggers

    def test_low_prosperity(self):
        metrics = PopulationMetrics(mean_prosperity=0.1, population_size=5)
        triggers = detect_triggers(metrics)
        assert "low_prosperity" in triggers

    def test_mass_struggle(self):
        metrics = PopulationMetrics(
            population_size=10,
            struggling_npcs=["a", "b", "c", "d", "e"],  # 50%
        )
        triggers = detect_triggers(metrics)
        assert "mass_struggle" in triggers

    def test_no_triggers_healthy_population(self):
        metrics = PopulationMetrics(
            population_size=5,
            mean_fitness=0.7,
            min_fitness=0.5,
            max_fitness=0.8,
            mean_social=0.6,
            mean_prosperity=0.5,
            struggling_npcs=[],
        )
        triggers = detect_triggers(metrics)
        assert triggers == []


class TestParseInterventions:
    def test_single_intervention(self):
        text = (
            "TYPE: parameter_tune\n"
            "TARGET: population\n"
            "ACTION: boost_conversation_chance\n"
            "REASON: social scores are low\n"
        )
        interventions = _parse_interventions(text)
        assert len(interventions) == 1
        assert interventions[0].action == "boost_conversation_chance"

    def test_multiple_interventions(self):
        text = (
            "TYPE: parameter_tune\n"
            "TARGET: population\n"
            "ACTION: boost_conversation_chance\n"
            "REASON: low social\n"
            "\n"
            "TYPE: policy_inject\n"
            "TARGET: Alice\n"
            "ACTION: merchant\n"
            "REASON: needs gold\n"
        )
        interventions = _parse_interventions(text)
        assert len(interventions) == 2

    def test_empty_response(self):
        assert _parse_interventions("") == []
        assert _parse_interventions("No interventions needed.") == []


class TestHeuristicInterventions:
    def test_stagnation_intervention(self):
        metrics = PopulationMetrics(population_size=5)
        interventions = _heuristic_interventions(["stagnation"], metrics, [])
        actions = [i.action for i in interventions]
        assert "increase_schedule_variety" in actions

    def test_low_social_intervention(self):
        metrics = PopulationMetrics(population_size=5)
        interventions = _heuristic_interventions(["low_social"], metrics, [])
        actions = [i.action for i in interventions]
        assert "boost_conversation_chance" in actions

    def test_mass_struggle_injects_survival_mode(self):
        metrics = PopulationMetrics(population_size=5)
        interventions = _heuristic_interventions(["mass_struggle"], metrics, [])
        actions = [i.action for i in interventions]
        assert "survival_mode" in actions

    def test_imbalance_helps_struggling(self):
        metrics = PopulationMetrics(
            population_size=5,
            struggling_npcs=["Alice", "Bob"],
        )
        interventions = _heuristic_interventions(["imbalance"], metrics, [])
        assert len(interventions) == 2
        assert all(i.intervention_type == "prompt_modifier" for i in interventions)


class TestOverseer:
    def test_evaluate_no_triggers(self):
        async def _run():
            npcs = [
                _make_npc("a", "Alice", health=0.9, energy=0.9, hunger=0.1, gold=80),
            ]
            overseer = Overseer()
            report = await overseer.evaluate(npcs, game_day=1)
            assert isinstance(report, EvaluationReport)
            assert report.game_day == 1
            assert report.metrics.population_size == 1
        asyncio.new_event_loop().run_until_complete(_run())

    def test_evaluate_with_triggers(self):
        async def _run():
            # Create NPCs that will trigger low_social and low_prosperity
            npcs = [
                _make_npc("a", "Alice", health=0.5, gold=0, last_conversation_time=0),
                _make_npc("b", "Bob", health=0.5, gold=0, last_conversation_time=0),
                _make_npc("c", "Carol", health=0.5, gold=0, last_conversation_time=0),
            ]
            overseer = Overseer()
            report = await overseer.evaluate(npcs, game_day=1)
            # Should detect at least some trigger
            assert report.metrics.mean_prosperity < 0.3
        asyncio.new_event_loop().run_until_complete(_run())

    def test_history_tracking(self):
        async def _run():
            npcs = [_make_npc()]
            overseer = Overseer()
            await overseer.evaluate(npcs, game_day=1)
            await overseer.evaluate(npcs, game_day=2)
            history = overseer.get_history()
            assert len(history) == 2
        asyncio.new_event_loop().run_until_complete(_run())

    def test_set_theme(self):
        overseer = Overseer(theme="default")
        overseer.set_theme("warzone")
        assert overseer.config.survival == 0.35


# ====================================================================
# MECHANISMS TESTS
# ====================================================================

class TestPolicyTemplate:
    def test_expiry(self):
        p = PolicyTemplate(name="test", description="test", applied_day=5, duration_days=3)
        assert not p.is_expired(6)
        assert not p.is_expired(7)
        assert p.is_expired(8)

    def test_to_dict(self):
        p = POLICY_TEMPLATES["merchant"]
        d = p.to_dict()
        assert d["name"] == "merchant"
        assert "trade_priority" in d["parameter_overrides"]


class TestPromptModifier:
    def test_expiry(self):
        m = PromptModifier(text="test", applied_day=10, ttl_days=2)
        assert not m.is_expired(11)
        assert m.is_expired(12)


class TestMechanismEngine:
    def test_apply_parameter_tune_population(self):
        engine = MechanismEngine()
        npcs = [_make_npc("a", "Alice"), _make_npc("b", "Bob")]
        intervention = _make_intervention(
            action="boost_conversation_chance",
            params={"conversation_chance_boost": 0.1},
        )
        result = engine.apply_intervention(intervention, npcs, current_day=1)
        assert result is True

    def test_apply_parameter_tune_individual(self):
        engine = MechanismEngine()
        npc = _make_npc("a", "Alice", gold=50)
        intervention = _make_intervention(target="Alice", params={"gold": 10})
        result = engine.apply_intervention(intervention, [npc], current_day=1)
        assert result is True
        assert npc.gold == 60

    def test_apply_parameter_tune_unknown_target(self):
        engine = MechanismEngine()
        intervention = _make_intervention(target="Unknown")
        result = engine.apply_intervention(intervention, [], current_day=1)
        assert result is False

    def test_apply_policy_inject(self):
        engine = MechanismEngine()
        npcs = [_make_npc("a", "Alice")]
        intervention = _make_intervention(
            itype="policy_inject", target="Alice", action="merchant",
        )
        result = engine.apply_intervention(intervention, npcs, current_day=1)
        assert result is True
        policies = engine.get_active_policies("a")
        assert len(policies) == 1
        assert policies[0].name == "merchant"

    def test_apply_policy_unknown_template(self):
        engine = MechanismEngine()
        intervention = _make_intervention(
            itype="policy_inject", action="nonexistent",
        )
        result = engine.apply_intervention(intervention, [], current_day=1)
        assert result is False

    def test_apply_prompt_modifier(self):
        engine = MechanismEngine()
        npcs = [_make_npc("a", "Alice")]
        intervention = _make_intervention(
            itype="prompt_modifier", target="Alice",
            params={"modifier": "You feel energised.", "ttl_days": 3},
        )
        result = engine.apply_intervention(intervention, npcs, current_day=1)
        assert result is True
        modifiers = engine.get_active_modifiers("a")
        assert "You feel energised." in modifiers

    def test_apply_prompt_modifier_population(self):
        engine = MechanismEngine()
        npcs = [_make_npc("a", "Alice"), _make_npc("b", "Bob")]
        intervention = _make_intervention(
            itype="prompt_modifier", target="population",
            params={"modifier": "The village celebrates."},
        )
        engine.apply_intervention(intervention, npcs, current_day=1)
        assert len(engine.get_active_modifiers("a")) == 1
        assert len(engine.get_active_modifiers("b")) == 1

    def test_expire_old(self):
        engine = MechanismEngine()
        npcs = [_make_npc("a", "Alice")]
        # Inject a policy on day 1
        intervention = _make_intervention(
            itype="policy_inject", target="Alice", action="survival_mode",
        )
        engine.apply_intervention(intervention, npcs, current_day=1)
        assert len(engine.get_active_policies("a")) == 1
        # survival_mode duration = 2 days, so day 3 should expire it
        removed = engine.expire_old(current_day=3)
        assert removed == 1
        assert len(engine.get_active_policies("a")) == 0

    def test_unknown_intervention_type(self):
        engine = MechanismEngine()
        intervention = _make_intervention(itype="unknown_type")
        result = engine.apply_intervention(intervention, [], current_day=1)
        assert result is False

    def test_get_stats(self):
        engine = MechanismEngine()
        stats = engine.get_stats()
        assert stats["active_policies"] == 0
        assert stats["active_modifiers"] == 0
        assert stats["total_applied"] == 0

    def test_population_params(self):
        engine = MechanismEngine()
        npcs = [_make_npc()]
        intervention = _make_intervention(
            action="increase_schedule_variety",
            params={"variety_boost": 0.2},
        )
        engine.apply_intervention(intervention, npcs, current_day=1)
        params = engine.get_population_params()
        assert "variety_boost" in params


# ====================================================================
# GUARDRAILS TESTS
# ====================================================================

class TestParamBounds:
    def test_healthy_npc_no_violations(self):
        npc = _make_npc()
        violations = check_param_bounds(npc)
        assert violations == []

    def test_health_above_max(self):
        npc = _make_npc(health=1.5)
        violations = check_param_bounds(npc)
        names = [v.rule_name for v in violations]
        assert "param_above_max_health" in names

    def test_gold_below_min(self):
        npc = _make_npc(gold=-10)
        violations = check_param_bounds(npc)
        names = [v.rule_name for v in violations]
        assert "param_below_min_gold" in names


class TestClampNpcParams:
    def test_clamp_over_max(self):
        npc = _make_npc(health=2.0, energy=1.5)
        clamped = clamp_npc_params(npc)
        assert clamped == 2
        assert npc.health == 1.0
        assert npc.energy == 1.0

    def test_clamp_under_min(self):
        npc = _make_npc(health=-0.5)
        clamp_npc_params(npc)
        assert npc.health == 0.0

    def test_no_clamp_needed(self):
        npc = _make_npc()
        clamped = clamp_npc_params(npc)
        assert clamped == 0


class TestRateLimiter:
    def test_allows_up_to_max(self):
        rl = RateLimiter(max_per_day=3)
        for _ in range(3):
            assert rl.check(1) is True
            rl.record(1)
        assert rl.check(1) is False

    def test_new_day_resets(self):
        rl = RateLimiter(max_per_day=2)
        rl.record(1)
        rl.record(1)
        assert rl.check(1) is False
        assert rl.check(2) is True

    def test_get_remaining(self):
        rl = RateLimiter(max_per_day=5)
        rl.record(1)
        rl.record(1)
        assert rl.get_remaining(1) == 3


class TestGuardrailEngine:
    def test_allows_valid_intervention(self):
        engine = GuardrailEngine()
        npcs = [_make_npc("a", "Alice"), _make_npc("b", "Bob"), _make_npc("c", "Carol")]
        intervention = _make_intervention()
        result = engine.check_intervention(intervention, npcs, game_day=1)
        assert result.allowed is True

    def test_blocks_rate_limited(self):
        engine = GuardrailEngine()
        engine._rate_limiter = RateLimiter(max_per_day=1)
        engine._rate_limiter.record(1)
        npcs = [_make_npc()]
        intervention = _make_intervention()
        result = engine.check_intervention(intervention, npcs, game_day=1)
        assert result.allowed is False

    def test_blocks_mass_policy_small_pop(self):
        engine = GuardrailEngine()
        npcs = [_make_npc("a", "Alice")]
        intervention = _make_intervention(
            itype="policy_inject", target="population", action="merchant",
        )
        result = engine.check_intervention(intervention, npcs, game_day=1)
        assert result.allowed is False
        assert any("mass_policy" in v.rule_name for v in result.violations)

    def test_adjusts_excessive_boost(self):
        engine = GuardrailEngine()
        npcs = [_make_npc("a", "Alice"), _make_npc("b", "Bob"), _make_npc("c", "Carol")]
        intervention = _make_intervention(
            params={"conversation_chance_boost": 0.9},
        )
        result = engine.check_intervention(intervention, npcs, game_day=1)
        # Should still be allowed but with adjust violation
        assert result.allowed is True
        adjust_violations = [v for v in result.violations if v.severity == "adjust"]
        assert len(adjust_violations) > 0

    def test_filter_interventions_caps_count(self):
        engine = GuardrailEngine()
        npcs = [_make_npc("a"), _make_npc("b"), _make_npc("c")]
        # Create more than MAX_INTERVENTIONS_PER_CYCLE
        interventions = [
            _make_intervention(action=f"action_{i}")
            for i in range(MAX_INTERVENTIONS_PER_CYCLE + 3)
        ]
        filtered = engine.filter_interventions(interventions, npcs, game_day=1)
        assert len(filtered) <= MAX_INTERVENTIONS_PER_CYCLE

    def test_filter_applies_parameter_adjustments(self):
        engine = GuardrailEngine()
        npcs = [_make_npc("a"), _make_npc("b"), _make_npc("c")]
        intervention = _make_intervention(
            params={"conversation_chance_boost": 0.9},
        )
        filtered = engine.filter_interventions([intervention], npcs, game_day=1)
        assert len(filtered) == 1
        # The excessive boost should have been adjusted down
        assert filtered[0].parameters["conversation_chance_boost"] == 0.5

    def test_enforce_bounds(self):
        engine = GuardrailEngine()
        npcs = [_make_npc(health=2.0), _make_npc(health=-1.0)]
        total = engine.enforce_bounds(npcs)
        assert total == 2
        assert npcs[0].health == 1.0
        assert npcs[1].health == 0.0

    def test_add_custom_rule(self):
        engine = GuardrailEngine()

        def custom_rule(intervention, npcs):
            if intervention.action == "forbidden":
                return RuleViolation(
                    rule_name="custom_block",
                    severity="block",
                    message="Forbidden action",
                )
            return None

        engine.add_rule(custom_rule)
        npcs = [_make_npc("a"), _make_npc("b"), _make_npc("c")]
        intervention = _make_intervention(action="forbidden")
        result = engine.check_intervention(intervention, npcs, game_day=1)
        assert result.allowed is False

    def test_remove_custom_rule(self):
        engine = GuardrailEngine()

        def custom_rule(intervention, npcs):
            return None

        engine.add_rule(custom_rule)
        assert engine.remove_rule(custom_rule) is True
        assert engine.remove_rule(custom_rule) is False  # already removed

    def test_violation_log(self):
        engine = GuardrailEngine()
        npcs = [_make_npc()]
        intervention = _make_intervention(
            itype="policy_inject", target="population", action="merchant",
        )
        engine.check_intervention(intervention, npcs, game_day=1)
        log = engine.get_violation_log()
        assert len(log) > 0
        assert log[0]["day"] == 1

    def test_get_stats(self):
        engine = GuardrailEngine()
        stats = engine.get_stats()
        assert stats["total_rules"] == 3  # 3 built-in rules
        assert stats["total_violations"] == 0


class TestConflictingPolicyWarning:
    def test_hermit_warns_about_politician(self):
        engine = GuardrailEngine()
        npcs = [_make_npc("a"), _make_npc("b"), _make_npc("c")]
        intervention = _make_intervention(
            itype="policy_inject", target="Alice", action="hermit",
        )
        result = engine.check_intervention(intervention, npcs, game_day=1)
        warnings = [v for v in result.violations if v.severity == "warn"]
        assert any("politician" in v.message for v in warnings)


# ====================================================================
# INTEGRATION: OVERSEER + MECHANISMS + GUARDRAILS
# ====================================================================

class TestEvolutionIntegration:
    def test_full_cycle(self):
        """End-to-end: overseer evaluates, guardrails filter, mechanisms apply."""
        async def _run():
            npcs = [
                _make_npc("a", "Alice", gold=0, last_conversation_time=0),
                _make_npc("b", "Bob", gold=0, last_conversation_time=0),
                _make_npc("c", "Carol", gold=0, last_conversation_time=0),
            ]

            overseer = Overseer()
            guardrails = GuardrailEngine()
            mechanisms = MechanismEngine()

            # Evaluate
            report = await overseer.evaluate(npcs, game_day=1)

            # Filter through guardrails
            allowed = guardrails.filter_interventions(
                report.interventions, npcs, game_day=1,
            )

            # Apply via mechanisms
            applied_count = 0
            for intervention in allowed:
                if mechanisms.apply_intervention(intervention, npcs, current_day=1):
                    guardrails.record_applied(game_day=1)
                    applied_count += 1

            # Verify the pipeline ran
            assert report.metrics.population_size == 3
            # The exact interventions depend on trigger detection, but the pipeline should work
        asyncio.new_event_loop().run_until_complete(_run())

    def test_stagnation_triggers_variety(self):
        """Two evaluations with same fitness should trigger stagnation."""
        async def _run():
            npcs = [_make_npc("a", "Alice"), _make_npc("b", "Bob"), _make_npc("c", "Carol")]
            overseer = Overseer()

            report1 = await overseer.evaluate(npcs, game_day=1)
            report2 = await overseer.evaluate(npcs, game_day=2)

            # Second eval should detect stagnation
            assert "stagnation" in report2.triggers
            actions = [i.action for i in report2.interventions]
            assert "increase_schedule_variety" in actions
        asyncio.new_event_loop().run_until_complete(_run())
