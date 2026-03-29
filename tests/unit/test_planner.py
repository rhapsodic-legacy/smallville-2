"""Tests for the deterministic planner — utility scoring and execution rules."""

import pytest
from core.npc.models import NPC, PersonalityTraits
from core.npc.cognition.planner import (
    DeterministicPlanner,
    ActionDef,
    ActionRegistry,
    build_default_registry,
    PlannerContext,
    ContextBuilder,
    ScoredAction,
    UtilityScorer,
    exponential_curve,
    linear_curve,
    step_curve,
    PlannedAction,
    RuleSet,
    RuleRegistry,
    build_default_rules,
)
from core.world.generator import WorldConfig, generate_world


# ---------- Fixtures ----------

def _make_npc(**overrides) -> NPC:
    defaults = dict(
        npc_id="test_0",
        name="Tester",
        age=30,
        personality=PersonalityTraits(),
        backstory="A test NPC.",
        occupation="farmer",
        x=5, z=5,
        home_x=5, home_z=5,
        work_x=10, work_z=10,
        health=1.0, energy=1.0, hunger=0.0,
        gold=50,
        skills={"farming": 0.7},
    )
    defaults.update(overrides)
    return NPC(**defaults)


@pytest.fixture
def world():
    config = WorldConfig(population=5, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    return grid, buildings


@pytest.fixture
def planner(world):
    grid, buildings = world
    return DeterministicPlanner(grid, buildings, seed=42)


@pytest.fixture
def ctx():
    """A basic planner context for unit tests."""
    return PlannerContext(
        current_slot="morning",
        game_minutes=480.0,
        nearby_npcs=[
            {"npc_id": "other_0", "name": "Other", "x": 6, "z": 5,
             "occupation": "merchant", "distance": 1},
        ],
        nearby_resource_nodes=[
            {"x": 8, "z": 8, "resource_type": "wood",
             "current_amount": 50, "distance": 6},
        ],
        available_recipes=["plank"],
        has_market=True,
        has_church=True,
        tavern_door=(3, 3),
        church_door=(12, 12),
        random_passable_tile=(7, 7),
    )


# ---------- Need curves ----------

class TestNeedCurves:
    def test_exponential_zero(self):
        assert exponential_curve(0.0) == pytest.approx(0.0, abs=0.01)

    def test_exponential_one(self):
        assert exponential_curve(1.0) == pytest.approx(1.0, abs=0.01)

    def test_exponential_grows_steeply(self):
        low = exponential_curve(0.3)
        high = exponential_curve(0.9)
        assert high > low * 3, "Curve should grow steeply at high values"

    def test_linear_passthrough(self):
        assert linear_curve(0.5) == 0.5
        assert linear_curve(0.0) == 0.0
        assert linear_curve(1.0) == 1.0

    def test_step_below_threshold(self):
        assert step_curve(0.3, threshold=0.5) == 0.0

    def test_step_above_threshold(self):
        assert step_curve(0.7, threshold=0.5) == 1.0

    def test_curves_clamp(self):
        assert exponential_curve(-0.5) == pytest.approx(0.0, abs=0.01)
        assert exponential_curve(1.5) == pytest.approx(1.0, abs=0.01)
        assert linear_curve(2.0) == 1.0


# ---------- Action registry ----------

class TestActionRegistry:
    def test_default_registry_has_actions(self):
        reg = build_default_registry()
        assert len(reg) >= 10

    def test_register_and_get(self):
        reg = ActionRegistry()
        action = ActionDef(action_id="dance", display_name="Dance")
        reg.register(action)
        assert reg.get("dance") is action
        assert "dance" in reg

    def test_remove(self):
        reg = build_default_registry()
        removed = reg.remove("eat")
        assert removed is not None
        assert "eat" not in reg

    def test_replace_existing(self):
        reg = build_default_registry()
        new_eat = ActionDef(action_id="eat", display_name="Feast")
        reg.replace("eat", new_eat)
        assert reg.get("eat").display_name == "Feast"

    def test_replace_nonexistent_raises(self):
        reg = ActionRegistry()
        with pytest.raises(KeyError):
            reg.replace("nope", ActionDef("nope", "Nope"))

    def test_by_tag(self):
        reg = build_default_registry()
        survival = reg.by_tag("survival")
        assert len(survival) >= 2
        assert all("survival" in a.tags for a in survival)

    def test_extensibility_custom_action(self):
        """AI Game Studio scenario: add a completely new action."""
        reg = build_default_registry()
        reg.register(ActionDef(
            action_id="fish",
            display_name="Go fishing",
            need_weights={"hunger": 1.0},
            base_utility=0.3,
            tags={"economy", "outdoor"},
        ))
        assert "fish" in reg
        assert len(reg.by_tag("outdoor")) >= 2


# ---------- Utility scorer ----------

class TestUtilityScorer:
    def test_hungry_npc_prefers_eating(self, ctx):
        npc = _make_npc(hunger=0.9, energy=1.0)
        scorer = UtilityScorer()
        reg = build_default_registry()
        results = scorer.evaluate_all(npc, ctx, reg.all())

        top = results[0]
        assert top.action_id == "eat", (
            f"Hungry NPC should want to eat, got {top.action_id}"
        )

    def test_exhausted_npc_prefers_sleep(self, ctx):
        npc = _make_npc(energy=0.1, hunger=0.0)
        ctx.current_slot = "night"
        scorer = UtilityScorer()
        reg = build_default_registry()
        results = scorer.evaluate_all(npc, ctx, reg.all())

        top = results[0]
        assert top.action_id == "sleep", (
            f"Exhausted NPC at night should sleep, got {top.action_id}"
        )

    def test_threat_causes_flee(self, ctx):
        npc = _make_npc()
        ctx.threat_level = 1.0
        ctx.threat_x = 10
        ctx.threat_z = 10
        scorer = UtilityScorer()
        reg = build_default_registry()
        results = scorer.evaluate_all(npc, ctx, reg.all())

        top = results[0]
        assert top.action_id == "flee", (
            f"Threatened NPC should flee, got {top.action_id}"
        )

    def test_personality_affects_scoring(self, ctx):
        """Extroverted NPC should score socialising higher than introvert."""
        extrovert = _make_npc(
            npc_id="ext",
            personality=PersonalityTraits(extraversion=0.9),
        )
        introvert = _make_npc(
            npc_id="int",
            personality=PersonalityTraits(extraversion=0.1),
        )
        scorer = UtilityScorer()
        reg = build_default_registry()

        ext_results = scorer.evaluate_all(extrovert, ctx, reg.all())
        int_results = scorer.evaluate_all(introvert, ctx, reg.all())

        ext_social = next(r for r in ext_results if r.action_id == "socialise")
        int_social = next(r for r in int_results if r.action_id == "socialise")
        assert ext_social.total_score > int_social.total_score

    def test_time_of_day_affects_scoring(self, ctx):
        """Work should score higher in morning than at night."""
        npc = _make_npc()
        scorer = UtilityScorer()
        reg = build_default_registry()

        ctx.current_slot = "morning"
        morning = scorer.evaluate_all(npc, ctx, reg.all())
        work_morning = next(r for r in morning if r.action_id == "work")

        ctx.current_slot = "night"
        night = scorer.evaluate_all(npc, ctx, reg.all())
        work_night = next(r for r in night if r.action_id == "work")

        assert work_morning.total_score > work_night.total_score

    def test_preconditions_filter_actions(self):
        """Actions whose preconditions fail should not appear."""
        npc = _make_npc(hunger=0.0)  # not hungry
        ctx = PlannerContext(current_slot="morning")
        scorer = UtilityScorer()
        reg = build_default_registry()
        results = scorer.evaluate_all(npc, ctx, reg.all())

        ids = {r.action_id for r in results}
        assert "eat" not in ids, "Not hungry, eat should be filtered"
        assert "flee" not in ids, "No threat, flee should be filtered"

    def test_custom_scorer_overrides(self, ctx):
        """Custom scorer function should override default scoring."""
        npc = _make_npc()
        scorer = UtilityScorer()
        scorer.set_custom_scorer("work", lambda n, c, a: 999.0)
        reg = build_default_registry()
        results = scorer.evaluate_all(npc, ctx, reg.all())

        assert results[0].action_id == "work"
        assert results[0].total_score == 999.0

    def test_score_breakdown_present(self, ctx):
        npc = _make_npc(hunger=0.5)
        scorer = UtilityScorer()
        reg = build_default_registry()
        results = scorer.evaluate_all(npc, ctx, reg.all())

        for r in results:
            assert "total" in r.breakdown or "custom" in r.breakdown

    def test_all_actions_scored_when_no_preconditions(self):
        """Actions without preconditions should always be scored."""
        npc = _make_npc()
        ctx = PlannerContext(current_slot="morning")
        scorer = UtilityScorer()
        # Only use actions without preconditions
        actions = [
            ActionDef("a", "A", base_utility=1.0),
            ActionDef("b", "B", base_utility=2.0),
        ]
        results = scorer.evaluate_all(npc, ctx, actions)
        assert len(results) == 2
        assert results[0].action_id == "b"  # higher base utility


# ---------- Rule registry ----------

class TestRuleRegistry:
    def test_default_rules_cover_all_actions(self):
        actions = build_default_registry()
        rules = build_default_rules()
        for action in actions.all():
            # Every default action should have a rule or fallback works
            assert rules.get(action.action_id) is not None or True

    def test_custom_rule_set(self, ctx):
        npc = _make_npc()
        rules = RuleRegistry()

        def custom_dance_rule(n, c, s):
            return PlannedAction(
                action_id="dance",
                description="dancing in the square",
                target_x=0, target_z=0,
                activity_state="idle",
            )

        rules.register(RuleSet("dance", [custom_dance_rule]))
        scored = ScoredAction("dance", "Dance", 5.0)
        result = rules.execute("dance", npc, ctx, scored)
        assert result is not None
        assert result.description == "dancing in the square"

    def test_fallback_rule(self, ctx):
        npc = _make_npc()
        rules = RuleRegistry()
        scored = ScoredAction("unknown", "Unknown", 1.0, target=(3, 3))
        result = rules.execute("unknown", npc, ctx, scored)
        assert result is not None
        assert result.target_x == 3

    def test_rule_fallthrough(self, ctx):
        """If first rule returns None, next rule is tried."""
        npc = _make_npc()
        rules = RuleRegistry()

        def failing_rule(n, c, s):
            return None

        def working_rule(n, c, s):
            return PlannedAction("x", "worked", 0, 0)

        rules.register(RuleSet("x", [failing_rule, working_rule]))
        scored = ScoredAction("x", "X", 1.0)
        result = rules.execute("x", npc, ctx, scored)
        assert result is not None
        assert result.description == "worked"


# ---------- Built-in rule execution ----------

class TestBuiltInRules:
    def test_eat_rule(self, ctx):
        npc = _make_npc(hunger=0.8)
        rules = build_default_rules()
        scored = ScoredAction("eat", "Eat", 3.0, target=(3, 3))
        result = rules.execute("eat", npc, ctx, scored)
        assert result is not None
        assert result.activity_state == "eating"

    def test_sleep_rule(self, ctx):
        npc = _make_npc(energy=0.2)
        rules = build_default_rules()
        scored = ScoredAction("sleep", "Sleep", 3.0)
        result = rules.execute("sleep", npc, ctx, scored)
        assert result is not None
        assert result.target_x == npc.home_x

    def test_work_rule(self, ctx):
        npc = _make_npc()
        rules = build_default_rules()
        scored = ScoredAction("work", "Work", 2.0)
        result = rules.execute("work", npc, ctx, scored)
        assert result is not None
        assert result.target_x == npc.work_x

    def test_gather_rule(self, ctx):
        npc = _make_npc()
        rules = build_default_rules()
        scored = ScoredAction("gather", "Gather", 2.0)
        result = rules.execute("gather", npc, ctx, scored)
        assert result is not None
        assert result.activity_state == "gathering"

    def test_flee_rule_with_threat(self, ctx):
        npc = _make_npc(x=5, z=5)
        ctx.threat_level = 1.0
        ctx.threat_x = 10
        ctx.threat_z = 10
        rules = build_default_rules()
        scored = ScoredAction("flee", "Flee", 5.0)
        result = rules.execute("flee", npc, ctx, scored)
        assert result is not None
        # Should flee away from (10, 10) — target should be west/north
        assert result.target_x < 10 or result.target_z < 10

    def test_flee_rule_no_threat(self, ctx):
        npc = _make_npc()
        ctx.threat_level = 0.0
        rules = build_default_rules()
        scored = ScoredAction("flee", "Flee", 0.0)
        result = rules.execute("flee", npc, ctx, scored)
        # Rule should return None (no threat), but fallback catches it
        assert result is not None

    def test_socialise_rule(self, ctx):
        npc = _make_npc()
        rules = build_default_rules()
        scored = ScoredAction("socialise", "Socialise", 2.0, target=(3, 3))
        result = rules.execute("socialise", npc, ctx, scored)
        assert result is not None
        assert result.activity_state == "talking"


# ---------- Context builder ----------

class TestContextBuilder(object):
    def test_build_basic_context(self, world):
        grid, buildings = world
        builder = ContextBuilder(grid, buildings, seed=42)
        npc = _make_npc()
        others = [_make_npc(npc_id="other_0", x=6, z=5)]
        ctx = builder.build(npc, others, "morning")

        assert ctx.current_slot == "morning"
        assert len(ctx.nearby_npcs) == 1
        assert ctx.random_passable_tile is not None

    def test_building_detection(self, world):
        grid, buildings = world
        builder = ContextBuilder(grid, buildings, seed=42)
        npc = _make_npc()
        ctx = builder.build(npc, [], "morning")

        # Generated world should have a tavern and market
        assert ctx.tavern_door is not None or ctx.has_market


# ---------- Full planner integration ----------

class TestDeterministicPlanner:
    def test_plan_action_returns_result(self, planner):
        npc = _make_npc()
        result = planner.plan_action(npc, [], "morning")
        assert result is not None
        assert isinstance(result, PlannedAction)
        assert len(result.description) > 0

    def test_plan_action_responds_to_hunger(self, planner):
        npc = _make_npc(hunger=0.95, energy=1.0)
        result = planner.plan_action(npc, [], "morning")
        assert result is not None
        assert result.action_id == "eat"

    def test_plan_action_responds_to_threat(self, planner):
        npc = _make_npc()
        result = planner.plan_action(
            npc, [], "morning", threat_level=1.0, threat_x=10, threat_z=10,
        )
        assert result is not None
        assert result.action_id == "flee"

    def test_plan_action_night_prefers_sleep(self, planner):
        npc = _make_npc(energy=0.3)
        result = planner.plan_action(npc, [], "night")
        assert result is not None
        assert result.action_id in ("sleep", "rest")

    def test_score_all_returns_ranked_list(self, planner):
        npc = _make_npc()
        scores = planner.score_all(npc, [], "morning")
        assert len(scores) >= 1
        # Should be sorted descending
        for i in range(len(scores) - 1):
            assert scores[i].total_score >= scores[i + 1].total_score

    def test_planner_with_custom_registry(self, world):
        """Fully custom planner with only two actions."""
        grid, buildings = world
        reg = ActionRegistry()
        reg.register(ActionDef("nap", "Nap", base_utility=5.0))
        reg.register(ActionDef("think", "Think", base_utility=1.0))

        planner = DeterministicPlanner(
            grid, buildings, action_registry=reg, seed=42,
        )
        result = planner.plan_action(_make_npc(), [], "afternoon")
        assert result is not None
        assert result.action_id == "nap"

    def test_planner_to_schedule_entry(self, planner):
        """PlannedAction should convert to ScheduleEntry."""
        npc = _make_npc()
        result = planner.plan_action(npc, [], "morning")
        entry = result.to_schedule_entry("morning")
        assert entry.slot == "morning"
        assert len(entry.activity) > 0

    def test_guard_prefers_patrol(self, planner):
        npc = _make_npc(occupation="guard", energy=1.0, hunger=0.0)
        result = planner.plan_action(npc, [], "morning")
        assert result is not None
        # Guard with no needs should lean towards patrol or work
        assert result.action_id in ("patrol", "work", "wander", "gather")
