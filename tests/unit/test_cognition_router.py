"""Tests for the cognition router — budget, policy, and routing decisions."""

import pytest
from core.npc.models import NPC, PersonalityTraits
from core.npc.cognition.router import (
    CognitionRouter,
    Route,
    RouteDecision,
    TokenBudget,
    CognitionPolicy,
    AutoConfig,
    ROUTE_LLM,
    ROUTE_DETERMINISTIC,
    ROUTE_AUTO,
    policy_all_llm,
    policy_all_deterministic,
    policy_conversations_only,
    policy_local_llm,
)
from core.npc.cognition.router.budget import BudgetSnapshot


def _make_npc(**overrides) -> NPC:
    defaults = dict(
        npc_id="test_0",
        name="Tester",
        age=30,
        personality=PersonalityTraits(),
        backstory="Test NPC.",
        occupation="farmer",
        x=5, z=5,
        home_x=5, home_z=5,
        work_x=10, work_z=10,
    )
    defaults.update(overrides)
    return NPC(**defaults)


# ---------- Token budget ----------

class TestTokenBudget:
    def test_initial_state(self):
        b = TokenBudget(daily_limit=1000)
        assert b.tokens_remaining == 1000
        assert b.budget_pressure == 0.0

    def test_spend_reduces_remaining(self):
        b = TokenBudget(daily_limit=1000)
        b.record_spend(300, "test")
        assert b.tokens_remaining == 700
        assert b.budget_pressure == pytest.approx(0.3)

    def test_can_spend_respects_limit(self):
        b = TokenBudget(daily_limit=1000, reserve_fraction=0.2)
        assert b.can_spend(800)
        assert not b.can_spend(801)  # exceeds unreserved (800)

    def test_priority_can_dip_into_reserve(self):
        b = TokenBudget(daily_limit=1000, reserve_fraction=0.2)
        assert b.can_spend(900, is_priority=True)
        assert not b.can_spend(900, is_priority=False)

    def test_unlimited_budget(self):
        b = TokenBudget(daily_limit=0)
        assert b.can_spend(999999)
        assert b.budget_pressure == 0.0

    def test_reset_clears_state(self):
        b = TokenBudget(daily_limit=1000)
        b.record_spend(500, "test")
        b.reset()
        assert b.tokens_remaining == 1000
        assert b._calls_total == 0

    def test_concurrent_tracking(self):
        b = TokenBudget(max_concurrent=2)
        b.begin_call()
        b.begin_call()
        assert not b.can_call()
        b.end_call()
        assert b.can_call()

    def test_snapshot(self):
        b = TokenBudget(daily_limit=1000)
        b.record_spend(300, "test")
        snap = b.get_snapshot()
        assert isinstance(snap, BudgetSnapshot)
        assert snap.tokens_used == 300
        assert snap.tokens_remaining == 700

    def test_per_purpose_tracking(self):
        b = TokenBudget(daily_limit=10000)
        b.record_spend(100, "conversation")
        b.record_spend(50, "schedule")
        b.record_spend(100, "conversation")
        stats = b.get_stats()
        assert stats["by_purpose"]["conversation"] == 200
        assert stats["by_purpose"]["schedule"] == 50


# ---------- Cognition policy ----------

class TestCognitionPolicy:
    def test_default_policy(self):
        p = CognitionPolicy()
        assert p.get_mode("conversation") == ROUTE_LLM
        assert p.get_mode("flee") == ROUTE_DETERMINISTIC
        assert p.get_mode("daily_schedule") == ROUTE_AUTO

    def test_set_mode(self):
        p = CognitionPolicy()
        p.set_mode("conversation", ROUTE_DETERMINISTIC)
        assert p.get_mode("conversation") == ROUTE_DETERMINISTIC

    def test_set_invalid_mode_raises(self):
        p = CognitionPolicy()
        with pytest.raises(ValueError):
            p.set_mode("conversation", "magic")

    def test_unknown_type_uses_default(self):
        p = CognitionPolicy(default_mode=ROUTE_DETERMINISTIC)
        assert p.get_mode("custom_type") == ROUTE_DETERMINISTIC

    def test_serialisation_roundtrip(self):
        p = CognitionPolicy(
            priority_npcs={"hero_0", "villain_0"},
            token_budget_daily=100_000,
        )
        data = p.to_dict()
        p2 = CognitionPolicy.from_dict(data)
        assert p2.token_budget_daily == 100_000
        assert "hero_0" in p2.priority_npcs

    def test_preset_all_llm(self):
        p = policy_all_llm()
        assert p.get_mode("flee") == ROUTE_LLM

    def test_preset_all_deterministic(self):
        p = policy_all_deterministic()
        assert p.get_mode("conversation") == ROUTE_DETERMINISTIC
        assert p.token_budget_daily == 0

    def test_preset_conversations_only(self):
        p = policy_conversations_only()
        assert p.get_mode("conversation") == ROUTE_LLM
        assert p.get_mode("daily_schedule") == ROUTE_DETERMINISTIC

    def test_preset_local_llm(self):
        p = policy_local_llm(max_concurrent=3)
        assert p.max_concurrent_llm_calls == 3
        assert p.token_budget_daily == 0


# ---------- Cognition router ----------

class TestCognitionRouter:
    def test_deterministic_mode_always_deterministic(self):
        policy = CognitionPolicy()
        policy.set_mode("flee", ROUTE_DETERMINISTIC)
        router = CognitionRouter(policy=policy)
        npc = _make_npc()
        decision = router.route(npc, "flee")
        assert decision.route == Route.DETERMINISTIC

    def test_llm_mode_routes_llm(self):
        policy = CognitionPolicy()
        policy.set_mode("conversation", ROUTE_LLM)
        router = CognitionRouter(policy=policy)
        npc = _make_npc()
        decision = router.route(npc, "conversation")
        assert decision.route == Route.LLM

    def test_llm_mode_falls_back_when_budget_empty(self):
        policy = CognitionPolicy(token_budget_daily=100)
        policy.set_mode("conversation", ROUTE_LLM)
        router = CognitionRouter(policy=policy)
        router.budget.record_spend(100, "test")
        npc = _make_npc()
        decision = router.route(npc, "conversation", estimated_tokens=50)
        assert decision.route == Route.DETERMINISTIC
        assert "budget" in decision.reason

    def test_priority_npc_gets_llm(self):
        policy = CognitionPolicy(priority_npcs={"hero_0"})
        router = CognitionRouter(policy=policy)
        npc = _make_npc(npc_id="hero_0")
        decision = router.route(npc, "flee")  # normally deterministic
        assert decision.route == Route.LLM
        assert "priority" in decision.reason

    def test_auto_routes_based_on_score(self):
        policy = CognitionPolicy()
        policy.set_mode("daily_schedule", ROUTE_AUTO)
        policy.auto_config.llm_threshold = 0.3
        router = CognitionRouter(policy=policy)
        # NPC near focus point = high proximity = high score
        npc = _make_npc(x=0, z=0)
        decision = router.route(npc, "daily_schedule", focus_x=0, focus_z=0)
        assert decision.route == Route.LLM

    def test_auto_downgrades_under_pressure(self):
        policy = CognitionPolicy(auto_downgrade_threshold=5)
        policy.set_mode("reaction", ROUTE_AUTO)
        router = CognitionRouter(policy=policy)
        npc = _make_npc()
        # Simulate high scene pressure
        decisions = [(npc, "reaction") for _ in range(10)]
        results = router.route_batch(decisions)
        deterministic_count = sum(
            1 for r in results if r.route == Route.DETERMINISTIC
        )
        assert deterministic_count >= 8  # most should downgrade

    def test_auto_budget_pressure_reduces_llm(self):
        """As budget depletes, auto decisions shift to deterministic."""
        policy = CognitionPolicy(token_budget_daily=1000)
        policy.set_mode("daily_schedule", ROUTE_AUTO)
        policy.auto_config.llm_threshold = 0.3
        router = CognitionRouter(policy=policy)

        npc = _make_npc(x=0, z=0)
        # Fresh budget
        d1 = router.route(npc, "daily_schedule", focus_x=0, focus_z=0)
        # Deplete budget to 90%
        router.budget.record_spend(900, "test")
        d2 = router.route(npc, "daily_schedule", focus_x=0, focus_z=0)
        # Second decision should be more likely deterministic
        # (budget_pressure = 0.9 → score multiplied by 0.1)
        assert d2.route == Route.DETERMINISTIC

    def test_stats_tracking(self):
        router = CognitionRouter()
        npc = _make_npc()
        router.route(npc, "flee")
        router.route(npc, "conversation")
        stats = router.get_stats()
        assert stats["total_decisions"] == 2
        assert stats["llm_decisions"] + stats["deterministic_decisions"] == 2

    def test_runtime_policy_swap(self):
        router = CognitionRouter()
        npc = _make_npc()
        d1 = router.route(npc, "conversation")
        assert d1.route == Route.LLM

        router.set_policy(policy_all_deterministic())
        d2 = router.route(npc, "conversation")
        assert d2.route == Route.DETERMINISTIC

    def test_runtime_route_change(self):
        router = CognitionRouter()
        npc = _make_npc()
        router.set_route("conversation", ROUTE_DETERMINISTIC)
        d = router.route(npc, "conversation")
        assert d.route == Route.DETERMINISTIC

    def test_add_remove_priority_npc(self):
        router = CognitionRouter()
        npc = _make_npc(npc_id="special")
        router.add_priority_npc("special")
        d = router.route(npc, "flee")
        assert d.route == Route.LLM

        router.remove_priority_npc("special")
        d2 = router.route(npc, "flee")
        assert d2.route == Route.DETERMINISTIC

    def test_custom_importance_scorer(self):
        """Pluggable importance scorer should override default."""
        def always_important(npc, dt, fx, fz, nov):
            return 1.0

        router = CognitionRouter()
        router.set_importance_scorer(always_important)
        npc = _make_npc()
        router.set_route("reaction", ROUTE_AUTO)
        d = router.route(npc, "reaction")
        assert d.route == Route.LLM

    def test_record_llm_spend(self):
        router = CognitionRouter()
        router.record_llm_spend(500, "conversation")
        assert router.budget._tokens_used == 500


# ---------- Mistral provider ----------

class TestMistralProvider:
    def test_import(self):
        from core.npc.mistral_provider import MistralProvider
        provider = MistralProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.npc_model == "mistral-small-latest"

    def test_implements_llm_provider(self):
        from core.npc.mistral_provider import MistralProvider
        from core.npc.llm_client import LLMProvider
        assert issubclass(MistralProvider, LLMProvider)

    def test_custom_models(self):
        from core.npc.mistral_provider import MistralProvider
        provider = MistralProvider(
            api_key="test",
            npc_model="custom-npc",
            overseer_model="custom-overseer",
        )
        assert provider.npc_model == "custom-npc"
        assert provider.overseer_model == "custom-overseer"
