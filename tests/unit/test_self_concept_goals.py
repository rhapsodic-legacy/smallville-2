"""Tests for the self_concept → long_term_goals mapper and scorer bias."""

import pytest

from core.npc.cognition.goal_mapper import (
    GOAL_ACTION_BONUS, GOAL_DEFINITIONS, GOAL_TAG_BONUS,
    GoalAffinity, aggregate_boost_actions, aggregate_boost_tags,
    propose_goals, sync_npc_goals,
)
from core.npc.cognition.planner import (
    DeterministicPlanner, UtilityScorer, build_default_registry,
)
from core.npc.cognition.planner.actions import ActionDef
from core.npc.cognition.planner.context import PlannerContext
from core.npc.models import NPC, PersonalityTraits


def _npc(**kwargs) -> NPC:
    return NPC(
        npc_id=kwargs.pop("npc_id", "alice"),
        name=kwargs.pop("name", "Alice"),
        age=30,
        personality=PersonalityTraits(),
        backstory="test",
        occupation=kwargs.pop("occupation", "farmer"),
        **kwargs,
    )


# ---------- propose_goals ----------

class TestProposeGoals:
    def test_empty_self_concept_returns_empty(self):
        assert propose_goals({}) == []

    def test_role_king_proposed_at_high_confidence(self):
        proposals = propose_goals({"role:king": 0.9})
        assert len(proposals) == 1
        assert "royal court" in proposals[0].text.lower()
        assert "construct" in proposals[0].boost_actions
        assert "socialise" in proposals[0].boost_actions

    def test_below_confidence_floor_skipped(self):
        definition = GOAL_DEFINITIONS["role:king"]
        assert propose_goals({"role:king": definition.min_confidence - 0.01}) == []

    def test_unknown_prefix_skipped(self):
        # "skill" has no definition — should produce nothing.
        assert propose_goals({"skill:weaving": 0.9}) == []

    def test_prefix_fallback_for_generic_enemy(self):
        proposals = propose_goals({"enemy_of:bran_1": 0.8})
        assert len(proposals) == 1
        assert "bran 1" in proposals[0].text
        assert "undermine" in proposals[0].text.lower()

    def test_target_underscores_become_spaces(self):
        proposals = propose_goals({"friend_of:old_merchant": 0.8})
        assert "old merchant" in proposals[0].text

    def test_multiple_beliefs_multiple_proposals(self):
        proposals = propose_goals({
            "role:king": 0.9,
            "enemy_of:bran_1": 0.8,
            "helped:town": 0.7,
        })
        texts = [p.text for p in proposals]
        assert len(proposals) == 3
        assert any("royal court" in t.lower() for t in texts)
        assert any("undermine" in t.lower() for t in texts)
        assert any("defend" in t.lower() for t in texts)

    def test_exact_key_wins_over_prefix(self):
        # role:king has an explicit definition distinct from the role:
        # prefix fallback. Confirm exact key is chosen.
        proposals = propose_goals({"role:king": 0.9})
        assert "construct" in proposals[0].boost_actions
        # The prefix fallback only boosts work + socialise; king adds
        # patrol + trade too.
        assert "patrol" in proposals[0].boost_actions


# ---------- sync_npc_goals ----------

class TestSyncNpcGoals:
    def test_strong_belief_adds_derived_goal(self):
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        added, removed = sync_npc_goals(npc)
        assert any("royal court" in g.lower() for g in npc.long_term_goals)
        assert len(added) == 1 and removed == []

    def test_hand_authored_goals_preserved(self):
        npc = _npc()
        npc.long_term_goals = ["Master the craft of weapon-smithing"]
        npc.self_concept["role:king"] = 0.9
        sync_npc_goals(npc)
        assert "Master the craft of weapon-smithing" in npc.long_term_goals

    def test_decayed_belief_removes_goal(self):
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        sync_npc_goals(npc)
        king_goal = next(
            g for g in npc.long_term_goals if "royal court" in g.lower()
        )
        assert king_goal in npc.goal_affinities

        # Belief decays below the confidence floor.
        npc.self_concept["role:king"] = 0.2
        added, removed = sync_npc_goals(npc)
        assert king_goal not in npc.long_term_goals
        assert king_goal not in npc.goal_affinities
        assert king_goal in removed and added == []

    def test_affinity_carries_boost_actions(self):
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        sync_npc_goals(npc)
        king_goal = next(
            g for g in npc.long_term_goals if "royal court" in g.lower()
        )
        aff = npc.goal_affinities[king_goal]
        assert isinstance(aff, GoalAffinity)
        assert "construct" in aff.boost_actions

    def test_repeated_sync_idempotent(self):
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        added1, _ = sync_npc_goals(npc)
        added2, removed2 = sync_npc_goals(npc)
        assert added1 and not added2 and not removed2


# ---------- Aggregation helpers ----------

class TestAggregators:
    def test_aggregate_unions_across_goals(self):
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        npc.self_concept["enemy_of:bran_1"] = 0.8
        sync_npc_goals(npc)

        actions = aggregate_boost_actions(npc)
        tags = aggregate_boost_tags(npc)
        assert "construct" in actions  # from king
        assert "patrol" in actions     # from king + enemy_of
        assert "social" in tags

    def test_empty_when_no_derived_goals(self):
        npc = _npc()
        assert aggregate_boost_actions(npc) == set()
        assert aggregate_boost_tags(npc) == set()


# ---------- Utility scorer bias ----------

def _bare_ctx() -> PlannerContext:
    """A minimal context — no buildings, no nearby NPCs, default slot."""
    return PlannerContext(current_slot="afternoon")


class TestUtilityScorerBias:
    def test_goal_action_bonus_applied(self):
        scorer = UtilityScorer()
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        sync_npc_goals(npc)

        construct = ActionDef(
            action_id="construct", display_name="Construct", tags={"economy"},
        )
        scored, breakdown = scorer.score_action(
            npc, _bare_ctx(), construct, needs={},
        )
        assert breakdown["goal_affinity"] >= GOAL_ACTION_BONUS
        assert scored >= GOAL_ACTION_BONUS

    def test_tag_bonus_applied_without_action_match(self):
        scorer = UtilityScorer()
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        sync_npc_goals(npc)

        # "community" is a king boost_tag but no action_id match.
        action = ActionDef(
            action_id="help_neighbour",
            display_name="Help a neighbour",
            tags={"community"},
        )
        _, breakdown = scorer.score_action(
            npc, _bare_ctx(), action, needs={},
        )
        assert breakdown["goal_affinity"] == pytest.approx(GOAL_TAG_BONUS)

    def test_no_bonus_when_no_goals(self):
        scorer = UtilityScorer()
        npc = _npc()  # empty self_concept
        action = ActionDef(
            action_id="construct", display_name="Construct", tags={"economy"},
        )
        _, breakdown = scorer.score_action(
            npc, _bare_ctx(), action, needs={},
        )
        assert breakdown["goal_affinity"] == 0.0

    def test_king_prefers_construct_over_gather(self):
        """King's affinity for construct should beat gather, which has
        no king-aligned tags."""
        scorer = UtilityScorer()
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        sync_npc_goals(npc)

        ctx = PlannerContext(
            current_slot="afternoon",
            nearby_resource_nodes=[
                {"x": 1, "z": 1, "resource_type": "wheat",
                 "current_amount": 5, "distance": 1},
            ],
            construction_sites=[
                {"site_id": "c1", "x": 5, "z": 5, "distance": 5,
                 "blueprint_id": "church", "progress": 10,
                 "needs_wood": 10, "needs_stone": 5, "needs_labour": 5},
            ],
        )
        registry = build_default_registry()
        scored = scorer.evaluate_all(npc, ctx, registry.all())
        by_id = {s.action_id: s for s in scored}
        assert by_id["construct"].total_score > by_id["gather"].total_score

    def test_farmer_prefers_gather_over_wander(self):
        """Farmer's gather-affinity should outrank an unrelated wander."""
        scorer = UtilityScorer()
        npc = _npc()
        npc.self_concept["role:farmer"] = 0.9
        sync_npc_goals(npc)

        ctx = PlannerContext(
            current_slot="afternoon",
            nearby_resource_nodes=[
                {"x": 1, "z": 1, "resource_type": "wheat",
                 "current_amount": 5, "distance": 1},
            ],
            random_passable_tile=(3, 3),
        )
        registry = build_default_registry()
        scored = scorer.evaluate_all(npc, ctx, registry.all())
        by_id = {s.action_id: s for s in scored}
        assert by_id["gather"].total_score > by_id["wander"].total_score

    def test_king_and_farmer_top_actions_diverge(self):
        """Same action pool, different self_concept → different top picks."""
        scorer = UtilityScorer()
        king = _npc(npc_id="king1", name="King")
        king.self_concept["role:king"] = 0.9
        sync_npc_goals(king)

        farmer = _npc(npc_id="farmer1", name="Farmer")
        farmer.self_concept["role:farmer"] = 0.9
        sync_npc_goals(farmer)

        ctx = PlannerContext(
            current_slot="afternoon",
            nearby_resource_nodes=[
                {"x": 1, "z": 1, "resource_type": "wheat",
                 "current_amount": 5, "distance": 1},
            ],
            construction_sites=[
                {"site_id": "c1", "x": 5, "z": 5, "distance": 5,
                 "blueprint_id": "church", "progress": 10,
                 "needs_wood": 10, "needs_stone": 5, "needs_labour": 5},
            ],
            random_passable_tile=(3, 3),
        )
        registry = build_default_registry()

        king_scored = scorer.evaluate_all(king, ctx, registry.all())
        farmer_scored = scorer.evaluate_all(farmer, ctx, registry.all())

        assert king_scored[0].action_id in {
            "construct", "socialise", "patrol", "trade",
        }
        assert farmer_scored[0].action_id in {"work", "gather"}


# ---------- NPCManager integration ----------

class TestManagerIntegration:
    def _mgr_with_alice(self):
        from core.npc.manager import NPCManager
        from core.npc.llm_client import MockProvider
        from core.world.generator import WorldConfig, generate_world

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=42,
        )
        mgr.spawn_population(2)
        return mgr, mgr.npcs[0]

    def test_claim_injection_syncs_goals(self):
        from core.memory.reflection import IdentityClaim

        mgr, npc = self._mgr_with_alice()
        # Two strong claims, each past the 0.5 floor.
        for _ in range(3):
            mgr._inject_self_concept_delta(
                npc,
                IdentityClaim(
                    key="role:king", confidence_delta=0.4,
                    source_text="You are a king.", speaker="Bran",
                ),
                current_minutes=0.0,
            )
        assert npc.self_concept["role:king"] >= 0.5
        assert any("royal court" in g.lower() for g in npc.long_term_goals)
        assert any(
            "royal court" in text.lower()
            for text in npc.goal_affinities
        )
