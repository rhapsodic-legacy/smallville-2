"""
Tests for Phase 5: Relationships and Social.

Covers sentiment dimensions, event impact system, faction structures,
and integration with existing memory/cognition systems.
"""

import pytest

from core.relationships.sentiment import (
    Sentiment, SentimentTracker, DIMENSIONS, DIMENSION_MIN, DIMENSION_MAX,
)
from core.relationships.structures import (
    Faction, FactionManager, FactionMember, FactionRelation, FactionRole,
    Agreement,
)
from core.events.impact import (
    EventImpactSystem, EventRule, EventEffect, EventCondition,
    GameEvent, DEFAULT_RULES,
)


# ---------- Sentiment ----------


class TestSentiment:
    """Unit tests for the Sentiment dataclass."""

    def test_default_is_neutral(self):
        s = Sentiment(npc_from="a", npc_to="b")
        assert s.is_default()
        assert s.overall_disposition() == 0.0

    def test_get_set_dimensions(self):
        s = Sentiment(npc_from="a", npc_to="b")
        for dim in DIMENSIONS:
            s.set(dim, 50.0)
            assert s.get(dim) == 50.0

    def test_clamping(self):
        s = Sentiment(npc_from="a", npc_to="b")
        s.set("trust", 999)
        assert s.get("trust") == DIMENSION_MAX
        s.set("trust", -999)
        assert s.get("trust") == DIMENSION_MIN

    def test_overall_disposition(self):
        s = Sentiment(npc_from="a", npc_to="b", trust=40, affection=60)
        disp = s.overall_disposition()
        # trust*0.25 + respect*0.2 + affection*0.3 - fear*0.15 + debt*0.1
        expected = 40 * 0.25 + 60 * 0.30
        assert disp == pytest.approx(expected)

    def test_is_default_false_with_values(self):
        s = Sentiment(npc_from="a", npc_to="b", trust=10)
        assert not s.is_default()

    def test_to_description_neutral(self):
        s = Sentiment(npc_from="a", npc_to="b")
        assert s.to_description() == "neutral acquaintance"

    def test_to_description_with_feelings(self):
        s = Sentiment(npc_from="a", npc_to="b", trust=30, fear=25)
        desc = s.to_description()
        assert "trust" in desc.lower()
        assert "fear" in desc.lower()

    def test_to_dict(self):
        s = Sentiment(npc_from="a", npc_to="b", trust=10)
        d = s.to_dict()
        assert d["from"] == "a"
        assert d["to"] == "b"
        assert d["trust"] == 10
        assert "disposition" in d
        assert "description" in d


class TestSentimentTracker:
    """Unit tests for SentimentTracker (SQLite-backed)."""

    @pytest.fixture
    def tracker(self):
        t = SentimentTracker()
        t.initialise()
        return t

    def test_get_default(self, tracker):
        s = tracker.get("npc_a", "npc_b")
        assert s.is_default()
        assert s.npc_from == "npc_a"

    def test_modify_and_retrieve(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 25)
        s = tracker.get("npc_a", "npc_b")
        assert s.trust == 25

    def test_modify_is_additive(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 10)
        tracker.modify("npc_a", "npc_b", "trust", 15)
        s = tracker.get("npc_a", "npc_b")
        assert s.trust == 25

    def test_directional(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 50)
        ab = tracker.get("npc_a", "npc_b")
        ba = tracker.get("npc_b", "npc_a")
        assert ab.trust == 50
        assert ba.trust == 0

    def test_modify_mutual(self, tracker):
        tracker.modify_mutual("npc_a", "npc_b", "respect", 20)
        ab = tracker.get("npc_a", "npc_b")
        ba = tracker.get("npc_b", "npc_a")
        assert ab.respect == 20
        assert ba.respect == 20

    def test_sparse_storage_prunes_defaults(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 10)
        tracker.modify("npc_a", "npc_b", "trust", -10)  # back to 0
        s = tracker.get("npc_a", "npc_b")
        assert s.is_default()
        # Row should be removed from DB
        stats = tracker.get_stats()
        assert stats["total_relationships"] == 0

    def test_get_all_for(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 10)
        tracker.modify("npc_a", "npc_c", "fear", 20)
        rels = tracker.get_all_for("npc_a")
        assert len(rels) == 2

    def test_get_all_towards(self, tracker):
        tracker.modify("npc_a", "npc_c", "trust", 10)
        tracker.modify("npc_b", "npc_c", "trust", 20)
        rels = tracker.get_all_towards("npc_c")
        assert len(rels) == 2

    def test_get_strongest(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 5)
        tracker.modify("npc_a", "npc_c", "trust", 50)
        tracker.modify("npc_a", "npc_d", "affection", 30)
        top = tracker.get_strongest_relationships("npc_a", limit=2)
        assert len(top) == 2
        # npc_c should be first (highest disposition)
        assert top[0].npc_to == "npc_c"

    def test_stats(self, tracker):
        tracker.modify("npc_a", "npc_b", "trust", 10)
        stats = tracker.get_stats()
        assert stats["total_relationships"] == 1
        assert stats["npcs_with_relationships"] == 1

    def test_unknown_dimension_ignored(self, tracker):
        s = tracker.modify("a", "b", "nonexistent", 10)
        assert s.is_default()

    def test_set_replaces(self, tracker):
        sent = Sentiment(npc_from="a", npc_to="b", trust=42, respect=10)
        tracker.set(sent, game_time=100)
        result = tracker.get("a", "b")
        assert result.trust == 42
        assert result.respect == 10


# ---------- Event Impact System ----------


class TestEventImpactSystem:
    """Unit tests for the data-driven event rules engine."""

    @pytest.fixture
    def tracker(self):
        t = SentimentTracker()
        t.initialise()
        return t

    @pytest.fixture
    def system(self, tracker):
        eis = EventImpactSystem(sentiment_tracker=tracker)
        eis.initialise()
        return eis

    def test_default_rules_loaded(self, system):
        stats = system.get_stats()
        assert stats["rule_count"] > 0
        assert "conversation" in stats["event_types"]

    def test_conversation_event_modifies_sentiment(self, system, tracker):
        event = GameEvent(
            event_type="conversation",
            participants=["npc_a", "npc_b"],
            game_time=100,
        )
        effects = system.process_event(event)
        assert len(effects) > 0

        # Check sentiment was modified
        s = tracker.get("npc_a", "npc_b")
        assert s.trust > 0
        assert s.affection > 0

    def test_trade_completed_event(self, system, tracker):
        event = GameEvent(
            event_type="trade_completed",
            participants=["npc_a", "npc_b"],
        )
        system.process_event(event)
        s = tracker.get("npc_a", "npc_b")
        assert s.trust == 5
        assert s.respect == 3

    def test_unknown_event_no_effects(self, system):
        event = GameEvent(event_type="nonexistent_event")
        effects = system.process_event(event)
        assert effects == []

    def test_conditional_rule_passes(self, system, tracker):
        # Boost affection first so proposal condition passes
        tracker.modify("npc_a", "npc_b", "affection", 50)

        event = GameEvent(
            event_type="proposal_accepted",
            participants=["npc_a", "npc_b"],
        )
        effects = system.process_event(event)
        assert len(effects) > 0

    def test_conditional_rule_fails(self, system, tracker):
        # Affection is 0, below the 30 threshold
        event = GameEvent(
            event_type="proposal_accepted",
            participants=["npc_a", "npc_b"],
        )
        effects = system.process_event(event)
        # Should not fire — condition not met
        assert effects == []

    def test_world_event_sets_flag(self, system):
        event = GameEvent(event_type="war_declared")
        system.process_event(event)
        assert system.get_world_flag("war") is True

    def test_world_event_modifies_param(self, system):
        event = GameEvent(event_type="war_declared")
        system.process_event(event)
        assert system.get_world_param("aggression_modifier") == 30

    def test_set_flag_individual_scope(self, system):
        event = GameEvent(
            event_type="proposal_accepted",
            participants=["npc_a", "npc_b"],
        )
        # Need to meet condition first
        system.sentiment.modify("npc_a", "npc_b", "affection", 50)
        system.process_event(event)
        assert system.get_npc_flag("npc_a", "engaged") is True
        assert system.get_npc_flag("npc_b", "engaged") is True

    def test_custom_rule_addition(self, system, tracker):
        system.add_rule({
            "event_type": "custom_event",
            "effects": [
                {"type": "modify_sentiment", "dimension": "fear", "delta": 99},
            ],
            "scope": "individual",
        })
        event = GameEvent(
            event_type="custom_event",
            participants=["npc_a", "npc_b"],
        )
        system.process_event(event)
        s = tracker.get("npc_a", "npc_b")
        assert s.fear == 99

    def test_event_rule_from_dict(self):
        data = {
            "event_type": "test",
            "effects": [{"type": "set_flag", "flag": "x", "value": True}],
            "conditions": [{"type": "always"}],
            "scope": "world",
        }
        rule = EventRule.from_dict(data)
        assert rule.event_type == "test"
        assert len(rule.effects) == 1
        assert len(rule.conditions) == 1

    def test_get_rules_returns_all(self, system):
        rules = system.get_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_world_state(self, system):
        system.set_world_flag("test_flag", True)
        state = system.get_world_state()
        assert state["flags"]["test_flag"] is True

    def test_flag_not_set_condition(self, system, tracker):
        # Add a rule with flag_not_set condition
        system.add_rule({
            "event_type": "conditional_test",
            "effects": [
                {"type": "modify_sentiment", "dimension": "trust", "delta": 10},
            ],
            "conditions": [{"type": "flag_not_set", "flag": "war"}],
            "scope": "individual",
        })
        event = GameEvent(
            event_type="conditional_test",
            participants=["a", "b"],
        )
        # Should fire (war not set)
        effects = system.process_event(event)
        assert len(effects) == 1

        # Set war flag, should not fire now
        system.set_world_flag("war", True)
        tracker.modify("a", "b", "trust", -10)  # reset
        effects = system.process_event(event)
        assert len(effects) == 0


# ---------- Factions ----------


class TestFaction:
    """Unit tests for the Faction data model."""

    def test_create_empty(self):
        f = Faction(faction_id="guild_1", name="Merchant Guild")
        assert f.faction_id == "guild_1"
        assert len(f.members) == 0
        assert f.leader is None

    def test_add_member(self):
        f = Faction(faction_id="guild_1", name="Merchant Guild")
        f.add_member("npc_a", FactionRole.LEADER)
        f.add_member("npc_b")
        assert len(f.members) == 2
        assert f.leader.npc_id == "npc_a"
        assert f.has_member("npc_b")

    def test_remove_member(self):
        f = Faction(faction_id="guild_1", name="Test")
        f.add_member("npc_a")
        assert f.remove_member("npc_a")
        assert not f.has_member("npc_a")
        assert not f.remove_member("nonexistent")

    def test_member_ids(self):
        f = Faction(faction_id="guild_1", name="Test")
        f.add_member("a")
        f.add_member("b")
        assert set(f.member_ids) == {"a", "b"}

    def test_faction_relations(self):
        f = Faction(faction_id="guild_1", name="Guild")
        f.set_relation("guild_2", FactionRelation.ALLIED)
        assert f.get_relation("guild_2") == FactionRelation.ALLIED
        assert f.get_relation("unknown") == FactionRelation.NEUTRAL

    def test_to_dict(self):
        f = Faction(faction_id="g1", name="Test Guild")
        f.add_member("npc_a", FactionRole.LEADER)
        d = f.to_dict()
        assert d["faction_id"] == "g1"
        assert d["leader"] == "npc_a"
        assert d["size"] == 1


class TestFactionManager:
    """Unit tests for the FactionManager."""

    @pytest.fixture
    def mgr(self):
        return FactionManager()

    def test_create_faction(self, mgr):
        f = mgr.create_faction("guild_1", "Merchant Guild", leader_id="npc_a")
        assert f.name == "Merchant Guild"
        assert f.leader.npc_id == "npc_a"

    def test_get_npc_faction(self, mgr):
        mgr.create_faction("g1", "Guild", leader_id="npc_a")
        assert mgr.get_npc_faction("npc_a").faction_id == "g1"
        assert mgr.get_npc_faction("npc_b") is None

    def test_join_faction(self, mgr):
        mgr.create_faction("g1", "Guild")
        assert mgr.join_faction("npc_a", "g1")
        assert mgr.get_npc_faction("npc_a").faction_id == "g1"

    def test_join_nonexistent_faction(self, mgr):
        assert not mgr.join_faction("npc_a", "nonexistent")

    def test_leave_faction(self, mgr):
        mgr.create_faction("g1", "Guild")
        mgr.join_faction("npc_a", "g1")
        assert mgr.leave_faction("npc_a")
        assert mgr.get_npc_faction("npc_a") is None

    def test_same_faction(self, mgr):
        mgr.create_faction("g1", "Guild")
        mgr.join_faction("npc_a", "g1")
        mgr.join_faction("npc_b", "g1")
        assert mgr.same_faction("npc_a", "npc_b")
        assert not mgr.same_faction("npc_a", "npc_c")

    def test_are_allies(self, mgr):
        mgr.create_faction("g1", "Guild A")
        mgr.create_faction("g2", "Guild B")
        mgr.join_faction("npc_a", "g1")
        mgr.join_faction("npc_b", "g2")
        mgr.set_faction_relation("g1", "g2", FactionRelation.ALLIED)
        assert mgr.are_allies("npc_a", "npc_b")

    def test_are_rivals(self, mgr):
        mgr.create_faction("g1", "Guild A")
        mgr.create_faction("g2", "Guild B")
        mgr.join_faction("npc_a", "g1")
        mgr.join_faction("npc_b", "g2")
        mgr.set_faction_relation("g1", "g2", FactionRelation.HOSTILE)
        assert mgr.are_rivals("npc_a", "npc_b")

    def test_switching_factions(self, mgr):
        mgr.create_faction("g1", "Old Guild")
        mgr.create_faction("g2", "New Guild")
        mgr.join_faction("npc_a", "g1")
        mgr.join_faction("npc_a", "g2")
        assert mgr.get_npc_faction("npc_a").faction_id == "g2"
        old = mgr.get_faction("g1")
        assert not old.has_member("npc_a")

    def test_agreements(self, mgr):
        agr = mgr.create_agreement(
            "trade", "g1", "g2",
            terms={"goods": "iron"}, game_time=100, duration=500,
        )
        assert agr.active
        assert not agr.is_expired(200)
        assert agr.is_expired(700)

    def test_expire_agreements(self, mgr):
        mgr.create_agreement("trade", "g1", "g2", game_time=0, duration=100)
        mgr.create_agreement("alliance", "g1", "g3", game_time=0, duration=500)
        expired = mgr.expire_agreements(200)
        assert expired == 1

    def test_get_agreements_for(self, mgr):
        mgr.create_agreement("trade", "g1", "g2")
        mgr.create_agreement("alliance", "g1", "g3")
        agrs = mgr.get_agreements_for("g1")
        assert len(agrs) == 2

    def test_faction_vote_simple_majority(self, mgr):
        mgr.create_faction("g1", "Council")
        mgr.join_faction("a", "g1")
        mgr.join_faction("b", "g1")
        mgr.join_faction("c", "g1")
        result, tally = mgr.get_faction_vote("g1", {"a": True, "b": True, "c": False})
        assert result is True
        assert tally["for"] == 2
        assert tally["against"] == 1

    def test_faction_vote_leader_double_weight(self, mgr):
        f = mgr.create_faction("g1", "Council", leader_id="a")
        mgr.join_faction("b", "g1")
        mgr.join_faction("c", "g1")
        # Leader votes against (weight 2), others vote for (weight 1 each)
        result, tally = mgr.get_faction_vote("g1", {"a": False, "b": True, "c": True})
        assert result is False  # 2 against (leader) vs 2 for → tie → False

    def test_get_social_context_no_faction(self, mgr):
        ctx = mgr.get_social_context("npc_a")
        assert "no faction" in ctx.lower()

    def test_get_social_context_with_faction(self, mgr):
        mgr.create_faction("g1", "Iron Brotherhood", leader_id="npc_a")
        mgr.join_faction("npc_b", "g1")
        ctx = mgr.get_social_context("npc_a")
        assert "Iron Brotherhood" in ctx
        assert "leader" in ctx

    def test_stats(self, mgr):
        mgr.create_faction("g1", "Guild", leader_id="a")
        stats = mgr.get_stats()
        assert stats["faction_count"] == 1
        assert stats["total_members"] == 1


# ---------- Integration ----------


class TestRelationshipContext:
    """Tests for enriched relationship context in memory manager."""

    @pytest.fixture
    def tracker(self):
        t = SentimentTracker()
        t.initialise()
        return t

    @pytest.fixture
    def factions(self):
        return FactionManager()

    def test_sentiment_appears_in_context(self, tracker, factions):
        from core.memory.structured import StructuredMemory
        from core.memory.manager import MemoryManager

        structured = StructuredMemory()
        structured.initialise()
        mgr = MemoryManager(
            structured=structured,
            sentiment=tracker,
            factions=factions,
        )

        tracker.modify("npc_a", "npc_b", "trust", 30)
        ctx = mgr.get_relationship_context("npc_a", "Bob", other_id="npc_b")
        assert "trust" in ctx.lower()

    def test_faction_appears_in_context(self, tracker, factions):
        from core.memory.structured import StructuredMemory
        from core.memory.manager import MemoryManager

        structured = StructuredMemory()
        structured.initialise()
        mgr = MemoryManager(
            structured=structured,
            sentiment=tracker,
            factions=factions,
        )

        factions.create_faction("g1", "Iron Brotherhood")
        factions.join_faction("npc_a", "g1")
        factions.join_faction("npc_b", "g1")

        ctx = mgr.get_relationship_context("npc_a", "Bob", other_id="npc_b")
        assert "Iron Brotherhood" in ctx

    def test_rival_factions_in_context(self, tracker, factions):
        from core.memory.structured import StructuredMemory
        from core.memory.manager import MemoryManager

        structured = StructuredMemory()
        structured.initialise()
        mgr = MemoryManager(
            structured=structured,
            sentiment=tracker,
            factions=factions,
        )

        factions.create_faction("g1", "Guild A")
        factions.create_faction("g2", "Guild B")
        factions.join_faction("npc_a", "g1")
        factions.join_faction("npc_b", "g2")
        factions.set_faction_relation("g1", "g2", FactionRelation.RIVAL)

        ctx = mgr.get_relationship_context("npc_a", "Bob", other_id="npc_b")
        assert "rival" in ctx.lower()

    def test_default_context_no_relationship(self, tracker, factions):
        from core.memory.structured import StructuredMemory
        from core.memory.manager import MemoryManager

        structured = StructuredMemory()
        structured.initialise()
        mgr = MemoryManager(
            structured=structured,
            sentiment=tracker,
            factions=factions,
        )

        ctx = mgr.get_relationship_context("npc_a", "Unknown", other_id="npc_x")
        assert "fellow townsperson" in ctx.lower()


class TestEventConversationIntegration:
    """Tests that conversations fire events and update sentiment."""

    @pytest.fixture
    def tracker(self):
        t = SentimentTracker()
        t.initialise()
        return t

    @pytest.fixture
    def system(self, tracker):
        eis = EventImpactSystem(sentiment_tracker=tracker)
        eis.initialise()
        return eis

    def test_conversation_event_builds_trust(self, system, tracker):
        """Simulates what NPCManager does after a conversation ends."""
        event = GameEvent(
            event_type="conversation",
            participants=["blacksmith_0", "merchant_1"],
            game_time=500,
        )
        effects = system.process_event(event)
        assert len(effects) > 0

        s = tracker.get("blacksmith_0", "merchant_1")
        assert s.trust > 0
        assert s.affection > 0

    def test_multiple_conversations_accumulate(self, system, tracker):
        for _ in range(5):
            system.process_event(GameEvent(
                event_type="conversation",
                participants=["npc_a", "npc_b"],
            ))
        s = tracker.get("npc_a", "npc_b")
        assert s.trust == 10  # 5 * 2
        assert s.affection == 5  # 5 * 1

    def test_betrayal_devastates_relationship(self, system, tracker):
        # Build up trust first
        for _ in range(10):
            system.process_event(GameEvent(
                event_type="conversation",
                participants=["npc_a", "npc_b"],
            ))
        pre_trust = tracker.get("npc_a", "npc_b").trust

        # Betrayal
        system.process_event(GameEvent(
            event_type="betrayal",
            participants=["npc_a", "npc_b"],
        ))
        post = tracker.get("npc_a", "npc_b")
        assert post.trust < pre_trust
        assert post.fear > 0
