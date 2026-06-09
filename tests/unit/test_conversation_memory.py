"""Tests for the conversation → memory pipeline.

Verifies that conversations are recorded in both participants' episodic
memory, structured facts, and sentiment when they end.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from core.npc.models import NPC, PersonalityTraits, ActivityState
from core.npc.cognition.converse import (
    end_conversation, initiate_conversation,
    _active_conversations, Conversation, ConversationExchange,
    _conversation_sentiment_deltas,
)
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.relationships.sentiment import SentimentTracker


def _make_npc(npc_id: str, name: str, occupation: str = "guard",
              skills: dict | None = None) -> NPC:
    """Create a minimal NPC for testing."""
    npc = NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory=f"{name} is a {occupation}.",
        occupation=occupation,
        x=5.0, z=5.0,
        home_x=5, home_z=5,
    )
    if skills:
        npc.skills = skills
    return npc


@pytest.fixture
def sentiment():
    """Fresh in-memory sentiment tracker."""
    st = SentimentTracker(db_path=":memory:")
    st.initialise()
    return st


@pytest.fixture
def memory(sentiment):
    """Fresh memory manager with mock LLM and sentiment."""
    episodic = EpisodicStore(fallback_only=True)
    mm = MemoryManager(llm=MockProvider(), sentiment=sentiment, episodic=episodic)
    mm.initialise()
    # Clear any leftover data from previous tests sharing the same
    # in-process ChromaDB collection.
    if mm.episodic._collection is not None:
        try:
            mm.episodic._client.delete_collection("npc_episodic_memory")
            mm.episodic._collection = mm.episodic._client.create_collection(
                name="npc_episodic_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass
    mm.episodic._fallback_memories.clear()
    mm.episodic._counter = 0
    return mm


@pytest.fixture
def npc_pair():
    """Two NPCs ready for conversation."""
    alice = _make_npc("alice_0", "Alice", "blacksmith")
    bob = _make_npc("bob_0", "Bob", "merchant")
    return alice, bob


class TestConversationMemory:
    def test_end_conversation_stores_episodic_memory(
        self, memory, npc_pair,
    ):
        """Both NPCs should get episodic memories after conversation ends."""
        async def _run():
            alice, bob = npc_pair

            # Set up a fake active conversation
            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Hello Bob, need any tools?")
            conv.add_exchange(bob.npc_id, "Bob", "Aye, I could use some nails.")
            _active_conversations[key] = conv

            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(
                alice, bob,
                current_game_minutes=120.0,
                memory_manager=memory,
            )

            # Both should have conversation memories
            alice_memories = memory.episodic.get_recent(alice.npc_id, limit=10)
            bob_memories = memory.episodic.get_recent(bob.npc_id, limit=10)

            alice_convo = [m for m in alice_memories if m.category == "conversation"]
            bob_convo = [m for m in bob_memories if m.category == "conversation"]

            assert len(alice_convo) >= 1, "Alice should have a conversation memory"
            assert len(bob_convo) >= 1, "Bob should have a conversation memory"

            # Memories should mention the other person
            assert "Bob" in alice_convo[0].description
            assert "Alice" in bob_convo[0].description
        asyncio.new_event_loop().run_until_complete(_run())

    def test_end_conversation_stores_spoke_with_fact(
        self, memory, npc_pair,
    ):
        """Both NPCs should get a 'spoke_with' structured fact."""
        async def _run():
            alice, bob = npc_pair

            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Good morning!")
            _active_conversations[key] = conv

            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(
                alice, bob, memory_manager=memory,
            )

            alice_facts = memory.structured.get_facts(alice.npc_id, limit=50)
            spoke = [f for f in alice_facts if f.predicate == "spoke_with"]
            assert len(spoke) >= 1
            assert any(f.obj == "Bob" for f in spoke)
        asyncio.new_event_loop().run_until_complete(_run())

    def test_end_conversation_without_memory_manager(
        self, npc_pair,
    ):
        """Should work fine without memory_manager (no crash)."""
        async def _run():
            alice, bob = npc_pair

            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Hello!")
            _active_conversations[key] = conv

            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            # Should not raise
            await end_conversation(alice, bob)

            assert alice.conversation_partner is None
            assert bob.conversation_partner is None
        asyncio.new_event_loop().run_until_complete(_run())

    def test_empty_conversation_no_memory(
        self, memory, npc_pair,
    ):
        """A conversation with no exchanges should not store memories."""
        async def _run():
            alice, bob = npc_pair

            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            # No exchanges added
            _active_conversations[key] = conv

            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(
                alice, bob, memory_manager=memory,
            )

            alice_memories = memory.episodic.get_recent(alice.npc_id, limit=10)
            convo_memories = [m for m in alice_memories if m.category == "conversation"]
            assert len(convo_memories) == 0
        asyncio.new_event_loop().run_until_complete(_run())

    def test_conversation_content_preserved(
        self, memory, npc_pair,
    ):
        """Exchange content should appear in the stored memory."""
        async def _run():
            alice, bob = npc_pair

            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "I forged a legendary sword!")
            conv.add_exchange(bob.npc_id, "Bob", "How much for it?")
            _active_conversations[key] = conv

            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(
                alice, bob, memory_manager=memory,
            )

            alice_memories = memory.episodic.get_recent(alice.npc_id, limit=10)
            convo = [m for m in alice_memories if m.category == "conversation"]
            assert len(convo) >= 1
            assert "legendary sword" in convo[0].description
        asyncio.new_event_loop().run_until_complete(_run())


class TestConversationSentiment:
    def test_conversation_increases_trust(
        self, memory, npc_pair,
    ):
        """Conversations should build trust between participants."""
        async def _run():
            alice, bob = npc_pair

            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Hello!")
            conv.add_exchange(bob.npc_id, "Bob", "Good day!")
            _active_conversations[key] = conv
            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(alice, bob, memory_manager=memory)

            s = memory.sentiment.get(alice.npc_id, bob.npc_id)
            assert s.trust > 0, "Trust should increase after conversation"
            assert s.affection > 0, "Affection should increase after conversation"
        asyncio.new_event_loop().run_until_complete(_run())

    def test_sentiment_is_mutual(
        self, memory, npc_pair,
    ):
        """Both NPCs should gain sentiment towards each other."""
        async def _run():
            alice, bob = npc_pair

            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Nice weather.")
            _active_conversations[key] = conv
            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(alice, bob, memory_manager=memory)

            s_ab = memory.sentiment.get(alice.npc_id, bob.npc_id)
            s_ba = memory.sentiment.get(bob.npc_id, alice.npc_id)
            assert s_ab.trust > 0
            assert s_ba.trust > 0
            assert s_ab.trust == s_ba.trust
        asyncio.new_event_loop().run_until_complete(_run())

    def test_longer_conversation_more_trust(
        self, memory,
    ):
        """More exchanges should produce more trust and affection."""
        async def _run():
            a = _make_npc("a_0", "A")
            b = _make_npc("b_0", "B")

            # Short conversation: 1 exchange
            key = frozenset({a.npc_id, b.npc_id})
            conv = Conversation(npc_a_id=a.npc_id, npc_b_id=b.npc_id)
            conv.add_exchange(a.npc_id, "A", "Hi.")
            _active_conversations[key] = conv
            a.conversation_partner = b.npc_id
            b.conversation_partner = a.npc_id
            await end_conversation(a, b, memory_manager=memory)
            short_trust = memory.sentiment.get(a.npc_id, b.npc_id).trust

            # Long conversation: 5 more exchanges on top
            conv2 = Conversation(npc_a_id=a.npc_id, npc_b_id=b.npc_id)
            for i in range(5):
                conv2.add_exchange(a.npc_id, "A", f"Message {i}")
            _active_conversations[key] = conv2
            a.conversation_partner = b.npc_id
            b.conversation_partner = a.npc_id
            await end_conversation(a, b, memory_manager=memory)
            long_trust = memory.sentiment.get(a.npc_id, b.npc_id).trust

            assert long_trust > short_trust
        asyncio.new_event_loop().run_until_complete(_run())

    def test_resonance_same_occupation(self):
        """NPCs with the same occupation should get a resonance boost."""
        a = _make_npc("a_0", "A", "blacksmith")
        b = _make_npc("b_0", "B", "blacksmith")
        deltas = _conversation_sentiment_deltas(a, b, 2)
        assert deltas.get("resonance", 0) == 5.0

    def test_resonance_shared_skills(self):
        """NPCs with overlapping skills should get moderate resonance."""
        a = _make_npc("a_0", "A", "blacksmith",
                      skills={"smithing": 0.7, "trading": 0.3})
        b = _make_npc("b_0", "B", "merchant",
                      skills={"trading": 0.8, "diplomacy": 0.5})
        deltas = _conversation_sentiment_deltas(a, b, 2)
        # 1 shared skill (trading) * 1.5 = 1.5
        assert deltas.get("resonance", 0) == 1.5

    def test_no_resonance_different_everything(self):
        """NPCs with no overlap should get no resonance."""
        a = _make_npc("a_0", "A", "blacksmith",
                      skills={"smithing": 0.7})
        b = _make_npc("b_0", "B", "priest",
                      skills={"diplomacy": 0.5})
        deltas = _conversation_sentiment_deltas(a, b, 2)
        assert "resonance" not in deltas or deltas["resonance"] == 0

    def test_personality_clash_reduces_affection(self):
        """Large agreeableness gap should reduce net affection gain."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        # No clash — default 0.5 agreeableness
        baseline = _conversation_sentiment_deltas(a, b, 2)

        # Create a clash: one very agreeable, one very blunt
        a.personality.agreeableness = 0.9
        b.personality.agreeableness = 0.2
        clashing = _conversation_sentiment_deltas(a, b, 2)

        assert clashing["affection"] < baseline["affection"]

    def test_high_neuroticism_reduces_trust(self):
        """High neuroticism should erode trust gain."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        baseline = _conversation_sentiment_deltas(a, b, 2)

        b.personality.neuroticism = 0.9
        neurotic = _conversation_sentiment_deltas(a, b, 2)

        assert neurotic["trust"] < baseline["trust"]

    def test_openness_gap_reduces_respect(self):
        """Large openness gap should slightly reduce respect."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        a.personality.openness = 0.1
        b.personality.openness = 0.9
        deltas = _conversation_sentiment_deltas(a, b, 2)
        # Respect still positive (base 1.0) but reduced
        assert deltas["respect"] < 1.0

    def test_trust_formula_precision(self):
        """Trust = 2.0 + max(0, count-1) * 0.5."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        d1 = _conversation_sentiment_deltas(a, b, 1)
        d3 = _conversation_sentiment_deltas(a, b, 3)
        d5 = _conversation_sentiment_deltas(a, b, 5)
        assert d1["trust"] == pytest.approx(2.0)
        assert d3["trust"] == pytest.approx(3.0)
        assert d5["trust"] == pytest.approx(4.0)

    def test_affection_formula_precision(self):
        """Affection = 1.0 + count * 0.3."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        d1 = _conversation_sentiment_deltas(a, b, 1)
        d4 = _conversation_sentiment_deltas(a, b, 4)
        assert d1["affection"] == pytest.approx(1.3)
        assert d4["affection"] == pytest.approx(2.2)

    def test_respect_always_one(self):
        """Respect base is flat +1.0 regardless of exchange count."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        for count in (1, 3, 5):
            d = _conversation_sentiment_deltas(a, b, count)
            assert d["respect"] == pytest.approx(1.0)

    def test_all_expected_keys_present(self):
        """Delta dict should always contain trust, affection, respect."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        d = _conversation_sentiment_deltas(a, b, 2)
        for key in ("trust", "affection", "respect"):
            assert key in d

    def test_negative_signals_cannot_flip_sign(self):
        """Negative personality clash should reduce but not flip deltas negative
        for a single short conversation."""
        a = _make_npc("a_0", "A")
        b = _make_npc("b_0", "B")
        # Maximum clash: extremes on all dimensions
        a.personality.agreeableness = 1.0
        b.personality.agreeableness = 0.0
        a.personality.neuroticism = 1.0
        a.personality.openness = 0.0
        b.personality.openness = 1.0
        d = _conversation_sentiment_deltas(a, b, 2)
        # With 2 exchanges, trust base=2.5, affection base=1.6, respect=1.0
        # Penalties should reduce but not make negative
        assert d["trust"] > 0, "Trust should stay positive after 2 exchanges"
        assert d["affection"] >= 0, "Affection should not go negative after 2 exchanges"

    def test_end_conversation_mutual_sentiment(
        self, memory, npc_pair,
    ):
        """Both directions of sentiment should be updated after conversation."""
        async def _run():
            alice, bob = npc_pair
            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Hello!")
            conv.add_exchange(bob.npc_id, "Bob", "Hi there!")
            _active_conversations[key] = conv
            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(alice, bob, memory_manager=memory)

            s_ab = memory.sentiment.get(alice.npc_id, bob.npc_id)
            s_ba = memory.sentiment.get(bob.npc_id, alice.npc_id)
            # Both should have trust, affection, and respect
            assert s_ab.trust > 0 and s_ba.trust > 0
            assert s_ab.affection > 0 and s_ba.affection > 0
            assert s_ab.respect > 0 and s_ba.respect > 0
            # Mutual — should be symmetric
            assert s_ab.trust == s_ba.trust
        asyncio.new_event_loop().run_until_complete(_run())

    def test_end_conversation_updates_last_time(self, npc_pair):
        """last_conversation_time should be set after conversation ends."""
        async def _run():
            alice, bob = npc_pair
            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Hello!")
            _active_conversations[key] = conv
            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(
                alice, bob, current_game_minutes=500.0,
            )
            assert alice.last_conversation_time == 500.0
            assert bob.last_conversation_time == 500.0
        asyncio.new_event_loop().run_until_complete(_run())

    def test_end_conversation_sets_dispatch_flag(self, npc_pair):
        """NPCs should be flagged for post-conversation dispatch."""
        async def _run():
            alice, bob = npc_pair
            key = frozenset({alice.npc_id, bob.npc_id})
            conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bob.npc_id)
            conv.add_exchange(alice.npc_id, "Alice", "Bye!")
            _active_conversations[key] = conv
            alice.conversation_partner = bob.npc_id
            bob.conversation_partner = alice.npc_id

            await end_conversation(alice, bob)
            assert alice.needs_post_convo_dispatch is True
            assert bob.needs_post_convo_dispatch is True
        asyncio.new_event_loop().run_until_complete(_run())


class TestConversationFactExtraction:
    """Tests for extracting structured facts from conversation content."""

    def test_parse_fact_triples_valid(self):
        """Should parse pipe-delimited triples."""
        from core.memory.manager import _parse_fact_triples
        response = "Bob | is_hungry | true\nMartha | wants_to_visit | the market"
        facts = _parse_fact_triples(response)
        assert len(facts) == 2
        assert facts[0] == ("Bob", "is_hungry", "true")
        assert facts[1] == ("Martha", "wants_to_visit", "the market")

    def test_parse_fact_triples_no_facts(self):
        """NO_FACTS response should return empty list."""
        from core.memory.manager import _parse_fact_triples
        assert _parse_fact_triples("NO_FACTS") == []

    def test_parse_fact_triples_numbered(self):
        """Should handle numbered lines from LLM."""
        from core.memory.manager import _parse_fact_triples
        response = "1. Bob | is_tired | true\n2. Alice | has_gold | plenty"
        facts = _parse_fact_triples(response)
        assert len(facts) == 2

    def test_parse_fact_triples_skips_malformed(self):
        """Malformed lines should be skipped."""
        from core.memory.manager import _parse_fact_triples
        response = "Bob | is_hungry | true\nthis is not a fact\n | | "
        facts = _parse_fact_triples(response)
        assert len(facts) == 1

    def test_heuristic_extracts_hunger(self):
        """Heuristic should detect 'I'm hungry' as a fact."""
        from core.memory.manager import _extract_facts_heuristic
        exchanges = [
            {"speaker": "Bob", "message": "I'm hungry, haven't eaten all day."},
        ]
        facts = _extract_facts_heuristic(exchanges)
        assert any(f[1] == "is_hungry" for f in facts)
        assert facts[0][0] == "Bob"

    def test_heuristic_extracts_tired(self):
        """Heuristic should detect 'I am tired'."""
        from core.memory.manager import _extract_facts_heuristic
        exchanges = [
            {"speaker": "Alice", "message": "I am so tired today."},
        ]
        facts = _extract_facts_heuristic(exchanges)
        assert any(f[1] == "is_tired" for f in facts)

    def test_heuristic_extracts_need_help(self):
        """Heuristic should detect 'need help' pattern."""
        from core.memory.manager import _extract_facts_heuristic
        exchanges = [
            {"speaker": "Bob", "message": "I need help with the crops."},
        ]
        facts = _extract_facts_heuristic(exchanges)
        assert any(f[1] == "needs_help" for f in facts)

    def test_heuristic_no_false_positives(self):
        """Normal conversation shouldn't produce spurious facts."""
        from core.memory.manager import _extract_facts_heuristic
        exchanges = [
            {"speaker": "Alice", "message": "Good day! Fine weather."},
            {"speaker": "Bob", "message": "Aye, indeed it is."},
        ]
        facts = _extract_facts_heuristic(exchanges)
        assert len(facts) == 0

    def test_heuristic_deduplicates(self):
        """Same fact mentioned twice should only appear once."""
        from core.memory.manager import _extract_facts_heuristic
        exchanges = [
            {"speaker": "Bob", "message": "I'm hungry."},
            {"speaker": "Bob", "message": "I'm so hungry I could faint."},
        ]
        facts = _extract_facts_heuristic(exchanges)
        hungry_facts = [f for f in facts if f[1] == "is_hungry"]
        assert len(hungry_facts) == 1

    def test_record_conversation_extracts_facts(self, memory):
        """record_conversation should extract and store facts."""
        async def _run():
            exchanges = [
                {"speaker": "Bob", "message": "I'm starving, haven't eaten since dawn."},
                {"speaker": "Alice", "message": "Oh no! Let me bring you something."},
            ]
            await memory.record_conversation(
                npc_a_id="bob_0", npc_b_id="alice_0",
                npc_a_name="Bob", npc_b_name="Alice",
                exchanges=exchanges, game_time=100.0,
            )
            # Both participants should have the fact
            bob_facts = memory.structured.get_facts("bob_0", limit=50)
            alice_facts = memory.structured.get_facts("alice_0", limit=50)
            bob_hungry = [f for f in bob_facts if f.predicate == "is_hungry"]
            alice_hungry = [f for f in alice_facts if f.predicate == "is_hungry"]
            assert len(bob_hungry) >= 1
            assert len(alice_hungry) >= 1
        asyncio.new_event_loop().run_until_complete(_run())


class TestSentimentDecay:
    def test_decay_reduces_positive_values(self, sentiment):
        """Positive sentiment should shrink toward zero over time."""
        sentiment.modify("a", "b", "trust", 50.0)
        before = sentiment.get("a", "b").trust
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        after = sentiment.get("a", "b").trust
        assert after < before
        assert after > 0  # should not overshoot to negative

    def test_decay_reduces_negative_values(self, sentiment):
        """Negative sentiment should shrink toward zero too."""
        sentiment.modify("a", "b", "fear", -30.0)
        before = sentiment.get("a", "b").fear
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        after = sentiment.get("a", "b").fear
        assert after > before  # closer to zero (less negative)
        assert after < 0

    def test_decay_proportional_to_magnitude(self, sentiment):
        """Larger values should decay by more in absolute terms."""
        sentiment.modify("a", "b", "trust", 80.0)
        sentiment.modify("c", "d", "trust", 20.0)
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        big_loss = 80.0 - sentiment.get("a", "b").trust
        small_loss = 20.0 - sentiment.get("c", "d").trust
        assert big_loss > small_loss

    def test_decay_prunes_near_zero(self, sentiment):
        """Tiny values should decay to zero and get pruned."""
        sentiment.modify("a", "b", "trust", 0.005)
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.5)
        s = sentiment.get("a", "b")
        assert s.is_default()

    def test_active_relationship_outpaces_decay(self, sentiment):
        """Conversation gains should easily outpace daily decay."""
        # Simulate: gain from conversation, then decay
        sentiment.modify("a", "b", "trust", 2.5)  # typical conversation gain
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.02)
        after = sentiment.get("a", "b").trust
        # At 2% decay, 2.5 loses only 0.05 — still strongly positive
        assert after > 2.0

    def test_decay_exact_formula(self, sentiment):
        """At 10% rate, a value of 50 should lose exactly 5."""
        sentiment.modify("a", "b", "trust", 50.0)
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        after = sentiment.get("a", "b").trust
        assert after == pytest.approx(45.0, abs=0.1)

    def test_multi_day_cumulative_decay(self, sentiment):
        """Multiple days of decay should compound (proportional decay)."""
        sentiment.modify("a", "b", "trust", 100.0)
        for _ in range(10):
            sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        after = sentiment.get("a", "b").trust
        # 100 * 0.9^10 ≈ 34.9
        assert 33.0 < after < 37.0

    def test_decay_all_dimensions(self, sentiment):
        """All sentiment dimensions should be decayed, not just trust."""
        sentiment.modify("a", "b", "trust", 50.0)
        sentiment.modify("a", "b", "affection", 40.0)
        sentiment.modify("a", "b", "respect", 30.0)
        sentiment.modify("a", "b", "fear", -20.0)
        sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        s = sentiment.get("a", "b")
        assert s.trust < 50.0
        assert s.affection < 40.0
        assert s.respect < 30.0
        assert s.fear > -20.0  # closer to zero

    def test_decay_returns_count(self, sentiment):
        """decay_all should return number of relationships updated."""
        sentiment.modify("a", "b", "trust", 50.0)
        sentiment.modify("c", "d", "trust", 30.0)
        count = sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        assert count == 2

    def test_decay_zero_relationships_returns_zero(self, sentiment):
        """No relationships → zero updated."""
        count = sentiment.decay_all(elapsed_game_minutes=1440, rate_per_day=0.1)
        assert count == 0
