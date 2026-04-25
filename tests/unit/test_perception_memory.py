"""Tests for the enhanced perception → memory pipeline.

Verifies that:
- Relationship sentiment boosts NPC perception importance
- Keyword heuristics score object/event importance correctly
- store_perception() applies relationship-based importance boosting
- Observations carry subject_npc_id for NPC-type perceptions
"""

import asyncio
import pytest

from core.npc.models import NPC, PersonalityTraits, ActivityState
from core.npc.cognition.perceive import (
    perceive, _npc_importance, _object_importance, Observation,
)
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.relationships.sentiment import SentimentTracker


def _make_npc(npc_id: str, name: str, occupation: str = "guard",
              x: float = 5.0, z: float = 5.0, tier: int = 2) -> NPC:
    """Create a minimal NPC for testing."""
    npc = NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory=f"{name} is a {occupation}.",
        occupation=occupation,
        x=x, z=z,
        home_x=5, home_z=5,
    )
    npc.cognition_tier = tier
    return npc


@pytest.fixture
def sentiment():
    st = SentimentTracker(db_path=":memory:")
    st.initialise()
    return st


@pytest.fixture
def memory(sentiment):
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


class TestNpcImportance:
    def test_baseline_importance(self):
        """NPC importance without sentiment should use proximity/activity."""
        a = _make_npc("a", "Alice", x=5.0, z=5.0)
        b = _make_npc("b", "Bob", x=8.0, z=8.0)  # farther away
        imp = _npc_importance(a, b, sentiment=None)
        assert 0.2 <= imp <= 0.6

    def test_strong_relationship_boosts_importance(self, sentiment):
        """NPC with strong positive feelings should perceive higher importance."""
        a = _make_npc("a", "Alice", x=5.0, z=5.0)
        b = _make_npc("b", "Bob", x=8.0, z=8.0)

        baseline = _npc_importance(a, b, sentiment=None)

        # Build a strong relationship
        sentiment.modify("a", "b", "trust", 60.0)
        sentiment.modify("a", "b", "affection", 50.0)
        boosted = _npc_importance(a, b, sentiment=sentiment)

        assert boosted > baseline

    def test_strong_negative_relationship_also_boosts(self, sentiment):
        """Fear/distrust should also increase salience (you watch enemies)."""
        a = _make_npc("a", "Alice", x=5.0, z=5.0)
        b = _make_npc("b", "Bob", x=8.0, z=8.0)

        baseline = _npc_importance(a, b, sentiment=None)

        sentiment.modify("a", "b", "fear", 60.0)
        boosted = _npc_importance(a, b, sentiment=sentiment)

        assert boosted > baseline

    def test_no_relationship_no_boost(self, sentiment):
        """Strangers should get no sentiment boost."""
        a = _make_npc("a", "Alice", x=5.0, z=5.0)
        b = _make_npc("b", "Bob", x=6.0, z=6.0)

        without = _npc_importance(a, b, sentiment=None)
        with_sent = _npc_importance(a, b, sentiment=sentiment)

        # Should be approximately equal (zero sentiment = zero boost)
        assert abs(without - with_sent) < 0.01


class TestObjectImportance:
    def test_high_importance_keywords(self):
        """Danger keywords should score high."""
        assert _object_importance("fire at (3, 5)") == 0.8
        assert _object_importance("dead body at (1, 2)") == 0.8

    def test_moderate_importance_keywords(self):
        """Economic/social keywords should score moderate."""
        assert _object_importance("trade goods at (4, 4)") == 0.5
        assert _object_importance("festival decorations at (0, 0)") == 0.5

    def test_mundane_objects_low(self):
        """Ordinary objects should score low."""
        assert _object_importance("barrel at (2, 3)") == 0.2
        assert _object_importance("rock at (7, 1)") == 0.2


class TestPerceiveIntegration:
    def test_observations_carry_subject_npc_id(self):
        """NPC observations should include the perceived NPC's ID."""
        class FakeGrid:
            def tiles_in_radius(self, x, z, r):
                return []
        a = _make_npc("a", "Alice", x=5.0, z=5.0, tier=2)
        b = _make_npc("b", "Bob", x=6.0, z=6.0, tier=2)

        obs = perceive(a, FakeGrid(), [a, b], 100.0)
        npc_obs = [o for o in obs if o.category == "npc"]
        assert len(npc_obs) == 1
        assert npc_obs[0].subject_npc_id == "b"

    def test_tier4_perceives_nothing(self):
        """Frozen NPCs should not perceive anything."""
        class FakeGrid:
            def tiles_in_radius(self, x, z, r):
                return []
        a = _make_npc("a", "Alice", tier=4)
        b = _make_npc("b", "Bob", x=6.0, z=6.0)
        obs = perceive(a, FakeGrid(), [a, b], 100.0)
        assert len(obs) == 0

    def test_close_npc_higher_importance_than_far(self):
        """NPCs within 2 tiles should have higher importance than those at 5+."""
        class FakeGrid:
            def tiles_in_radius(self, x, z, r):
                return []
        a = _make_npc("a", "Alice", x=5.0, z=5.0, tier=1)
        close = _make_npc("b", "Bob", x=6.0, z=5.0)   # dist 1
        far = _make_npc("c", "Carol", x=10.0, z=5.0)   # dist 5

        obs = perceive(a, FakeGrid(), [a, close, far], 100.0)
        npc_obs = {o.subject_npc_id: o for o in obs if o.category == "npc"}
        assert npc_obs["b"].importance > npc_obs["c"].importance

    def test_same_occupation_bonus(self):
        """Same occupation should increase importance."""
        a = _make_npc("a", "Alice", occupation="blacksmith", x=5.0, z=5.0, tier=2)
        same = _make_npc("b", "Bob", occupation="blacksmith", x=10.0, z=10.0)
        diff = _make_npc("c", "Carol", occupation="farmer", x=10.0, z=10.0)

        imp_same = _npc_importance(a, same, sentiment=None)
        imp_diff = _npc_importance(a, diff, sentiment=None)
        assert imp_same > imp_diff

    def test_activity_bonus(self):
        """NPCs doing interesting activities should get importance boost."""
        a = _make_npc("a", "Alice", x=5.0, z=5.0)
        b_idle = _make_npc("b", "Bob", x=8.0, z=8.0)
        b_talking = _make_npc("c", "Carol", x=8.0, z=8.0)
        b_talking.activity = ActivityState.TALKING

        imp_idle = _npc_importance(a, b_idle, sentiment=None)
        imp_talking = _npc_importance(a, b_talking, sentiment=None)
        assert imp_talking > imp_idle

    def test_importance_capped_at_one(self, sentiment):
        """Importance should never exceed 1.0 regardless of bonuses."""
        a = _make_npc("a", "Alice", occupation="blacksmith", x=5.0, z=5.0)
        b = _make_npc("b", "Bob", occupation="blacksmith", x=6.0, z=5.0)
        b.activity = ActivityState.TALKING
        # Very strong relationship too
        sentiment.modify("a", "b", "trust", 100.0)
        sentiment.modify("a", "b", "affection", 100.0)

        imp = _npc_importance(a, b, sentiment=sentiment)
        assert imp <= 1.0

    def test_retention_window_deduplication(self):
        """Previously perceived observations should not appear again."""
        class FakeGrid:
            def tiles_in_radius(self, x, z, r):
                return []
        a = _make_npc("a", "Alice", x=5.0, z=5.0, tier=2)
        b = _make_npc("b", "Bob", x=6.0, z=6.0, tier=2)

        # First perception
        obs1 = perceive(a, FakeGrid(), [a, b], 100.0)
        assert len(obs1) == 1

        # Second perception — same scene, should be deduplicated
        obs2 = perceive(a, FakeGrid(), [a, b], 200.0)
        assert len(obs2) == 0

    def test_bandwidth_limits_observations(self):
        """Perception should respect attention bandwidth (tier-dependent)."""
        class FakeGrid:
            def tiles_in_radius(self, x, z, r):
                return []
        a = _make_npc("a", "Alice", x=5.0, z=5.0, tier=3)  # bandwidth=2
        # Create many nearby NPCs
        others = [
            _make_npc(f"n{i}", f"NPC{i}", x=5.0 + i * 0.5, z=5.0)
            for i in range(6)
        ]
        all_npcs = [a] + others

        obs = perceive(a, FakeGrid(), all_npcs, 100.0)
        assert len(obs) <= 2  # tier 3 bandwidth


class TestStorePerception:
    def test_store_perception_boosts_for_relationship(
        self, memory, sentiment,
    ):
        """store_perception should boost importance for strong relationships."""
        async def _run():
            sentiment.modify("alice", "bob", "affection", 60.0)
            sentiment.modify("alice", "bob", "trust", 40.0)

            mem_id = await memory.store_perception(
                npc_id="alice",
                description="Bob the merchant is working nearby",
                category="npc",
                importance=0.4,
                game_time=100.0,
                mentioned_npc_id="bob",
            )

            # Retrieve and check boosted importance
            recent = memory.episodic.get_recent("alice", limit=1)
            assert len(recent) == 1
            assert recent[0].importance > 0.4  # should be boosted
        asyncio.new_event_loop().run_until_complete(_run())

    def test_store_perception_no_boost_for_strangers(
        self, memory, sentiment,
    ):
        """No relationship → no importance boost."""
        async def _run():
            mem_id = await memory.store_perception(
                npc_id="alice",
                description="Stranger is walking nearby",
                category="npc",
                importance=0.3,
                game_time=100.0,
                mentioned_npc_id="stranger_0",
            )

            recent = memory.episodic.get_recent("alice", limit=1)
            assert len(recent) == 1
            assert recent[0].importance == 0.3  # unchanged
        asyncio.new_event_loop().run_until_complete(_run())

    def test_store_perception_no_npc_id_no_boost(self, memory):
        """Non-NPC perceptions (objects) should not get relationship boost."""
        async def _run():
            mem_id = await memory.store_perception(
                npc_id="alice",
                description="barrel at (3, 4)",
                category="object",
                importance=0.2,
                game_time=100.0,
            )

            recent = memory.episodic.get_recent("alice", limit=1)
            assert len(recent) == 1
            assert recent[0].importance == 0.2
        asyncio.new_event_loop().run_until_complete(_run())

    def test_store_perception_boost_capped_at_one(
        self, memory, sentiment,
    ):
        """Boosted importance should never exceed 1.0."""
        async def _run():
            sentiment.modify("alice", "bob", "trust", 100.0)
            sentiment.modify("alice", "bob", "affection", 100.0)

            await memory.store_perception(
                npc_id="alice",
                description="Bob is nearby",
                category="npc",
                importance=0.9,  # already high
                game_time=100.0,
                mentioned_npc_id="bob",
            )

            recent = memory.episodic.get_recent("alice", limit=1)
            assert recent[0].importance <= 1.0
        asyncio.new_event_loop().run_until_complete(_run())

    def test_store_perception_weak_relationship_no_boost(
        self, memory, sentiment,
    ):
        """Disposition below threshold (10) should not boost importance."""
        async def _run():
            # Very small sentiment — disposition < 10
            sentiment.modify("alice", "bob", "trust", 2.0)

            await memory.store_perception(
                npc_id="alice",
                description="Bob walking by",
                category="npc",
                importance=0.3,
                game_time=100.0,
                mentioned_npc_id="bob",
            )

            recent = memory.episodic.get_recent("alice", limit=1)
            assert recent[0].importance == 0.3  # no boost
        asyncio.new_event_loop().run_until_complete(_run())

    def test_store_perception_feeds_importance_accumulator(
        self, memory, sentiment,
    ):
        """Stored perceptions should contribute to reflection importance tracking."""
        async def _run():
            sentiment.modify("alice", "bob", "trust", 80.0)
            sentiment.modify("alice", "bob", "affection", 80.0)

            for i in range(5):
                await memory.store_perception(
                    npc_id="alice",
                    description=f"Bob is doing thing {i}",
                    category="npc",
                    importance=0.6,
                    game_time=float(100 + i),
                    mentioned_npc_id="bob",
                )

            # Total importance since time 0 should include boosted values
            total = memory.episodic.importance_since("alice", 0.0)
            assert total > 3.0  # 5 * 0.6 = 3.0 minimum, boosted should exceed
        asyncio.new_event_loop().run_until_complete(_run())
