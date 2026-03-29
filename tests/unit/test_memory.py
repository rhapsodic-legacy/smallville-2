"""
Unit tests for the Phase 4 memory system.

Tests structured storage, episodic memory, spatial memory,
memory manager, and reflection system.
"""

import asyncio
import pytest

from core.memory.structured import StructuredMemory, Fact, GoalRecord, EventRecord
from core.memory.episodic import EpisodicStore, EpisodicMemory, RetrievalResult
from core.memory.spatial import SpatialMemory
from core.memory.manager import MemoryManager, MemoryContext
from core.npc.models import NPC, PersonalityTraits
from core.npc.llm_client import MockProvider


# ---------- Fixtures ----------

@pytest.fixture
def structured():
    """In-memory SQLite structured storage."""
    store = StructuredMemory(db_path=None)
    store.initialise()
    yield store
    store.close()


@pytest.fixture
def episodic():
    """In-memory episodic store (forced fallback mode for test isolation)."""
    store = EpisodicStore()
    store._fallback_mode = True  # skip ChromaDB to avoid shared state
    return store


@pytest.fixture
def spatial():
    return SpatialMemory()


@pytest.fixture
def mock_llm():
    return MockProvider()


@pytest.fixture
def memory_mgr(mock_llm):
    """Full memory manager with in-memory backends."""
    episodic = EpisodicStore()
    episodic._fallback_mode = True  # force isolation
    mgr = MemoryManager(episodic=episodic, llm=mock_llm)
    mgr.initialise()
    yield mgr
    mgr.close()


@pytest.fixture
def sample_npc():
    return NPC(
        npc_id="blacksmith_0",
        name="Thorin",
        age=45,
        personality=PersonalityTraits(
            openness=0.4, conscientiousness=0.8,
            extraversion=0.3, agreeableness=0.5, neuroticism=0.2,
        ),
        backstory="A veteran blacksmith.",
        occupation="blacksmith",
    )


# ======== Structured Storage Tests ========

class TestStructuredMemory:

    def test_add_and_get_fact(self, structured):
        fact_id = structured.add_fact(
            npc_id="npc_1", subject="Alice", predicate="is_a",
            obj="blacksmith", game_time=100.0,
        )
        assert fact_id > 0

        facts = structured.get_facts("npc_1")
        assert len(facts) == 1
        assert facts[0].subject == "Alice"
        assert facts[0].predicate == "is_a"
        assert facts[0].obj == "blacksmith"

    def test_upsert_fact(self, structured):
        """Same SPO triple should update, not duplicate."""
        id1 = structured.add_fact(
            "npc_1", "Alice", "is_a", "blacksmith", confidence=0.8, game_time=100,
        )
        id2 = structured.add_fact(
            "npc_1", "Alice", "is_a", "blacksmith", confidence=0.95, game_time=200,
        )
        assert id1 == id2

        facts = structured.get_facts("npc_1")
        assert len(facts) == 1
        assert facts[0].confidence == 0.95

    def test_get_facts_filtered(self, structured):
        structured.add_fact("npc_1", "Alice", "is_a", "blacksmith", game_time=100)
        structured.add_fact("npc_1", "Alice", "lives_at", "home", game_time=101)
        structured.add_fact("npc_1", "Bob", "is_a", "farmer", game_time=102)

        facts = structured.get_facts("npc_1", subject="Alice")
        assert len(facts) == 2

        facts = structured.get_facts("npc_1", predicate="is_a")
        assert len(facts) == 2

    def test_get_facts_about(self, structured):
        structured.add_fact("npc_1", "Alice", "trusts", "Bob", game_time=100)
        structured.add_fact("npc_1", "Bob", "owes_gold", "Alice", game_time=101)

        facts = structured.get_facts_about("npc_1", "Bob")
        assert len(facts) == 2

    def test_remove_fact(self, structured):
        fid = structured.add_fact("npc_1", "A", "B", "C", game_time=100)
        structured.remove_fact(fid)
        facts = structured.get_facts("npc_1")
        assert len(facts) == 0

    def test_goals_crud(self, structured):
        gid = structured.add_goal("npc_1", "Master smithing", importance=0.9)
        goals = structured.get_active_goals("npc_1")
        assert len(goals) == 1
        assert goals[0].description == "Master smithing"

        structured.update_goal_status(gid, "completed")
        goals = structured.get_active_goals("npc_1")
        assert len(goals) == 0

    def test_events_crud(self, structured):
        eid = structured.record_event(
            event_type="conversation",
            description="Thorin talked to Elara",
            participants=["npc_1", "npc_2"],
            game_time=500.0,
            importance=0.6,
        )
        assert eid > 0

        events = structured.get_events(event_type="conversation")
        assert len(events) == 1
        assert "Thorin" in events[0].description

    def test_events_by_participant(self, structured):
        structured.record_event(
            "conversation", "A talked to B",
            participants=["npc_1", "npc_2"], game_time=100,
        )
        structured.record_event(
            "trade", "C traded with D",
            participants=["npc_3", "npc_4"], game_time=200,
        )

        events = structured.get_events(participant="npc_1")
        assert len(events) == 1

    def test_stats(self, structured):
        structured.add_fact("npc_1", "A", "B", "C", game_time=100)
        structured.add_goal("npc_1", "Goal 1")
        structured.record_event("test", "Test event", game_time=100)

        stats = structured.get_stats()
        assert stats["total_facts"] == 1
        assert stats["active_goals"] == 1
        assert stats["total_events"] == 1


# ======== Episodic Memory Tests ========

class TestEpisodicStore:

    def test_add_and_count(self, episodic):
        mem_id = episodic.add_memory(
            npc_id="npc_1",
            description="Saw Alice working at the forge",
            category="observation",
            importance=0.5,
            game_time=100.0,
        )
        assert mem_id.startswith("npc_1_mem_")
        assert episodic.count("npc_1") == 1
        assert episodic.count() == 1

    def test_get_recent(self, episodic):
        episodic.add_memory("npc_1", "First observation", game_time=100)
        episodic.add_memory("npc_1", "Second observation", game_time=200)
        episodic.add_memory("npc_1", "Third observation", game_time=300)

        recent = episodic.get_recent("npc_1", limit=2)
        assert len(recent) == 2
        assert recent[0].game_time == 300  # most recent first

    def test_retrieve_by_relevance(self, episodic):
        episodic.add_memory("npc_1", "Alice the blacksmith is working", game_time=100)
        episodic.add_memory("npc_1", "Bob the farmer is sleeping", game_time=200)
        episodic.add_memory("npc_1", "The forge is hot today", game_time=300)

        results = episodic.retrieve(
            "npc_1", query="blacksmith forge", current_game_time=300,
        )
        assert len(results) > 0
        # The blacksmith and forge memories should score higher
        descriptions = [r.memory.description for r in results]
        assert any("blacksmith" in d for d in descriptions)

    def test_retrieve_scores_composite(self, episodic):
        episodic.add_memory(
            "npc_1", "Important event",
            importance=0.9, game_time=100,
        )
        episodic.add_memory(
            "npc_1", "Mundane event",
            importance=0.1, game_time=100,
        )

        results = episodic.retrieve("npc_1", "event", current_game_time=100)
        assert len(results) == 2
        # All results should have scores
        for r in results:
            assert r.composite_score > 0

    def test_importance_since(self, episodic):
        episodic.add_memory("npc_1", "Obs 1", importance=0.5, game_time=100)
        episodic.add_memory("npc_1", "Obs 2", importance=0.8, game_time=200)
        episodic.add_memory("npc_1", "Obs 3", importance=0.3, game_time=300)

        # >= 150 should include Obs 2 (t=200) and Obs 3 (t=300)
        total = episodic.importance_since("npc_1", since_game_time=150)
        assert abs(total - 1.1) < 0.01  # 0.8 + 0.3

    def test_category_filter(self, episodic):
        episodic.add_memory("npc_1", "Saw something", category="observation")
        episodic.add_memory("npc_1", "Talked to someone", category="conversation")

        obs = episodic.get_recent("npc_1", category="observation")
        assert len(obs) == 1
        assert obs[0].category == "observation"

    def test_npc_isolation(self, episodic):
        """Memories from different NPCs should not mix."""
        episodic.add_memory("npc_1", "NPC 1 memory")
        episodic.add_memory("npc_2", "NPC 2 memory")

        assert episodic.count("npc_1") == 1
        assert episodic.count("npc_2") == 1
        assert episodic.count() == 2

    def test_stats(self, episodic):
        episodic.add_memory("npc_1", "Obs", category="observation")
        episodic.add_memory("npc_1", "Conv", category="conversation")

        stats = episodic.get_stats()
        assert stats["total_memories"] == 2
        assert stats["backend"] == "in-memory fallback"


# ======== Spatial Memory Tests ========

class TestSpatialMemory:

    def test_update_and_query(self, spatial):
        spatial.update_from_perception(
            npc_id="npc_1",
            sector="market_district",
            arena="blacksmith_shop",
            objects=["anvil", "forge"],
            game_time=100.0,
        )

        sectors = spatial.get_known_sectors("npc_1")
        assert "market_district" in sectors

        arenas = spatial.get_known_arenas("npc_1", "market_district")
        assert "blacksmith_shop" in arenas

        objects = spatial.get_arena_objects("npc_1", "market_district", "blacksmith_shop")
        assert "anvil" in objects
        assert "forge" in objects

    def test_find_object(self, spatial):
        spatial.update_from_perception(
            "npc_1", "market", "stall", objects=["sword", "shield"],
        )
        spatial.update_from_perception(
            "npc_1", "residential", "home", objects=["bed", "table"],
        )

        results = spatial.find_object("npc_1", "sword")
        assert len(results) == 1
        assert "market:stall" in results[0]

    def test_world_summary(self, spatial):
        spatial.update_from_perception(
            "npc_1", "market_district", "blacksmith_shop",
        )
        spatial.update_from_perception(
            "npc_1", "residential", "home_1",
        )

        summary = spatial.get_world_summary("npc_1")
        assert "market_district" in summary
        assert "residential" in summary

    def test_unknown_npc(self, spatial):
        assert spatial.get_known_sectors("nonexistent") == []
        summary = spatial.get_world_summary("nonexistent")
        assert "do not yet know" in summary

    def test_notes(self, spatial):
        spatial.update_from_perception(
            "npc_1", "market", "stall",
            note="Alice was here selling apples",
        )

        tree = spatial.get_tree("npc_1")
        notes = tree["market"]["arenas"]["stall"]["notes"]
        assert "Alice was here selling apples" in notes

    def test_stats(self, spatial):
        spatial.update_from_perception("npc_1", "sector_a", "arena_1")
        spatial.update_from_perception("npc_1", "sector_a", "arena_2")
        spatial.update_from_perception("npc_2", "sector_b", "arena_3")

        stats = spatial.get_stats()
        assert stats["npcs_with_spatial"] == 2
        assert stats["total_arenas_known"] == 3


# ======== Memory Manager Tests ========

class TestMemoryManager:

    @pytest.mark.asyncio
    async def test_record_observation(self, memory_mgr):
        mem_id = await memory_mgr.record_observation(
            npc_id="npc_1",
            description="Saw Alice working at the forge",
            importance=0.6,
            game_time=100.0,
            location_x=5,
            location_z=10,
            tile_sector="market",
            tile_arena="blacksmith",
        )
        assert mem_id

        # Check episodic stored
        recent = memory_mgr.episodic.get_recent("npc_1")
        assert len(recent) == 1

        # Check spatial updated
        sectors = memory_mgr.spatial.get_known_sectors("npc_1")
        assert "market" in sectors

    @pytest.mark.asyncio
    async def test_record_conversation(self, memory_mgr):
        await memory_mgr.record_conversation(
            npc_a_id="npc_1",
            npc_b_id="npc_2",
            npc_a_name="Thorin",
            npc_b_name="Elara",
            exchanges=[
                {"speaker": "Thorin", "message": "Good day!"},
                {"speaker": "Elara", "message": "Hello, how's the forge?"},
            ],
            game_time=200.0,
        )

        # Both NPCs should have conversation memories
        mem_a = memory_mgr.episodic.get_recent("npc_1")
        mem_b = memory_mgr.episodic.get_recent("npc_2")
        assert len(mem_a) >= 1
        assert len(mem_b) >= 1

        # Event should be recorded
        events = memory_mgr.structured.get_events(event_type="conversation")
        assert len(events) == 1

        # Relationship fact should exist
        facts = memory_mgr.structured.get_facts("npc_1")
        assert any(f.predicate == "spoke_with" for f in facts)

    def test_retrieve_context(self, memory_mgr):
        memory_mgr.episodic.add_memory(
            "npc_1", "Saw Alice at market", importance=0.5, game_time=100,
        )
        memory_mgr.structured.add_fact(
            "npc_1", "Alice", "is_a", "merchant", game_time=100,
        )
        memory_mgr.spatial.update_from_perception(
            "npc_1", "market", "stall",
        )

        ctx = memory_mgr.retrieve_context(
            "npc_1", "market Alice", cognition_tier=1, current_game_time=100,
        )
        assert isinstance(ctx, MemoryContext)
        assert len(ctx.facts) >= 1

        prompt_text = ctx.to_prompt_text()
        assert "Alice" in prompt_text

    def test_retrieve_context_tier_4_empty(self, memory_mgr):
        memory_mgr.episodic.add_memory("npc_1", "Something", game_time=100)

        ctx = memory_mgr.retrieve_context("npc_1", "anything", cognition_tier=4)
        assert len(ctx.episodic) == 0
        assert len(ctx.facts) == 0

    def test_relationship_context(self, memory_mgr):
        memory_mgr.structured.add_fact(
            "npc_1", "npc_1", "spoke_with", "Alice", game_time=100,
        )
        memory_mgr.structured.add_fact(
            "npc_1", "Alice", "is_a", "merchant", game_time=100,
        )

        ctx = memory_mgr.get_relationship_context("npc_1", "Alice")
        assert "Alice" in ctx

    def test_relationship_context_unknown(self, memory_mgr):
        ctx = memory_mgr.get_relationship_context("npc_1", "Unknown Person")
        assert "fellow townsperson" in ctx

    def test_should_reflect_false_initially(self, memory_mgr):
        assert not memory_mgr.should_reflect("npc_1", 100.0)

    def test_get_stats(self, memory_mgr):
        stats = memory_mgr.get_stats()
        assert "structured" in stats
        assert "episodic" in stats
        assert "spatial" in stats

    def test_npc_memory_summary(self, memory_mgr):
        memory_mgr.episodic.add_memory("npc_1", "Test memory", game_time=100)
        memory_mgr.structured.add_fact("npc_1", "A", "B", "C", game_time=100)

        summary = memory_mgr.get_npc_memory_summary("npc_1")
        assert summary["npc_id"] == "npc_1"
        assert len(summary["recent_memories"]) >= 1
        assert len(summary["facts"]) >= 1

    @pytest.mark.asyncio
    async def test_record_reflection(self, memory_mgr):
        mem_id = await memory_mgr.record_reflection(
            "npc_1", "I should visit the market more often", game_time=500,
        )
        assert mem_id

        recent = memory_mgr.episodic.get_recent("npc_1", category="reflection")
        assert len(recent) == 1
        assert "market" in recent[0].description

    def test_fact_extraction(self, memory_mgr):
        """Heuristic fact extraction from observation text."""
        memory_mgr._extract_facts(
            "npc_1", "Alice the blacksmith is working nearby", 100.0,
        )
        facts = memory_mgr.structured.get_facts("npc_1", subject="Alice")
        assert any(f.predicate == "is_a" and f.obj == "blacksmith" for f in facts)
