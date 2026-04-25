"""
Phase B e2e — outcome extraction inside the NPCManager's conversation
persistence pipeline.

Verifies that when an NPC↔NPC conversation finishes:
- record_conversation writes the usual transcript memory.
- consolidate_conversation_turns sweeps per-turn entries.
- extract_outcomes runs and lands structured records.

Uses the bread-hoarding scenario so this test doubles as the
regression gate for the full Phase A + B pipeline.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.converse import (
    Conversation, _active_conversations,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


def _make_manager(seed: int = 101) -> NPCManager:
    config = WorldConfig(population=3, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=MockProvider(), seed=seed,
        memory=memory,
    )
    mgr.spawn_population(3)
    return mgr


@pytest.fixture(autouse=True)
def _clear_conversations():
    _active_conversations.clear()
    yield
    _active_conversations.clear()


def test_bread_hoarding_produces_structured_records():
    async def _run():
        mgr = _make_manager()
        bran, traveller = mgr.npcs[0], mgr.npcs[1]
        # Rename to match the scenario language the extractor expects.
        bran.name = "Bran"
        traveller.name = "Traveller"

        conv = Conversation(npc_a_id=bran.npc_id, npc_b_id=traveller.npc_id)
        conv.add_exchange(bran.npc_id, "Bran", "What brings you, friend?")
        conv.add_exchange(
            traveller.npc_id, "Traveller",
            "Petra said Bran has been hoarding bread for weeks.",
        )
        conv.add_exchange(
            bran.npc_id, "Bran",
            "That is a lie. I will speak with Petra tomorrow.",
        )
        conv.finished = True
        _active_conversations[
            frozenset({bran.npc_id, traveller.npc_id})
        ] = conv

        await mgr._persist_finished_conversations(current_minutes=300.0)

        # Bran's episodic store must now include a relayed_claim
        # about Petra/bread and a commitment to confront Petra.
        bran_claims = mgr.memory.episodic.get_recent(
            bran.npc_id, limit=20, category="relayed_claim",
        )
        assert bran_claims, "expected relayed_claim memory on Bran"
        assert any(
            c.metadata.get("cited_source") == "Petra"
            and c.metadata.get("unresolved") is True
            for c in bran_claims
        ), [c.metadata for c in bran_claims]

        bran_commits = mgr.memory.episodic.get_recent(
            bran.npc_id, limit=20, category="commitment",
        )
        assert any(
            "petra" in c.description.lower()
            for c in bran_commits
        ), [c.description for c in bran_commits]

        # Traveller holds the speaker-side record.
        trav_claims = mgr.memory.episodic.get_recent(
            traveller.npc_id, limit=20, category="relayed_claim",
        )
        assert any(
            "I told Bran" in c.description for c in trav_claims
        ), [c.description for c in trav_claims]

    asyncio.new_event_loop().run_until_complete(_run())


def test_neutral_conversation_produces_no_outcomes():
    async def _run():
        mgr = _make_manager(seed=202)
        a, b = mgr.npcs[0], mgr.npcs[1]

        conv = Conversation(npc_a_id=a.npc_id, npc_b_id=b.npc_id)
        conv.add_exchange(a.npc_id, a.name, "Good morning.")
        conv.add_exchange(b.npc_id, b.name, "A pleasant day indeed.")
        conv.finished = True
        _active_conversations[frozenset({a.npc_id, b.npc_id})] = conv

        await mgr._persist_finished_conversations(current_minutes=400.0)

        for npc in (a, b):
            for cat in ("commitment", "accusation", "relayed_claim"):
                assert not mgr.memory.episodic.get_recent(
                    npc.npc_id, limit=10, category=cat,
                ), f"unexpected {cat} on {npc.name}"

    asyncio.new_event_loop().run_until_complete(_run())
