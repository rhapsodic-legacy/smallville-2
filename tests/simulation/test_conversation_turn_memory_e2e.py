"""
Phase A end-to-end — conversation turn memory through the NPCManager.

Verifies that:
1. An NPC↔NPC conversation writes per-turn memories each time
   `_run_conversations` advances it.
2. `_persist_finished_conversations` writes the consolidated summary
   and sweeps the per-turn entries by `conv_id`.
3. memory_formed events are broadcast via the drain queue.
"""

from __future__ import annotations

import asyncio

import pytest

from core.npc.cognition.converse import (
    Conversation, _active_conversations,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


def _make_manager(seed: int = 77) -> NPCManager:
    config = WorldConfig(population=4, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=MockProvider(), seed=seed,
    )
    mgr.spawn_population(4)
    return mgr


@pytest.fixture(autouse=True)
def _clear_conversations():
    _active_conversations.clear()
    yield
    _active_conversations.clear()


def test_per_turn_memories_then_consolidation():
    async def _run():
        mgr = _make_manager()
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        # Place them adjacent so conversation logic is legal.
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name,
                          "You are accusing me of hoarding bread?!")
        conv.add_exchange(alice.npc_id, alice.name,
                          "I only meant it was unusual.")
        _active_conversations[frozenset({alice.npc_id, bran.npc_id})] = conv

        # Persist per-turn entries without waiting for close.
        await mgr.memory.persist_new_exchanges(
            conv, alice, bran, game_time=200.0,
        )
        # Both NPCs should now have two per-turn memories each.
        assert mgr.memory.episodic.count(alice.npc_id) >= 2
        assert mgr.memory.episodic.count(bran.npc_id) >= 2

        # High-keyword "accusing" should have fired memory events.
        events = mgr.memory.drain_memory_events()
        assert any("accusing" in e["summary"].lower() for e in events)

        # Mark finished and trigger consolidation via the manager hook.
        conv.finished = True
        await mgr._persist_finished_conversations(current_minutes=210.0)

        # After consolidation the turn memories are gone, replaced by
        # the consolidated "Had a conversation" summary entry.
        alice_recent = mgr.memory.episodic.get_recent(alice.npc_id, limit=10)
        assert all(
            m.category != mgr.memory.TURN_MEMORY_CATEGORY
            for m in alice_recent
        ), f"turn memories still present: {[m.category for m in alice_recent]}"
        assert any(
            m.category == "conversation" for m in alice_recent
        ), "consolidated conversation memory missing"

    asyncio.new_event_loop().run_until_complete(_run())


def test_idempotent_multiple_runs():
    """Two cognition ticks on the same unchanged conversation must
    not double-write turn memories."""
    async def _run():
        mgr = _make_manager(seed=91)
        alice, bran = mgr.npcs[0], mgr.npcs[1]

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name, "A single line.")
        _active_conversations[frozenset({alice.npc_id, bran.npc_id})] = conv

        await mgr.memory.persist_new_exchanges(conv, alice, bran, game_time=1)
        await mgr.memory.persist_new_exchanges(conv, alice, bran, game_time=2)

        # Count only conversation_turn entries — seed memories are
        # present too but use other categories.
        turn_mems = mgr.memory.episodic.get_recent(
            alice.npc_id, limit=20,
            category=mgr.memory.TURN_MEMORY_CATEGORY,
        )
        assert len(turn_mems) == 1

    asyncio.new_event_loop().run_until_complete(_run())
