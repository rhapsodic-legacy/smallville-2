"""
Regression: a player chat must not wedge the cognition tick and
freeze every NPC in their houses.

The bug this guards against (reported 2026-04-22, third time):

  1. Player chats with an NPC.
  2. After the chat ends, `_persist_finished_conversations` runs on
     the next cognition tick and calls `memory.record_conversation`.
  3. `record_conversation` performs an LLM fact-extraction pass. If
     the LLM raises (timeout, provider hiccup), the exception
     propagates up to `cognition_tick`. The tick's outer handler
     logs and returns.
  4. The bad conversation stays in `_active_conversations` because
     `clear_finished_conversations()` (called AFTER the throwing
     line) was skipped. Every subsequent tick hits the same
     exception at the same line. The cognition tick effectively
     crashes on every tick — step 6b (overlap resolution),
     step 7 (reflection), and all post-persistence work are
     skipped repeatedly. NPCs never advance their schedules, never
     form perceptions beyond basic ones, and appear "stuck in
     their houses all day".

This file has two tests:

- `test_record_conversation_exception_does_not_freeze_tick`:
  a finished conversation whose `record_conversation` raises is
  marked `persisted=True` and swept away by
  `clear_finished_conversations`. A second tick does NOT re-raise.

- `test_many_ticks_do_not_accumulate_stuck_conversations`:
  10 ticks after a throwing conversation leave
  `_active_conversations` empty — not a growing pile of zombies.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.converse import (
    Conversation, _active_conversations,
    clear_finished_conversations,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


@pytest.fixture(autouse=True)
def _clear_conv_registry():
    _active_conversations.clear()
    yield
    _active_conversations.clear()


def _mgr() -> NPCManager:
    config = WorldConfig(population=3, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=MockProvider(), seed=42,
        memory=memory,
    )
    mgr.spawn_population(3)
    return mgr


def _seed_finished_conversation(mgr: NPCManager) -> Conversation:
    """Inject a finished conversation into the registry as if the
    player just closed chat with the first NPC."""
    a, b = mgr.npcs[0], mgr.npcs[1]
    conv = Conversation(npc_a_id=a.npc_id, npc_b_id=b.npc_id)
    conv.add_exchange(a.npc_id, a.name, "I'll do something about this.")
    conv.add_exchange(b.npc_id, b.name, "Good luck with that.")
    conv.finished = True
    _active_conversations[frozenset([a.npc_id, b.npc_id])] = conv
    return conv


def test_record_conversation_exception_does_not_freeze_tick():
    """If `record_conversation` raises, the conversation is still
    marked persisted + swept, so the next tick is free of zombie
    state. And the crash does NOT propagate out of
    `_persist_finished_conversations`."""
    async def _run():
        mgr = _mgr()
        conv = _seed_finished_conversation(mgr)

        # Make record_conversation raise on the first (and only) call.
        mgr.memory.record_conversation = AsyncMock(
            side_effect=RuntimeError("simulated LLM timeout"),
        )

        # First tick — persistence hits the exception, logs, flags
        # persisted=True so the conversation doesn't live forever.
        await mgr._persist_finished_conversations(current_minutes=60.0)
        assert conv.persisted, (
            "Conversation must be flagged persisted even on failure "
            "— otherwise we re-crash the tick next time."
        )

        # cleanup sweep takes it out of the active registry.
        removed = clear_finished_conversations()
        assert removed == 1
        assert not _active_conversations

        # Second tick — no zombies, no exceptions, no retry.
        await mgr._persist_finished_conversations(current_minutes=120.0)

    asyncio.run(_run())


def test_many_ticks_do_not_accumulate_stuck_conversations():
    """Ten consecutive ticks after a throwing conversation should
    leave the active registry empty — not accumulate ghost entries
    that re-crash the tick indefinitely."""
    async def _run():
        mgr = _mgr()
        _seed_finished_conversation(mgr)
        mgr.memory.record_conversation = AsyncMock(
            side_effect=RuntimeError("bad LLM"),
        )

        for tick in range(10):
            await mgr._persist_finished_conversations(current_minutes=60.0 * (tick + 1))
            clear_finished_conversations()

        # After 10 ticks the registry is empty and record_conversation
        # was called exactly once (on the first tick — the rest
        # short-circuit via `if conv.persisted: continue`).
        assert not _active_conversations
        assert mgr.memory.record_conversation.call_count == 1, (
            f"record_conversation should be called once for a "
            f"throwing conversation, not retried indefinitely; "
            f"got {mgr.memory.record_conversation.call_count} calls"
        )

    asyncio.run(_run())


def test_successful_conversation_still_persists_normally():
    """Sanity guard: the fix must not break the happy path. A
    conversation whose record_conversation succeeds is still
    marked persisted and swept."""
    async def _run():
        mgr = _mgr()
        conv = _seed_finished_conversation(mgr)
        await mgr._persist_finished_conversations(current_minutes=60.0)
        assert conv.persisted
        removed = clear_finished_conversations()
        assert removed == 1
        # The conversation summary landed in both participants' memory.
        a_recents = mgr.memory.episodic.get_recent(conv.npc_a_id, limit=20)
        conv_mems = [m for m in a_recents if m.category == "conversation"]
        assert conv_mems, "happy-path conversation memory missing"

    asyncio.run(_run())
