"""
Regression gates for the three live-play issues Jesse hit on 2026-04-20:

1. `RuntimeError: dictionary changed size during iteration` in
   `_persist_finished_conversations` — the chat task adds a new
   conversation mid-iteration, crashes the cognition tick, and
   wedges every NPC indoors the following day.

2. NPCs never draw conclusions from notable conversations. The
   transcript is stored verbatim but the router often declined the
   reflection LLM call, so no insight landed.

3. (UX, separately tested at the client layer): a stale NPC reply
   that arrived after the NPC walked out of range was silently
   dropped from the chat panel even though the line existed in
   memory.

This file covers (1) and (2) on the server side.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.conversation_outcomes import (
    Accusation, Commitment, ConversationOutcome, RelayedClaim,
)
from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.reflection import _format_outcome_for_reflection
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.converse import (
    Conversation, _active_conversations,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


def _make_manager(seed: int = 711) -> NPCManager:
    config = WorldConfig(population=4, terrain="riverside", seed=seed)
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
    mgr.spawn_population(4)
    return mgr


@pytest.fixture(autouse=True)
def _clear_conversations():
    _active_conversations.clear()
    yield
    _active_conversations.clear()


# ---------- Issue #1: iteration race ----------

def test_concurrent_conversation_insert_does_not_crash_tick():
    """Simulates the race that bricked day 38 in the wild.

    While `_persist_finished_conversations` is iterating one
    finished conversation, another task inserts a brand-new
    conversation into `_active_conversations`. The old code raised
    `RuntimeError: dictionary changed size during iteration` from
    inside the loop, which took down the whole cognition tick.
    The snapshot fix (`list(_active_conversations.items())`) must
    keep the loop running to completion.
    """
    async def _run():
        mgr = _make_manager()
        alice, bran, cedric = mgr.npcs[0], mgr.npcs[1], mgr.npcs[2]

        conv1 = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv1.add_exchange(alice.npc_id, alice.name, "Good morning.")
        conv1.add_exchange(bran.npc_id, bran.name, "A fair day.")
        conv1.finished = True
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv1

        # A second conversation inserts MID-iteration via a task
        # scheduled to fire during the first `record_conversation`
        # await. Simulate by adding it before the call — the snapshot
        # must still cover both (iterate over both) or skip the new
        # one cleanly without raising.
        async def _insert_mid_run():
            # Let the loop start its first iteration
            await asyncio.sleep(0)
            conv2 = Conversation(
                npc_a_id=alice.npc_id, npc_b_id=cedric.npc_id,
            )
            conv2.add_exchange(alice.npc_id, alice.name, "Also hello.")
            conv2.add_exchange(cedric.npc_id, cedric.name, "Greetings.")
            conv2.finished = True
            _active_conversations[
                frozenset({alice.npc_id, cedric.npc_id})
            ] = conv2

        # Run the insert in parallel with the persistence loop.
        await asyncio.gather(
            mgr._persist_finished_conversations(current_minutes=200.0),
            _insert_mid_run(),
        )
        # No exception = regression is fixed.

    asyncio.new_event_loop().run_until_complete(_run())


# ---------- Issue #2: forced reflection on notable outcomes ----------

def test_accusation_conversation_produces_reflection():
    """A conversation that extracts an accusation must leave a
    reflection memory on both participants, regardless of what the
    router says about budget."""
    async def _run():
        mgr = _make_manager(seed=722)
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        # Both tier 1 so the underlying LLM pathway is allowed;
        # MockProvider always returns a canned reply.
        alice.cognition_tier = 1
        bran.cognition_tier = 1

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(
            alice.npc_id, alice.name,
            "You are a liar! Everyone knows you stole bread from "
            "the market!",
        )
        conv.add_exchange(
            bran.npc_id, bran.name,
            "That is a baseless slander.",
        )
        conv.finished = True
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        await mgr._persist_finished_conversations(current_minutes=300.0)

        for npc in (alice, bran):
            reflections = mgr.memory.episodic.get_recent(
                npc.npc_id, limit=10, category="reflection",
            )
            assert reflections, (
                f"{npc.name} should have a reflection memory after an "
                f"accusation conversation. got none."
            )

    asyncio.new_event_loop().run_until_complete(_run())


def test_neutral_conversation_does_not_force_reflection():
    """Chit-chat — no outcomes — shouldn't force an LLM reflection.
    With our MockProvider and the router's default policy for a
    low-importance decision, the pre-fix baseline was: no reflection.
    We want that behaviour preserved so chit-chat stays cheap."""
    async def _run():
        mgr = _make_manager(seed=733)
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        # Router routes reflection to deterministic for low-importance
        # decisions by default; force tier to one that avoids LLM.
        alice.cognition_tier = 3
        bran.cognition_tier = 3

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(alice.npc_id, alice.name, "Good morning.")
        conv.add_exchange(bran.npc_id, bran.name, "A fair day indeed.")
        conv.finished = True
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        await mgr._persist_finished_conversations(current_minutes=400.0)

        for npc in (alice, bran):
            reflections = mgr.memory.episodic.get_recent(
                npc.npc_id, limit=10, category="reflection",
            )
            assert not reflections, (
                f"{npc.name} should NOT reflect on chit-chat. "
                f"got: {[r.description for r in reflections]}"
            )

    asyncio.new_event_loop().run_until_complete(_run())


# ---------- Outcome summary helper ----------

class TestFormatOutcomeForReflection:
    def test_empty_returns_empty(self):
        assert _format_outcome_for_reflection(None) == ""
        assert _format_outcome_for_reflection(ConversationOutcome()) == ""

    def test_renders_all_three_shapes(self):
        outcome = ConversationOutcome(
            commitments=[Commitment(speaker="Alice", subject="confront Bran")],
            accusations=[Accusation(
                accuser="Alice", accused="Bran", claim="hoarding bread",
            )],
            relayed_claims=[RelayedClaim(
                subject="Bran", claim="is a thief",
                cited_source="Petra", relayed_by="Traveller",
            )],
        )
        out = _format_outcome_for_reflection(outcome)
        assert "Alice committed to confront Bran" in out
        assert "Alice accused Bran of hoarding bread" in out
        assert "Traveller relayed that Petra said Bran is a thief" in out
