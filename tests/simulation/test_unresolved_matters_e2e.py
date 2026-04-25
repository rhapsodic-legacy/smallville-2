"""
Phase C e2e — unresolved-matter retrieval and resolution inside the
full conversation persistence pipeline.

Two scenarios:
1. Bran carries an unresolved relayed_claim about Petra into a later
   conversation — retrieve_unresolved_matters + format_unresolved_matters
   produce a prompt block naming the bread topic.
2. When Bran and Petra have a conversation that actually airs the
   topic, the original matter flips to resolved and no longer
   surfaces for the next meeting.

Uses deterministic canned transcripts; does not depend on LLM output.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.conversation_outcomes import (
    Commitment, ConversationOutcome, RelayedClaim,
)
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


def _make_manager(seed: int = 171) -> NPCManager:
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


def test_matter_surfaces_for_bran_against_petra():
    """After Traveller tells Bran 'Petra said you hoard bread', Bran's
    unresolved-matters call naming Petra must return the claim."""
    mgr = _make_manager()
    bran = mgr.npcs[0]
    bran.name = "Bran"

    outcome = ConversationOutcome(
        relayed_claims=[RelayedClaim(
            subject="Bran",
            claim="is hoarding bread",
            cited_source="Petra",
            relayed_by="Traveller",
        )],
    )
    mgr.memory.store_conversation_outcomes(
        outcome,
        participants={bran.npc_id: "Bran", "trav_id": "Traveller"},
    )

    matters = mgr.memory.retrieve_unresolved_matters(
        bran.npc_id, partner_id="petra_id", partner_name="Petra",
    )
    assert matters, "expected bread-hoarding matter naming Petra"
    block = mgr.memory.format_unresolved_matters(matters, "Petra")
    assert "hoarding bread" in block.lower()
    assert block.startswith("Matters you want to raise with Petra:")


def test_bran_meets_petra_resolves_the_matter():
    """Full pipeline: Bran carries the relayed claim, meets Petra,
    discusses the bread topic, matter flips to resolved, next meeting
    no longer surfaces it."""
    async def _run():
        mgr = _make_manager(seed=181)
        bran = mgr.npcs[0]
        petra = mgr.npcs[1]
        bran.name = "Bran"
        petra.name = "Petra"

        # Pre-seed Bran's outcome memory (as if from an earlier
        # conversation with a Traveller).
        mgr.memory.store_conversation_outcomes(
            ConversationOutcome(relayed_claims=[RelayedClaim(
                subject="Bran",
                claim="is hoarding bread",
                cited_source="Petra",
                relayed_by="Traveller",
            )]),
            participants={bran.npc_id: "Bran", "trav_id": "Traveller"},
        )

        # Confirm it's retrievable before the resolution pass.
        before = mgr.memory.retrieve_unresolved_matters(
            bran.npc_id, partner_id=petra.npc_id, partner_name="Petra",
        )
        assert any(
            m.category == "relayed_claim" for m in before
        ), "setup: matter should be retrievable"

        # Bran ↔ Petra conversation in which the topic is actually aired.
        conv = Conversation(npc_a_id=bran.npc_id, npc_b_id=petra.npc_id)
        conv.add_exchange(
            bran.npc_id, "Bran",
            "Petra, did you tell the traveller I was hoarding bread?",
        )
        conv.add_exchange(
            petra.npc_id, "Petra",
            "I never said any such thing about bread.",
        )
        conv.finished = True
        _active_conversations[frozenset({bran.npc_id, petra.npc_id})] = conv

        await mgr._persist_finished_conversations(current_minutes=500.0)

        # Matter should now be resolved for Bran.
        after = mgr.memory.retrieve_unresolved_matters(
            bran.npc_id, partner_id=petra.npc_id, partner_name="Petra",
        )
        assert not any(
            m.category == "relayed_claim" for m in after
        ), (
            "relayed_claim should be resolved after the topic was aired. "
            f"still present: {[m.description for m in after]}"
        )

    asyncio.new_event_loop().run_until_complete(_run())


def test_incidental_meeting_does_not_resolve():
    """Bran meets Petra and they exchange pleasantries without
    mentioning bread. The relayed_claim stays unresolved."""
    async def _run():
        mgr = _make_manager(seed=191)
        bran = mgr.npcs[0]
        petra = mgr.npcs[1]
        bran.name = "Bran"
        petra.name = "Petra"

        mgr.memory.store_conversation_outcomes(
            ConversationOutcome(relayed_claims=[RelayedClaim(
                subject="Bran",
                claim="is hoarding bread",
                cited_source="Petra",
                relayed_by="Traveller",
            )]),
            participants={bran.npc_id: "Bran", "trav_id": "Traveller"},
        )

        conv = Conversation(npc_a_id=bran.npc_id, npc_b_id=petra.npc_id)
        conv.add_exchange(bran.npc_id, "Bran", "A fair morning, Petra.")
        conv.add_exchange(petra.npc_id, "Petra", "Indeed, Bran. Safe travels.")
        conv.finished = True
        _active_conversations[frozenset({bran.npc_id, petra.npc_id})] = conv

        await mgr._persist_finished_conversations(current_minutes=600.0)

        still = mgr.memory.retrieve_unresolved_matters(
            bran.npc_id, partner_id=petra.npc_id, partner_name="Petra",
        )
        assert any(
            m.category == "relayed_claim" for m in still
        ), "chit-chat should not close an open accusation"

    asyncio.new_event_loop().run_until_complete(_run())
