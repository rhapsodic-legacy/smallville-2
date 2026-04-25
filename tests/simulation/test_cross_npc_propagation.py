"""
Phase D — full cross-NPC propagation end-to-end.

Jesse's scenario:
1. Player (Traveller) tells Bran that Petra accused him of hoarding bread.
2. Bran and Petra later converse — Bran raises the accusation, Petra denies.
3. The NEXT time the player meets Petra, she has her own memory naming
   Traveller as the rumour-spreader and brings it up.

This is the acceptance test for the holistic-world claim. It doesn't
tune — it just verifies the chain runs end-to-end through
`_persist_finished_conversations` with the Phase A/B/C machinery live.

All three conversations here are scripted — the test does NOT depend
on LLM output; it exercises the extraction + retrieval pipeline
deterministically.
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


def _make_manager(seed: int = 321) -> NPCManager:
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


def _finish_conv(mgr: NPCManager, a, b, lines, minutes: float):
    """Inject a scripted conversation and run it through the full
    persistence pipeline (extraction + resolution included)."""
    conv = Conversation(npc_a_id=a.npc_id, npc_b_id=b.npc_id)
    for speaker, msg in lines:
        speaker_id = a.npc_id if speaker == a.name else b.npc_id
        conv.add_exchange(speaker_id, speaker, msg)
    conv.finished = True
    _active_conversations[frozenset({a.npc_id, b.npc_id})] = conv
    return asyncio.get_event_loop().run_until_complete(
        mgr._persist_finished_conversations(current_minutes=minutes),
    )


def test_player_to_bran_to_petra_back_to_player():
    async def _run():
        mgr = _make_manager()

        # Cast the players with fixed names so the heuristic extractor
        # can reliably key off proper nouns.
        traveller = mgr.npcs[0]
        bran = mgr.npcs[1]
        petra = mgr.npcs[2]
        traveller.name = "Traveller"
        bran.name = "Bran"
        petra.name = "Petra"

        # --- Turn 1: Traveller → Bran, planting the accusation. ---
        conv1 = Conversation(
            npc_a_id=traveller.npc_id, npc_b_id=bran.npc_id,
        )
        conv1.add_exchange(
            traveller.npc_id, "Traveller",
            "Bran, I heard Petra said you have been hoarding bread "
            "from everyone in town.",
        )
        conv1.add_exchange(
            bran.npc_id, "Bran",
            "That is outrageous. I will speak with Petra first thing "
            "tomorrow.",
        )
        conv1.finished = True
        _active_conversations[
            frozenset({traveller.npc_id, bran.npc_id})
        ] = conv1
        await mgr._persist_finished_conversations(current_minutes=100.0)
        _active_conversations.clear()

        # Bran must now hold a relayed_claim naming Petra as the
        # rumour source (via Traveller). This is what feeds Phase C.
        bran_claims = mgr.memory.episodic.get_recent(
            bran.npc_id, limit=20, category="relayed_claim",
        )
        assert any(
            c.metadata.get("cited_source") == "Petra"
            and c.metadata.get("relayed_by") == "Traveller"
            and c.metadata.get("unresolved") is True
            for c in bran_claims
        ), f"turn 1: expected relayed_claim, got: {[c.metadata for c in bran_claims]}"

        # --- Turn 2: Bran → Petra, airing the accusation. ---
        conv2 = Conversation(
            npc_a_id=bran.npc_id, npc_b_id=petra.npc_id,
        )
        conv2.add_exchange(
            bran.npc_id, "Bran",
            "Petra, Traveller told me you said I was hoarding bread. "
            "What truth is in that?",
        )
        conv2.add_exchange(
            petra.npc_id, "Petra",
            "I said no such thing about bread. Whoever Traveller "
            "is, they have been lying to you.",
        )
        conv2.finished = True
        _active_conversations[
            frozenset({bran.npc_id, petra.npc_id})
        ] = conv2
        await mgr._persist_finished_conversations(current_minutes=200.0)
        _active_conversations.clear()

        # Bran's original relayed_claim should now be resolved —
        # he's aired it with Petra.
        post_bran_claims = mgr.memory.episodic.get_recent(
            bran.npc_id, limit=20, category="relayed_claim",
        )
        assert any(
            c.metadata.get("cited_source") == "Petra"
            and c.metadata.get("unresolved") is False
            for c in post_bran_claims
        ), "turn 2: Bran's original claim should be resolved"

        # Petra must now hold a record that names Traveller so the
        # NEXT time she talks to the player, Phase C surfaces it.
        # Acceptable shapes: a relayed_claim with relayed_by=Bran
        # and cited_source=Traveller, OR any outcome memory whose
        # description or metadata mentions "Traveller".
        petra_mems = []
        for cat in ("relayed_claim", "accusation", "commitment"):
            petra_mems.extend(mgr.memory.episodic.get_recent(
                petra.npc_id, limit=20, category=cat,
            ))
        assert any(
            "traveller" in (m.description or "").lower()
            or (m.metadata.get("cited_source", "") or "").lower() == "traveller"
            or (m.metadata.get("relayed_by", "") or "").lower() == "traveller"
            or (m.metadata.get("subject", "") or "").lower() == "traveller"
            for m in petra_mems
        ), (
            f"turn 2: Petra should hold a record naming Traveller. "
            f"got: {[(m.category, m.description, m.metadata) for m in petra_mems]}"
        )

        # --- Turn 3: Traveller meets Petra. ---
        # Petra's prompt-side retrieval when Traveller is the partner
        # must surface the Traveller-named record. That's the hook
        # Jesse wanted: "she might ask me why I told Bran that she
        # said he hoards bread".
        matters = mgr.memory.retrieve_unresolved_matters(
            petra.npc_id,
            partner_id=traveller.npc_id,
            partner_name="Traveller",
        )
        assert matters, (
            "turn 3: Petra should have an unresolved matter to raise "
            "with Traveller"
        )
        # Format block should read sensibly too.
        block = mgr.memory.format_unresolved_matters(matters, "Traveller")
        assert block.startswith("Matters you want to raise with Traveller:")

    asyncio.new_event_loop().run_until_complete(_run())


def test_incidental_chain_does_not_propagate_false_accusations():
    """Negative control: if Bran and Petra talk without raising the
    bread topic, no Traveller-naming memory forms on Petra's side.

    This ensures the extractor isn't over-eager — casual meetings
    between the accused and the alleged source don't implicate the
    player.
    """
    async def _run():
        mgr = _make_manager(seed=331)
        traveller = mgr.npcs[0]
        bran = mgr.npcs[1]
        petra = mgr.npcs[2]
        traveller.name = "Traveller"
        bran.name = "Bran"
        petra.name = "Petra"

        conv1 = Conversation(
            npc_a_id=traveller.npc_id, npc_b_id=bran.npc_id,
        )
        conv1.add_exchange(
            traveller.npc_id, "Traveller",
            "Bran, Petra said you have been hoarding bread.",
        )
        conv1.add_exchange(
            bran.npc_id, "Bran",
            "I shall consider what to do next.",
        )
        conv1.finished = True
        _active_conversations[
            frozenset({traveller.npc_id, bran.npc_id})
        ] = conv1
        await mgr._persist_finished_conversations(current_minutes=100.0)
        _active_conversations.clear()

        # Bran meets Petra but they exchange pleasantries only.
        conv2 = Conversation(
            npc_a_id=bran.npc_id, npc_b_id=petra.npc_id,
        )
        conv2.add_exchange(bran.npc_id, "Bran", "Good morning, Petra.")
        conv2.add_exchange(petra.npc_id, "Petra", "Good morning, Bran.")
        conv2.finished = True
        _active_conversations[
            frozenset({bran.npc_id, petra.npc_id})
        ] = conv2
        await mgr._persist_finished_conversations(current_minutes=200.0)
        _active_conversations.clear()

        # Petra should NOT have anything naming Traveller.
        petra_mems = []
        for cat in ("relayed_claim", "accusation", "commitment"):
            petra_mems.extend(mgr.memory.episodic.get_recent(
                petra.npc_id, limit=20, category=cat,
            ))
        for m in petra_mems:
            assert "traveller" not in (m.description or "").lower(), (
                f"Petra should not have a Traveller-naming memory after "
                f"an incidental meeting. got: {m.description}"
            )

        matters = mgr.memory.retrieve_unresolved_matters(
            petra.npc_id,
            partner_id=traveller.npc_id,
            partner_name="Traveller",
        )
        assert matters == [], (
            "Petra should not have unresolved matters to raise with "
            "Traveller if the chain never actually discussed anything."
        )

    asyncio.new_event_loop().run_until_complete(_run())
