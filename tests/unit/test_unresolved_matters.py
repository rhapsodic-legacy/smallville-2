"""
Phase C — unresolved-matter retrieval, prompt injection, and
transcript-driven resolution.

Covers:
- retrieve_unresolved_matters filters by partner via metadata fields
  (cited_source, accused, subject, accuser) AND description match.
- It respects the unresolved flag.
- format_unresolved_matters renders a prompt-friendly one-liner.
- resolve_matters_from_transcript flips the flag only when distinctive
  claim tokens appear in the transcript.
- update_metadata round-trips cleanly through EpisodicStore fallback.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.conversation_outcomes import (
    Accusation, Commitment, ConversationOutcome, RelayedClaim,
)
from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory


def _mgr() -> MemoryManager:
    mgr = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mgr.initialise()
    return mgr


def _seed_bread_scenario(mgr: MemoryManager) -> None:
    """Write the bread-hoarding outcomes so Bran has unresolved matters
    mentioning Petra."""
    outcome = ConversationOutcome(
        relayed_claims=[RelayedClaim(
            subject="Bran",
            claim="is hoarding bread",
            cited_source="Petra",
            relayed_by="Traveller",
        )],
        commitments=[Commitment(
            speaker="Bran",
            subject="speak with Petra tomorrow",
        )],
    )
    mgr.store_conversation_outcomes(
        outcome,
        participants={"bran_id": "Bran", "trav_id": "Traveller"},
    )


# ---------- EpisodicStore update_metadata ----------

class TestEpisodicUpdate:
    def test_round_trip_fallback(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        mid = store.add_memory(
            npc_id="a", description="x", category="c",
            extra_metadata={"unresolved": True, "foo": "bar"},
        )
        assert store.update_metadata(mid, {"unresolved": False}) is True
        got = store.get_recent("a", limit=1)[0]
        assert got.metadata.get("unresolved") is False
        # Other metadata preserved.
        assert got.metadata.get("foo") == "bar"

    def test_unknown_id_returns_false(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        assert store.update_metadata("nope", {"unresolved": False}) is False

    def test_empty_updates_returns_false(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        mid = store.add_memory(npc_id="a", description="x")
        assert store.update_metadata(mid, {}) is False


# ---------- retrieve_unresolved_matters ----------

class TestRetrieve:
    def test_relayed_claim_surfaces_by_cited_source(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        matters = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="petra_id", partner_name="Petra",
        )
        assert matters, "expected relayed claim + commitment naming Petra"
        # At least one record should be a relayed_claim about the bread topic.
        assert any(
            "hoarding bread" in m.description.lower() for m in matters
        )

    def test_no_match_for_unrelated_partner(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        # Fiona has no relationship to the bread claim.
        matters = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="fiona_id", partner_name="Fiona",
        )
        assert matters == []

    def test_respects_unresolved_flag(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        # Flip the relayed_claim record to resolved directly.
        claims = mgr.episodic.get_recent(
            "bran_id", limit=10, category="relayed_claim",
        )
        assert claims
        mgr.episodic.update_metadata(claims[0].memory_id, {"unresolved": False})

        matters = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="petra_id", partner_name="Petra",
        )
        # Commitment still unresolved (mentions "Petra" in description).
        assert matters
        assert all(
            m.category != "relayed_claim" for m in matters
        ), "resolved relayed_claim should be filtered"

    def test_commitment_surfaces_by_description(self):
        mgr = _mgr()
        # Commitment with no explicit `about` field — relies on the
        # partner's name appearing in the free-text description.
        outcome = ConversationOutcome(commitments=[
            Commitment(speaker="Bran", subject="speak with Petra tomorrow"),
        ])
        mgr.store_conversation_outcomes(
            outcome, participants={"bran_id": "Bran", "a_id": "A"},
        )
        matters = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="p_id", partner_name="Petra",
        )
        assert any(m.category == "commitment" for m in matters)

    def test_limit_applied(self):
        mgr = _mgr()
        for i in range(5):
            outcome = ConversationOutcome(accusations=[
                Accusation(accuser="Alice", accused="Bran",
                           claim=f"charge number {i}"),
            ])
            mgr.store_conversation_outcomes(
                outcome,
                participants={"alice_id": "Alice", "bran_id": "Bran"},
                game_time=float(i),
            )
        matters = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="alice_id", partner_name="Alice",
            limit=2,
        )
        assert len(matters) == 2


# ---------- format_unresolved_matters ----------

class TestFormatMatters:
    def test_empty_returns_empty_string(self):
        mgr = _mgr()
        assert mgr.format_unresolved_matters([], "Petra") == ""

    def test_renders_clean_block(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        matters = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="petra_id", partner_name="Petra",
        )
        block = mgr.format_unresolved_matters(matters, "Petra")
        assert block.startswith("Matters you want to raise with Petra:")
        assert "hoarding bread" in block.lower()


# ---------- resolve_matters_from_transcript ----------

class TestResolveFromTranscript:
    def test_relevant_keyword_resolves(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)

        transcript = (
            "Bran: Petra, did you tell people I was hoarding bread? "
            "Petra: I never said any such thing."
        )
        resolved = mgr.resolve_matters_from_transcript(
            npc_id="bran_id",
            partner_id="petra_id",
            partner_name="Petra",
            transcript_text=transcript,
        )
        assert resolved >= 1

        # The relayed claim is now marked resolved, so retrieval no
        # longer surfaces it.
        remaining = mgr.retrieve_unresolved_matters(
            "bran_id", partner_id="petra_id", partner_name="Petra",
        )
        assert not any(
            m.category == "relayed_claim" for m in remaining
        )

    def test_chitchat_does_not_resolve(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        transcript = "Petra: Good day, Bran. Bran: Good day to you."
        resolved = mgr.resolve_matters_from_transcript(
            npc_id="bran_id",
            partner_id="petra_id",
            partner_name="Petra",
            transcript_text=transcript,
        )
        assert resolved == 0

    def test_no_partner_name_no_op(self):
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        resolved = mgr.resolve_matters_from_transcript(
            npc_id="bran_id",
            partner_id="",
            partner_name="",
            transcript_text="blah blah bread",
        )
        assert resolved == 0

    def test_partner_name_alone_is_not_enough(self):
        """The partner's name in the transcript without any claim
        keyword must not resolve the matter — we need the topic aired."""
        mgr = _mgr()
        _seed_bread_scenario(mgr)
        transcript = "Bran: Petra, hello. Petra: Hello Bran."
        resolved = mgr.resolve_matters_from_transcript(
            npc_id="bran_id",
            partner_id="petra_id",
            partner_name="Petra",
            transcript_text=transcript,
        )
        assert resolved == 0
