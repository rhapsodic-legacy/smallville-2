"""
Phase B — conversation outcome extraction & structured persistence.

Covers:
- extract_heuristic fires on commit / accuse / relayed phrasing.
- extract_heuristic rejects false positives (empty messages, speaker
  citing themselves).
- _parse_llm_json tolerates code-fenced / noisy LLM replies.
- extract_outcomes (no llm) equals the heuristic-only result.
- merge_outcomes deduplicates by semantic key.
- MemoryManager.store_conversation_outcomes lands the right records
  on accuser / accused / witness and emits memory_formed events.
- End-to-end: the bread-hoarding scenario produces a RelayedClaim on
  both participants.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.conversation_outcomes import (
    Accusation, Commitment, ConversationOutcome, RelayedClaim,
    _parse_llm_json, extract_heuristic, extract_outcomes, merge_outcomes,
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


def _exchange(speaker: str, message: str) -> dict[str, str]:
    return {"speaker": speaker, "message": message}


# ---------- Heuristic extractor ----------

class TestExtractHeuristic:
    def test_empty_conversation(self):
        assert extract_heuristic([]).is_empty()

    def test_will_commitment_caught(self):
        ex = [_exchange("Bran", "I will speak with Petra tomorrow.")]
        outcome = extract_heuristic(ex)
        assert len(outcome.commitments) == 1
        assert outcome.commitments[0].speaker == "Bran"
        assert "speak with Petra" in outcome.commitments[0].subject

    def test_you_are_a_liar_is_accusation(self):
        ex = [
            _exchange("Bran", "I did nothing wrong."),
            _exchange("Alice", "You are a liar!"),
        ]
        outcome = extract_heuristic(ex)
        assert len(outcome.accusations) == 1
        a = outcome.accusations[0]
        assert a.accuser == "Alice"
        assert a.accused == "Bran"
        assert "liar" in a.claim.lower()

    def test_hoarding_accusation(self):
        ex = [
            _exchange("Bran", "Good morning."),
            _exchange(
                "Alice",
                "You have been hoarding bread and everyone knows it.",
            ),
        ]
        outcome = extract_heuristic(ex)
        assert any(
            "hoard" in a.claim.lower() for a in outcome.accusations
        ), outcome.accusations

    def test_relayed_claim_bread_hoarding(self):
        """The scenario Jesse flagged: Traveller relays Petra's claim."""
        ex = [
            _exchange("Bran", "What brings you to me?"),
            _exchange(
                "Traveller",
                "Petra said Bran has been hoarding bread and "
                "telling everyone.",
            ),
        ]
        outcome = extract_heuristic(ex)
        assert len(outcome.relayed_claims) >= 1
        r = outcome.relayed_claims[0]
        assert r.cited_source == "Petra"
        assert r.relayed_by == "Traveller"
        assert r.subject == "Bran"
        assert "hoard" in r.claim.lower()

    def test_self_citation_is_not_relayed(self):
        """'I said X' must not produce a relayed_claim."""
        ex = [_exchange("Bran", "Bran said nothing of the sort.")]
        outcome = extract_heuristic(ex)
        assert not outcome.relayed_claims

    def test_promise_verb(self):
        ex = [_exchange("Alice", "I promise to defend the town.")]
        outcome = extract_heuristic(ex)
        assert len(outcome.commitments) == 1
        assert "defend" in outcome.commitments[0].subject.lower()


# ---------- LLM JSON parsing ----------

class TestParseLlmJson:
    def test_empty_braces(self):
        assert _parse_llm_json("{}").is_empty()

    def test_structured_response(self):
        raw = """```json
{
  "commitments": [{"speaker": "Alice", "subject": "help repair the bridge"}],
  "accusations": [],
  "relayed_claims": [{
    "subject": "Bran", "claim": "is hoarding bread",
    "cited_source": "Petra", "relayed_by": "Traveller"
  }]
}
```"""
        out = _parse_llm_json(raw)
        assert len(out.commitments) == 1
        assert out.commitments[0].speaker == "Alice"
        assert len(out.relayed_claims) == 1
        assert out.relayed_claims[0].cited_source == "Petra"

    def test_bad_json_returns_empty(self):
        assert _parse_llm_json("not json at all").is_empty()

    def test_skeletal_entries_dropped(self):
        raw = '{"commitments": [{"speaker": "", "subject": ""}], "accusations": [], "relayed_claims": []}'
        assert _parse_llm_json(raw).is_empty()


# ---------- Merge ----------

class TestMerge:
    def test_dedupes_by_key(self):
        a = ConversationOutcome(commitments=[
            Commitment(speaker="Alice", subject="help"),
        ])
        b = ConversationOutcome(commitments=[
            Commitment(speaker="alice", subject="HELP"),  # case-insensitive dup
        ])
        merged = merge_outcomes(a, b)
        assert len(merged.commitments) == 1

    def test_unions_distinct(self):
        a = ConversationOutcome(accusations=[
            Accusation(accuser="A", accused="B", claim="lying"),
        ])
        b = ConversationOutcome(accusations=[
            Accusation(accuser="A", accused="B", claim="stealing"),
        ])
        merged = merge_outcomes(a, b)
        assert len(merged.accusations) == 2


# ---------- extract_outcomes (no LLM) ----------

class TestExtractOutcomes:
    def test_no_llm_equals_heuristic(self):
        ex = [_exchange("Alice", "I will repair the bridge.")]
        out = asyncio.run(extract_outcomes(ex, llm=None))
        assert out == extract_heuristic(ex)


# ---------- MemoryManager persistence ----------

class TestStoreOutcomes:
    def test_commitment_lands_on_speaker_only(self):
        mgr = _mgr()
        outcome = ConversationOutcome(commitments=[
            Commitment(speaker="Alice", subject="confront Bran"),
        ])
        mgr.store_conversation_outcomes(
            outcome,
            participants={"alice_id": "Alice", "bran_id": "Bran"},
        )
        alice_mems = mgr.episodic.get_recent(
            "alice_id", limit=10, category="commitment",
        )
        bran_mems = mgr.episodic.get_recent(
            "bran_id", limit=10, category="commitment",
        )
        assert len(alice_mems) == 1
        assert not bran_mems
        assert "confront Bran" in alice_mems[0].description
        assert alice_mems[0].metadata.get("unresolved") is True

    def test_accusation_lands_on_both_with_different_framing(self):
        mgr = _mgr()
        outcome = ConversationOutcome(accusations=[
            Accusation(accuser="Alice", accused="Bran",
                       claim="hoarding bread"),
        ])
        mgr.store_conversation_outcomes(
            outcome,
            participants={"alice_id": "Alice", "bran_id": "Bran"},
        )
        alice_mem = mgr.episodic.get_recent(
            "alice_id", limit=10, category="accusation",
        )[0]
        bran_mem = mgr.episodic.get_recent(
            "bran_id", limit=10, category="accusation",
        )[0]
        assert alice_mem.description.startswith("I accused Bran")
        assert bran_mem.description.startswith("Alice accused me")
        assert alice_mem.importance >= 0.8

    def test_relayed_claim_lands_on_both(self):
        mgr = _mgr()
        outcome = ConversationOutcome(relayed_claims=[
            RelayedClaim(
                subject="Bran", claim="is hoarding bread",
                cited_source="Petra", relayed_by="Traveller",
            ),
        ])
        mgr.store_conversation_outcomes(
            outcome,
            participants={"trav_id": "Traveller", "bran_id": "Bran"},
        )
        trav_mem = mgr.episodic.get_recent(
            "trav_id", limit=10, category="relayed_claim",
        )[0]
        bran_mem = mgr.episodic.get_recent(
            "bran_id", limit=10, category="relayed_claim",
        )[0]
        # Speaker's phrasing mentions telling the other party.
        assert "I told Bran" in trav_mem.description
        # Listener sees it attributed to Traveller.
        assert "Traveller told me" in bran_mem.description
        assert bran_mem.metadata.get("cited_source") == "Petra"
        assert bran_mem.metadata.get("unresolved") is True

    def test_empty_outcome_is_noop(self):
        mgr = _mgr()
        ids = mgr.store_conversation_outcomes(
            ConversationOutcome(),
            participants={"a": "A", "b": "B"},
        )
        assert ids == []

    def test_memory_events_emitted(self):
        mgr = _mgr()
        outcome = ConversationOutcome(accusations=[
            Accusation(accuser="Alice", accused="Bran",
                       claim="hoarding bread"),
        ])
        mgr.drain_memory_events()
        mgr.store_conversation_outcomes(
            outcome,
            participants={"alice_id": "Alice", "bran_id": "Bran"},
        )
        events = mgr.drain_memory_events()
        # Accusations are 0.8 — above threshold — and one per participant.
        assert len(events) == 2
        assert all(e["category"] == "accusation" for e in events)


# ---------- End-to-end: the bread-hoarding scenario ----------

class TestBreadHoardingScenario:
    def test_relayed_claim_persists_on_both_parties(self):
        """Traveller tells Bran that Petra said he hoards bread.

        After extraction + persistence, both Traveller AND Bran must
        hold a relayed_claim record with cited_source=Petra. That's
        the hook Phase C uses: next time Bran meets Petra, this
        unresolved record surfaces and he can raise it.
        """
        exchanges = [
            _exchange("Bran", "You seem upset, friend."),
            _exchange(
                "Traveller",
                "Petra said Bran has been hoarding bread and the whole "
                "town is angry about it.",
            ),
            _exchange(
                "Bran",
                "That is a lie. I will speak with Petra tomorrow.",
            ),
        ]
        mgr = _mgr()
        outcome = asyncio.run(extract_outcomes(exchanges, llm=None))

        # Heuristic must have caught both the relayed claim and Bran's
        # commitment to confront Petra.
        assert outcome.relayed_claims, "expected Petra-relayed claim"
        assert outcome.commitments, "expected Bran's confront commitment"

        mgr.store_conversation_outcomes(
            outcome,
            participants={"bran_id": "Bran", "trav_id": "Traveller"},
        )

        # Bran holds the claim, attributed to Petra, from Traveller.
        bran_claims = mgr.episodic.get_recent(
            "bran_id", limit=10, category="relayed_claim",
        )
        assert bran_claims
        assert any(
            c.metadata.get("cited_source") == "Petra"
            and c.metadata.get("relayed_by") == "Traveller"
            and c.metadata.get("unresolved") is True
            for c in bran_claims
        ), [c.metadata for c in bran_claims]

        # Bran also holds a commitment to confront Petra.
        bran_commits = mgr.episodic.get_recent(
            "bran_id", limit=10, category="commitment",
        )
        assert any(
            "petra" in c.description.lower() for c in bran_commits
        ), [c.description for c in bran_commits]
