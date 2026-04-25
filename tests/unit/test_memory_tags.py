"""
Phase K — tag-based specific retention.

Covers:
- normalise_tag / normalise_tags canonicalise raw strings.
- EpisodicMemory.tags round-trips through add_memory + retrieval
  (fallback and semantic paths).
- The per-NPC tag index lights up on add, stays in sync after
  `delete_by_metadata` and `update_metadata`.
- retrieve_by_tags finds the Seren/Bran bread memory by cited_source.
- tags_for_* derivation covers commitment / accusation / relayed /
  town-event shapes.
- infer_tags_from_context unions partner + agenda + self_concept.
- retrieve_with_tag_boost lifts tag-matched memories above
  recency-only hits.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.conversation_outcomes import (
    Accusation, Commitment, ConversationOutcome, RelayedClaim,
)
from core.memory.episodic import (
    EpisodicStore, normalise_tag, normalise_tags,
)
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.models import NPC, PersonalityTraits


def _mgr() -> MemoryManager:
    mgr = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mgr.initialise()
    return mgr


# ---------- Normalisation ----------

class TestNormaliseTag:
    def test_lowercases(self):
        assert normalise_tag("PETRA") == "petra"

    def test_spaces_become_underscore(self):
        assert normalise_tag("missing bread") == "missing_bread"

    def test_punctuation_stripped(self):
        assert normalise_tag("agenda:repair_bridge!") == "agenda:repair_bridge"

    def test_empty_and_garbage_return_empty(self):
        assert normalise_tag("") == ""
        assert normalise_tag("!!!") == ""


class TestNormaliseTags:
    def test_list_input(self):
        assert normalise_tags(["Bread", "MISSING"]) == {"bread", "missing"}

    def test_string_input(self):
        assert normalise_tags("Bread missing") == {"bread", "missing"}

    def test_dedupes(self):
        assert normalise_tags(["bread", "BREAD", "Bread"]) == {"bread"}


# ---------- Store round-trip ----------

class TestStoreRoundTrip:
    def test_add_sets_tags(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        mid = store.add_memory(
            npc_id="a", description="hello",
            tags=["bread", "Traveller"],
        )
        mem = store.get_by_id(mid)
        assert mem.tags == {"bread", "traveller"}

    def test_tag_index_populates(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        mid = store.add_memory(
            npc_id="a", description="x",
            tags=["bread"],
        )
        assert mid in store._tag_index["a"]["bread"]

    def test_retrieve_by_tags_matches(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        store.add_memory(npc_id="a", description="x", tags=["bread"])
        store.add_memory(npc_id="a", description="y", tags=["festival"])
        hits = store.retrieve_by_tags("a", ["bread"])
        assert len(hits) == 1
        assert hits[0].description == "x"

    def test_retrieve_by_tags_scoped_per_npc(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        store.add_memory(npc_id="a", description="x", tags=["bread"])
        store.add_memory(npc_id="b", description="y", tags=["bread"])
        hits = store.retrieve_by_tags("a", ["bread"])
        assert len(hits) == 1
        assert hits[0].description == "x"

    def test_delete_maintains_index(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        store.add_memory(
            npc_id="a", description="x",
            tags=["bread"],
            extra_metadata={"conversation_id": "c1"},
        )
        assert store.retrieve_by_tags("a", ["bread"])
        store.delete_by_metadata("conversation_id", "c1")
        assert not store.retrieve_by_tags("a", ["bread"])

    def test_update_metadata_patches_tags(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        mid = store.add_memory(
            npc_id="a", description="x",
            tags=["bread"],
        )
        store.update_metadata(mid, {"tags": "festival"})
        assert not store.retrieve_by_tags("a", ["bread"])
        assert store.retrieve_by_tags("a", ["festival"])


# ---------- tags_for_* derivation ----------

class TestOutcomeTagDerivation:
    def test_commitment_tags(self):
        mgr = _mgr()
        c = Commitment(speaker="Bran", subject="speak with Petra tomorrow")
        tags = mgr.tags_for_commitment(c)
        assert "outcome:commitment" in tags
        assert "petra" in tags or "speak" in tags or "tomorrow" in tags

    def test_accusation_tags(self):
        mgr = _mgr()
        a = Accusation(
            accuser="Alice", accused="Bran", claim="hoarding bread",
        )
        tags = mgr.tags_for_accusation(a)
        assert "outcome:accusation" in tags
        assert "accused:bran" in tags
        assert "accuser:alice" in tags
        assert "hoarding" in tags or "bread" in tags

    def test_relayed_claim_tags_preserve_chain(self):
        mgr = _mgr()
        r = RelayedClaim(
            subject="Bran", claim="is hoarding bread",
            cited_source="Petra", relayed_by="Traveller",
        )
        tags = mgr.tags_for_relayed_claim(r)
        assert "outcome:relayed_claim" in tags
        assert "subject:bran" in tags
        assert "cited:petra" in tags
        assert "from:traveller" in tags

    def test_town_event_tags(self):
        mgr = _mgr()
        tags = mgr.tags_for_town_event("repair_bridge", "town_event")
        assert "agenda:repair_bridge" in tags


# ---------- Bread-scenario e2e at the memory layer ----------

class TestBreadScenarioTags:
    def test_retrieve_by_cited_source_finds_relayed_claim(self):
        mgr = _mgr()
        outcome = ConversationOutcome(relayed_claims=[RelayedClaim(
            subject="Bran", claim="is hoarding bread",
            cited_source="Petra", relayed_by="Traveller",
        )])
        mgr.store_conversation_outcomes(
            outcome,
            participants={"seren_id": "Seren", "trav_id": "Traveller"},
        )
        # Seren's tag probe for "petra" surfaces the relayed claim
        # even though Seren never directly spoke to Petra about bread.
        hits = mgr.retrieve_by_tags(
            "seren_id", ["cited:petra"],
        )
        assert hits, "Expected a bread-hoarding memory in Seren's cited:petra bucket"
        assert any("bread" in m.description.lower() for m in hits)


# ---------- infer_tags_from_context ----------

def _npc(**kwargs) -> NPC:
    return NPC(
        npc_id=kwargs.pop("npc_id", "alice"),
        name=kwargs.pop("name", "Alice"),
        age=30,
        personality=PersonalityTraits(),
        backstory="test",
        occupation="farmer",
        **kwargs,
    )


class TestInferTags:
    def test_unions_partner_agenda_self_concept(self):
        mgr = _mgr()
        npc = _npc()
        npc.self_concept["role:king"] = 0.9
        tags = mgr.infer_tags_from_context(
            npc,
            partner_name="Petra",
            active_agenda_titles=["Repair the old bridge"],
            recent_text="There is talk of hoarded bread.",
        )
        assert "petra" in tags
        assert "agenda:repair" in tags or "repair" in tags or "bridge" in tags
        assert "role_king" in tags
        assert any("hoard" in t or "bread" in t for t in tags)

    def test_empty_inputs_return_empty(self):
        mgr = _mgr()
        npc = _npc()
        assert mgr.infer_tags_from_context(npc) == set()


# ---------- retrieve_with_tag_boost ----------

class TestTagBoost:
    def test_tag_hit_rises_above_untagged(self):
        mgr = _mgr()
        # Older tagged memory
        mgr.episodic.add_memory(
            npc_id="a", description="Petra told me Bran hoards bread.",
            importance=0.5, game_time=100.0,
            tags=["bread", "cited:petra"],
        )
        # Newer untagged memory
        mgr.episodic.add_memory(
            npc_id="a", description="Gathered wheat today.",
            importance=0.5, game_time=200.0,
        )
        results = mgr.retrieve_with_tag_boost(
            npc_id="a",
            query="something about bread and petra",
            context_tags=["cited:petra"],
            current_game_time=200.0,
            limit=5,
        )
        assert results
        assert "bread" in results[0].memory.description.lower()

    def test_tag_only_hit_still_returned(self):
        mgr = _mgr()
        mgr.episodic.add_memory(
            npc_id="a",
            description="Hidden note about the old festival.",
            importance=0.3, game_time=50.0,
            tags=["festival"],
        )
        mgr.episodic.add_memory(
            npc_id="a",
            description="Completely unrelated thought about soup.",
            importance=0.5, game_time=200.0,
        )
        results = mgr.retrieve_with_tag_boost(
            npc_id="a",
            query="unrelated soup recipe",
            context_tags=["festival"],
            current_game_time=200.0,
            limit=5,
        )
        descriptions = [r.memory.description for r in results]
        assert any("festival" in d.lower() for d in descriptions)
