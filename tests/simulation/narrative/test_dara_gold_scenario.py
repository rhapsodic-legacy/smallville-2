"""
Narrative scenario: Dara hears about Bran's thousand gold.

The story:

  Traveller walks up to Dara the blacksmith and tells her
  "Bran said he wants to give you a thousand gold."

What the sim should exhibit:

  1. Dara's immediate reply is a real LLM answer — not a canned
     fallback like "Indeed, quite so."
  2. Dara forms a structured RELAYED_CLAIM memory naming Bran as
     the cited source, tagged with "bran" / "gold".
  3. The topic is discoverable via the Phase K tag index and
     Phase C unresolved-matters query.
  4. (Aspirational, not yet asserted) Dara eventually schedules a
     visit to Bran to confirm — this is Phase I territory and will
     become an assertion once progress-aware objectives ship.

Run:

    pytest -m narrative tests/simulation/narrative/test_dara_gold_scenario.py -v

Slow: each scenario makes several real Gemma calls, expect tens of
seconds per case. Auto-skips when Ollama / gemma model unavailable.
"""

from __future__ import annotations

import pytest

from tests.simulation.narrative.framework import (
    NarrativeSim, narrative_scenario,
)


# ---------- Core outcome: player chat is never canned ----------

@narrative_scenario
async def test_dara_response_is_not_a_canned_fallback(sim: NarrativeSim):
    """The headline promise of the force_llm fix: Dara answers in
    her own voice, not `_fallback_response`. Pipeline regression
    gate at the narrative level — complements the unit-level
    `test_player_chat_never_canned.py`."""
    sim.cast(dara="blacksmith", bran="merchant")
    reply = await sim.player_says(
        "Dara",
        "Hi Dara. Bran said he has a thousand gold he wants to give you.",
    )
    from core.npc.cognition.converse import _FALLBACK_RESPONSES_FOR_TEST
    assert reply, "Dara returned an empty reply"
    assert reply not in _FALLBACK_RESPONSES_FOR_TEST, (
        f"Dara's reply was a canned fallback: {reply!r}. "
        f"The force_llm=True path should NEVER produce a canned "
        f"string on player chats."
    )
    # Minimal sanity: reply should be non-trivial (more than a
    # four-word stub).
    assert len(reply.split()) >= 4, (
        f"Reply is suspiciously short: {reply!r}"
    )


# ---------- Outcome extraction: relayed claim forms ----------

@narrative_scenario
async def test_dara_forms_relayed_claim_about_bran_gold(sim: NarrativeSim):
    """After the chat ends, Dara's memory should contain a Phase B
    relayed_claim naming Bran and the gold topic."""
    sim.cast(dara="blacksmith", bran="merchant")
    await sim.player_says(
        "Dara",
        "Bran said he wants to give you a thousand gold. You should go talk to him.",
    )
    await sim.player_says(
        "Dara",
        "Seriously. Bran told me directly that he has a thousand gold for you.",
    )
    # Close the chat so `_persist_finished_conversations` runs,
    # which is what actually extracts + stores Phase B outcomes.
    await sim.player_closes_chat("Dara")
    # Let at least one cognition tick run so persistence fires.
    await sim.advance(minutes=20)

    sim.assert_has_memory(
        "Dara",
        category="relayed_claim",
        matches=("bran",),
    )


# ---------- Tag retention: the gold topic is discoverable ----------

@narrative_scenario
async def test_dara_memory_is_tagged_for_bran(sim: NarrativeSim):
    """Phase K: the relayed_claim persisted above should carry a
    tag anchoring it to Bran, so retrieval by tag surfaces it."""
    sim.cast(dara="blacksmith", bran="merchant")
    await sim.player_says(
        "Dara",
        "Bran said he has a thousand gold for you.",
    )
    await sim.player_closes_chat("Dara")
    await sim.advance(minutes=20)

    # Accept either the `cited:<name>` outcome tag or the bare
    # name tag — Phase K derivation produces both.
    sim.assert_tags_present(
        "Dara",
        tags=("bran", "cited:bran"),
    )


# ---------- Unresolved matters: the Phase C prompt block mentions it ----------

@narrative_scenario
async def test_dara_carries_unresolved_matter_about_bran(sim: NarrativeSim):
    """Phase C: the claim should surface when Dara's cognition
    pulls unresolved matters relating to Bran (e.g. next time she
    runs into Bran, the planner sees it in the prompt block)."""
    sim.cast(dara="blacksmith", bran="merchant")
    await sim.player_says(
        "Dara",
        "Bran told me he wants to give you a thousand gold.",
    )
    await sim.player_closes_chat("Dara")
    await sim.advance(minutes=20)

    bran = sim.npc("Bran")
    matters = sim.memory.retrieve_unresolved_matters(
        sim.npc("Dara").npc_id,
        partner_id=bran.npc_id,
        partner_name=bran.name,
    )
    if not matters:
        raise AssertionError(
            "No unresolved matter naming Bran after the chat. "
            "Phase C retrieval is not seeing the relayed_claim "
            "Dara just stored.\n\n"
            + sim.dump_memories("Dara")
        )
    # The matter's text should mention Bran.
    assert any(
        "bran" in (m.description or "").lower() for m in matters
    ), (
        "Matters found, but none reference Bran explicitly. "
        "Phase B outcome shape may have changed.\n\n"
        + sim.dump_memories("Dara")
    )
