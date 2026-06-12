"""Emergent write-paths arc — the pipes from LLM signal to durable state.

The diagnosis: the persona-conditioned LLM now GENERATES friction and
identity, but content-blind heuristics discarded it on arrival —
sentiment was written only by a talking-is-bonding baseline, and
self_concept only by regexes over other people's words. These tests
pin the new pipes (tone → sentiment, accusation → sentiment,
reflection SELF → self_concept) and the failure modes: hallucinated
keys must never reach self_concept, invalid tones must be dropped,
deltas must stay one-directional, and the shrunk baseline must allow
a personality clash to net negative.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.reflection import (
    REFLECTION_CLAIM_DELTA,
    parse_reflection_extras,
    reflect_on_conversation,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.models import NPC, PersonalityTraits
from core.npc.persona import PersonaForge
from core.relationships.sentiment import (
    CONVERSATION_TONE_DELTAS,
    SentimentTracker,
)


# ---------- Fixtures ----------

def _make_npc(name: str = "Vex", npc_id: str = "vex_1", **traits) -> NPC:
    npc = NPC(
        npc_id=npc_id,
        name=name,
        age=40,
        personality=PersonalityTraits(**traits),
        backstory="b",
        occupation="blacksmith",
        persona=PersonaForge.from_seed(7).forge("blacksmith"),
    )
    npc.cognition_tier = 1
    return npc


def _make_memory() -> MemoryManager:
    sentiment = SentimentTracker(db_path=":memory:")
    sentiment.initialise()
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
        sentiment=sentiment,
    )
    memory.initialise()
    return memory


class _StubLLM:
    """Returns a fixed response for the reflection call."""

    def __init__(self, response: str):
        self.response = response

    async def complete(self, **kwargs) -> str:
        return self.response


# ---------- Parser ----------

class TestParseReflectionExtras:
    def test_tone_parsed_and_stripped(self):
        for tone in ("warm", "neutral", "tense", "hostile"):
            insight, parsed, claim = parse_reflection_extras(
                f"I see Bran differently now.\nTONE: {tone}"
            )
            assert parsed == tone
            assert insight == "I see Bran differently now."
            assert claim is None

    def test_tone_case_and_trailing_commentary(self):
        _, tone, _ = parse_reflection_extras(
            "Something.\nTone: HOSTILE — he called me a thief."
        )
        assert tone == "hostile"

    def test_invalid_tone_dropped(self):
        insight, tone, _ = parse_reflection_extras(
            "Something.\nTONE: furious"
        )
        assert tone is None
        assert insight == "Something."

    def test_self_claim_parsed_normalised(self):
        _, _, claim = parse_reflection_extras(
            "I will not back this repair.\nSELF: opposes:repair bridge"
        )
        assert claim is not None
        assert claim.key == "opposes:repair_bridge"
        assert claim.confidence_delta == REFLECTION_CLAIM_DELTA

    def test_self_disallowed_prefix_dropped(self):
        for bad in ("built:bridge", "helped:town", "god:me", "unreliable:self"):
            _, _, claim = parse_reflection_extras(f"X.\nSELF: {bad}")
            assert claim is None, f"{bad} should never reach self_concept"

    def test_self_malformed_dropped(self):
        for bad in ("opposes", "role:", ":target", "role:a$b!", "SELF SELF"):
            _, _, claim = parse_reflection_extras(f"X.\nSELF: {bad}")
            assert claim is None

    def test_first_occurrence_wins(self):
        _, tone, claim = parse_reflection_extras(
            "X.\nTONE: tense\nTONE: warm\nSELF: role:mediator\nSELF: role:king"
        )
        assert tone == "tense"
        assert claim.key == "role:mediator"

    def test_no_extras_passthrough(self):
        text = "Just an ordinary insight about turnips."
        insight, tone, claim = parse_reflection_extras(text)
        assert (insight, tone, claim) == (text, None, None)

    def test_extras_only_yields_empty_insight(self):
        insight, tone, _ = parse_reflection_extras("TONE: warm")
        assert insight == ""
        assert tone == "warm"


# ---------- Tone → sentiment (one-directional) ----------

class TestToneSentiment:
    async def _reflect(self, response: str, memory=None, claim_sink=None):
        npc = _make_npc()
        memory = memory or _make_memory()
        insight = await reflect_on_conversation(
            npc, "Bran",
            [{"speaker": "Bran", "message": "That bridge is rot."}],
            memory, _StubLLM(response), current_game_time=100.0,
            other_id="bran_1", claim_sink=claim_sink,
        )
        return npc, memory, insight

    async def test_hostile_tone_writes_negative_one_directional(self):
        npc, memory, _ = await self._reflect(
            "Bran insulted my forge work.\nTONE: hostile"
        )
        towards = memory.sentiment.get(npc.npc_id, "bran_1")
        expected = CONVERSATION_TONE_DELTAS["hostile"]
        assert towards.trust == expected["trust"] < 0
        assert towards.affection == expected["affection"] < 0
        # One-directional: Bran's view of Vex is untouched.
        back = memory.sentiment.get("bran_1", npc.npc_id)
        assert back.is_default()

    async def test_warm_tone_writes_positive(self):
        npc, memory, _ = await self._reflect("Good talk.\nTONE: warm")
        towards = memory.sentiment.get(npc.npc_id, "bran_1")
        assert towards.trust > 0 and towards.affection > 0

    async def test_neutral_tone_writes_nothing(self):
        npc, memory, _ = await self._reflect("Fine.\nTONE: neutral")
        assert memory.sentiment.get(npc.npc_id, "bran_1").is_default()

    async def test_insight_recorded_without_trailer_lines(self):
        npc, memory, insight = await self._reflect(
            "Bran is hiding something about the timber.\nTONE: tense"
        )
        assert insight == "Bran is hiding something about the timber."
        mems = memory.episodic.get_recent(npc.npc_id, limit=5)
        assert any(
            "hiding something" in m.description
            and "TONE" not in m.description
            for m in mems
        )

    async def test_missing_other_id_no_crash_no_write(self):
        npc = _make_npc()
        memory = _make_memory()
        insight = await reflect_on_conversation(
            npc, "Bran", [{"speaker": "Bran", "message": "Hm."}],
            memory, _StubLLM("Insight.\nTONE: hostile"),
            current_game_time=1.0,
        )
        assert insight == "Insight."
        assert memory.sentiment.get(npc.npc_id, "bran_1").is_default()


# ---------- Reflection SELF → claim sink ----------

class TestReflectionSelfClaims:
    async def test_self_claim_routed_to_sink(self):
        npc = _make_npc()
        memory = _make_memory()
        received = []
        await reflect_on_conversation(
            npc, "Bran",
            [{"speaker": "Bran", "message": "The repair is on again."}],
            memory,
            _StubLLM(
                "I will not lend my hammer to patchwork.\n"
                "TONE: tense\nSELF: opposes:repair_bridge"
            ),
            current_game_time=5.0,
            other_id="bran_1",
            claim_sink=received.append,
        )
        assert len(received) == 1
        claim = received[0]
        assert claim.key == "opposes:repair_bridge"
        assert claim.speaker == npc.name
        assert claim.confidence_delta == REFLECTION_CLAIM_DELTA

    async def test_sink_error_does_not_break_reflection(self):
        npc = _make_npc()
        memory = _make_memory()

        def explode(claim):
            raise RuntimeError("sink boom")

        insight = await reflect_on_conversation(
            npc, "Bran", [{"speaker": "Bran", "message": "Hm."}],
            memory, _StubLLM("Insight.\nSELF: role:objector"),
            current_game_time=1.0, other_id="bran_1", claim_sink=explode,
        )
        assert insight == "Insight."


# ---------- Baseline shrink (clash can net negative) ----------

class TestContactBaseline:
    def test_clash_pair_nets_negative(self):
        from core.npc.cognition.converse import _conversation_sentiment_deltas

        blunt = _make_npc(
            "Vex", "vex_1",
            agreeableness=0.1, openness=0.1, neuroticism=0.9,
        )
        gentle = _make_npc(
            "Mira", "mira_1",
            agreeableness=0.9, openness=0.9, neuroticism=0.2,
        )
        gentle.occupation = "priest"
        gentle.skills = {}
        blunt.skills = {}
        deltas = _conversation_sentiment_deltas(blunt, gentle, exchange_count=1)
        assert sum(deltas.values()) < 0, (
            f"personality clash must be able to outweigh the contact "
            f"baseline, got {deltas}"
        )

    def test_friendly_pair_still_mildly_positive(self):
        from core.npc.cognition.converse import _conversation_sentiment_deltas

        a = _make_npc("Vex", "vex_1")
        b = _make_npc("Bran", "bran_1")
        deltas = _conversation_sentiment_deltas(a, b, exchange_count=3)
        total = sum(deltas.values())
        assert 0 < total < 6, (
            f"baseline should be mild bonding, not the old +9-ish: {deltas}"
        )

    def test_zero_deltas_filtered(self):
        from core.npc.cognition.converse import _conversation_sentiment_deltas

        a = _make_npc("Vex", "vex_1")
        b = _make_npc("Bran", "bran_1")
        deltas = _conversation_sentiment_deltas(a, b, exchange_count=1)
        assert all(abs(v) > 0 for v in deltas.values())


# ---------- Accusation → sentiment (manager path) ----------

class TestAccusationSentiment:
    def _make_manager(self):
        from core.npc.llm_client import MockProvider
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world

        config = WorldConfig(population=2, terrain="riverside", seed=55)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=55,
        )
        mgr.spawn_population(2)
        return mgr

    def test_direct_accusation_penalises_both_directions(self):
        from core.memory.conversation_outcomes import (
            Accusation, ConversationOutcome,
        )

        mgr = self._make_manager()
        a, b = mgr.npcs[0], mgr.npcs[1]
        # Spawn may seed initial relationships — measure the delta.
        before_b_a = mgr.sentiment.get(b.npc_id, a.npc_id).trust
        before_a_b_trust = mgr.sentiment.get(a.npc_id, b.npc_id).trust
        before_a_b_respect = mgr.sentiment.get(a.npc_id, b.npc_id).respect
        outcome = ConversationOutcome(accusations=[
            Accusation(accuser=a.name, accused=b.name, claim="stole timber"),
        ])
        applied = mgr._apply_accusation_sentiment(
            outcome,
            participants={a.npc_id: a.name, b.npc_id: b.name},
            current_minutes=10.0,
        )
        assert applied == 1
        # Accused resents accuser; accuser distrusts accused.
        assert mgr.sentiment.get(b.npc_id, a.npc_id).trust < before_b_a
        assert mgr.sentiment.get(a.npc_id, b.npc_id).trust < before_a_b_trust
        assert mgr.sentiment.get(a.npc_id, b.npc_id).respect < before_a_b_respect

    def test_third_party_accusation_not_applied(self):
        from core.memory.conversation_outcomes import (
            Accusation, ConversationOutcome,
        )

        mgr = self._make_manager()
        a, b = mgr.npcs[0], mgr.npcs[1]
        before = mgr.sentiment.get(b.npc_id, a.npc_id).trust
        outcome = ConversationOutcome(accusations=[
            Accusation(accuser=a.name, accused="Theron", claim="lied"),
        ])
        applied = mgr._apply_accusation_sentiment(
            outcome,
            participants={a.npc_id: a.name, b.npc_id: b.name},
            current_minutes=10.0,
        )
        assert applied == 0
        assert mgr.sentiment.get(b.npc_id, a.npc_id).trust == before


# ---------- End-to-end wiring through the real manager path ----------

class TestEndToEndWiring:
    """A finished conversation flowing through
    `_persist_finished_conversations` must land tone in the sentiment
    table and a SELF assertion in self_concept — this is the test
    that fails if anyone disconnects the pipes (drops the claim_sink
    lambda, stops passing other_id, reorders the reflection pass).
    """

    async def test_tone_and_self_flow_from_conversation_end(self):
        from core.npc.llm_client import MockProvider
        from core.npc.manager import NPCManager
        from core.npc.cognition.converse import (
            Conversation, _active_conversations,
        )
        from core.world.generator import WorldConfig, generate_world

        class ScriptedReflectionProvider(MockProvider):
            async def complete(self, system="", messages=None,
                               max_tokens=300, temperature=0.7,
                               purpose="general", **kwargs):
                if purpose == "reflection":
                    return (
                        "I'll not be spoken to like that over honest "
                        "timber.\nTONE: hostile\nSELF: rival_of:bran"
                    )
                return await super().complete(
                    system=system, messages=messages or [],
                    max_tokens=max_tokens, temperature=temperature,
                    purpose=purpose, **kwargs,
                )

        config = WorldConfig(population=2, terrain="riverside", seed=55)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings,
            llm=ScriptedReflectionProvider(), seed=55,
        )
        a, b = mgr.spawn_population(2)
        a.cognition_tier = 1
        b.cognition_tier = 1

        conv = Conversation(npc_a_id=a.npc_id, npc_b_id=b.npc_id)
        conv.add_exchange(a.npc_id, a.name, "Your beams are warped rubbish.")
        conv.add_exchange(b.npc_id, b.name, "Say that again and mean it.")
        conv.finished = True
        key = frozenset({a.npc_id, b.npc_id})
        _active_conversations[key] = conv

        trust_before = mgr.sentiment.get(a.npc_id, b.npc_id).trust
        try:
            await mgr._persist_finished_conversations(current_minutes=50.0)
        finally:
            _active_conversations.pop(key, None)

        # Tone: hostile reflection dropped trust a→b (and b→a).
        assert mgr.sentiment.get(a.npc_id, b.npc_id).trust < trust_before
        # Self: the reflection's SELF line reached self_concept via
        # the manager's contradiction-damped applier.
        assert a.self_concept.get("rival_of:bran", 0) > 0
        assert b.self_concept.get("rival_of:bran", 0) > 0
