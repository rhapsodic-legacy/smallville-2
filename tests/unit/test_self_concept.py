"""Tests for NPC self_concept and identity-claim injection."""

import pytest

from core.memory.reflection import (
    IdentityClaim,
    detect_identity_claims,
)
from core.npc.models import NPC, PersonalityTraits


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


# ---------- NPC.apply_self_concept_delta ----------

class TestSelfConceptDelta:
    def test_new_key_inserted(self):
        npc = _npc()
        npc.apply_self_concept_delta("role:king", 0.4)
        assert npc.self_concept == {"role:king": 0.4}

    def test_accumulates(self):
        npc = _npc()
        npc.apply_self_concept_delta("role:king", 0.4)
        npc.apply_self_concept_delta("role:king", 0.3)
        assert npc.self_concept["role:king"] == pytest.approx(0.7)

    def test_clamped_upper(self):
        npc = _npc()
        for _ in range(10):
            npc.apply_self_concept_delta("role:king", 0.4)
        assert npc.self_concept["role:king"] == 1.0

    def test_removed_when_below_floor(self):
        npc = _npc()
        npc.apply_self_concept_delta("role:king", 0.3)
        npc.apply_self_concept_delta("role:king", -0.3)
        # Now at zero — removed
        assert "role:king" not in npc.self_concept


# ---------- NPC.self_concept_summary ----------

class TestSelfConceptSummary:
    def test_empty_returns_empty(self):
        assert _npc().self_concept_summary() == ""

    def test_strong_belief_framed_unhedged(self):
        npc = _npc()
        npc.apply_self_concept_delta("role:king", 0.8)
        summary = npc.self_concept_summary()
        assert "a king" in summary
        assert "perhaps" not in summary

    def test_medium_belief_hedged(self):
        npc = _npc()
        npc.apply_self_concept_delta("role:king", 0.45)
        summary = npc.self_concept_summary()
        assert "perhaps" in summary.lower() or "wondering" in summary.lower()

    def test_sorts_by_confidence(self):
        npc = _npc()
        npc.apply_self_concept_delta("role:knight", 0.3)
        npc.apply_self_concept_delta("role:king", 0.9)
        summary = npc.self_concept_summary()
        # king (stronger) should appear before knight (weaker)
        assert summary.index("king") < summary.index("knight")

    def test_enemy_phrasing(self):
        npc = _npc()
        npc.apply_self_concept_delta("enemy_of:bran_1", 0.8)
        assert "enemy" in npc.self_concept_summary().lower()


# ---------- detect_identity_claims ----------

class TestDetectIdentityClaims:
    def test_role_assertion(self):
        claims = detect_identity_claims(
            [{"speaker": "Bran", "message": "You are a king among men."}],
            listener_name="Alice",
            speaker_id="bran_1",
        )
        assert any(c.key.startswith("role:king") for c in claims), claims

    def test_stopword_filtered(self):
        # "You are a fool" shouldn't become role:fool — fool is a stopword
        claims = detect_identity_claims(
            [{"speaker": "Bran", "message": "You are a fool, Alice."}],
            listener_name="Alice",
            speaker_id="bran_1",
        )
        for c in claims:
            assert c.key != "role:fool", claims

    def test_listener_own_line_ignored(self):
        claims = detect_identity_claims(
            [{"speaker": "Alice", "message": "I am a king."}],
            listener_name="Alice",
        )
        # Alice claiming about herself is not external; zero claims.
        assert claims == []

    def test_helper_claim(self):
        claims = detect_identity_claims(
            [{"speaker": "Bran", "message": "You helped us win the battle."}],
            listener_name="Alice",
            speaker_id="bran_1",
        )
        assert any(c.key.startswith("helped:") for c in claims), claims

    def test_enemy_claim(self):
        claims = detect_identity_claims(
            [{"speaker": "Bran", "message": "You are my enemy."}],
            listener_name="Alice",
            speaker_id="bran_1",
        )
        assert any(c.key.startswith("enemy_of:") for c in claims), claims

    def test_friend_claim(self):
        claims = detect_identity_claims(
            [{"speaker": "Bran", "message": "You are my friend."}],
            listener_name="Alice",
            speaker_id="bran_1",
        )
        assert any(c.key.startswith("friend_of:") for c in claims), claims

    def test_betrayed_claim(self):
        claims = detect_identity_claims(
            [{"speaker": "Bran", "message": "You betrayed us all."}],
            listener_name="Alice",
            speaker_id="bran_1",
        )
        assert any(c.key.startswith("betrayed:") for c in claims), claims

    def test_multiple_messages_scan(self):
        exchanges = [
            {"speaker": "Alice", "message": "Hello Bran."},
            {"speaker": "Bran", "message": "You are the king of Smallville!"},
            {"speaker": "Alice", "message": "You flatter me."},
        ]
        claims = detect_identity_claims(
            exchanges, listener_name="Alice", speaker_id="bran_1",
        )
        assert any("king" in c.key for c in claims), claims


# ---------- NPCManager._inject_self_concept_delta ----------

class TestInjectSelfConceptDelta:
    def _mgr_with_alice(self):
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=42,
        )
        mgr.spawn_population(2)
        return mgr, mgr.npcs[0]

    def test_claim_applied(self):
        mgr, npc = self._mgr_with_alice()
        claim = IdentityClaim(
            key="role:king", confidence_delta=0.4,
            source_text="You are a king.", speaker="Bran",
        )
        ok = mgr._inject_self_concept_delta(npc, claim, current_minutes=0.0)
        assert ok is True
        assert npc.self_concept.get("role:king") == pytest.approx(0.4)

    def test_contradicting_belief_rejects(self):
        mgr, npc = self._mgr_with_alice()
        # Seed strong friendship toward bran_1
        npc.self_concept["friend_of:bran_1"] = 0.8

        claim = IdentityClaim(
            key="enemy_of:bran_1", confidence_delta=0.4,
            source_text="You are my enemy.", speaker="Bran",
        )
        ok = mgr._inject_self_concept_delta(npc, claim, current_minutes=0.0)
        assert ok is False
        # Enemy belief was NOT added
        assert "enemy_of:bran_1" not in npc.self_concept
        # Friend belief unchanged
        assert npc.self_concept["friend_of:bran_1"] == 0.8

    def test_weak_contradiction_dampens(self):
        mgr, npc = self._mgr_with_alice()
        # Seed mild friendship
        npc.self_concept["friend_of:bran_1"] = 0.3

        claim = IdentityClaim(
            key="enemy_of:bran_1", confidence_delta=0.4,
            source_text="You are my enemy.", speaker="Bran",
        )
        ok = mgr._inject_self_concept_delta(npc, claim, current_minutes=0.0)
        assert ok is True
        # Applied but dampened — strictly less than the raw 0.4
        confidence = npc.self_concept.get("enemy_of:bran_1", 0.0)
        assert 0.0 < confidence < 0.4

    def test_multiple_reinforcing_claims_saturate(self):
        mgr, npc = self._mgr_with_alice()
        claim = IdentityClaim(
            key="role:king", confidence_delta=0.4,
            source_text="You are a king.", speaker="Crowd",
        )
        for _ in range(5):
            mgr._inject_self_concept_delta(npc, claim, current_minutes=0.0)
        assert npc.self_concept["role:king"] == 1.0


# ---------- Conversation prompt integration ----------

class TestConversationPromptIntegration:
    """self_concept should be threaded into the conversation prompts."""

    def test_summary_appears_in_prompt_template_args(self):
        # Rather than invoking LLM flow, we verify format_prompt
        # accepts the self_concept kwarg without crashing.
        from core.npc.llm_client import format_prompt

        out = format_prompt(
            "conversation_respond",
            name="Alice", age=30, occupation="farmer",
            personality="balanced",
            self_concept="You see yourself as: a king.",
            other_name="Bran", other_occupation="guard",
            other_message="Hello.",
            relationship_context="You are allies.",
        )
        assert "You see yourself as: a king." in out
