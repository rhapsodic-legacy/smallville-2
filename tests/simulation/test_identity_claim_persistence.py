"""
Identity-claim persistence simulation test.

Exercises the full loop: a conversation partner makes an identity
claim ("you are a king"), the listener's self_concept is updated,
and the listener's *subsequent* conversation prompt reflects that
new identity.

This is the end-to-end guarantee that makes the feature visible:
- Reflection detects the claim
- Manager injects into self_concept via the contradiction filter
- Conversation layer reads from self_concept when building prompts
"""

from __future__ import annotations

import asyncio

import pytest

from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.npc.cognition.converse import (
    Conversation, ConversationExchange, _active_conversations,
    continue_conversation, initiate_conversation,
)
from core.world.generator import WorldConfig, generate_world


def _make_manager(pop: int = 4, seed: int = 99) -> NPCManager:
    config = WorldConfig(population=pop, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=MockProvider(), seed=seed,
    )
    mgr.spawn_population(pop)
    return mgr


def _inject_conversation(
    mgr: NPCManager, npc_a, npc_b, lines: list[tuple[str, str]],
) -> None:
    """Build and finish a conversation synthetically.

    Used to place a specific identity claim into the record without
    waiting for natural NPC-NPC chat. `lines` is [(speaker_name, text)].
    """
    conv = Conversation(npc_a_id=npc_a.npc_id, npc_b_id=npc_b.npc_id)
    for speaker_name, text in lines:
        speaker = npc_a if speaker_name == npc_a.name else npc_b
        conv.add_exchange(speaker.npc_id, speaker.name, text)
    conv.finished = True
    key = frozenset({npc_a.npc_id, npc_b.npc_id})
    _active_conversations[key] = conv


@pytest.fixture(autouse=True)
def _clear_conversations():
    _active_conversations.clear()
    yield
    _active_conversations.clear()


def test_role_claim_lands_in_self_concept():
    """A "you are a king" line persists into the listener's self_concept."""
    async def _run():
        mgr = _make_manager()
        alice, bran = mgr.npcs[0], mgr.npcs[1]

        _inject_conversation(mgr, bran, alice, [
            (bran.name, "You are a king among men, truly."),
            (alice.name, "You flatter me."),
            (bran.name, "I mean it — you are our rightful king."),
        ])

        await mgr._persist_finished_conversations(current_minutes=300.0)

        assert any(k.startswith("role:king") for k in alice.self_concept), (
            f"Expected role:king in alice.self_concept, got: {alice.self_concept}"
        )

    asyncio.new_event_loop().run_until_complete(_run())


def test_subsequent_conversation_prompt_contains_new_identity():
    """After the claim, the NPC's next conversation prompt carries the belief."""
    async def _run():
        mgr = _make_manager()
        alice, bran = mgr.npcs[0], mgr.npcs[1]

        # Plant a strong king belief directly (bypasses detection so the
        # test focuses on prompt reading, not detection). We still want
        # detection covered — that's what the first test is for.
        alice.self_concept["role:king"] = 0.9

        # Force both into tier 1 so continue_conversation uses the LLM,
        # and place them adjacent so continue_conversation accepts them.
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0

        # Spin up a fresh conversation with Bran speaking first.
        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name, "Good morrow, friend.")
        _active_conversations[frozenset({alice.npc_id, bran.npc_id})] = conv

        # Alice responds — this is the prompt we want to inspect.
        await continue_conversation(
            alice, bran, mgr.llm, memory_manager=mgr.memory,
        )

        # MockProvider records every call; the last conversation call
        # should include the self-concept line.
        convo_calls = [
            c for c in mgr.llm.call_log
            if c["purpose"] == "conversation"
        ]
        assert convo_calls, "Expected at least one conversation LLM call"
        prompt_text = " ".join(
            m["content"] for m in convo_calls[-1]["messages"]
        )
        assert "king" in prompt_text.lower(), (
            "Alice's self-concept (role:king) should be in the prompt. "
            f"Prompt was:\n{prompt_text}"
        )

    asyncio.new_event_loop().run_until_complete(_run())


def test_contradicting_claim_does_not_flip_identity():
    """Pre-existing friendship blocks an opportunistic 'you are my enemy'."""
    async def _run():
        mgr = _make_manager()
        alice, bran = mgr.npcs[0], mgr.npcs[1]

        # Alice already strongly considers bran a friend.
        alice.self_concept[f"friend_of:{bran.npc_id}"] = 0.85

        # Bran claims they're enemies — contradicts, must be rejected.
        _inject_conversation(mgr, bran, alice, [
            (bran.name, "You are my enemy, Alice."),
        ])
        await mgr._persist_finished_conversations(current_minutes=400.0)

        assert f"enemy_of:{bran.npc_id}" not in alice.self_concept
        assert alice.self_concept[f"friend_of:{bran.npc_id}"] == pytest.approx(
            0.85,
        )

    asyncio.new_event_loop().run_until_complete(_run())


def test_conversation_drift_nudges_personality():
    """A heavy-valence conversation moves Big-5 away from spawn baseline."""
    async def _run():
        mgr = _make_manager()
        alice, bran = mgr.npcs[0], mgr.npcs[1]

        baseline_neuroticism = alice.personality.neuroticism

        # Flood the conversation with fear-laden language.
        _inject_conversation(mgr, bran, alice, [
            (bran.name, "I am so afraid. I fear what the night brings."),
            (alice.name, "Terrifying. Anxiety has been my only companion."),
            (bran.name, "Dread hangs over the town. We are scared."),
        ])
        await mgr._persist_finished_conversations(current_minutes=500.0)

        # Fear content → neuroticism up, agreeableness unchanged or up.
        assert alice.personality.neuroticism > baseline_neuroticism, (
            f"Neuroticism should have risen. baseline={baseline_neuroticism} "
            f"now={alice.personality.neuroticism}"
        )
        # And stay bounded.
        assert 0.0 <= alice.personality.neuroticism <= 1.0

    asyncio.new_event_loop().run_until_complete(_run())
