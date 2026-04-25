"""
Regression gate: a player chat must NEVER produce a canned
`_fallback_response` string, regardless of the NPC's cognition
tier.

This is the pipeline that should have prevented the "Indeed, quite
so." / "Interesting. I hadn't thought of it that way." silent
stub replies reported on 2026-04-22 (v2 of the house-staying bug
— the LLM wasn't being called at all for player chats against
tier-3/4 NPCs, and the canned fallback was indistinguishable from
a real reply).

Two guarantees:

1. `continue_conversation(..., force_llm=True)` takes the LLM path
   even when the NPC is at a tier whose config says `uses_llm=False`.
2. When `force_llm=True` and the LLM raises, the error propagates
   rather than being swallowed into a canned string. Player-facing
   code can then surface the real cause to the UI.

The assertion set also actively hunts for `_fallback_response`
strings in the reply, so any future regression that routes the
player path back into canned-output territory trips this test.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.converse import (
    _FALLBACK_RESPONSES_FOR_TEST,
    Conversation, _active_conversations,
    continue_conversation,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


@pytest.fixture(autouse=True)
def _clear_conv_registry():
    _active_conversations.clear()
    yield
    _active_conversations.clear()


def _mgr() -> NPCManager:
    config = WorldConfig(population=3, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=MockProvider(), seed=42, memory=memory,
    )
    mgr.spawn_population(3)
    return mgr


def _seed_chat(mgr: NPCManager, player_text: str) -> tuple:
    """Start a fresh conversation with one player utterance."""
    target, player = mgr.npcs[0], mgr.npcs[1]
    player.npc_id = "player"
    player.name = "Traveller"
    key = frozenset({target.npc_id, player.npc_id})
    conv = Conversation(npc_a_id=target.npc_id, npc_b_id=player.npc_id)
    conv.add_exchange(player.npc_id, player.name, player_text)
    _active_conversations[key] = conv
    return target, player, conv


class TestForceLlmBypassesTierGate:
    """A tier-3 NPC (uses_llm=False in its TierConfig) must still
    hit the LLM path when force_llm=True."""

    def test_tier_3_force_llm_still_calls_provider(self):
        async def _run():
            mgr = _mgr()
            target, player, conv = _seed_chat(
                mgr, "Bran said you hoard bread."
            )
            target.cognition_tier = 3  # would normally return fallback

            fake_llm = AsyncMock()
            fake_llm.complete = AsyncMock(
                return_value="That's quite the accusation.",
            )

            await continue_conversation(
                target, player, fake_llm, mgr.memory,
                force_llm=True,
            )

            # LLM was called (tier gate bypassed).
            assert fake_llm.complete.await_count == 1
            # Reply is the LLM's actual output.
            npc_reply = conv.exchanges[-1].message
            assert npc_reply == "That's quite the accusation."

        asyncio.run(_run())

    def test_tier_3_without_force_still_gets_fallback(self):
        """Sanity: the default (non-force) path preserves the
        existing tier-gate behaviour for NPC↔NPC chats so we
        don't accidentally spend LLM budget on background NPCs."""
        async def _run():
            mgr = _mgr()
            target, player, conv = _seed_chat(mgr, "Hello.")
            target.cognition_tier = 3

            fake_llm = AsyncMock()
            fake_llm.complete = AsyncMock()
            await continue_conversation(
                target, player, fake_llm, mgr.memory,
                force_llm=False,
            )
            assert fake_llm.complete.await_count == 0
            # Canned fallback path used instead.
            assert conv.exchanges[-1].message in _FALLBACK_RESPONSES_FOR_TEST

        asyncio.run(_run())


class TestForceLlmPropagatesErrors:
    """When the LLM raises during a forced player chat, the error
    must surface — not be quietly swallowed into a canned string."""

    def test_llm_exception_raises_not_fallback(self):
        async def _run():
            mgr = _mgr()
            target, player, conv = _seed_chat(mgr, "Hi Dara.")
            target.cognition_tier = 1

            fake_llm = AsyncMock()
            fake_llm.complete = AsyncMock(
                side_effect=RuntimeError("network down"),
            )

            with pytest.raises(RuntimeError, match="network down"):
                await continue_conversation(
                    target, player, fake_llm, mgr.memory,
                    force_llm=True,
                )

            # The bogus canned string must NOT have landed as an
            # exchange — the caller should have seen the exception.
            assert len(conv.exchanges) == 1  # only the player's line

        asyncio.run(_run())


class TestCannedStringDetector:
    """The real pipeline: given the actual list of canned
    fallback strings, assert that no response ever matches one
    on a forced player chat."""

    @pytest.mark.parametrize("tier", [1, 2, 3, 4])
    def test_forced_player_chat_response_is_never_canned(self, tier):
        """For every tier, force_llm=True uses the LLM and never
        returns a canned fallback string."""
        async def _run():
            mgr = _mgr()
            target, player, conv = _seed_chat(
                mgr, "Hi Dara. Bran said he has 1000 gold for you.",
            )
            target.cognition_tier = tier
            fake_llm = AsyncMock()
            fake_llm.complete = AsyncMock(
                return_value="Bran? A thousand gold? Hmph, I'll believe it when I see it.",
            )

            await continue_conversation(
                target, player, fake_llm, mgr.memory,
                force_llm=True,
            )

            reply = conv.exchanges[-1].message
            assert reply not in _FALLBACK_RESPONSES_FOR_TEST, (
                f"Tier {tier} forced chat landed on canned reply: "
                f"{reply!r}. This is exactly the bug "
                f"test_player_chat_never_canned guards against."
            )
            assert fake_llm.complete.await_count == 1

        asyncio.run(_run())
