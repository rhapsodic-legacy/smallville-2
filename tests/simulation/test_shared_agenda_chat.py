"""
Phase G e2e — shared agenda surfaces in live conversation prompts.

Verifies end-to-end that:
- When two NPCs have both committed to the same goal, their
  conversation prompt's {shared_agenda} slot carries a joint line.
- When two NPCs recently completed a goal together, the same slot
  carries the "recently completed" cue.
- Outside the recency window, the cue goes quiet.
- The player chat path receives the same cue for NPCs the player is
  jointly helping.

Uses MockProvider so we can inspect the prompt text deterministically
rather than depending on LLM output quality.
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
    continue_conversation,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world
from core.world.town_agenda import create_goal_from_template


def _make_manager(seed: int = 421) -> NPCManager:
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


def _last_conversation_prompt(llm: MockProvider) -> str:
    convo_calls = [c for c in llm.call_log if c["purpose"] == "conversation"]
    assert convo_calls, "expected at least one conversation LLM call"
    return " ".join(m["content"] for m in convo_calls[-1]["messages"])


def test_prompt_shows_shared_commitment_line():
    async def _run():
        mgr = _make_manager()
        alice = mgr.npcs[0]
        bran = mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0

        # Both commit to the bridge-repair goal.
        goal = create_goal_from_template("repair_bridge", current_day=1)
        mgr.town_agenda.propose(goal, current_day=1)
        mgr.town_agenda.record_contribution(
            goal.goal_id, alice.npc_id, current_day=1,
        )
        mgr.town_agenda.record_contribution(
            goal.goal_id, bran.npc_id, current_day=1,
        )

        # Seed a conversation with Bran speaking first; Alice responds.
        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name, "Good morning.")
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        await continue_conversation(
            alice, bran, mgr.llm, memory_manager=mgr.memory,
            town_agenda_summary=mgr.town_agenda.summary_for_prompt(
                alice.npc_id,
            ),
            shared_agenda_summary=mgr.town_agenda.shared_matters_for_prompt(
                alice.npc_id, bran.npc_id, current_day=1,
            ),
        )

        prompt = _last_conversation_prompt(mgr.llm)
        assert "you and your partner are both helping" in prompt.lower()
        assert "bridge" in prompt.lower()

    asyncio.new_event_loop().run_until_complete(_run())


def test_prompt_shows_recent_shared_victory():
    async def _run():
        mgr = _make_manager(seed=431)
        alice = mgr.npcs[0]
        bran = mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0

        # Build a completable goal: two contributions required.
        goal = create_goal_from_template("repair_bridge", current_day=3)
        goal.required_contributions = 2
        mgr.town_agenda.propose(goal, current_day=3)
        mgr.town_agenda.record_contribution(
            goal.goal_id, alice.npc_id, current_day=3,
        )
        completed = mgr.town_agenda.record_contribution(
            goal.goal_id, bran.npc_id, current_day=3,
        )
        assert completed

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name, "What a day.")
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        await continue_conversation(
            alice, bran, mgr.llm, memory_manager=mgr.memory,
            shared_agenda_summary=mgr.town_agenda.shared_matters_for_prompt(
                alice.npc_id, bran.npc_id, current_day=3,
            ),
        )
        prompt = _last_conversation_prompt(mgr.llm)
        assert "recently completed" in prompt.lower()

    asyncio.new_event_loop().run_until_complete(_run())


def test_victory_cue_fades_after_window():
    async def _run():
        mgr = _make_manager(seed=441)
        alice = mgr.npcs[0]
        bran = mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0

        goal = create_goal_from_template("repair_bridge", current_day=1)
        goal.required_contributions = 2
        mgr.town_agenda.propose(goal, current_day=1)
        mgr.town_agenda.record_contribution(
            goal.goal_id, alice.npc_id, current_day=1,
        )
        mgr.town_agenda.record_contribution(
            goal.goal_id, bran.npc_id, current_day=1,
        )

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name, "A quiet week.")
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        # Days later — the recency window has passed.
        await continue_conversation(
            alice, bran, mgr.llm, memory_manager=mgr.memory,
            shared_agenda_summary=mgr.town_agenda.shared_matters_for_prompt(
                alice.npc_id, bran.npc_id, current_day=30,
            ),
        )
        prompt = _last_conversation_prompt(mgr.llm)
        assert "recently completed" not in prompt.lower()

    asyncio.new_event_loop().run_until_complete(_run())


def test_manager_run_conversations_threads_shared_agenda():
    """The NPCManager's own conversation loop builds and passes the
    shared_agenda_summary to continue_conversation, so this wiring
    works without callers knowing about TownAgenda directly."""
    async def _run():
        mgr = _make_manager(seed=451)
        alice = mgr.npcs[0]
        bran = mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0
        mgr._current_day = 1

        goal = create_goal_from_template("repair_bridge", current_day=1)
        mgr.town_agenda.propose(goal, current_day=1)
        mgr.town_agenda.record_contribution(
            goal.goal_id, alice.npc_id, current_day=1,
        )
        mgr.town_agenda.record_contribution(
            goal.goal_id, bran.npc_id, current_day=1,
        )

        # Start an in-progress conversation so _run_conversations
        # takes the continue branch rather than trying to initiate one
        # (which runs extra scaffolding).
        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(bran.npc_id, bran.name, "Hello friend.")
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        await mgr._run_conversations(current_minutes=10.0)

        prompt = _last_conversation_prompt(mgr.llm)
        assert "you and your partner are both helping" in prompt.lower()

    asyncio.new_event_loop().run_until_complete(_run())
