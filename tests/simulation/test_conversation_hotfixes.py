"""
Hotfixes shipped 2026-04-20 (second pass) after live-play regression:

1. Reflection must not block the cognition tick indefinitely — each
   call is bounded at 15s and the two participants reflect
   concurrently. Previously a slow LLM froze every NPC for a whole
   game day.

2. `conversation_respond` now carries a short "Recent conversation
   so far" block so the NPC doesn't forget what was discussed two
   turns ago.

3. After a notable conversation, the reflection produces an insight
   AND runs `classify_insight` to see whether a physical action is
   needed ("go talk to Dara"). Purely internal reflections
   ("I distrust Bob") classify as NO_ACTION and don't schedule.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.reflection import ActionIntent
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.converse import (
    Conversation, ConversationExchange,
    _active_conversations, _format_recent_history, continue_conversation,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


def _make_manager(seed: int = 811) -> NPCManager:
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


# ---------- #2 Recent history rendering ----------

class TestRecentHistory:
    def test_empty_when_only_one_exchange(self):
        convo = [ConversationExchange("a", "A", "hi")]
        assert _format_recent_history(convo) == ""

    def test_empty_when_zero_exchanges(self):
        assert _format_recent_history([]) == ""

    def test_renders_prior_turns_excluding_latest(self):
        convo = [
            ConversationExchange("a", "Alice", "You hoard bread."),
            ConversationExchange("b", "Bran", "That is a lie."),
            ConversationExchange("a", "Alice", "Petra said so."),
        ]
        out = _format_recent_history(convo)
        # Last exchange ("Petra said so.") is excluded — it's the
        # line being replied to.
        assert "Alice: \"You hoard bread.\"" in out
        assert "Bran: \"That is a lie.\"" in out
        assert "Petra said so." not in out

    def test_caps_at_five_turns(self):
        convo = [
            ConversationExchange("a", f"S{i}", f"line {i}")
            for i in range(10)
        ]
        out = _format_recent_history(convo)
        # Drops the last (being replied to); keeps the 5 before it.
        # So it should contain lines 4-8 but not 0-3 or 9.
        assert "line 9" not in out  # excluded (latest)
        assert "line 8" in out
        assert "line 4" in out
        assert "line 3" not in out


# ---------- Continue-conversation prompt includes history ----------

def test_continue_conversation_prompt_includes_recent_history():
    async def _run():
        mgr = _make_manager()
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        alice.x, alice.z = 5.0, 5.0
        bran.x, bran.z = 6.0, 5.0

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(alice.npc_id, alice.name, "Dara sent me.")
        conv.add_exchange(bran.npc_id, bran.name, "Dara has what to say?")
        conv.add_exchange(
            alice.npc_id, alice.name,
            "She said you're planning a wedding with Seren.",
        )
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        # Bran replies to Alice's last message — the prompt must
        # carry the earlier "Dara sent me" context so Bran doesn't
        # drift off topic.
        await continue_conversation(
            bran, alice, mgr.llm, memory_manager=mgr.memory,
        )
        convo_calls = [
            c for c in mgr.llm.call_log if c["purpose"] == "conversation"
        ]
        assert convo_calls
        prompt = " ".join(
            m["content"] for m in convo_calls[-1]["messages"]
        )
        assert "Recent conversation so far" in prompt
        assert "Dara sent me" in prompt
        # Most recent line still rendered separately.
        assert "wedding with Seren" in prompt

    asyncio.new_event_loop().run_until_complete(_run())


# ---------- #1 Reflection is bounded ----------

def test_reflection_timeout_does_not_block_tick():
    """Simulate a hanging reflection LLM. The cognition tick must
    complete within a bounded window regardless."""
    async def _run():
        mgr = _make_manager(seed=822)
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1

        # Install a hang-forever patch on the LLM's `complete` so
        # reflection blocks. Other LLM calls should not be routed
        # through here during this single _persist call — if they
        # are, they'll hit the same timeout and we recover anyway.
        async def _hang(*_a, **_kw):
            await asyncio.sleep(60)  # would hang the tick before fix

        original = mgr.llm.complete
        mgr.llm.complete = _hang  # type: ignore[assignment]

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(
            alice.npc_id, alice.name,
            "You are a thief. I accuse you of stealing bread!",
        )
        conv.add_exchange(
            bran.npc_id, bran.name, "That is slander.",
        )
        conv.finished = True
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        try:
            # The 30s guard here is the test's own safety rail —
            # the production code should return well under 16s
            # (one 15s reflection timeout + small overhead).
            await asyncio.wait_for(
                mgr._persist_finished_conversations(current_minutes=1000.0),
                timeout=30.0,
            )
        finally:
            mgr.llm.complete = original  # type: ignore[assignment]

    asyncio.new_event_loop().run_until_complete(_run())


# ---------- #3 Action intent injection ----------

def test_reflection_with_action_intent_injects_schedule_entry():
    """When reflection produces an insight that classifies as
    actionable, the NPC's schedule gets a temporary entry."""
    async def _run():
        mgr = _make_manager(seed=833)
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        # Give both NPCs non-empty schedules so _inject_reflection_entry
        # has something to insert into.
        from core.npc.models import ScheduleEntry
        for npc_ in (alice, bran):
            npc_.daily_schedule = [
                ScheduleEntry(
                    slot="afternoon", activity="work", location="work",
                    priority=5, duration_minutes=240,
                ),
            ]

        # Stub the LLM to deterministically return a reflection and
        # then a parseable ACTION block for classification.
        async def _scripted(*, system: str, messages, purpose: str,
                            **_kw):
            if purpose == "reflection":
                # First call of the pair is the insight; the second
                # is the classify. Infer by message content.
                user = messages[0]["content"]
                if "physical action" in user.lower():
                    return (
                        "ACTION: go speak with Dara about the matter\n"
                        "LOCATION: tavern\n"
                        "DURATION: 30"
                    )
                return (
                    "I must speak with Dara and clear up the matter."
                )
            return "ok"

        mgr.llm.complete = _scripted  # type: ignore[assignment]

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(
            alice.npc_id, alice.name,
            "Dara told me you were hoarding bread — she accused you.",
        )
        conv.add_exchange(
            bran.npc_id, bran.name,
            "This is the first I've heard of it.",
        )
        conv.finished = True
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        before_len = len(bran.daily_schedule)
        await mgr._persist_finished_conversations(current_minutes=500.0)

        # Either Alice or Bran should now have an extra schedule entry
        # representing the action intent ("go speak with Dara").
        after_len = max(
            len(alice.daily_schedule), len(bran.daily_schedule),
        )
        assert after_len > before_len, (
            f"expected a schedule injection from the action intent, "
            f"alice={alice.daily_schedule}, bran={bran.daily_schedule}"
        )
        all_activities = [
            e.activity for e in alice.daily_schedule + bran.daily_schedule
        ]
        assert any(
            "dara" in a.lower() for a in all_activities
        ), all_activities

    asyncio.new_event_loop().run_until_complete(_run())


def test_no_action_classification_does_not_inject():
    """An internal-only insight ('I distrust Bob') must NOT produce a
    schedule injection."""
    async def _run():
        mgr = _make_manager(seed=844)
        alice, bran = mgr.npcs[0], mgr.npcs[1]
        alice.cognition_tier = 1
        bran.cognition_tier = 1
        from core.npc.models import ScheduleEntry
        for npc_ in (alice, bran):
            npc_.daily_schedule = [
                ScheduleEntry(
                    slot="afternoon", activity="work", location="work",
                    priority=5, duration_minutes=240,
                ),
            ]

        async def _scripted(*, system: str, messages, purpose: str,
                            **_kw):
            if purpose == "reflection":
                user = messages[0]["content"]
                if "physical action" in user.lower():
                    return "NO_ACTION"
                return "I distrust Alice a little after that exchange."
            return "ok"

        mgr.llm.complete = _scripted  # type: ignore[assignment]

        conv = Conversation(npc_a_id=alice.npc_id, npc_b_id=bran.npc_id)
        conv.add_exchange(
            alice.npc_id, alice.name,
            "You are a liar and I won't forget it.",
        )
        conv.add_exchange(bran.npc_id, bran.name, "Think as you will.")
        conv.finished = True
        _active_conversations[
            frozenset({alice.npc_id, bran.npc_id})
        ] = conv

        before_alice = len(alice.daily_schedule)
        before_bran = len(bran.daily_schedule)
        await mgr._persist_finished_conversations(current_minutes=600.0)
        assert len(alice.daily_schedule) == before_alice
        assert len(bran.daily_schedule) == before_bran

    asyncio.new_event_loop().run_until_complete(_run())
