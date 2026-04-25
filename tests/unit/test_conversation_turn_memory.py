"""
Phase A — per-turn conversation memory persistence.

Covers:
- MemoryManager.persist_conversation_turn writes one episodic memory
  per participant, tagged with conv_id.
- _score_turn_importance lifts emotionally-weighted lines.
- persist_new_exchanges is idempotent and cursor-driven.
- consolidate_conversation_turns removes per-turn entries by conv_id.
- memory_formed events are queued for notable turns only.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.memory.structured import StructuredMemory
from core.memory.spatial import SpatialMemory


def _mgr() -> MemoryManager:
    mgr = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mgr.initialise()
    return mgr


class _FakeNpc:
    def __init__(self, npc_id: str, name: str, x: int = 0, z: int = 0):
        self.npc_id = npc_id
        self.name = name
        self.x = x
        self.z = z


class _FakeExchange:
    def __init__(self, speaker_name: str, message: str):
        self.speaker_name = speaker_name
        self.message = message


class _FakeConv:
    def __init__(self, conv_id: str = "conv-xyz"):
        self.conv_id = conv_id
        self.exchanges: list[_FakeExchange] = []
        self.persisted_exchange_count = 0

    def add(self, speaker: str, msg: str) -> None:
        self.exchanges.append(_FakeExchange(speaker, msg))


# ---------- persist_conversation_turn ----------

class TestPersistTurn:
    def test_writes_one_memory_per_participant(self):
        mgr = _mgr()
        ids = asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="alice", npc_b_id="bran",
            npc_a_name="Alice", npc_b_name="Bran",
            exchange={"speaker": "Alice", "message": "Hello there."},
            game_time=100.0,
        ))
        assert len(ids) == 2

        alice_mem = mgr.episodic.get_recent("alice", limit=5)
        bran_mem = mgr.episodic.get_recent("bran", limit=5)
        assert len(alice_mem) == 1 and len(bran_mem) == 1
        assert "Alice" in alice_mem[0].description
        assert "Hello there" in alice_mem[0].description
        assert alice_mem[0].metadata.get("conversation_id") == "c1"

    def test_empty_message_skipped(self):
        mgr = _mgr()
        ids = asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchange={"speaker": "A", "message": "   "},
        ))
        assert ids == []

    def test_high_keyword_gets_high_importance(self):
        mgr = _mgr()
        asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchange={"speaker": "A", "message": "You are accusing me of being a liar!"},
        ))
        stored = mgr.episodic.get_recent("a", limit=1)[0]
        assert stored.importance >= mgr.TURN_MEMORY_HIGH_IMPORTANCE

    def test_neutral_turn_default_importance(self):
        mgr = _mgr()
        asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchange={"speaker": "A", "message": "Good morning."},
        ))
        stored = mgr.episodic.get_recent("a", limit=1)[0]
        assert stored.importance == pytest.approx(
            mgr.TURN_MEMORY_DEFAULT_IMPORTANCE,
        )


# ---------- persist_new_exchanges (cursor) ----------

class TestPersistNewExchanges:
    def test_cursor_prevents_duplicates(self):
        mgr = _mgr()
        a, b = _FakeNpc("a", "A"), _FakeNpc("b", "B")
        conv = _FakeConv("c1")
        conv.add("A", "First line.")

        ids1 = asyncio.run(mgr.persist_new_exchanges(conv, a, b))
        assert len(ids1) == 2  # one memory per participant
        assert conv.persisted_exchange_count == 1

        # Re-run without adding — should no-op
        ids2 = asyncio.run(mgr.persist_new_exchanges(conv, a, b))
        assert ids2 == []
        assert conv.persisted_exchange_count == 1

        # Add a new line; only that one gets persisted
        conv.add("B", "Second line.")
        ids3 = asyncio.run(mgr.persist_new_exchanges(conv, a, b))
        assert len(ids3) == 2
        assert conv.persisted_exchange_count == 2
        assert mgr.episodic.count("a") == 2


# ---------- consolidate_conversation_turns ----------

class TestConsolidate:
    def test_sweeps_by_conv_id(self):
        mgr = _mgr()
        a, b = _FakeNpc("a", "A"), _FakeNpc("b", "B")
        conv = _FakeConv("c1")
        conv.add("A", "X")
        conv.add("B", "Y")
        asyncio.run(mgr.persist_new_exchanges(conv, a, b))
        assert mgr.episodic.count("a") == 2

        removed = mgr.consolidate_conversation_turns("c1")
        assert removed == 4  # 2 turns x 2 participants
        assert mgr.episodic.count("a") == 0
        assert mgr.episodic.count("b") == 0

    def test_unknown_conv_id_returns_zero(self):
        mgr = _mgr()
        assert mgr.consolidate_conversation_turns("never-seen") == 0


# ---------- memory_events broadcast queue ----------

class TestMemoryEvents:
    def test_notable_turn_emits_event(self):
        mgr = _mgr()
        asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchange={"speaker": "A", "message": "You are a liar and I accuse you!"},
        ))
        events = mgr.drain_memory_events()
        assert len(events) == 2
        assert {e["npc_id"] for e in events} == {"a", "b"}
        assert all(e["importance"] >= mgr.MEMORY_EVENT_THRESHOLD for e in events)

    def test_neutral_turn_emits_nothing(self):
        mgr = _mgr()
        asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchange={"speaker": "A", "message": "Good morning."},
        ))
        # default importance (0.45) is below the 0.6 threshold
        assert mgr.drain_memory_events() == []

    def test_drain_clears_queue(self):
        mgr = _mgr()
        asyncio.run(mgr.persist_conversation_turn(
            conversation_id="c1",
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchange={"speaker": "A", "message": "Enemy approaching the town!"},
        ))
        assert len(mgr.drain_memory_events()) == 2
        assert mgr.drain_memory_events() == []

    def test_record_conversation_emits_event(self):
        mgr = _mgr()
        asyncio.run(mgr.record_conversation(
            npc_a_id="a", npc_b_id="b",
            npc_a_name="A", npc_b_name="B",
            exchanges=[
                {"speaker": "A", "message": "Hello."},
                {"speaker": "B", "message": "Hi."},
            ],
            game_time=200.0,
        ))
        events = mgr.drain_memory_events()
        # record_conversation writes at importance 0.6, so both NPCs
        # should produce one event each.
        assert len(events) == 2


# ---------- Conversation dataclass fields ----------

class TestConversationFields:
    def test_conv_id_auto_assigned(self):
        from core.npc.cognition.converse import Conversation
        c1 = Conversation(npc_a_id="a", npc_b_id="b")
        c2 = Conversation(npc_a_id="a", npc_b_id="b")
        assert c1.conv_id and c2.conv_id
        assert c1.conv_id != c2.conv_id

    def test_persisted_cursor_starts_at_zero(self):
        from core.npc.cognition.converse import Conversation
        c = Conversation(npc_a_id="a", npc_b_id="b")
        assert c.persisted_exchange_count == 0
