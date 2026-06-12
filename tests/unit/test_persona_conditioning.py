"""Persona conditioning — every NPC-voiced LLM call must carry the
caller's character sheet in its system prompt.

This is the drowning regression guard: the diagnosis behind the
vectorization arc was that the persona signal was thin AND buried,
with the system slot — the strongest conditioning channel — carrying
one generic string shared by the whole town. Each test drives a real
call site against MockProvider and audits `call_log` for the sheet.
A new call site that ships with the old generic string should fail
the eval in tests/simulation/eval_persona_conditioning.py; these
tests pin the sites we have today.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


def _make_world(population: int = 3, seed: int = 55):
    config = WorldConfig(population=population, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    memory.initialise()
    llm = MockProvider()
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed, memory=memory,
    )
    npcs = mgr.spawn_population(population)
    for npc in npcs:
        npc.cognition_tier = 1  # LLM path everywhere
    return mgr, npcs, llm, memory


def _assert_conditioned(call: dict, npc) -> None:
    system = call["system"]
    assert npc.name in system, f"{call['purpose']}: name missing from system"
    assert npc.persona.speech_style in system, (
        f"{call['purpose']}: speech style missing from system prompt"
    )
    assert "character sheet" in system, (
        f"{call['purpose']}: persona sheet header missing"
    )
    assert "medieval NPC" not in system, (
        f"{call['purpose']}: generic shared identity string survived"
    )


def _calls(llm: MockProvider, purpose: str) -> list[dict]:
    return [c for c in llm.call_log if c["purpose"] == purpose]


class TestConversationConditioning:
    async def test_initiate_and_respond_carry_personas(self):
        from core.npc.cognition.converse import (
            initiate_conversation, continue_conversation,
        )

        mgr, npcs, llm, memory = _make_world()
        a, b = npcs[0], npcs[1]

        await initiate_conversation(
            a, b, llm, current_game_minutes=0.0, memory_manager=memory,
        )
        conv_calls = _calls(llm, "conversation")
        assert len(conv_calls) == 1
        _assert_conditioned(conv_calls[0], a)

        await continue_conversation(b, a, llm, memory_manager=memory)
        conv_calls = _calls(llm, "conversation")
        assert len(conv_calls) == 2
        # The responder's persona, not the initiator's.
        _assert_conditioned(conv_calls[1], b)
        assert conv_calls[1]["system"] != conv_calls[0]["system"]


class TestReflectionConditioning:
    async def test_post_conversation_reflection(self):
        from core.memory.reflection import reflect_on_conversation

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        await reflect_on_conversation(
            npc, "Bran",
            [{"speaker": "Bran", "message": "The bridge is failing."}],
            memory, llm, current_game_time=10.0,
        )
        calls = _calls(llm, "reflection")
        assert calls, "reflection produced no LLM call"
        _assert_conditioned(calls[0], npc)

    async def test_focal_questions_and_insight(self):
        from core.memory.reflection import (
            _generate_focal_questions, _synthesise_insight,
        )

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        await _generate_focal_questions(npc, ["Saw the bridge crack."], llm)
        await _synthesise_insight(
            npc, "What of the bridge?", "- Saw the bridge crack.", llm,
        )
        calls = _calls(llm, "reflection")
        assert len(calls) == 2
        for call in calls:
            _assert_conditioned(call, npc)


class TestPlanningConditioning:
    async def test_daily_schedule(self):
        from core.npc.cognition.plan import _llm_schedule

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        await _llm_schedule(npc, llm, current_day=1)
        calls = _calls(llm, "daily_plan")
        assert calls
        _assert_conditioned(calls[0], npc)

    async def test_reaction(self):
        from core.npc.cognition.plan import decide_reaction

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        await decide_reaction(npc, "A stranger enters the square.", llm)
        calls = _calls(llm, "reaction")
        assert calls
        _assert_conditioned(calls[0], npc)

    async def test_replan(self):
        from core.npc.cognition.plan import replan_schedule
        from core.npc.models import ScheduleEntry

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        npc.daily_schedule = [
            ScheduleEntry(
                slot="afternoon", activity="work the fields",
                location="farm", duration_minutes=240,
            ),
        ]
        npc.schedule_index = 0
        await replan_schedule(npc, llm, current_minutes=600.0)
        calls = _calls(llm, "daily_plan")
        assert calls
        _assert_conditioned(calls[0], npc)


class TestMemoryVoiceConditioning:
    async def test_day_summary(self):
        from core.memory.compaction import _summarise_with_llm

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        await _summarise_with_llm(llm, npc, day=1, mems=[])
        calls = _calls(llm, "day_summary")
        assert calls
        _assert_conditioned(calls[0], npc)

    async def test_self_review(self):
        from core.memory.self_review import _run_review_with_llm

        mgr, npcs, llm, memory = _make_world()
        npc = npcs[0]
        await _run_review_with_llm(
            llm, npc, day=1, commitments=[], long_term=[], day_summary=None,
        )
        calls = _calls(llm, "self_review")
        assert calls
        _assert_conditioned(calls[0], npc)
