"""
Phase I.1 — `_run_daily_self_review` wiring into NPCManager.

Covers the manager-level plumbing that runs the bedtime review
right after `_run_daily_compaction` on day rollover:

- `self_review` is a registered router decision type with the
  expected default-LLM routing.
- `_run_daily_self_review(day)` is a no-op on negative days.
- One call per autonomous, non-frozen NPC writes a
  `commitment_review` memory and stamps `_last_self_reviewed_day`.
- Tier-4 (frozen) NPCs are skipped.
- Cursor guard makes re-runs a cheap no-op — router is not consulted
  a second time.
- Router `DETERMINISTIC` verdict bypasses the LLM provider entirely
  (heuristic fallback path), and `LLM` verdict drives the provider.
- `ActionIntent` returned by the review is injected into the NPC's
  daily schedule via the shared reflection-entry path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.reflection import ActionIntent
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.router import CognitionRouter, Route
from core.npc.cognition.router.policy import (
    CognitionPolicy, DECISION_TYPES, ROUTE_LLM,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.time_system.clock import MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world


# ---------- Helpers ----------


def _clean_memory() -> MemoryManager:
    mem = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mem.initialise()
    return mem


def _manager(router: CognitionRouter | None = None) -> NPCManager:
    config = WorldConfig(population=3, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    return NPCManager(
        grid=grid, buildings=buildings,
        llm=MockProvider(), seed=42, router=router,
        memory=_clean_memory(),
    )


def _seed_commitment(mgr: NPCManager, npc_id: str, day: int = 0) -> str:
    """Give the NPC one open self-commitment so the review has
    something to talk about."""
    return mgr.memory.episodic.add_memory(
        npc_id=npc_id,
        description="I promised to check the south field.",
        category="commitment",
        importance=0.75,
        game_time=day * MINUTES_PER_DAY + 120,
        extra_metadata={
            "outcome_kind": "commitment",
            "source_speaker": npc_id,
            "unresolved": True,
        },
        tags={"field"},
    )


# ---------- Router stubs ----------


class _FakeRouter:
    """Counts calls and returns a fixed verdict. Mirrors the stub in
    test_compaction_wiring so self-review gating is testable without
    the full router scoring machinery."""

    def __init__(self, verdict: Route = Route.DETERMINISTIC):
        self.verdict = verdict
        self.total_calls = 0
        self.calls_by_type: dict[str, int] = {}

    def route(self, npc, decision_type: str, **_kwargs):
        self.total_calls += 1
        self.calls_by_type[decision_type] = (
            self.calls_by_type.get(decision_type, 0) + 1
        )

        class _D:
            def __init__(self, route):
                self.route = route

        return _D(self.verdict)


class _RecordingLLM(MockProvider):
    def __init__(self, response: str = ""):
        super().__init__()
        self.call_count = 0
        self._response = response

    async def complete(self, **kwargs):  # type: ignore[override]
        self.call_count += 1
        # Return the canned response for self_review so the parser
        # has something structured to work with; fall back to the
        # MockProvider for every other purpose (classify_insight,
        # conversation, reflection, etc.).
        if kwargs.get("purpose") == "self_review" and self._response:
            return self._response
        return await super().complete(**kwargs)


# ---------- Policy registration ----------


class TestSelfReviewDecisionType:
    def test_registered(self):
        assert "self_review" in DECISION_TYPES

    def test_default_routing_is_llm(self):
        """Voice-per-NPC feature — we opt in to the LLM cost for
        every non-tier-4 NPC by default."""
        policy = CognitionPolicy()
        assert policy.get_mode("self_review") == ROUTE_LLM


# ---------- _run_daily_self_review ----------


class TestRunDailySelfReview:
    def test_noop_on_negative_day(self):
        mgr = _manager()
        mgr.spawn_population(2)
        asyncio.run(mgr._run_daily_self_review(-1))
        assert mgr._last_self_reviewed_day == {}

    def test_reviews_all_autonomous_npcs(self):
        router = _FakeRouter(verdict=Route.DETERMINISTIC)
        mgr = _manager(router=router)
        mgr.spawn_population(3)
        for npc in mgr.npcs:
            _seed_commitment(mgr, npc.npc_id, day=0)

        asyncio.run(mgr._run_daily_self_review(0))

        for npc in mgr.npcs:
            assert mgr._last_self_reviewed_day[npc.npc_id] == 0
            reviews = [
                m for m in mgr.memory.episodic.get_recent(
                    npc.npc_id, limit=20,
                )
                if m.category == "commitment_review"
            ]
            assert reviews, f"no commitment_review for {npc.npc_id}"

        # Router consulted exactly once per NPC.
        assert router.calls_by_type["self_review"] == len(mgr.npcs)

    def test_cursor_guard_prevents_rerun(self):
        router = _FakeRouter(verdict=Route.DETERMINISTIC)
        mgr = _manager(router=router)
        mgr.spawn_population(2)
        for npc in mgr.npcs:
            _seed_commitment(mgr, npc.npc_id, day=0)

        asyncio.run(mgr._run_daily_self_review(0))
        first_calls = router.calls_by_type["self_review"]
        assert first_calls == len(mgr.npcs)

        asyncio.run(mgr._run_daily_self_review(0))
        # No further router calls — the cursor short-circuited.
        assert router.calls_by_type["self_review"] == first_calls

    def test_frozen_tier_npcs_are_skipped(self):
        mgr = _manager()
        mgr.spawn_population(2)
        # Freeze the first NPC; the second should still run.
        mgr.npcs[0].cognition_tier = 4
        for npc in mgr.npcs:
            _seed_commitment(mgr, npc.npc_id, day=0)

        asyncio.run(mgr._run_daily_self_review(0))
        assert mgr.npcs[0].npc_id not in mgr._last_self_reviewed_day
        assert mgr._last_self_reviewed_day[mgr.npcs[1].npc_id] == 0

    def test_runs_on_blank_slate_without_crash(self):
        """NPCs with no commitments and no day_summary must still
        advance the cursor (review returns None cleanly)."""
        mgr = _manager()
        mgr.spawn_population(1)
        asyncio.run(mgr._run_daily_self_review(0))
        assert mgr._last_self_reviewed_day[mgr.npcs[0].npc_id] == 0


# ---------- Router-driven LLM gating ----------


class TestRouterGating:
    def test_deterministic_verdict_uses_fallback(self):
        router = _FakeRouter(verdict=Route.DETERMINISTIC)
        mgr = _manager(router=router)
        mgr.llm = _RecordingLLM()
        mgr.memory.llm = None  # force manager.llm to be the only path
        mgr.spawn_population(1)
        npc_id = mgr.npcs[0].npc_id
        _seed_commitment(mgr, npc_id, day=0)

        asyncio.run(mgr._run_daily_self_review(0))

        # No LLM calls for the review — fallback heuristic ran.
        review = [
            m for m in mgr.memory.episodic.get_recent(npc_id, limit=20)
            if m.category == "commitment_review"
        ][0]
        assert review.description.startswith("Day 0")
        assert mgr.llm.call_count == 0

    def test_llm_verdict_invokes_provider(self):
        router = _FakeRouter(verdict=Route.LLM)
        mgr = _manager(router=router)
        mgr.llm = _RecordingLLM(
            response=(
                "SUMMARY: Field not checked.\n"
                "GOAL: check the south field\n"
                "STATUS: stalled\n"
                "NOTE: rain all day.\n"
                "NEXT: NO_ACTION\n"
            ),
        )
        mgr.memory.llm = None
        mgr.spawn_population(1)
        npc_id = mgr.npcs[0].npc_id
        _seed_commitment(mgr, npc_id, day=0)

        asyncio.run(mgr._run_daily_self_review(0))

        review = [
            m for m in mgr.memory.episodic.get_recent(npc_id, limit=20)
            if m.category == "commitment_review"
        ][0]
        assert mgr.llm.call_count >= 1
        assert "Field not checked" in review.description
        assert "[stalled]" in review.description


# ---------- Action-intent injection ----------


class TestActionIntentInjection:
    def test_intent_injected_into_schedule(self, monkeypatch):
        """When the review produces an ActionIntent, the manager
        should call `_inject_reflection_entry` so tomorrow's schedule
        picks up a reaction-priority entry. We monkey-patch the
        memory layer so the test isn't coupled to LLM behaviour."""
        router = _FakeRouter(verdict=Route.LLM)
        mgr = _manager(router=router)
        mgr.spawn_population(1)
        npc = mgr.npcs[0]
        _seed_commitment(mgr, npc.npc_id, day=0)

        intent = ActionIntent(
            activity="visit the south field at dawn",
            location="south_field",
            duration_minutes=45,
        )

        async def _fake_review(npc_id, game_day, *, npc=None, llm=None):
            from core.memory.self_review import SelfReviewResult
            return SelfReviewResult(
                memory_id="stub_id",
                summary_text="ok",
                per_goal=[],
                action_intent=intent,
            )

        monkeypatch.setattr(mgr.memory, "daily_self_review", _fake_review)

        injected: list[Any] = []
        original_inject = mgr._inject_reflection_entry

        def _spy(npc_arg, intent_arg):
            injected.append(intent_arg)
            return original_inject(npc_arg, intent_arg)

        monkeypatch.setattr(mgr, "_inject_reflection_entry", _spy)

        asyncio.run(mgr._run_daily_self_review(0))

        assert len(injected) == 1
        assert injected[0].activity == intent.activity
