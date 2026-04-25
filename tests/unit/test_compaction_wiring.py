"""
Phase H.6 — `_run_daily_compaction` wiring into NPCManager.

Covers:
- `compaction` is a known router decision type with an auto-mode
  default.
- `_run_daily_compaction(day)` is a no-op when day < 0.
- One call compacts that day for every autonomous, non-frozen NPC,
  stamps `_last_compacted_day`, and re-running the same day is a
  cheap no-op (cursor guard).
- Tier-4 (frozen) NPCs are skipped.
- Week rollup piggybacks on the last day of a week (day % 7 == 6)
  and bumps `_last_compacted_week`.
- Router mode "deterministic" causes the fallback heuristic to be
  used (no LLM call) — verified by observing that the day_summary
  description starts with the fallback `"Day N:"` prefix.
- Router mode "llm" passes `manager.llm` through (verified via a
  stub provider that records the call).

These tests exercise the manager method directly; they do NOT spin
the full cognition_tick loop, which has heavy setup. The daily-tick
integration — that compaction actually fires on day-flip — is
exercised end-to-end by the Phase H.7 simulation test.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.router import CognitionRouter, Route
from core.npc.cognition.router.policy import (
    CognitionPolicy, DECISION_TYPES, ROUTE_AUTO, ROUTE_DETERMINISTIC,
    ROUTE_LLM,
)
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.time_system.clock import MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world


def _clean_memory() -> MemoryManager:
    """Fresh, isolated memory manager per test.

    ChromaDB's `Client()` (no persist dir) caches the ephemeral
    store globally by settings — which means multiple NPCManager
    instances in the same test session end up sharing memories
    across tests. Using `fallback_only=True` sidesteps the cache
    entirely by bypassing ChromaDB for a simple in-memory dict.
    """
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


def _seed_raws_for_day(mgr: NPCManager, npc_id: str, day: int) -> list[str]:
    """Drop a few untagged observations into day `day` so the
    compaction pass has something to do."""
    ids = []
    for i in range(3):
        ids.append(mgr.memory.episodic.add_memory(
            npc_id=npc_id,
            description=f"Thing {i} happened on day {day}",
            category="observation",
            game_time=day * MINUTES_PER_DAY + 60 + i * 10,
        ))
    return ids


# ---------- Policy registration ----------

class TestCompactionDecisionType:
    def test_registered_decision_type(self):
        assert "compaction" in DECISION_TYPES

    def test_default_routing_is_auto(self):
        policy = CognitionPolicy()
        assert policy.get_mode("compaction") == ROUTE_AUTO


# ---------- Daily hook ----------

class TestRunDailyCompaction:
    def test_noop_on_negative_day(self):
        mgr = _manager()
        mgr.spawn_population(2)
        asyncio.run(mgr._run_daily_compaction(-1))
        # No compaction state recorded.
        assert mgr._last_compacted_day == {}

    def test_compacts_all_autonomous_npcs(self):
        mgr = _manager()
        mgr.spawn_population(3)
        for npc in mgr.npcs:
            _seed_raws_for_day(mgr, npc.npc_id, 0)

        asyncio.run(mgr._run_daily_compaction(0))

        for npc in mgr.npcs:
            assert mgr._last_compacted_day[npc.npc_id] == 0
            # At least one day_summary exists for each NPC.
            summaries = [
                m for m in mgr.memory.episodic.get_recent(
                    npc.npc_id, limit=20,
                )
                if m.category == "day_summary"
            ]
            assert summaries, f"no day_summary for {npc.npc_id}"

    def test_cursor_guard_prevents_rerun(self):
        """A second call for the same day should be a cursor-level
        no-op — not even dispatching through the router."""
        router = _FakeRouter()
        mgr = _manager(router=router)
        mgr.spawn_population(2)
        for npc in mgr.npcs:
            _seed_raws_for_day(mgr, npc.npc_id, 0)

        asyncio.run(mgr._run_daily_compaction(0))
        first_calls = router.total_calls
        assert first_calls == len(mgr.npcs)

        asyncio.run(mgr._run_daily_compaction(0))
        # No further router calls.
        assert router.total_calls == first_calls

    def test_frozen_tier_npcs_are_skipped(self):
        mgr = _manager()
        mgr.spawn_population(2)
        # Freeze one NPC explicitly.
        mgr.npcs[0].cognition_tier = 4
        for npc in mgr.npcs:
            _seed_raws_for_day(mgr, npc.npc_id, 0)

        asyncio.run(mgr._run_daily_compaction(0))

        assert mgr.npcs[0].npc_id not in mgr._last_compacted_day
        assert mgr._last_compacted_day[mgr.npcs[1].npc_id] == 0


# ---------- Week rollup piggyback ----------

class TestWeekRollupPiggyback:
    def test_rolls_week_on_final_day_of_week(self):
        mgr = _manager()
        mgr.spawn_population(1)
        npc_id = mgr.npcs[0].npc_id

        # Pre-seed one day_summary per day 0..6 so compact_week has
        # something to roll up when day 6 completes.
        for d in range(7):
            mgr.memory.episodic.add_memory(
                npc_id=npc_id,
                description=f"Day {d} summary",
                category="day_summary", importance=0.6,
                game_time=(d + 1) * MINUTES_PER_DAY - 1.0,
                extra_metadata={"day": d},
            )

        asyncio.run(mgr._run_daily_compaction(6))
        assert mgr._last_compacted_week[npc_id] == 0

        # A week_summary now exists, rolled up from those day_summaries.
        recents = mgr.memory.episodic.get_recent(npc_id, limit=20)
        weeks = [m for m in recents if m.category == "week_summary"]
        assert weeks
        assert weeks[0].metadata.get("week") == 0

    def test_no_week_rollup_on_midweek_day(self):
        mgr = _manager()
        mgr.spawn_population(1)
        npc_id = mgr.npcs[0].npc_id
        _seed_raws_for_day(mgr, npc_id, 3)
        asyncio.run(mgr._run_daily_compaction(3))
        assert npc_id not in mgr._last_compacted_week


# ---------- Router-driven LLM gating ----------

class _FakeRouter:
    """Minimal CognitionRouter stand-in: always routes to a fixed
    verdict and counts calls. Used to assert the manager actually
    consults the router for each compaction."""

    def __init__(self, verdict: Route = Route.DETERMINISTIC):
        self.verdict = verdict
        self.total_calls = 0
        self.last_decision_type: str = ""

    def route(self, npc, decision_type: str, **_kwargs):
        self.total_calls += 1
        self.last_decision_type = decision_type

        class _D:
            def __init__(self, route):
                self.route = route

        return _D(self.verdict)


class _RecordingLLM(MockProvider):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    async def complete(self, **kwargs):  # type: ignore[override]
        self.call_count += 1
        return await super().complete(**kwargs)


class TestRouterGating:
    def test_deterministic_verdict_uses_fallback(self):
        router = _FakeRouter(verdict=Route.DETERMINISTIC)
        mgr = _manager(router=router)
        mgr.llm = _RecordingLLM()
        mgr.memory.llm = None  # ensure manager.llm is the only path
        mgr.spawn_population(1)
        npc_id = mgr.npcs[0].npc_id
        _seed_raws_for_day(mgr, npc_id, 0)

        asyncio.run(mgr._run_daily_compaction(0))

        # Fallback summary prefix confirms no LLM path.
        summaries = [
            m for m in mgr.memory.episodic.get_recent(npc_id, limit=20)
            if m.category == "day_summary"
        ]
        assert summaries
        assert summaries[0].description.startswith("Day 0")
        assert mgr.llm.call_count == 0
        assert router.last_decision_type == "compaction"

    def test_llm_verdict_invokes_provider(self):
        router = _FakeRouter(verdict=Route.LLM)
        mgr = _manager(router=router)
        mgr.llm = _RecordingLLM()
        mgr.memory.llm = None  # force the explicit llm kwarg path
        mgr.spawn_population(1)
        npc_id = mgr.npcs[0].npc_id
        _seed_raws_for_day(mgr, npc_id, 0)

        asyncio.run(mgr._run_daily_compaction(0))

        assert mgr.llm.call_count >= 1
