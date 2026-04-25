"""
Phase H.7 — end-to-end: Phase K tagged memories survive day-level
compaction and remain retrievable a few game-days later.

The guarantee being tested is the load-bearing contract for Phase H:
compaction collapses the untagged firehose, but anything carrying a
Phase K tag (outcome records, town events, notes) must stay findable
— otherwise the whole point of the tagged-specific-retention Phase K
introduced evaporates.

Scenario:
1. Seed an NPC on day 0 with:
   - A tagged accusation memory ("Petra accused me of hoarding bread")
     carrying tags {"bread", "accused:<npc>", "outcome:accusation"}.
   - A dozen untagged observation memories (the "firehose").
2. Run the sim through day rollover at least twice so the
   NPCManager's day-rollover hook fires and compacts day 0 (and
   eventually day 1).
3. Assert:
   - A `day_summary` memory for day 0 exists on the NPC.
   - Every seeded raw observation is tombstoned (`compacted_into`
     points at the day_summary).
   - The tagged accusation memory is UNTOMBSTONED, discoverable
     via `retrieve_by_tags(["bread"])`, and appears in
     `retrieve_unresolved_matters` for the original partner.
"""

from __future__ import annotations

import asyncio
import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world


def _make_sim(seed: int = 701) -> tuple[NPCManager, GameClock]:
    config = WorldConfig(population=3, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    # Fallback-only episodic keeps the test deterministic and isolated
    # — Phase H is memory-surface logic; the ChromaDB path is covered
    # by the Phase K unit tests.
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
        llm=llm,
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed, memory=memory,
    )
    mgr.spawn_population(3)
    return mgr, GameClock()


def _seed_day_zero_memories(
    mgr: NPCManager, npc_id: str,
) -> tuple[str, list[str]]:
    """Drop one tagged accusation + a dozen untagged observations
    into day 0 of this NPC's memory. Returns
    `(tagged_id, [raw_id, ...])`.
    """
    tagged_id = mgr.memory.episodic.add_memory(
        npc_id=npc_id,
        description="Petra accused me of hoarding bread.",
        category="accusation",
        importance=0.85,
        game_time=120.0,
        tags={"bread", f"accused:{npc_id}", "outcome:accusation"},
        extra_metadata={
            "outcome_kind": "accusation",
            "accuser": "Petra",
            "accused": npc_id,
            "claim": "hoarding bread",
            "unresolved": True,
        },
    )
    raw_ids: list[str] = []
    for i in range(12):
        raw_ids.append(mgr.memory.episodic.add_memory(
            npc_id=npc_id,
            description=f"Small observation {i} on day 0.",
            category="observation",
            importance=0.35,
            game_time=200.0 + i * 30.0,
        ))
    return tagged_id, raw_ids


async def _run_through_day(
    mgr: NPCManager, clock: GameClock, num_days: int,
) -> None:
    """Advance the sim by `num_days` in-game days, ticking fast.

    Mirrors the pattern in `test_multiday_invariants.py`: at default
    clock speed each real-second is ~1.2 game-minutes, so a
    real_delta of 8 per tick covers ~9.6 game-minutes. For a 3-day
    run that's roughly 450 ticks. The exact count isn't material —
    what matters is that at least two day-rollovers fire, triggering
    `_run_daily_compaction` twice.
    """
    real_delta = 8.0
    game_minutes_per_tick = 9.6
    total_game_minutes = num_days * MINUTES_PER_DAY
    num_ticks = int(total_game_minutes / game_minutes_per_tick) + 1
    for _ in range(num_ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)


@pytest.mark.timeout(300)
def test_tagged_memory_survives_day_compaction() -> None:
    """The headline H.7 invariant: tagged Phase K memories still
    retrievable after compaction collapses the raw firehose."""
    async def _run():
        mgr, clock = _make_sim()
        npc = mgr.npcs[0]
        tagged_id, raw_ids = _seed_day_zero_memories(mgr, npc.npc_id)

        # Advance ~3 game-days. Day-rollover at day 1 should compact
        # day 0; at day 2 compact day 1 (largely empty).
        await _run_through_day(mgr, clock, num_days=3)

        # --- The compaction cursor actually ran on day 0 ---
        assert mgr._last_compacted_day.get(npc.npc_id, -1) >= 0, (
            "expected _run_daily_compaction to have fired for this NPC"
        )

        # --- A day_summary for day 0 exists ---
        all_mems = mgr.memory.episodic.get_recent(
            npc.npc_id, limit=100, include_compacted=True,
        )
        day0_summaries = [
            m for m in all_mems
            if m.category == "day_summary"
            and (m.metadata or {}).get("day") == 0
        ]
        assert day0_summaries, (
            "no day_summary produced for day 0 — compaction did not run"
        )
        summary = day0_summaries[0]

        # --- Every seeded raw observation is tombstoned to the summary ---
        for rid in raw_ids:
            raw = mgr.memory.episodic.get_raw_by_id(rid)
            assert raw is not None
            assert raw.metadata.get("compacted_into") == summary.memory_id, (
                f"raw {rid} was not tombstoned into the day_summary "
                f"(got compacted_into={raw.metadata.get('compacted_into')!r})"
            )

        # --- The tagged accusation memory survived intact ---
        tagged = mgr.memory.episodic.get_raw_by_id(tagged_id)
        assert tagged is not None
        assert "compacted_into" not in (tagged.metadata or {}), (
            "tagged Phase K memory was tombstoned — Phase K retention "
            "contract violated by compaction"
        )
        assert "bread" in tagged.tags

        # --- It's still retrievable via the tag index ---
        hits = mgr.memory.episodic.retrieve_by_tags(
            npc.npc_id, ["bread"],
        )
        assert any(m.memory_id == tagged_id for m in hits), (
            "tagged accusation memory no longer discoverable via "
            "retrieve_by_tags — Phase K/H interaction broken"
        )

        # --- And via the Phase C unresolved-matters path ---
        matters = mgr.memory.retrieve_unresolved_matters(
            npc.npc_id,
            partner_id="petra_id",
            partner_name="Petra",
        )
        assert any(
            m.memory_id == tagged_id for m in matters
        ), (
            "unresolved matter lookup no longer surfaces the tagged "
            "accusation — Phase C retrieval path broken by compaction"
        )

        # --- The summary's kept_tags records the bread topic ---
        kept = set(
            (summary.metadata.get("kept_tags") or "").split()
        )
        assert "bread" in kept, (
            "day_summary's kept_tags did not aggregate the bread tag — "
            "the provenance audit trail is lossy"
        )

    asyncio.run(_run())


@pytest.mark.timeout(300)
def test_default_retrieval_shows_summary_not_raws() -> None:
    """After compaction, `get_recent` (the default retrieval path)
    surfaces the day_summary and hides the tombstoned originals.
    This is the H.3 retrieval-preference invariant, measured
    end-to-end rather than in isolation."""
    async def _run():
        mgr, clock = _make_sim(seed=703)
        npc = mgr.npcs[0]
        _, raw_ids = _seed_day_zero_memories(mgr, npc.npc_id)

        await _run_through_day(mgr, clock, num_days=3)

        recents = mgr.memory.episodic.get_recent(
            npc.npc_id, limit=100,
        )
        recent_ids = {m.memory_id for m in recents}

        # Seeded raws are all tombstoned → hidden.
        assert not (set(raw_ids) & recent_ids), (
            "raw day-0 observations still surfaced by default "
            "retrieval after compaction — H.3 hierarchy broken"
        )

        # At least one day_summary is visible.
        categories = {m.category for m in recents}
        assert "day_summary" in categories
    asyncio.run(_run())
