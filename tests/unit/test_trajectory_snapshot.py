"""Trajectory snapshot instrument (diagnostic_bridge_objector).

The 30-day run is steered by daily metric snapshots; if the snapshot
maths is wrong, the whole trajectory read is wrong. These pin the
metric computation, the per-day tone delta, pariah detection, and
that the table renders without crashing on empty/populated input.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "simulation"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "diag_bridge", ROOT / "tests" / "simulation" / "diagnostic_bridge_objector.py"
)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.memory.reflection import TONE_TALLY, reset_tone_tally
from core.world.generator import WorldConfig, generate_world


def _mgr(pop=6, seed=42):
    grid, buildings = generate_world(
        WorldConfig(population=pop, terrain="riverside", seed=seed)
    )
    mgr = NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=seed)
    npcs = mgr.spawn_population(pop)
    return mgr, npcs


class TestSnapshotMetrics:
    def test_distribution_and_pariah(self):
        mgr, npcs = _mgr()
        a, b, c, d = npcs[0], npcs[1], npcs[2], npcs[3]
        # b is widely disliked; c<->d are warm.
        mgr.sentiment.modify(a.npc_id, b.npc_id, "trust", -40)
        mgr.sentiment.modify(c.npc_id, b.npc_id, "affection", -40)
        mgr.sentiment.modify(c.npc_id, d.npc_id, "affection", 40)
        a.self_concept["opposes:repair_bridge"] = 0.9
        b.self_concept["role:baker"] = 0.7

        reset_tone_tally()
        snap, tone_cum = diag._snapshot_metrics(
            mgr, npcs, day=3, prev_tone={}, events_today=["bridge_proposed#1"],
        )
        assert snap["day"] == 3
        assert snap["rels"] >= 3
        assert 0.0 <= snap["neg_pct"] <= 1.0
        assert snap["neg_pct"] > 0  # the seeded grudges register
        assert snap["min"] < 0
        # b should be the most-disliked (two strong negatives against it).
        assert snap["most_disliked"] == b.name
        assert snap["most_disliked_mean"] < 0
        assert snap["self_keys_mean"] > 0
        assert snap["events"] == ["bridge_proposed#1"]

    def test_tone_delta_is_per_day(self):
        mgr, npcs = _mgr()
        reset_tone_tally()
        TONE_TALLY.update({"tense": 10, "warm": 4, "neutral": 6})
        # prev cumulative was 7 tense / 1 warm — today's delta is the rest.
        snap, cum = diag._snapshot_metrics(
            mgr, npcs, day=2,
            prev_tone={"tense": 7, "warm": 1, "neutral": 6, "hostile": 0},
            events_today=[],
        )
        assert snap["tone_today"] == {"tense": 3, "warm": 3, "neutral": 0, "hostile": 0}
        assert cum["tense"] == 10  # returns the new cumulative for next call

    def test_empty_sentiment_no_crash(self):
        mgr, npcs = _mgr()
        snap, _ = diag._snapshot_metrics(
            mgr, npcs, day=1, prev_tone={}, events_today=[],
        )
        # Fresh town may have no non-default sentiment; must not divide by zero.
        assert snap["day"] == 1
        assert 0.0 <= snap["pos_pct"] <= 1.0


class TestTrajectoryTable:
    def test_renders_empty_and_populated(self):
        diag._print_trajectory_table([])  # no crash on empty
        diag._print_trajectory_table([{
            "day": 1, "neg_pct": 0.1, "neu_pct": 0.3, "pos_pct": 0.6,
            "mean": 12.3, "min": -5.2, "self_keys_mean": 2.1,
            "tone_today": {"tense": 5, "neutral": 3, "warm": 8, "hostile": 1},
            "events": ["bridge_completed"],
        }])
