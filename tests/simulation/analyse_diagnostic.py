"""
Post-run analysis of the diagnostic simulation log.

Reads diagnostic_log.jsonl and produces:
1. Timeline report — per-NPC goal progress over simulated days
2. Sync detection — ticks where ≥3 NPCs change state simultaneously
3. Behaviour degradation — does subtask variety decrease over time?
4. Idle analysis — when do NPCs go idle? Why?
5. Memory/activity report — per-NPC breakdown
6. Heat map — where NPCs spend time (text-based)

Run: python3 tests/simulation/analyse_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

LOG_PATH = Path(__file__).parent / "diagnostic_log.jsonl"


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def analyse(events: list[dict]) -> None:
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS")
    print(f"  Total events: {len(events)}")
    print("=" * 80)

    # ---------- 1. Timeline: Goal progress ----------
    print("\n--- 1. GOAL PROGRESS TIMELINE ---\n")
    goal_events = [e for e in events if e["event_type"] in (
        "GOAL_STEP_COMPLETE", "GOAL_COMPLETE",
    )]
    by_npc: dict[str, list[dict]] = defaultdict(list)
    for e in goal_events:
        by_npc[e["npc_name"]].append(e)

    if not goal_events:
        print("  No goal events found.")
    else:
        for name in sorted(by_npc):
            evts = by_npc[name]
            print(f"  {name}:")
            for e in evts:
                d = e["data"]
                print(
                    f"    {e['game_time']:>16s}  "
                    f"{e['event_type']:20s}  "
                    f"step={d.get('step_index', '?')}  "
                    f"{d.get('step_desc', '')[:40]}  "
                    f"progress={d.get('progress', 0):.0%}"
                )

    # ---------- 2. Sync detection ----------
    print("\n--- 2. SYNCHRONISATION DETECTION ---\n")

    departures_by_tick: dict[int, list[str]] = defaultdict(list)
    arrivals_by_tick: dict[int, list[str]] = defaultdict(list)
    for e in events:
        if e["event_type"] == "DEPARTURE":
            departures_by_tick[e["tick"]].append(e["npc_name"])
        elif e["event_type"] == "ARRIVAL":
            arrivals_by_tick[e["tick"]].append(e["npc_name"])

    sync_dep_ticks = {
        t: names for t, names in departures_by_tick.items()
        if len(names) >= 3
    }
    sync_arr_ticks = {
        t: names for t, names in arrivals_by_tick.items()
        if len(names) >= 3
    }

    total_ticks = max((e["tick"] for e in events), default=0) + 1
    print(f"  Total ticks analysed: {total_ticks}")
    print(f"  Sync departure ticks (≥3 NPCs): {len(sync_dep_ticks)}")
    print(f"  Sync arrival ticks (≥3 NPCs): {len(sync_arr_ticks)}")

    # Sync score: fraction of ticks with synchronised departures
    sync_score = len(sync_dep_ticks) / max(total_ticks, 1)
    print(f"  Sync score (departure): {sync_score:.4f}")
    sync_threshold = 0.2
    if sync_score < sync_threshold:
        print(f"  ✓ PASS: sync score {sync_score:.4f} < {sync_threshold}")
    else:
        print(f"  ✗ FAIL: sync score {sync_score:.4f} >= {sync_threshold}")

    if sync_dep_ticks:
        print(f"\n  Worst sync departure ticks (top 10):")
        worst = sorted(sync_dep_ticks.items(), key=lambda x: -len(x[1]))[:10]
        for t, names in worst:
            gt = _find_game_time(events, t)
            print(f"    tick {t:6d} ({gt}): {len(names)} NPCs — {', '.join(names)}")

    # ---------- 3. Behaviour degradation ----------
    print("\n--- 3. BEHAVIOUR DEGRADATION ---\n")

    tick_states = [e for e in events if e["event_type"] == "TICK_STATE"]
    # Group by day
    day_subtasks: dict[int, set[str]] = defaultdict(set)
    day_activities: dict[int, Counter] = defaultdict(Counter)
    for e in tick_states:
        gt = e["game_time"]
        day = _extract_day(gt)
        st = e["data"].get("subtask")
        if st:
            day_subtasks[day].add(st)
        act = e["data"].get("activity", "idle")
        day_activities[day][act] += 1

    print("  Unique subtask descriptions per day:")
    days_sorted = sorted(day_subtasks.keys())
    variety_values = []
    for d in days_sorted:
        count = len(day_subtasks[d])
        variety_values.append(count)
        flag = " ← LOW" if count < 8 else ""
        print(f"    Day {d:3d}: {count:3d} unique subtasks{flag}")

    if variety_values:
        avg = sum(variety_values) / len(variety_values)
        print(f"  Average: {avg:.1f} unique subtasks/day")
        if avg >= 8:
            print(f"  ✓ PASS: variety average {avg:.1f} >= 8")
        else:
            print(f"  ✗ FAIL: variety average {avg:.1f} < 8")

    # ---------- 4. Idle analysis ----------
    print("\n--- 4. IDLE ANALYSIS ---\n")

    idle_ticks: dict[str, int] = defaultdict(int)
    walking_ticks: dict[str, int] = defaultdict(int)
    working_ticks: dict[str, int] = defaultdict(int)
    total_samples: dict[str, int] = defaultdict(int)

    for e in tick_states:
        npc = e["npc_name"]
        act = e["data"].get("activity", "idle")
        total_samples[npc] += 1
        if act == "idle":
            idle_ticks[npc] += 1
        elif act == "walking":
            walking_ticks[npc] += 1
        elif act in ("working", "gathering"):
            working_ticks[npc] += 1

    print(f"  {'NPC':12s} {'Idle%':>7s} {'Walk%':>7s} {'Work%':>7s} {'Samples':>8s}")
    for npc in sorted(total_samples):
        tot = total_samples[npc]
        if tot == 0:
            continue
        print(
            f"  {npc:12s} "
            f"{idle_ticks[npc]/tot:6.1%} "
            f"{walking_ticks[npc]/tot:6.1%} "
            f"{working_ticks[npc]/tot:6.1%} "
            f"{tot:8d}"
        )

    # ---------- 5. Per-NPC goal summary ----------
    print("\n--- 5. GOAL SUMMARY ---\n")

    goal_completes = [e for e in events if e["event_type"] == "GOAL_COMPLETE"]
    completed_npcs = {e["npc_id"] for e in goal_completes}

    # Get final goal progress from last TICK_STATE per NPC
    final_progress: dict[str, float] = {}
    for e in reversed(tick_states):
        npc_id = e["npc_id"]
        if npc_id not in final_progress:
            prog = e["data"].get("goal_progress")
            if prog is not None:
                final_progress[npc_id] = prog

    step_events = [e for e in events if e["event_type"] == "GOAL_STEP_COMPLETE"]
    steps_by_npc: dict[str, int] = Counter()
    for e in step_events:
        steps_by_npc[e["npc_id"]] += 1

    npc_ids = sorted(set(e["npc_id"] for e in tick_states))
    npcs_with_3_steps = 0
    for npc_id in npc_ids:
        npc_name = _find_npc_name(events, npc_id)
        steps = steps_by_npc.get(npc_id, 0)
        complete = npc_id in completed_npcs
        progress = final_progress.get(npc_id, 0.0)
        status = "COMPLETE" if complete else f"{steps} steps"
        if steps >= 3:
            npcs_with_3_steps += 1
        print(f"  {npc_name:12s}: {status:12s} (final progress: {progress:.0%})")

    print(f"\n  NPCs with ≥3 steps complete: {npcs_with_3_steps}/{len(npc_ids)}")
    if npcs_with_3_steps >= 7:
        print(f"  ✓ PASS: {npcs_with_3_steps} >= 7 NPCs with ≥3 steps")
    else:
        print(f"  ✗ FAIL: {npcs_with_3_steps} < 7 NPCs with ≥3 steps")

    # ---------- 6. Position heat map ----------
    print("\n--- 6. POSITION HEAT MAP (top 15 tiles) ---\n")

    pos_counter: Counter = Counter()
    for e in tick_states:
        tile = e["data"].get("tile")
        if tile:
            pos_counter[(tile[0], tile[1])] += 1

    print(f"  {'Tile':>12s}  {'Visits':>7s}  {'%':>6s}")
    total_pos = sum(pos_counter.values())
    for (x, z), count in pos_counter.most_common(15):
        print(f"  ({x:4d},{z:4d})  {count:7d}  {count/total_pos:5.1%}")

    # ---------- Summary ----------
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Sync score:              {sync_score:.4f} (target < 0.2)")
    avg_variety = (
        sum(variety_values) / len(variety_values) if variety_values else 0
    )
    print(f"  Avg subtask variety/day: {avg_variety:.1f} (target >= 8)")
    print(f"  Goals with ≥3 steps:     {npcs_with_3_steps}/{len(npc_ids)} (target >= 7)")
    print(f"  Goals fully complete:    {len(completed_npcs)}/{len(npc_ids)}")

    all_pass = (
        sync_score < 0.2
        and avg_variety >= 8
        and npcs_with_3_steps >= 7
    )
    if all_pass:
        print("\n  ✓ ALL CRITERIA MET — experiment succeeded")
    else:
        print("\n  ✗ SOME CRITERIA NOT MET — see details above")
    print("=" * 80)


# ---------- Helpers ----------

def _find_game_time(events: list[dict], tick: int) -> str:
    for e in events:
        if e["tick"] == tick:
            return e.get("game_time", "?")
    return "?"


def _extract_day(game_time: str) -> int:
    try:
        return int(game_time.split("Day")[1].split(",")[0].strip())
    except (IndexError, ValueError):
        return 0


def _find_npc_name(events: list[dict], npc_id: str) -> str:
    for e in events:
        if e["npc_id"] == npc_id:
            return e.get("npc_name", npc_id)
    return npc_id


def main():
    if not LOG_PATH.exists():
        print(f"ERROR: Log file not found: {LOG_PATH}")
        print("Run diagnostic_instrumented_sim.py first.")
        sys.exit(1)

    events = load_events(LOG_PATH)
    analyse(events)


if __name__ == "__main__":
    main()
