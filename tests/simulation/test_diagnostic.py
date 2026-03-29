"""
Diagnostic simulation — dumps detailed per-tick NPC state to find
the behavioral regression that hits ~30 seconds in.

Outputs a tick-by-tick log showing:
- Each NPC's activity, position, subtask, path status
- Synchronisation metrics (how many NPCs change state per tick)
- Overlap counts
- Departure/arrival events

Run: python3 tests/simulation/test_diagnostic.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.world.generator import TownGenerator, WorldConfig
from core.world.grid import Terrain
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.world.spatial_awareness import get_occupied_tiles

TICKS = 120  # 2 minutes at 1s per tick
TICK_DELTA = 1.0
POPULATION = 10
SEED = 42


async def run_diagnostic():
    cfg = WorldConfig(seed=SEED, grid_width=60, grid_height=60, population=POPULATION)
    gen = TownGenerator(cfg)
    gen.generate()
    mgr = NPCManager(gen.grid, gen.buildings, seed=SEED)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    # Track previous state
    prev: dict[str, dict] = {}
    for npc in mgr.npcs:
        prev[npc.npc_id] = {
            "x": npc.x, "z": npc.z,
            "activity": npc.activity.value,
            "subtask": None,
            "walking": False,
        }

    # Metrics per tick
    print("=" * 100)
    print(f"DIAGNOSTIC SIMULATION — {TICKS} ticks, {POPULATION} NPCs, seed={SEED}")
    print("=" * 100)

    # Summary buckets for every 10-tick window
    window_sync_departures = []
    window_sync_arrivals = []
    window_overlaps = []
    window_idle_counts = []

    for tick in range(TICKS):
        await mgr.tick(clock, TICK_DELTA)
        clock.tick(TICK_DELTA)

        # Per-tick counters
        departures = 0
        arrivals = 0
        idle_no_subtask = 0
        walking_count = 0
        overlap_count = 0
        state_changes = 0
        positions = []

        for npc in mgr.npcs:
            p = prev[npc.npc_id]
            is_walking = npc.activity == ActivityState.WALKING
            was_walking = p["walking"]
            curr_subtask = npc.current_subtask.description if npc.current_subtask else None

            if is_walking and not was_walking:
                departures += 1
            if not is_walking and was_walking:
                arrivals += 1
            if is_walking:
                walking_count += 1
            if npc.activity.value != p["activity"]:
                state_changes += 1
            if (not is_walking and not npc.current_subtask
                    and not npc.subtask_queue):
                idle_no_subtask += 1

            if not is_walking:
                positions.append((npc.tile_x, npc.tile_z))

            prev[npc.npc_id] = {
                "x": npc.x, "z": npc.z,
                "activity": npc.activity.value,
                "subtask": curr_subtask,
                "walking": is_walking,
            }

        # Count position overlaps
        pos_counts = Counter(positions)
        overlap_count = sum(c - 1 for c in pos_counts.values() if c > 1)

        window_sync_departures.append(departures)
        window_sync_arrivals.append(arrivals)
        window_overlaps.append(overlap_count)
        window_idle_counts.append(idle_no_subtask)

        # Print per-tick detail for key events
        flags = []
        if departures >= 3:
            flags.append(f"SYNC_DEPART={departures}")
        if arrivals >= 3:
            flags.append(f"SYNC_ARRIVE={arrivals}")
        if overlap_count > 0:
            flags.append(f"OVERLAPS={overlap_count}")
        if idle_no_subtask >= 3:
            flags.append(f"MASS_IDLE={idle_no_subtask}")
        if state_changes >= 4:
            flags.append(f"STATE_CHURN={state_changes}")

        if flags or tick % 10 == 0:
            flag_str = " *** " + " | ".join(flags) if flags else ""
            print(
                f"  tick {tick:4d}  "
                f"walk={walking_count} dep={departures} arr={arrivals} "
                f"idle_empty={idle_no_subtask} overlaps={overlap_count} "
                f"changes={state_changes}{flag_str}"
            )

        # Every 10 ticks, print individual NPC state
        if tick % 30 == 29 or (flags and any("SYNC" in f or "MASS" in f for f in flags)):
            print(f"    --- NPC State at tick {tick} ---")
            for npc in mgr.npcs:
                st = npc.current_subtask
                st_desc = st.description[:35] if st else "NONE"
                st_remain = f"{npc.subtask_time_remaining:.1f}m" if st else "-"
                q_len = len(npc.subtask_queue)
                path_len = len(npc.current_path)
                print(
                    f"    {npc.name:12s} "
                    f"({npc.tile_x:3d},{npc.tile_z:3d}) "
                    f"{npc.activity.value:10s} "
                    f"subtask=[{st_desc:35s}] remain={st_remain:6s} "
                    f"queue={q_len} path={path_len} "
                    f"desc=\"{npc.current_action_description[:40]}\""
                )

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY BY 10-TICK WINDOWS")
    print("=" * 100)
    for i in range(0, TICKS, 10):
        chunk = slice(i, i + 10)
        dep = sum(window_sync_departures[chunk])
        arr = sum(window_sync_arrivals[chunk])
        ovlp = max(window_overlaps[chunk])
        idle = max(window_idle_counts[chunk])
        sync_dep_ticks = sum(1 for d in window_sync_departures[chunk] if d >= 3)
        sync_arr_ticks = sum(1 for a in window_sync_arrivals[chunk] if a >= 3)
        flags = []
        if sync_dep_ticks:
            flags.append(f"SYNC_DEP({sync_dep_ticks})")
        if sync_arr_ticks:
            flags.append(f"SYNC_ARR({sync_arr_ticks})")
        if ovlp > 1:
            flags.append(f"OVERLAP_PEAK({ovlp})")
        if idle >= 3:
            flags.append(f"MASS_IDLE({idle})")
        flag_str = " *** " + " ".join(flags) if flags else ""
        print(
            f"  ticks {i:3d}-{i+9:3d}: "
            f"departures={dep:2d} arrivals={arr:2d} "
            f"peak_overlap={ovlp} peak_idle={idle}"
            f"{flag_str}"
        )

    # Final NPC state
    print()
    print("FINAL NPC STATE:")
    for npc in mgr.npcs:
        st = npc.current_subtask
        print(
            f"  {npc.name:12s} {npc.activity.value:10s} "
            f"subtask={st.description[:40] if st else 'NONE':40s} "
            f"queue={len(npc.subtask_queue)} "
            f"desc=\"{npc.current_action_description[:50]}\""
        )


def main():
    asyncio.run(run_diagnostic())


if __name__ == "__main__":
    main()
