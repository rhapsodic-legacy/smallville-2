"""
Multi-day simulation test: no resting NPCs may share a tile.

Runs a headless simulation for N game-days and asserts that after
every movement_tick, no two non-walking NPCs occupy the same tile.
Logs detailed diagnostics when violations occur so the root cause
can be traced.
"""

import asyncio
import logging

from core.npc.manager import NPCManager
from core.npc.llm_client import MockProvider
from core.npc.models import ActivityState
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world
from core.world.spatial_awareness import get_occupied_tiles

logger = logging.getLogger(__name__)

# --------------- helpers ---------------

def find_overlaps(npcs):
    """Return list of (tile, [npc_names]) for every tile with 2+ resting NPCs."""
    tile_map: dict[tuple[int, int], list] = {}
    for npc in npcs:
        if npc.activity == ActivityState.WALKING:
            continue
        pos = (npc.tile_x, npc.tile_z)
        tile_map.setdefault(pos, []).append(npc)
    return [
        (pos, [n.name for n in group])
        for pos, group in tile_map.items()
        if len(group) > 1
    ]


def npc_debug(npc):
    """One-line debug string for an NPC."""
    entry_desc = ""
    if npc.daily_schedule and npc.schedule_index < len(npc.daily_schedule):
        e = npc.daily_schedule[npc.schedule_index]
        entry_desc = (
            f"entry={e.activity!r} loc={e.location!r} "
            f"target=({e.target_x},{e.target_z})"
        )
    return (
        f"{npc.name} ({npc.npc_id}) at ({npc.tile_x},{npc.tile_z}) "
        f"activity={npc.activity.value} home=({npc.home_x},{npc.home_z}) "
        f"path_len={len(npc.current_path)} sched_idx={npc.schedule_index} "
        f"{entry_desc}"
    )


# --------------- simulation ---------------

async def run_simulation(
    population: int = 10,
    days: int = 7,
    ticks_per_day: int = 120,
    seed: int = 42,
) -> list[dict]:
    """Run a headless sim and return all overlap violations found."""
    config = WorldConfig(population=population, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed,
        deterministic=True,
    )
    mgr.spawn_population(population)

    clock = GameClock(speed=1200.0)  # 1 day = 20 min real time
    real_delta = 0.25  # 4 Hz tick

    violations: list[dict] = []
    total_ticks = days * ticks_per_day

    for tick_num in range(total_ticks):
        clock.tick(real_delta)

        # Cognition tick (async)
        await mgr.cognition_tick(clock, real_delta)

        # Movement tick
        mgr.movement_tick(clock, real_delta)

        # Check for overlaps
        overlaps = find_overlaps(mgr.npcs)
        if overlaps:
            day = clock.day
            minutes = clock.minutes
            for pos, names in overlaps:
                detail = {
                    "tick": tick_num,
                    "day": day,
                    "time": f"{int(minutes // 60):02d}:{int(minutes % 60):02d}",
                    "tile": pos,
                    "npcs": names,
                    "npc_details": [],
                }
                for npc in mgr.npcs:
                    if npc.name in names:
                        detail["npc_details"].append(npc_debug(npc))
                violations.append(detail)

    return violations


# --------------- entry point ---------------

def main():
    """Run multiple simulation configs and report results."""
    logging.basicConfig(level=logging.WARNING)

    configs = [
        {"population": 10, "days": 14, "seed": 42, "label": "10 NPCs / seed 42"},
        {"population": 10, "days": 14, "seed": 99, "label": "10 NPCs / seed 99"},
        {"population": 15, "days": 14, "seed": 7,  "label": "15 NPCs / seed 7"},
        {"population": 20, "days": 10, "seed": 13, "label": "20 NPCs / seed 13"},
    ]

    all_passed = True
    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        violations = asyncio.run(run_simulation(**cfg))

        if violations:
            all_passed = False
            # Deduplicate: count unique (tile, frozenset(npcs)) pairs
            seen = set()
            unique = []
            for v in violations:
                key = (v["tile"], tuple(sorted(v["npcs"])))
                if key not in seen:
                    seen.add(key)
                    unique.append(v)

            print(f"FAIL: {len(violations)} overlap ticks, "
                  f"{len(unique)} unique overlap pairs")
            for v in unique[:10]:  # show first 10
                print(f"  Day {v['day']} {v['time']} at {v['tile']}: "
                      f"{v['npcs']}")
                for d in v["npc_details"]:
                    print(f"    {d}")
        else:
            print(f"PASS: 0 overlaps across {cfg.get('days', 14)} days")

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL CONFIGS PASSED — no overlaps detected")
    else:
        print("FAILURES DETECTED — see details above")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
