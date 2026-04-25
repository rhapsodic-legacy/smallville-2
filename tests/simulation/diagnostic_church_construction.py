"""
Church construction simulation test.

Creates a church construction site and verifies NPCs autonomously
gather resources, contribute materials, provide labour, and complete
the building. Produces rich per-NPC diagnostics showing what everyone
was doing, who contributed, and whether NPCs maintained life balance.

Run: python3 tests/simulation/diagnostic_church_construction.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.world.generator import TownGenerator, WorldConfig
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider

from tests.simulation.npc_metrics import NPCMetricsTracker

# ---------- Configuration ----------

POPULATION = 10
SEED = 42
MAX_SIMULATED_DAYS = 10
TICK_DELTA = 1.0
TICKS_PER_DAY = 1200
SAMPLE_INTERVAL = 10  # sample metrics every N ticks


def _find_clear_site(grid, buildings, npcs, width=5, height=5):
    """Find a clear passable area near the NPC population centre.

    Uses flood-fill from the NPC majority cluster to ensure the site
    is on the same side of any rivers or terrain barriers.
    """
    if npcs:
        cx = sum(n.x for n in npcs) // len(npcs)
        cz = sum(n.z for n in npcs) // len(npcs)
    else:
        cx, cz = grid.width // 2, grid.height // 2

    # Flood-fill from each NPC; pick the cluster containing the most NPCs
    # so the site is on the same side of any river/barrier as the majority.
    best_reachable: set[tuple[int, int]] = set()
    best_count = 0
    checked_npcs: set[str] = set()
    for npc in npcs:
        if npc.npc_id in checked_npcs:
            continue
        r = _flood_reachable(grid, round(npc.x), round(npc.z))
        count = sum(1 for n in npcs if (round(n.x), round(n.z)) in r)
        for n in npcs:
            if (round(n.x), round(n.z)) in r:
                checked_npcs.add(n.npc_id)
        if count > best_count:
            best_count = count
            best_reachable = r
    reachable = best_reachable

    # Update centroid to the cluster's centre
    cluster_npcs = [n for n in npcs if (round(n.x), round(n.z)) in reachable]
    if cluster_npcs:
        cx = sum(round(n.x) for n in cluster_npcs) // len(cluster_npcs)
        cz = sum(round(n.z) for n in cluster_npcs) // len(cluster_npcs)

    for radius in range(3, 30):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x, z = cx + dx, cz + dz
                # Approach tile must be in the reachable set
                approach_x = x + width // 2
                approach_z = z + height
                if (approach_x, approach_z) not in reachable:
                    continue
                clear = True
                for bx in range(-1, width + 1):
                    for bz in range(-1, height + 1):
                        tile = grid.get_tile(x + bx, z + bz)
                        if tile is None or not tile.is_passable or tile.objects:
                            clear = False
                            break
                    if not clear:
                        break
                if clear:
                    return x, z
    return None, None


def _flood_reachable(grid, start_x: int, start_z: int) -> set[tuple[int, int]]:
    """BFS flood-fill of all passable tiles reachable from start."""
    visited: set[tuple[int, int]] = set()
    start = grid.get_tile(start_x, start_z)
    if start is None or not start.is_passable:
        # Find nearest passable tile
        for r in range(1, 10):
            for ddx in range(-r, r + 1):
                for ddz in range(-r, r + 1):
                    t = grid.get_tile(start_x + ddx, start_z + ddz)
                    if t and t.is_passable:
                        start_x, start_z = start_x + ddx, start_z + ddz
                        break
                else:
                    continue
                break
            else:
                continue
            break

    queue = [(start_x, start_z)]
    visited.add((start_x, start_z))
    while queue:
        cx, cz = queue.pop()
        for ddx, ddz in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, nz = cx + ddx, cz + ddz
            if (nx, nz) in visited:
                continue
            tile = grid.get_tile(nx, nz)
            if tile and tile.is_passable:
                visited.add((nx, nz))
                queue.append((nx, nz))
    return visited


async def run_church_construction_test(seed: int = SEED, quiet: bool = False):
    """Run one church construction simulation. Returns (passed, summary_dict)."""
    log = (lambda *a, **k: None) if quiet else print

    log("=" * 90)
    log("CHURCH CONSTRUCTION SIMULATION")
    log(f"  Population: {POPULATION}  |  Max days: {MAX_SIMULATED_DAYS}  |  Seed: {seed}")
    log("=" * 90)

    config = WorldConfig(population=POPULATION, terrain="riverside", seed=seed)
    gen = TownGenerator(config)
    gen.generate()
    grid, buildings = gen.grid, gen.buildings

    from core.npc.cognition.router import CognitionRouter, policy_all_deterministic
    router = CognitionRouter(policy=policy_all_deterministic())
    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=MockProvider(), seed=seed, router=router,
    )
    npcs = mgr.spawn_population(POPULATION)
    clock = GameClock()

    # Boost construct action priority (overseer would do this in production)
    from core.npc.cognition.planner.actions import ActionDef
    orig = mgr.planner.actions.get("construct")
    mgr.planner.actions.register(ActionDef(
        action_id="construct",
        display_name="Contribute to construction",
        personality_weights={"conscientiousness": 0.8, "agreeableness": 0.4},
        time_weights={"morning": 2.5, "afternoon": 2.5, "early_morning": 1.0},
        base_utility=2.5,
        precondition=orig.precondition,
        target_selector=orig.target_selector,
        tags={"economy", "community"},
    ))

    # Find a clear site
    site_x, site_z = _find_clear_site(grid, buildings, npcs)
    if site_x is None:
        return False, {"error": "no clear site"}

    site, msg = mgr.economy.construction.start_construction(
        "church", site_x, site_z, grid, game_time=0.0,
    )
    if site is None:
        return False, {"error": msg}

    log(f"\nChurch site: ({site_x}, {site_z})")
    log(f"  Requires: wood=100, stone=50, labour=120 min")

    # Give NPCs resources (spread so everyone needs to contribute)
    wood_per_npc = 20
    stone_per_npc = 10
    for npc in npcs:
        npc.inventory["wood"] = npc.inventory.get("wood", 0) + wood_per_npc
        npc.inventory["stone"] = npc.inventory.get("stone", 0) + stone_per_npc

    total_wood = sum(n.inventory.get("wood", 0) for n in npcs)
    total_stone = sum(n.inventory.get("stone", 0) for n in npcs)

    log(f"  Distributed: wood={total_wood} stone={total_stone}\n")
    for npc in npcs:
        dist = abs(npc.x - site_x) + abs(npc.z - site_z)
        log(f"  {npc.name:12s} ({npc.occupation:12s}) dist={dist:2d}")

    # --- Metrics tracker ---
    tracker = NPCMetricsTracker(npcs)

    start_time = time.time()
    completed = False
    completion_day = -1
    phases_seen: set[str] = set()
    total_ticks = MAX_SIMULATED_DAYS * TICKS_PER_DAY
    last_reported_phase = ""

    for tick in range(total_ticks):
        clock.tick(TICK_DELTA)
        await mgr.tick(clock, TICK_DELTA)

        # Sample metrics
        if tick % SAMPLE_INTERVAL == 0:
            tracker.sample(npcs, day=clock.day, game_minutes=clock.minutes)

        # Check construction every 60 ticks
        if tick % 60 == 0:
            current_site = mgr.economy.construction.get_site(site.site_id)
            if current_site is None:
                completed = True
                completion_day = clock.day
                break

            phase = current_site.phase.value
            phases_seen.add(phase)

            if phase != last_reported_phase:
                elapsed = time.time() - start_time
                log(f"  Day {clock.day:2d} tick {tick:5d} | "
                    f"phase={phase:12s} {current_site.progress:5.1%} | "
                    f"res={current_site.resource_progress:.0%} "
                    f"lab={current_site.labour_progress:.0%} | "
                    f"{elapsed:.0f}s")
                last_reported_phase = phase

            if current_site.is_complete:
                mgr.economy.construction.check_and_complete(
                    current_site.site_id, grid,
                )
                completed = True
                completion_day = clock.day
                break

    elapsed = time.time() - start_time

    # ---------- Results ----------
    log("\n" + "=" * 90)
    log("RESULTS")
    log("=" * 90)

    # 1. Completion
    log(f"\n1. CONSTRUCTION: {'COMPLETED day ' + str(completion_day) if completed else 'INCOMPLETE'}")
    if not completed:
        s = mgr.economy.construction.get_site(site.site_id)
        if s:
            log(f"   Progress: {s.progress:.1%} (res={s.resource_progress:.1%} lab={s.labour_progress:.1%})")
            log(f"   Still needs: {s.resources_still_needed()}")

    # 2. Resources
    remaining_wood = sum(n.inventory.get("wood", 0) for n in npcs)
    remaining_stone = sum(n.inventory.get("stone", 0) for n in npcs)
    wood_used = total_wood - remaining_wood
    stone_used = total_stone - remaining_stone
    log(f"\n2. RESOURCES: wood {wood_used}/{total_wood}, stone {stone_used}/{total_stone}")

    # 3. Full NPC diagnostics
    tracker.print_full_report() if not quiet else None

    # 4. Non-contributors analysis
    log(f"\n--- NON-CONTRIBUTOR ANALYSIS ---")
    for p in tracker.profiles.values():
        if p.construction_samples == 0:
            pcts = p.activity_pcts
            log(f"  {p.name:12s} ({p.occupation}): NEVER CONSTRUCTED")
            log(f"    Instead: {', '.join(f'{k}={v:.0f}%' for k,v in sorted(pcts.items(), key=lambda x:-x[1])[:4])}")
            log(f"    Avg energy={p.avg_energy:.2f} hunger={p.avg_hunger:.2f} tiles_visited={len(p.unique_tiles_visited)}")

    # Pass/fail
    participation = sum(1 for p in tracker.profiles.values() if p.construction_samples > 0)
    all_pass = completed and participation >= 2 and wood_used > 0

    log(f"\nElapsed: {elapsed:.1f}s  |  Participation: {participation}/{POPULATION}")
    log(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    log("=" * 90)

    summary = tracker.get_summary()
    summary.update({
        "completed": completed,
        "completion_day": completion_day,
        "wood_used": wood_used,
        "stone_used": stone_used,
        "participation": participation,
        "phases_seen": len(phases_seen),
        "seed": seed,
    })
    return all_pass, summary


async def run_multi(n_runs: int = 5):
    """Run the simulation N times with different seeds and report outliers."""
    print("\n" + "=" * 90)
    print(f"MULTI-RUN ANALYSIS ({n_runs} runs)")
    print("=" * 90)

    results: list[dict] = []
    seeds = [42 + i * 17 for i in range(n_runs)]
    passes = 0

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{n_runs} (seed={seed}) ---")
        passed, summary = await run_church_construction_test(seed=seed, quiet=True)
        results.append(summary)
        status = "PASS" if passed else "FAIL"
        day = summary.get("completion_day", -1)
        bal = summary.get("avg_balance", 0)
        part = summary.get("participation", 0)
        print(f"  {status} | day={day} | balance={bal:.0f} | participation={part}/{POPULATION}")
        if passed:
            passes += 1

    # Aggregate analysis
    print(f"\n{'=' * 90}")
    print(f"AGGREGATE ({passes}/{n_runs} passed)")
    print(f"{'=' * 90}")

    completed_runs = [r for r in results if r.get("completed")]
    if completed_runs:
        days = [r["completion_day"] for r in completed_runs]
        print(f"  Completion days: min={min(days)} max={max(days)} avg={sum(days)/len(days):.1f}")

    balances = [r.get("avg_balance", 0) for r in results]
    print(f"  Life balance: min={min(balances):.0f} max={max(balances):.0f} avg={sum(balances)/len(balances):.0f}")

    participations = [r.get("participation", 0) for r in results]
    print(f"  Participation: min={min(participations)} max={max(participations)} avg={sum(participations)/len(participations):.1f}")

    # Flag outlier runs
    print(f"\n--- OUTLIER RUNS ---")
    import statistics
    if len(balances) > 2:
        mean_bal = statistics.mean(balances)
        std_bal = statistics.stdev(balances)
        for i, r in enumerate(results):
            bal = r.get("avg_balance", 0)
            if std_bal > 0 and abs(bal - mean_bal) / std_bal > 1.5:
                print(f"  Run {i+1} (seed={r['seed']}): balance={bal:.0f} "
                      f"(z={abs(bal - mean_bal)/std_bal:.1f})")
    else:
        print("  Not enough runs for outlier detection")

    # Wonky outliers
    never_slept = [r.get("npcs_never_slept", 0) for r in results]
    never_ate = [r.get("npcs_never_ate", 0) for r in results]
    print(f"\n  NPCs never slept (per run): {never_slept}")
    print(f"  NPCs never ate (per run):   {never_ate}")

    print(f"\n{'=' * 90}")
    return passes == n_runs


if __name__ == "__main__":
    if "--multi" in sys.argv:
        n = 5
        for arg in sys.argv:
            if arg.startswith("--runs="):
                n = int(arg.split("=")[1])
        result = asyncio.run(run_multi(n))
    else:
        result, _ = asyncio.run(run_church_construction_test())
    sys.exit(0 if result else 1)
