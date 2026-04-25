"""
60-day stability simulation test.

Runs 10 NPCs for 60 simulated days, tracking per-NPC life balance,
sync scores, action variety, and outlier detection. Produces rich
diagnostics showing whether NPCs maintain human-like daily routines
or degrade into robotic loops.

Run: python3 tests/simulation/diagnostic_60day_stability.py
Multi: python3 tests/simulation/diagnostic_60day_stability.py --multi --runs=5
"""

from __future__ import annotations

import asyncio
import statistics
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
SIMULATED_DAYS = 60
TICK_DELTA = 1.0
TICKS_PER_DAY = 1200
TOTAL_TICKS = SIMULATED_DAYS * TICKS_PER_DAY
SAMPLE_INTERVAL = 10

# Thresholds
MAX_SYNC_SCORE = 0.35
MIN_AVG_BALANCE = 20        # Minimum average life-balance score
MAX_OUTLIER_COUNT = 5       # Max outlier flags across all NPCs


async def run_stability_test(seed: int = SEED, quiet: bool = False):
    """Run one 60-day simulation. Returns (passed, summary_dict)."""
    log = (lambda *a, **k: None) if quiet else print

    log("=" * 90)
    log(f"60-DAY STABILITY SIMULATION (seed={seed})")
    log(f"  Population: {POPULATION}  |  Days: {SIMULATED_DAYS}  |  Ticks: {TOTAL_TICKS}")
    log("=" * 90)

    config = WorldConfig(population=POPULATION, terrain="riverside", seed=seed)
    gen = TownGenerator(config)
    gen.generate()
    grid, buildings = gen.grid, gen.buildings

    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=MockProvider(), seed=seed,
    )
    npcs = mgr.spawn_population(POPULATION)
    clock = GameClock()

    log(f"\nSpawned {len(npcs)} NPCs:")
    for npc in npcs:
        log(f"  {npc.name} ({npc.occupation}) at ({npc.x}, {npc.z})")

    # --- Metrics tracker ---
    tracker = NPCMetricsTracker(npcs)

    start_time = time.time()
    report_interval = TICKS_PER_DAY * 10

    for tick in range(TOTAL_TICKS):
        clock.tick(TICK_DELTA)
        await mgr.tick(clock, TICK_DELTA)

        # Sample metrics
        if tick % SAMPLE_INTERVAL == 0:
            tracker.sample(npcs, day=clock.day, game_minutes=clock.minutes)

        # Progress report
        if tick > 0 and tick % report_interval == 0:
            elapsed = time.time() - start_time
            s = tracker.get_summary()
            log(
                f"  Day {clock.day:3d} | sync={s['avg_sync']:.3f} | "
                f"balance={s['avg_balance']:.0f} | "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - start_time

    # ---------- Results ----------
    log("\n" + "=" * 90)
    log("RESULTS")
    log("=" * 90)

    summary = tracker.get_summary()

    # 1. Sync
    sync_pass = summary["avg_sync"] <= MAX_SYNC_SCORE
    log(f"\n1. SYNC: {summary['avg_sync']:.3f} (threshold: {MAX_SYNC_SCORE}) {'PASS' if sync_pass else 'FAIL'}")

    # 2. Life balance
    balance_pass = summary["avg_balance"] >= MIN_AVG_BALANCE
    log(f"2. BALANCE: avg={summary['avg_balance']:.0f} min={summary['min_balance']:.0f} "
        f"max={summary['max_balance']:.0f} (threshold: {MIN_AVG_BALANCE}) {'PASS' if balance_pass else 'FAIL'}")

    # 3. Outliers
    outlier_pass = summary["outlier_count"] <= MAX_OUTLIER_COUNT
    log(f"3. OUTLIERS: {summary['outlier_count']} flags (threshold: {MAX_OUTLIER_COUNT}) "
        f"{'PASS' if outlier_pass else 'FAIL'}")

    # 4. Basic survival
    never_slept = summary["npcs_never_slept"]
    never_ate = summary["npcs_never_ate"]
    survival_pass = never_slept < POPULATION and never_ate < POPULATION
    log(f"4. SURVIVAL: never_slept={never_slept} never_ate={never_ate} "
        f"{'PASS' if survival_pass else 'FAIL'}")

    # Full diagnostics
    if not quiet:
        tracker.print_full_report()

    all_pass = sync_pass and balance_pass and outlier_pass and survival_pass

    log(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    log("=" * 90)

    summary["seed"] = seed
    summary["passed"] = all_pass
    summary["elapsed"] = elapsed
    return all_pass, summary


async def run_multi(n_runs: int = 5):
    """Run the simulation N times with different seeds and report outliers."""
    print("\n" + "=" * 90)
    print(f"MULTI-RUN STABILITY ANALYSIS ({n_runs} runs x {SIMULATED_DAYS} days)")
    print("=" * 90)

    results: list[dict] = []
    seeds = [42 + i * 17 for i in range(n_runs)]
    passes = 0

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{n_runs} (seed={seed}) ---")
        passed, summary = await run_stability_test(seed=seed, quiet=True)
        results.append(summary)
        status = "PASS" if passed else "FAIL"
        print(f"  {status} | sync={summary['avg_sync']:.3f} | balance={summary['avg_balance']:.0f} "
              f"| outliers={summary['outlier_count']} | elapsed={summary['elapsed']:.0f}s")
        if passed:
            passes += 1

    # Aggregate
    print(f"\n{'=' * 90}")
    print(f"AGGREGATE ({passes}/{n_runs} passed)")
    print(f"{'=' * 90}")

    syncs = [r["avg_sync"] for r in results]
    bals = [r["avg_balance"] for r in results]
    tiles = [r["avg_tiles_visited"] for r in results]
    energies = [r["avg_energy"] for r in results]

    print(f"  Sync:    min={min(syncs):.3f}  max={max(syncs):.3f}  avg={statistics.mean(syncs):.3f}")
    print(f"  Balance: min={min(bals):.0f}  max={max(bals):.0f}  avg={statistics.mean(bals):.0f}")
    print(f"  Tiles:   min={min(tiles):.0f}  max={max(tiles):.0f}  avg={statistics.mean(tiles):.0f}")
    print(f"  Energy:  min={min(energies):.2f}  max={max(energies):.2f}  avg={statistics.mean(energies):.2f}")

    # Cross-run outliers
    print(f"\n--- CROSS-RUN OUTLIERS ---")
    if len(bals) > 2:
        mean_bal = statistics.mean(bals)
        std_bal = statistics.stdev(bals) if len(bals) > 1 else 0
        for i, r in enumerate(results):
            if std_bal > 0 and abs(r["avg_balance"] - mean_bal) / std_bal > 1.5:
                print(f"  Run {i+1} (seed={r['seed']}): balance={r['avg_balance']:.0f} "
                      f"(z={abs(r['avg_balance']-mean_bal)/std_bal:.1f})")
    else:
        print("  Not enough runs for cross-run outlier detection")

    never_slept_runs = [r["npcs_never_slept"] for r in results]
    never_ate_runs = [r["npcs_never_ate"] for r in results]
    print(f"\n  NPCs never slept (per run): {never_slept_runs}")
    print(f"  NPCs never ate (per run):   {never_ate_runs}")

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
        result, _ = asyncio.run(run_stability_test())
    sys.exit(0 if result else 1)
