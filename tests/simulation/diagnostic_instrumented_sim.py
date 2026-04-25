"""
Instrumented simulation runner — Phase 2/3 of the diagnostic experiment.

Runs 10 NPCs for 14 simulated days with:
  - Per-NPC concrete goals with substep tracking
  - Structured JSON-lines logging of every tick and event
  - Post-run analysis via analyse_diagnostic.py

Run: python3 tests/simulation/diagnostic_instrumented_sim.py
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

from tests.simulation.goals import assign_goals
from tests.simulation.diagnostic_logger import DiagnosticLogger

# ---------- Configuration ----------

POPULATION = 10
SEED = 42
SIMULATED_DAYS = 14
TICK_DELTA = 1.0  # 1 real second per tick

# At default clock speed (1 day = 20 real min = 1200 ticks),
# 14 days = 16,800 ticks. We accelerate by ticking faster.
# With TICK_DELTA=1.0 the clock advances normally.
# Total ticks = SIMULATED_DAYS * ticks_per_day
TICKS_PER_DAY = 1200  # 20 minutes * 60 seconds
TOTAL_TICKS = SIMULATED_DAYS * TICKS_PER_DAY

LOG_PATH = Path(__file__).parent / "diagnostic_log.jsonl"


def _format_game_time(clock: GameClock) -> str:
    """Human-readable game time string."""
    h = int(clock.minutes // 60) % 24
    m = int(clock.minutes % 60)
    return f"Day {clock.day}, {h:02d}:{m:02d}"


async def run_instrumented_simulation():
    print("=" * 80)
    print("INSTRUMENTED DIAGNOSTIC SIMULATION")
    print(f"  Population: {POPULATION}")
    print(f"  Days: {SIMULATED_DAYS}")
    print(f"  Total ticks: {TOTAL_TICKS}")
    print(f"  Log file: {LOG_PATH}")
    print("=" * 80)

    # ---------- Setup ----------
    cfg = WorldConfig(
        seed=SEED, grid_width=60, grid_height=60, population=POPULATION,
    )
    gen = TownGenerator(cfg)
    gen.generate()
    mgr = NPCManager(gen.grid, gen.buildings, seed=SEED)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    # Assign goals
    goals = assign_goals(mgr.npcs)
    print("\nGoal assignments:")
    for npc in mgr.npcs:
        g = goals[npc.npc_id]
        print(f"  {npc.name:12s} ({npc.occupation:14s}): {g.description}")
    print()

    # Logger
    logger = DiagnosticLogger(LOG_PATH)

    # Track previous state for event detection
    prev_state: dict[str, dict] = {}
    for npc in mgr.npcs:
        prev_state[npc.npc_id] = {
            "activity": npc.activity.value,
            "subtask": None,
            "tile": (npc.tile_x, npc.tile_z),
            "walking": False,
            "schedule_slot": "",
        }

    # ---------- Main loop ----------
    wall_start = time.monotonic()
    last_day = -1
    last_slot = ""

    for tick in range(TOTAL_TICKS):
        await mgr.tick(clock, TICK_DELTA)
        clock.tick(TICK_DELTA)

        game_time = _format_game_time(clock)
        current_slot = clock.schedule_slot.value
        game_minutes_elapsed = TICK_DELTA / clock._real_seconds_per_game_minute()

        # Day change report
        if clock.day != last_day:
            last_day = clock.day
            elapsed = time.monotonic() - wall_start
            pct = tick / TOTAL_TICKS * 100
            completed_goals = sum(1 for g in goals.values() if g.completed)
            print(
                f"  Day {clock.day:3d} | tick {tick:6d}/{TOTAL_TICKS} "
                f"({pct:5.1f}%) | wall={elapsed:6.1f}s | "
                f"goals complete: {completed_goals}/{POPULATION}"
            )

        # Slot transition logging
        if current_slot != last_slot:
            last_slot = current_slot
            for npc in mgr.npcs:
                entry = npc.get_current_schedule_entry(current_slot)
                logger.log_event(
                    tick, game_time, npc.npc_id, npc.name,
                    "SLOT_TRANSITION",
                    {
                        "new_slot": current_slot,
                        "schedule_activity": (
                            entry.activity if entry else "none"
                        ),
                        "schedule_location": (
                            entry.location if entry else "none"
                        ),
                    },
                )

        # Per-NPC tick processing
        for npc in mgr.npcs:
            goal = goals.get(npc.npc_id)
            prev = prev_state[npc.npc_id]

            # Detect events
            is_walking = npc.activity == ActivityState.WALKING
            was_walking = prev["walking"]
            curr_subtask = (
                npc.current_subtask.description
                if npc.current_subtask else None
            )
            curr_tile = (npc.tile_x, npc.tile_z)

            # Departure
            if is_walking and not was_walking:
                logger.log_event(
                    tick, game_time, npc.npc_id, npc.name,
                    "DEPARTURE",
                    {"from": list(prev["tile"]), "activity": npc.activity.value},
                )

            # Arrival
            if not is_walking and was_walking:
                logger.log_event(
                    tick, game_time, npc.npc_id, npc.name,
                    "ARRIVAL",
                    {"at": list(curr_tile)},
                )

            # Subtask change
            if curr_subtask != prev["subtask"]:
                if curr_subtask:
                    logger.log_event(
                        tick, game_time, npc.npc_id, npc.name,
                        "SUBTASK_START",
                        {
                            "description": curr_subtask,
                            "duration": round(
                                npc.subtask_time_remaining, 1,
                            ),
                        },
                    )
                elif prev["subtask"]:
                    logger.log_event(
                        tick, game_time, npc.npc_id, npc.name,
                        "SUBTASK_COMPLETE",
                        {"description": prev["subtask"]},
                    )

            # Goal progress
            if goal:
                # Determine NPC's effective location for goal matching
                npc_location = _resolve_npc_location(npc, mgr)
                event = goal.tick(
                    npc.activity.value,
                    npc_location,
                    game_minutes_elapsed,
                    tick,
                )
                if event:
                    step_idx = goal.current_step_index
                    if event == "GOAL_STEP_COMPLETE":
                        step_idx -= 1  # just completed previous
                    logger.log_event(
                        tick, game_time, npc.npc_id, npc.name,
                        event,
                        {
                            "goal": goal.description,
                            "step_index": step_idx,
                            "step_desc": (
                                goal.steps[step_idx].description
                                if step_idx < len(goal.steps) else "all"
                            ),
                            "progress": round(goal.progress_fraction, 3),
                        },
                    )

            # Log full tick state (every 10 ticks to keep file manageable)
            if tick % 10 == 0:
                logger.log_tick_state(tick, game_time, npc, goal)

            # Update prev state
            prev_state[npc.npc_id] = {
                "activity": npc.activity.value,
                "subtask": curr_subtask,
                "tile": curr_tile,
                "walking": is_walking,
                "schedule_slot": current_slot,
            }

    # ---------- Final report ----------
    logger.close()
    elapsed = time.monotonic() - wall_start

    print()
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Log events: {logger.event_count}")
    print(f"  Log file: {LOG_PATH}")
    print()
    print("GOAL RESULTS:")
    for npc in mgr.npcs:
        g = goals[npc.npc_id]
        steps_done = sum(1 for s in g.steps if s.completed)
        status = "COMPLETE" if g.completed else f"{steps_done}/{len(g.steps)}"
        print(
            f"  {npc.name:12s} ({npc.occupation:14s}): "
            f"{g.description:40s} [{status}] "
            f"({g.progress_fraction:.0%})"
        )

    completed = sum(1 for g in goals.values() if g.completed)
    partial = sum(
        1 for g in goals.values()
        if not g.completed and g.current_step_index > 0
    )
    print(f"\n  Goals complete: {completed}/{POPULATION}")
    print(f"  Goals with partial progress: {partial}/{POPULATION}")
    print("=" * 80)


def _resolve_npc_location(npc, mgr: NPCManager) -> str:
    """Map NPC position to a location name for goal matching.

    Uses proximity (Manhattan distance ≤ 3) to match locations,
    and returns ALL matching location names so goals can match
    flexibly.
    """
    tx, tz = npc.tile_x, npc.tile_z
    # Home proximity (within 3 tiles)
    if abs(tx - npc.home_x) + abs(tz - npc.home_z) <= 3:
        return "home"
    # Work proximity (within 3 tiles)
    if abs(tx - npc.work_x) + abs(tz - npc.work_z) <= 3:
        return "work"
    # Check building proximity (within 3 tiles of door)
    for b in mgr.buildings:
        if abs(tx - b.door_x) + abs(tz - b.door_z) <= 3:
            return b.building_type
    return "any"


def main():
    asyncio.run(run_instrumented_simulation())


if __name__ == "__main__":
    main()
