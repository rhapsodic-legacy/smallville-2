"""
Multi-day simulation invariants.

Runs the core simulation (no server, no websocket — direct NPCManager
drive) for several in-game days and asserts the town is well-behaved:

  1. Schedule bloat: no NPC's schedule may grow without bound. A
     healthy schedule is 5–12 entries; we flag anything over 20.
  2. Position drift: no NPC may park itself at the grid border or
     more than N tiles from their home/work — the "top-left corner"
     bug the user reported on day 40.
  3. Night-time bed invariant: at night, NPCs should be home (or on
     their way there), not wandering the town square.
  4. Schedule-day stamp: each dawn the schedule_day should advance
     so NPCs regenerate fresh plans instead of accumulating stale
     entries from replans.

These cover the class of bugs that silently worsen over many game
days and are invisible in a 30-second e2e test.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# How far an NPC may be from their home at night before we flag it.
NIGHT_HOME_RADIUS = 6
# Absolute cap on any NPC's position distance from the centre — NPCs
# parked at the map border (e.g. x=-30 on a 60-wide grid) fail this.
BORDER_SAFE_MARGIN = 4
# Bloat threshold: a healthy day has ~7 entries; >20 means an unbounded
# growth bug somewhere (replan appending without replacing, etc.).
SCHEDULE_BLOAT_LIMIT = 20


def make_sim():
    config = WorldConfig(population=8, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(
        grid=grid,
        buildings=buildings,
        llm=llm,
        seed=42,
        memory=memory,
        deterministic=False,  # exercise the LLM/replan path too
    )
    mgr.spawn_population(config.population)
    clock = GameClock()
    return mgr, clock, grid, buildings


async def _run_days(mgr, clock, days: int):
    """Advance the simulation by `days` in-game days, ticking at a
    rate that lets schedules cycle. 1 tick = 1 game-minute of advance
    via clock.tick so we can cover several days in seconds of real time."""
    # clock.tick expects real-seconds-of-wall-time. At the default
    # 1 game-day = 20 real-minutes, 1 real-second ~= 1.2 game-minutes.
    # We want each cognition tick to advance ~10 game-minutes. A
    # real_delta of 8 works out to ~9.6 game-minutes per tick.
    real_delta = 8.0
    game_minutes_per_tick = 9.6
    total_game_minutes = days * MINUTES_PER_DAY
    num_ticks = int(total_game_minutes / game_minutes_per_tick)
    for _ in range(num_ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)


def _assert_schedules_bounded(mgr) -> None:
    offenders = [
        (n.name, len(n.daily_schedule))
        for n in mgr.npcs
        if n.npc_id != "player"
        and len(n.daily_schedule) > SCHEDULE_BLOAT_LIMIT
    ]
    assert not offenders, (
        f"Schedule bloat detected (limit={SCHEDULE_BLOAT_LIMIT}). "
        f"Offenders: {offenders}. Likely cause: replan appending "
        "full-day templates onto an already in-progress schedule."
    )


def _assert_not_parked_at_border(mgr, grid) -> None:
    """No resting NPC may be parked at the very edge of the map."""
    half_w = grid.width // 2
    half_h = grid.height // 2
    offenders = []
    for n in mgr.npcs:
        if n.npc_id == "player":
            continue
        if n.activity == ActivityState.WALKING:
            continue
        # Grid is centred at (0, 0) with half_w tiles on each side.
        max_x = half_w - BORDER_SAFE_MARGIN
        max_z = half_h - BORDER_SAFE_MARGIN
        if abs(n.x) > max_x or abs(n.z) > max_z:
            offenders.append((n.name, round(n.x), round(n.z)))
    assert not offenders, (
        f"NPC(s) parked at the map border (safe margin={BORDER_SAFE_MARGIN} "
        f"from the edge): {offenders}. They should never get this far from "
        "any meaningful landmark — see _reanchor_strays."
    )


def _assert_home_at_night(mgr, clock) -> None:
    """At night, every non-walking NPC should be near their home."""
    if clock.phase.value != "night":
        return  # not night — invariant doesn't apply
    offenders = []
    for n in mgr.npcs:
        if n.npc_id == "player":
            continue
        if n.activity == ActivityState.WALKING:
            continue  # on their way home
        dist = abs(n.x - n.home_x) + abs(n.z - n.home_z)
        if dist > NIGHT_HOME_RADIUS:
            offenders.append((
                n.name, (round(n.x), round(n.z)),
                (n.home_x, n.home_z), int(dist),
            ))
    assert not offenders, (
        f"At night, these NPCs are more than {NIGHT_HOME_RADIUS} tiles "
        f"from home (name, pos, home, distance): {offenders}. "
        f"Time: day {clock.day} {clock.time_string}."
    )


async def test_five_day_simulation_invariants() -> None:
    """Run the sim for 5 in-game days and check all invariants.

    Covers: schedule bloat, border parking, night bed-check.
    """
    mgr, clock, grid, buildings = make_sim()

    # Check invariants periodically as days elapse — a lot of bugs
    # only manifest on specific day transitions (day 2 when first
    # regen happens, day 5 when replan has had time to fire).
    for day_chunk in range(5):
        await _run_days(mgr, clock, 1)
        _assert_schedules_bounded(mgr)
        _assert_not_parked_at_border(mgr, grid)
        _assert_home_at_night(mgr, clock)

    # Final snapshot.
    schedule_sizes = [
        (n.name, len(n.daily_schedule))
        for n in mgr.npcs if n.npc_id != "player"
    ]
    positions = [
        (n.name, round(n.x), round(n.z))
        for n in mgr.npcs if n.npc_id != "player"
    ]
    logger.info("Final schedule sizes after 5 days: %s", schedule_sizes)
    logger.info("Final positions after 5 days: %s", positions)

    _assert_schedules_bounded(mgr)
    _assert_not_parked_at_border(mgr, grid)


async def test_schedule_day_advances_daily() -> None:
    """After each dawn, every NPC's schedule_day should match the
    current day. Without this, schedules never reset across days."""
    mgr, clock, grid, buildings = make_sim()

    await _run_days(mgr, clock, 3)

    for n in mgr.npcs:
        if n.npc_id == "player":
            continue
        assert hasattr(n, "schedule_day"), f"{n.name} has no schedule_day"
        # schedule_day should equal the current day (NPC regenerated
        # at most one day ago, never stuck on day 1's schedule).
        lag = clock.day - n.schedule_day
        assert 0 <= lag <= 1, (
            f"{n.name}: schedule_day={n.schedule_day}, current_day={clock.day} "
            f"— a lag of {lag} days means the NPC is running on a stale schedule."
        )


async def test_night_home_invariant() -> None:
    """Run through a full night phase and assert NPCs are home.

    The night phase is the most legible failure mode: if NPCs are
    wandering the town at 2 AM, something's seriously wrong with
    either scheduling or movement. This test exists to catch the
    exact scenario the user reported ('it's night, NPCs in corner').
    """
    mgr, clock, grid, buildings = make_sim()

    # Run until night falls on day 2 (let them settle for one full day).
    deadline = 4 * MINUTES_PER_DAY
    while (
        clock.day * MINUTES_PER_DAY + clock.minutes
    ) < deadline and clock.phase.value != "night":
        clock.tick(8.0)
        mgr.movement_tick(clock, 8.0)
        await mgr.cognition_tick(clock, 8.0)

    if clock.phase.value != "night":
        # Advance through the day until night
        for _ in range(500):
            clock.tick(8.0)
            mgr.movement_tick(clock, 8.0)
            await mgr.cognition_tick(clock, 8.0)
            if clock.phase.value == "night":
                break

    assert clock.phase.value == "night", (
        f"Could not reach night phase; stuck on {clock.phase.value}"
    )

    # Run deeper into the night (let them finish walking home).
    for _ in range(40):
        clock.tick(8.0)
        mgr.movement_tick(clock, 8.0)
        await mgr.cognition_tick(clock, 8.0)

    _assert_home_at_night(mgr, clock)


async def _run_all() -> int:
    tests = [
        ("five-day invariants", test_five_day_simulation_invariants),
        ("schedule_day advances daily", test_schedule_day_advances_daily),
        ("night home invariant", test_night_home_invariant),
    ]
    fails = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            await fn()
            print("  PASS")
        except AssertionError as e:
            fails.append((name, str(e)))
            print(f"  FAIL: {e}")
        except Exception as e:
            fails.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR: {e}")

    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
