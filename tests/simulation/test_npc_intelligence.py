"""
NPC Intelligence Validation Pipeline.

Tests the Stanford 3-level cognition hierarchy:
  Level 1: Daily schedule (exists)
  Level 2: Hourly decomposition into 2-6 sub-tasks
  Level 3: Moment-to-moment task execution

Validates that NPCs always have concrete sub-tasks, descriptions
are varied and occupation-specific, and the sub-task timer system
advances correctly.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from core.world.generator import TownGenerator, WorldConfig
from core.npc.manager import NPCManager
from core.npc.models import ScheduleEntry, ActivityState
from core.npc.cognition.decompose import decompose_schedule_entry
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock

TICK_DELTA = 1.0   # real seconds per tick (matches live sim)
POPULATION = 10
TICKS = 600        # ~10 real minutes — tests past the 45s subtask depletion point


# ---------- Test infrastructure ----------

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    metrics: dict = field(default_factory=dict)


def _print_report(results: list[TestResult]) -> bool:
    """Print report and return True if all passed."""
    print("\n" + "=" * 70)
    print("  NPC INTELLIGENCE TEST REPORT")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        print(f"  [{tag}] {r.name}")
        print(f"         {r.message}")
        if r.metrics:
            print(f"         metrics: {r.metrics}")

    print("-" * 70)
    print(f"  {passed} passed, {failed} failed, {len(results)} total")
    if failed == 0:
        print("  ALL CLEAR")
    else:
        print("  FAILED")
    print("=" * 70)
    return failed == 0


# ---------- Unit tests (no simulation needed) ----------

def test_decompose_produces_subtasks() -> TestResult:
    """Every occupation produces 2+ sub-tasks for a work entry."""
    from core.npc.models import NPC, PersonalityTraits
    import random

    occupations = ["blacksmith", "farmer", "merchant", "tavern_keeper", "priest", "guard"]
    rng = random.Random(42)
    failures = []

    for occ in occupations:
        npc = NPC(
            npc_id=f"test_{occ}", name="Test", age=30,
            personality=PersonalityTraits(),
            backstory="", occupation=occ,
            x=0, z=0, home_x=0, home_z=0, work_x=5, work_z=5,
        )
        entry = ScheduleEntry(slot="morning", activity="work", location="workplace")
        tasks = decompose_schedule_entry(npc, entry, rng)
        if len(tasks) < 2:
            failures.append(f"{occ}: only {len(tasks)} tasks")

    if failures:
        return TestResult("decompose_produces_subtasks", False,
                          f"Failures: {'; '.join(failures)}")
    return TestResult("decompose_produces_subtasks", True,
                      f"All {len(occupations)} occupations produce 2+ sub-tasks",
                      {"occupations_tested": len(occupations)})


def test_decompose_description_variety() -> TestResult:
    """Across 10 decompositions of the same entry, descriptions vary."""
    from core.npc.models import NPC, PersonalityTraits
    import random

    npc = NPC(
        npc_id="test_bs", name="Test", age=30,
        personality=PersonalityTraits(),
        backstory="", occupation="blacksmith",
        x=0, z=0, home_x=0, home_z=0, work_x=5, work_z=5,
    )
    entry = ScheduleEntry(slot="morning", activity="work", location="workplace")

    all_descs: set[str] = set()
    for seed in range(10):
        rng = random.Random(seed)
        tasks = decompose_schedule_entry(npc, entry, rng)
        for t in tasks:
            all_descs.add(t.description)

    if len(all_descs) < 5:
        return TestResult("decompose_description_variety", False,
                          f"Only {len(all_descs)} unique descriptions across 10 runs",
                          {"unique": len(all_descs)})
    return TestResult("decompose_description_variety", True,
                      f"{len(all_descs)} unique descriptions across 10 runs",
                      {"unique": len(all_descs)})


def test_decompose_all_activity_types() -> TestResult:
    """Decomposition handles eat, sleep, socialise, work, and wander."""
    from core.npc.models import NPC, PersonalityTraits
    import random

    npc = NPC(
        npc_id="test", name="Test", age=30,
        personality=PersonalityTraits(),
        backstory="", occupation="farmer",
        x=0, z=0, home_x=0, home_z=0, work_x=5, work_z=5,
    )
    rng = random.Random(42)
    activities = {
        "sleep": "night",
        "eat breakfast": "early_morning",
        "socialise at the tavern": "evening",
        "work at the farm": "morning",
        "wander through town": "afternoon",
    }
    failures = []
    for activity, slot in activities.items():
        entry = ScheduleEntry(slot=slot, activity=activity, location="home")
        tasks = decompose_schedule_entry(npc, entry, rng)
        if not tasks:
            failures.append(f"{activity}: no tasks produced")

    if failures:
        return TestResult("decompose_all_activity_types", False,
                          f"Failures: {'; '.join(failures)}")
    return TestResult("decompose_all_activity_types", True,
                      f"All {len(activities)} activity types produce sub-tasks",
                      {"types_tested": len(activities)})


def test_subtask_has_valid_states() -> TestResult:
    """All sub-tasks have valid activity_state values."""
    from core.npc.models import NPC, PersonalityTraits
    import random

    valid_states = {"idle", "working", "eating", "sleeping", "talking", "gathering"}
    npc = NPC(
        npc_id="test", name="Test", age=30,
        personality=PersonalityTraits(),
        backstory="", occupation="blacksmith",
        x=0, z=0, home_x=0, home_z=0, work_x=5, work_z=5,
    )
    rng = random.Random(42)

    bad_states = []
    for activity in ["work", "eat lunch", "sleep", "socialise", "wander"]:
        entry = ScheduleEntry(slot="morning", activity=activity, location="home")
        tasks = decompose_schedule_entry(npc, entry, rng)
        for t in tasks:
            if t.activity_state not in valid_states:
                bad_states.append(f"{activity}/{t.description}: {t.activity_state}")

    if bad_states:
        return TestResult("subtask_valid_states", False,
                          f"Invalid states: {'; '.join(bad_states)}")
    return TestResult("subtask_valid_states", True,
                      "All sub-tasks have valid activity states")


def test_building_objects_populated() -> TestResult:
    """Buildings have interior objects after generation."""
    cfg = WorldConfig(seed=42, grid_width=60, grid_height=60, population=10)
    gen = TownGenerator(cfg)
    gen.generate()

    buildings_with_objects = sum(1 for b in gen.buildings if b.interior_objects)
    total = len(gen.buildings)

    if buildings_with_objects == 0:
        return TestResult("building_objects_populated", False,
                          "No buildings have interior objects")
    return TestResult("building_objects_populated", True,
                      f"{buildings_with_objects}/{total} buildings have interior objects",
                      {"with_objects": buildings_with_objects, "total": total})


# ---------- Simulation tests ----------

def test_never_idle_simulation() -> TestResult:
    """After 300 ticks, most NPCs have active sub-tasks (not generic idle)."""
    cfg = WorldConfig(seed=42, grid_width=60, grid_height=60, population=POPULATION)
    gen = TownGenerator(cfg)
    gen.generate()
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(gen.grid, gen.buildings, llm=llm, seed=42, memory=memory)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    idle_snapshots = 0
    total_snapshots = 0

    async def _run():
        nonlocal idle_snapshots, total_snapshots
        for tick in range(TICKS):
            await mgr.tick(clock, TICK_DELTA)
            clock.tick(TICK_DELTA)

            for npc in mgr.npcs:
                if npc.activity == ActivityState.WALKING:
                    continue  # walking NPCs don't need sub-tasks
                total_snapshots += 1
                if (npc.current_action_description in ("idle", "")
                        and not npc.current_subtask
                        and not npc.subtask_queue):
                    idle_snapshots += 1

    asyncio.new_event_loop().run_until_complete(_run())

    idle_pct = (idle_snapshots / max(total_snapshots, 1)) * 100
    # Allow up to 20% idle (transition gaps, slot changes, etc.)
    passed = idle_pct < 20
    return TestResult(
        "never_idle_simulation", passed,
        f"{idle_pct:.1f}% of resting snapshots were generic idle (threshold: 20%)",
        {"idle_pct": round(idle_pct, 1), "idle_snapshots": idle_snapshots,
         "total_snapshots": total_snapshots},
    )


def test_description_variety_simulation() -> TestResult:
    """Across a full simulation, NPCs produce 15+ unique action descriptions."""
    cfg = WorldConfig(seed=42, grid_width=60, grid_height=60, population=POPULATION)
    gen = TownGenerator(cfg)
    gen.generate()
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(gen.grid, gen.buildings, llm=llm, seed=42, memory=memory)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    all_descs: set[str] = set()

    async def _run():
        for _ in range(TICKS):
            await mgr.tick(clock, TICK_DELTA)
            clock.tick(TICK_DELTA)
            for npc in mgr.npcs:
                if npc.current_action_description:
                    all_descs.add(npc.current_action_description)

    asyncio.new_event_loop().run_until_complete(_run())

    # Filter out generic ones
    subtask_descs = {d for d in all_descs
                     if not d.startswith("heading")
                     and not d.startswith("talking to")
                     and d not in ("idle", "finishing up", "")}

    passed = len(subtask_descs) >= 10
    return TestResult(
        "description_variety_simulation", passed,
        f"{len(subtask_descs)} unique sub-task descriptions (threshold: 10)",
        {"unique_subtask_descs": len(subtask_descs), "total_unique": len(all_descs)},
    )


def test_subtask_timer_advances() -> TestResult:
    """Sub-task timers decrease over time and sub-tasks rotate."""
    cfg = WorldConfig(seed=42, grid_width=60, grid_height=60, population=POPULATION)
    gen = TownGenerator(cfg)
    gen.generate()
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(gen.grid, gen.buildings, llm=llm, seed=42, memory=memory)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    # Run enough ticks for sub-tasks to expire
    subtask_changes = 0
    prev_subtasks: dict[str, str | None] = {}

    async def _run():
        nonlocal subtask_changes
        for _ in range(TICKS):
            await mgr.tick(clock, TICK_DELTA)
            clock.tick(TICK_DELTA)

            for npc in mgr.npcs:
                curr = npc.current_subtask.description if npc.current_subtask else None
                prev = prev_subtasks.get(npc.npc_id)
                if prev is not None and curr != prev:
                    subtask_changes += 1
                prev_subtasks[npc.npc_id] = curr

    asyncio.new_event_loop().run_until_complete(_run())

    # Expect at least some sub-task rotations
    passed = subtask_changes >= 5
    return TestResult(
        "subtask_timer_advances", passed,
        f"{subtask_changes} sub-task rotations over {TICKS} ticks (threshold: 5)",
        {"rotations": subtask_changes},
    )


def test_no_idle_synchronisation() -> TestResult:
    """NPCs should not all go idle on the same tick (synchronisation bug)."""
    cfg = WorldConfig(seed=42, grid_width=60, grid_height=60, population=POPULATION)
    gen = TownGenerator(cfg)
    gen.generate()
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(gen.grid, gen.buildings, llm=llm, seed=42, memory=memory)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    sync_threshold = max(2, int(POPULATION * 0.6))
    sync_events = 0
    prev_idle: dict[str, bool] = {}

    async def _run():
        nonlocal sync_events
        for tick in range(TICKS):
            await mgr.tick(clock, TICK_DELTA)
            clock.tick(TICK_DELTA)

            # Count NPCs that became idle THIS tick (weren't idle before)
            new_idle_count = 0
            for npc in mgr.npcs:
                is_idle = (npc.activity != ActivityState.WALKING
                           and not npc.current_subtask
                           and not npc.subtask_queue)
                was_idle = prev_idle.get(npc.npc_id, False)
                if is_idle and not was_idle:
                    new_idle_count += 1
                prev_idle[npc.npc_id] = is_idle

            if new_idle_count >= sync_threshold:
                sync_events += 1

    asyncio.new_event_loop().run_until_complete(_run())

    passed = sync_events == 0
    return TestResult(
        "no_idle_synchronisation", passed,
        f"{sync_events} ticks where {sync_threshold}+ NPCs went idle simultaneously",
        {"sync_events": sync_events, "threshold": sync_threshold},
    )


# ---------- Main ----------

def main() -> int:
    # Unit tests
    unit_results = [
        test_decompose_produces_subtasks(),
        test_decompose_description_variety(),
        test_decompose_all_activity_types(),
        test_subtask_has_valid_states(),
        test_building_objects_populated(),
    ]

    # Simulation tests
    sim_results = [
        test_never_idle_simulation(),
        test_description_variety_simulation(),
        test_subtask_timer_advances(),
        test_no_idle_synchronisation(),
    ]

    all_results = unit_results + sim_results
    success = _print_report(all_results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
