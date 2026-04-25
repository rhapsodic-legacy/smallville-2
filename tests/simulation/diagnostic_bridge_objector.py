"""
Bridge Objector diagnostic — non-deterministic emergent-behaviour sim.

Seeds one NPC with `opposes:repair_bridge = 0.9` and keeps the town's
`repair_bridge` goal on the docket. Logs what emerges:
- Bridge goal progress and objector participation probability each day
- Whether the objector was injected with the goal (sample outcome)
- Any episodic memory on the objector mentioning the bridge
- Final self_concept + sentiment snapshot for the objector

No pass/fail assertions — this is a logging harness. We read the logs.

Uses local Gemma via Ollama for non-deterministic LLM output. Fails
fast if Ollama isn't reachable.

Run:
  python3 tests/simulation/diagnostic_bridge_objector.py
  python3 tests/simulation/diagnostic_bridge_objector.py --days=60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.world.generator import TownGenerator, WorldConfig
from core.world.town_agenda import (
    GoalStatus, create_goal_from_template,
)
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager
from core.npc.gemma_provider import GemmaProvider, ollama_available


POPULATION = 10
SEED = 42
TICK_DELTA = 1.0
TICKS_PER_DAY = 1200
DEFAULT_DAYS = 30
PROGRESS_REPORT_DAYS = 5


def _bridge_memories(memory, npc_id: str) -> list:
    """All non-compacted memories for this NPC mentioning the bridge."""
    mems = memory.episodic.get_recent(npc_id, limit=200)
    return [m for m in mems if "bridge" in m.description.lower()]


async def run(days: int = DEFAULT_DAYS) -> None:
    if not ollama_available():
        print("ERROR: Ollama not reachable at http://localhost:11434.")
        print("Start it with: brew services start ollama")
        sys.exit(1)

    print("=" * 90)
    print(f"BRIDGE OBJECTOR DIAGNOSTIC  (days={days}, pop={POPULATION}, seed={SEED})")
    print("=" * 90)

    config = WorldConfig(
        population=POPULATION, terrain="riverside", seed=SEED,
    )
    gen = TownGenerator(config)
    gen.generate()
    grid, buildings = gen.grid, gen.buildings

    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=GemmaProvider(), seed=SEED,
    )
    npcs = mgr.spawn_population(POPULATION)
    clock = GameClock()

    # Pick the most conscientious NPC as objector — their personality
    # would normally pull them strongly toward repairing the bridge, so
    # the opposition belief has something real to fight against.
    objector = max(npcs, key=lambda n: n.personality.conscientiousness)
    objector.self_concept["opposes:repair_bridge"] = 0.9

    print(f"\nObjector: {objector.name} ({objector.occupation})")
    print(f"  conscientiousness = {objector.personality.conscientiousness:.2f}")
    print(f"  injected belief: opposes:repair_bridge = 0.9")
    print(f"  self_concept_summary: {objector.self_concept_summary()!r}")

    # Other NPCs — show their conscientiousness for context
    print(f"\nOther NPCs (conscientiousness):")
    for n in sorted(npcs, key=lambda x: -x.personality.conscientiousness):
        if n.npc_id == objector.npc_id:
            continue
        print(f"  {n.name:20s} {n.occupation:14s} {n.personality.conscientiousness:.2f}")
    print()

    daily_log: list[str] = []
    bridge_cycles = 0
    seen_bridge_mem_ids: set = set()

    def _log(line: str) -> None:
        """Append AND stream immediately — so a watcher sees events as they
        happen rather than waiting for end-of-run accumulation."""
        daily_log.append(line)
        print(line, flush=True)

    total_ticks = days * TICKS_PER_DAY
    start = time.time()
    last_reported_day = -1

    for tick in range(total_ticks):
        clock.tick(TICK_DELTA)
        await mgr.tick(clock, TICK_DELTA)

        # Once per game-day (first tick of the day after tick 0)
        if clock.day != last_reported_day:
            last_reported_day = clock.day
            day = clock.day

            # Propose bridge goal if none active and cooldown allows.
            active_bridges = [
                g for g in mgr.town_agenda.active_and_proposed()
                if g.goal_id == "repair_bridge"
            ]
            if not active_bridges:
                goal = create_goal_from_template("repair_bridge", day)
                if goal and mgr.town_agenda.propose(goal, day):
                    bridge_cycles += 1
                    _log(
                        f"[day {day:3d}] PROPOSED repair_bridge "
                        f"(cycle #{bridge_cycles}, deadline day {goal.deadline_day})"
                    )

            # Log current bridge goal state + objector stance.
            active_bridges = [
                g for g in mgr.town_agenda.active_and_proposed()
                if g.goal_id == "repair_bridge"
            ]
            for g in active_bridges:
                p = g.participation_probability(objector)
                score = g.participation_score(objector)
                joined = objector.npc_id in g.contributors
                _log(
                    f"[day {day:3d}] BRIDGE status={g.status.value:9s} "
                    f"progress={g.progress}/{g.required_contributions} "
                    f"objector_score={score:+.2f} p={p:.3f} "
                    f"objector_joined={joined}"
                )

            # Log any recently-completed or expired bridge goals.
            for g in mgr.town_agenda.completed():
                if g.goal_id == "repair_bridge" and g.completed_day == day:
                    _log(
                        f"[day {day:3d}] BRIDGE COMPLETED "
                        f"contributors={sorted(g.contributors)}"
                    )

            # New bridge-related memories for the objector.
            for m in _bridge_memories(mgr.memory, objector.npc_id):
                mid = getattr(m, "memory_id", None) or id(m)
                if mid in seen_bridge_mem_ids:
                    continue
                seen_bridge_mem_ids.add(mid)
                desc = m.description[:120].replace("\n", " ")
                _log(
                    f"[day {day:3d}] OBJECTOR_MEM [{m.category}] {desc!r}"
                )

        # Progress print every N sim days.
        if tick > 0 and tick % (TICKS_PER_DAY * PROGRESS_REPORT_DAYS) == 0:
            elapsed = time.time() - start
            print(
                f"  day {clock.day:3d}  cycles={bridge_cycles}  "
                f"log_lines={len(daily_log)}  elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - start

    # --- Final report ---
    print("\n" + "=" * 90)
    print(f"DAILY LOG  ({len(daily_log)} lines across {days} sim days, "
          f"{bridge_cycles} bridge cycles)")
    print("=" * 90)
    for line in daily_log:
        print(line)

    # --- Final objector state ---
    print("\n" + "=" * 90)
    print("FINAL OBJECTOR STATE")
    print("=" * 90)
    print(f"Name: {objector.name}")
    print(f"Self-concept:")
    for k, v in sorted(objector.self_concept.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} {v:+.2f}")
    print(f"Summary: {objector.self_concept_summary()!r}")

    # Sentiment from objector toward others.
    rels = mgr.sentiment.get_all_for(objector.npc_id)
    print(f"\nSentiment from {objector.name} ({len(rels)} relationships):")
    for r in sorted(rels, key=lambda r: -abs(r.overall_disposition()))[:15]:
        name = next((n.name for n in npcs if n.npc_id == r.npc_to), r.npc_to)
        print(
            f"  → {name:20s} overall={r.overall_disposition():+.2f}  "
            f"{r.to_description()}"
        )

    # Bridge goal history on the agenda.
    print("\nBridge goal history:")
    for gid, g in mgr.town_agenda._goals.items():
        if "bridge" not in gid:
            continue
        print(
            f"  {gid}: status={g.status.value}  progress={g.progress}/"
            f"{g.required_contributions}  contributors={sorted(g.contributors)}"
        )

    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Simulated days (default {DEFAULT_DAYS}).",
    )
    args = parser.parse_args()
    asyncio.run(run(days=args.days))
