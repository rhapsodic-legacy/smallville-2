"""Foundation behavioural eval — the dashboard the rebuild is steered by.

NOT a pass/fail test. It runs a deterministic MockProvider sim and emits
metrics that capture whether the scheduling / planning / town-goal layer
is BEHAVING, not just whether components pass in isolation. Run it as a
baseline on the current code, then re-run after each rebuild phase to see
the numbers move toward target.

Why MockProvider: the foundation bugs (replan wipes commitments, schedule
bloat, no organic crediting) are provider-independent plumbing, so a
deterministic sim reproduces them in seconds and gives stable numbers to
compare phase-over-phase. Emergent-quality (does Jasper argue, does
sentiment shift) is a separate, expensive Gemma eval — see
diagnostic_bridge_objector.py.

Metrics (and their rebuild target):
  goal_completion_rate   organic cycles COMPLETED / total cycles      target > 0
  contributions_total    record_contribution calls that stuck         target > 0
  committed_but_uncredited  NPC-days holding a goal entry that never
                            credited                                   target ~ 0
  schedule_len_max/mean  daily_schedule size (bloat)                   target <= 12
  replan_growth          net entries added by replan over a day        target ~ 0
  sec_per_sim_day        wall-clock per simulated day, per pop         scalability

Run:
  python3 tests/simulation/eval_foundation.py
  python3 tests/simulation/eval_foundation.py --pops 10,30 --days 8
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Determinism: Python randomises string hashing per process, which makes
# set-of-npc_id iteration order (and thus conversation encounters) vary
# run-to-run — a steering instrument must be reproducible. Pin the hash
# seed by re-exec'ing once before any heavy imports if it isn't set.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.npc.cognition.router import CognitionRouter
from core.npc.cognition.router.policy import CognitionPolicy, ROUTE_DETERMINISTIC
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world
from core.world.town_agenda import create_goal_from_template, GoalStatus

SCHEDULE_CAP = 12  # mirror the Phase 0 bounded-schedule target


def _make_sim(pop, seed, deterministic=False):
    config = WorldConfig(population=pop, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(), llm=llm)
    memory.initialise()
    policy = CognitionPolicy()
    policy.set_mode("self_review", ROUTE_DETERMINISTIC)
    # deterministic=True forces the SOUND occupation-template schedule
    # everywhere (bypasses the broken LLM-parser + 1-entry planner paths)
    # to validate whether the foundation completes goals given a good day.
    mgr = NPCManager(grid=grid, buildings=buildings, llm=llm, seed=seed,
                     memory=memory, router=CognitionRouter(policy=policy),
                     deterministic=deterministic)
    mgr.spawn_population(pop)
    return mgr, GameClock()


class _SnapCounter(logging.Handler):
    """Counts pathing-failure ('snapping') log records — a movement-health
    signal the rebuild could disturb (schedule supplies nav targets)."""
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.snaps = 0

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "snapping" in msg or "no path" in msg:
            self.snaps += 1


async def _eval_one(pop, days, seed, deterministic=False):
    from core.npc import manager as _mgrmod

    mgr, clock = _make_sim(pop, seed, deterministic)

    contributions = []
    orig = mgr.town_agenda.record_contribution

    def wrapped(goal_id, npc_id, current_day=None):
        ok = orig(goal_id, npc_id, current_day=current_day)
        contributions.append((goal_id, npc_id))
        return ok
    mgr.town_agenda.record_contribution = wrapped

    # --- pathing-failure counter (logging handler, no core changes) ---
    snap = _SnapCounter()
    mlog = logging.getLogger("core.npc.manager")
    prev_level = mlog.level
    mlog.setLevel(logging.INFO)
    mlog.addHandler(snap)

    # --- conversation-churn counter (monkeypatch the initiation call) ---
    convo = {"total": 0, "pair_day": {}}
    _orig_init = _mgrmod.initiate_conversation

    async def _init_wrap(npc, other, llm, current_minutes, *a, **k):
        convo["total"] += 1
        key = (frozenset((npc.npc_id, other.npc_id)),
               int(current_minutes // MINUTES_PER_DAY))
        convo["pair_day"][key] = convo["pair_day"].get(key, 0) + 1
        return await _orig_init(npc, other, llm, current_minutes, *a, **k)
    _mgrmod.initiate_conversation = _init_wrap

    # Seed willing contributors so the participation GATE isn't the
    # variable — this eval isolates the credit/plan PATH.
    for npc in mgr.npcs:
        npc.self_concept["supports:repair_bridge"] = 0.9

    cycles = 0
    sched_lens = []
    committed_uncredited_npc_days = 0
    commit_live_max = 0  # most live commitments any NPC holds (bounded?)
    # Completion funnel: which stage drops off? Each is a set of npc_ids.
    committed_npcs = set()   # ever held a bridge commitment
    projected_npcs = set()   # ever had a bridge goal entry in-schedule
    cycle_records = []       # per-cycle {progress, required, status}
    _captured = set()        # id()s of cycles already captured

    real_delta = 8.0
    gm_per_tick = 9.6
    ticks = int(days * MINUTES_PER_DAY / gm_per_tick) + 1
    last_day = -1
    start = time.time()

    for _ in range(ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)

        for n in mgr.npcs:
            sched_lens.append(len(n.daily_schedule))

        if clock.day != last_day:
            last_day = clock.day
            # Keep a bridge cycle on the docket each day.
            active = [g for g in mgr.town_agenda.active_and_proposed()
                      if g.goal_id == "repair_bridge"]
            if not active:
                # The previous cycle (about to be overwritten) has
                # resolved — capture its distinct-contributor count.
                old = mgr.town_agenda.get("repair_bridge")
                if old is not None and id(old) not in _captured:
                    _captured.add(id(old))
                    cycle_records.append({
                        "progress": old.progress,
                        "required": old.required_contributions,
                        "status": old.status.value,
                    })
                g = create_goal_from_template("repair_bridge", clock.day)
                if g and mgr.town_agenda.propose(g, clock.day):
                    cycles += 1
            # NPC-days where a goal entry is sitting in the schedule.
            for n in mgr.npcs:
                if any(e.goal_id for e in n.daily_schedule):
                    committed_uncredited_npc_days += 1
            # Durable commitments held (Phase 2) — must stay bounded.
            commit_live_max = max(
                commit_live_max,
                max((len(n.commitments) for n in mgr.npcs), default=0),
            )
            # Completion funnel sampling.
            for n in mgr.npcs:
                if any(c.goal_id == "repair_bridge" for c in n.commitments):
                    committed_npcs.add(n.npc_id)
                if any(e.goal_id == "repair_bridge" for e in n.daily_schedule):
                    projected_npcs.add(n.npc_id)

    elapsed = time.time() - start

    # Restore the global hooks.
    _mgrmod.initiate_conversation = _orig_init
    mlog.removeHandler(snap)
    mlog.setLevel(prev_level)

    # Capture the final (still-live) cycle for the funnel.
    final = mgr.town_agenda.get("repair_bridge")
    if final is not None and id(final) not in _captured:
        cycle_records.append({
            "progress": final.progress,
            "required": final.required_contributions,
            "status": final.status.value,
        })

    # Count completion PER CYCLE (the agenda overwrites _goals by goal_id,
    # so counting from _goals only ever sees the last cycle — a
    # measurement bug that masked real completions).
    completed = sum(1 for r in cycle_records if r["status"] == "completed")
    expired = sum(1 for r in cycle_records if r["status"] == "expired")

    goal_required = cycle_records[0]["required"] if cycle_records else 0
    max_distinct = max((r["progress"] for r in cycle_records), default=0)
    cycle_statuses = [r["status"] for r in cycle_records]

    convos = convo["total"]
    max_pair_day = max(convo["pair_day"].values()) if convo["pair_day"] else 0
    distinct_pair_days = len(convo["pair_day"])
    repeat_rate = (1 - distinct_pair_days / convos) if convos else 0.0

    return {
        "pop": pop, "days": days,
        "cycles": cycles, "completed": completed, "expired": expired,
        "completion_rate": (completed / cycles) if cycles else 0.0,
        "contributions_total": len(contributions),
        "committed_uncredited_npc_days": committed_uncredited_npc_days,
        "sched_len_max": max(sched_lens) if sched_lens else 0,
        "sched_len_mean": (sum(sched_lens) / len(sched_lens)) if sched_lens else 0.0,
        "convos_total": convos,
        "max_pair_day": max_pair_day,   # churn: most repeats of one pair in a day
        "repeat_rate": repeat_rate,     # share of convos that re-hit a same-day pair
        "path_snaps": snap.snaps,       # pathing failures
        "commit_live_max": commit_live_max,  # durable commitments held (bounded?)
        # Completion funnel — which stage collapses?
        "f_committed": len(committed_npcs),
        "f_projected": len(projected_npcs),
        "f_required": goal_required,
        "f_max_distinct": max_distinct,  # best distinct-contributor count any cycle hit
        "cycle_statuses": cycle_statuses,
        "sec_per_sim_day": elapsed / days,
    }


async def main(pops, days, seed, deterministic=False):
    print("=" * 78)
    print(f"FOUNDATION EVAL  (MockProvider, days={days}, seed={seed})")
    print("contributors seeded supports:repair_bridge=0.9 — isolates credit/plan path")
    print("=" * 78)
    rows = []
    for pop in pops:
        rows.append(await _eval_one(pop, days, seed, deterministic))

    print("GOAL & SCHEDULE health:")
    print(f"{'pop':>4} {'cycles':>6} {'compl':>5} {'expir':>5} "
          f"{'compl_rate':>10} {'contribs':>8} {'uncredited':>10} "
          f"{'sched_max':>9} {'sched_mean':>10}")
    for r in rows:
        print(f"{r['pop']:>4} {r['cycles']:>6} {r['completed']:>5} "
              f"{r['expired']:>5} {r['completion_rate']:>10.2f} "
              f"{r['contributions_total']:>8} "
              f"{r['committed_uncredited_npc_days']:>10} "
              f"{r['sched_len_max']:>9} {r['sched_len_mean']:>10.1f}")

    print("\nADJACENT health (rebuild could disturb these):")
    print(f"{'pop':>4} {'convos':>7} {'max_pair_day':>12} {'repeat_rate':>11} "
          f"{'path_snaps':>10} {'commit_max':>10} {'s/day':>7}")
    for r in rows:
        print(f"{r['pop']:>4} {r['convos_total']:>7} {r['max_pair_day']:>12} "
              f"{r['repeat_rate']:>11.2f} {r['path_snaps']:>10} "
              f"{r['commit_live_max']:>10} {r['sec_per_sim_day']:>7.2f}")

    print("\nCOMPLETION FUNNEL (where does it collapse? required vs reached):")
    print(f"{'pop':>4} {'committed':>9} {'projected':>9} {'max_distinct':>12} "
          f"{'required':>8}  -> {'completes?':>10}")
    for r in rows:
        ok = "yes" if r["f_max_distinct"] >= r["f_required"] and r["f_required"] else "NO"
        print(f"{r['pop']:>4} {r['f_committed']:>9} {r['f_projected']:>9} "
              f"{r['f_max_distinct']:>12} {r['f_required']:>8}  -> {ok:>10}")
        print(f"       per-cycle status: {r['cycle_statuses']}")

    print("-" * 78)
    print(f"TARGETS:  compl_rate > 0   contribs > 0   sched_max <= {SCHEDULE_CAP}"
          "   uncredited ~ 0   (adjacent: hold or improve, don't regress)")
    worst_max = max(r["sched_len_max"] for r in rows)
    any_completed = any(r["completed"] for r in rows)
    print(f"VERDICT:  goal completion {'OK' if any_completed else 'BROKEN (0)'}"
          f" | schedule {'OK' if worst_max <= SCHEDULE_CAP else f'BLOATED ({worst_max})'}")
    print("NOTE: desync is covered separately by diagnostic_instrumented_sim.py "
          "+ analyse_diagnostic.py")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pops", default="10,30",
                    help="comma-separated population sizes")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template", action="store_true",
                    help="force the sound occupation-template schedule "
                         "(deterministic) instead of the LLM-parse path")
    args = ap.parse_args()
    pops = [int(p) for p in args.pops.split(",")]
    asyncio.run(main(pops, args.days, args.seed, args.template))
