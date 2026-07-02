"""
Bridge Objector diagnostic — non-deterministic emergent-behaviour sim.

Seeds one NPC with `opposes:repair_bridge = 0.9` and keeps the town's
`repair_bridge` goal on the docket. Logs what emerges:
- Bridge goal progress and objector participation probability each day
- Whether the objector was injected with the goal (sample outcome)
- Any episodic memory on the objector mentioning the bridge
- Final self_concept + sentiment snapshot for the objector

Non-deterministic cognition is required (MockProvider is deterministic and
hides exactly the emergent cases we want). Two engines:
  --provider mistral  (default) — Mistral API: non-deterministic at API speed.
                        Fast path for de-risking the harness + the criteria.
  --provider gemma    — local Gemma via Ollama: the production engine, truer
                        but ~30 wall-min per sim day here. Confirmatory run.

The run ends with a PRE-REGISTERED CRITERIA VERDICT: the four read-signals
(voiced dissent, indecision calibration, social consequence, organic belief
formation) scored against thresholds fixed before the run, plus a
pre-committed meta-verdict mapping the outcome to a conclusion.

Run:
  python3 tests/simulation/diagnostic_bridge_objector.py            # mistral, 30d
  python3 tests/simulation/diagnostic_bridge_objector.py --days=5   # quick smoke
  python3 tests/simulation/diagnostic_bridge_objector.py --provider gemma --days=30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

from core.world.generator import TownGenerator, WorldConfig
from core.world.town_agenda import (
    GoalStatus, create_goal_from_template,
)
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager
from core.npc.gemma_provider import GemmaProvider, ollama_available
from core.npc.mistral_provider import MistralProvider

load_dotenv()  # MistralProvider reads MISTRAL_API_KEY from the environment


POPULATION = 10
SEED = 42
TICK_DELTA = 1.0
TICKS_PER_DAY = 1200
DEFAULT_DAYS = 30
PROGRESS_REPORT_DAYS = 5
# Observability / watchdog for long unattended runs.
HEARTBEAT_SECONDS = 60        # flushed proof-of-life + progress-rate line
TICK_TIMEOUT_SECONDS = 1200   # abort if ONE tick hangs > 20 min (LLM hung)

# --- Pre-registered read-criteria (Option 3) ---------------------------------
# Thresholds FIXED before the run so the logs answer a falsifiable question
# rather than confirm a hunch. Rationale in MEMORY_V2_ROADMAP.md ("Open
# questions the diagnostic is meant to answer") and AGENT_DIRECTION.md
# ("Dependency order"). The meta-verdict pre-commits which outcome implies
# which conclusion — including the outcome that KILLS the rebuild case.
JOIN_RATE_BAND = (0.05, 0.30)   # C2: designed "human-like indecision" (~14%)
SENTIMENT_DRIFT_MIN = 3.0       # C3: min RELATIVE cooling toward objector (vs
                                #     the town-wide drift) to count as real
                                #     social consequence. Disposition is on a
                                #     +/-100 scale (DIMENSION_MIN/MAX); 3.0 is a
                                #     few points of net cooling, provisional and
                                #     tunable like the rest of the watchlist.
# C1 is judged deterministically — see _count_voiced_opposition.


def _bridge_memories(memory, npc_id: str) -> list:
    """All non-compacted memories for this NPC mentioning the bridge."""
    mems = memory.episodic.get_recent(npc_id, limit=200)
    return [m for m in mems if "bridge" in m.description.lower()]


def _count_voiced_opposition(npc_name: str, memories: list):
    """C1 (deterministic) — count utterances where the objector HIMSELF
    voices opposition to repairing the bridge.

    Replaces the old LLM judge, which false-negatived twice: it sampled
    only the FIRST 15 newest-first conversation memories and truncated
    each at 400 chars, so on a 30-day run it judged only the final days
    (and often never saw the objector's own words at all). This scan is
    speaker-attributed (only lines the objector speaks), covers EVERY
    conversation memory with no truncation, dedupes utterances (turn
    memories + consolidated summaries overlap), skips canned fallback
    lines, and excludes pro-repair idioms ("won't fix itself") that a
    naive token scan would flag. Deterministic: same dump, same answer.
    """
    support_idioms = ("won't fix itself", "wont fix itself",
                      "won't repair itself", "needs repair",
                      "needs fixing", "must repair", "help repair")
    oppose_tokens = ("oppose", "opposed", "won't", "wont", "refuse",
                     "fool's", "waste", "death-trap", "death trap",
                     "no good", "against", "rotten", "not be lendin",
                     "patchwork", "crumbling", "safer rebuilding")
    prefix_a = f"{npc_name}:"
    prefix_b = f"{npc_name} said:"
    seen: set[str] = set()
    quotes: list[str] = []
    for m in memories:
        cat = getattr(m, "category", "")
        if cat not in ("conversation", "conversation_turn"):
            continue
        desc = m.description
        if "bridge" not in desc.lower():
            continue
        if (getattr(m, "metadata", None) or {}).get("fallback"):
            continue  # canned stub, not the objector's voice
        for seg in desc.split(" | "):
            s = seg.strip()
            if s.startswith("Had a conversation with"):
                _, _, s = s.partition(". ")
                s = s.strip()
            if not (s.startswith(prefix_a) or s.startswith(prefix_b)):
                continue
            low = s.lower()
            if "bridge" not in low:
                continue
            key = low[:120]
            if key in seen:
                continue
            if any(t in low for t in support_idioms) and not any(
                t in low for t in ("won't be", "refuse", "oppose")
            ):
                continue
            if any(t in low for t in oppose_tokens):
                seen.add(key)
                quotes.append(s[:170])
    return len(quotes), quotes


def _bridge_self_concept_keys(npc) -> dict:
    """Self-concept keys mentioning the bridge — for C4 organic-formation diff."""
    return {k: v for k, v in npc.self_concept.items() if "bridge" in k.lower()}


def _mean_sentiment_towards(sentiment, npc_id: str):
    """C3 — mean overall disposition of every other NPC *toward* npc_id.
    Returns (mean, count). Sparse storage means count may be 0 at baseline."""
    rels = sentiment.get_all_towards(npc_id)
    if not rels:
        return 0.0, 0
    vals = [r.overall_disposition() for r in rels]
    return sum(vals) / len(vals), len(vals)


def _town_mean_sentiment(sentiment, npcs, exclude_id: str) -> float:
    """C3 control — mean disposition toward every NPC except exclude_id. The
    objector's drift is read RELATIVE to this so general warming/cooling over
    the run doesn't masquerade as (or mask) social consequence aimed at him."""
    means = []
    for n in npcs:
        if n.npc_id == exclude_id:
            continue
        m, cnt = _mean_sentiment_towards(sentiment, n.npc_id)
        if cnt:
            means.append(m)
    return (sum(means) / len(means)) if means else 0.0


def _build_provider(provider: str):
    """Select the non-deterministic LLM backend.

    'mistral' (default) is the fast-path: non-deterministic cognition at API
    speed, used to de-risk the harness and pin criteria. 'gemma' is the
    production cognition engine — truer but ~30 wall-min per sim day on this
    hardware, so reserve it for the confirmatory run once the harness is sound.
    """
    if provider == "gemma":
        if not ollama_available():
            print("ERROR: Ollama not reachable at http://localhost:11434.")
            print("Start it with: brew services start ollama")
            sys.exit(1)
        return GemmaProvider(), f"Gemma ({GemmaProvider.NPC_MODEL})"
    if provider == "mistral":
        try:
            llm = MistralProvider()
        except Exception as exc:  # missing key / SDK
            print(f"ERROR: could not init MistralProvider: {exc}")
            sys.exit(1)
        return llm, f"Mistral ({MistralProvider.NPC_MODEL})"
    if provider == "mock":
        # Deterministic harness de-risking: exercises the full run loop,
        # day-rollover snapshots, sidecar, and table in seconds. Tone is
        # deterministic (no real verdicts), so sentiment trajectory is
        # flat — use it to verify plumbing, not behaviour.
        from core.npc.llm_client import MockProvider
        return MockProvider(), "Mock (deterministic, plumbing-only)"
    print(f"ERROR: unknown provider {provider!r} (use 'mistral'/'gemma'/'mock').")
    sys.exit(1)


def _snapshot_metrics(mgr, npcs, day, prev_tone, events_today):
    """Lightweight daily metrics over LIVE state (no dump).

    The trajectory instrument: called once per snapshot day so a long
    run yields a time series — letting us see whether sentiment
    converges smoothly, oscillates, or drifts, and correlate inflections
    with events (bridge cycles) and, via the final full dump, with what
    NPCs were actually saying. Cheap enough to run daily (a few hundred
    directed pairs); voice is NOT computed here (it needs the whole
    dialogue corpus) — that stays a one-shot at end-of-run.
    """
    import statistics as _st
    from core.memory.reflection import get_tone_tally
    from collections import defaultdict

    disp: list[float] = []
    incoming: dict[str, list[float]] = defaultdict(list)
    for n in npcs:
        for s in mgr.sentiment.get_all_for(n.npc_id):
            d = s.overall_disposition()
            disp.append(d)
            tgt = mgr.get_npc(s.npc_to)
            incoming[tgt.name if tgt else s.npc_to].append(d)

    total = len(disp) or 1
    neg = sum(1 for v in disp if v < -5)
    neu = sum(1 for v in disp if -5 <= v <= 5)
    pos = sum(1 for v in disp if v > 5)
    self_keys = [len(x.self_concept) for x in npcs]

    tone_cum = get_tone_tally()
    tone_day = {k: tone_cum.get(k, 0) - prev_tone.get(k, 0)
                for k in ("tense", "neutral", "warm", "hostile")}

    # Most-disliked NPC this snapshot (emergent-pariah tracking).
    pariah, pariah_mean = None, 0.0
    if incoming:
        pariah, vals = min(incoming.items(), key=lambda kv: _st.mean(kv[1]))
        pariah_mean = _st.mean(vals)

    snap = {
        "day": day,
        "rels": len(disp),
        "neg_pct": round(neg / total, 3),
        "neu_pct": round(neu / total, 3),
        "pos_pct": round(pos / total, 3),
        "mean": round(_st.mean(disp), 2) if disp else 0.0,
        "stdev": round(_st.pstdev(disp), 2) if len(disp) > 1 else 0.0,
        "min": round(min(disp), 1) if disp else 0.0,
        "max": round(max(disp), 1) if disp else 0.0,
        "self_keys_mean": round(_st.mean(self_keys), 2) if self_keys else 0.0,
        "tone_today": tone_day,
        "most_disliked": pariah,
        "most_disliked_mean": round(pariah_mean, 1),
        "events": list(events_today),
    }
    return snap, tone_cum


def _print_trajectory_table(timeseries: list[dict]) -> None:
    """One table = the whole run's social arc at a glance."""
    if not timeseries:
        return
    print("\n" + "=" * 90)
    print("SENTIMENT TRAJECTORY (per snapshot day)")
    print("=" * 90)
    print(f"{'day':>3} {'neg%':>5} {'neu%':>5} {'pos%':>5} {'mean':>6} "
          f"{'min':>6} {'self':>5} {'tone t/n/w/h':>14}  events")
    for s in timeseries:
        t = s["tone_today"]
        tone_str = f"{t['tense']}/{t['neutral']}/{t['warm']}/{t['hostile']}"
        ev = ",".join(s["events"]) if s["events"] else ""
        print(f"{s['day']:>3} "
              f"{s['neg_pct']*100:>4.0f}% {s['neu_pct']*100:>4.0f}% "
              f"{s['pos_pct']*100:>4.0f}% {s['mean']:>+6.1f} {s['min']:>6.1f} "
              f"{s['self_keys_mean']:>5.1f} {tone_str:>14}  {ev}")
    print("=" * 90)


async def run(days: int = DEFAULT_DAYS, provider: str = "mistral",
              dump_path: str | None = None,
              snapshot_every: int = 1,
              timeseries_path: str | None = None) -> None:
    llm, llm_label = _build_provider(provider)

    print("=" * 90)
    print(f"BRIDGE OBJECTOR DIAGNOSTIC  (days={days}, pop={POPULATION}, seed={SEED})")
    print(f"  cognition engine: {llm_label}")
    print("=" * 90)

    config = WorldConfig(
        population=POPULATION, terrain="riverside", seed=SEED,
    )
    gen = TownGenerator(config)
    gen.generate()
    grid, buildings = gen.grid, gen.buildings

    # When dumping, also journal every memory to daily-bucket text
    # files as it is written — crash-safe (a run killed at hour 12
    # keeps everything to there) and mid-run readable (tail/grep
    # <dump>_memories/<npc_id>.txt while the sim runs).
    memory = None
    if dump_path:
        from core.memory.manager import MemoryManager
        from core.memory.episodic import EpisodicStore
        from core.memory.structured import StructuredMemory
        from core.memory.spatial import SpatialMemory
        journal_dir = dump_path.rsplit(".", 1)[0] + "_memories"
        memory = MemoryManager(
            structured=StructuredMemory(":memory:"),
            episodic=EpisodicStore(persist_directory=journal_dir),
            spatial=SpatialMemory(),
            llm=llm,
        )
        memory.initialise()
        print(f"  memory journal: {journal_dir}/<npc_id>.txt "
              "(live, day-bucketed)", flush=True)

    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=llm, seed=SEED, memory=memory,
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

    # --- Baselines for the criteria diff (captured post-spawn, pre-sim) ---
    baseline_sentiment, _baseline_n = _mean_sentiment_towards(
        mgr.sentiment, objector.npc_id,
    )
    baseline_town = _town_mean_sentiment(mgr.sentiment, npcs, objector.npc_id)
    # The objector's own opposes:repair_bridge is the SEED, not organic — so
    # exclude him and snapshot every other NPC's pre-existing bridge keys (C4).
    baseline_bridge_keys = {
        n.npc_id: set(_bridge_self_concept_keys(n))
        for n in npcs if n.npc_id != objector.npc_id
    }

    daily_log: list[str] = []
    bridge_cycles = 0
    cycles: list[dict] = []   # one finalised record per resolved bridge cycle
    current_goal = None        # held reference — survives _goals overwrite
    seen_bridge_mem_ids: set = set()

    # --- Trajectory instrumentation (daily snapshots) ---
    from core.memory.reflection import reset_tone_tally
    reset_tone_tally()  # run-scoped tone tally
    timeseries: list[dict] = []
    _tone_cumulative: dict[str, int] = {}
    if timeseries_path is None and dump_path is not None:
        timeseries_path = dump_path.replace(".json", "") + "_timeseries.json"

    def _log(line: str) -> None:
        """Append AND stream immediately — so a watcher sees events as they
        happen rather than waiting for end-of-run accumulation."""
        daily_log.append(line)
        print(line, flush=True)

    total_ticks = days * TICKS_PER_DAY
    start = time.time()
    last_reported_day = -1
    last_hb = start

    print(f"Starting {total_ticks} ticks ({days} sim-days). Heartbeat every "
          f"{HEARTBEAT_SECONDS}s; per-tick watchdog {TICK_TIMEOUT_SECONDS}s.",
          flush=True)

    for tick in range(total_ticks):
        clock.tick(TICK_DELTA)
        # Watchdog: a single tick must complete within the budget. If the
        # LLM backend hangs (e.g. a dead Ollama socket after a sleep/wake),
        # abort loudly instead of stalling silently for hours.
        try:
            await asyncio.wait_for(
                mgr.tick(clock, TICK_DELTA), timeout=TICK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            mins = (time.time() - start) / 60
            print(
                f"\n[WATCHDOG] tick {tick} (sim-day {clock.day}) did not "
                f"complete within {TICK_TIMEOUT_SECONDS}s — the LLM backend is "
                f"almost certainly hung. Aborting after {mins:.1f} min "
                f"wall-clock. Re-run under `caffeinate` (prevent sleep) and "
                f"stream to a logfile, not `| tail`.",
                flush=True,
            )
            return

        # Heartbeat: flushed proof-of-life + progress rate, so a stall (or
        # slow-but-alive) is visible immediately rather than after hours.
        now = time.time()
        if now - last_hb >= HEARTBEAT_SECONDS:
            elapsed = now - start
            rate = (tick + 1) / elapsed * 60.0
            eta_m = (total_ticks - tick - 1) / max(rate, 1e-9)
            print(
                f"[hb] tick={tick+1}/{total_ticks} sim-day={clock.day} "
                f"elapsed={elapsed/60:.1f}m rate={rate:.1f} ticks/min "
                f"eta~{eta_m:.0f}m",
                flush=True,
            )
            last_hb = now

        # Once per game-day (first tick of the day after tick 0)
        if clock.day != last_reported_day:
            last_reported_day = clock.day
            day = clock.day
            events_today: list[str] = []

            # Finalise the previous cycle once it has resolved. We read it from
            # the held reference because propose() overwrites _goals by id.
            if current_goal is not None and current_goal.status in (
                GoalStatus.COMPLETED, GoalStatus.EXPIRED,
            ):
                cycles.append({
                    "cycle": len(cycles) + 1,
                    "status": current_goal.status.value,
                    "joined": objector.npc_id in current_goal.contributors,
                    "progress": current_goal.progress,
                    "required": current_goal.required_contributions,
                })
                events_today.append(f"bridge_{current_goal.status.value}")
                current_goal = None

            # Propose bridge goal if none active and cooldown allows.
            active_bridges = [
                g for g in mgr.town_agenda.active_and_proposed()
                if g.goal_id == "repair_bridge"
            ]
            if not active_bridges:
                goal = create_goal_from_template("repair_bridge", day)
                if goal and mgr.town_agenda.propose(goal, day):
                    bridge_cycles += 1
                    current_goal = goal
                    events_today.append(f"bridge_proposed#{bridge_cycles}")
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

            # Daily trajectory snapshot (lightweight metrics over live
            # state) — written incrementally so a run stopped early
            # still leaves a complete time series to here.
            if day > 0 and (day % snapshot_every == 0):
                snap, _tone_cumulative = _snapshot_metrics(
                    mgr, npcs, day, _tone_cumulative, events_today,
                )
                timeseries.append(snap)
                _log(
                    f"[day {day:3d}] SNAPSHOT neg={snap['neg_pct']:.0%} "
                    f"neu={snap['neu_pct']:.0%} pos={snap['pos_pct']:.0%} "
                    f"mean={snap['mean']:+.1f} self={snap['self_keys_mean']:.1f} "
                    f"tone(t/n/w/h)={snap['tone_today']['tense']}/"
                    f"{snap['tone_today']['neutral']}/{snap['tone_today']['warm']}/"
                    f"{snap['tone_today']['hostile']} "
                    f"pariah={snap['most_disliked']}({snap['most_disliked_mean']:+.0f})"
                )
                if timeseries_path:
                    try:
                        with open(timeseries_path, "w") as _f:
                            json.dump({"seed": SEED, "pop": POPULATION,
                                       "days": days, "series": timeseries}, _f,
                                      indent=2)
                    except Exception as _e:
                        print(f"[ts] write failed: {_e}", flush=True)

        # Progress print every N sim days.
        if tick > 0 and tick % (TICKS_PER_DAY * PROGRESS_REPORT_DAYS) == 0:
            elapsed = time.time() - start
            print(
                f"  day {clock.day:3d}  cycles={bridge_cycles}  "
                f"log_lines={len(daily_log)}  elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - start

    # --- Trajectory table (the whole social arc at a glance) ---
    _print_trajectory_table(timeseries)

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

    # Finalise the still-open trailing cycle (if the run ended mid-cycle).
    if current_goal is not None:
        cycles.append({
            "cycle": len(cycles) + 1,
            "status": current_goal.status.value,
            "joined": objector.npc_id in current_goal.contributors,
            "progress": current_goal.progress,
            "required": current_goal.required_contributions,
        })

    # ===================== PRE-REGISTERED CRITERIA VERDICT =====================
    # C1 — voiced dissent (deterministic speaker-attributed scan over
    # ALL of the objector's dialogue memories; no LLM, no sampling cap)
    obj_mems = mgr.memory.episodic.get_recent(
        objector.npc_id, limit=10_000_000,
    )
    c1_count, c1_quotes = _count_voiced_opposition(objector.name, obj_mems)
    c1_pass = c1_count >= 1

    # C2 — indecision calibration + bridge outcome (tuning, not the verdict axis)
    n_cycles = len(cycles)
    n_joined = sum(1 for c in cycles if c["joined"])
    n_completed = sum(1 for c in cycles if c["status"] == "completed")
    n_expired = sum(1 for c in cycles if c["status"] == "expired")
    join_rate = (n_joined / n_cycles) if n_cycles else 0.0
    c2_in_band = JOIN_RATE_BAND[0] <= join_rate <= JOIN_RATE_BAND[1]

    # C3 — social consequence: cooling toward the objector RELATIVE to the
    # town-wide drift (controls for everyone generally warming/cooling).
    final_sentiment, final_n = _mean_sentiment_towards(
        mgr.sentiment, objector.npc_id,
    )
    final_town = _town_mean_sentiment(mgr.sentiment, npcs, objector.npc_id)
    obj_drift = final_sentiment - baseline_sentiment
    town_drift = final_town - baseline_town
    rel_drift = obj_drift - town_drift             # negative = cooled vs town
    c3_cooled = rel_drift <= -SENTIMENT_DRIFT_MIN

    # C4 — organic belief formation/propagation in OTHER NPCs
    organic = []
    for n in npcs:
        if n.npc_id == objector.npc_id:
            continue
        new_keys = (set(_bridge_self_concept_keys(n))
                    - baseline_bridge_keys.get(n.npc_id, set()))
        if new_keys:
            organic.append((n.name, sorted(new_keys)))
    c4_pass = len(organic) >= 1

    # Meta-verdict — pre-committed interpretation. Uses C1 as a validity gate
    # and the C3/C4 axis as the emergence read; C2 is calibration only.
    resolved_cycles = n_completed + n_expired
    if not c1_pass:
        meta = ("NO VOICED DISSENT — the deterministic scan found no dialogue "
                "line where the objector opposes the repair. Before reading "
                "C3/C4, verify the belief survived (self_concept) and the run "
                "wasn't throttle-degraded (canned-line count in the Stage 1.5 "
                "summary). If both check out, this is a real behavioural "
                "finding, not a harness bug.")
    elif resolved_cycles == 0:
        meta = (f"INCONCLUSIVE — the objector voices opposition (C1 ok), but no "
                f"bridge cycle resolved over {n_cycles} proposed, so C3 (social "
                f"consequence of sitting one out) and C4 (propagation) had no "
                f"triggering event. Run longer; this is not an emergence read.")
    elif c3_cooled and c4_pass:
        meta = ("EMERGENCE-RICH — the mechanism layer produced BOTH social "
                "consequence and organic belief formation. The AGENT_DIRECTION "
                "rebuild is NOT indicated by this run.")
    elif (not c3_cooled) and (not c4_pass):
        meta = ("EMERGENCE-THIN — no social consequence AND no organic belief "
                "formation. Reinforces AGENT_DIRECTION's diagnosis. Indicated "
                "next step: privatise sentiment.")
    else:
        meta = ("MIXED / INCONCLUSIVE — exactly one of {social consequence, "
                "organic formation} fired. Discriminating needs the deferred "
                "Traveller-contradictory-claim scenario, or a retune-and-rerun.")

    def _mark(b: bool) -> str:
        return "PASS" if b else "----"

    print("\n" + "=" * 90)
    print("PRE-REGISTERED CRITERIA VERDICT")
    print("=" * 90)
    print(f"C1 voiced dissent  (validity): {_mark(c1_pass):4s}  "
          f"{c1_count} line(s) where objector opposes the repair "
          f"(deterministic speaker-attributed scan)")
    print(f"C2 indecision      (calibr.) : {'IN-BAND' if c2_in_band else 'OUT-BAND'}  "
          f"join_rate={join_rate:.2f} band={JOIN_RATE_BAND[0]:.2f}-{JOIN_RATE_BAND[1]:.2f} "
          f"over {n_cycles} cycle(s); completed={n_completed} expired={n_expired}")
    print(f"C3 social conseq.  (verdict) : {_mark(c3_cooled):4s}  "
          f"objector {baseline_sentiment:+.1f}->{final_sentiment:+.1f} "
          f"(drift {obj_drift:+.1f}) vs town drift {town_drift:+.1f} "
          f"=> relative {rel_drift:+.1f} (threshold -{SENTIMENT_DRIFT_MIN:.1f}, n={final_n})")
    print(f"C4 organic belief  (verdict) : {_mark(c4_pass):4s}  "
          f"{len(organic)} other NPC(s) formed a bridge stance: {organic}")
    print()
    print(f"META-VERDICT: {meta}")
    if c1_quotes:
        print("\nC1 — objector's own opposition lines (sample):")
        for q in c1_quotes[:15]:
            print(f"  {q}")

    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 90)

    # Harvest the run's NPC memories + state for offline review / synopsis
    # (python3 tests/simulation/run_memory.py <path>). Reusable across events.
    if dump_path:
        sys.path.insert(0, str(Path(__file__).parent))
        from run_memory import dump_run_state
        from core.memory.reflection import get_tone_tally
        meta = {
            "event": "repair_bridge", "provider": provider, "days": days,
            "seed": SEED, "population": POPULATION,
            "objector_id": objector.npc_id, "objector_name": objector.name,
            "elapsed_s": round(elapsed),
            "cycles": cycles,
            # Arc-A mechanism evidence: how often NPCs actually judged
            # conversations tense/hostile vs warm/neutral. Lets us read
            # WHY sentiment moved, not just that it did.
            "tone_tally": get_tone_tally(),
            # Full per-day trajectory (also written incrementally to the
            # _timeseries.json sidecar during the run).
            "timeseries": timeseries,
        }
        written = dump_run_state(mgr, npcs, meta, dump_path)
        print(f"[dump] wrote run memories/state -> {written}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Simulated days (default {DEFAULT_DAYS}).",
    )
    parser.add_argument(
        "--provider", choices=("mistral", "gemma", "mock"), default="mistral",
        help="Cognition engine. 'mistral' (default) = fast API path for "
             "harness de-risking; 'gemma' = production engine, slow, for the "
             "confirmatory run; 'mock' = deterministic plumbing smoke.",
    )
    parser.add_argument(
        "--dump", default=None, metavar="PATH",
        help="Write the run's NPC memories + state to PATH (JSON) at the end, "
             "for review/synopsis via run_memory.py.",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=1, metavar="N",
        help="Take a lightweight metrics snapshot every N sim-days "
             "(default 1 = daily). Builds the sentiment trajectory time "
             "series + end-of-run table.",
    )
    parser.add_argument(
        "--timeseries", default=None, metavar="PATH",
        help="Write the per-day trajectory to PATH (JSON), incrementally. "
             "Defaults to <dump>_timeseries.json when --dump is given.",
    )
    args = parser.parse_args()
    asyncio.run(run(days=args.days, provider=args.provider,
                    dump_path=args.dump,
                    snapshot_every=args.snapshot_every,
                    timeseries_path=args.timeseries))
