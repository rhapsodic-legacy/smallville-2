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
C1_JUDGE_MAX = 15               # C1: cap excerpts sent to the LLM judge per run.


def _bridge_memories(memory, npc_id: str) -> list:
    """All non-compacted memories for this NPC mentioning the bridge."""
    mems = memory.episodic.get_recent(npc_id, limit=200)
    return [m for m in mems if "bridge" in m.description.lower()]


async def _judge_voiced_opposition(llm, npc_name: str, memories: list):
    """C1 (LLM judge) — among the objector's bridge conversations, count those
    where {npc_name} HIMSELF voices reluctance, refusal, or opposition to
    repairing the bridge.

    Judges speaker-attributed intent rather than keywords, so pro-repair idioms
    like "the bridge won't fix itself" (which a token scan flags as opposition)
    no longer false-positive. Returns ``(count, quotes)``; ``count == -1``
    signals the judge call itself failed (validity unknown, not a real zero)."""
    convos = [
        m for m in memories
        if getattr(m, "category", "") == "conversation"
        and "bridge" in m.description.lower()
    ][:C1_JUDGE_MAX]
    if not convos:
        return 0, []

    listing = "\n\n".join(
        f"[{i}] {m.description[:400]}" for i, m in enumerate(convos)
    )
    prompt = (
        f"Below are excerpts of conversations involving {npc_name}. Each line "
        f"is prefixed by the speaker's name.\n\n"
        f"For EACH excerpt, judge ONLY what {npc_name} himself says (lines "
        f"prefixed '{npc_name}:'). Does {npc_name} express reluctance, refusal, "
        f"or opposition to REPAIRING THE BRIDGE?\n"
        f"IMPORTANT: saying the bridge needs fixing, offering to help, or "
        f"phrases like 'the bridge won't fix itself' are SUPPORT, not "
        f"opposition.\n\n"
        f"{listing}\n\n"
        f"Reply with one line per excerpt where {npc_name} genuinely opposes "
        f"the repair, formatted 'OPPOSE [n]: <his own words>'. If none, reply "
        f"exactly 'NONE'."
    )
    try:
        resp = await llm.complete(
            system="You are a precise dialogue annotator.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.0, purpose="conversation",
        )
    except Exception as exc:  # judge unavailable — don't fake a zero
        return -1, [f"judge call failed: {exc}"]

    quotes = [
        ln.strip() for ln in resp.splitlines()
        if ln.strip().upper().startswith("OPPOSE")
    ]
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
    print(f"ERROR: unknown provider {provider!r} (use 'mistral' or 'gemma').")
    sys.exit(1)


async def run(days: int = DEFAULT_DAYS, provider: str = "mistral",
              dump_path: str | None = None) -> None:
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

    mgr = NPCManager(
        grid=grid, buildings=buildings,
        llm=llm, seed=SEED,
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
    # C1 — voiced dissent (LLM judge; harness-validity gate)
    obj_mems = mgr.memory.episodic.get_recent(objector.npc_id, limit=400)
    c1_count, c1_quotes = await _judge_voiced_opposition(
        llm, objector.name, obj_mems,
    )
    c1_judge_failed = c1_count < 0
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
    if c1_judge_failed:
        meta = ("UNSCORED — the C1 opposition judge could not run, so dialogue "
                "validity is unknown. Re-run; do NOT read C3/C4 as architecture "
                "evidence without a working validity gate.")
    elif not c1_pass:
        meta = ("INVALID — the opposition belief never surfaced in dialogue. "
                "Treat as a wiring bug (self_concept -> prompt path), not as "
                "evidence about emergence.")
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
    c1_label = "FAIL?" if c1_judge_failed else _mark(c1_pass)
    c1_n = "?" if c1_judge_failed else str(c1_count)
    print(f"C1 voiced dissent  (validity): {c1_label:4s}  "
          f"{c1_n} line(s) where objector opposes the repair (LLM-judged)")
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
        title = ("C1 judge could not run" if c1_judge_failed
                 else "C1 judge output (objector's own opposition lines)")
        print(f"\n{title}:")
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
        "--provider", choices=("mistral", "gemma"), default="mistral",
        help="Cognition engine. 'mistral' (default) = fast API path for "
             "harness de-risking; 'gemma' = production engine, slow, for the "
             "confirmatory run.",
    )
    parser.add_argument(
        "--dump", default=None, metavar="PATH",
        help="Write the run's NPC memories + state to PATH (JSON) at the end, "
             "for review/synopsis via run_memory.py.",
    )
    args = parser.parse_args()
    asyncio.run(run(days=args.days, provider=args.provider,
                    dump_path=args.dump))
