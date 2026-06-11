"""Persona-conditioning audit — the failure-mode eval for the
vectorization foundation.

Runs a deterministic MockProvider sim with every NPC pinned to tier 1
(so all LLM cognition paths fire), then audits the COMPLETE LLM
traffic post-hoc. The unit tests in tests/unit/test_persona_*.py pin
each call site in isolation; this eval catches what they can't — a
new or missed call site shipping with the old shared generic system
prompt, persona loss in the live spawn path, or the persona signal
being drowned at the whole-prompt level.

Checks:
  spawn integrity     every NPC has a persona; speech styles and
                      temperaments unique across the town    target 100%
  conditioned rate    audited-purpose calls whose system prompt
                      carries the calling NPC's character sheet
                                                             target 100%
  generic leakage     audited-purpose calls still using a "medieval
                      NPC"-style shared system string         target 0
  unattributable      audited calls whose system identifies no
                      spawned NPC                             target 0
  dominance share     persona block chars / whole prompt chars on
                      conversation calls (the "drowned" measure;
                      informational, printed for trend)
  purpose coverage    audited purposes that produced zero traffic
                      (eval blind spots; warning, not failure)

Run:
  python3 tests/simulation/eval_persona_conditioning.py
  python3 tests/simulation/eval_persona_conditioning.py --pop 10 --days 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Determinism: pin string-hash seed (set iteration order affects
# encounter order) before any heavy imports — same trick as
# eval_foundation.py.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world

# LLM purposes spoken IN THE NPC'S VOICE — these must carry the
# caller's character sheet. Clerk purposes (fact extraction, action
# classification, importance scoring) are deliberately absent.
AUDITED_PURPOSES = {
    "conversation", "reflection", "daily_plan", "reaction",
    "day_summary", "week_summary", "self_review",
}

SHEET_MARKER = "This is your character sheet"
GENERIC_MARKER = "medieval NPC"

# Clerk calls that share an audited purpose but are deliberately NOT
# spoken in the NPC's voice (they process the NPC's data from outside).
# Explicit whitelist: a future NPC-voiced call site that ships without
# conditioning will NOT match and will still fail the audit.
CLERK_SYSTEM_PREFIXES = (
    "You extract factual information from NPC conversations.",
    "You decide which facts an NPC should remember verbatim.",
    "You classify NPC reflections as actionable or not.",
)


def _make_sim(pop: int, seed: int):
    config = WorldConfig(population=pop, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(), llm=llm,
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=seed, memory=memory,
    )
    mgr.spawn_population(pop)
    return mgr, llm, GameClock()


def _pin_tier_one(mgr) -> None:
    """Force every NPC onto the full-LLM path for the whole run, so
    the audit sees traffic from every cognition purpose."""
    from core.npc import manager as _mgrmod

    def _pinned(npcs, focus_x, focus_z):
        for n in npcs:
            n.cognition_tier = 1
        return {1: [n.npc_id for n in npcs], 2: [], 3: [], 4: []}

    _mgrmod.update_all_tiers = _pinned
    for n in mgr.npcs:
        n.cognition_tier = 1


def _audit_call(call: dict, npcs_by_name: dict) -> tuple[str, str]:
    """Classify one audited-purpose call.

    Returns (status, detail) where status is one of:
    conditioned / generic / unattributable / sheet_mismatch.
    """
    system = call.get("system") or ""
    if system.startswith(CLERK_SYSTEM_PREFIXES):
        return "clerk", system[:50]
    if GENERIC_MARKER in system:
        return "generic", system[:80]
    speaker = None
    for name, npc in npcs_by_name.items():
        if system.startswith(f"You are {name},"):
            speaker = npc
            break
    if speaker is None:
        return "unattributable", system[:80]
    if (
        SHEET_MARKER in system
        and speaker.persona is not None
        and speaker.persona.speech_style in system
    ):
        return "conditioned", speaker.name
    return "sheet_mismatch", f"{speaker.name}: {system[:80]}"


def _dominance_share(call: dict, npcs_by_name: dict) -> float | None:
    """Persona block chars / whole prompt chars for one call."""
    system = call.get("system") or ""
    for name, npc in npcs_by_name.items():
        if system.startswith(f"You are {name},") and npc.persona:
            block = npc.persona.to_prompt_block(name)
            user = "".join(
                m.get("content", "") for m in call.get("messages", [])
            )
            total = len(system) + len(user)
            return len(block) / total if total else None
    return None


async def run_eval(pop: int, days: int, seed: int) -> int:
    mgr, llm, clock = _make_sim(pop, seed)
    _pin_tier_one(mgr)
    npcs_by_name = {n.name: n for n in mgr.npcs}

    # --- 1. Spawn integrity ---
    missing = [n.npc_id for n in mgr.npcs if n.persona is None]
    styles = [n.persona.speech_style for n in mgr.npcs if n.persona]
    temps = [n.persona.temperament for n in mgr.npcs if n.persona]
    style_dupes = len(styles) - len(set(styles))
    temp_dupes = len(temps) - len(set(temps))
    serialise_ok = all(
        isinstance(n.to_full_dict().get("persona"), dict)
        for n in mgr.npcs
    )

    # --- 2. Drive the sim ---
    real_delta = 8.0
    gm_per_tick = 9.6
    ticks = int(days * MINUTES_PER_DAY / gm_per_tick) + 1
    start = time.time()
    for _ in range(ticks):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)
    elapsed = time.time() - start

    # --- 3. Audit the traffic ---
    audited = [
        c for c in llm.call_log if c.get("purpose") in AUDITED_PURPOSES
    ]
    by_purpose: dict[str, int] = {}
    statuses: dict[str, list[str]] = {
        "conditioned": [], "clerk": [], "generic": [],
        "unattributable": [], "sheet_mismatch": [],
    }
    shares: list[float] = []
    for call in audited:
        by_purpose[call["purpose"]] = by_purpose.get(call["purpose"], 0) + 1
        status, detail = _audit_call(call, npcs_by_name)
        statuses[status].append(f"[{call['purpose']}] {detail}")
        if call["purpose"] == "conversation":
            share = _dominance_share(call, npcs_by_name)
            if share is not None:
                shares.append(share)

    silent = sorted(AUDITED_PURPOSES - set(by_purpose))
    conditioned = len(statuses["conditioned"])
    voiced = len(audited) - len(statuses["clerk"])
    rate = conditioned / voiced if voiced else 0.0

    # --- 4. Dashboard ---
    print(f"\n=== Persona-conditioning audit "
          f"(pop={pop}, days={days}, seed={seed}) ===")
    print(f"sim wall-clock: {elapsed:.1f}s, "
          f"LLM calls total: {len(llm.call_log)}, audited: {len(audited)}")
    print("\n-- spawn integrity --")
    print(f"personas missing:        {len(missing)} {missing or ''}")
    print(f"duplicate speech styles: {style_dupes}")
    print(f"duplicate temperaments:  {temp_dupes}")
    print(f"serialisation intact:    {serialise_ok}")
    print("\n-- traffic by purpose --")
    for purpose in sorted(by_purpose):
        print(f"{purpose:14s} {by_purpose[purpose]}")
    if silent:
        print(f"WARNING — no traffic for: {', '.join(silent)} "
              "(audit blind spot, raise --days or check routing)")
    print("\n-- conditioning --")
    print(f"conditioned:    {conditioned}/{voiced} NPC-voiced ({rate:.1%}); "
          f"{len(statuses['clerk'])} clerk calls excluded")
    for kind in ("generic", "unattributable", "sheet_mismatch"):
        rows = statuses[kind]
        print(f"{kind + ':':16s}{len(rows)}")
        for row in rows[:5]:
            print(f"    {row}")
    if shares:
        print(f"\npersona dominance share on conversation calls: "
              f"mean {sum(shares) / len(shares):.1%}, "
              f"min {min(shares):.1%}")

    # --- 5. Verdict ---
    failures = []
    if missing:
        failures.append("NPCs spawned without personas")
    if style_dupes or temp_dupes:
        failures.append("duplicate personas within town")
    if not serialise_ok:
        failures.append("persona lost in to_full_dict")
    if not audited:
        failures.append("no audited LLM traffic at all — eval did not run")
    if statuses["generic"]:
        failures.append("generic shared system prompts still in use")
    if statuses["unattributable"] or statuses["sheet_mismatch"]:
        failures.append("calls without the caller's character sheet")

    print("\n=== CRITERIA VERDICT ===")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: every audited cognition call carries its caller's "
          "character sheet; personas unique, persistent, serialised.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()
    return asyncio.run(run_eval(args.pop, args.days, args.seed))


if __name__ == "__main__":
    raise SystemExit(main())
