"""Harvest, review, and synopsise a simulation run's NPC memories.

Two halves:
  dump_run_state(mgr, npcs, meta, path)  -> write the full per-NPC state
        (episodic memories, self-concept, sentiments, commitments, goals)
        plus run metadata to a JSON file. Call at the end of a diagnostic.
  synopsise(path)                        -> read the JSON and print a
        review: town-level behaviour, the response to the event, the
        objector's arc, a per-NPC digest, automated bug/outlier flags,
        and a compact comparison block for stacking runs side by side.

Purpose (exploratory): (1) surface clearly-buggy behaviour, (2) draw
conclusions about how the town acts overall and relative to an injected
event — so that across events/runs a longer comparative picture forms.

CLI:  python3 tests/simulation/run_memory.py <dump.json>
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #
def dump_run_state(mgr, npcs, meta: dict, path: str) -> str:
    """Serialise every NPC's memories + state and the run metadata to JSON."""
    id_to_name = {n.npc_id: n.name for n in npcs}
    out = {"meta": meta, "npcs": []}
    for n in npcs:
        # No cap. limit=5000 silently discarded the first ~15 days of a
        # 30-day run (newest-first) — the store's "full picture" promise
        # has to hold at the dump too. Text is cheap; keep everything.
        try:
            mems = mgr.memory.episodic.get_recent(
                n.npc_id, limit=10_000_000, include_compacted=True,
            )
        except TypeError:
            mems = mgr.memory.episodic.get_recent(n.npc_id, limit=10_000_000)
        sentiments = []
        try:
            for s in mgr.sentiment.get_all_for(n.npc_id):
                sentiments.append({
                    "to": s.npc_to,
                    "to_name": id_to_name.get(s.npc_to, s.npc_to),
                    "disposition": round(s.overall_disposition(), 1),
                    "desc": s.to_description(),
                })
        except Exception:
            pass
        out["npcs"].append({
            "npc_id": n.npc_id,
            "name": n.name,
            "occupation": n.occupation,
            "personality": (n.personality.to_dict()
                            if hasattr(n.personality, "to_dict") else {}),
            "self_concept": dict(n.self_concept),
            "long_term_goals": list(getattr(n, "long_term_goals", [])),
            "commitments": [{
                "goal_id": c.goal_id,
                "status": getattr(c.status, "value", str(c.status)),
                "activity": c.activity,
                "deadline_day": c.deadline_day,
            } for c in getattr(n, "commitments", [])],
            "sentiments": sentiments,
            "n_memories": len(mems),
            "memories": [{
                "description": m.description,
                "category": getattr(m, "category", ""),
                "importance": round(float(getattr(m, "importance", 0.0)), 3),
                "game_time": getattr(m, "game_time", 0),
                "tags": sorted(getattr(m, "tags", []) or []),
                # Canned-line provenance (only present when set).
                **({"fallback": True}
                   if (getattr(m, "metadata", None) or {}).get("fallback")
                   else {}),
                **({"fallback_turns":
                    (getattr(m, "metadata", None) or {})["fallback_turns"]}
                   if (getattr(m, "metadata", None) or {}).get("fallback_turns")
                   else {}),
            } for m in mems],
        })
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    return str(p)


# --------------------------------------------------------------------------- #
# Synopsis
# --------------------------------------------------------------------------- #
_OPPOSE_TOKENS = ("oppose", "opposed", "won't", "wont", "refuse", "fool's",
                  "waste", "death-trap", "death trap", "no good", "against",
                  "rotten", "won't be", "not be lendin")
_SUPPORT_TOKENS = ("help repair", "will help", "lend a hand", "agreed to",
                   "must repair", "needs repair", "support", "fix the")


def _stance(npc: dict, event_kw: str) -> str:
    sc = npc["self_concept"]
    if any(k.startswith("opposes:") and event_kw in k for k in sc):
        return "opposes (self-concept)"
    if any(k.startswith(("built:", "helped:", "supports:")) and event_kw in k
           for k in sc):
        return "backed (self-concept)"
    own = " ".join(
        m["description"].lower() for m in npc["memories"]
        if m["category"] == "conversation" and event_kw in m["description"].lower()
    )
    if not own:
        return "no recorded stance"
    opp = sum(t in own for t in _OPPOSE_TOKENS)
    sup = sum(t in own for t in _SUPPORT_TOKENS)
    if opp > sup:
        return "opposes (dialogue)"
    if sup > opp:
        return "backs (dialogue)"
    return "engaged, neutral"


def _dup_count(npc: dict) -> int:
    """Near-duplicate memories: same category + first 45 chars seen 3+ times."""
    sig = Counter((m["category"], m["description"][:45]) for m in npc["memories"])
    return sum(c for c in sig.values() if c >= 3)


def synopsise(path: str) -> None:
    data = json.loads(pathlib.Path(path).read_text())
    meta, npcs = data["meta"], data["npcs"]
    event = meta.get("event", "repair_bridge")
    event_kw = event.split("_")[-1] if "_" in event else event  # e.g. "bridge"
    objector_id = meta.get("objector_id")

    line = "=" * 84
    print(line)
    print(f"RUN SYNOPSIS — event={event}  provider={meta.get('provider')}  "
          f"days={meta.get('days')}  pop={len(npcs)}  seed={meta.get('seed')}")
    if "cycles" in meta:
        statuses = Counter(c.get("status") for c in meta["cycles"])
        print(f"  event cycles: {dict(statuses)}  "
              f"(elapsed {meta.get('elapsed_s', '?')}s)")
    print(line)

    # ---- Town-level ----
    total_mem = sum(n["n_memories"] for n in npcs)
    cat = Counter()
    for n in npcs:
        cat.update(m["category"] for m in n["memories"])
    occ = Counter(n["occupation"] for n in npcs)
    all_disp = [s["disposition"] for n in npcs for s in n["sentiments"]]
    avg_disp = sum(all_disp) / len(all_disp) if all_disp else 0.0
    print("\nTOWN")
    print(f"  occupations: {dict(occ)}")
    print(f"  memories: {total_mem} total, {total_mem/len(npcs):.0f}/NPC; "
          f"by category: {dict(cat.most_common())}")
    print(f"  sentiment landscape: {len(all_disp)} relationships, "
          f"mean disposition {avg_disp:+.1f}")

    # ---- Response to the event ----
    engaged = [n for n in npcs
               if any(event_kw in m["description"].lower() for m in n["memories"])]
    backers = [n for n in npcs
               if any(k.startswith(("built:", "helped:")) and event_kw in k
                      for k in n["self_concept"])]
    opposers = [n for n in npcs
                if any(k.startswith("opposes:") and event_kw in k
                       for k in n["self_concept"])]
    print(f"\nEVENT RESPONSE ({event})")
    print(f"  engaged (have {event_kw} memories): {len(engaged)}/{len(npcs)}")
    print(f"  came away identifying as backers (built/helped): {len(backers)} "
          f"-> {[n['name'] for n in backers]}")
    print(f"  hold an opposition self-concept: {len(opposers)} "
          f"-> {[n['name'] for n in opposers]}")

    # ---- Objector arc ----
    obj = next((n for n in npcs if n["npc_id"] == objector_id), None)
    if obj:
        toward = []  # how others feel about the objector
        for n in npcs:
            for s in n["sentiments"]:
                if s["to"] == objector_id:
                    toward.append(s["disposition"])
        opp_lines = [m["description"][:120] for m in obj["memories"]
                     if m["category"] == "conversation"
                     and event_kw in m["description"].lower()
                     and any(t in m["description"].lower() for t in _OPPOSE_TOKENS)]
        print(f"\nOBJECTOR — {obj['name']} ({obj['occupation']})")
        print(f"  stance: {_stance(obj, event_kw)}; "
              f"self-concept: { {k: v for k, v in obj['self_concept'].items() if event_kw in k} }")
        print(f"  town feeling toward them: mean "
              f"{(sum(toward)/len(toward) if toward else 0):+.1f} (n={len(toward)})")
        print(f"  opposition lines recorded: {len(opp_lines)}")
        for l in opp_lines[:3]:
            print(f"    - {l!r}")

    # ---- Per-NPC digest ----
    print("\nPER-NPC")
    for n in sorted(npcs, key=lambda x: -x["n_memories"]):
        top_sc = sorted(n["self_concept"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {n['name']:14s} {n['occupation']:14s} mem={n['n_memories']:4d} "
              f"stance={_stance(n, event_kw):22s} "
              f"self={[f'{k}={v:.1f}' for k, v in top_sc]}")

    # ---- Automated bug / outlier flags ----
    print("\nBUG / OUTLIER FLAGS")
    flags = []
    for n in npcs:
        if n["n_memories"] == 0:
            flags.append(f"{n['name']}: ZERO memories (frozen / never perceived?)")
        sc = n["self_concept"]
        opposes_evt = any(k.startswith("opposes:") and event_kw in k for k in sc)
        backed_evt = any(k.startswith(("built:", "helped:")) and event_kw in k for k in sc)
        if opposes_evt and backed_evt:
            flags.append(f"{n['name']}: CONTRADICTION — opposes AND backed {event_kw}")
        live = [c for c in n["commitments"]
                if c["status"] in ("pending", "active")]
        if live:
            flags.append(f"{n['name']}: {len(live)} commitment(s) never resolved "
                         f"({[c['goal_id'] for c in live]})")
        dups = _dup_count(n)
        if dups >= 6:
            flags.append(f"{n['name']}: {dups} near-duplicate memories "
                         f"(conversation churn?)")
        bad_disp = [s for s in n["sentiments"] if abs(s["disposition"]) > 100.5]
        if bad_disp:
            flags.append(f"{n['name']}: disposition out of [-100,100] range")
    if flags:
        for f in flags:
            print(f"  ! {f}")
    else:
        print("  (none detected by current heuristics)")

    # ---- Comparison block (stack across runs/events) ----
    print("\nCOMPARISON BLOCK (for cross-run/event diffing)")
    comp = {
        "event": event,
        "provider": meta.get("provider"),
        "days": meta.get("days"),
        "pop": len(npcs),
        "event_cycles": dict(Counter(c.get("status") for c in meta.get("cycles", []))),
        "backers": len(backers),
        "opposers": len(opposers),
        "engaged_frac": round(len(engaged) / len(npcs), 2),
        "mem_per_npc": round(total_mem / len(npcs), 1),
        "mean_disposition": round(avg_disp, 1),
        "flags": len(flags),
    }
    print("  " + json.dumps(comp))
    print(line)


# --------------------------------------------------------------------------- #
# Stage 1.5 — thorough, file-written run summary (for run-to-run comparison)
# --------------------------------------------------------------------------- #
def _top_repeated(memories: list[dict], top: int = 8) -> list[tuple[str, int]]:
    """Most-repeated memory lines (category + first 60 chars), count >= 2.
    Surfaces the repetitive-thinking / inane-observation behaviour
    (e.g. "Sam was talking near me" recorded 30+ times)."""
    sig = Counter(
        f"[{m['category']}] {m['description'][:60]}" for m in memories
    )
    return [(text, c) for text, c in sig.most_common(top) if c >= 2]


def summarise_run(path: str, out_path: str | None = None) -> str:
    """Write a thorough, human-readable summary of a run to a text file
    so it (and a context-wiped future reader) can compare one run to a
    previous one without re-reading the raw dump. Returns the path
    written. Descriptive only — what the NPCs thought, how much, what
    they repeated, where they ended up. No scored metrics."""
    data = json.loads(pathlib.Path(path).read_text())
    meta, npcs = data["meta"], data["npcs"]
    if out_path is None:
        out_path = path.rsplit(".", 1)[0] + "_summary.txt"

    L: list[str] = []
    def emit(s: str = "") -> None:
        L.append(s)

    bar = "=" * 84
    emit(bar)
    emit(f"RUN SUMMARY — event={meta.get('event')} provider={meta.get('provider')} "
         f"days={meta.get('days')} pop={len(npcs)} seed={meta.get('seed')}")
    if "cycles" in meta:
        emit(f"  event cycles: {dict(Counter(c.get('status') for c in meta['cycles']))}")
    emit(bar)

    # ---- Town roll-up ----
    total_mem = sum(n["n_memories"] for n in npcs)
    cat = Counter()
    for n in npcs:
        cat.update(m["category"] for m in n["memories"])
    disp = [s["disposition"] for n in npcs for s in n["sentiments"]]
    # Canned-line provenance: fallback-flagged turns/reflections plus
    # fallback_turns counts folded into consolidated conversations.
    canned = sum(
        1 for n in npcs for m in n["memories"] if m.get("fallback")
    ) + sum(
        m.get("fallback_turns", 0) for n in npcs for m in n["memories"]
    )
    dialogue_mems = sum(
        1 for n in npcs for m in n["memories"]
        if m["category"] in ("conversation_turn", "conversation")
    )
    emit("\nTOWN")
    emit(f"  memories: {total_mem} total, {total_mem/max(len(npcs),1):.0f}/NPC")
    emit(f"  by category: {dict(cat.most_common())}")
    emit(f"  canned/fallback lines: {canned}"
         + (f" (~{canned/dialogue_mems:.0%} of {dialogue_mems} dialogue "
            f"memories) — run-validity canary" if dialogue_mems else ""))
    if disp:
        emit(f"  relationships: {len(disp)}, mean disposition "
             f"{sum(disp)/len(disp):+.1f}, "
             f"range [{min(disp):+.0f}, {max(disp):+.0f}]")

    # ---- Repetition / spam (town-wide) ----
    all_mem = [m for n in npcs for m in n["memories"]]
    rep = _top_repeated(all_mem, top=12)
    emit("\nMOST-REPEATED THOUGHTS (town-wide — repetition / inane-observation check)")
    if rep:
        for text, c in rep:
            emit(f"  x{c:<4d} {text!r}")
    else:
        emit("  (no line repeated)")

    # ---- Per-NPC digest ----
    emit("\nPER-NPC")
    for n in sorted(npcs, key=lambda x: -x["n_memories"]):
        ncat = Counter(m["category"] for m in n["memories"])
        top_sc = sorted(n["self_concept"].items(), key=lambda kv: -kv[1])[:5]
        sents = sorted(n["sentiments"], key=lambda s: s["disposition"])
        liked = [s for s in sents if s["disposition"] > 5][-3:]
        disliked = [s for s in sents if s["disposition"] < -5][:3]
        emit(f"\n  {n['name']} ({n['occupation']}) — {n['n_memories']} memories "
             f"{dict(ncat.most_common())}")
        emit(f"    self-concept: {[f'{k}={v:.1f}' for k, v in top_sc] or '(none)'}")
        if liked:
            emit(f"    likes: {[(s['to_name'], round(s['disposition'])) for s in reversed(liked)]}")
        if disliked:
            emit(f"    dislikes: {[(s['to_name'], round(s['disposition'])) for s in disliked]}")
        # The NPC's own day-by-day recollections (already LLM-generated
        # in-sim with their guardrails) — the thread of what they thought.
        recolls = sorted(
            (m for m in n["memories"]
             if m["category"] in ("day_summary", "week_summary")),
            key=lambda m: m.get("game_time", 0),
        )
        if recolls:
            emit("    recollections:")
            for m in recolls:
                emit(f"      d{int(m.get('game_time',0)//1440)}: {m['description']}")
        npc_rep = _top_repeated(n["memories"], top=3)
        if npc_rep:
            emit(f"    repeated: {[f'x{c} {t[:40]}' for t, c in npc_rep]}")

    # ---- Compact comparison line (diff two runs at a glance) ----
    emit("\nCOMPARISON LINE")
    emit("  " + json.dumps({
        "event": meta.get("event"), "provider": meta.get("provider"),
        "days": meta.get("days"), "pop": len(npcs),
        "mem_per_npc": round(total_mem / max(len(npcs), 1), 1),
        "mean_disposition": round(sum(disp)/len(disp), 1) if disp else 0,
        "zero_memory_npcs": sum(1 for n in npcs if n["n_memories"] == 0),
        "town_repeats": len(rep),
        "canned_lines": canned,
    }))
    emit(bar)

    text = "\n".join(L)
    pathlib.Path(out_path).write_text(text)
    print(text)
    print(f"\n[summary] written -> {out_path}")
    return out_path


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "summary":
        summarise_run(args[1])
    elif len(args) == 1:
        synopsise(args[0])
    else:
        print("usage: python3 tests/simulation/run_memory.py <dump.json>")
        print("       python3 tests/simulation/run_memory.py summary <dump.json>")
        sys.exit(1)
