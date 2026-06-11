"""NPC individuality / vectorization metrics (measurement suite — Layer 1).

Complements the live `npc_metrics.NPCMetricsTracker` (activity/needs/
life-balance) with an *offline, cognitive* read from a run dump
(tests/simulation/run_memory.py output).

The concern (Stanford baseline): each NPC should read as a DISTINCT
individual whose vectorized self (personality, self-concept, goals)
drives behaviour. The worry is that the unique self is drowned in
high-volume, near-duplicate conversation/observation noise — homogenising
the population. This quantifies that and tries to tell *one bug* (e.g.
pure conversation volume) from *systemic* (weak self-formation AND uniform
sentiment AND homogeneous memory AND volume).

CLI:  python3 tests/simulation/npc_individuality.py <dump.json>
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
from collections import Counter
from itertools import combinations

# Memory categories that represent the NPC's distinctive inner life.
SIGNAL_CATS = {
    "reflection", "note", "identity", "commitment", "commitment_review",
    "motivation", "aspiration", "concern", "day_summary", "week_summary",
    "relationship", "knowledge", "relayed_claim", "accusation",
}
CONVO_CATS = {"conversation", "conversation_turn", "conversation_fact"}
OBSERVE_CATS = {"npc", "object", "observation"}


def _self_keys(npc: dict) -> set:
    return set(npc["self_concept"].keys())


def _near_dup_rate(npc: dict) -> float:
    if not npc["memories"]:
        return 0.0
    sig = Counter((m["category"], m["description"][:45]) for m in npc["memories"])
    dups = sum(c for c in sig.values() if c >= 3)
    return dups / len(npc["memories"])


def measure(path: str) -> dict:
    data = json.loads(pathlib.Path(path).read_text())
    npcs = data["npcs"]
    n = len(npcs)
    line = "=" * 84
    print(line)
    print(f"NPC INDIVIDUALITY / VECTORIZATION  (Layer 1)  "
          f"event={data['meta'].get('event')} provider={data['meta'].get('provider')} "
          f"days={data['meta'].get('days')} pop={n}")
    print(line)

    # ---- 1. Signal vs volume ----
    print("\n1. SIGNAL (the vectorized self) vs VOLUME (conversation/observation)")
    sig_ratios = []
    for npc in npcs:
        c = Counter(m["category"] for m in npc["memories"])
        total = sum(c.values()) or 1
        sig = sum(v for k, v in c.items() if k in SIGNAL_CATS)
        convo = sum(v for k, v in c.items() if k in CONVO_CATS)
        obs = sum(v for k, v in c.items() if k in OBSERVE_CATS)
        sig_ratios.append(sig / total)
        print(f"  {npc['name']:12s} mem={total:5d} "
              f"signal={sig:4d}({sig/total:5.1%}) convo={convo:5d} obs={obs:4d}")
    pop_sig = statistics.mean(sig_ratios)
    print(f"  -> town signal ratio: mean {pop_sig:.1%}  "
          f"(the distinctive self is ~{pop_sig:.0%} of recorded memory)")

    # ---- 2. Self-concept vectorization ----
    print("\n2. SELF-CONCEPT vectorization (distinct identities?)")
    key_counts = [len(_self_keys(x)) for x in npcs]
    empties = sum(1 for k in key_counts if k == 0)
    all_keys = Counter(k for x in npcs for k in _self_keys(x))
    shared = [k for k, c in all_keys.items() if c >= max(2, n // 2)]
    sets = [_self_keys(x) for x in npcs]
    jac = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
    mean_jac = statistics.mean(jac) if jac else 0.0
    print(f"  self-concept keys per NPC: mean {statistics.mean(key_counts):.1f}, "
          f"min {min(key_counts)}, max {max(key_counts)}; "
          f"EMPTY self-concepts: {empties}/{n}")
    print(f"  keys held by >=half the town: {shared}")
    print(f"  mean pairwise self-concept overlap (Jaccard): {mean_jac:.2f}  "
          f"(1.0 = identical selves, 0 = fully distinct)")

    # ---- 3. Sentiment differentiation ----
    print("\n3. SENTIMENT differentiation (individuated relationships?)")
    disp = [s["disposition"] for x in npcs for s in x["sentiments"]]
    sdev = statistics.pstdev(disp) if disp else 0.0
    if disp:
        neg = sum(1 for d in disp if d < -5) / len(disp)
        neu = sum(1 for d in disp if -5 <= d <= 5) / len(disp)
        pos = sum(1 for d in disp if d > 5) / len(disp)
        print(f"  {len(disp)} relationships: mean {statistics.mean(disp):+.1f}, "
              f"stdev {sdev:.1f}")
        print(f"  distribution: negative {neg:.0%} | neutral {neu:.0%} | "
              f"positive {pos:.0%}")
    else:
        print("  (no sentiment recorded)")

    # ---- 4. Memory homogeneity ----
    print("\n4. MEMORY homogeneity (near-duplicate churn)")
    dup_rates = [_near_dup_rate(x) for x in npcs]
    print(f"  near-duplicate memory rate: mean {statistics.mean(dup_rates):.0%}, "
          f"max {max(dup_rates):.0%}")

    # ---- 5. Behavioural diversity ----
    occ = Counter(x["occupation"] for x in npcs)
    goals = Counter(g for x in npcs for g in x.get("long_term_goals", []))
    print("\n5. BEHAVIOURAL diversity")
    print(f"  occupations: {dict(occ)}; distinct long-term goals: {len(goals)}")

    # ---- Verdict: localise the homogenisation ----
    print("\nVERDICT — sources of homogenisation")
    sources = []
    if pop_sig < 0.10:
        sources.append(f"VOLUME DROWNING: only {pop_sig:.0%} of memory is the "
                       f"distinctive self")
    if empties >= n / 3 or statistics.mean(key_counts) < 3:
        sources.append(f"WEAK SELF-FORMATION: {empties}/{n} empty self-concepts, "
                       f"mean {statistics.mean(key_counts):.1f} keys/NPC")
    if mean_jac > 0.3:
        sources.append(f"SHARED IDENTITIES: self-concept overlap {mean_jac:.2f}")
    if disp and sdev < 25 and statistics.mean(disp) > 25:
        sources.append("UNIFORM SENTIMENT: town ~equally warm; not individuated")
    if statistics.mean(dup_rates) > 0.15:
        sources.append(f"CHURN: {statistics.mean(dup_rates):.0%} near-duplicate "
                       f"memories")
    for s in sources:
        print(f"  ! {s}")
    if sources:
        kind = ("SYSTEMIC (multiple independent sources)" if len(sources) >= 3
                else "MULTI-FACTOR" if len(sources) == 2 else "LOCALISED")
        print(f"  => {len(sources)} source(s) -> {kind}")
    else:
        print("  No strong homogenisation signal by current thresholds.")

    print(line)
    return {
        "signal_ratio": round(pop_sig, 3),
        "self_keys_mean": round(statistics.mean(key_counts), 2),
        "empty_selves": empties,
        "self_overlap": round(mean_jac, 2),
        "sentiment_stdev": round(sdev, 1),
        "dup_rate_mean": round(statistics.mean(dup_rates), 3),
        "homogenisation_sources": len(sources),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 tests/simulation/npc_individuality.py <dump.json>")
        sys.exit(1)
    measure(sys.argv[1])
