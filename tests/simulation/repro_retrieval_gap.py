"""Fast, LLM-free reproduction harness for the episodic RETRIEVAL GAP.

The 30-day run showed 3/10 NPCs returning 0 from `get_recent` despite
heavy activity (VECTORIZATION_ROADMAP.md scale bug). A synthetic
dummy-embedding ChromaDB test at 53k docs did NOT reproduce it, so this
exercises the REAL `EpisodicStore` (real ONNX embeddings, in-memory
client — matching the diagnostic) at scale, with the realism the
synthetic test lacked: round-robin interleaving across NPCs, the global
id counter, tombstone (compaction) churn, and tag metadata.

It checks `added_count` (ground truth) vs `get_recent` (the suspect)
per NPC at checkpoints, and reports the first divergence — catching the
gap in minutes instead of a 25h sim. No LLM, no game loop.

Run:  python3 tests/simulation/repro_retrieval_gap.py --npcs 10 --per 5500
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.memory.episodic import EpisodicStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

_CATS = ["conversation", "conversation_turn", "observation", "reflection",
         "npc", "object", "note"]
_WORDS = ("bridge timber river flood market grain forge ale festival "
          "council debt promise rumour harvest stranger gossip trust "
          "quarrel kindness").split()


def _text(rng_state: int) -> str:
    # Deterministic varied text without Math.random (cheap LCG).
    s = rng_state
    out = []
    for _ in range(12):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(_WORDS[s % len(_WORDS)])
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npcs", type=int, default=10)
    ap.add_argument("--per", type=int, default=5500,
                    help="memories per NPC (30-day scale ~5500)")
    ap.add_argument("--tombstone-frac", type=float, default=0.4,
                    help="fraction of memories to tombstone (compaction churn)")
    ap.add_argument("--checkpoint", type=int, default=1000,
                    help="check added-vs-retrieved every N adds-per-NPC")
    args = ap.parse_args()

    store = EpisodicStore()  # real ChromaDB, in-memory — matches diagnostic
    store.initialise()
    if store._fallback_mode:
        print("ERROR: ChromaDB unavailable — repro needs the real store.")
        return 2

    npcs = [f"npc_{i}" for i in range(args.npcs)]
    ids_by_npc: dict[str, list[str]] = {n: [] for n in npcs}
    start = time.time()
    seed = 1

    def checkpoint(label: str) -> bool:
        """Return True if a gap is found."""
        found = False
        rows = []
        for n in npcs:
            added = store.added_count(n)
            retrieved = len(store.get_recent(n, limit=100000,
                                             include_compacted=True))
            gap = added > 0 and retrieved == 0
            found = found or gap
            rows.append(f"{n}={retrieved}/{added}" + ("!!GAP" if gap else ""))
        print(f"[{label}] t={time.time()-start:.0f}s  "
              + "  ".join(rows), flush=True)
        return found

    # Round-robin interleave adds across NPCs (like a real sim), with
    # periodic tombstone churn, to the target per-NPC count.
    for k in range(args.per):
        for n in npcs:
            seed += 1
            mid = store.add_memory(
                npc_id=n, description=_text(seed),
                category=_CATS[seed % len(_CATS)],
                game_time=float(k), importance=0.5,
                tags={_WORDS[seed % len(_WORDS)]},
            )
            ids_by_npc[n].append(mid)
        # Compaction-style tombstone churn: tombstone older memories.
        if k > 0 and k % 200 == 0:
            for n in npcs:
                cutoff = int(len(ids_by_npc[n]) * args.tombstone_frac)
                for mid in ids_by_npc[n][:cutoff]:
                    store.update_metadata(mid, {"compacted_into": "sum"})
        if k > 0 and k % args.checkpoint == 0:
            if checkpoint(f"k={k}"):
                print(">>> RETRIEVAL GAP REPRODUCED at per-NPC count ~%d" % k)
                return 1

    print(f"total adds: {sum(store.added_count(n) for n in npcs)} "
          f"in {time.time()-start:.0f}s")
    if checkpoint("FINAL"):
        print(">>> RETRIEVAL GAP REPRODUCED at final scale")
        return 1
    print(">>> NO GAP reproduced — real-embedding store retrieved all NPCs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
