"""Delete-churn reproduction for the episodic RETRIEVAL GAP.

The scale/embedding/tombstone repro came back clean at 80k docs. The one
big untested difference between that harness and the real sim is
DELETION: the sim consolidates every conversation by deleting its
per-turn entries (`consolidate_conversation_turns` -> `delete_by_metadata
("conversation_id", ...)`), a massive add-then-delete cycle on
`conversation_turn` rows. This harness models exactly that lifecycle and
hammers it, checking whether the per-NPC `get(where={npc_id})` query
(get_recent's mechanism) ever drops an NPC after enough delete churn.

It replicates EpisodicStore's ChromaDB mechanics directly (raw collection,
the same get/delete calls) with DUMMY embeddings so it runs in minutes —
the delete bug, if real, is in the ChromaDB metadata-index churn, not the
embedding values. (If this comes back clean, the next step is the same
lifecycle with real embeddings / the actual store.)

Run:  python3 tests/simulation/repro_delete_churn.py --npcs 10 --ticks 4000
"""

from __future__ import annotations

import argparse
import sys
import time

import chromadb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npcs", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=4000,
                    help="conversation rounds per NPC")
    ap.add_argument("--turns", type=int, default=6,
                    help="per-turn entries per conversation (added then deleted)")
    ap.add_argument("--checkpoint", type=int, default=500)
    ap.add_argument("--dim", type=int, default=8)
    args = ap.parse_args()

    client = chromadb.Client()
    col = client.get_or_create_collection(
        name="delete_churn", metadata={"hnsw:space": "cosine"},
    )
    npcs = [f"npc_{i}" for i in range(args.npcs)]
    expected_live = {n: 0 for n in npcs}   # surviving summaries/obs per NPC
    gid = 0
    conv = 0
    start = time.time()

    def emb():
        # cheap deterministic dummy embedding
        nonlocal gid
        return [((gid * 2654435761 + j * 40503) % 1000) / 1000.0
                for j in range(args.dim)]

    def checkpoint(label: str) -> bool:
        found = False
        rows = []
        for n in npcs:
            try:
                got = len(col.get(where={"npc_id": n},
                                  include=["metadatas"])["ids"])
            except Exception as e:
                got = -1
                print(f"  {n}: get EXCEPTION {e}")
            exp = expected_live[n]
            # Gap = retrieval returns 0 (or far below survivors) despite
            # many surviving adds.
            bad = exp > 50 and got <= 0
            found = found or bad
            rows.append(f"{n}={got}/{exp}" + ("!!GAP" if bad else ""))
        print(f"[{label}] t={time.time()-start:.0f}s  " + "  ".join(rows),
              flush=True)
        return found

    for k in range(args.ticks):
        for n in npcs:
            conv += 1
            cid = f"conv_{conv}"
            # Add per-turn entries (transient) — mimics persist_conversation_turn
            tids = []
            for _ in range(args.turns):
                gid += 1
                mid = f"{n}_mem_{gid}"
                tids.append(mid)
                col.add(ids=[mid], embeddings=[emb()],
                        metadatas=[{"npc_id": n, "conversation_id": cid,
                                    "category": "conversation_turn",
                                    "game_time": float(k)}],
                        documents=[f"turn {gid}"])
            # Add the consolidated summary (persists) — mimics record_conversation
            gid += 1
            sid = f"{n}_mem_{gid}"
            col.add(ids=[sid], embeddings=[emb()],
                    metadatas=[{"npc_id": n, "category": "conversation",
                                "game_time": float(k)}],
                    documents=[f"summary {gid}"])
            expected_live[n] += 1
            # Delete the turns by conversation_id — mimics
            # consolidate_conversation_turns -> delete_by_metadata
            r = col.get(where={"conversation_id": cid})
            if r["ids"]:
                col.delete(ids=r["ids"])
        if k > 0 and k % args.checkpoint == 0:
            if checkpoint(f"k={k}"):
                print(">>> RETRIEVAL GAP REPRODUCED under delete churn "
                      f"at ~{k} conversations/NPC")
                return 1

    print(f"total live adds: {sum(expected_live.values())}, "
          f"conversations: {conv}, in {time.time()-start:.0f}s")
    if checkpoint("FINAL"):
        print(">>> RETRIEVAL GAP REPRODUCED under delete churn")
        return 1
    print(">>> NO GAP under delete churn (dummy embeddings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
