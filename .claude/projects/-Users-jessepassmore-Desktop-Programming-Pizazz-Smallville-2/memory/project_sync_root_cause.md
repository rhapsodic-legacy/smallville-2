---
name: NPC Sync Root Cause
description: After 43 simulated days NPCs synchronise — caused by shared RNG + identical schedule templates. Fix requires per-NPC RNG and schedule variation.
type: project
---

After extended simulation (43+ days), all NPCs move in synchronised lockstep.

**Root cause:** Single shared `random.Random(seed)` in NPCManager generates ALL stagger delays, subtask selections, and queue refills. After thousands of deterministic calls, the RNG produces identical sequences for same-occupation NPCs. Combined with identical schedule templates per occupation, NPCs lock into perfect sync.

**Why:** manager.py:79 seeds RNG once. Slot transitions (manager.py:283) fire for ALL NPCs simultaneously. Stagger delays (manager.py:396,422) and subtask decomposition (decompose.py) all draw from same RNG sequentially.

**How to apply:** Fix requires: (1) per-NPC RNG seeded from hash(global_seed, npc_id), (2) schedule time jitter per NPC, (3) subtask duration jitter. Full plan in DIAGNOSTIC_EXPERIMENT_PLAN.md. Memory system is also in-memory only — no persistence configured.
