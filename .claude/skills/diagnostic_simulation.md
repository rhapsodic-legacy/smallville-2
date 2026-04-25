---
name: diagnostic_simulation
description: Diagnostic simulation that dumps per-tick NPC state logs — positions, activities, subtasks, synchronisation metrics, overlaps. Use to debug visual behavior issues that tests alone can't catch.
---

# Diagnostic Simulation

## What It Does
Runs a headless simulation (120 ticks at 1s/tick = 2 minutes) and dumps a detailed per-tick log showing:

- Per-tick: walker count, departures, arrivals, idle count, overlaps, state changes
- Flagged ticks: SYNC_DEPART, SYNC_ARRIVE, OVERLAPS, MASS_IDLE, STATE_CHURN
- Every 30 ticks: full NPC state dump (position, activity, subtask, queue, path, description)
- Summary by 10-tick windows
- Final NPC state

## When To Use
- When visual behavior doesn't match what tests show
- When the user reports jitter, stacking, or synchronised movement
- After changing client rendering code (the diagnostic proves server state is correct)
- Before/after any movement or intelligence system change

## How To Run
```bash
python3 tests/simulation/diagnostic_per_tick_state.py
```

## Key Insight
Server-side logic and client-side rendering are separate systems. Tests validate server logic. The diagnostic log proves whether the server is producing correct state. If the diagnostic looks clean but the user sees problems, the bug is in the **client renderer** (npc_renderer.js), not the server.

## Common Client-Side Issues
- **Drift correction jitter**: Client slowly interpolates to server position every frame, causing micro-movement when resting
- **Path snap-back**: Client resets pathIndex=0 every tick, re-traversing already-covered waypoints
- **Synchronised visual movement**: Server stagger is correct but client applies uniform animation timing
