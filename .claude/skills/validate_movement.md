---
name: validate_movement
description: Automated NPC movement validation — unit tests + live 300-tick simulation that observes actual NPC behaviour. Runs automatically via hook on file changes.
---

# Validate Movement Pipeline

## What It Does
Two-stage automated validation that runs whenever movement-related code changes:

### Stage 1: Unit Tests (`test_npc_movement.py`)
15 targeted checks on world generation and pathfinding:
1. Door tiles passable
2. All doors reachable from town centre
3. Building clearance (no trapped buildings)
4. **Building impermeability** — every non-door building tile is impassable
5. **Diagonal paths exist** — A* produces diagonal segments between buildings
6. **Diagonal no corner clipping** — diagonal steps never clip building corners
7. Path waypoint/distance ratio
8. NPC reaches destination within time limit
9. No water tile violations
10. No building intrusions
11. No overlapping resting NPCs
12. Multi-NPC collision safety
13. Stuck detection/recovery
14. Varied NPC speeds (unique per NPC)
15. **Speed range wide enough** — fastest >1.5x slowest

### Stage 2: Live Simulation (`test_live_simulation.py`)
Boots the full world, runs NPCManager.tick() for 300 ticks, observes every NPC every tick:
- **Teleportation** — NPC moves >5 tiles in one tick
- **Building intrusion** — resting NPC on impassable tile
- **Water entry** — NPC stands on water
- **Stuck** — walking NPC makes no path progress for 10+ ticks
- **Resting overlap** — two non-walking NPCs share a tile
- **Synchronized departure** — too many NPCs start walking on the same tick
- **Synchronized arrival** — too many NPCs stop walking on the same tick
- **Departure spread** — no single tick has >30% of population departing

Reports: NPCs that moved, max simultaneous walkers, travel distances, path assignments, departure histogram.

## How It Runs
**Automatically** — a PostToolUse hook in `.claude/settings.json` triggers both stages whenever these files are edited:
- `pathfinding.py`, `generator.py`, `spatial_awareness.py`
- `execute.py`, `manager.py`, `npc_renderer.js`

If either stage fails, the hook blocks further edits until the issue is fixed.

## Manual Run (if needed)
```bash
python3 tests/simulation/test_npc_movement.py    # Stage 1
python3 tests/simulation/test_live_simulation.py  # Stage 2
```

## Adding New Checks
- Unit checks: add to `ALL_TESTS` or `STANDALONE_TESTS` in `test_npc_movement.py`
- Behavioural checks: add anomaly detection in the tick loop in `test_live_simulation.py`
