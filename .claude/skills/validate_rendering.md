---
name: validate_rendering
description: Simulates client-side rendering at 60fps to detect teleporting, snap-backs, and building clips. Run after ANY change to NPC movement, pathfinding, execute.py, models.py to_dict, or npc_renderer.js.
---

# Client Rendering Validation

## What It Does
Runs a headless server simulation (120 ticks), captures the exact WebSocket data
each tick (to_dict output), then simulates 60fps client-side rendering to detect:

- **TELEPORT**: NPC moves >3 tiles in a single frame (visual pop)
- **SNAPBACK**: NPC reverses direction suddenly (walks forward then jumps back)
- **BUILDING_CLIP**: NPC visual position inside a building tile

## When To Use
- After changing `core/npc/cognition/execute.py` (_follow_path, _tick_trail)
- After changing `core/npc/models.py` (to_dict, position fields, trail)
- After changing `client/js/npc_renderer.js` (movement, lerp, trail walking)
- After changing `core/world/pathfinding.py` (path routing)
- After changing movement speeds, tick intervals, or timing
- When the user reports jitter, teleporting, or building clipping

## How To Run
```bash
python3 tests/simulation/diagnostic_client_rendering.py
```

## How It Works
The simulation mirrors the exact client JavaScript logic in Python:
1. `update_from_tick()` mirrors `_updateMoveState()` — processes trail waypoints
2. `simulate_frame()` mirrors `_walkTrail()` — walks through waypoints at computed speed
3. `check_anomaly()` detects visual problems per frame

## Architecture (Stanford Approach)
- Server sends `trail`: tiles traversed THIS tick (max 2-3 waypoints)
- Client walks through trail waypoints smoothly between ticks
- No full remaining path sent. No client-side path following. No sync state.
- Eliminates all snap-back and teleporting bugs from dual-state sync.

## Key Constants (must match client)
- `TELEPORT_DISTANCE = 4.0` — jump threshold (snap instead of lerp)
- `ARRIVAL_FACTOR = 0.85` — arrive before next tick
- `TICK_INTERVAL = 1.0` — seconds between server ticks

## Also Run
- `python3 tests/simulation/diagnostic_door_placement.py` — doors adjacent to buildings
- `python3 tests/simulation/test_npc_movement.py` — server-side movement validation
