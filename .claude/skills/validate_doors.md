---
name: validate_doors
description: Validates that building doors are correctly placed adjacent to their buildings — both server data and client rendering coordinates. Run after any change to world generation or building rendering.
---

# Door Placement Validation

## What It Does
Generates a world and validates every building's door placement:
- Server-side: door tile is adjacent to the building footprint (not inside, not displaced)
- Client-side: door 3D coordinate (door_x+0.5, door_z+0.5) is within expected distance of building centre

## When To Use
- After changing `core/world/generator.py` (building placement, door calculation)
- After changing `client/js/world_renderer.js` (`_buildDoorMarkers`, `_createBuilding`)
- When the user reports doors appearing in wrong locations

## How To Run
```bash
python3 tests/simulation/diagnostic_door_placement.py
```

## Key Insight
Doors are placed in **grid tile coordinates** by the generator (`door_x`, `door_z`).
The client must convert to 3D coordinates by adding 0.5 to centre within the tile.
The door tile is always OUTSIDE the building footprint (one tile past the south face):
`door_z = building.z + building.height`.

Previous bugs occurred because the client ignored `door_x`/`door_z` and tried to calculate door position from building geometry — this always produced wrong results.
