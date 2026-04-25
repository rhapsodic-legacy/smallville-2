---
name: spatial_awareness
description: Rules and patterns for NPC positioning, rest-tile selection, and spatial integrity
---

# Spatial Awareness Skill

## Core Rules (non-negotiable)

1. **Pass-through OK**: NPCs may move through each other freely while walking. Never add movement-phase collision.
2. **No resting overlap**: Two or more NPCs must NEVER share a tile when at rest (any state except WALKING).
3. **Conversations require adjacency**: Conversing NPCs must be on adjacent tiles (Manhattan distance = 1) before dialogue begins.

## Module: `core/world/spatial_awareness.py`

This is the single source of truth for safe tile selection. All rest-placement goes through it.

### Key Functions

- `get_occupied_tiles(npcs)` — returns `set[tuple[int,int]]` of tiles with resting NPCs
- `find_rest_tile(target_x, target_z, grid, occupied, ...)` — nearest free passable tile to target
- `find_conversation_positions(npc_a, npc_b, grid, occupied)` — two adjacent free tiles for a conversation pair
- `resolve_overlaps(npcs, grid)` — safety net that nudges stacked resting NPCs apart

### When to Use

- **NPC arrives at destination** (`execute.py:_arrive_at_destination`): call `find_rest_tile` before settling
- **Conversation starts** (`converse.py:initiate_conversation`): call `find_conversation_positions` and move both NPCs
- **Slot transitions** (`manager.py:_spread_destination`): call `find_rest_tile` when assigning schedule locations
- **Every tick** (`manager.py:tick`): call `resolve_overlaps` as a safety net after all movement

### When NOT to Use

- During pathfinding/A* — NPCs walk through each other
- For walking NPCs — they are transient, not resting

## Testing Invariant

The spatial integrity test (`tests/unit/test_spatial_awareness.py`) enforces:
- No two resting NPCs share a tile after any operation
- `find_rest_tile` never returns an occupied tile
- `find_conversation_positions` returns adjacent tiles (distance = 1)
- Multi-NPC simulation with 20 NPCs over 100 ticks: zero resting overlaps

## Adding New NPC Placement

Any new code that places an NPC at rest (building, crafting, trading, sleeping) MUST:
1. Call `get_occupied_tiles(all_npcs)` to get current occupancy
2. Call `find_rest_tile(target, grid, occupied)` to get a safe position
3. Set `npc.x, npc.z` to the returned position

Never set an NPC's position directly without checking occupancy first.
