# Skill: World Generation

## When to Use
When working on procedural town generation, terrain layout, or building placement.

## Generation Philosophy
Subtractive constraints (from AI Game Studio pattern):
1. Start with all candidate positions
2. Apply terrain constraints (no buildings on water)
3. Apply spacing constraints (buildings need clearance)
4. Apply connectivity constraints (all buildings reachable via roads)
5. Place in priority order: central buildings first, homes last

## Generation Driven by Parameters

    world = generate_town(
        population=30,
        terrain="riverside",      # riverside, plains, forest_clearing, hilltop
        has_ruler=True,
        ruler_type="good_king",   # good_king, tyrant, council, absent
        economy="trade",          # trade, farming, mining, mixed
        seed=42,                  # reproducible generation
    )

## Building Types and Priority
1. **Town centre** — Market square, notice board (always placed first)
2. **Governance** — Castle/town hall (if has_ruler)
3. **Commerce** — Shops, tavern, market stalls
4. **Religion** — Church or church plot (can be unbuilt for construction quest)
5. **Production** — Blacksmith, farm plots, mine entrance
6. **Residential** — Homes (one per NPC household)
7. **Infrastructure** — Roads connecting all buildings, bridges over water

## Tile Types
- grass, dirt_road, stone_road
- water, bridge
- forest, rock
- building_floor, building_wall, door

## Spatial Rules
- Buildings must be at least 2 tiles apart
- Roads must connect every building to the town centre
- Water features create natural boundaries
- Trees and rocks serve as decorative obstacles
- Resource nodes (trees for wood, rocks for stone) placed at map edges

## Output Format
Grid of tiles, each with:

    {
        "terrain": "grass",
        "object": None,           # or "tree", "rock", "building_wall", etc.
        "building_id": None,      # or "tavern_1", "home_3", etc.
        "address": "smallville:town_centre:market_square",
        "walkable": True,
        "resource": None,         # or {"type": "wood", "yield": 5}
    }
