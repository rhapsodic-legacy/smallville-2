"""
Terrain feature definitions and placement logic.

Features are specific terrain elements (bridges, ponds, walls, ruins, etc.)
that can be placed on the grid during or after standard town generation.
Each feature type has its own placement function that respects existing terrain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.world.grid import Grid, Terrain


# ---------- Data model ----------

@dataclass
class TerrainFeature:
    """A terrain feature to place during generation."""
    type: str           # "bridge", "pond", "wall", "ruins", "garden", "watchtower"
    count: int = 1
    placement: str = "auto"   # "auto", "near_river", "edge", "centre", "outskirts"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "count": self.count,
            "placement": self.placement,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TerrainFeature:
        return cls(
            type=data["type"],
            count=data.get("count", 1),
            placement=data.get("placement", "auto"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TownMood:
    """Aesthetic/feel modifiers derived from descriptors like 'cozy'."""
    descriptor: str
    population_modifier: float = 1.0
    building_bias: dict = field(default_factory=dict)
    economy_hint: str = ""       # suggests economy type if not explicit

    def to_dict(self) -> dict:
        return {
            "descriptor": self.descriptor,
            "population_modifier": self.population_modifier,
            "building_bias": self.building_bias,
            "economy_hint": self.economy_hint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TownMood:
        return cls(
            descriptor=data["descriptor"],
            population_modifier=data.get("population_modifier", 1.0),
            building_bias=data.get("building_bias", {}),
            economy_hint=data.get("economy_hint", ""),
        )


# ---------- Mood presets ----------

MOOD_PRESETS: dict[str, TownMood] = {
    "cozy": TownMood(
        descriptor="cozy",
        population_modifier=0.7,
        building_bias={"tavern": 1, "home": 2},
        economy_hint="mixed",
    ),
    "bustling": TownMood(
        descriptor="bustling",
        population_modifier=1.5,
        building_bias={"market_stall": 2, "tavern": 1},
        economy_hint="trading",
    ),
    "fortified": TownMood(
        descriptor="fortified",
        population_modifier=1.0,
        building_bias={"town_hall": 1},
        economy_hint="mining",
    ),
    "rustic": TownMood(
        descriptor="rustic",
        population_modifier=0.8,
        building_bias={"farm": 2},
        economy_hint="farming",
    ),
    "sacred": TownMood(
        descriptor="sacred",
        population_modifier=1.0,
        building_bias={"church": 1},
        economy_hint="mixed",
    ),
    "frontier": TownMood(
        descriptor="frontier",
        population_modifier=0.6,
        building_bias={"blacksmith": 1, "farm": 1},
        economy_hint="mining",
    ),
}

# Valid feature types that the system recognises
VALID_FEATURE_TYPES = frozenset({
    "bridge", "pond", "wall", "ruins", "garden", "watchtower",
    "orchard", "well", "campfire",
})


# ---------- Placement functions ----------

def place_features(
    grid: Grid,
    features: list[TerrainFeature],
    rng: random.Random,
) -> list[dict]:
    """Place all terrain features on the grid. Returns placement records."""
    from core.world.grid import Terrain, WorldObject

    records: list[dict] = []
    for feat in features:
        placer = _PLACERS.get(feat.type)
        if placer is None:
            continue
        for i in range(feat.count):
            result = placer(grid, feat, rng, i)
            if result:
                records.append(result)
    return records


def _place_bridge(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a bridge across a river. Finds water tiles and spans them."""
    from core.world.grid import Terrain, WorldObject

    water_tiles = grid.find_tiles_by_terrain(Terrain.WATER)
    if not water_tiles:
        return None

    # Group water tiles by z-coordinate to find rows with continuous water
    z_rows: dict[int, list[int]] = {}
    for t in water_tiles:
        z_rows.setdefault(t.z, []).append(t.x)

    # Find good bridge locations: rows where water spans 2-4 tiles wide
    candidates = []
    for z, x_coords in z_rows.items():
        x_coords.sort()
        width = x_coords[-1] - x_coords[0] + 1
        if 2 <= width <= 6:
            candidates.append((z, x_coords[0], x_coords[-1]))

    if not candidates:
        return None

    # Pick a random candidate, avoiding previously placed bridges
    rng.shuffle(candidates)
    z, x_start, x_end = candidates[0]

    # Place bridge tiles: make water passable, change terrain to road
    bridge_id = f"bridge_{index + 1}"
    for x in range(x_start - 1, x_end + 2):
        tile = grid.get_tile(x, z)
        if tile is None:
            continue
        if tile.terrain == Terrain.WATER:
            tile.terrain = Terrain.ROAD
            tile.walkable = True
        # Also set adjacent sand tiles to road for smooth approach
        if tile.terrain == Terrain.SAND:
            tile.terrain = Terrain.ROAD

    # Place bridge object on centre tile
    centre_x = (x_start + x_end) // 2
    obj = WorldObject(
        object_id=bridge_id,
        object_type="structure",
        name=f"Bridge {index + 1}",
        walkable=True,
        metadata={"feature": "bridge", "spans_z": z},
    )
    grid.place_object(centre_x, z, obj)

    return {"type": "bridge", "x": centre_x, "z": z, "id": bridge_id}


def _place_pond(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a small pond (3x3 to 5x5 water area)."""
    from core.world.grid import Terrain, WorldObject

    min_x, min_z, max_x, max_z = grid.bounds
    size = rng.choice([3, 4, 5])
    half = size // 2

    # Try to place in outskirts
    for _ in range(100):
        cx = rng.randint(min_x + size, max_x - size)
        cz = rng.randint(min_z + size, max_z - size)

        # Prefer placement away from centre
        if abs(cx) < 10 and abs(cz) < 10:
            continue

        # Check all tiles are grass and passable
        clear = True
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                tile = grid.get_tile(cx + dx, cz + dz)
                if tile is None or tile.terrain != Terrain.GRASS:
                    clear = False
                    break
                if tile.objects:
                    clear = False
                    break
            if not clear:
                break

        if not clear:
            continue

        # Carve the pond with an organic shape (skip some corners)
        pond_id = f"pond_{index + 1}"
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                # Skip corners for organic shape
                if abs(dx) == half and abs(dz) == half:
                    continue
                tile = grid.get_tile(cx + dx, cz + dz)
                if tile:
                    tile.terrain = Terrain.WATER
                    tile.walkable = False

        # Sandy banks around the pond
        for dx in range(-half - 1, half + 2):
            for dz in range(-half - 1, half + 2):
                tile = grid.get_tile(cx + dx, cz + dz)
                if tile and tile.terrain == Terrain.GRASS:
                    dist = max(abs(dx), abs(dz))
                    if dist == half + 1:
                        tile.terrain = Terrain.SAND

        obj = WorldObject(
            object_id=pond_id,
            object_type="structure",
            name=f"Pond",
            walkable=False,
            metadata={"feature": "pond", "size": size},
        )
        grid.place_object(cx, cz, obj)
        return {"type": "pond", "x": cx, "z": cz, "id": pond_id, "size": size}

    return None


def _place_wall(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a defensive wall/palisade ring around the town centre."""
    from core.world.grid import Terrain, WorldObject

    # Wall forms a square ring at a set radius from centre
    radius = feat.metadata.get("radius", 12)
    wall_id = f"wall_{index + 1}"
    placed = 0

    for x in range(-radius, radius + 1):
        for z in range(-radius, radius + 1):
            # Only place on the perimeter
            if abs(x) != radius and abs(z) != radius:
                continue

            tile = grid.get_tile(x, z)
            if tile is None:
                continue
            # Don't overwrite buildings, water, or roads
            if tile.terrain not in (Terrain.GRASS, Terrain.DIRT):
                continue
            if tile.objects:
                continue

            tile.terrain = Terrain.STONE
            tile.walkable = False
            placed += 1

    # Leave gates (gaps) on each cardinal side
    for gx, gz in [(0, -radius), (0, radius), (-radius, 0), (radius, 0)]:
        for offset in range(-1, 2):
            # Gate is 3 tiles wide
            if abs(gx) == radius:
                tile = grid.get_tile(gx, gz + offset)
            else:
                tile = grid.get_tile(gx + offset, gz)
            if tile and tile.terrain == Terrain.STONE:
                tile.terrain = Terrain.ROAD
                tile.walkable = True

    if placed == 0:
        return None

    return {"type": "wall", "radius": radius, "id": wall_id, "tiles": placed}


def _place_ruins(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a small cluster of ruins (3x3 to 4x4 area)."""
    from core.world.grid import Terrain, WorldObject

    min_x, min_z, max_x, max_z = grid.bounds
    size = rng.choice([3, 4])

    for _ in range(80):
        x = rng.randint(min_x + 5, max_x - 5)
        z = rng.randint(min_z + 5, max_z - 5)

        # Place in outskirts
        if abs(x) < 12 and abs(z) < 12:
            continue

        clear = True
        for dx in range(size):
            for dz in range(size):
                tile = grid.get_tile(x + dx, z + dz)
                if tile is None or tile.terrain != Terrain.GRASS or tile.objects:
                    clear = False
                    break
            if not clear:
                break

        if not clear:
            continue

        ruin_id = f"ruins_{index + 1}"
        # Scatter stone tiles with gaps for atmosphere
        for dx in range(size):
            for dz in range(size):
                tile = grid.get_tile(x + dx, z + dz)
                if tile and rng.random() < 0.6:
                    tile.terrain = Terrain.STONE
                    tile.walkable = False

        obj = WorldObject(
            object_id=ruin_id,
            object_type="structure",
            name="Ancient Ruins",
            walkable=False,
            metadata={"feature": "ruins"},
        )
        grid.place_object(x, z, obj)
        return {"type": "ruins", "x": x, "z": z, "id": ruin_id}

    return None


def _place_garden(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a decorative garden (3x3 passable area with flower objects)."""
    from core.world.grid import Terrain, WorldObject

    min_x, min_z, max_x, max_z = grid.bounds

    for _ in range(60):
        x = rng.randint(min_x + 4, max_x - 4)
        z = rng.randint(min_z + 4, max_z - 4)

        clear = True
        for dx in range(3):
            for dz in range(3):
                tile = grid.get_tile(x + dx, z + dz)
                if tile is None or tile.terrain != Terrain.GRASS or tile.objects:
                    clear = False
                    break
            if not clear:
                break

        if not clear:
            continue

        garden_id = f"garden_{index + 1}"
        plants = ["wildflowers", "herb patch", "rose bush", "lavender"]
        for dx in range(3):
            for dz in range(3):
                tile = grid.get_tile(x + dx, z + dz)
                if tile:
                    tile.terrain = Terrain.DIRT
                    # Place a plant on ~half the tiles
                    if rng.random() < 0.5:
                        plant = rng.choice(plants)
                        obj = WorldObject(
                            object_id=f"{garden_id}_{plant}_{dx}_{dz}",
                            object_type="decoration",
                            name=plant.title(),
                            walkable=True,
                            metadata={"feature": "garden"},
                        )
                        grid.place_object(x + dx, z + dz, obj)

        return {"type": "garden", "x": x, "z": z, "id": garden_id}

    return None


def _place_watchtower(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a watchtower (2x2 building) near the map edge."""
    from core.world.grid import Terrain, WorldObject

    min_x, min_z, max_x, max_z = grid.bounds

    # Try edge positions
    edge_candidates = []
    margin = 4
    for x in range(min_x + margin, max_x - margin):
        for z in [min_z + margin, max_z - margin]:
            edge_candidates.append((x, z))
    for z in range(min_z + margin, max_z - margin):
        for x in [min_x + margin, max_x - margin]:
            edge_candidates.append((x, z))

    rng.shuffle(edge_candidates)

    for x, z in edge_candidates:
        clear = True
        for dx in range(2):
            for dz in range(2):
                tile = grid.get_tile(x + dx, z + dz)
                if tile is None or tile.terrain != Terrain.GRASS or tile.objects:
                    clear = False
                    break
            if not clear:
                break

        if not clear:
            continue

        tower_id = f"watchtower_{index + 1}"
        for dx in range(2):
            for dz in range(2):
                tile = grid.get_tile(x + dx, z + dz)
                if tile:
                    tile.terrain = Terrain.STONE
                    tile.walkable = False

        # Door on south side
        door_tile = grid.get_tile(x, z + 1)
        if door_tile:
            door_tile.walkable = True
            door_tile.terrain = Terrain.ROAD

        obj = WorldObject(
            object_id=tower_id,
            object_type="building",
            name="Watchtower",
            walkable=False,
            metadata={"feature": "watchtower", "width": 2, "height": 2},
        )
        grid.place_object(x, z, obj)
        return {"type": "watchtower", "x": x, "z": z, "id": tower_id}

    return None


def _place_well(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a well (single tile) near the town centre."""
    from core.world.grid import Terrain, WorldObject

    for _ in range(40):
        x = rng.randint(-8, 8)
        z = rng.randint(-8, 8)
        tile = grid.get_tile(x, z)
        if tile is None:
            continue
        if tile.terrain not in (Terrain.GRASS, Terrain.DIRT):
            continue
        if tile.objects:
            continue

        well_id = f"well_{index + 1}"
        obj = WorldObject(
            object_id=well_id,
            object_type="structure",
            name="Well",
            walkable=True,
            metadata={"feature": "well"},
        )
        grid.place_object(x, z, obj)
        return {"type": "well", "x": x, "z": z, "id": well_id}

    return None


def _place_campfire(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place a campfire (single tile) in outskirts."""
    from core.world.grid import Terrain, WorldObject

    min_x, min_z, max_x, max_z = grid.bounds

    for _ in range(40):
        x = rng.randint(min_x + 5, max_x - 5)
        z = rng.randint(min_z + 5, max_z - 5)
        if abs(x) < 10 and abs(z) < 10:
            continue
        tile = grid.get_tile(x, z)
        if tile is None or tile.terrain != Terrain.GRASS or tile.objects:
            continue

        fire_id = f"campfire_{index + 1}"
        obj = WorldObject(
            object_id=fire_id,
            object_type="structure",
            name="Campfire",
            walkable=True,
            metadata={"feature": "campfire"},
        )
        grid.place_object(x, z, obj)
        tile.terrain = Terrain.DIRT
        return {"type": "campfire", "x": x, "z": z, "id": fire_id}

    return None


def _place_orchard(
    grid: Grid, feat: TerrainFeature, rng: random.Random, index: int,
) -> dict | None:
    """Place an orchard (4x4 grid of fruit trees)."""
    from core.world.grid import Terrain, WorldObject

    min_x, min_z, max_x, max_z = grid.bounds

    for _ in range(60):
        x = rng.randint(min_x + 5, max_x - 9)
        z = rng.randint(min_z + 5, max_z - 9)
        if abs(x) < 10 and abs(z) < 10:
            continue

        clear = True
        for dx in range(4):
            for dz in range(4):
                tile = grid.get_tile(x + dx, z + dz)
                if tile is None or tile.terrain != Terrain.GRASS or tile.objects:
                    clear = False
                    break
            if not clear:
                break

        if not clear:
            continue

        orchard_id = f"orchard_{index + 1}"
        trees = ["apple tree", "pear tree", "cherry tree"]
        for dx in range(4):
            for dz in range(4):
                tile = grid.get_tile(x + dx, z + dz)
                if tile:
                    tile.terrain = Terrain.DIRT
                    # Place trees in a grid pattern (every other tile)
                    if dx % 2 == 0 and dz % 2 == 0:
                        tree = rng.choice(trees)
                        obj = WorldObject(
                            object_id=f"{orchard_id}_{dx}_{dz}",
                            object_type="resource",
                            name=tree.title(),
                            walkable=True,
                            metadata={
                                "feature": "orchard",
                                "resource": "food",
                                "yield": rng.randint(2, 5),
                            },
                        )
                        grid.place_object(x + dx, z + dz, obj)

        return {"type": "orchard", "x": x, "z": z, "id": orchard_id}

    return None


# ---------- Placer registry ----------

_PLACERS = {
    "bridge": _place_bridge,
    "pond": _place_pond,
    "wall": _place_wall,
    "ruins": _place_ruins,
    "garden": _place_garden,
    "watchtower": _place_watchtower,
    "well": _place_well,
    "campfire": _place_campfire,
    "orchard": _place_orchard,
}
