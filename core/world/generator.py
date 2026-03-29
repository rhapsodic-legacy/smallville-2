"""
Procedural town generator.

Creates a complete town layout on a Grid based on configurable parameters:
population, terrain style, economy type, and optional ruler.
Handles building placement, road networks, and resource nodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.world.grid import Grid, Terrain, Tile, WorldObject


# ---------- Configuration ----------

@dataclass
class BuildingDef:
    """Blueprint for a building type."""
    building_type: str
    name: str
    width: int
    height: int
    required: bool = False  # must appear in every town
    max_count: int = 1
    sector: str = ""


# Default building catalogue — order matters for placement priority
DEFAULT_BUILDINGS: list[BuildingDef] = [
    BuildingDef("tavern", "Tavern", 4, 4, required=True, sector="market"),
    BuildingDef("blacksmith", "Blacksmith", 3, 3, required=True, sector="market"),
    BuildingDef("market_stall", "Market Stall", 2, 2, required=True,
                max_count=3, sector="market"),
    BuildingDef("church", "Church", 5, 5, sector="centre"),
    BuildingDef("town_hall", "Town Hall", 4, 4, sector="centre"),
    BuildingDef("home", "Home", 2, 3, max_count=20, sector="residential"),
    BuildingDef("farm", "Farm", 4, 4, max_count=4, sector="outskirts"),
]


@dataclass
class WorldConfig:
    """Parameters that shape the generated town."""
    population: int = 10
    terrain: str = "riverside"   # riverside, plains, forest_edge, hillside
    economy: str = "mixed"       # mixed, farming, trading, mining
    has_ruler: bool = False
    seed: int | None = None
    grid_width: int = 60
    grid_height: int = 60


# ---------- Generator ----------

@dataclass
class PlacedBuilding:
    """Record of a building placed on the grid."""
    building_type: str
    name: str
    x: int
    z: int
    width: int
    height: int
    sector: str
    door_x: int = 0
    door_z: int = 0
    interior_objects: list[str] = field(default_factory=list)


# Maps building type -> list of interior objects for flavour/sub-task targeting.
# NPCs reference these objects in their sub-task descriptions.
BUILDING_OBJECTS: dict[str, list[str]] = {
    "blacksmith": ["anvil", "furnace", "water trough", "whetstone", "bellows", "tool rack"],
    "tavern": ["bar counter", "hearth", "ale barrel", "kitchen", "dining table", "bard's corner"],
    "market_stall": ["display shelf", "ledger desk", "back room", "scales", "cash box"],
    "church": ["altar", "pews", "candle stand", "herb garden", "chronicle desk"],
    "town_hall": ["armoury", "guard post", "notice board", "meeting table"],
    "farm": ["chicken coop", "vegetable patch", "well", "compost heap", "tool shed", "beehives"],
    "home": ["bed", "hearth", "table", "wash basin", "storage chest"],
}


class TownGenerator:
    """Generates a complete town on a Grid from a WorldConfig."""

    def __init__(self, config: WorldConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.grid = Grid(config.grid_width, config.grid_height)
        self.buildings: list[PlacedBuilding] = []
        self._road_tiles: set[tuple[int, int]] = set()

    def generate(self) -> Grid:
        """Run the full generation pipeline and return the populated grid."""
        self._apply_base_terrain()
        self._place_town_centre()
        self._place_buildings()
        self._connect_roads()
        self._place_resource_nodes()
        self._assign_homes()
        self._validate_building_integrity()
        return self.grid

    def _validate_building_integrity(self) -> None:
        """Post-generation check: every non-door building tile must be impassable."""
        for b in self.buildings:
            door = (b.door_x, b.door_z)
            for dx in range(b.width):
                for dz in range(b.height):
                    tx, tz = b.x + dx, b.z + dz
                    if (tx, tz) == door:
                        continue
                    tile = self.grid.get_tile(tx, tz)
                    if tile and tile.is_passable:
                        # Road or other post-processing made a building tile walkable.
                        # Force it back to impassable.
                        tile.walkable = False

    # ---------- Terrain ----------

    def _apply_base_terrain(self) -> None:
        """Set base terrain based on the terrain style parameter."""
        terrain_map = self.config.terrain
        min_x, min_z, max_x, max_z = self.grid.bounds

        for tile in self.grid:
            tile.terrain = Terrain.GRASS

        if terrain_map == "riverside":
            self._carve_river()
        elif terrain_map == "forest_edge":
            self._place_forest_edge()
        elif terrain_map == "hillside":
            self._apply_hills()
        # "plains" is default grass — no modifications needed

    def _carve_river(self) -> None:
        """Carve a winding river across the map."""
        min_x, min_z, max_x, max_z = self.grid.bounds
        # Need enough room for river + banks
        if max_x - min_x < 22:
            return
        river_x = self.rng.randint(min_x + 10, max_x - 10)

        for z in range(min_z, max_z + 1):
            drift = self.rng.randint(-1, 1)
            river_x = max(min_x + 2, min(max_x - 2, river_x + drift))
            for dx in range(-1, 2):
                self.grid.set_terrain(river_x + dx, z, Terrain.WATER)
            # Sandy banks
            for dx in [-2, 2]:
                x = river_x + dx
                if self.grid.in_bounds(x, z):
                    self.grid.set_terrain(x, z, Terrain.SAND)

    def _place_forest_edge(self) -> None:
        """Place dense forest on one side of the map."""
        min_x, min_z, max_x, max_z = self.grid.bounds
        forest_boundary = min_x + self.grid.width // 3
        for tile in self.grid:
            if tile.x < forest_boundary:
                tile.terrain = Terrain.FOREST
                if self.rng.random() < 0.6:
                    tile.walkable = False

    def _apply_hills(self) -> None:
        """Apply elevation variation for hillside terrain."""
        for tile in self.grid:
            dist_from_centre = abs(tile.x) + abs(tile.z)
            tile.elevation = max(0.0, dist_from_centre * 0.1 - 1.0)
            if tile.elevation > 3.0:
                tile.terrain = Terrain.STONE
                tile.walkable = False

    # ---------- Town Centre ----------

    def _place_town_centre(self) -> None:
        """Mark the central area and place the town square."""
        # Town square: 6x6 dirt area at origin
        for x in range(-3, 4):
            for z in range(-3, 4):
                if self.grid.in_bounds(x, z):
                    self.grid.set_terrain(x, z, Terrain.DIRT)
                    self.grid.set_sector(x, z, "centre", "town_square")

    # ---------- Building Placement ----------

    def _place_buildings(self) -> None:
        """Place all buildings according to population and economy."""
        buildings_to_place = self._compute_building_list()

        # Sort by size descending so big buildings get placed first
        buildings_to_place.sort(key=lambda b: b.width * b.height, reverse=True)

        for bdef in buildings_to_place:
            pos = self._find_building_site(bdef)
            if pos is None:
                continue
            x, z = pos
            self._stamp_building(bdef, x, z)

    def _compute_building_list(self) -> list[BuildingDef]:
        """Decide which buildings to place based on config."""
        result: list[BuildingDef] = []
        homes_needed = self.config.population

        for bdef in DEFAULT_BUILDINGS:
            if bdef.building_type == "home":
                count = min(homes_needed, bdef.max_count)
                result.extend([bdef] * count)
            elif bdef.required:
                result.extend([bdef] * min(bdef.max_count, max(1, 1)))
            elif bdef.building_type == "church":
                if self.config.population >= 8:
                    result.append(bdef)
            elif bdef.building_type == "town_hall":
                if self.config.has_ruler:
                    result.append(bdef)
            elif bdef.building_type == "farm":
                if self.config.economy in ("farming", "mixed"):
                    count = min(bdef.max_count, self.config.population // 5 + 1)
                    result.extend([bdef] * count)
            elif bdef.building_type == "market_stall":
                if self.config.economy in ("trading", "mixed"):
                    count = min(bdef.max_count, 2)
                    result.extend([bdef] * count)

        return result

    def _find_building_site(self, bdef: BuildingDef) -> tuple[int, int] | None:
        """Find a suitable location for a building, respecting zones."""
        min_x, min_z, max_x, max_z = self.grid.bounds

        # Define zone preferences
        zone_ranges = {
            "centre": ((-6, -6), (6, 6)),
            "market": ((-10, -10), (10, 10)),
            "residential": ((-15, -15), (15, 15)),
            "outskirts": ((min_x + 3, min_z + 3), (max_x - 3, max_z - 3)),
        }

        zone = bdef.sector or "residential"
        (zx1, zz1), (zx2, zz2) = zone_ranges.get(
            zone, ((min_x + 3, min_z + 3), (max_x - 3, max_z - 3))
        )

        # Shuffle candidate positions within the zone
        candidates = [
            (x, z)
            for x in range(zx1, zx2 - bdef.width + 1)
            for z in range(zz1, zz2 - bdef.height + 1)
        ]
        self.rng.shuffle(candidates)

        for cx, cz in candidates:
            if self._can_place(cx, cz, bdef.width, bdef.height):
                return (cx, cz)

        # Fallback: try anywhere
        all_candidates = [
            (x, z)
            for x in range(min_x + 2, max_x - bdef.width - 1)
            for z in range(min_z + 2, max_z - bdef.height - 1)
        ]
        self.rng.shuffle(all_candidates)
        for cx, cz in all_candidates:
            if self._can_place(cx, cz, bdef.width, bdef.height):
                return (cx, cz)

        return None

    def _can_place(self, x: int, z: int, w: int, h: int) -> bool:
        """Check if a w×h rectangle at (x, z) is clear for building.

        Enforces a 2-tile buffer around the building footprint so NPCs
        always have room to walk between buildings and doors stay accessible.
        """
        buffer = 2
        for dx in range(-buffer, w + buffer):
            for dz in range(-buffer, h + buffer):
                tile = self.grid.get_tile(x + dx, z + dz)
                if tile is None:
                    return False
                # Check buffer zone — must be passable, not water
                is_interior = 0 <= dx < w and 0 <= dz < h
                if not is_interior:
                    if tile.terrain == Terrain.WATER:
                        return False
                    # Buffer tiles must not contain buildings
                    if any(o.object_type == "building" for o in tile.objects):
                        return False
                    continue
                # Interior must be grass/dirt and have no objects
                if tile.terrain not in (Terrain.GRASS, Terrain.DIRT):
                    return False
                if tile.objects:
                    return False
        return True

    def _stamp_building(self, bdef: BuildingDef, x: int, z: int) -> None:
        """Place a building's footprint on the grid."""
        instance_num = sum(
            1 for b in self.buildings if b.building_type == bdef.building_type
        ) + 1
        instance_name = (
            bdef.name if bdef.max_count == 1
            else f"{bdef.name} {instance_num}"
        )
        arena_name = f"{bdef.building_type}_{instance_num}"

        obj = WorldObject(
            object_id=arena_name,
            object_type="building",
            name=instance_name,
            walkable=False,
            metadata={"width": bdef.width, "height": bdef.height},
        )

        # Door is on the south face, centre — ON the building's last row
        door_x = x + bdef.width // 2
        door_z = z + bdef.height - 1

        for dx in range(bdef.width):
            for dz in range(bdef.height):
                tx, tz = x + dx, z + dz
                tile = self.grid.get_tile(tx, tz)
                if tile is None:
                    continue
                tile.terrain = Terrain.STONE if bdef.building_type == "church" else Terrain.DIRT
                tile.walkable = False
                self.grid.set_sector(tx, tz, bdef.sector or "residential", arena_name)
                # Only place the object on the first tile (top-left corner)
                if dx == 0 and dz == 0:
                    self.grid.place_object(tx, tz, obj)

        # Make the door tile walkable (entrance into building)
        door_tile = self.grid.get_tile(door_x, door_z)
        if door_tile is not None:
            door_tile.walkable = True
            door_tile.terrain = Terrain.ROAD

        # Ensure the approach tile (one south of door) is also passable
        approach_tile = self.grid.get_tile(door_x, door_z + 1)
        if approach_tile is not None:
            approach_tile.walkable = True
            if approach_tile.terrain == Terrain.GRASS:
                approach_tile.terrain = Terrain.ROAD

        placed = PlacedBuilding(
            building_type=bdef.building_type,
            name=instance_name,
            x=x, z=z,
            width=bdef.width, height=bdef.height,
            sector=bdef.sector or "residential",
            door_x=door_x, door_z=door_z,
            interior_objects=list(BUILDING_OBJECTS.get(bdef.building_type, [])),
        )
        self.buildings.append(placed)

    # ---------- Roads ----------

    def _connect_roads(self) -> None:
        """Build roads connecting all building doors to the town centre."""
        centre = (0, 0)

        for building in self.buildings:
            # Connect from the approach tile (one south of door) to town centre.
            # The door itself is on the building's south wall; the approach tile
            # is the first walkable tile outside the building.
            approach_z = building.door_z + 1
            self._lay_road(building.door_x, approach_z, centre[0], centre[1])

    def _lay_road(self, x1: int, z1: int, x2: int, z2: int) -> None:
        """Lay an L-shaped road between two points."""
        # Horizontal first, then vertical
        step_x = 1 if x2 >= x1 else -1
        step_z = 1 if z2 >= z1 else -1

        x = x1
        while x != x2:
            self._set_road_tile(x, z1)
            x += step_x
        z = z1
        while z != z2:
            self._set_road_tile(x2, z)
            z += step_z
        self._set_road_tile(x2, z2)

    def _set_road_tile(self, x: int, z: int) -> None:
        """Set a tile as road if it's currently passable and not a building interior."""
        tile = self.grid.get_tile(x, z)
        if tile is None:
            return
        # Never overwrite impassable tiles (building interiors, stone, etc.)
        if not tile.walkable:
            return
        if tile.terrain in (Terrain.GRASS, Terrain.DIRT, Terrain.SAND, Terrain.ROAD):
            tile.terrain = Terrain.ROAD
            tile.walkable = True
            self._road_tiles.add((x, z))

    # ---------- Resource Nodes ----------

    def _place_resource_nodes(self) -> None:
        """Scatter resource nodes (trees, mines, fields) in outskirts."""
        min_x, min_z, max_x, max_z = self.grid.bounds
        resources = self._resource_types_for_economy()

        for res_type, count, terrain_pref in resources:
            placed = 0
            attempts = 0
            while placed < count and attempts < count * 20:
                attempts += 1
                x = self.rng.randint(min_x + 2, max_x - 2)
                z = self.rng.randint(min_z + 2, max_z - 2)
                tile = self.grid.get_tile(x, z)
                if tile is None:
                    continue
                if not tile.is_passable:
                    continue
                if tile.objects:
                    continue
                # Prefer outskirts
                if abs(x) < 8 and abs(z) < 8:
                    continue

                obj = WorldObject(
                    object_id=f"{res_type}_{placed + 1}",
                    object_type="resource",
                    name=res_type.replace("_", " ").title(),
                    walkable=True,
                    metadata={"resource": res_type, "yield": self.rng.randint(3, 10)},
                )
                self.grid.place_object(x, z, obj)
                self.grid.set_sector(x, z, "outskirts")
                placed += 1

    def _resource_types_for_economy(self) -> list[tuple[str, int, Terrain]]:
        """Return (resource_type, count, preferred_terrain) for the economy."""
        base = [
            ("oak_tree", 15, Terrain.GRASS),
            ("berry_bush", 6, Terrain.GRASS),
        ]
        economy = self.config.economy
        if economy in ("farming", "mixed"):
            base.append(("wheat_field", 8, Terrain.GRASS))
        if economy in ("mining", "mixed"):
            base.append(("iron_deposit", 5, Terrain.STONE))
            base.append(("stone_quarry", 3, Terrain.STONE))
        if economy == "trading":
            base.append(("trade_post", 2, Terrain.ROAD))
        return base

    # ---------- Home Assignment ----------

    def _assign_homes(self) -> None:
        """Prepare home buildings for NPC assignment (stored in metadata)."""
        homes = [b for b in self.buildings if b.building_type == "home"]
        for i, home in enumerate(homes):
            # Tag with occupant slot — NPC system will fill these
            tile = self.grid.get_tile(home.x, home.z)
            if tile and tile.objects:
                tile.objects[0].metadata["home_slot"] = i
                tile.objects[0].metadata["occupant"] = None


def generate_world(config: WorldConfig | None = None) -> tuple[Grid, list[PlacedBuilding]]:
    """
    Convenience function: generate a complete world from config.

    Returns (grid, buildings) tuple.
    """
    if config is None:
        config = WorldConfig()
    gen = TownGenerator(config)
    grid = gen.generate()
    return grid, gen.buildings
