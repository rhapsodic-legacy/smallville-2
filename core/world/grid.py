"""
Spatial grid system for the Smallville world.

Provides a tile-based grid with terrain types, collision, object placement,
and hierarchical addressing (world:sector:arena:object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class Terrain(Enum):
    """Terrain types that affect movement, building, and resource gathering."""
    GRASS = "grass"
    DIRT = "dirt"
    ROAD = "road"
    WATER = "water"
    STONE = "stone"
    SAND = "sand"
    FOREST = "forest"


@dataclass
class WorldObject:
    """An object placed on a tile (building, resource node, furniture, etc.)."""
    object_id: str
    object_type: str       # e.g. "building", "resource", "furniture"
    name: str              # e.g. "Tavern", "Oak Tree"
    walkable: bool = True  # can NPCs walk through/on this?
    metadata: dict = field(default_factory=dict)


@dataclass
class Tile:
    """A single cell in the world grid."""
    x: int
    z: int
    terrain: Terrain = Terrain.GRASS
    walkable: bool = True
    objects: list[WorldObject] = field(default_factory=list)
    sector: str = ""   # hierarchical zone name
    arena: str = ""    # sub-zone (e.g. "blacksmith_shop")
    elevation: float = 0.0
    interior: bool = False  # building interior — passable but not routable

    @property
    def address(self) -> str:
        """Hierarchical address: world:sector:arena or world:sector."""
        parts = ["smallville"]
        if self.sector:
            parts.append(self.sector)
        if self.arena:
            parts.append(self.arena)
        return ":".join(parts)

    @property
    def is_passable(self) -> bool:
        """Can an NPC walk on this tile?"""
        if not self.walkable:
            return False
        if self.terrain == Terrain.WATER:
            return False
        return all(obj.walkable for obj in self.objects)


class Grid:
    """
    2D spatial grid representing the game world.

    Coordinate system matches Three.js: x is east/west, z is north/south.
    Origin (0, 0) is the centre of the map.
    """

    def __init__(self, width: int, height: int):
        if width < 1 or height < 1:
            raise ValueError("Grid dimensions must be positive")
        self.width = width
        self.height = height
        # Offset so (0,0) is near centre
        self._x_offset = width // 2
        self._z_offset = height // 2
        self._tiles: dict[tuple[int, int], Tile] = {}
        self._init_tiles()

    def _init_tiles(self) -> None:
        """Create all tiles with default terrain."""
        for x in range(self.width):
            for z in range(self.height):
                gx = x - self._x_offset
                gz = z - self._z_offset
                self._tiles[(gx, gz)] = Tile(x=gx, z=gz)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return (min_x, min_z, max_x, max_z) inclusive."""
        min_x = -self._x_offset
        min_z = -self._z_offset
        max_x = self.width - self._x_offset - 1
        max_z = self.height - self._z_offset - 1
        return (min_x, min_z, max_x, max_z)

    def in_bounds(self, x: int, z: int) -> bool:
        return (x, z) in self._tiles

    def get_tile(self, x: int, z: int) -> Tile | None:
        return self._tiles.get((x, z))

    def set_terrain(self, x: int, z: int, terrain: Terrain) -> None:
        tile = self.get_tile(x, z)
        if tile is None:
            raise ValueError(f"Tile ({x}, {z}) out of bounds")
        tile.terrain = terrain
        if terrain == Terrain.WATER:
            tile.walkable = False

    def place_object(self, x: int, z: int, obj: WorldObject) -> None:
        tile = self.get_tile(x, z)
        if tile is None:
            raise ValueError(f"Tile ({x}, {z}) out of bounds")
        tile.objects.append(obj)

    def set_sector(self, x: int, z: int, sector: str, arena: str = "") -> None:
        tile = self.get_tile(x, z)
        if tile is None:
            raise ValueError(f"Tile ({x}, {z}) out of bounds")
        tile.sector = sector
        tile.arena = arena

    def get_neighbours(self, x: int, z: int, diagonal: bool = False) -> list[Tile]:
        """Return adjacent tiles (4-directional, optionally 8-directional)."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        result = []
        for dx, dz in directions:
            tile = self.get_tile(x + dx, z + dz)
            if tile is not None:
                result.append(tile)
        return result

    def get_passable_neighbours(
        self, x: int, z: int, diagonal: bool = False
    ) -> list[Tile]:
        """Return adjacent tiles that NPCs can walk on."""
        return [t for t in self.get_neighbours(x, z, diagonal) if t.is_passable]

    def tiles_in_radius(self, cx: int, cz: int, radius: int) -> list[Tile]:
        """Return all tiles within Manhattan distance of (cx, cz)."""
        result = []
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if abs(dx) + abs(dz) <= radius:
                    tile = self.get_tile(cx + dx, cz + dz)
                    if tile is not None:
                        result.append(tile)
        return result

    def tiles_in_rect(
        self, x1: int, z1: int, x2: int, z2: int
    ) -> list[Tile]:
        """Return all tiles in a rectangular region (inclusive)."""
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_z, max_z = min(z1, z2), max(z1, z2)
        result = []
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                tile = self.get_tile(x, z)
                if tile is not None:
                    result.append(tile)
        return result

    def find_tiles_by_sector(self, sector: str) -> list[Tile]:
        """Return all tiles in a given sector."""
        return [t for t in self._tiles.values() if t.sector == sector]

    def find_tiles_by_terrain(self, terrain: Terrain) -> list[Tile]:
        return [t for t in self._tiles.values() if t.terrain == terrain]

    def __iter__(self) -> Iterator[Tile]:
        return iter(self._tiles.values())

    def to_dict(self) -> dict:
        """Serialise grid for WebSocket transmission."""
        tiles = []
        for tile in self._tiles.values():
            tile_data = {
                "x": tile.x,
                "z": tile.z,
                "terrain": tile.terrain.value,
                "walkable": tile.is_passable,
                "sector": tile.sector,
                "arena": tile.arena,
                "elevation": tile.elevation,
                "objects": [
                    {
                        "object_id": obj.object_id,
                        "object_type": obj.object_type,
                        "name": obj.name,
                        "metadata": obj.metadata,
                    }
                    for obj in tile.objects
                ],
            }
            tiles.append(tile_data)
        return {
            "width": self.width,
            "height": self.height,
            "tiles": tiles,
        }
