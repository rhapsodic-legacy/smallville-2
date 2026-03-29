"""World module — spatial grid, procedural town generator, pathfinding, spatial awareness."""

from core.world.grid import Grid, Tile, Terrain, WorldObject
from core.world.pathfinding import find_path

__all__ = [
    "Grid", "Tile", "Terrain", "WorldObject", "find_path",
]

# spatial_awareness is imported directly where needed to avoid
# circular imports (it references core.npc.models.ActivityState).
# Import as: from core.world.spatial_awareness import ...
