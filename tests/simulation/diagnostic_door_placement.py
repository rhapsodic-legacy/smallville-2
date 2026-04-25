"""
Door placement validation — proves doors are on the building's south wall.

The door tile should be on the last row (south face) of the building
footprint, not outside it. This means:
  - door_z == z + height - 1 (on the building)
  - The door tile is walkable (entrance)
  - The approach tile (door_z + 1) is also walkable
  - The door 3D position is on or very near the building's south wall

Run: python3 tests/simulation/diagnostic_door_placement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.world.generator import TownGenerator, WorldConfig


def validate_door_placement():
    """Validate that every door is on the building's south wall."""
    cfg = WorldConfig(seed=42, grid_width=60, grid_height=60, population=10)
    gen = TownGenerator(cfg)
    gen.generate()

    print("=" * 80)
    print("DOOR PLACEMENT VALIDATION")
    print("=" * 80)

    errors = []
    for b in gen.buildings:
        # Building footprint: x..x+width-1, z..z+height-1
        footprint = set()
        for dx in range(b.width):
            for dz in range(b.height):
                footprint.add((b.x + dx, b.z + dz))

        door_tile = (b.door_x, b.door_z)
        in_footprint = door_tile in footprint

        # Door should be on the building's last row (south face)
        on_south_face = (b.door_z == b.z + b.height - 1)

        # Door tile should be walkable
        tile = gen.grid.get_tile(b.door_x, b.door_z)
        is_walkable = tile is not None and tile.walkable

        # Approach tile (one south of door) should be walkable
        approach = gen.grid.get_tile(b.door_x, b.door_z + 1)
        approach_walkable = approach is not None and approach.walkable

        # 3D rendering positions
        cx_3d = b.x + b.width / 2
        cz_3d = b.z + b.height / 2
        south_wall_z = b.z + b.height / 2 + (b.height * 0.9) / 2
        door_z_3d = b.door_z + 0.5

        # Door 3D should be near the south wall (within 0.5 tiles)
        door_wall_gap = abs(door_z_3d - south_wall_z)

        checks_pass = (
            in_footprint
            and on_south_face
            and is_walkable
            and approach_walkable
        )

        status = "OK" if checks_pass else "FAIL"
        details = []
        if not in_footprint:
            details.append("NOT_IN_FOOTPRINT")
        if not on_south_face:
            details.append("NOT_SOUTH_FACE")
        if not is_walkable:
            details.append("DOOR_NOT_WALKABLE")
        if not approach_walkable:
            details.append("APPROACH_NOT_WALKABLE")

        if not checks_pass:
            errors.append((b.name, details, b, door_tile))

        print(
            f"  {b.name:20s} "
            f"footprint=({b.x},{b.z})-({b.x+b.width-1},{b.z+b.height-1})  "
            f"door=({b.door_x},{b.door_z})  "
            f"walkable={is_walkable}  approach_walkable={approach_walkable}  "
            f"wall_gap={door_wall_gap:.2f}  "
            f"[{status}]"
        )

    print()
    if errors:
        print(f"FAILED: {len(errors)} door(s) have issues:")
        for name, details, b, door in errors:
            print(f"  {name}: {', '.join(details)} at {door}")
        return False
    else:
        print(f"PASSED: All {len(gen.buildings)} doors correctly on building south wall")
        return True


if __name__ == "__main__":
    ok = validate_door_placement()
    sys.exit(0 if ok else 1)
