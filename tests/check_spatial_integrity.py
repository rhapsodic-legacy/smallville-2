#!/usr/bin/env python3
"""
Spatial integrity check — runs as a standalone script or via hook.

Verifies that spatial awareness code hasn't regressed by running
the core invariant tests. Returns exit code 1 on failure.

Usage:
  python3 tests/check_spatial_integrity.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.world.grid import Grid
from core.world.spatial_awareness import (
    get_occupied_tiles, find_rest_tile, find_conversation_positions,
    resolve_overlaps,
)
from core.npc.models import NPC, PersonalityTraits, ActivityState

import random


def _make_npc(npc_id: str, x: int, z: int) -> NPC:
    return NPC(
        npc_id=npc_id, name=npc_id, age=30,
        personality=PersonalityTraits(), backstory="", occupation="labourer",
        x=x, z=z, home_x=x, home_z=z,
    )


def check_no_resting_overlaps(npcs: list[NPC]) -> list[str]:
    """Return list of violation descriptions (empty = pass)."""
    violations = []
    occupied: dict[tuple[int, int], str] = {}
    for npc in npcs:
        if npc.activity == ActivityState.WALKING:
            continue
        pos = (npc.x, npc.z)
        existing = occupied.get(pos)
        if existing:
            violations.append(
                f"OVERLAP: {npc.npc_id} and {existing} both resting at {pos}"
            )
        occupied[pos] = npc.npc_id
    return violations


def run_checks() -> bool:
    """Run all spatial integrity checks. Returns True if all pass."""
    passed = True
    rng = random.Random(42)
    grid = Grid(40, 40)

    # Test 1: find_rest_tile never returns occupied
    print("  [1/4] find_rest_tile avoids occupied tiles...", end=" ")
    for _ in range(100):
        occupied = {(rng.randint(-10, 10), rng.randint(-10, 10)) for _ in range(15)}
        pos = find_rest_tile(0, 0, grid, occupied)
        if pos in occupied:
            print(f"FAIL — returned occupied tile {pos}")
            passed = False
            break
    else:
        print("OK")

    # Test 2: conversation positions are adjacent
    print("  [2/4] find_conversation_positions returns adjacent tiles...", end=" ")
    for _ in range(50):
        a = _make_npc("a", rng.randint(-5, 5), rng.randint(-5, 5))
        b = _make_npc("b", rng.randint(-5, 5), rng.randint(-5, 5))
        pa, pb = find_conversation_positions(a, b, grid, set())
        dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
        if dist != 1:
            print(f"FAIL — positions {pa} and {pb} have distance {dist}")
            passed = False
            break
    else:
        print("OK")

    # Test 3: resolve_overlaps fixes violations
    print("  [3/4] resolve_overlaps fixes stacked NPCs...", end=" ")
    npcs = [_make_npc(f"npc_{i}", 0, 0) for i in range(10)]
    resolve_overlaps(npcs, grid)
    violations = check_no_resting_overlaps(npcs)
    if violations:
        print(f"FAIL — {len(violations)} overlaps remain")
        for v in violations:
            print(f"    {v}")
        passed = False
    else:
        print("OK")

    # Test 4: 20-NPC simulation stress test
    print("  [4/4] 20-NPC simulation (100 steps)...", end=" ")
    npcs = [_make_npc(f"sim_{i}", rng.randint(-5, 5), rng.randint(-5, 5))
            for i in range(20)]
    resolve_overlaps(npcs, grid)
    for step in range(100):
        npc = rng.choice(npcs)
        occupied = get_occupied_tiles(npcs)
        safe = find_rest_tile(
            rng.randint(-3, 3), rng.randint(-3, 3), grid, occupied,
            exclude_npc_id=npc.npc_id, npcs=npcs,
        )
        npc.x, npc.z = safe
        resolve_overlaps(npcs, grid)
        violations = check_no_resting_overlaps(npcs)
        if violations:
            print(f"FAIL at step {step} — {len(violations)} overlaps")
            for v in violations:
                print(f"    {v}")
            passed = False
            break
    else:
        print("OK")

    return passed


if __name__ == "__main__":
    print("Spatial integrity check:")
    if run_checks():
        print("All spatial checks PASSED")
        sys.exit(0)
    else:
        print("Spatial checks FAILED", file=sys.stderr)
        sys.exit(1)
