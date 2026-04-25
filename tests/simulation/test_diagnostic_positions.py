"""
Diagnostic position test — captures raw NPC state data.

Instead of asserting pass/fail, this test captures and prints
detailed per-tick data so we can see EXACTLY what's happening:

1. Where is each NPC standing (tile coords)?
2. Is that tile passable? Inside a building? A door tile?
3. What's the NPC's activity, path state, and action description?
4. Is the NPC overlapping with another resting NPC?
5. When do NPCs change position and why?

Run: python3 tests/simulation/test_diagnostic_positions.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock
from core.world.generator import WorldConfig, generate_world


# ---------- Data capture ----------

@dataclass
class TickSnapshot:
    tick: int
    game_time: str
    slot: str
    npcs: list[dict] = field(default_factory=list)


@dataclass
class NPCDiagnostic:
    """Per-NPC diagnostic data accumulated across the sim."""
    npc_id: str
    name: str
    occupation: str
    home_x: int
    home_z: int
    work_x: int
    work_z: int

    # Position timeline: (tick, x, z)
    positions: list[tuple[int, int, int]] = field(default_factory=list)

    # Every position change with reason
    position_changes: list[dict] = field(default_factory=list)

    # Ticks spent on impassable tiles
    impassable_ticks: int = 0

    # Ticks spent inside a building (non-door building tile)
    inside_building_ticks: int = 0

    # Ticks overlapping with another resting NPC
    overlap_ticks: int = 0

    # Reversal count (A→B→A patterns)
    reversal_count: int = 0

    # Activity breakdown
    activity_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def run_diagnostic(days: int = 2, population: int = 7, seed: int = 99):
    """Run sim and capture detailed position data for every NPC."""
    config = WorldConfig(population=population, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm,
        seed=seed, memory=memory,
    )
    mgr.spawn_population(population)

    total_ticks = days * 1200
    clock = GameClock()

    # Build building tile lookup: (x, z) → building info
    building_tiles: dict[tuple[int, int], dict] = {}
    door_tiles: set[tuple[int, int]] = set()
    for b in buildings:
        door_tiles.add((b.door_x, b.door_z))
        for dx in range(b.width):
            for dz in range(b.height):
                tx, tz = b.x + dx, b.z + dz
                is_door = (tx == b.door_x and tz == b.door_z)
                building_tiles[(tx, tz)] = {
                    "building": b.name or b.building_type,
                    "is_door": is_door,
                }

    # Init diagnostics
    diagnostics: dict[str, NPCDiagnostic] = {}
    for npc in mgr.npcs:
        diagnostics[npc.npc_id] = NPCDiagnostic(
            npc_id=npc.npc_id, name=npc.name,
            occupation=npc.occupation,
            home_x=npc.home_x, home_z=npc.home_z,
            work_x=npc.work_x, work_z=npc.work_z,
        )

    # Track previous positions for change detection
    prev_positions: dict[str, tuple[int, int]] = {}
    prev_activities: dict[str, str] = {}

    async def _run():
        for tick in range(total_ticks):
            clock.tick(1.0)
            await mgr.tick(clock, 1.0)

            slot = clock.schedule_slot.value

            # Build resting position map for overlap detection
            resting_at: dict[tuple[int, int], list[str]] = defaultdict(list)
            for npc in mgr.npcs:
                if npc.activity != ActivityState.WALKING:
                    resting_at[(npc.tile_x, npc.tile_z)].append(npc.npc_id)

            for npc in mgr.npcs:
                d = diagnostics[npc.npc_id]
                tx, tz = npc.tile_x, npc.tile_z
                pos = (tx, tz)
                activity = npc.activity.value

                d.positions.append((tick, tx, tz))
                d.activity_counts[activity] += 1

                # Check tile properties
                tile = grid.get_tile(tx, tz)
                is_passable = tile.is_passable if tile else False
                in_building = pos in building_tiles
                is_door = pos in door_tiles

                if not is_passable:
                    d.impassable_ticks += 1

                if in_building and not is_door:
                    d.inside_building_ticks += 1

                # Overlap detection (resting NPCs only)
                if activity != "walking":
                    others = resting_at.get(pos, [])
                    if len(others) > 1:
                        d.overlap_ticks += 1

                # Position change detection
                prev = prev_positions.get(npc.npc_id)
                prev_act = prev_activities.get(npc.npc_id, "")
                if prev and prev != pos:
                    bld_info = building_tiles.get(pos, {})
                    d.position_changes.append({
                        "tick": tick,
                        "slot": slot,
                        "from": prev,
                        "to": pos,
                        "activity": activity,
                        "prev_activity": prev_act,
                        "has_path": bool(npc.current_path),
                        "path_len": len(npc.current_path),
                        "path_idx": npc.path_index,
                        "action_desc": npc.current_action_description,
                        "on_building": bld_info.get("building", ""),
                        "is_door": bld_info.get("is_door", False),
                        "is_passable": is_passable,
                        "convo_partner": npc.conversation_partner,
                    })

                prev_positions[npc.npc_id] = pos
                prev_activities[npc.npc_id] = activity

            # Detect reversals in a window
            if tick > 0 and tick % 100 == 0:
                for npc_id, d in diagnostics.items():
                    recent = d.positions[-100:]
                    positions = [(x, z) for _, x, z in recent]
                    for i in range(2, len(positions)):
                        if (positions[i] == positions[i - 2]
                                and positions[i] != positions[i - 1]):
                            d.reversal_count += 1

    asyncio.new_event_loop().run_until_complete(_run())

    return diagnostics, buildings, grid


def print_report(diagnostics, buildings, grid):
    """Print the full diagnostic report."""
    total_ticks = max(
        len(d.positions) for d in diagnostics.values()
    )

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC POSITION REPORT — {total_ticks} ticks")
    print(f"{'='*70}")

    # Building summary
    print(f"\n--- Buildings ({len(buildings)}) ---")
    for b in buildings:
        print(f"  {b.name or b.building_type:15s} "
              f"origin=({b.x},{b.z}) size={b.width}x{b.height} "
              f"door=({b.door_x},{b.door_z})")

    # Per-NPC summary
    print(f"\n--- NPC Summaries ---")
    for npc_id, d in sorted(diagnostics.items()):
        total = sum(d.activity_counts.values())
        pcts = {k: f"{v/total*100:.0f}%" for k, v in d.activity_counts.items()}

        print(f"\n  {d.name} ({d.occupation}) [{npc_id}]")
        print(f"    Home: ({d.home_x},{d.home_z})  Work: ({d.work_x},{d.work_z})")
        print(f"    Activities: {dict(pcts)}")
        print(f"    Position changes: {len(d.position_changes)}")
        print(f"    Impassable ticks: {d.impassable_ticks}")
        print(f"    Inside building (non-door): {d.inside_building_ticks}")
        print(f"    Overlap ticks: {d.overlap_ticks}")
        print(f"    Reversals (A→B→A): {d.reversal_count}")

        # Show the most suspicious position changes
        if d.impassable_ticks > 0 or d.inside_building_ticks > 0:
            print(f"    !! INSIDE BUILDING or IMPASSABLE — sample changes:")
            suspicious = [
                c for c in d.position_changes
                if not c["is_passable"] or c["on_building"]
            ]
            for c in suspicious[:5]:
                print(f"       tick={c['tick']} slot={c['slot']} "
                      f"{c['from']}→{c['to']} act={c['activity']} "
                      f"path={c['has_path']}(len={c['path_len']}) "
                      f"bld={c['on_building']} door={c['is_door']} "
                      f"passable={c['is_passable']} "
                      f"desc=\"{c['action_desc']}\"")

        if d.reversal_count > 3:
            print(f"    !! OSCILLATION — sample reversals:")
            # Find reversal sequences
            positions = [(x, z) for _, x, z in d.positions]
            shown = 0
            for i in range(2, len(positions)):
                if (positions[i] == positions[i - 2]
                        and positions[i] != positions[i - 1]):
                    tick_num = d.positions[i][0]
                    # Find the matching position change entry
                    matching = [
                        c for c in d.position_changes
                        if c["tick"] == tick_num
                    ]
                    if matching:
                        c = matching[0]
                        print(f"       tick={c['tick']} {c['from']}→{c['to']} "
                              f"act={c['activity']} path={c['has_path']} "
                              f"desc=\"{c['action_desc']}\" "
                              f"convo={c['convo_partner']}")
                    shown += 1
                    if shown >= 8:
                        break

        if d.overlap_ticks > 10:
            print(f"    !! OVERLAPS — sample:")
            overlap_changes = [
                c for c in d.position_changes
                if c["activity"] != "walking"
            ]
            for c in overlap_changes[:5]:
                print(f"       tick={c['tick']} at {c['to']} "
                      f"act={c['activity']} desc=\"{c['action_desc']}\"")

    # Aggregate issues
    print(f"\n--- Aggregate Issues ---")
    total_impassable = sum(d.impassable_ticks for d in diagnostics.values())
    total_inside = sum(d.inside_building_ticks for d in diagnostics.values())
    total_overlaps = sum(d.overlap_ticks for d in diagnostics.values())
    total_reversals = sum(d.reversal_count for d in diagnostics.values())

    print(f"  Total impassable ticks:       {total_impassable}")
    print(f"  Total inside-building ticks:  {total_inside}")
    print(f"  Total overlap ticks:          {total_overlaps}")
    print(f"  Total reversals:              {total_reversals}")

    # Home/work tile passability check
    print(f"\n--- Home/Work Tile Passability ---")
    for npc_id, d in sorted(diagnostics.items()):
        home_tile = grid.get_tile(d.home_x, d.home_z)
        work_tile = grid.get_tile(d.work_x, d.work_z)
        home_passable = home_tile.is_passable if home_tile else False
        work_passable = work_tile.is_passable if work_tile else False
        home_in_bld = (d.home_x, d.home_z) in {
            (b.door_x, b.door_z) for b in buildings
        }
        work_in_bld = (d.work_x, d.work_z) in {
            (b.door_x, b.door_z) for b in buildings
        }

        issues = []
        if not home_passable:
            issues.append("HOME IMPASSABLE")
        if not work_passable:
            issues.append("WORK IMPASSABLE")
        if not home_in_bld:
            # Check if home is a building tile at all
            if (d.home_x, d.home_z) in {
                (b.x + dx, b.z + dz)
                for b in buildings
                for dx in range(b.width)
                for dz in range(b.height)
            }:
                issues.append("HOME IS NON-DOOR BUILDING TILE")
        if not work_in_bld:
            if (d.work_x, d.work_z) in {
                (b.x + dx, b.z + dz)
                for b in buildings
                for dx in range(b.width)
                for dz in range(b.height)
            }:
                issues.append("WORK IS NON-DOOR BUILDING TILE")

        status = " | ".join(issues) if issues else "OK"
        print(f"  {d.name:10s} home=({d.home_x},{d.home_z}) "
              f"pass={home_passable} door={home_in_bld}  "
              f"work=({d.work_x},{d.work_z}) "
              f"pass={work_passable} door={work_in_bld}  {status}")

    # Door tile analysis
    print(f"\n--- Door Tile Passability ---")
    for b in buildings:
        door_tile = grid.get_tile(b.door_x, b.door_z)
        approach_tile = grid.get_tile(b.door_x, b.door_z + 1)
        door_pass = door_tile.is_passable if door_tile else False
        approach_pass = approach_tile.is_passable if approach_tile else False
        issues = []
        if not door_pass:
            issues.append("DOOR IMPASSABLE")
        if not approach_pass:
            issues.append("APPROACH IMPASSABLE")
        status = " | ".join(issues) if issues else "OK"
        print(f"  {b.name or b.building_type:15s} "
              f"door=({b.door_x},{b.door_z}) pass={door_pass}  "
              f"approach=({b.door_x},{b.door_z+1}) pass={approach_pass}  "
              f"{status}")

    # Unique rest positions per NPC — where do they actually stop?
    print(f"\n--- Rest Positions (where NPCs stop when not walking) ---")
    for npc_id, d in sorted(diagnostics.items()):
        rest_positions: dict[tuple[int, int], int] = defaultdict(int)
        for change in d.position_changes:
            if change["activity"] != "walking":
                rest_positions[change["to"]] += 1
        # Also count final positions from timeline
        for _, x, z in d.positions:
            pos = (x, z)
            # We already counted changes, just show unique rests
        if rest_positions:
            top = sorted(rest_positions.items(), key=lambda x: -x[1])[:5]
            parts = [f"({x},{z}):{count}" for (x, z), count in top]
            print(f"  {d.name:10s}: {', '.join(parts)}")


# ---------- Pytest wrapper ----------

import pytest


@pytest.fixture(scope="module")
def diagnostic_data():
    return run_diagnostic(days=2, population=7, seed=99)


class TestDiagnosticPositions:
    """Data-driven tests based on captured diagnostic data."""

    def test_no_impassable_resting(self, diagnostic_data):
        """No NPC should rest on an impassable tile."""
        diagnostics, _, _ = diagnostic_data
        violations = {
            d.name: d.impassable_ticks
            for d in diagnostics.values()
            if d.impassable_ticks > 0
        }
        assert len(violations) == 0, (
            f"NPCs on impassable tiles: {violations}"
        )

    def test_npcs_use_building_interiors(self, diagnostic_data):
        """NPCs should spend time on building interior tiles (home, work)."""
        diagnostics, _, _ = diagnostic_data
        npcs_inside = {
            d.name: d.inside_building_ticks
            for d in diagnostics.values()
            if d.inside_building_ticks > 0
        }
        assert len(npcs_inside) > 0, (
            "No NPCs spent any time inside buildings"
        )

    def test_no_excessive_reversals(self, diagnostic_data):
        """No NPC should have more than 10 reversals across the sim."""
        diagnostics, _, _ = diagnostic_data
        violations = {
            d.name: d.reversal_count
            for d in diagnostics.values()
            if d.reversal_count > 10
        }
        assert len(violations) == 0, (
            f"NPCs oscillating: {violations}"
        )

    def test_no_excessive_overlaps(self, diagnostic_data):
        """No NPC should spend >5% of time overlapping."""
        diagnostics, _, _ = diagnostic_data
        violations = {}
        for d in diagnostics.values():
            total = sum(d.activity_counts.values())
            if total > 0 and d.overlap_ticks / total > 0.05:
                violations[d.name] = f"{d.overlap_ticks}/{total} ({d.overlap_ticks/total:.0%})"
        assert len(violations) == 0, (
            f"NPCs overlapping: {violations}"
        )

    def test_home_tiles_passable(self, diagnostic_data):
        """All NPC home tiles should be passable."""
        diagnostics, buildings, grid = diagnostic_data
        violations = {}
        for d in diagnostics.values():
            tile = grid.get_tile(d.home_x, d.home_z)
            if tile and not tile.is_passable:
                violations[d.name] = f"({d.home_x},{d.home_z})"
        assert len(violations) == 0, (
            f"Impassable home tiles: {violations}"
        )

    def test_work_tiles_passable(self, diagnostic_data):
        """All NPC work tiles should be passable."""
        diagnostics, buildings, grid = diagnostic_data
        violations = {}
        for d in diagnostics.values():
            tile = grid.get_tile(d.work_x, d.work_z)
            if tile and not tile.is_passable:
                violations[d.name] = f"({d.work_x},{d.work_z})"
        assert len(violations) == 0, (
            f"Impassable work tiles: {violations}"
        )

    def test_door_tiles_passable(self, diagnostic_data):
        """All building door tiles should be passable."""
        _, buildings, grid = diagnostic_data
        violations = {}
        for b in buildings:
            tile = grid.get_tile(b.door_x, b.door_z)
            if tile and not tile.is_passable:
                violations[b.name or b.building_type] = (
                    f"({b.door_x},{b.door_z})"
                )
        assert len(violations) == 0, (
            f"Impassable door tiles: {violations}"
        )


# ---------- Standalone ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic Position Test")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--population", type=int, default=7)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    print(f"Running {args.days}-day diagnostic "
          f"({args.population} NPCs, seed={args.seed})...")

    diagnostics, buildings, grid = run_diagnostic(
        days=args.days, population=args.population, seed=args.seed,
    )
    print_report(diagnostics, buildings, grid)
