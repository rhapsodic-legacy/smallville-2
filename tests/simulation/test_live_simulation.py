"""
Live simulation test — runs the actual game loop and observes NPC behaviour.

Boots the full world, runs NPCManager.tick() for many ticks, watches
every NPC every tick for anomalies AND movement timing patterns:
  - Teleportation (moves > threshold in one tick)
  - Building intrusion (rests on impassable tile)
  - Water entry
  - Permanent stuckness
  - Resting overlap
  - Synchronized departure (too many NPCs start walking on the same tick)
  - Synchronized arrival (too many NPCs stop walking on the same tick)

Run:
    python tests/simulation/test_live_simulation.py
    python -m pytest tests/simulation/test_live_simulation.py -v
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.world.generator import WorldConfig, generate_world
from core.world.grid import Terrain
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager
from core.npc.llm_client import MockProvider
from core.npc.models import ActivityState
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore

logger = logging.getLogger(__name__)

# ---------- Configuration ----------

SIM_TICKS = 600   # ~10 real minutes — tests behaviour well past the 45s mark
TICK_DELTA = 1.0
TELEPORT_THRESHOLD = 5
POPULATION = 10
SEED = 42

# If more than this fraction of NPCs start or stop walking on the same tick,
# flag it as synchronized movement.
SYNC_THRESHOLD = 0.6  # 60% of population


# ---------- Data structures ----------

@dataclass
class Anomaly:
    tick: int
    npc_name: str
    npc_id: str
    category: str
    details: str
    x: float
    z: float


@dataclass
class NPCSnapshot:
    x: float
    z: float
    tile_x: int
    tile_z: int
    activity: str
    path_len: int


@dataclass
class SimulationReport:
    ticks_run: int
    population: int
    anomalies: list[Anomaly] = field(default_factory=list)
    npc_travel_distance: dict[str, float] = field(default_factory=dict)
    npcs_that_moved: int = 0
    max_simultaneous_walkers: int = 0
    total_path_assignments: int = 0

    # Movement timing
    departure_ticks: list[int] = field(default_factory=list)
    arrival_ticks: list[int] = field(default_factory=list)
    sync_departure_events: int = 0
    sync_arrival_events: int = 0
    departure_spread: float = 0.0  # std dev of departure ticks

    @property
    def passed(self) -> bool:
        return len(self.anomalies) == 0

    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  LIVE SIMULATION REPORT")
        lines.append("=" * 70)
        lines.append(f"  Ticks: {self.ticks_run} | Population: {self.population}")
        lines.append(f"  NPCs that moved: {self.npcs_that_moved}/{self.population}")
        lines.append(f"  Max simultaneous walkers: {self.max_simultaneous_walkers}")
        lines.append(f"  Total path assignments: {self.total_path_assignments}")

        if self.npc_travel_distance:
            dists = sorted(self.npc_travel_distance.values(), reverse=True)
            lines.append(f"  Travel distances: max={dists[0]:.1f}, min={dists[-1]:.1f}, "
                         f"avg={sum(dists)/len(dists):.1f}")

        lines.append("")
        lines.append("  --- Movement Timing ---")
        lines.append(f"  Departures: {len(self.departure_ticks)} total")
        lines.append(f"  Arrivals:   {len(self.arrival_ticks)} total")
        lines.append(f"  Departure spread (std dev): {self.departure_spread:.1f} ticks")
        lines.append(f"  Sync departure events (>{SYNC_THRESHOLD*100:.0f}% same tick): "
                     f"{self.sync_departure_events}")
        lines.append(f"  Sync arrival events   (>{SYNC_THRESHOLD*100:.0f}% same tick): "
                     f"{self.sync_arrival_events}")

        # Show per-tick departure histogram for first slot transition
        if self.departure_ticks:
            from collections import Counter
            dep_counts = Counter(self.departure_ticks)
            first_deps = sorted(dep_counts.items())[:20]
            lines.append(f"  Departure histogram (first 20 ticks with departures):")
            for tick, count in first_deps:
                bar = "#" * count
                lines.append(f"    tick {tick:4d}: {bar} ({count})")

        lines.append("")
        lines.append("-" * 70)

        if not self.anomalies:
            lines.append("  NO ANOMALIES DETECTED")
        else:
            by_cat = {}
            for a in self.anomalies:
                by_cat.setdefault(a.category, []).append(a)

            for cat, items in sorted(by_cat.items()):
                lines.append(f"  [{cat.upper()}] — {len(items)} occurrences:")
                for a in items[:5]:
                    lines.append(f"    tick {a.tick}: {a.npc_name} at ({a.x:.1f},{a.z:.1f}) "
                                 f"— {a.details}")
                if len(items) > 5:
                    lines.append(f"    ... and {len(items) - 5} more")

        lines.append("-" * 70)
        status = "ALL CLEAR" if self.passed else f"FAILED — {len(self.anomalies)} anomalies"
        lines.append(f"  {status}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------- Simulation runner ----------

async def run_simulation(
    ticks: int = SIM_TICKS,
    population: int = POPULATION,
    seed: int = SEED,
) -> SimulationReport:
    """Run a headless simulation and observe NPC behaviour + timing."""

    config = WorldConfig(population=population, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    clock = GameClock()

    manager = NPCManager(
        grid=grid, buildings=buildings, llm=llm,
        seed=seed, memory=memory,
    )
    manager.spawn_population(population)

    # Pre-compute tile sets
    impassable_building_tiles = set()
    for b in buildings:
        for dx in range(b.width):
            for dz in range(b.height):
                tile = grid.get_tile(b.x + dx, b.z + dz)
                if tile and not tile.is_passable:
                    impassable_building_tiles.add((b.x + dx, b.z + dz))

    water_tiles = set()
    for tile in grid:
        if tile.terrain == Terrain.WATER:
            water_tiles.add((tile.x, tile.z))

    # Tracking
    report = SimulationReport(ticks_run=ticks, population=len(manager.npcs))
    prev_snapshots: dict[str, NPCSnapshot] = {}
    travel_dist: dict[str, float] = {npc.npc_id: 0.0 for npc in manager.npcs}
    moved_npcs: set[str] = set()
    stuck_counters: dict[str, int] = {}

    # Movement timing tracking
    all_departure_ticks: list[int] = []
    all_arrival_ticks: list[int] = []
    sync_min_count = max(2, int(len(manager.npcs) * SYNC_THRESHOLD))

    # Initial snapshots
    for npc in manager.npcs:
        prev_snapshots[npc.npc_id] = NPCSnapshot(
            x=npc.x, z=npc.z, tile_x=npc.tile_x, tile_z=npc.tile_z,
            activity=npc.activity.value, path_len=len(npc.current_path),
        )

    # Run simulation
    for tick in range(ticks):
        await manager.tick(clock, TICK_DELTA)
        clock.tick(TICK_DELTA)

        walkers = 0
        departures_this_tick = 0
        arrivals_this_tick = 0

        for npc in manager.npcs:
            prev = prev_snapshots.get(npc.npc_id)
            if not prev:
                continue

            # --- Departure detection ---
            if npc.activity == ActivityState.WALKING and prev.activity != "walking":
                departures_this_tick += 1
                all_departure_ticks.append(tick)

            # --- Arrival detection ---
            if npc.activity != ActivityState.WALKING and prev.activity == "walking":
                arrivals_this_tick += 1
                all_arrival_ticks.append(tick)

            # --- Teleportation ---
            dx = abs(npc.x - prev.x)
            dz = abs(npc.z - prev.z)
            dist = dx + dz
            travel_dist[npc.npc_id] += dist

            if dist > TELEPORT_THRESHOLD and prev.activity == "walking":
                report.anomalies.append(Anomaly(
                    tick=tick, npc_name=npc.name, npc_id=npc.npc_id,
                    category="teleport",
                    details=f"moved {dist:.1f} tiles in one tick",
                    x=npc.x, z=npc.z,
                ))

            # --- Building intrusion ---
            if npc.activity != ActivityState.WALKING:
                if (npc.tile_x, npc.tile_z) in impassable_building_tiles:
                    report.anomalies.append(Anomaly(
                        tick=tick, npc_name=npc.name, npc_id=npc.npc_id,
                        category="building_intrusion",
                        details=f"resting inside impassable tile ({npc.tile_x},{npc.tile_z})",
                        x=npc.x, z=npc.z,
                    ))

            # --- Water ---
            if (npc.tile_x, npc.tile_z) in water_tiles:
                report.anomalies.append(Anomaly(
                    tick=tick, npc_name=npc.name, npc_id=npc.npc_id,
                    category="water",
                    details=f"on water tile ({npc.tile_x},{npc.tile_z})",
                    x=npc.x, z=npc.z,
                ))

            # --- Stuck ---
            if npc.activity == ActivityState.WALKING:
                walkers += 1
                prev_idx = stuck_counters.get(npc.npc_id + "_idx", -1)
                if npc.path_index == prev_idx and npc.current_path:
                    stuck_counters[npc.npc_id] = stuck_counters.get(npc.npc_id, 0) + 1
                else:
                    stuck_counters[npc.npc_id] = 0
                stuck_counters[npc.npc_id + "_idx"] = npc.path_index

                if stuck_counters.get(npc.npc_id, 0) >= 10:
                    report.anomalies.append(Anomaly(
                        tick=tick, npc_name=npc.name, npc_id=npc.npc_id,
                        category="stuck",
                        details=f"no path progress for 10+ ticks at ({npc.tile_x},{npc.tile_z})",
                        x=npc.x, z=npc.z,
                    ))
                    stuck_counters[npc.npc_id] = 0
            else:
                stuck_counters[npc.npc_id] = 0

            # --- Track movement ---
            if dist > 0.1:
                moved_npcs.add(npc.npc_id)
            if npc.current_path and not prev.path_len:
                report.total_path_assignments += 1

            prev_snapshots[npc.npc_id] = NPCSnapshot(
                x=npc.x, z=npc.z, tile_x=npc.tile_x, tile_z=npc.tile_z,
                activity=npc.activity.value, path_len=len(npc.current_path),
            )

        report.max_simultaneous_walkers = max(report.max_simultaneous_walkers, walkers)

        # --- Synchronized movement detection ---
        if departures_this_tick >= sync_min_count:
            report.sync_departure_events += 1
            report.anomalies.append(Anomaly(
                tick=tick, npc_name="(multiple)", npc_id="",
                category="sync_departure",
                details=f"{departures_this_tick}/{len(manager.npcs)} NPCs started walking "
                        f"on the same tick",
                x=0, z=0,
            ))

        if arrivals_this_tick >= sync_min_count:
            report.sync_arrival_events += 1
            report.anomalies.append(Anomaly(
                tick=tick, npc_name="(multiple)", npc_id="",
                category="sync_arrival",
                details=f"{arrivals_this_tick}/{len(manager.npcs)} NPCs stopped walking "
                        f"on the same tick",
                x=0, z=0,
            ))

        # --- Resting overlaps ---
        resting_positions: dict[tuple[int, int], str] = {}
        for npc in manager.npcs:
            if npc.activity == ActivityState.WALKING:
                continue
            pos = (npc.tile_x, npc.tile_z)
            if pos in resting_positions:
                report.anomalies.append(Anomaly(
                    tick=tick, npc_name=npc.name, npc_id=npc.npc_id,
                    category="overlap",
                    details=f"same tile as {resting_positions[pos]} at {pos}",
                    x=npc.x, z=npc.z,
                ))
            resting_positions[pos] = npc.name

    # Compute timing stats
    report.npcs_that_moved = len(moved_npcs)
    report.npc_travel_distance = travel_dist
    report.departure_ticks = all_departure_ticks
    report.arrival_ticks = all_arrival_ticks

    if all_departure_ticks:
        mean = sum(all_departure_ticks) / len(all_departure_ticks)
        variance = sum((t - mean) ** 2 for t in all_departure_ticks) / len(all_departure_ticks)
        report.departure_spread = variance ** 0.5

    return report


# ---------- pytest integration ----------

import pytest

@pytest.mark.simulation
def test_live_simulation():
    """Run live simulation — no anomalies, NPCs actually move."""
    report = asyncio.run(run_simulation(ticks=SIM_TICKS))
    print(report.summary())
    assert report.passed, (
        f"{len(report.anomalies)} anomalies detected — see report"
    )
    assert report.npcs_that_moved > 0, "No NPCs moved"


@pytest.mark.simulation
def test_departure_spread():
    """Departures must be spread over real time, not clustered."""
    report = asyncio.run(run_simulation(ticks=SIM_TICKS))
    if not report.departure_ticks:
        return  # no departures = nothing to check
    from collections import Counter
    dep_counts = Counter(report.departure_ticks)
    # No single tick should have >30% of population departing
    max_on_one_tick = max(dep_counts.values())
    threshold = max(2, int(report.population * 0.3))
    assert max_on_one_tick <= threshold, (
        f"Too many NPCs departed on one tick: {max_on_one_tick} "
        f"(threshold: {threshold})"
    )


@pytest.mark.simulation
def test_high_population():
    """Stress test with 20 NPCs."""
    report = asyncio.run(run_simulation(ticks=100, population=20, seed=99))
    print(report.summary())
    assert report.passed, f"{len(report.anomalies)} anomalies"


# ---------- Standalone ----------

if __name__ == "__main__":
    report = asyncio.run(run_simulation())
    print(report.summary())
    sys.exit(0 if report.passed else 1)
