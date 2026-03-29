"""
Client rendering simulation — simulates EXACTLY what the browser sees.

Runs the server, captures to_dict() each tick (same as WebSocket data),
then simulates 60fps client-side rendering to detect anomalies:
  - TELEPORT: NPC moves >3 tiles in one frame
  - SNAPBACK: NPC reverses direction suddenly
  - BUILDING_CLIP: NPC visual position inside a building tile

Tests the trail-based approach: server sends tiles traversed this tick
(max 2-3 waypoints). Client walks through them. No full path, no sync.

Run: python3 tests/simulation/test_client_rendering.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.world.generator import TownGenerator, WorldConfig
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager

TICKS = 120
TICK_DELTA = 1.0
POPULATION = 10
SEED = 42
CLIENT_FPS = 60
FRAMES_PER_TICK = int(CLIENT_FPS * TICK_DELTA)
FRAME_DT = 1.0 / CLIENT_FPS

# Must match client constants
TELEPORT_DISTANCE = 4.0
ARRIVAL_FACTOR = 0.85

# Anomaly thresholds
FRAME_TELEPORT_THRESHOLD = 3.0
SNAPBACK_DOT_THRESHOLD = -0.3


@dataclass
class AnomalyRecord:
    tick: int
    frame: int
    npc_id: str
    anomaly_type: str
    detail: str


@dataclass
class ClientState:
    """Mirrors the JS NPCRenderer movement state exactly."""
    x: float = 0.0
    z: float = 0.0
    trail: list = field(default_factory=list)
    trail_index: int = 0
    walk_speed: float = 0.0
    prev_x: float = 0.0
    prev_z: float = 0.0
    last_dx: float = 0.0
    last_dz: float = 0.0


def update_from_tick(state: ClientState, data: dict):
    """Mirrors JS _updateMoveState exactly."""
    trail = data.get("trail", [])

    if trail:
        # Compute total trail distance
        total = 0.0
        px, pz = state.x, state.z
        for wp in trail:
            dx = wp[0] - px
            dz = wp[1] - pz
            total += math.sqrt(dx * dx + dz * dz)
            px, pz = wp[0], wp[1]

        if total > TELEPORT_DISTANCE:
            last = trail[-1]
            state.x = last[0]
            state.z = last[1]
            state.trail = []
            state.trail_index = 0
            state.walk_speed = 0
        else:
            state.trail = trail
            state.trail_index = 0
            state.walk_speed = total / (TICK_DELTA * ARRIVAL_FACTOR) if total > 0.01 else 2.0
    else:
        state.trail = []
        state.trail_index = 0
        state.walk_speed = 0
        state.x = data["x"]
        state.z = data["z"]


def simulate_frame(state: ClientState, dt: float):
    """Mirrors JS _walkTrail exactly."""
    state.prev_x = state.x
    state.prev_z = state.z

    if not state.trail or state.trail_index >= len(state.trail):
        return

    remaining = state.walk_speed * dt
    while remaining > 0 and state.trail_index < len(state.trail):
        wp = state.trail[state.trail_index]
        dx = wp[0] - state.x
        dz = wp[1] - state.z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < 0.01:
            state.x = wp[0]
            state.z = wp[1]
            state.trail_index += 1
            continue
        if remaining >= dist:
            state.x = wp[0]
            state.z = wp[1]
            remaining -= dist
            state.trail_index += 1
        else:
            frac = remaining / dist
            state.x += dx * frac
            state.z += dz * frac
            remaining = 0


def check_anomaly(state: ClientState, tick: int, frame: int, npc_id: str) -> AnomalyRecord | None:
    dx = state.x - state.prev_x
    dz = state.z - state.prev_z
    dist = math.sqrt(dx * dx + dz * dz)

    if dist > FRAME_TELEPORT_THRESHOLD:
        return AnomalyRecord(
            tick, frame, npc_id, "TELEPORT",
            f"moved {dist:.2f} tiles in 1 frame "
            f"({state.prev_x:.1f},{state.prev_z:.1f})->({state.x:.1f},{state.z:.1f})")

    if dist > 0.5 and (abs(state.last_dx) > 0.01 or abs(state.last_dz) > 0.01):
        dot = dx * state.last_dx + dz * state.last_dz
        if dot < SNAPBACK_DOT_THRESHOLD:
            return AnomalyRecord(
                tick, frame, npc_id, "SNAPBACK",
                f"reversed direction ({state.prev_x:.1f},{state.prev_z:.1f})->"
                f"({state.x:.1f},{state.z:.1f})")

    if dist > 0.01:
        state.last_dx = dx
        state.last_dz = dz
    return None


async def run_simulation():
    cfg = WorldConfig(seed=SEED, grid_width=60, grid_height=60, population=POPULATION)
    gen = TownGenerator(cfg)
    gen.generate()
    mgr = NPCManager(gen.grid, gen.buildings, seed=SEED)
    mgr.spawn_population(POPULATION)
    clock = GameClock()

    states: dict[str, ClientState] = {}
    anomalies: list[AnomalyRecord] = []

    building_tiles = set()
    for b in gen.buildings:
        for dx in range(b.width):
            for dz in range(b.height):
                building_tiles.add((b.x + dx, b.z + dz))

    print("=" * 100)
    print(f"CLIENT RENDERING SIMULATION (trail-based approach)")
    print(f"  {TICKS} ticks × {FRAMES_PER_TICK} frames/tick = {TICKS * FRAMES_PER_TICK} total frames")
    print(f"  {POPULATION} NPCs, seed={SEED}")
    print("=" * 100)

    for tick in range(TICKS):
        await mgr.tick(clock, TICK_DELTA)
        clock.tick(TICK_DELTA)

        tick_data = [npc.to_dict() for npc in mgr.npcs]

        for data in tick_data:
            nid = data["npc_id"]
            if nid not in states:
                states[nid] = ClientState(
                    x=data["x"], z=data["z"],
                    prev_x=data["x"], prev_z=data["z"],
                )
            update_from_tick(states[nid], data)

        tick_anomalies = 0
        for frame in range(FRAMES_PER_TICK):
            for nid, state in states.items():
                simulate_frame(state, FRAME_DT)
                anom = check_anomaly(state, tick, frame, nid)
                if anom:
                    anomalies.append(anom)
                    tick_anomalies += 1

                tx = round(state.x)
                tz = round(state.z)
                if (tx, tz) in building_tiles:
                    anomalies.append(AnomalyRecord(
                        tick, frame, nid, "BUILDING_CLIP",
                        f"inside building at ({tx},{tz})"))
                    tick_anomalies += 1

        if tick % 20 == 0 or tick_anomalies > 0:
            flag = f" *** {tick_anomalies} ANOMALIES" if tick_anomalies else ""
            print(f"  tick {tick:4d}: anomalies_total={len(anomalies)}{flag}")

    # Results
    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    teleports = sum(1 for a in anomalies if a.anomaly_type == "TELEPORT")
    snapbacks = sum(1 for a in anomalies if a.anomaly_type == "SNAPBACK")
    clips = sum(1 for a in anomalies if a.anomaly_type == "BUILDING_CLIP")

    if not anomalies:
        print("  NO ANOMALIES — rendering is clean")
    else:
        by_type = defaultdict(list)
        for a in anomalies:
            by_type[a.anomaly_type].append(a)
        for atype, records in sorted(by_type.items()):
            print(f"\n  {atype}: {len(records)}")
            for r in records[:3]:
                print(f"    tick={r.tick} frame={r.frame} {r.npc_id}: {r.detail}")
            if len(records) > 3:
                print(f"    ... and {len(records) - 3} more")

    print(f"\n  TELEPORTS:      {teleports}")
    print(f"  SNAP-BACKS:     {snapbacks}")
    print(f"  BUILDING CLIPS: {clips}")
    print(f"\n  {'PASS' if teleports == 0 and snapbacks == 0 else 'FAIL'}")

    return anomalies


if __name__ == "__main__":
    asyncio.run(run_simulation())
