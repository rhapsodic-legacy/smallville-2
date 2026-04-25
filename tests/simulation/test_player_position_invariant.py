"""
Player position invariant — server-side code MUST NEVER write to the
player's (x, z) on its own.

This is a class-of-bugs test. Three separate "avatar moves by itself"
regressions have landed over the past few sessions:

  1. resolve_overlaps nudged the player off NPCs' tiles.
  2. _arrive() inside execute_tick nudged the player when their
     activity was WALKING.
  3. resolve_overlaps' accumulated drift + stray push accumulated
     over hours.

Each fix patched the specific call site. A better invariant is
stronger: for the entire simulation, if the player's NPC record is
otherwise in a realistic "near NPC / mid-chat / walking" state, the
manager's tick loop must not mutate its position.

This test constructs every scenario we've seen, drives the sim
directly (no server), and asserts the player's position is
unchanged after many ticks.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock
from core.world.generator import WorldConfig, generate_world
from core.player.player_agent import PlayerAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_sim():
    config = WorldConfig(population=8, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    llm = MockProvider()
    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=llm, seed=42,
        memory=memory, deterministic=False,
    )
    mgr.spawn_population(config.population)
    clock = GameClock()

    player = PlayerAgent.create(name="Traveller", spawn_x=0.0, spawn_z=0.0)
    player.npc.has_custom_schedule = True
    player.npc.daily_schedule = []
    mgr.npcs.append(player.npc)
    mgr._npc_map[player.npc_id] = player.npc
    mgr.player_agent = player
    return mgr, clock, player, grid, buildings


async def _drive_ticks(mgr, clock, n: int, real_delta: float = 0.25) -> None:
    for _ in range(n):
        clock.tick(real_delta)
        mgr.movement_tick(clock, real_delta)
        await mgr.cognition_tick(clock, real_delta)


async def test_player_stationary_on_own_tile_never_moves() -> None:
    """With no input and no nearby NPCs, the player stays put."""
    mgr, clock, player, _, _ = _make_sim()
    player.npc.x = 10.0
    player.npc.z = 10.0
    player.npc.activity = ActivityState.IDLE

    start = (player.npc.x, player.npc.z)
    await _drive_ticks(mgr, clock, 30)
    assert (player.npc.x, player.npc.z) == start, (
        f"Player moved without input: {start} -> {(player.npc.x, player.npc.z)}"
    )


async def test_player_standing_on_npc_tile_never_gets_nudged() -> None:
    """REGRESSION: resolve_overlaps and _arrive both used to nudge the
    player off NPCs' tiles. The player's position is input-
    authoritative — if they stand on an NPC, the NPC moves aside."""
    mgr, clock, player, _, _ = _make_sim()

    # Pick the first real NPC and place the player exactly on their tile.
    target = next(n for n in mgr.npcs if n.npc_id != "player")
    player.npc.x = float(target.tile_x)
    player.npc.z = float(target.tile_z)
    # A WALKING player was the specific case that triggered _arrive's
    # nudge — exercise it here.
    player.npc.activity = ActivityState.WALKING

    start = (player.npc.x, player.npc.z)
    await _drive_ticks(mgr, clock, 50)

    assert (player.npc.x, player.npc.z) == start, (
        f"Player nudged off NPC tile: {start} -> {(player.npc.x, player.npc.z)} "
        f"(target NPC was at ({target.tile_x}, {target.tile_z}))"
    )


async def test_player_in_conversation_never_moves() -> None:
    """While chatting with an NPC, the player must stay where the user
    parked them. `start_player_conversation` writes to the NPC's
    position (they freeze in TALKING) but must leave the player alone."""
    from core.npc.cognition.converse import start_player_conversation

    mgr, clock, player, _, _ = _make_sim()

    target = next(n for n in mgr.npcs if n.npc_id != "player")
    player.npc.x = float(target.tile_x + 1)
    player.npc.z = float(target.tile_z)

    start = (player.npc.x, player.npc.z)
    start_player_conversation(target, player.npc, "hello")
    await _drive_ticks(mgr, clock, 20)

    assert (player.npc.x, player.npc.z) == start, (
        f"Player moved during chat: {start} -> {(player.npc.x, player.npc.z)}"
    )


async def test_player_at_night_not_teleported_home() -> None:
    """`_enforce_bedtime` teleports NPCs to their sleep entry at night.
    The player has no schedule — they must be exempt."""
    mgr, clock, player, _, _ = _make_sim()

    # Fast-forward to the night phase.
    for _ in range(5000):
        clock.tick(8.0)
        mgr.movement_tick(clock, 8.0)
        await mgr.cognition_tick(clock, 8.0)
        if clock.phase.value == "night":
            break
    assert clock.phase.value == "night", "Could not reach night in test"

    player.npc.x = 12.0
    player.npc.z = -8.0
    start = (player.npc.x, player.npc.z)

    await _drive_ticks(mgr, clock, 20, real_delta=0.25)
    assert (player.npc.x, player.npc.z) == start, (
        f"Player teleported at night: {start} -> {(player.npc.x, player.npc.z)}"
    )


async def test_player_far_from_home_not_reanchored() -> None:
    """`_reanchor_strays` teleports NPCs back to home if they've
    drifted past a threshold. Must not apply to the player."""
    mgr, clock, player, _, _ = _make_sim()
    player.npc.x = -28.0
    player.npc.z = -28.0
    player.npc.activity = ActivityState.IDLE
    start = (player.npc.x, player.npc.z)

    await _drive_ticks(mgr, clock, 30)

    assert (player.npc.x, player.npc.z) == start, (
        f"Stray-catcher re-anchored the player: "
        f"{start} -> {(player.npc.x, player.npc.z)}"
    )


async def _run_all() -> int:
    tests = [
        ("stationary player never moves", test_player_stationary_on_own_tile_never_moves),
        ("player on NPC tile never nudged", test_player_standing_on_npc_tile_never_gets_nudged),
        ("player in chat never moves", test_player_in_conversation_never_moves),
        ("player at night not teleported home", test_player_at_night_not_teleported_home),
        ("player stray not re-anchored", test_player_far_from_home_not_reanchored),
    ]
    fails = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            await fn()
            print("  PASS")
        except AssertionError as e:
            fails.append((name, str(e)))
            print(f"  FAIL: {e}")
        except Exception as e:
            import traceback
            fails.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
