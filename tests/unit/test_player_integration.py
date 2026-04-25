"""
Tests for Phase 8 — Player Integration.

Covers: PlayerAgent creation, movement, NPC interaction,
chat routing, trade routing, and server wiring.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.player.player_agent import (
    PlayerAgent, AwarenessMode, find_player_spawn,
    INTERACTION_RADIUS, PLAYER_MOVE_SPEED,
)
from core.npc.models import NPC, ActivityState, PersonalityTraits, Direction


# ---------- Helpers ----------

def _make_grid(width=20, height=20):
    """Create a minimal grid for testing.

    Grid auto-creates all tiles as walkable grass.
    Coordinates are centred: a 20x20 grid goes from -10 to 9.
    """
    from core.world.grid import Grid
    return Grid(width, height)


def _make_npc(npc_id="npc_1", name="Aldric", x=5.0, z=5.0, occupation="blacksmith"):
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory=f"{name} is a {occupation}.",
        occupation=occupation,
        x=x, z=z,
    )


# ---------- PlayerAgent creation ----------

class TestPlayerAgentCreation:
    def test_create_default(self):
        pa = PlayerAgent.create()
        assert pa.npc_id == "player"
        assert pa.name == "Traveller"
        assert pa.npc.occupation == "traveller"
        assert pa.npc.gold == 50
        assert pa.npc.move_speed == PLAYER_MOVE_SPEED
        assert pa.awareness == AwarenessMode.INDISTINGUISHABLE

    def test_create_custom(self):
        pa = PlayerAgent.create(
            name="Hero",
            spawn_x=10.0,
            spawn_z=15.0,
            awareness=AwarenessMode.KNOWN_HUMAN,
        )
        assert pa.name == "Hero"
        assert pa.x == 10.0
        assert pa.z == 15.0
        assert pa.awareness == AwarenessMode.KNOWN_HUMAN

    def test_npc_id_is_player(self):
        pa = PlayerAgent.create()
        assert pa.npc.npc_id == "player"

    def test_to_dict_includes_player_fields(self):
        pa = PlayerAgent.create()
        d = pa.to_dict()
        assert d["is_player"] is True
        assert "awareness" in d
        assert "interaction_radius" in d
        assert "gold" in d
        assert "inventory" in d


# ---------- Player movement ----------

class TestPlayerMovement:
    def test_move_north(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        pa.set_move_direction("north")
        moved = pa.movement_tick(grid, 0.25)
        assert moved is True
        assert pa.z < 0.0
        assert pa.npc.activity == ActivityState.WALKING

    def test_move_south(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        pa.set_move_direction("south")
        moved = pa.movement_tick(grid, 0.25)
        assert moved is True
        assert pa.z > 0.0

    def test_move_east(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        pa.set_move_direction("east")
        moved = pa.movement_tick(grid, 0.25)
        assert moved is True
        assert pa.x > 0.0

    def test_move_west(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        pa.set_move_direction("west")
        moved = pa.movement_tick(grid, 0.25)
        assert moved is True
        assert pa.x < 0.0

    def test_no_move_without_direction(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        moved = pa.movement_tick(grid, 0.25)
        assert moved is False

    def test_stop_after_clearing_direction(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        pa.set_move_direction("north")
        pa.movement_tick(grid, 0.25)
        assert pa.npc.activity == ActivityState.WALKING
        # Direction is consumed after one tick
        moved = pa.movement_tick(grid, 0.25)
        assert moved is False

    def test_blocked_by_impassable_tile(self):
        from core.world.grid import Grid, Terrain
        grid = Grid(20, 20)
        # Block the tile to the north of (0, 0) -> (0, -1)
        tile = grid.get_tile(0, -1)
        tile.walkable = False

        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        pa.set_move_direction("north")
        moved = pa.movement_tick(grid, 0.25)
        assert moved is False

    def test_blocked_by_out_of_bounds(self):
        grid = _make_grid()  # 20x20 -> coords -10 to 9
        # Spawn at northern edge
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=-10.0)
        pa.set_move_direction("north")  # Would go to z=-11 (out of bounds)
        moved = pa.movement_tick(grid, 0.25)
        assert moved is False

    def test_move_speed_applied(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        grid = _make_grid()
        pa.set_move_direction("south")
        pa.movement_tick(grid, 0.5)
        # Should not overshoot the next tile
        assert pa.z <= 1.0
        assert pa.z > 0.0


# ---------- Nearby NPC detection ----------

class TestNearbyNPCs:
    def test_nearby_npcs_within_radius(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        nearby_npc = _make_npc(x=1.0, z=0.0)
        far_npc = _make_npc(npc_id="npc_2", name="Far", x=20.0, z=20.0)
        result = pa.get_nearby_npcs([nearby_npc, far_npc])
        assert len(result) == 1
        assert result[0].npc_id == "npc_1"

    def test_excludes_self(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        result = pa.get_nearby_npcs([pa.npc])
        assert len(result) == 0

    def test_closest_npc(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        close = _make_npc(npc_id="close", name="Close", x=2.0, z=0.0)
        closer = _make_npc(npc_id="closer", name="Closer", x=1.0, z=0.0)
        result = pa.get_closest_npc([close, closer])
        assert result is not None
        assert result.npc_id == "closer"

    def test_no_nearby(self):
        pa = PlayerAgent.create(spawn_x=0.0, spawn_z=0.0)
        far = _make_npc(x=50.0, z=50.0)
        result = pa.get_closest_npc([far])
        assert result is None


# ---------- Awareness modes ----------

class TestAwareness:
    def test_indistinguishable_description(self):
        pa = PlayerAgent.create(awareness=AwarenessMode.INDISTINGUISHABLE)
        desc = pa.npc_awareness_description()
        assert "ordinary" in desc.lower() or "traveller" in desc.lower()

    def test_known_human_description(self):
        pa = PlayerAgent.create(awareness=AwarenessMode.KNOWN_HUMAN)
        desc = pa.npc_awareness_description()
        assert "visitor" in desc.lower() or "different" in desc.lower()


# ---------- Spawn point ----------

class TestSpawnPoint:
    def test_finds_passable_tile(self):
        grid = _make_grid()
        x, z = find_player_spawn(grid, [])
        tile = grid.get_tile(round(x), round(z))
        assert tile is not None
        assert tile.is_passable

    def test_prefers_centre(self):
        grid = _make_grid()
        x, z = find_player_spawn(grid, [])
        # Grid is centred at (0, 0)
        assert abs(x) <= 1
        assert abs(z) <= 1


# ---------- Player serialisation ----------

class TestPlayerSerialisation:
    def test_to_dict_roundtrip(self):
        pa = PlayerAgent.create(spawn_x=3.0, spawn_z=5.0)
        pa.npc.gold = 100
        pa.npc.inventory = {"wood": 5, "stone": 3}
        d = pa.to_dict()
        assert d["x"] == 3.0
        assert d["z"] == 5.0
        assert d["gold"] == 100
        assert d["inventory"]["wood"] == 5
        assert d["is_player"] is True

    def test_chat_state_in_dict(self):
        pa = PlayerAgent.create()
        pa.is_chatting = True
        pa.chat_target_id = "npc_1"
        d = pa.to_dict()
        assert d["is_chatting"] is True
        assert d["chat_target_id"] == "npc_1"

    def test_trade_state_in_dict(self):
        pa = PlayerAgent.create()
        pa.is_trading = True
        pa.trade_target_id = "merchant_1"
        d = pa.to_dict()
        assert d["is_trading"] is True
        assert d["trade_target_id"] == "merchant_1"
