"""
Player agent — human player modelled as an NPC.

Same data model as NPC with a human flag and player-specific
interaction logic. NPCs treat the player according to the
configured awareness mode.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from core.npc.models import (
    NPC, ActivityState, Direction, PersonalityTraits,
)

if TYPE_CHECKING:
    from core.world.grid import Grid
    from core.world.generator import PlacedBuilding

logger = logging.getLogger(__name__)

# How far the player can interact with NPCs / objects
INTERACTION_RADIUS = 3  # tiles (Manhattan distance)

# Movement: tiles per real second (slightly faster than NPCs)
PLAYER_MOVE_SPEED = 3.0


class AwarenessMode(Enum):
    """How NPCs perceive the player."""
    INDISTINGUISHABLE = "indistinguishable"  # NPCs treat player as another NPC
    KNOWN_HUMAN = "known_human"  # NPCs know this is a human player


@dataclass
class PlayerAgent:
    """Human player wrapped as an NPC-compatible entity.

    Uses composition: holds an NPC instance and adds player-specific
    state. The underlying NPC participates in the normal cognition
    cycle (perception, conversations, memory) but movement is
    driven by player input instead of the schedule system.
    """

    npc: NPC
    awareness: AwarenessMode = AwarenessMode.INDISTINGUISHABLE
    interaction_radius: int = INTERACTION_RADIUS

    # When True, the player NPC acts autonomously (schedules, conversations,
    # perception, reflections) like any other NPC when the player isn't
    # actively controlling it. When False, only player input drives behaviour.
    autonomous: bool = True

    # Player-specific state
    is_chatting: bool = False
    chat_target_id: str | None = None
    is_trading: bool = False
    trade_target_id: str | None = None

    # Input state (set by server from WebSocket messages)
    _move_direction: Direction | None = None
    _move_held: bool = False

    @staticmethod
    def create(
        name: str = "Traveller",
        spawn_x: float = 0.0,
        spawn_z: float = 0.0,
        awareness: AwarenessMode = AwarenessMode.INDISTINGUISHABLE,
    ) -> PlayerAgent:
        """Create a new player agent at the given spawn point."""
        npc = NPC(
            npc_id="player",
            name=name,
            age=25,
            personality=PersonalityTraits(
                openness=0.7,
                conscientiousness=0.5,
                extraversion=0.6,
                agreeableness=0.6,
                neuroticism=0.3,
            ),
            backstory=f"{name} is a traveller who has recently arrived in town.",
            occupation="traveller",
            x=spawn_x,
            z=spawn_z,
            home_x=round(spawn_x),
            home_z=round(spawn_z),
            work_x=round(spawn_x),
            work_z=round(spawn_z),
            health=1.0,
            energy=1.0,
            hunger=0.0,
            gold=50,
            skills={"diplomacy": 0.5, "trading": 0.5},
            cognition_tier=1,
            move_speed=PLAYER_MOVE_SPEED,
            archetype="traveller",
            _rng=random.Random(hash("player")),
        )
        npc.long_term_goals = ["Explore the town", "Meet the townsfolk"]
        npc.short_term_goals = ["Look around"]

        return PlayerAgent(npc=npc, awareness=awareness)

    @property
    def npc_id(self) -> str:
        return self.npc.npc_id

    @property
    def name(self) -> str:
        return self.npc.name

    @property
    def x(self) -> float:
        return self.npc.x

    @property
    def z(self) -> float:
        return self.npc.z

    @property
    def tile_x(self) -> int:
        return self.npc.tile_x

    @property
    def tile_z(self) -> int:
        return self.npc.tile_z

    def set_move_direction(self, direction: str | None) -> None:
        """Set movement direction from player input. None = stop."""
        if direction is None:
            self._move_direction = None
            self._move_held = False
            return
        try:
            self._move_direction = Direction(direction)
            self._move_held = True
        except ValueError:
            logger.warning("Invalid direction: %s", direction)

    def movement_tick(self, grid: Grid, real_delta: float) -> bool:
        """Process one movement tick. Returns True if position changed.

        Server-authoritative: validates the move against the grid
        before applying it. Player moves tile-by-tile like NPCs.
        """
        if not self._move_held or self._move_direction is None:
            if self.npc.activity == ActivityState.WALKING:
                self.npc.activity = ActivityState.IDLE
            return False

        # Calculate target tile
        dx, dz = _direction_delta(self._move_direction)
        target_x = self.npc.tile_x + dx
        target_z = self.npc.tile_z + dz

        # Validate: passable and in bounds
        if not grid.in_bounds(target_x, target_z):
            return False
        tile = grid.get_tile(target_x, target_z)
        if tile is None or not tile.is_passable:
            return False

        # Smooth movement: lerp toward target
        speed = self.npc.move_speed * real_delta
        nx = self.npc.x + dx * speed
        nz = self.npc.z + dz * speed

        # Clamp to not overshoot target
        if dx > 0:
            nx = min(nx, float(target_x))
        elif dx < 0:
            nx = max(nx, float(target_x))
        if dz > 0:
            nz = min(nz, float(target_z))
        elif dz < 0:
            nz = max(nz, float(target_z))

        self.npc.x = nx
        self.npc.z = nz
        self.npc.direction = self._move_direction
        self.npc.activity = ActivityState.WALKING
        self.npc.current_action_description = "exploring"

        # Stop holding after one tick (client must re-send for continuous)
        self._move_held = False

        return True

    def get_nearby_npcs(
        self, all_npcs: list[NPC], radius: int | None = None,
    ) -> list[NPC]:
        """Get NPCs within interaction radius."""
        r = radius or self.interaction_radius
        return [
            n for n in all_npcs
            if n.npc_id != self.npc_id and n.distance_to(self.tile_x, self.tile_z) <= r
        ]

    def get_closest_npc(self, all_npcs: list[NPC]) -> NPC | None:
        """Get the closest NPC within interaction radius."""
        nearby = self.get_nearby_npcs(all_npcs)
        if not nearby:
            return None
        return min(nearby, key=lambda n: n.distance_to(self.tile_x, self.tile_z))

    def to_dict(self) -> dict[str, Any]:
        """Serialise player state for WebSocket transmission."""
        base = self.npc.to_dict()
        base["is_player"] = True
        base["awareness"] = self.awareness.value
        base["interaction_radius"] = self.interaction_radius
        base["is_chatting"] = self.is_chatting
        base["chat_target_id"] = self.chat_target_id
        base["is_trading"] = self.is_trading
        base["trade_target_id"] = self.trade_target_id
        base["gold"] = self.npc.gold
        base["inventory"] = dict(self.npc.inventory)
        return base

    def npc_awareness_description(self) -> str:
        """Description of the player for NPC prompts, based on awareness mode."""
        if self.awareness == AwarenessMode.KNOWN_HUMAN:
            return (
                f"{self.name} is a visitor to the town. They seem different from "
                f"the other townsfolk — more curious, more unpredictable."
            )
        return (
            f"{self.name} is a traveller who recently arrived in town. "
            f"They seem like an ordinary person."
        )


def _direction_delta(direction: Direction) -> tuple[int, int]:
    """Convert direction enum to (dx, dz) delta."""
    return {
        Direction.NORTH: (0, -1),
        Direction.SOUTH: (0, 1),
        Direction.EAST: (1, 0),
        Direction.WEST: (-1, 0),
    }[direction]


def find_player_spawn(
    grid: Grid,
    buildings: list[PlacedBuilding],
) -> tuple[float, float]:
    """Find a good spawn point for the player near the town centre.

    Prefers a passable tile near (0, 0) — the grid's centre.
    """
    # Grid is centred at (0, 0)
    cx, cz = 0, 0

    # Spiral outward from centre to find a passable tile
    for radius in range(0, max(grid.width, grid.height)):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dz) != radius:
                    continue
                tx, tz = cx + dx, cz + dz
                if not grid.in_bounds(tx, tz):
                    continue
                tile = grid.get_tile(tx, tz)
                if tile and tile.is_passable:
                    return float(tx), float(tz)

    return float(cx), float(cz)
