"""
Planner context — lightweight world state snapshot.

Gathers everything the deterministic planner needs to make decisions
without any LLM calls or memory queries. Designed to be cheap to
construct every tick for Tier 3 NPCs.

The context is a plain data object — no game logic, no side effects.
Game systems populate it; the planner reads it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid
    from core.world.generator import PlacedBuilding


@dataclass
class PlannerContext:
    """
    Snapshot of the world state relevant to one NPC's decision.

    All fields are plain data — no references to live game objects.
    This makes the context safe to cache, log, or pass across threads.
    """
    # Time
    current_slot: str = ""
    game_minutes: float = 0.0
    current_day: int = 0

    # Nearby NPCs: list of {npc_id, name, x, z, occupation, distance}
    nearby_npcs: list[dict[str, Any]] = field(default_factory=list)

    # Nearby resource nodes: list of {x, z, resource_type, current_amount, distance}
    nearby_resource_nodes: list[dict[str, Any]] = field(default_factory=list)

    # Available recipes the NPC can craft (list of recipe IDs)
    available_recipes: list[str] = field(default_factory=list)

    # Building availability
    has_market: bool = False
    has_church: bool = False
    tavern_door: tuple[int, int] | None = None
    church_door: tuple[int, int] | None = None

    # Active construction sites: list of {site_id, x, z, blueprint_id, progress, distance,
    #   needs_wood, needs_stone, needs_labour}
    construction_sites: list[dict[str, Any]] = field(default_factory=list)

    # Threat (0.0 = peaceful, 1.0 = extreme danger)
    threat_level: float = 0.0
    threat_x: int = 0
    threat_z: int = 0

    # Random passable tile for wandering
    random_passable_tile: tuple[int, int] | None = None


class ContextBuilder:
    """
    Builds PlannerContext from live game state.

    Pluggable — the AI Game Studio can subclass this to inject
    custom context fields (e.g. weather, faction war status).
    """

    def __init__(
        self,
        grid: Grid,
        buildings: list[PlacedBuilding],
        perception_radius: int = 15,
        seed: int | None = None,
    ) -> None:
        self.grid = grid
        self.buildings = buildings
        self.perception_radius = perception_radius
        self.rng = random.Random(seed)

        # Pre-compute building lookups
        self._tavern_door: tuple[int, int] | None = None
        self._church_door: tuple[int, int] | None = None
        self._has_market = False
        self._has_church = False
        self._precompute_buildings()

    def _precompute_buildings(self) -> None:
        for b in self.buildings:
            if b.building_type == "tavern" and self._tavern_door is None:
                self._tavern_door = (b.door_x, b.door_z)
            if b.building_type == "church":
                self._church_door = (b.door_x, b.door_z)
                self._has_church = True
            if b.building_type == "market_stall":
                self._has_market = True

    def build(
        self,
        npc: NPC,
        all_npcs: list[NPC],
        current_slot: str,
        game_minutes: float = 0.0,
        current_day: int = 0,
        resource_nodes: list[dict[str, Any]] | None = None,
        available_recipes: list[str] | None = None,
        construction_sites: list[dict[str, Any]] | None = None,
        threat_level: float = 0.0,
        threat_x: int = 0,
        threat_z: int = 0,
    ) -> PlannerContext:
        """Build a context snapshot for the given NPC."""
        nearby_npcs = self._gather_nearby_npcs(npc, all_npcs)
        nearby_resources = self._gather_nearby_resources(
            npc, resource_nodes or [],
        )
        nearby_sites = self._gather_nearby_sites(
            npc, construction_sites or [],
        )
        wander_tile = self._pick_wander_tile(npc)

        return PlannerContext(
            current_slot=current_slot,
            game_minutes=game_minutes,
            current_day=current_day,
            nearby_npcs=nearby_npcs,
            nearby_resource_nodes=nearby_resources,
            available_recipes=available_recipes or [],
            construction_sites=nearby_sites,
            has_market=self._has_market,
            has_church=self._has_church,
            tavern_door=self._tavern_door,
            church_door=self._church_door,
            threat_level=threat_level,
            threat_x=threat_x,
            threat_z=threat_z,
            random_passable_tile=wander_tile,
        )

    def _gather_nearby_npcs(
        self, npc: NPC, all_npcs: list[NPC],
    ) -> list[dict[str, Any]]:
        result = []
        for other in all_npcs:
            if other.npc_id == npc.npc_id:
                continue
            dist = npc.distance_to(other.x, other.z)
            if dist <= self.perception_radius:
                result.append({
                    "npc_id": other.npc_id,
                    "name": other.name,
                    "x": other.x,
                    "z": other.z,
                    "occupation": other.occupation,
                    "distance": dist,
                })
        result.sort(key=lambda n: n["distance"])
        return result

    def _gather_nearby_resources(
        self, npc: NPC, resource_nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter and sort resource nodes by distance from NPC."""
        result = []
        for node in resource_nodes:
            dist = abs(npc.x - node["x"]) + abs(npc.z - node["z"])
            if dist <= self.perception_radius:
                result.append({**node, "distance": dist})
        result.sort(key=lambda n: n["distance"])
        return result

    def _gather_nearby_sites(
        self, npc: NPC, construction_sites: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter and sort construction sites by distance from NPC.

        Construction is a community-wide event — NPCs hear about it
        across the whole town, so we use a much wider radius than
        the normal perception radius.
        """
        # Construction sites visible from 3x normal perception (town-wide)
        construction_radius = self.perception_radius * 3
        result = []
        for site in construction_sites:
            dist = abs(npc.x - site["x"]) + abs(npc.z - site["z"])
            if dist <= construction_radius:
                result.append({**site, "distance": dist})
        result.sort(key=lambda s: s["distance"])
        return result

    def _pick_wander_tile(self, npc: NPC) -> tuple[int, int] | None:
        """Pick a random passable tile within wandering range."""
        for _ in range(10):
            dx = self.rng.randint(-8, 8)
            dz = self.rng.randint(-8, 8)
            tx, tz = npc.tile_x + dx, npc.tile_z + dz
            tile = self.grid.get_tile(tx, tz)
            if tile and tile.is_passable:
                return (tx, tz)
        return (npc.home_x, npc.home_z)
