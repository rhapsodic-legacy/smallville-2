"""
Construction system — blueprints, build sites, phased progress.

NPCs contribute resources and labour to construction sites.
Buildings progress through visual phases (planned → foundation → walls →
roofing → complete) based on the fraction of required resources delivered.
Once all resources and labour are in, the site converts to a finished building.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid

logger = logging.getLogger(__name__)


# ---------- Build Phases ----------

class BuildPhase(Enum):
    """Visual phases of a building under construction."""
    PLANNED = "planned"           # 0 %
    FOUNDATION = "foundation"     # 25 %
    WALLS = "walls"               # 50 %
    ROOFING = "roofing"           # 75 %
    COMPLETE = "complete"         # 100 %


PHASE_THRESHOLDS: list[tuple[float, BuildPhase]] = [
    (0.0, BuildPhase.PLANNED),
    (0.25, BuildPhase.FOUNDATION),
    (0.50, BuildPhase.WALLS),
    (0.75, BuildPhase.ROOFING),
    (1.0, BuildPhase.COMPLETE),
]


def phase_for_progress(progress: float) -> BuildPhase:
    """Return the build phase for a given progress fraction (0.0–1.0)."""
    result = BuildPhase.PLANNED
    for threshold, phase in PHASE_THRESHOLDS:
        if progress >= threshold:
            result = phase
    return result


# ---------- Blueprint ----------

@dataclass
class Blueprint:
    """
    Template defining what a building requires to construct.

    required_resources: resource_type → amount needed.
    labour_required: total game minutes of NPC work on-site.
    """
    blueprint_id: str
    building_type: str
    name: str
    width: int
    height: int
    required_resources: dict[str, int] = field(default_factory=dict)
    labour_required: float = 60.0

    @property
    def total_resource_units(self) -> int:
        """Sum of all resource amounts required."""
        return sum(self.required_resources.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "building_type": self.building_type,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "required_resources": dict(self.required_resources),
            "labour_required": self.labour_required,
        }


# Default blueprints available for construction
DEFAULT_BLUEPRINTS: dict[str, Blueprint] = {
    "church": Blueprint(
        blueprint_id="church",
        building_type="church",
        name="Church",
        width=5, height=5,
        required_resources={"wood": 100, "stone": 50},
        labour_required=120.0,
    ),
    "market_stall": Blueprint(
        blueprint_id="market_stall",
        building_type="market_stall",
        name="Market Stall",
        width=2, height=2,
        required_resources={"wood": 30, "stone": 10},
        labour_required=30.0,
    ),
    "watchtower": Blueprint(
        blueprint_id="watchtower",
        building_type="watchtower",
        name="Watchtower",
        width=3, height=3,
        required_resources={"wood": 40, "stone": 60},
        labour_required=90.0,
    ),
    "home": Blueprint(
        blueprint_id="home",
        building_type="home",
        name="Home",
        width=2, height=3,
        required_resources={"wood": 50, "stone": 20},
        labour_required=60.0,
    ),
    "bridge": Blueprint(
        blueprint_id="bridge",
        building_type="bridge",
        name="Bridge",
        width=2, height=4,
        required_resources={"wood": 20, "stone": 40},
        labour_required=45.0,
    ),
}


# ---------- Construction Site ----------

@dataclass
class ConstructionSite:
    """
    An active building under construction on the grid.

    Tracks resources contributed, labour done, current phase,
    and which NPCs have contributed (for credit and memory).
    """
    site_id: str
    blueprint: Blueprint
    x: int
    z: int

    contributed: dict[str, int] = field(default_factory=dict)
    labour_contributed: float = 0.0
    contributors: dict[str, int] = field(default_factory=dict)  # npc_id → units
    created_at: float = 0.0  # game minutes

    @property
    def resource_progress(self) -> float:
        """Fraction of required resources contributed (0.0–1.0)."""
        total_needed = self.blueprint.total_resource_units
        if total_needed <= 0:
            return 1.0
        total_contributed = sum(
            min(self.contributed.get(res, 0), needed)
            for res, needed in self.blueprint.required_resources.items()
        )
        return total_contributed / total_needed

    @property
    def labour_progress(self) -> float:
        """Fraction of required labour completed (0.0–1.0)."""
        if self.blueprint.labour_required <= 0:
            return 1.0
        return min(1.0, self.labour_contributed / self.blueprint.labour_required)

    @property
    def progress(self) -> float:
        """
        Overall completion (0.0–1.0).

        Weighted: 70% resources, 30% labour.
        """
        return self.resource_progress * 0.7 + self.labour_progress * 0.3

    @property
    def phase(self) -> BuildPhase:
        return phase_for_progress(self.progress)

    @property
    def is_complete(self) -> bool:
        return self.resource_progress >= 1.0 and self.labour_progress >= 1.0

    def resources_still_needed(self) -> dict[str, int]:
        """Return {resource: amount_still_needed} for incomplete resources."""
        needed: dict[str, int] = {}
        for res, total in self.blueprint.required_resources.items():
            have = self.contributed.get(res, 0)
            if have < total:
                needed[res] = total - have
        return needed

    def contribute_resources(
        self, npc: NPC, resource: str, amount: int,
    ) -> tuple[int, str]:
        """
        NPC contributes resources from their inventory.

        Only accepts resources the blueprint requires, and only up to
        the remaining need. Deducts from NPC inventory.
        Returns (amount_accepted, message).
        """
        if resource not in self.blueprint.required_resources:
            return 0, f"{resource} not needed for this building"

        needed = self.blueprint.required_resources[resource]
        have = self.contributed.get(resource, 0)
        remaining = needed - have
        if remaining <= 0:
            return 0, f"{resource} requirement already met"

        available = npc.inventory.get(resource, 0)
        if available <= 0:
            return 0, f"no {resource} in inventory"

        actual = min(amount, remaining, available)
        npc.inventory[resource] = available - actual
        self.contributed[resource] = have + actual
        self.contributors[npc.npc_id] = self.contributors.get(npc.npc_id, 0) + actual

        logger.debug(
            "%s contributed %d %s to %s (%d/%d)",
            npc.name, actual, resource, self.blueprint.name,
            self.contributed[resource], needed,
        )
        return actual, "ok"

    def contribute_labour(self, npc: NPC, game_minutes: float) -> tuple[float, str]:
        """
        NPC works on the construction site.

        Only accepts labour if some resources have been contributed
        (can't build with nothing). Returns (minutes_accepted, message).
        """
        if self.resource_progress <= 0:
            return 0.0, "no resources contributed yet"

        remaining = self.blueprint.labour_required - self.labour_contributed
        if remaining <= 0:
            return 0.0, "labour complete"

        actual = min(game_minutes, remaining)
        self.labour_contributed += actual
        # Count labour as 1 unit per 10 minutes for contributor tracking
        labour_units = max(1, int(actual / 10))
        self.contributors[npc.npc_id] = self.contributors.get(npc.npc_id, 0) + labour_units

        return actual, "ok"

    def summary(self) -> str:
        """Human-readable status for LLM prompts."""
        parts = [f"{self.blueprint.name} at ({self.x}, {self.z})"]
        parts.append(f"Phase: {self.phase.value}")
        parts.append(f"Progress: {self.progress:.0%}")
        needed = self.resources_still_needed()
        if needed:
            res_strs = [f"{amt} {res}" for res, amt in needed.items()]
            parts.append(f"Still needs: {', '.join(res_strs)}")
        if self.labour_progress < 1.0:
            remaining_labour = self.blueprint.labour_required - self.labour_contributed
            parts.append(f"Labour remaining: {remaining_labour:.0f} minutes")
        return ". ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "blueprint": self.blueprint.to_dict(),
            "x": self.x,
            "z": self.z,
            "contributed": dict(self.contributed),
            "labour_contributed": round(self.labour_contributed, 1),
            "progress": round(self.progress, 3),
            "phase": self.phase.value,
            "is_complete": self.is_complete,
            "resources_needed": self.resources_still_needed(),
            "contributors": dict(self.contributors),
        }


# ---------- Construction Manager ----------

class ConstructionManager:
    """
    Manages all active construction sites.

    Handles site creation, resource/labour contributions,
    completion (converting site to finished building on the grid),
    and queries for NPC planning.
    """

    def __init__(
        self,
        blueprints: dict[str, Blueprint] | None = None,
        on_event: Any | None = None,
    ):
        self._blueprints = dict(blueprints or DEFAULT_BLUEPRINTS)
        self._on_event = on_event  # callback: (event_type, participants, data) -> None
        self._sites: dict[str, ConstructionSite] = {}
        self._completed: list[ConstructionSite] = []

    # ---------- Blueprints ----------

    def get_blueprint(self, blueprint_id: str) -> Blueprint | None:
        return self._blueprints.get(blueprint_id)

    def get_all_blueprints(self) -> list[Blueprint]:
        return list(self._blueprints.values())

    def add_blueprint(self, blueprint: Blueprint) -> None:
        self._blueprints[blueprint.blueprint_id] = blueprint

    # ---------- Site Creation ----------

    def start_construction(
        self,
        blueprint_id: str,
        x: int,
        z: int,
        grid: Grid,
        game_time: float = 0.0,
    ) -> tuple[ConstructionSite | None, str]:
        """
        Begin construction at a grid location.

        Validates the blueprint exists, the footprint is clear
        (all tiles passable, no objects), and places a construction
        WorldObject on the grid.
        Returns (site, message).
        """
        from core.world.grid import WorldObject

        blueprint = self._blueprints.get(blueprint_id)
        if blueprint is None:
            return None, f"unknown blueprint: {blueprint_id}"

        # Check footprint is clear
        for dx in range(blueprint.width):
            for dz in range(blueprint.height):
                tile = grid.get_tile(x + dx, z + dz)
                if tile is None:
                    return None, f"tile ({x + dx}, {z + dz}) out of bounds"
                if not tile.is_passable:
                    return None, f"tile ({x + dx}, {z + dz}) not passable"
                if tile.objects:
                    return None, f"tile ({x + dx}, {z + dz}) already occupied"

        site_id = f"site_{uuid.uuid4().hex[:8]}"
        site = ConstructionSite(
            site_id=site_id,
            blueprint=blueprint,
            x=x, z=z,
            created_at=game_time,
        )

        # Place construction object on the grid (top-left tile only)
        obj = WorldObject(
            object_id=site_id,
            object_type="construction",
            name=f"{blueprint.name} (under construction)",
            walkable=True,
            metadata={
                "blueprint_id": blueprint_id,
                "phase": BuildPhase.PLANNED.value,
            },
        )
        grid.place_object(x, z, obj)

        # Mark footprint tiles as occupied (not walkable) except the
        # access tile: south face centre, like building doors.
        access_dx = blueprint.width // 2
        access_dz = blueprint.height - 1  # south face
        for dx in range(blueprint.width):
            for dz in range(blueprint.height):
                if dx == access_dx and dz == access_dz:
                    continue  # access tile stays walkable
                if dx == 0 and dz == 0:
                    continue  # object tile stays walkable
                tile = grid.get_tile(x + dx, z + dz)
                if tile is not None:
                    tile.walkable = False

        # Ensure approach tile (one south of access) is walkable
        approach = grid.get_tile(x + access_dx, z + blueprint.height)
        if approach is not None:
            approach.walkable = True

        self._sites[site_id] = site
        logger.info(
            "Construction started: %s at (%d, %d)", blueprint.name, x, z,
        )
        return site, "ok"

    # ---------- Contributions ----------

    def contribute_resources(
        self,
        npc: NPC,
        site_id: str,
        resource: str,
        amount: int,
    ) -> tuple[int, str]:
        """NPC contributes resources to a construction site."""
        site = self._sites.get(site_id)
        if site is None:
            return 0, "site not found"
        if site.is_complete:
            return 0, "construction already complete"
        return site.contribute_resources(npc, resource, amount)

    def contribute_labour(
        self,
        npc: NPC,
        site_id: str,
        game_minutes: float,
    ) -> tuple[float, str]:
        """NPC works on a construction site."""
        site = self._sites.get(site_id)
        if site is None:
            return 0.0, "site not found"
        if site.is_complete:
            return 0.0, "construction already complete"
        return site.contribute_labour(npc, game_minutes)

    # ---------- Completion ----------

    def check_and_complete(
        self,
        site_id: str,
        grid: Grid,
    ) -> tuple[bool, str]:
        """
        Check if a site is complete and finalise the building.

        Replaces the construction WorldObject with a finished building,
        marks footprint as non-walkable, and fires construction_complete.
        """
        from core.world.grid import WorldObject

        site = self._sites.get(site_id)
        if site is None:
            return False, "site not found"
        if not site.is_complete:
            return False, f"not complete ({site.progress:.0%})"

        # Remove construction object from grid
        tile = grid.get_tile(site.x, site.z)
        if tile is not None:
            tile.objects = [
                o for o in tile.objects if o.object_id != site_id
            ]

        # Place finished building object
        building_obj = WorldObject(
            object_id=f"{site.blueprint.building_type}_{site_id}",
            object_type="building",
            name=site.blueprint.name,
            walkable=False,
            metadata={
                "width": site.blueprint.width,
                "height": site.blueprint.height,
                "built_by": list(site.contributors.keys()),
            },
        )
        grid.place_object(site.x, site.z, building_obj)

        # Ensure entire footprint is non-walkable
        for dx in range(site.blueprint.width):
            for dz in range(site.blueprint.height):
                ft = grid.get_tile(site.x + dx, site.z + dz)
                if ft is not None:
                    ft.walkable = False

        # Move site to completed list
        del self._sites[site_id]
        self._completed.append(site)

        # Fire event
        contributor_ids = list(site.contributors.keys())
        if self._on_event and contributor_ids:
            self._on_event(
                "construction_complete",
                contributor_ids,
                {
                    "building_type": site.blueprint.building_type,
                    "name": site.blueprint.name,
                    "x": site.x,
                    "z": site.z,
                },
            )

        logger.info("Construction complete: %s at (%d, %d)", site.blueprint.name, site.x, site.z)
        return True, "ok"

    # ---------- Queries ----------

    def get_site(self, site_id: str) -> ConstructionSite | None:
        return self._sites.get(site_id)

    def get_all_sites(self) -> list[ConstructionSite]:
        return list(self._sites.values())

    def get_sites_needing(self, resource: str) -> list[ConstructionSite]:
        """Return active sites that still need a specific resource."""
        return [
            site for site in self._sites.values()
            if resource in site.resources_still_needed()
        ]

    def get_nearest_site(
        self,
        x: int,
        z: int,
        resource: str | None = None,
    ) -> ConstructionSite | None:
        """
        Find the closest construction site by Manhattan distance.

        If resource is given, only considers sites needing that resource.
        """
        candidates = (
            self.get_sites_needing(resource) if resource
            else self.get_all_sites()
        )
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda s: abs(s.x - x) + abs(s.z - z),
        )

    def evaluate_contribution(
        self,
        npc: NPC,
        site: ConstructionSite,
    ) -> tuple[bool, str]:
        """
        Heuristic: should this NPC contribute to this site?

        Considers occupation affinity, available resources, and distance.
        """
        # Labourers and builders always willing
        if npc.occupation in ("labourer", "guard"):
            return True, "occupation suited to construction"

        # Check if NPC has any needed resources
        needed = site.resources_still_needed()
        has_something = any(
            npc.inventory.get(res, 0) > 0
            for res in needed
        )
        if has_something:
            return True, "has needed resources"

        # Far away NPCs less likely
        dist = abs(npc.x - site.x) + abs(npc.z - site.z)
        if dist > 20:
            return False, "too far away"

        return True, "community contribution"

    # ---------- State ----------

    def get_state(self) -> dict[str, Any]:
        return {
            "active_sites": [s.to_dict() for s in self._sites.values()],
            "completed_count": len(self._completed),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "active_sites": len(self._sites),
            "completed_buildings": len(self._completed),
            "blueprints_available": len(self._blueprints),
        }
