"""
Economy tick orchestrator.

Thin wrapper that holds references to all economy managers and runs
their per-tick logic. Extracted from NPCManager to keep it under
the 750-line limit.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from core.economy.resources import ResourceManager
from core.economy.trading import TradeManager
from core.economy.crafting import CraftingManager
from core.economy.construction import ConstructionManager

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid

logger = logging.getLogger(__name__)


class EconomyTick:
    """
    Orchestrates per-tick economy updates.

    Holds all four economy managers and exposes a single tick()
    method that NPCManager calls each game loop iteration.
    """

    def __init__(
        self,
        grid: Grid,
        resources: ResourceManager | None = None,
        trading: TradeManager | None = None,
        crafting: CraftingManager | None = None,
        construction: ConstructionManager | None = None,
    ) -> None:
        self.resources = resources or ResourceManager()
        self.trading = trading or TradeManager()
        self.crafting = crafting or CraftingManager()
        self.construction = construction or ConstructionManager()
        self._grid = grid

        # Initialise resources from the grid's world objects
        self.resources.initialise_from_grid(grid)

    def tick(
        self,
        npcs: list[NPC],
        game_minutes: float,
        current_game_time: float,
    ) -> None:
        """
        Run all economy updates for one tick.

        - Regenerate resource nodes
        - Complete any finished gathering sessions
        - Complete any finished crafting sessions
        """
        # Resource regeneration
        self.resources.tick(game_minutes)

        # Auto-complete gathering sessions
        for npc in npcs:
            if self.resources.is_gathering(npc.npc_id):
                completed, result = self.resources.complete_gathering(
                    npc, current_game_time,
                )
                if completed:
                    logger.debug(
                        "%s finished gathering %d %s",
                        npc.name,
                        result["amount"],
                        result["resource_type"],
                    )

        # Auto-complete crafting sessions
        for npc in npcs:
            if self.crafting.is_crafting(npc.npc_id):
                completed, result = self.crafting.complete_crafting(
                    npc, current_game_time,
                )
                if completed:
                    logger.debug(
                        "%s finished crafting %s",
                        npc.name, result.get("item", "unknown"),
                    )

        # Auto-contribute labour for NPCs at construction sites
        self._tick_construction(npcs, game_minutes)

    def _tick_construction(
        self, npcs: list[NPC], game_minutes: float,
    ) -> None:
        """Auto-contribute labour for NPCs near active construction sites."""
        from core.npc.models import ActivityState
        sites = self.construction.get_all_sites()
        if not sites:
            return
        for npc in npcs:
            if npc.activity == ActivityState.WALKING:
                continue
            # Check current description or schedule entry for construction
            desc = getattr(npc, "current_action_description", "") or ""
            on_construct = "construct" in desc.lower()
            if not on_construct:
                # Also check if NPC's current schedule is a construct action
                # (gap-fill subtasks like "taking a break" don't mention construct
                # but the NPC is still assigned to construction)
                schedule = getattr(npc, "daily_schedule", [])
                for entry in schedule:
                    if "construct" in (getattr(entry, "action_id", "") or ""):
                        on_construct = True
                        break
            if not on_construct:
                continue
            # Find nearest site within working distance (5 tiles from approach)
            for site in sites:
                bp = site.blueprint
                # Approach tile is south face centre
                approach_x = site.x + bp.width // 2
                approach_z = site.z + bp.height
                dist = abs(npc.x - approach_x) + abs(npc.z - approach_z)
                if dist <= 5 and not site.is_complete:
                    # Contribute any resources NPC has
                    needed = site.resources_still_needed()
                    for res in list(needed.keys()):
                        amt = npc.inventory.get(res, 0)
                        if amt > 0:
                            site.contribute_resources(npc, res, amt)
                    # Contribute labour
                    site.contribute_labour(npc, game_minutes)
                    # Check completion
                    if site.is_complete:
                        self.construction.check_and_complete(
                            site.site_id, self._grid,
                        )
                    break

    def get_resource_node_dicts(self) -> list[dict[str, Any]]:
        """Resource nodes as dicts for the planner context."""
        return [
            {
                "node_id": n.node_id,
                "resource_type": n.resource_type.value,
                "x": n.x,
                "z": n.z,
                "available": not n.is_depleted,
            }
            for n in self.resources.get_all_nodes()
            if not n.is_depleted
        ]

    def get_construction_site_dicts(self) -> list[dict[str, Any]]:
        """Active construction sites as dicts for the planner context."""
        result = []
        for site in self.construction.get_all_sites():
            if site.is_complete:
                continue
            needed = site.resources_still_needed()
            bp = site.blueprint
            # Access tile: south face centre approach
            access_dx = bp.width // 2
            access_dz = bp.height  # one tile south of footprint
            result.append({
                "site_id": site.site_id,
                "x": site.x,
                "z": site.z,
                "blueprint_id": bp.blueprint_id,
                "progress": site.progress,
                "needs_wood": needed.get("wood", 0),
                "needs_stone": needed.get("stone", 0),
                "needs_labour": bp.labour_required - site.labour_contributed,
                "access_dx": access_dx,
                "access_dz": access_dz,
            })
        return result

    def get_available_recipes(self) -> list[str]:
        """Recipe IDs for the planner context."""
        return [r.recipe_id for r in self.crafting.get_all_recipes()]

    def get_state(self) -> dict[str, Any]:
        """Economy state for broadcast."""
        return {
            "resources": self.resources.get_state(),
            "trading": self.trading.get_stats(),
            "crafting": self.crafting.get_stats(),
            "construction": self.construction.get_stats(),
        }
