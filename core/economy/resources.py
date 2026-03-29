"""
Resource system — types, nodes, gathering mechanics.

Resource nodes exist on the world grid (placed by the generator).
ResourceManager wraps them with gameplay data: capacity, regeneration,
gathering time, and skill requirements. NPCs gather resources by
starting a gathering action and completing it after the required time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid, Tile

logger = logging.getLogger(__name__)


# ---------- Resource Types ----------

class ResourceType(Enum):
    """Harvestable resource categories."""
    WOOD = "wood"
    STONE = "stone"
    IRON = "iron"
    GOLD_ORE = "gold_ore"
    FOOD = "food"
    WHEAT = "wheat"
    BERRIES = "berries"


# Map from generator resource names to (ResourceType, display_name)
RESOURCE_NAME_MAP: dict[str, tuple[ResourceType, str]] = {
    "oak_tree": (ResourceType.WOOD, "Oak Tree"),
    "berry_bush": (ResourceType.BERRIES, "Berry Bush"),
    "wheat_field": (ResourceType.WHEAT, "Wheat Field"),
    "iron_deposit": (ResourceType.IRON, "Iron Deposit"),
    "stone_quarry": (ResourceType.STONE, "Stone Quarry"),
}


# ---------- Node Templates ----------

@dataclass
class NodeTemplate:
    """Default parameters for a resource node type."""
    resource_type: ResourceType
    capacity: int              # max harvestable units
    gather_time: float         # game minutes per gathering action
    base_yield: int            # units per successful gather
    regen_per_day: float       # units regenerated per game day
    required_skill: str        # skill name checked during gathering
    min_skill_level: float     # 0.0–1.0 minimum to attempt


NODE_TEMPLATES: dict[str, NodeTemplate] = {
    "oak_tree": NodeTemplate(
        resource_type=ResourceType.WOOD,
        capacity=20,
        gather_time=15.0,
        base_yield=3,
        regen_per_day=4.0,
        required_skill="gathering",
        min_skill_level=0.0,
    ),
    "berry_bush": NodeTemplate(
        resource_type=ResourceType.BERRIES,
        capacity=10,
        gather_time=8.0,
        base_yield=2,
        regen_per_day=6.0,
        required_skill="gathering",
        min_skill_level=0.0,
    ),
    "wheat_field": NodeTemplate(
        resource_type=ResourceType.WHEAT,
        capacity=30,
        gather_time=20.0,
        base_yield=4,
        regen_per_day=5.0,
        required_skill="farming",
        min_skill_level=0.1,
    ),
    "iron_deposit": NodeTemplate(
        resource_type=ResourceType.IRON,
        capacity=15,
        gather_time=25.0,
        base_yield=2,
        regen_per_day=2.0,
        required_skill="mining",
        min_skill_level=0.2,
    ),
    "stone_quarry": NodeTemplate(
        resource_type=ResourceType.STONE,
        capacity=25,
        gather_time=20.0,
        base_yield=3,
        regen_per_day=3.0,
        required_skill="mining",
        min_skill_level=0.1,
    ),
}


# ---------- Resource Node ----------

@dataclass
class ResourceNode:
    """
    A gatherable resource instance on the world grid.

    Wraps a WorldObject placed by the generator with gameplay state:
    capacity tracking, regeneration, and skill requirements.
    """
    node_id: str
    resource_name: str          # generator name, e.g. "oak_tree"
    resource_type: ResourceType
    x: int
    z: int

    # Capacity
    capacity: int = 20
    current_amount: int = 20

    # Gathering
    gather_time: float = 15.0   # game minutes
    base_yield: int = 3
    required_skill: str = "gathering"
    min_skill_level: float = 0.0

    # Regeneration
    regen_per_day: float = 4.0
    _regen_accumulator: float = field(default=0.0, repr=False)

    @property
    def is_depleted(self) -> bool:
        return self.current_amount <= 0

    @property
    def fullness(self) -> float:
        """0.0 (empty) to 1.0 (full)."""
        if self.capacity <= 0:
            return 0.0
        return self.current_amount / self.capacity

    def can_gather(self, npc: NPC) -> tuple[bool, str]:
        """Check whether an NPC can gather from this node."""
        if self.is_depleted:
            return False, "depleted"
        skill_level = npc.skills.get(self.required_skill, 0.0)
        if skill_level < self.min_skill_level:
            return False, f"requires {self.required_skill} >= {self.min_skill_level:.1f}"
        return True, "ok"

    def calculate_yield(self, npc: NPC) -> int:
        """
        Calculate gathering yield based on NPC skill.

        Yield = base_yield * (0.5 + skill_level).
        Capped at current_amount so we never go negative.
        """
        skill_level = npc.skills.get(self.required_skill, 0.0)
        multiplier = 0.5 + skill_level
        raw_yield = max(1, int(self.base_yield * multiplier))
        return min(raw_yield, self.current_amount)

    def extract(self, amount: int) -> int:
        """Remove resources from the node. Returns actual amount extracted."""
        taken = min(amount, self.current_amount)
        self.current_amount -= taken
        return taken

    def regenerate(self, game_minutes: float) -> None:
        """Accumulate regeneration over time. Called each tick."""
        if self.current_amount >= self.capacity:
            self._regen_accumulator = 0.0
            return
        day_fraction = game_minutes / 1440.0  # 1440 minutes per game day
        self._regen_accumulator += self.regen_per_day * day_fraction
        if self._regen_accumulator >= 1.0:
            restore = int(self._regen_accumulator)
            self.current_amount = min(self.capacity, self.current_amount + restore)
            self._regen_accumulator -= restore

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "resource_name": self.resource_name,
            "resource_type": self.resource_type.value,
            "x": self.x,
            "z": self.z,
            "capacity": self.capacity,
            "current_amount": self.current_amount,
            "is_depleted": self.is_depleted,
            "gather_time": self.gather_time,
        }


# ---------- Gathering Session ----------

@dataclass
class GatheringSession:
    """Tracks an in-progress gathering action."""
    npc_id: str
    node_id: str
    started_at: float     # game minutes (absolute)
    duration: float       # game minutes required
    expected_yield: int

    @property
    def completes_at(self) -> float:
        return self.started_at + self.duration


# ---------- Resource Manager ----------

class ResourceManager:
    """
    Manages all resource nodes in the world.

    Scans the grid for resource WorldObjects on init, wraps them
    with ResourceNode gameplay data, and handles gathering sessions.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ResourceNode] = {}
        self._nodes_by_pos: dict[tuple[int, int], ResourceNode] = {}
        self._sessions: dict[str, GatheringSession] = {}  # npc_id -> session

    def initialise_from_grid(self, grid: Grid) -> int:
        """
        Scan grid for resource WorldObjects and create ResourceNodes.

        Returns the number of nodes created.
        """
        self._nodes.clear()
        self._nodes_by_pos.clear()
        count = 0

        for tile in grid:
            for obj in tile.objects:
                if obj.object_type != "resource":
                    continue
                res_name = obj.metadata.get("resource", "")
                if res_name not in NODE_TEMPLATES:
                    continue

                template = NODE_TEMPLATES[res_name]
                gen_yield = obj.metadata.get("yield", template.base_yield)

                node = ResourceNode(
                    node_id=obj.object_id,
                    resource_name=res_name,
                    resource_type=template.resource_type,
                    x=tile.x,
                    z=tile.z,
                    capacity=template.capacity,
                    current_amount=template.capacity,
                    gather_time=template.gather_time,
                    base_yield=gen_yield,
                    required_skill=template.required_skill,
                    min_skill_level=template.min_skill_level,
                    regen_per_day=template.regen_per_day,
                )
                self._nodes[node.node_id] = node
                self._nodes_by_pos[(tile.x, tile.z)] = node
                count += 1

        logger.info("Resource manager initialised with %d nodes", count)
        return count

    # ---------- Queries ----------

    def get_node(self, node_id: str) -> ResourceNode | None:
        return self._nodes.get(node_id)

    def get_node_at(self, x: int, z: int) -> ResourceNode | None:
        return self._nodes_by_pos.get((x, z))

    def get_all_nodes(self) -> list[ResourceNode]:
        return list(self._nodes.values())

    def get_nodes_by_type(self, resource_type: ResourceType) -> list[ResourceNode]:
        return [n for n in self._nodes.values() if n.resource_type == resource_type]

    def get_nearest_node(
        self,
        x: int,
        z: int,
        resource_type: ResourceType | None = None,
        only_available: bool = True,
    ) -> ResourceNode | None:
        """
        Find the closest resource node by Manhattan distance.

        If resource_type is given, only considers that type.
        If only_available is True, skips depleted nodes.
        """
        best: ResourceNode | None = None
        best_dist = float("inf")

        for node in self._nodes.values():
            if resource_type and node.resource_type != resource_type:
                continue
            if only_available and node.is_depleted:
                continue
            dist = abs(node.x - x) + abs(node.z - z)
            if dist < best_dist:
                best_dist = dist
                best = node

        return best

    # ---------- Gathering ----------

    def start_gathering(
        self,
        npc: NPC,
        node: ResourceNode,
        current_game_time: float,
    ) -> tuple[bool, str]:
        """
        Begin a gathering session for an NPC at a resource node.

        Validates proximity (must be on or adjacent to node),
        skill requirements, and node availability.
        Returns (success, message).
        """
        # Already gathering?
        if npc.npc_id in self._sessions:
            return False, "already gathering"

        # Proximity check — must be within Manhattan distance 1
        dist = abs(npc.x - node.x) + abs(npc.z - node.z)
        if dist > 1:
            return False, "too far from node"

        # Skill and depletion check
        can, reason = node.can_gather(npc)
        if not can:
            return False, reason

        expected = node.calculate_yield(npc)

        session = GatheringSession(
            npc_id=npc.npc_id,
            node_id=node.node_id,
            started_at=current_game_time,
            duration=node.gather_time,
            expected_yield=expected,
        )
        self._sessions[npc.npc_id] = session
        return True, "ok"

    def complete_gathering(
        self,
        npc: NPC,
        current_game_time: float,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Complete a gathering session if enough time has passed.

        Returns (completed, result_dict). The result_dict contains
        resource_type, amount, and node_id on success.
        """
        session = self._sessions.get(npc.npc_id)
        if session is None:
            return False, {"reason": "no active session"}

        if current_game_time < session.completes_at:
            return False, {"reason": "not finished", "remaining": session.completes_at - current_game_time}

        node = self._nodes.get(session.node_id)
        if node is None:
            del self._sessions[npc.npc_id]
            return False, {"reason": "node no longer exists"}

        # Recalculate yield in case node was partially depleted by another NPC
        actual_yield = min(session.expected_yield, node.current_amount)
        if actual_yield <= 0:
            del self._sessions[npc.npc_id]
            return False, {"reason": "node depleted during gathering"}

        extracted = node.extract(actual_yield)

        # Add to NPC inventory
        resource_key = node.resource_type.value
        npc.inventory[resource_key] = npc.inventory.get(resource_key, 0) + extracted

        del self._sessions[npc.npc_id]

        result = {
            "resource_type": resource_key,
            "amount": extracted,
            "node_id": node.node_id,
            "node_remaining": node.current_amount,
        }
        logger.debug(
            "%s gathered %d %s from %s",
            npc.name, extracted, resource_key, node.node_id,
        )
        return True, result

    def cancel_gathering(self, npc_id: str) -> bool:
        """Cancel an in-progress gathering session."""
        if npc_id in self._sessions:
            del self._sessions[npc_id]
            return True
        return False

    def get_session(self, npc_id: str) -> GatheringSession | None:
        return self._sessions.get(npc_id)

    def is_gathering(self, npc_id: str) -> bool:
        return npc_id in self._sessions

    # ---------- Tick ----------

    def tick(self, game_minutes: float) -> None:
        """Regenerate depleted nodes. Called each simulation tick."""
        for node in self._nodes.values():
            node.regenerate(game_minutes)

    # ---------- Serialisation ----------

    def get_state(self) -> dict[str, Any]:
        """Full state for API / save."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "active_sessions": len(self._sessions),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_nodes": len(self._nodes),
            "depleted_nodes": sum(1 for n in self._nodes.values() if n.is_depleted),
            "active_sessions": len(self._sessions),
            "by_type": {
                rt.value: len(self.get_nodes_by_type(rt))
                for rt in ResourceType
                if self.get_nodes_by_type(rt)
            },
        }
