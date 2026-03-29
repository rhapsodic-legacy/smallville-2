"""
Spatial memory — hierarchical world knowledge tree.

Each NPC maintains a mental model of the world: known locations,
what's in each area, and where to find things. Updated from
perception observations and used by planning for location resolution.

Mirrors the grid's world:sector:arena addressing scheme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArenaKnowledge:
    """What an NPC knows about a specific arena (sub-zone)."""
    name: str
    objects: list[str] = field(default_factory=list)
    last_visited: float = 0.0  # game minutes
    notes: list[str] = field(default_factory=list)  # free-form observations

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objects": list(self.objects),
            "last_visited": self.last_visited,
            "notes": self.notes[-5:],  # keep recent notes
        }


@dataclass
class SectorKnowledge:
    """What an NPC knows about a sector (zone)."""
    name: str
    arenas: dict[str, ArenaKnowledge] = field(default_factory=dict)
    last_visited: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arenas": {k: v.to_dict() for k, v in self.arenas.items()},
            "last_visited": self.last_visited,
        }


class SpatialMemory:
    """
    Per-NPC hierarchical world knowledge tree.

    Structure: world → sectors → arenas → objects/notes
    Updated when NPCs perceive their surroundings.
    Queried by planning to decide where to go.
    """

    def __init__(self) -> None:
        # npc_id → {sector_name → SectorKnowledge}
        self._trees: dict[str, dict[str, SectorKnowledge]] = {}

    def _ensure_npc(self, npc_id: str) -> dict[str, SectorKnowledge]:
        if npc_id not in self._trees:
            self._trees[npc_id] = {}
        return self._trees[npc_id]

    # ---------- Updates from perception ----------

    def update_from_perception(
        self,
        npc_id: str,
        sector: str,
        arena: str,
        objects: list[str] | None = None,
        note: str | None = None,
        game_time: float = 0.0,
    ) -> None:
        """Update spatial knowledge from a perception observation."""
        tree = self._ensure_npc(npc_id)

        if sector not in tree:
            tree[sector] = SectorKnowledge(name=sector)

        sk = tree[sector]
        sk.last_visited = max(sk.last_visited, game_time)

        if arena:
            if arena not in sk.arenas:
                sk.arenas[arena] = ArenaKnowledge(name=arena)

            ak = sk.arenas[arena]
            ak.last_visited = max(ak.last_visited, game_time)

            if objects:
                for obj in objects:
                    if obj not in ak.objects:
                        ak.objects.append(obj)

            if note and note not in ak.notes:
                ak.notes.append(note)
                # Keep notes bounded
                if len(ak.notes) > 20:
                    ak.notes = ak.notes[-20:]

    def update_from_tile(
        self,
        npc_id: str,
        tile_sector: str,
        tile_arena: str,
        tile_objects: list[str],
        game_time: float = 0.0,
    ) -> None:
        """Convenience: update from a grid tile's properties."""
        self.update_from_perception(
            npc_id=npc_id,
            sector=tile_sector or "unknown",
            arena=tile_arena or "",
            objects=tile_objects,
            game_time=game_time,
        )

    # ---------- Queries ----------

    def get_known_sectors(self, npc_id: str) -> list[str]:
        """List all sectors this NPC knows about."""
        tree = self._trees.get(npc_id, {})
        return list(tree.keys())

    def get_known_arenas(self, npc_id: str, sector: str) -> list[str]:
        """List all arenas in a sector that this NPC knows about."""
        tree = self._trees.get(npc_id, {})
        sk = tree.get(sector)
        if not sk:
            return []
        return list(sk.arenas.keys())

    def get_arena_objects(
        self, npc_id: str, sector: str, arena: str,
    ) -> list[str]:
        """Get known objects in a specific arena."""
        tree = self._trees.get(npc_id, {})
        sk = tree.get(sector)
        if not sk:
            return []
        ak = sk.arenas.get(arena)
        if not ak:
            return []
        return list(ak.objects)

    def find_object(self, npc_id: str, object_name: str) -> list[str]:
        """
        Search for a named object across all known locations.

        Returns list of addresses like "sector:arena" where the object is known.
        """
        tree = self._trees.get(npc_id, {})
        results = []
        search = object_name.lower()
        for sector_name, sk in tree.items():
            for arena_name, ak in sk.arenas.items():
                for obj in ak.objects:
                    if search in obj.lower():
                        results.append(f"{sector_name}:{arena_name}")
                        break
        return results

    def get_world_summary(self, npc_id: str) -> str:
        """
        Generate a natural language summary of what the NPC knows
        about the world. Used in LLM prompts for planning.
        """
        tree = self._trees.get(npc_id, {})
        if not tree:
            return "You do not yet know much about the layout of Smallville."

        parts = ["You know about these areas in Smallville:"]
        for sector_name, sk in tree.items():
            arena_names = list(sk.arenas.keys())
            if arena_names:
                parts.append(
                    f"- {sector_name}: contains {', '.join(arena_names)}"
                )
            else:
                parts.append(f"- {sector_name}")

        return "\n".join(parts)

    # ---------- Full tree access (for inspector) ----------

    def get_tree(self, npc_id: str) -> dict[str, Any]:
        """Return the full knowledge tree as a dict (for UI inspector)."""
        tree = self._trees.get(npc_id, {})
        return {
            sector: sk.to_dict() for sector, sk in tree.items()
        }

    def get_all_npc_ids(self) -> list[str]:
        """List all NPCs that have spatial memory."""
        return list(self._trees.keys())

    def get_stats(self) -> dict[str, Any]:
        """Summary stats for the memory inspector."""
        total_sectors = 0
        total_arenas = 0
        for tree in self._trees.values():
            total_sectors += len(tree)
            for sk in tree.values():
                total_arenas += len(sk.arenas)

        return {
            "npcs_with_spatial": len(self._trees),
            "total_sectors_known": total_sectors,
            "total_arenas_known": total_arenas,
        }
