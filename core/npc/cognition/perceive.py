"""
Perception module.

NPCs observe their surroundings: nearby NPCs, objects, events, and terrain.
Perception is filtered by vision radius and attention bandwidth.
A retention window prevents re-perceiving the same things repeatedly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.world.grid import Grid

logger = logging.getLogger(__name__)

# Vision radius per tier (in tiles, Manhattan distance)
VISION_RADIUS = {1: 8, 2: 6, 3: 4, 4: 0}

# Max perceptions to keep in the retention window
MAX_RECENT_PERCEPTIONS = 20

# How many new observations per perception cycle
ATTENTION_BANDWIDTH = {1: 8, 2: 4, 3: 2, 4: 0}


@dataclass
class Observation:
    """A single thing the NPC noticed."""
    description: str
    category: str     # "npc", "object", "terrain", "event"
    x: int
    z: int
    importance: float  # 0.0–1.0 estimated importance


def perceive(
    npc: NPC,
    grid: Grid,
    all_npcs: list[NPC],
    current_game_minutes: float,
) -> list[Observation]:
    """
    Run the perception cycle for an NPC.

    Scans nearby tiles for NPCs, objects, and notable terrain.
    Updates npc.recent_perceptions with new observations.

    Returns the list of new observations.
    """
    radius = VISION_RADIUS.get(npc.cognition_tier, 0)
    bandwidth = ATTENTION_BANDWIDTH.get(npc.cognition_tier, 0)

    if radius == 0 or bandwidth == 0:
        return []

    observations: list[Observation] = []

    # Perceive nearby NPCs
    for other in all_npcs:
        if other.npc_id == npc.npc_id:
            continue
        if npc.distance_to(other.x, other.z) <= radius:
            desc = _describe_npc(other)
            observations.append(Observation(
                description=desc,
                category="npc",
                x=other.x, z=other.z,
                importance=_npc_importance(npc, other),
            ))

    # Perceive nearby objects
    nearby_tiles = grid.tiles_in_radius(npc.tile_x, npc.tile_z, radius)
    for tile in nearby_tiles:
        for obj in tile.objects:
            if obj.object_type == "building":
                continue  # buildings are static, don't re-perceive
            desc = f"{obj.name} at ({tile.x}, {tile.z})"
            observations.append(Observation(
                description=desc,
                category="object",
                x=tile.x, z=tile.z,
                importance=0.2,
            ))

    # Sort by importance and limit to attention bandwidth
    observations.sort(key=lambda o: o.importance, reverse=True)
    observations = observations[:bandwidth]

    # Filter out observations already in retention window
    existing = set(npc.recent_perceptions)
    new_observations = [
        o for o in observations if o.description not in existing
    ]

    # Update retention window
    for obs in new_observations:
        npc.recent_perceptions.append(obs.description)
    # Trim retention window
    if len(npc.recent_perceptions) > MAX_RECENT_PERCEPTIONS:
        npc.recent_perceptions = npc.recent_perceptions[-MAX_RECENT_PERCEPTIONS:]

    npc.last_perception_tick = current_game_minutes

    if new_observations:
        logger.debug(
            "%s perceived %d new things (tier %d)",
            npc.name, len(new_observations), npc.cognition_tier,
        )

    return new_observations


def _describe_npc(other: NPC) -> str:
    """Generate a natural language description of a perceived NPC."""
    activity_descriptions = {
        "idle": "standing around",
        "walking": "walking",
        "working": "working",
        "sleeping": "sleeping",
        "talking": "having a conversation",
        "eating": "eating",
        "gathering": "gathering resources",
    }
    activity = activity_descriptions.get(other.activity.value, other.activity.value)
    return f"{other.name} the {other.occupation} is {activity} nearby"


def _npc_importance(npc: NPC, other: NPC) -> float:
    """Estimate how important another NPC is to perceive."""
    importance = 0.3  # base

    # Closer NPCs are more important
    dist = npc.distance_to(other.x, other.z)
    if dist <= 2:
        importance += 0.3
    elif dist <= 4:
        importance += 0.1

    # NPCs doing something interesting
    if other.activity.value in ("talking", "gathering"):
        importance += 0.1

    # Same occupation = professional interest
    if other.occupation == npc.occupation:
        importance += 0.1

    return min(1.0, importance)
