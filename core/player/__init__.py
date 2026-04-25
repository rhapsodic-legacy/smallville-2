"""Player module — player-as-NPC model, interaction handling."""

from core.player.player_agent import (
    PlayerAgent,
    AwarenessMode,
    find_player_spawn,
    INTERACTION_RADIUS,
    PLAYER_MOVE_SPEED,
)

__all__ = [
    "PlayerAgent",
    "AwarenessMode",
    "find_player_spawn",
    "INTERACTION_RADIUS",
    "PLAYER_MOVE_SPEED",
]
