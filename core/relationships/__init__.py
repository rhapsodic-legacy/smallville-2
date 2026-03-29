"""Relationships module — sentiment dimensions, factions, formal structures."""

from core.relationships.sentiment import (
    Sentiment,
    SentimentTracker,
    DIMENSIONS,
)
from core.relationships.structures import (
    Faction,
    FactionManager,
    FactionMember,
    FactionRelation,
    FactionRole,
    Agreement,
)

__all__ = [
    "Sentiment",
    "SentimentTracker",
    "DIMENSIONS",
    "Faction",
    "FactionManager",
    "FactionMember",
    "FactionRelation",
    "FactionRole",
    "Agreement",
]
