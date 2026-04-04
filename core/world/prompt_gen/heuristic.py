"""
Keyword-based fallback parser for town descriptions.

When no LLM is available, extracts WorldConfig parameters and terrain
features from natural language using pattern matching. Not as nuanced
as the LLM parser, but handles common descriptions reliably.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.world.prompt_gen.features import (
    MOOD_PRESETS,
    VALID_FEATURE_TYPES,
    TerrainFeature,
    TownMood,
)


@dataclass
class HeuristicResult:
    """Output of the heuristic parser."""
    terrain: str = "plains"
    economy: str = "mixed"
    population_hint: int | None = None
    has_ruler: bool = False
    mood: TownMood | None = None
    features: list[TerrainFeature] | None = None
    name_suggestion: str | None = None

    def __post_init__(self):
        if self.features is None:
            self.features = []


# ---------- Keyword maps ----------

_TERRAIN_KEYWORDS: list[tuple[list[str], str]] = [
    (["river", "riverside", "brook", "creek", "stream"], "riverside"),
    (["forest", "woods", "woodland", "trees", "timber"], "forest_edge"),
    (["hill", "hillside", "mountain", "highland", "elevated"], "hillside"),
    (["plain", "plains", "flat", "meadow", "grassland", "prairie"], "plains"),
]

_ECONOMY_KEYWORDS: list[tuple[list[str], str]] = [
    (["farm", "farming", "agricultural", "crops", "harvest"], "farming"),
    (["trade", "trading", "merchant", "commerce", "market"], "trading"),
    (["mine", "mining", "quarry", "ore", "iron", "stone"], "mining"),
]

_RULER_KEYWORDS = [
    "ruler", "lord", "king", "queen", "mayor", "governor", "chief",
    "baron", "duke", "duchess", "ruled", "governed",
]

_SIZE_KEYWORDS: dict[str, int] = {
    "tiny": 4,
    "small": 6,
    "little": 6,
    "modest": 7,
    "medium": 10,
    "large": 14,
    "big": 14,
    "huge": 18,
    "sprawling": 18,
    "vast": 20,
}

_FEATURE_KEYWORDS: dict[str, str] = {
    "bridge": "bridge",
    "bridges": "bridge",
    "pond": "pond",
    "lake": "pond",
    "pool": "pond",
    "wall": "wall",
    "walls": "wall",
    "palisade": "wall",
    "fortification": "wall",
    "ruin": "ruins",
    "ruins": "ruins",
    "ancient": "ruins",
    "abandoned": "ruins",
    "garden": "garden",
    "gardens": "garden",
    "flowers": "garden",
    "watchtower": "watchtower",
    "tower": "watchtower",
    "lookout": "watchtower",
    "orchard": "orchard",
    "orchards": "orchard",
    "fruit trees": "orchard",
    "well": "well",
    "campfire": "campfire",
    "bonfire": "campfire",
    "fire pit": "campfire",
}

# Number words for count extraction
_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "couple": 2, "few": 2, "several": 3, "many": 4,
}


def parse_description(description: str) -> HeuristicResult:
    """Parse a natural language town description using keyword matching."""
    text = description.lower().strip()
    result = HeuristicResult()

    result.terrain = _extract_terrain(text)
    result.economy = _extract_economy(text)
    result.population_hint = _extract_population(text)
    result.has_ruler = _extract_ruler(text)
    result.mood = _extract_mood(text)
    result.features = _extract_features(text, result.terrain)
    result.name_suggestion = None  # Heuristic can't generate names

    return result


def _extract_terrain(text: str) -> str:
    """Find the best-matching terrain type."""
    for keywords, terrain in _TERRAIN_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return terrain
    return "plains"


def _extract_economy(text: str) -> str:
    """Find the best-matching economy type."""
    for keywords, economy in _ECONOMY_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return economy
    return "mixed"


def _extract_population(text: str) -> int | None:
    """Extract population hint from size keywords."""
    for keyword, pop in _SIZE_KEYWORDS.items():
        if keyword in text:
            return pop
    return None


def _extract_ruler(text: str) -> bool:
    """Check if the description mentions a ruler."""
    return any(kw in text for kw in _RULER_KEYWORDS)


def _extract_mood(text: str) -> TownMood | None:
    """Find the best-matching mood preset."""
    for mood_name, mood in MOOD_PRESETS.items():
        if mood_name in text:
            return mood
    return None


def _extract_features(text: str, terrain: str) -> list[TerrainFeature]:
    """Extract terrain features and their counts from the description."""
    features: dict[str, int] = {}

    # Multi-word patterns first
    for phrase, feat_type in sorted(
        _FEATURE_KEYWORDS.items(), key=lambda x: -len(x[0])
    ):
        if phrase in text:
            # Try to find a count modifier before the keyword
            count = _find_count_before(text, phrase)
            # Bridge only valid with riverside
            if feat_type == "bridge" and terrain != "riverside":
                continue
            features[feat_type] = max(features.get(feat_type, 0), count)

    return [
        TerrainFeature(type=ft, count=ct)
        for ft, ct in features.items()
    ]


def _find_count_before(text: str, keyword: str) -> int:
    """Look for a number or number word before a keyword in the text."""
    idx = text.find(keyword)
    if idx <= 0:
        return 1

    # Look at the ~20 characters before the keyword
    prefix = text[max(0, idx - 20):idx].strip()
    words = prefix.split()
    if not words:
        return 1

    last_word = words[-1]

    # Try numeric
    if last_word.isdigit():
        return min(int(last_word), 4)

    # Try number word
    if last_word in _NUMBER_WORDS:
        return _NUMBER_WORDS[last_word]

    return 1
