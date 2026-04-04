"""
LLM-based natural language town description parser.

Sends the description to an LLM (Haiku tier) with few-shot examples
and a structured output schema. Falls back to heuristic parsing
if no LLM provider is available or the call fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from core.world.prompt_gen.features import (
    MOOD_PRESETS,
    VALID_FEATURE_TYPES,
    TerrainFeature,
    TownMood,
)
from core.world.prompt_gen.heuristic import parse_description as heuristic_parse
from core.world.prompt_gen.templates import SYSTEM_PROMPT, build_messages

logger = logging.getLogger(__name__)

# Valid enum values for validation
_VALID_TERRAINS = frozenset({"riverside", "plains", "forest_edge", "hillside"})
_VALID_ECONOMIES = frozenset({"mixed", "farming", "trading", "mining"})


@dataclass
class ParsedTownDescription:
    """Structured output from parsing a natural language town description."""
    terrain: str = "plains"
    economy: str = "mixed"
    population_hint: int | None = None
    has_ruler: bool = False
    mood: TownMood | None = None
    features: list[TerrainFeature] = field(default_factory=list)
    name_suggestion: str | None = None
    source: str = "llm"  # "llm" or "heuristic"

    def to_dict(self) -> dict:
        return {
            "terrain": self.terrain,
            "economy": self.economy,
            "population_hint": self.population_hint,
            "has_ruler": self.has_ruler,
            "mood": self.mood.to_dict() if self.mood else None,
            "features": [f.to_dict() for f in self.features],
            "name_suggestion": self.name_suggestion,
            "source": self.source,
        }


async def parse_town_description(
    description: str,
    llm_provider=None,
) -> ParsedTownDescription:
    """Parse a natural language town description into structured parameters.

    Uses the LLM provider if available, otherwise falls back to heuristic.

    Args:
        description: Natural language description like
            "a cozy riverside town with two bridges".
        llm_provider: An LLMProvider instance (optional). If None, uses
            heuristic fallback.

    Returns:
        ParsedTownDescription with extracted parameters.
    """
    if llm_provider is None:
        return _from_heuristic(description)

    try:
        return await _from_llm(description, llm_provider)
    except Exception as e:
        logger.warning("LLM parsing failed, falling back to heuristic: %s", e)
        return _from_heuristic(description)


async def _from_llm(
    description: str,
    llm_provider,
) -> ParsedTownDescription:
    """Parse using LLM with few-shot prompting."""
    messages = build_messages(description)

    raw = await llm_provider.complete(
        system=SYSTEM_PROMPT,
        messages=messages,
        max_tokens=400,
        temperature=0.3,
        purpose="town_prompt_parse",
    )

    # Extract JSON from the response (handle markdown code blocks)
    json_str = _extract_json(raw)
    data = json.loads(json_str)

    return _validate_and_build(data, source="llm")


def _from_heuristic(description: str) -> ParsedTownDescription:
    """Parse using keyword matching fallback."""
    result = heuristic_parse(description)
    return ParsedTownDescription(
        terrain=result.terrain,
        economy=result.economy,
        population_hint=result.population_hint,
        has_ruler=result.has_ruler,
        mood=result.mood,
        features=result.features or [],
        name_suggestion=result.name_suggestion,
        source="heuristic",
    )


def _extract_json(raw: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    # Find the JSON object boundaries
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")

    return raw[start:end]


def _validate_and_build(
    data: dict,
    source: str = "llm",
) -> ParsedTownDescription:
    """Validate LLM output and build a ParsedTownDescription."""
    # Validate terrain
    terrain = data.get("terrain", "plains")
    if terrain not in _VALID_TERRAINS:
        terrain = "plains"

    # Validate economy
    economy = data.get("economy", "mixed")
    if economy not in _VALID_ECONOMIES:
        economy = "mixed"

    # Validate population
    pop = data.get("population_hint")
    if pop is not None:
        pop = max(4, min(20, int(pop)))

    # Validate ruler
    has_ruler = bool(data.get("has_ruler", False))

    # Validate mood
    mood = None
    mood_str = data.get("mood")
    if mood_str and mood_str in MOOD_PRESETS:
        mood = MOOD_PRESETS[mood_str]

    # Validate features
    features: list[TerrainFeature] = []
    for feat_data in data.get("features", []):
        feat_type = feat_data.get("type", "")
        if feat_type not in VALID_FEATURE_TYPES:
            continue
        # Bridge only valid with riverside
        if feat_type == "bridge" and terrain != "riverside":
            continue
        count = max(1, min(4, int(feat_data.get("count", 1))))
        placement = feat_data.get("placement", "auto")
        features.append(TerrainFeature(
            type=feat_type,
            count=count,
            placement=placement,
        ))

    name = data.get("name_suggestion")
    if name and len(name) > 30:
        name = name[:30]

    return ParsedTownDescription(
        terrain=terrain,
        economy=economy,
        population_hint=pop,
        has_ruler=has_ruler,
        mood=mood,
        features=features,
        name_suggestion=name,
        source=source,
    )
