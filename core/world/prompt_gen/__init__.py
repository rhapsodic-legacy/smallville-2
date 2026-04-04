"""
Town generation prompt system.

Converts natural language town descriptions into WorldConfig and
terrain features that drive the procedural generator.

Usage:
    from core.world.prompt_gen import TownPromptGenerator

    gen = TownPromptGenerator(llm_provider=my_provider)  # or None for heuristic
    spec = await gen.generate_config("cozy riverside town with two bridges")
    # spec.config  -> WorldConfig
    # spec.features -> [TerrainFeature(...)]
    # spec.town_name -> "Willowbrook"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.world.generator import WorldConfig
from core.world.prompt_gen.features import TerrainFeature, TownMood
from core.world.prompt_gen.parser import ParsedTownDescription, parse_town_description

logger = logging.getLogger(__name__)


@dataclass
class GeneratedWorldSpec:
    """Everything needed to generate a world from a prompt."""
    config: WorldConfig
    features: list[TerrainFeature]
    town_name: str
    parsed: ParsedTownDescription  # for inspection/debugging

    def to_dict(self) -> dict:
        return {
            "config": {
                "population": self.config.population,
                "terrain": self.config.terrain,
                "economy": self.config.economy,
                "has_ruler": self.config.has_ruler,
                "seed": self.config.seed,
                "grid_width": self.config.grid_width,
                "grid_height": self.config.grid_height,
            },
            "features": [f.to_dict() for f in self.features],
            "town_name": self.town_name,
            "parsed": self.parsed.to_dict(),
        }


class TownPromptGenerator:
    """Converts natural language descriptions to world generation specs.

    Args:
        llm_provider: An LLMProvider instance for LLM-based parsing.
            If None, uses heuristic keyword matching.
        default_population: Base population before mood modifiers.
        seed: Random seed for deterministic generation.
        grid_width: Grid width (default 60).
        grid_height: Grid height (default 60).
    """

    def __init__(
        self,
        llm_provider=None,
        default_population: int = 10,
        seed: int | None = None,
        grid_width: int = 60,
        grid_height: int = 60,
    ):
        self.llm_provider = llm_provider
        self.default_population = default_population
        self.seed = seed
        self.grid_width = grid_width
        self.grid_height = grid_height

    async def generate_config(self, description: str) -> GeneratedWorldSpec:
        """Parse a description and produce a GeneratedWorldSpec.

        Args:
            description: Natural language like "cozy riverside town
                with forest and two bridges".

        Returns:
            GeneratedWorldSpec containing WorldConfig, features, and
            a suggested town name.
        """
        parsed = await parse_town_description(
            description,
            llm_provider=self.llm_provider,
        )

        config = self._build_config(parsed)
        features = parsed.features
        town_name = parsed.name_suggestion or "Smallville"

        logger.info(
            "Prompt parsed [%s]: terrain=%s economy=%s pop=%d features=%d name=%s",
            parsed.source, config.terrain, config.economy,
            config.population, len(features), town_name,
        )

        return GeneratedWorldSpec(
            config=config,
            features=features,
            town_name=town_name,
            parsed=parsed,
        )

    def _build_config(self, parsed: ParsedTownDescription) -> WorldConfig:
        """Convert parsed description into a WorldConfig."""
        # Start with population hint or default
        population = parsed.population_hint or self.default_population

        # Apply mood modifier
        if parsed.mood:
            population = int(population * parsed.mood.population_modifier)
            population = max(4, min(20, population))

        # Mood can suggest economy if parser didn't find an explicit one
        economy = parsed.economy
        if economy == "mixed" and parsed.mood and parsed.mood.economy_hint:
            economy = parsed.mood.economy_hint

        # Fortified mood implies a ruler
        has_ruler = parsed.has_ruler
        if parsed.mood and parsed.mood.descriptor == "fortified":
            has_ruler = True

        # Wall feature also implies fortification/ruler
        if any(f.type == "wall" for f in parsed.features):
            has_ruler = True

        return WorldConfig(
            population=population,
            terrain=parsed.terrain,
            economy=economy,
            has_ruler=has_ruler,
            seed=self.seed,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
        )
