"""
Prompt templates for LLM-based town description parsing.

Contains the system prompt, output schema, and few-shot examples
used to convert natural language town descriptions into structured
WorldConfig parameters and terrain features.
"""

from __future__ import annotations

# JSON schema that the LLM must output
OUTPUT_SCHEMA = """\
{
  "terrain": "riverside" | "plains" | "forest_edge" | "hillside",
  "economy": "mixed" | "farming" | "trading" | "mining",
  "population_hint": <integer 4-20 or null>,
  "has_ruler": <boolean>,
  "mood": "<one of: cozy, bustling, fortified, rustic, sacred, frontier>" | null,
  "features": [
    {
      "type": "<bridge | pond | wall | ruins | garden | watchtower | orchard | well | campfire>",
      "count": <integer 1-4>,
      "placement": "auto" | "near_river" | "edge" | "centre" | "outskirts"
    }
  ],
  "name_suggestion": "<a fitting town name>" | null
}"""

SYSTEM_PROMPT = f"""\
You are a world-building assistant for a medieval fantasy town generator.
Given a natural language description of a town, extract structured parameters.

Rules:
- terrain must be one of: riverside, plains, forest_edge, hillside
- economy must be one of: mixed, farming, trading, mining
- population_hint: infer from description size cues (small=5-7, medium=8-12, large=13-20), or null if unclear
- has_ruler: true if description mentions a ruler, lord, king, mayor, or government
- mood: pick the SINGLE best match from [cozy, bustling, fortified, rustic, sacred, frontier], or null if none fit
- features: list specific terrain features mentioned or strongly implied
  - bridge: only valid with riverside terrain (rivers to cross)
  - wall: implies fortification
  - pond: a small body of still water, distinct from rivers
  - ruins: ancient or abandoned structures
  - garden: decorative planted areas, flower gardens
  - watchtower: lookout structures, usually at edges
  - orchard: fruit tree groves
  - well: town well, water source
  - campfire: outdoor gathering spot
- name_suggestion: invent a fitting name if the description suggests a feel/theme
- Do NOT add features that are not mentioned or strongly implied
- Output ONLY valid JSON matching this schema, no other text

Output schema:
{OUTPUT_SCHEMA}"""

# Few-shot examples for in-context learning
FEW_SHOT_EXAMPLES = [
    {
        "description": "A cozy riverside town with forest and two bridges",
        "output": {
            "terrain": "riverside",
            "economy": "mixed",
            "population_hint": 7,
            "has_ruler": False,
            "mood": "cozy",
            "features": [
                {"type": "bridge", "count": 2, "placement": "auto"},
            ],
            "name_suggestion": "Willowbrook",
        },
    },
    {
        "description": "A bustling trading hub on the plains with a large market and watchtower",
        "output": {
            "terrain": "plains",
            "economy": "trading",
            "population_hint": 15,
            "has_ruler": False,
            "mood": "bustling",
            "features": [
                {"type": "watchtower", "count": 1, "placement": "edge"},
            ],
            "name_suggestion": "Crossroads",
        },
    },
    {
        "description": "A fortified mining town in the hills, ruled by a stern lord, with defensive walls",
        "output": {
            "terrain": "hillside",
            "economy": "mining",
            "population_hint": 12,
            "has_ruler": True,
            "mood": "fortified",
            "features": [
                {"type": "wall", "count": 1, "placement": "auto"},
            ],
            "name_suggestion": "Ironhaven",
        },
    },
    {
        "description": "A small sacred village near ancient ruins with a garden and well",
        "output": {
            "terrain": "plains",
            "economy": "farming",
            "population_hint": 5,
            "has_ruler": False,
            "mood": "sacred",
            "features": [
                {"type": "ruins", "count": 1, "placement": "outskirts"},
                {"type": "garden", "count": 1, "placement": "centre"},
                {"type": "well", "count": 1, "placement": "centre"},
            ],
            "name_suggestion": "Sanctuary",
        },
    },
    {
        "description": "A frontier farming settlement at the edge of a great forest with orchards",
        "output": {
            "terrain": "forest_edge",
            "economy": "farming",
            "population_hint": 6,
            "has_ruler": False,
            "mood": "frontier",
            "features": [
                {"type": "orchard", "count": 1, "placement": "outskirts"},
            ],
            "name_suggestion": "Timberstead",
        },
    },
]


def build_messages(description: str) -> list[dict[str, str]]:
    """Build the message list for the LLM call with few-shot examples."""
    import json

    messages: list[dict[str, str]] = []

    # Add few-shot examples as user/assistant pairs
    for example in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": example["description"],
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["output"]),
        })

    # Add the actual request
    messages.append({
        "role": "user",
        "content": description,
    })

    return messages
