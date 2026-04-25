"""
Seed memories — foundational memories planted at NPC creation.

Based on Stanford Generative Agents' approach where seed memories
drive initial behaviour. Each NPC wakes up with identity, craft
knowledge, motivation, and community awareness.

Also seeds inter-NPC relationship facts and initial sentiment so
NPCs know each other from day one.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from core.npc.models import SEED_MEMORIES, UNIVERSAL_SEED_MEMORIES

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.memory.manager import MemoryManager
    from core.relationships.sentiment import SentimentTracker

logger = logging.getLogger(__name__)


# ---------- Occupational relationship rules ----------
# Maps (occupation_a, occupation_b) -> (predicate, description_template, sentiment)
# description_template uses {a_name}, {b_name}, {a_occ}, {b_occ}
# sentiment is dict of dimension -> initial value

OCCUPATIONAL_BONDS: list[dict] = [
    {
        "occupations": ("blacksmith", "merchant"),
        "predicate": "trades_with",
        "desc": "{a_name} supplies tools and metalwork to {b_name} the {b_occ}.",
        "sentiment": {"trust": 15, "respect": 10},
    },
    {
        "occupations": ("farmer", "tavern_keeper"),
        "predicate": "supplies",
        "desc": "{a_name} supplies fresh produce to {b_name} at the tavern.",
        "sentiment": {"trust": 15, "affection": 5},
    },
    {
        "occupations": ("farmer", "merchant"),
        "predicate": "trades_with",
        "desc": "{a_name} sells surplus crops to {b_name} the {b_occ}.",
        "sentiment": {"trust": 10, "respect": 5},
    },
    {
        "occupations": ("guard", "priest"),
        "predicate": "respects",
        "desc": "{a_name} respects {b_name} the {b_occ} — they both serve the town.",
        "sentiment": {"respect": 20, "trust": 10},
    },
    {
        "occupations": ("tavern_keeper", "guard"),
        "predicate": "knows_well",
        "desc": "{a_name} sees {b_name} the {b_occ} most evenings at the tavern.",
        "sentiment": {"trust": 10, "affection": 10},
    },
    {
        "occupations": ("priest", "tavern_keeper"),
        "predicate": "knows_well",
        "desc": "{a_name} and {b_name} are pillars of the community, though they disagree on drink.",
        "sentiment": {"respect": 15, "affection": 5},
    },
    {
        "occupations": ("labourer", "blacksmith"),
        "predicate": "works_for",
        "desc": "{a_name} helps {b_name} at the forge when extra hands are needed.",
        "sentiment": {"respect": 15, "trust": 5},
    },
    {
        "occupations": ("labourer", "farmer"),
        "predicate": "works_for",
        "desc": "{a_name} helps {b_name} in the fields during harvest.",
        "sentiment": {"trust": 10, "respect": 5},
    },
]

# Neighbour relationship templates — for NPCs sharing nearby homes
NEIGHBOUR_DESCS: list[str] = [
    "{a_name} and {b_name} are neighbours. They see each other every morning.",
    "{a_name} lives near {b_name}. They sometimes share meals.",
    "{a_name} and {b_name} are neighbours and often chat over the fence.",
]


def seed_npc_memories(npc: NPC, memory: MemoryManager) -> int:
    """
    Plant foundational memories for a single NPC.

    Returns the number of memories created.
    """
    count = 0

    # Occupation-specific seeds
    occ_seeds = SEED_MEMORIES.get(npc.occupation, SEED_MEMORIES["labourer"])
    for desc_template, category, importance in occ_seeds:
        desc = desc_template.format(name=npc.name, occupation=npc.occupation)
        memory.episodic.add_memory(
            npc_id=npc.npc_id,
            description=desc,
            category=category,
            importance=importance,
            game_time=0.0,
            location_x=npc.home_x,
            location_z=npc.home_z,
        )
        count += 1

    # Universal town knowledge
    for desc, category, importance in UNIVERSAL_SEED_MEMORIES:
        memory.episodic.add_memory(
            npc_id=npc.npc_id,
            description=desc,
            category=category,
            importance=importance,
            game_time=0.0,
        )
        count += 1

    # Backstory as a memory
    memory.episodic.add_memory(
        npc_id=npc.npc_id,
        description=npc.backstory,
        category="identity",
        importance=0.8,
        game_time=0.0,
    )
    count += 1

    # Seed structured goals
    for goal in npc.long_term_goals:
        memory.structured.add_goal(
            npc_id=npc.npc_id,
            description=goal,
            importance=0.8,
            game_time=0.0,
        )

    # Seed identity fact
    memory.structured.add_fact(
        npc_id=npc.npc_id,
        subject=npc.name,
        predicate="is_a",
        obj=npc.occupation,
        confidence=1.0,
        source="identity",
        game_time=0.0,
    )

    return count


def seed_relationship_facts(
    npcs: list[NPC],
    memory: MemoryManager,
    sentiment: SentimentTracker | None = None,
    rng: random.Random | None = None,
) -> int:
    """
    Seed inter-NPC relationship facts, episodic memories, and initial
    sentiment. Returns the number of relationships created.

    Three sources of relationships:
    1. Occupational bonds (blacksmith ↔ merchant, farmer ↔ tavern_keeper)
    2. Neighbours (NPCs with nearby homes)
    3. Random acquaintances (everyone knows a few people)
    """
    rng = rng or random.Random(42)
    npc_map = {n.npc_id: n for n in npcs}
    count = 0

    # 1. Occupational bonds
    for bond in OCCUPATIONAL_BONDS:
        occ_a, occ_b = bond["occupations"]
        npcs_a = [n for n in npcs if n.occupation == occ_a]
        npcs_b = [n for n in npcs if n.occupation == occ_b]

        for a in npcs_a:
            for b in npcs_b:
                if a.npc_id == b.npc_id:
                    continue
                _seed_one_relationship(
                    a, b, bond["predicate"],
                    bond["desc"], bond["sentiment"],
                    memory, sentiment,
                )
                count += 1

    # 2. Neighbours (homes within 5 tiles)
    for i, a in enumerate(npcs):
        for b in npcs[i + 1:]:
            dist = abs(a.home_x - b.home_x) + abs(a.home_z - b.home_z)
            if dist <= 5:
                desc = rng.choice(NEIGHBOUR_DESCS)
                _seed_one_relationship(
                    a, b, "neighbour_of", desc,
                    {"trust": 10, "affection": 10},
                    memory, sentiment,
                )
                # Bidirectional
                _seed_one_relationship(
                    b, a, "neighbour_of", desc,
                    {"trust": 10, "affection": 10},
                    memory, sentiment,
                )
                count += 2

    # 3. Random acquaintances — each NPC knows 1-2 random others
    for npc in npcs:
        others = [n for n in npcs if n.npc_id != npc.npc_id]
        if not others:
            continue
        acquaintances = rng.sample(others, min(2, len(others)))
        for other in acquaintances:
            # Skip if already connected
            existing = memory.structured.get_facts_about(
                npc.npc_id, about=other.name,
            )
            already = any(
                f.predicate in ("trades_with", "supplies", "respects",
                                "knows_well", "works_for", "neighbour_of",
                                "knows")
                for f in existing
            )
            if already:
                continue
            desc = (
                f"{npc.name} knows {other.name} the {other.occupation} "
                f"from around town."
            )
            _seed_one_relationship(
                npc, other, "knows", desc,
                {"trust": 5},
                memory, sentiment,
            )
            count += 1

    logger.info("Seeded %d inter-NPC relationships", count)
    return count


def _seed_one_relationship(
    npc_a: NPC,
    npc_b: NPC,
    predicate: str,
    desc_template: str,
    sentiment_values: dict[str, float],
    memory: MemoryManager,
    sentiment: SentimentTracker | None = None,
) -> None:
    """Seed a single directional relationship: fact + memory + sentiment."""
    desc = desc_template.format(
        a_name=npc_a.name, b_name=npc_b.name,
        a_occ=npc_a.occupation, b_occ=npc_b.occupation,
    )

    # Structured fact: ("Alice", "trades_with", "Bob")
    memory.structured.add_fact(
        npc_id=npc_a.npc_id,
        subject=npc_a.name,
        predicate=predicate,
        obj=npc_b.name,
        confidence=1.0,
        source="relationship",
        game_time=0.0,
    )

    # Also store a fact about the other's occupation for retrieval
    memory.structured.add_fact(
        npc_id=npc_a.npc_id,
        subject=npc_b.name,
        predicate="is_a",
        obj=npc_b.occupation,
        confidence=1.0,
        source="relationship",
        game_time=0.0,
    )

    # Episodic memory of the relationship
    memory.episodic.add_memory(
        npc_id=npc_a.npc_id,
        description=desc,
        category="relationship",
        importance=0.7,
        game_time=0.0,
        location_x=npc_a.home_x,
        location_z=npc_a.home_z,
    )

    # Initial sentiment
    if sentiment and sentiment_values:
        from core.relationships.sentiment import Sentiment
        s = sentiment.get(npc_a.npc_id, npc_b.npc_id)
        for dim, value in sentiment_values.items():
            current = s.get(dim)
            s.set(dim, current + value)
        sentiment.set(s, game_time=0.0)


def seed_population_memories(
    npcs: list[NPC],
    memory: MemoryManager,
    sentiment: SentimentTracker | None = None,
    rng: random.Random | None = None,
) -> None:
    """Seed memories and relationships for all NPCs in the population."""
    total = 0
    for npc in npcs:
        total += seed_npc_memories(npc, memory)

    # Seed inter-NPC relationships
    rel_count = seed_relationship_facts(npcs, memory, sentiment, rng)

    logger.info(
        "Seeded memories for %d NPCs (%d episodic, %d relationships)",
        len(npcs), total, rel_count,
    )
