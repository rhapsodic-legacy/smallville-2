"""
Seed memories — foundational memories planted at NPC creation.

Based on Stanford Generative Agents' approach where seed memories
drive initial behaviour. Each NPC wakes up with identity, craft
knowledge, motivation, and community awareness.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.npc.models import SEED_MEMORIES, UNIVERSAL_SEED_MEMORIES

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


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


def seed_population_memories(
    npcs: list[NPC], memory: MemoryManager,
) -> None:
    """Seed memories for all NPCs in the population."""
    total = 0
    for npc in npcs:
        total += seed_npc_memories(npc, memory)

    logger.info(
        "Seeded memories for %d NPCs (%d total episodic memories)",
        len(npcs), total,
    )
