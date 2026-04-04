"""
Reflection system.

Generates higher-level insights from accumulated experiences.
Triggered when the importance accumulator exceeds a threshold.
Produces focal points, synthesises insights via LLM, and stores
results back as high-importance episodic memories.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.manager import MemoryManager
    from core.npc.llm_client import LLMProvider
    from core.npc.models import NPC

logger = logging.getLogger(__name__)

# Template-based reflections for when LLM is unavailable
_CONVERSATION_REFLECTION_TEMPLATES = [
    "I enjoyed talking with {other_name}. We should speak again soon.",
    "That conversation with {other_name} gave me something to think about.",
    "{other_name} seems like someone I can trust. Or perhaps I need to be more careful.",
    "I wonder what {other_name} really thinks about life in Smallville.",
    "Talking with {other_name} reminded me why I became a {occupation}.",
    "I should remember what {other_name} said — it might matter later.",
]

_GENERAL_REFLECTION_TEMPLATES = [
    "I've been busy lately. I should take stock of how things are going.",
    "Life as a {occupation} in Smallville has its rhythms. I'm settling into mine.",
    "I've noticed some changes around town. Things feel different lately.",
    "I should think about what I really want for my future here.",
    "The people of Smallville are interesting. I'm learning who to trust.",
]

# Prompt templates for reflection
FOCAL_POINT_PROMPT = (
    "You are {name}, a {occupation} in Smallville.\n"
    "Here are your most notable recent experiences:\n"
    "{experiences}\n\n"
    "Given these experiences, what are the 3 most important questions "
    "or themes you should think about? Be specific and personal.\n"
    "List them, one per line."
)

INSIGHT_PROMPT = (
    "You are {name}, a {occupation} in Smallville.\n"
    "Personality: {personality}\n\n"
    "You are reflecting on this question: {focal_point}\n\n"
    "Relevant memories:\n{memories}\n\n"
    "What insight or conclusion do you draw? "
    "Answer in 1-2 sentences, in first person, as {name}."
)

POST_CONVERSATION_PROMPT = (
    "You are {name}, a {occupation} in Smallville.\n"
    "You just finished talking with {other_name}.\n\n"
    "The conversation:\n{conversation}\n\n"
    "What did you learn or feel from this conversation? "
    "Answer in 1 sentence, in first person."
)


async def run_reflection(
    npc: NPC,
    memory: MemoryManager,
    llm: LLMProvider,
    current_game_time: float,
) -> list[str]:
    """
    Run a full reflection cycle for an NPC.

    1. Generate focal points from high-importance recent memories
    2. For each focal point, retrieve relevant memories
    3. Synthesise an insight via LLM
    4. Store insights as high-importance memories

    Returns the list of generated insights.
    """
    focal_points = memory.get_focal_points(npc.npc_id, limit=3)
    if not focal_points:
        logger.debug("%s has no focal points for reflection", npc.name)
        return []

    # Generate thematic focal points via LLM
    focal_questions = await _generate_focal_questions(
        npc, focal_points, llm,
    )

    insights: list[str] = []
    for question in focal_questions:
        # Retrieve memories relevant to this focal point
        context = memory.retrieve_context(
            npc_id=npc.npc_id,
            query=question,
            cognition_tier=1,  # full retrieval for reflection
            current_game_time=current_game_time,
        )

        # Synthesise insight
        insight = await _synthesise_insight(
            npc, question, context.to_prompt_text(), llm,
        )

        if insight:
            await memory.record_reflection(
                npc_id=npc.npc_id,
                insight=insight,
                game_time=current_game_time,
            )
            insights.append(insight)

    # If LLM produced no insights, generate a template fallback
    # so reflection still advances the importance accumulator
    if not insights:
        fallback = random.choice(_GENERAL_REFLECTION_TEMPLATES).format(
            occupation=npc.occupation,
        )
        await memory.record_reflection(
            npc_id=npc.npc_id,
            insight=fallback,
            game_time=current_game_time,
        )
        insights.append(fallback)

    logger.info(
        "%s reflected and generated %d insights", npc.name, len(insights),
    )
    return insights


async def reflect_on_conversation(
    npc: NPC,
    other_name: str,
    exchanges: list[dict[str, str]],
    memory: MemoryManager,
    llm: LLMProvider,
    current_game_time: float,
) -> str | None:
    """
    Quick reflection after a conversation ends.

    Generates a single insight about what was discussed/learnt.
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config

    config = get_tier_config(npc.cognition_tier)
    if not config.uses_llm:
        return None

    conversation_text = "\n".join(
        f"{e.get('speaker', '?')}: {e.get('message', '')}"
        for e in exchanges
    )

    try:
        prompt = POST_CONVERSATION_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            other_name=other_name,
            conversation=conversation_text,
        )

        insight = await llm.complete(
            system="You are a medieval NPC reflecting on a conversation.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
            purpose="reflection",
        )

        insight = insight.strip()
        if insight:
            await memory.record_reflection(
                npc_id=npc.npc_id,
                insight=insight,
                game_time=current_game_time,
            )

        return insight

    except Exception as e:
        logger.warning("Post-conversation reflection failed for %s: %s", npc.name, e)
        # Template fallback so NPCs always reflect after conversations
        insight = random.choice(_CONVERSATION_REFLECTION_TEMPLATES).format(
            other_name=other_name, occupation=npc.occupation,
        )
        await memory.record_reflection(
            npc_id=npc.npc_id,
            insight=insight,
            game_time=current_game_time,
        )
        return insight


async def _generate_focal_questions(
    npc: NPC,
    experiences: list[str],
    llm: LLMProvider,
) -> list[str]:
    """Generate reflection focal points from recent experiences."""
    try:
        prompt = FOCAL_POINT_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            experiences="\n".join(f"- {e}" for e in experiences),
        )

        response = await llm.complete(
            system="You are helping a medieval NPC identify themes for reflection.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            purpose="reflection",
        )

        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.strip().split("\n")
            if line.strip()
        ]
        return lines[:3]

    except Exception as e:
        logger.warning("Focal point generation failed for %s: %s", npc.name, e)
        # Fallback: use the raw experiences as focal points
        return experiences[:3]


async def _synthesise_insight(
    npc: NPC,
    focal_point: str,
    memory_text: str,
    llm: LLMProvider,
) -> str | None:
    """Generate an insight for a single focal point."""
    try:
        prompt = INSIGHT_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            personality=npc.personality.to_description(),
            focal_point=focal_point,
            memories=memory_text,
        )

        response = await llm.complete(
            system="You are a medieval NPC synthesising an insight from memories.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
            purpose="reflection",
        )

        return response.strip() or None

    except Exception as e:
        logger.warning("Insight synthesis failed for %s: %s", npc.name, e)
        return None
