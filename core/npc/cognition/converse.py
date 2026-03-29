"""
Conversation system.

NPCs decide whether to engage in conversation, generate dialogue,
and record conversation outcomes. Conversations are 2-4 exchanges
between two NPCs within speaking distance.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.npc.models import ActivityState

if TYPE_CHECKING:
    from core.npc.llm_client import LLMProvider
    from core.npc.models import NPC
    from core.relationships.sentiment import SentimentTracker

logger = logging.getLogger(__name__)

# NPCs must be within this distance to talk
CONVERSATION_RANGE = 3  # tiles (Manhattan distance)

# Max exchanges per conversation
MAX_EXCHANGES = 4


@dataclass
class ConversationExchange:
    """A single line of dialogue in a conversation."""
    speaker_id: str
    speaker_name: str
    message: str


@dataclass
class Conversation:
    """An active or completed conversation between two NPCs."""
    npc_a_id: str
    npc_b_id: str
    exchanges: list[ConversationExchange] = field(default_factory=list)
    finished: bool = False

    def add_exchange(self, speaker_id: str, speaker_name: str, message: str) -> None:
        self.exchanges.append(ConversationExchange(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            message=message,
        ))

    def to_dict(self) -> dict:
        return {
            "participants": [self.npc_a_id, self.npc_b_id],
            "exchanges": [
                {"speaker": e.speaker_name, "message": e.message}
                for e in self.exchanges
            ],
            "finished": self.finished,
        }


# Active conversations tracked by frozenset of participant IDs
_active_conversations: dict[frozenset[str], Conversation] = {}

# Module-level sentiment tracker (set by NPCManager at init)
_sentiment_tracker: SentimentTracker | None = None


def set_sentiment_tracker(tracker: SentimentTracker | None) -> None:
    """Inject sentiment tracker for conversation initiation decisions."""
    global _sentiment_tracker
    _sentiment_tracker = tracker


def get_active_conversations() -> list[Conversation]:
    """Return all currently active conversations."""
    return [c for c in _active_conversations.values() if not c.finished]


def clear_finished_conversations() -> int:
    """Remove finished conversations. Returns count removed."""
    to_remove = [k for k, v in _active_conversations.items() if v.finished]
    for k in to_remove:
        del _active_conversations[k]
    return len(to_remove)


def should_initiate_conversation(
    npc: NPC,
    other: NPC,
    current_game_minutes: float,
) -> bool:
    """
    Decide if an NPC should try to start a conversation.

    Considers: distance, cooldowns, personality, current activity.
    """
    # Already in a conversation
    if npc.conversation_partner or other.conversation_partner:
        return False

    # Too far apart
    if npc.distance_to(other.x, other.z) > CONVERSATION_RANGE:
        return False

    # Cooldown not expired
    if current_game_minutes - npc.last_conversation_time < npc.conversation_cooldown:
        return False
    if current_game_minutes - other.last_conversation_time < other.conversation_cooldown:
        return False

    # Don't interrupt sleeping or important work
    busy_states = {ActivityState.SLEEPING}
    if npc.activity in busy_states or other.activity in busy_states:
        return False

    # Personality-based probability: extraverts initiate more
    base_chance = 0.3
    extraversion_bonus = npc.personality.extraversion * 0.4
    chance = base_chance + extraversion_bonus

    # Sentiment modifier: strong positive feelings increase chance,
    # strong negative feelings decrease it (but don't eliminate — rivals talk too)
    if _sentiment_tracker is not None:
        sent = _sentiment_tracker.get(npc.npc_id, other.npc_id)
        disposition = sent.overall_disposition()
        # ±0.2 max adjustment
        chance += max(-0.2, min(0.2, disposition / 100))

    return random.random() < chance


async def initiate_conversation(
    npc: NPC,
    other: NPC,
    llm: LLMProvider,
    current_game_minutes: float,
    memory_manager: object | None = None,
    grid: object | None = None,
    all_npcs: list | None = None,
) -> Conversation | None:
    """
    Start a conversation between two NPCs.

    Moves both NPCs to adjacent tiles before dialogue begins.
    Returns the Conversation object, or None if initiation fails.
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config

    config = get_tier_config(npc.cognition_tier)
    key = frozenset({npc.npc_id, other.npc_id})

    # Don't duplicate
    if key in _active_conversations and not _active_conversations[key].finished:
        return _active_conversations[key]

    conv = Conversation(npc_a_id=npc.npc_id, npc_b_id=other.npc_id)

    if config.uses_llm:
        try:
            # Pull relationship context from memory if available
            relationship_context = "You know them as a fellow townsperson."
            if memory_manager is not None:
                try:
                    relationship_context = memory_manager.get_relationship_context(
                        npc.npc_id, other.name,
                    )
                except Exception:
                    pass

            prompt = format_prompt(
                "conversation_initiate",
                name=npc.name,
                age=npc.age,
                occupation=npc.occupation,
                personality=npc.personality.to_description(),
                backstory=npc.backstory,
                other_name=other.name,
                other_occupation=other.occupation,
                relationship_context=relationship_context,
                recent_perceptions="; ".join(npc.recent_perceptions[-3:]) or "nothing notable",
            )

            message = await llm.complete(
                system="You are a medieval NPC having a casual conversation.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.8,
                purpose="conversation",
            )
        except Exception as e:
            logger.warning("Conversation initiation failed for %s: %s", npc.name, e)
            message = _fallback_greeting(npc)
    else:
        message = _fallback_greeting(npc)

    conv.add_exchange(npc.npc_id, npc.name, message)

    # Move both NPCs to adjacent tiles before they start talking
    if grid is not None and all_npcs is not None:
        from core.world.spatial_awareness import (
            get_occupied_tiles, find_conversation_positions,
        )
        occupied = get_occupied_tiles(all_npcs)
        pos_a, pos_b = find_conversation_positions(
            npc, other, grid, occupied,
        )
        npc.x, npc.z = float(pos_a[0]), float(pos_a[1])
        other.x, other.z = float(pos_b[0]), float(pos_b[1])
        # Clear any in-progress paths
        npc.current_path = []
        npc.path_index = 0
        other.current_path = []
        other.path_index = 0

    # Set conversation state
    npc.conversation_partner = other.npc_id
    other.conversation_partner = npc.npc_id
    npc.activity = ActivityState.TALKING
    other.activity = ActivityState.TALKING
    npc.current_action_description = f"talking to {other.name}"
    other.current_action_description = f"talking to {npc.name}"

    _active_conversations[key] = conv

    logger.debug("%s started talking to %s", npc.name, other.name)
    return conv


async def continue_conversation(
    npc: NPC,
    other: NPC,
    llm: LLMProvider,
    memory_manager: object | None = None,
) -> bool:
    """
    Generate the next exchange in an active conversation.

    The responder is `npc`, responding to the last thing `other` said.
    Returns False if the conversation should end.
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config

    key = frozenset({npc.npc_id, other.npc_id})
    conv = _active_conversations.get(key)
    if conv is None or conv.finished:
        return False

    # Check if we've hit the exchange limit
    if len(conv.exchanges) >= MAX_EXCHANGES:
        await end_conversation(npc, other)
        return False

    config = get_tier_config(npc.cognition_tier)
    last_message = conv.exchanges[-1].message if conv.exchanges else ""

    if config.uses_llm:
        try:
            # Pull relationship context if available
            relationship_context = ""
            if memory_manager is not None:
                try:
                    relationship_context = memory_manager.get_relationship_context(
                        npc.npc_id, other.name, other_id=other.npc_id,
                    )
                except Exception:
                    pass

            prompt = format_prompt(
                "conversation_respond",
                name=npc.name,
                age=npc.age,
                occupation=npc.occupation,
                personality=npc.personality.to_description(),
                other_name=other.name,
                other_occupation=other.occupation,
                other_message=last_message,
                relationship_context=relationship_context or "You know them as a fellow townsperson.",
            )

            message = await llm.complete(
                system="You are a medieval NPC responding in conversation.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.8,
                purpose="conversation",
            )
        except Exception:
            message = _fallback_response(npc)
    else:
        message = _fallback_response(npc)

    conv.add_exchange(npc.npc_id, npc.name, message)

    # Random chance to end after each exchange (increases with count)
    end_chance = len(conv.exchanges) / (MAX_EXCHANGES + 2)
    if random.random() < end_chance:
        await end_conversation(npc, other)
        return False

    return True


async def end_conversation(
    npc: NPC,
    other: NPC,
    current_game_minutes: float = 0,
) -> None:
    """End an active conversation between two NPCs."""
    key = frozenset({npc.npc_id, other.npc_id})
    conv = _active_conversations.get(key)
    if conv:
        conv.finished = True

    npc.conversation_partner = None
    other.conversation_partner = None
    npc.activity = ActivityState.IDLE
    other.activity = ActivityState.IDLE
    npc.current_action_description = ""
    other.current_action_description = ""

    if current_game_minutes > 0:
        npc.last_conversation_time = current_game_minutes
        other.last_conversation_time = current_game_minutes

    logger.debug(
        "%s finished talking to %s (%d exchanges)",
        npc.name, other.name,
        len(conv.exchanges) if conv else 0,
    )


def _fallback_greeting(npc: NPC) -> str:
    """Simple canned greeting for tier 3+ NPCs."""
    greetings = [
        f"Good day! I'm {npc.name}, the {npc.occupation}.",
        "Hello there, fine day isn't it?",
        "Greetings, friend. How goes your work?",
        f"Ah, good to see a friendly face. {npc.name}, at your service.",
        "Well met! What brings you this way?",
    ]
    return random.choice(greetings)


def _fallback_response(npc: NPC) -> str:
    """Simple canned response for tier 3+ NPCs."""
    responses = [
        "Indeed, quite so.",
        "Aye, I suppose you're right about that.",
        "Interesting. I hadn't thought of it that way.",
        "Ha! You always know what to say.",
        "Well, I must be getting on with things soon.",
        "That's good to hear. Things are well enough on my end too.",
    ]
    return random.choice(responses)
