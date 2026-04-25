"""
Conversation system.

NPCs decide whether to engage in conversation, generate dialogue,
and record conversation outcomes. Conversations are 2-4 exchanges
between two NPCs within speaking distance.
"""

from __future__ import annotations

import logging
import random
import uuid
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
    """An active or completed conversation between two NPCs.

    `conv_id` is a process-unique id used to tag per-turn memories so
    the consolidation sweep (see MemoryManager.consolidate_conversation_turns)
    can find and remove them once the final summary is written.
    """
    npc_a_id: str
    npc_b_id: str
    exchanges: list[ConversationExchange] = field(default_factory=list)
    finished: bool = False
    conv_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Count of exchanges already persisted as per-turn memories, so
    # the server's after-reply hook can write only the new ones.
    persisted_exchange_count: int = 0
    # Set to True once `_persist_finished_conversations` has made a
    # persistence attempt (successful OR failed). Without this flag,
    # a `record_conversation` exception would re-crash every tick
    # because the conversation stays in `_active_conversations`.
    # `clear_finished_conversations` also removes persisted convos.
    persisted: bool = False

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
            "conv_id": self.conv_id,
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


# How many trailing exchanges to pass into the response prompt as
# short-term conversational context. Five is enough to hold the
# shape of a back-and-forth without bloating the prompt. The
# newest exchange — the one being responded to — is excluded
# because it's rendered separately as `other_message`.
_RECENT_HISTORY_TURNS = 5


def _format_recent_history(
    exchanges: list["ConversationExchange"],
) -> str:
    """Render recent turns as a "Recent conversation so far" block.

    Returns the empty string when there aren't enough prior turns to
    bother with (a brand-new chat has only the line we're replying
    to). Formats with speaker names so the LLM can follow who said
    what, and keeps each line terse so the block stays compact.
    """
    if not exchanges or len(exchanges) < 2:
        return ""
    # Drop the final exchange — that's the message being replied to
    # and is rendered separately in the prompt. Take the N before it.
    tail = exchanges[:-1][-_RECENT_HISTORY_TURNS:]
    if not tail:
        return ""
    lines = [
        f"{e.speaker_name}: \"{e.message}\""
        for e in tail
    ]
    return (
        "Recent conversation so far:\n"
        + "\n".join(lines)
        + "\n\n"
    )


def clear_finished_conversations() -> int:
    """Remove finished conversations that have been persisted (or
    whose persistence was attempted and swallowed). Returns the
    number removed.

    A conversation stays in `_active_conversations` if
    `finished=True` but `persisted=False` — which only happens
    between the moment `end_conversation` flips `finished` and the
    NEXT cognition tick's persistence step. Once a persistence
    attempt runs (success or logged failure), `persisted=True` is
    set and this cleanup removes it.
    """
    to_remove = [
        k for k, v in _active_conversations.items()
        if v.finished and v.persisted
    ]
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

    # Don't interrupt sleeping or walking NPCs.
    # Walking interruption was the #1 cause of oscillation — NPCs get
    # teleported mid-walk, dispatched back, meet again, repeat forever.
    busy_states = {ActivityState.SLEEPING, ActivityState.WALKING}
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

    return npc._rng.random() < chance


async def initiate_conversation(
    npc: NPC,
    other: NPC,
    llm: LLMProvider,
    current_game_minutes: float,
    memory_manager: object | None = None,
    grid: object | None = None,
    all_npcs: list | None = None,
    town_agenda_summary: str = "",
    unresolved_matters_summary: str = "",
    shared_agenda_summary: str = "",
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
                self_concept=npc.self_concept_summary(),
                town_agenda=town_agenda_summary,
                shared_agenda=shared_agenda_summary,
                unresolved_matters=unresolved_matters_summary,
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


def start_player_conversation(
    npc: NPC,
    player: NPC,
    player_message: str,
) -> Conversation:
    """Start (or reuse) a conversation initiated by the player.

    Unlike initiate_conversation, this skips the NPC-greeting LLM call —
    the NPC responds directly to the player's first message via a
    subsequent call to continue_conversation. One LLM call per round,
    not two. Critical for responsiveness: the greeting call uses Gemma's
    thinking mode (1024 tokens, ~25s) and is unnecessary when the
    player has already opened the conversation with their own line.

    Returns the Conversation (new or existing-not-finished).
    """
    key = frozenset({player.npc_id, npc.npc_id})
    existing = _active_conversations.get(key)
    if existing and not existing.finished:
        existing.add_exchange(player.npc_id, player.name, player_message)
        return existing

    conv = Conversation(npc_a_id=player.npc_id, npc_b_id=npc.npc_id)
    conv.add_exchange(player.npc_id, player.name, player_message)
    _active_conversations[key] = conv

    # Set conversation state on the NPC so other systems don't
    # pull them away (post-convo dispatch, tier promotion, etc.).
    npc.conversation_partner = player.npc_id
    player.conversation_partner = npc.npc_id
    npc.activity = ActivityState.TALKING
    npc.current_action_description = f"talking to {player.name}"
    # Stop any in-flight schedule walk. Without this, an NPC who was
    # mid-path when the player opened chat would keep walking during
    # the LLM call and could leave interaction range before the reply
    # even arrives — the proximity check then closes the chat and the
    # player never sees a response to their first message.
    npc.current_path = []
    npc.path_index = 0
    # Player's activity is driven by input — do not force TALKING.

    logger.debug("%s opened conversation with %s", player.name, npc.name)
    return conv


async def continue_conversation(
    npc: NPC,
    other: NPC,
    llm: LLMProvider,
    memory_manager: object | None = None,
    allow_auto_end: bool = True,
    max_exchanges: int | None = None,
    town_agenda_summary: str = "",
    unresolved_matters_summary: str = "",
    shared_agenda_summary: str = "",
    force_llm: bool = False,
) -> bool:
    """
    Generate the next exchange in an active conversation.

    The responder is `npc`, responding to the last thing `other` said.
    Returns False if the conversation should end.

    Args:
        allow_auto_end: If False, suppress the random end-after-exchange
            roll. Player chats pass False so the conversation doesn't
            randomly end while the player is actively engaging — only
            explicit close, walking out of range, or max_exchanges ends it.
        max_exchanges: Override the default exchange cap. Player chats
            pass a higher value so a single chat window stays usable.
        force_llm: Require the LLM path regardless of this NPC's
            cognition tier, and SURFACE any LLM exception (don't
            silently fall back to a canned string). Set by player
            chats — if the player is typing at an NPC, a mock
            response like "Indeed, quite so." is a worse experience
            than an honest error; the whole point of player-NPC
            interaction is that the NPC actually engages with what
            was said. A tier-update race (NPC still at tier 3 when
            focus hasn't caught up to the player's latest step)
            used to drop player chats onto the canned-fallback path
            and produce the infamous "Indeed, quite so." reply.
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config

    cap = max_exchanges if max_exchanges is not None else MAX_EXCHANGES
    key = frozenset({npc.npc_id, other.npc_id})
    conv = _active_conversations.get(key)
    if conv is None or conv.finished:
        return False

    # Check if we've hit the exchange limit
    if len(conv.exchanges) >= cap:
        await end_conversation(npc, other, memory_manager=memory_manager)
        return False

    config = get_tier_config(npc.cognition_tier)
    last_message = conv.exchanges[-1].message if conv.exchanges else ""

    # Build a compact history block covering the last few exchanges
    # before the line the NPC is replying to. Without this, the LLM
    # only ever sees the single latest utterance and loses the thread
    # of a multi-turn conversation — NPCs appeared to forget what was
    # just discussed, because they literally had no prior context in
    # the prompt.
    recent_history = _format_recent_history(conv.exchanges)

    use_llm = config.uses_llm or force_llm
    if use_llm:
        # Pull relationship context if available (cheap; errors
        # here are cosmetic and shouldn't abort the chat).
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
            self_concept=npc.self_concept_summary(),
            town_agenda=town_agenda_summary,
            shared_agenda=shared_agenda_summary,
            unresolved_matters=unresolved_matters_summary,
            other_name=other.name,
            other_occupation=other.occupation,
            other_message=last_message,
            recent_history=recent_history,
            relationship_context=relationship_context or "You know them as a fellow townsperson.",
        )

        try:
            message = await llm.complete(
                system="You are a medieval NPC responding in conversation.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.8,
                purpose="conversation",
            )
        except Exception:
            if force_llm:
                # Player chats must NEVER silently fall back to the
                # canned "Indeed, quite so." string — that produced
                # a misleading UX where the player couldn't tell a
                # real reply from a stub. Surface the error so the
                # player sees something is wrong and the log has
                # the real cause.
                logger.exception(
                    "Player-chat LLM call failed for %s — surfacing "
                    "the error rather than stub-replying",
                    npc.name,
                )
                raise
            message = _fallback_response(npc)
    else:
        message = _fallback_response(npc)

    conv.add_exchange(npc.npc_id, npc.name, message)

    if allow_auto_end:
        # Random chance to end after each exchange (increases with count)
        end_chance = len(conv.exchanges) / (cap + 2)
        if npc._rng.random() < end_chance:
            await end_conversation(npc, other, memory_manager=memory_manager)
            return False

    return True


def _conversation_sentiment_deltas(
    npc: NPC, other: NPC, exchange_count: int,
) -> dict[str, float]:
    """Compute sentiment changes from a conversation using heuristics.

    Trust: grows with every conversation (people who talk build trust).
    Affection: grows slightly per exchange (time spent = bonding).
    Resonance: boost if shared occupation or overlapping skills.
    Respect: small boost — you respect someone who takes time to talk.
    """
    deltas: dict[str, float] = {}

    # Trust: base +2, +0.5 per exchange beyond the first
    deltas["trust"] = 2.0 + max(0, exchange_count - 1) * 0.5

    # Affection: +1 base, +0.3 per exchange
    deltas["affection"] = 1.0 + exchange_count * 0.3

    # Respect: small flat boost
    deltas["respect"] = 1.0

    # Resonance: shared occupation = strong, overlapping skills = moderate
    if npc.occupation == other.occupation:
        deltas["resonance"] = 5.0
    else:
        shared_skills = set(npc.skills.keys()) & set(other.skills.keys())
        if shared_skills:
            deltas["resonance"] = len(shared_skills) * 1.5

    # --- Negative signals (personality clashes) ---
    # Large agreeableness gap → friction (blunt vs cooperative)
    agree_gap = abs(npc.personality.agreeableness - other.personality.agreeableness)
    if agree_gap > 0.5:
        deltas["affection"] -= agree_gap * 1.5  # up to -1.5

    # High neuroticism on either side → trust erosion
    max_neuroticism = max(npc.personality.neuroticism, other.personality.neuroticism)
    if max_neuroticism > 0.7:
        deltas["trust"] -= (max_neuroticism - 0.5) * 1.0  # up to -0.5

    # Very different openness → slight respect penalty (can't relate)
    open_gap = abs(npc.personality.openness - other.personality.openness)
    if open_gap > 0.6:
        deltas["respect"] -= open_gap * 0.5  # up to -0.5

    return deltas


async def end_conversation(
    npc: NPC,
    other: NPC,
    current_game_minutes: float = 0,
    memory_manager: object | None = None,
) -> None:
    """End an active conversation and store it in both NPCs' memory."""
    key = frozenset({npc.npc_id, other.npc_id})
    conv = _active_conversations.get(key)
    if conv:
        conv.finished = True
        exchange_count = len(conv.exchanges)

        # Store conversation in memory for both participants
        if memory_manager is not None and conv.exchanges:
            try:
                await memory_manager.record_conversation(
                    npc_a_id=npc.npc_id,
                    npc_b_id=other.npc_id,
                    npc_a_name=npc.name,
                    npc_b_name=other.name,
                    exchanges=[
                        {"speaker": e.speaker_name, "message": e.message}
                        for e in conv.exchanges
                    ],
                    game_time=current_game_minutes,
                    location_x=int(npc.x),
                    location_z=int(npc.z),
                )
            except Exception as e:
                logger.warning(
                    "Failed to record conversation %s↔%s: %s",
                    npc.name, other.name, e,
                )

            # Update sentiment for both participants
            if memory_manager.sentiment is not None and exchange_count > 0:
                deltas = _conversation_sentiment_deltas(
                    npc, other, exchange_count,
                )
                for dim, delta in deltas.items():
                    memory_manager.sentiment.modify_mutual(
                        npc.npc_id, other.npc_id, dim, delta,
                        game_time=current_game_minutes,
                    )

    npc.conversation_partner = None
    other.conversation_partner = None
    npc.activity = ActivityState.IDLE
    other.activity = ActivityState.IDLE
    npc.current_action_description = ""
    other.current_action_description = ""
    # Flag for post-conversation dispatch (manager picks up schedule)
    npc._needs_post_convo_dispatch = True
    other._needs_post_convo_dispatch = True

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
    return npc._rng.choice(greetings)


_FALLBACK_RESPONSES: tuple[str, ...] = (
    "Indeed, quite so.",
    "Aye, I suppose you're right about that.",
    "Interesting. I hadn't thought of it that way.",
    "Ha! You always know what to say.",
    "Well, I must be getting on with things soon.",
    "That's good to hear. Things are well enough on my end too.",
)

# Exposed for the `test_player_chat_never_canned` regression gate.
# Renaming the alias is fine; do not rename the tuple itself without
# updating that test.
_FALLBACK_RESPONSES_FOR_TEST = _FALLBACK_RESPONSES


def _fallback_response(npc: NPC) -> str:
    """Simple canned response for tier 3+ NPCs."""
    return npc._rng.choice(_FALLBACK_RESPONSES)
