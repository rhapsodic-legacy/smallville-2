"""
Reflection system.

Generates higher-level insights from accumulated experiences.
Triggered when the importance accumulator exceeds a threshold.
Produces focal points, synthesises insights via LLM, and stores
results back as high-importance episodic memories.

Action intents: insights classified as action_intent trigger
dynamic schedule injection — the NPC gets a temporary schedule
entry inserted so they act on their reflection.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
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

ACTION_INTENT_PROMPT = (
    "You are {name}, a {occupation} in Smallville.\n"
    "You just had this insight:\n\"{insight}\"\n\n"
    "Does this insight suggest a specific physical action you should take "
    "RIGHT NOW? Examples: bringing someone food, visiting a friend, checking "
    "on the forge, delivering a message, going to see something.\n\n"
    "If yes, respond with EXACTLY this format (one line each):\n"
    "ACTION: <short activity description>\n"
    "LOCATION: <one of: home, work, tavern, town_square, market_stall, "
    "church, farm, outskirts, or a person's name>\n"
    "DURATION: <minutes, between 15 and 60>\n\n"
    "If the insight is purely internal (a feeling, opinion, or abstract "
    "thought with no immediate physical action), respond with:\n"
    "NO_ACTION"
)

# Heuristic keywords that suggest an actionable intent (for non-LLM tiers)
_ACTION_KEYWORDS = [
    "should bring", "should visit", "should check", "should go",
    "should deliver", "should talk to", "should help", "should see",
    "need to bring", "need to visit", "need to check", "need to go",
    "must bring", "must visit", "must check", "must go",
    "I will bring", "I will visit", "I will go", "I will check",
    "want to bring", "want to visit", "want to see",
]


@dataclass
class ActionIntent:
    """An actionable intent extracted from a reflection insight."""
    activity: str       # e.g. "bring lunch to Bob at the bridge"
    location: str       # symbolic location or NPC name
    duration_minutes: int  # 15–60


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


async def run_reflection_with_intents(
    npc: NPC,
    memory: MemoryManager,
    llm: LLMProvider,
    current_game_time: float,
) -> tuple[list[str], list[ActionIntent]]:
    """
    Run reflection and classify each insight for action intents.

    Returns (insights, action_intents). Action intents are insights
    that imply a physical action the NPC should take — these get
    injected into the NPC's schedule by the manager.
    """
    insights = await run_reflection(npc, memory, llm, current_game_time)

    action_intents: list[ActionIntent] = []
    for insight in insights:
        intent = await classify_insight(npc, insight, llm)
        if intent:
            logger.info(
                "%s action intent: %s at %s (%d min)",
                npc.name, intent.activity, intent.location,
                intent.duration_minutes,
            )
            action_intents.append(intent)

    return insights, action_intents


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


# ---------- Action intent classification ----------

async def classify_insight(
    npc: NPC,
    insight: str,
    llm: LLMProvider,
) -> ActionIntent | None:
    """
    Classify whether an insight implies a physical action the NPC
    should take. Returns an ActionIntent if so, None otherwise.

    Uses LLM for Tier 1 NPCs, heuristic for others.
    """
    from core.npc.cognition.tiers import get_tier_config

    config = get_tier_config(npc.cognition_tier)
    if not config.uses_llm:
        return _classify_insight_heuristic(insight)

    try:
        prompt = ACTION_INTENT_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            insight=insight,
        )

        response = await llm.complete(
            system="You classify NPC reflections as actionable or not.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
            purpose="reflection",
        )

        return _parse_action_intent(response.strip())

    except Exception as e:
        logger.warning(
            "Action intent classification failed for %s: %s", npc.name, e,
        )
        return _classify_insight_heuristic(insight)


def _parse_action_intent(response: str) -> ActionIntent | None:
    """Parse the structured ACTION/LOCATION/DURATION response."""
    if "NO_ACTION" in response:
        return None

    action = ""
    location = ""
    duration = 30  # default

    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()
        elif line.upper().startswith("LOCATION:"):
            location = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("DURATION:"):
            raw = line.split(":", 1)[1].strip()
            # Extract digits
            digits = "".join(c for c in raw if c.isdigit())
            if digits:
                duration = max(15, min(60, int(digits)))

    if not action:
        return None

    # Default location if not provided
    if not location:
        location = "town_square"

    return ActionIntent(
        activity=action,
        location=location,
        duration_minutes=duration,
    )


def _classify_insight_heuristic(insight: str) -> ActionIntent | None:
    """Keyword-based fallback for non-LLM tiers."""
    lower = insight.lower()
    for keyword in _ACTION_KEYWORDS:
        if keyword in lower:
            # Extract a rough activity from the insight itself
            return ActionIntent(
                activity=insight[:80],
                location="town_square",
                duration_minutes=30,
            )
    return None


# ---------- Emotional valence → personality drift ----------

# Big-5 deltas per keyword category. Magnitudes are intentionally
# small; the manager scales them by importance so a 0.5-importance
# reflection moves the vector by ~0.01 and a 1.0 by ~0.03.
_VALENCE_RULES: list[tuple[tuple[str, ...], dict[str, float]]] = [
    # Joy / warmth
    (
        ("happy", "joy", "love", "grateful", "warm", "glad", "fond", "cherish"),
        {"extraversion": +1.0, "agreeableness": +1.0, "neuroticism": -1.0},
    ),
    # Pride / accomplishment
    (
        ("proud", "accomplished", "achieved", "mastered", "triumph", "succeed", "earned"),
        {"conscientiousness": +1.0, "neuroticism": -0.5, "extraversion": +0.5},
    ),
    # Curiosity / wonder
    (
        ("curious", "wonder", "explore", "discover", "new", "strange", "intrigued"),
        {"openness": +1.0},
    ),
    # Fear / anxiety
    (
        ("afraid", "fear", "scared", "anxious", "worried", "dread", "terrified"),
        {"neuroticism": +1.0, "openness": -0.5, "extraversion": -0.5},
    ),
    # Anger / hostility
    (
        ("angry", "furious", "hate", "resent", "betrayed", "enemy", "vengeance"),
        {"agreeableness": -1.0, "neuroticism": +0.5},
    ),
    # Sadness / grief
    (
        ("sad", "grief", "mourn", "lonely", "hopeless", "despair"),
        {"neuroticism": +1.0, "extraversion": -0.5},
    ),
    # Social warmth / connection
    (
        ("friend", "trust", "bond", "together", "companion", "ally"),
        {"agreeableness": +0.5, "extraversion": +0.5},
    ),
    # Isolation / withdrawal
    (
        ("alone", "avoid", "distance", "withdraw", "solitude"),
        {"extraversion": -1.0, "openness": -0.5},
    ),
]

# Base epsilon per matched rule — importance scales this up to ~3× so
# a 1.0-importance reflection nudges a trait by ~0.03, a 0.5-importance
# one nudges it by ~0.01. Under clamping at [0,1] this stays bounded.
_DRIFT_EPSILON: float = 0.01


def classify_emotional_valence(
    text: str,
    importance: float = 0.5,
) -> dict[str, float]:
    """Return a Big-5 delta dict from the emotional content of a text.

    Multiple rules may fire for the same text — their deltas sum. The
    result is scaled by (0.5 + importance), giving 0.5–1.5× at the
    normal memory importance range. Empty dict when nothing matches.
    """
    if not text:
        return {}

    lower = text.lower()
    combined: dict[str, float] = {}
    for keywords, deltas in _VALENCE_RULES:
        if any(kw in lower for kw in keywords):
            for trait, raw in deltas.items():
                combined[trait] = combined.get(trait, 0.0) + raw

    if not combined:
        return {}

    scale = _DRIFT_EPSILON * (0.5 + max(0.0, min(1.5, importance)))
    return {trait: delta * scale for trait, delta in combined.items()}


def apply_personality_drift(
    npc: NPC, text: str, importance: float = 0.5,
) -> dict[str, float]:
    """Mutate the NPC's personality from a reflection/conversation text.

    Returns the applied delta dict (may be empty). The NPC must have
    a personality field (guaranteed) — spawn_baseline is handled by
    the manager on a separate daily decay tick, not here.
    """
    deltas = classify_emotional_valence(text, importance)
    if deltas:
        npc.personality.apply_deltas(deltas)
    return deltas


# ---------- Identity-claim detection ----------

@dataclass
class IdentityClaim:
    """An identity proposition made about an NPC in conversation.

    key: namespaced self-concept key (e.g. "role:king", "enemy_of:bran_1").
    confidence_delta: how much to nudge that belief's confidence (±1.0).
    source_text: the raw line that triggered detection (for memory).
    speaker: who made the claim — used by the manager to reject self-
             flattery and weight strangers vs friends.
    """
    key: str
    confidence_delta: float
    source_text: str
    speaker: str = ""


# Patterns that identify claims about the listener. Each rule is
# (compiled regex, key_template, delta). The regex must produce at
# least one capture group; for role claims group(1) is the role,
# for relational claims group(1) is the target (who the role or
# feeling is toward). key_template is formatted with the captured
# group lowercased and whitespace-collapsed.
#
# Negative claims (enemy, betray) get smaller positive deltas because
# the NPC shouldn't internalise hostility from one line as fast as
# they internalise praise. The manager weights these further.
_IDENTITY_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    # "You are (a|the|our) king" / "you are king" — role assertion
    (
        re.compile(
            r"\byou\s+(?:are|'re)\s+(?:a|an|the|our|my)?\s*([a-zA-Z]{3,20})\b",
            re.IGNORECASE,
        ),
        "role:{target}",
        +0.4,
    ),
    # "You helped us/me win/...": helper
    (
        re.compile(
            r"\byou\s+(?:helped|saved|rescued|aided)\s+(us|me|the town|our people)",
            re.IGNORECASE,
        ),
        "helped:{target}",
        +0.4,
    ),
    # "You are my/our enemy|rival|foe"
    (
        re.compile(
            r"\byou\s+(?:are|'re)\s+(?:my|our)\s+(enemy|rival|foe|adversary)\b",
            re.IGNORECASE,
        ),
        "enemy_of:{speaker}",
        +0.3,
    ),
    # "You betrayed X"
    (
        re.compile(
            r"\byou\s+betrayed\s+(us|me|the town)",
            re.IGNORECASE,
        ),
        "betrayed:{target}",
        +0.3,
    ),
    # "You are my friend" / "my ally"
    (
        re.compile(
            r"\byou\s+(?:are|'re)\s+(?:my|our)\s+(friend|ally|companion)\b",
            re.IGNORECASE,
        ),
        "friend_of:{speaker}",
        +0.3,
    ),
]

# Words that follow "you are a/the/our ..." but are NOT a meaningful
# role — they'd produce noisy self-concepts. Filter them out.
_ROLE_STOPWORDS: frozenset[str] = frozenset({
    "good", "bad", "fine", "kind", "sure", "right", "wrong", "nice",
    "great", "terrible", "awful", "lovely", "just", "really", "very",
    "here", "there", "back", "late", "early", "alone", "together",
    "stranger", "person", "one", "someone", "somebody", "thing",
    "man", "woman", "boy", "girl", "fool", "idiot", "liar",
    "tired", "hungry", "sick", "hurt", "afraid", "scared",
    "you", "me", "us", "him", "her", "they", "it",
    "well", "okay", "ok", "today", "tomorrow", "now",
})


def _normalise_target(raw: str, speaker: str = "") -> str:
    """Canonical form for a captured role/target string."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if cleaned in ("us", "me", "the_town", "our_people", "i"):
        # Collective targets collapse to the speaker or 'town'.
        return speaker or "town"
    return cleaned


def detect_identity_claims(
    exchanges: list[dict[str, str]],
    listener_name: str,
    speaker_id: str = "",
) -> list[IdentityClaim]:
    """Scan a conversation for identity claims made about the listener.

    Only lines whose speaker is NOT the listener are scanned — self-
    claims don't seed the listener's self-concept. Returns a list of
    IdentityClaim, one per matched pattern per line. Heuristic only;
    LLM enrichment can be added later without changing the API.
    """
    claims: list[IdentityClaim] = []

    for exchange in exchanges:
        spk_name = exchange.get("speaker", "").strip()
        message = exchange.get("message") or ""
        if not message:
            continue
        if spk_name.lower() == (listener_name or "").lower():
            continue  # ignore listener's own lines

        for pattern, template, delta in _IDENTITY_PATTERNS:
            for match in pattern.finditer(message):
                raw_target = match.group(1) if match.groups() else ""
                if not raw_target:
                    continue
                target_lower = raw_target.strip().lower()
                if template.startswith("role:") and target_lower in _ROLE_STOPWORDS:
                    continue
                target = _normalise_target(
                    raw_target, speaker=speaker_id or spk_name.lower(),
                )
                if not target:
                    continue
                key = template.format(
                    target=target,
                    speaker=speaker_id or _normalise_target(spk_name),
                )
                claims.append(IdentityClaim(
                    key=key,
                    confidence_delta=delta,
                    source_text=message.strip()[:160],
                    speaker=spk_name,
                ))

    return claims
