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
from typing import Any, TYPE_CHECKING

from core.relationships.sentiment import CONVERSATION_TONE_DELTAS

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
    "The conversation:\n{conversation}\n"
    "{outcome_summary}\n"
    "What conclusion do you draw from this — about {other_name}, "
    "about yourself, about the town, or about someone who was "
    "mentioned? Name the specific insight, not just how you felt. "
    "Answer in 1-2 sentences, in first person.\n\n"
    "Then, on its own final line, judge how the conversation "
    "actually felt toward {other_name}, by YOUR standards — argument "
    "may be bracing to one character and an insult to another:\n"
    "TONE: warm|neutral|tense|hostile\n"
    "If — and only if — your conclusion asserts something about WHO "
    "YOU ARE (a stance, a role, a bond), add one more line:\n"
    "SELF: <prefix>:<target>\n"
    "where <prefix> is one of: role, opposes, supports, friend_of, "
    "enemy_of, rival_of, skill, reputation. Omit the SELF line "
    "otherwise."
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

# ---------- Reflection extras: tone + self-assertion (write paths) ----------
# The emergent-write-paths arc: the persona-conditioned reflection is
# where the LLM expresses friction and identity — these parse the two
# structured trailer lines (TONE:, SELF:) the prompt requests, so that
# signal lands in durable state instead of dying in episodic memory.

VALID_TONES = ("warm", "neutral", "tense", "hostile")

# Self-concept prefixes a reflection may assert about ONESELF. Matches
# the phrase_map in NPC.self_concept_summary; `helped`/`built`/`joined`
# are excluded — those are earned through I.4 goal completion, not
# claimed in a moment of reflection.
ALLOWED_SELF_PREFIXES = (
    "role", "opposes", "supports", "friend_of", "enemy_of",
    "rival_of", "skill", "reputation",
)

# One reflection nudges identity gently; conviction comes from
# repetition, mirroring IDENTITY_DELTA/REINFORCEMENT_DELTA magnitudes.
REFLECTION_CLAIM_DELTA = 0.10

_SELF_KEY_RE = re.compile(r"^([a-z_]+)\s*:\s*([a-z0-9 _'\-]{1,40})$")


def parse_reflection_extras(
    text: str,
) -> tuple[str, str | None, "IdentityClaim | None"]:
    """Split an LLM reflection into (insight, tone, self_claim).

    Scans for the `TONE:` and `SELF:` trailer lines, validates them
    strictly (unknown tones and disallowed/malformed SELF keys are
    dropped silently — a hallucinated key must never reach
    self_concept), and returns the insight with those lines removed.
    First valid occurrence of each wins.
    """
    tone: str | None = None
    claim: IdentityClaim | None = None
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TONE:"):
            if tone is None:
                value = stripped.split(":", 1)[1].strip().lower()
                first_word = value.split()[0].rstrip(".,;") if value else ""
                if first_word in VALID_TONES:
                    tone = first_word
            continue
        if upper.startswith("SELF:"):
            if claim is None:
                value = stripped.split(":", 1)[1].strip().lower()
                match = _SELF_KEY_RE.match(value)
                if match:
                    prefix = match.group(1)
                    target = re.sub(r"[\s\-]+", "_", match.group(2).strip())
                    target = target.strip("_'")[:32]
                    if prefix in ALLOWED_SELF_PREFIXES and target:
                        claim = IdentityClaim(
                            key=f"{prefix}:{target}",
                            confidence_delta=REFLECTION_CLAIM_DELTA,
                            source_text=text.strip()[:160],
                        )
            continue
        kept.append(line)
    return "\n".join(kept).strip(), tone, claim


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


IMPORTANT_NOTE_PROMPT = (
    "You are {name}, a {occupation} in Smallville.\n"
    "You just had this insight after a conversation with {other_name}:\n"
    "\"{insight}\"\n\n"
    "Is there a specific fact you should WRITE DOWN and remember "
    "about this — a concrete detail that, weeks from now, would let "
    "you explain why something happened? (Example: \"Traveller told "
    "me on day 12 that Petra is hoarding bread.\")\n\n"
    "If yes, reply in this exact format — ONE line, concise:\n"
    "NOTE: <the single factual line>\n"
    "TAGS: <2-4 short, lowercase, underscore-joined tags, "
    "space-separated>\n\n"
    "If the insight is purely emotional, a generic impression, or "
    "doesn't contain a specific fact worth remembering, reply:\n"
    "NO_NOTE"
)


async def extract_important_note(
    npc: "NPC",
    insight: str,
    other_name: str,
    llm: "LLMProvider",
) -> tuple[str, set[str]] | None:
    """Phase K.5 — ask the LLM whether a reflection insight contains
    a surgical fact worth keeping verbatim (with tags) even after
    Phase H compaction.

    Returns `(note_text, tag_set)` or None when nothing worth noting.
    Uses the same tier gate as `classify_insight`: only tier-1 NPCs
    burn LLM budget here; others just skip.
    """
    from core.npc.cognition.tiers import get_tier_config
    from core.memory.episodic import normalise_tags

    config = get_tier_config(npc.cognition_tier)
    if not config.uses_llm:
        return None
    try:
        prompt = IMPORTANT_NOTE_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            other_name=other_name,
            insight=insight,
        )
        response = await llm.complete(
            system="You decide which facts an NPC should remember verbatim.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
            purpose="reflection",
        )
    except Exception as e:
        logger.debug("Important-note extraction failed for %s: %s", npc.name, e)
        return None

    text = (response or "").strip()
    if "NO_NOTE" in text.upper():
        return None
    note_line = ""
    tag_line = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("NOTE:"):
            note_line = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("TAGS:"):
            tag_line = stripped.split(":", 1)[1].strip()
    if not note_line:
        return None
    tag_set = normalise_tags(tag_line)
    if not tag_set:
        # Fall back to two words from the note as tags so the note
        # remains retrievable.
        import re
        words = [
            w.lower() for w in re.findall(r"[A-Za-z]+", note_line)
            if len(w) > 4
        ]
        tag_set = normalise_tags(words[:3])
    return (note_line, tag_set)


def _format_outcome_for_reflection(outcome: Any) -> str:
    """Render a ConversationOutcome as a short prompt-ready block.

    Returns "" when there's nothing to say; otherwise produces
    bullet-style lines the reflection prompt can lean on. The
    generated lines stay first-person-neutral so either participant
    can consume the same block without pronoun surgery.
    """
    if outcome is None:
        return ""
    parts: list[str] = []
    for c in getattr(outcome, "commitments", []):
        parts.append(
            f"- {c.speaker} committed to {c.subject.strip().rstrip('.')}."
        )
    for a in getattr(outcome, "accusations", []):
        parts.append(
            f"- {a.accuser} accused {a.accused} of "
            f"{a.claim.strip().rstrip('.')}."
        )
    for r in getattr(outcome, "relayed_claims", []):
        subject = r.subject or "someone"
        parts.append(
            f"- {r.relayed_by} relayed that {r.cited_source} "
            f"said {subject} {r.claim.strip().rstrip('.')}."
        )
    if not parts:
        return ""
    return "Notable outcomes from this conversation:\n" + "\n".join(parts)


async def reflect_on_conversation(
    npc: NPC,
    other_name: str,
    exchanges: list[dict[str, str]],
    memory: MemoryManager,
    llm: LLMProvider,
    current_game_time: float,
    outcome: Any = None,
    other_id: str = "",
    claim_sink: Any = None,
) -> str | None:
    """
    Quick reflection after a conversation ends.

    Generates a single insight about what was discussed/learnt. When
    a structured `ConversationOutcome` is passed in, the prompt is
    primed with its contents (commitments, accusations, relayed
    claims) so the NPC's conclusion actually engages with what
    happened rather than producing vague "I had a nice chat" lines.

    Write paths (emergent-write-paths arc): the response's TONE
    verdict is applied one-directionally to this NPC's sentiment
    toward `other_id` (when given), and a SELF assertion is routed
    through `claim_sink` — the manager passes its
    `_inject_self_concept_delta` wrapper so reflection-born identity
    gets the same contradiction damping as conversation claims.
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config
    from core.npc.persona import persona_system_prompt

    config = get_tier_config(npc.cognition_tier)
    if not config.uses_llm:
        return None

    conversation_text = "\n".join(
        f"{e.get('speaker', '?')}: {e.get('message', '')}"
        for e in exchanges
    )

    outcome_summary = _format_outcome_for_reflection(outcome)

    try:
        prompt = POST_CONVERSATION_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            other_name=other_name,
            conversation=conversation_text,
            outcome_summary=outcome_summary,
        )

        insight = await llm.complete(
            system=persona_system_prompt(
                npc,
                "You are reflecting privately on a conversation you "
                "just had. Draw the conclusion YOUR character would "
                "draw — filtered through your temperament, values, "
                "fears, and agenda.",
            ),
            messages=[{"role": "user", "content": prompt}],
            # Headroom for the TONE/SELF trailer lines — they render
            # LAST, so truncation would silently eat the write-path
            # signal while leaving the insight looking fine.
            max_tokens=160,
            temperature=0.7,
            purpose="reflection",
            npc_id=npc.npc_id,
        )

        insight, tone, self_claim = parse_reflection_extras(insight.strip())

        if tone and other_id:
            sentiment = getattr(memory, "sentiment", None)
            deltas = CONVERSATION_TONE_DELTAS.get(tone, {})
            if sentiment is not None and deltas:
                for dim, delta in deltas.items():
                    sentiment.modify(
                        npc.npc_id, other_id, dim, delta,
                        game_time=current_game_time,
                    )
                logger.info(
                    "TONE %s→%s: %s %s",
                    npc.name, other_name, tone, deltas,
                )

        if self_claim is not None and claim_sink is not None:
            self_claim.speaker = npc.name  # own assertion, not hearsay
            try:
                claim_sink(self_claim)
            except Exception:
                logger.exception(
                    "Reflection self-claim sink failed for %s (%s)",
                    npc.name, self_claim.key,
                )

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
    from core.npc.persona import persona_system_prompt

    try:
        prompt = FOCAL_POINT_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            experiences="\n".join(f"- {e}" for e in experiences),
        )

        response = await llm.complete(
            system=persona_system_prompt(
                npc,
                "You are deciding what questions about your recent "
                "experiences matter enough to think on — the ones "
                "YOUR character, with these values and fears, would "
                "actually dwell on.",
            ),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            purpose="reflection",
            npc_id=npc.npc_id,
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
    from core.npc.persona import persona_system_prompt

    try:
        prompt = INSIGHT_PROMPT.format(
            name=npc.name,
            occupation=npc.occupation,
            personality=npc.personality.to_description(),
            focal_point=focal_point,
            memories=memory_text,
        )

        response = await llm.complete(
            system=persona_system_prompt(
                npc,
                "You are synthesising an insight from your own "
                "memories. Conclude what YOUR character would "
                "conclude — coloured by your temperament, values, "
                "and private agenda.",
            ),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
            purpose="reflection",
            npc_id=npc.npc_id,
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
