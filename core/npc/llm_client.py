"""
LLM integration layer.

Wraps the Anthropic Claude API with rate limiting, cost tracking,
and a prompt template system. Haiku for NPCs, Opus for overseer.
Pluggable provider interface for future local model support.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------- Response cache ----------

class ResponseCache:
    """
    In-memory LRU cache for LLM responses.

    Keyed on (system_prompt, messages, temperature, purpose).
    TTL is configurable per purpose — conversations expire quickly,
    daily plans last all day, town parsing lasts forever.
    """

    # TTL in seconds per purpose (0 = no caching for that purpose)
    DEFAULT_TTLS: dict[str, float] = {
        "daily_plan": 1200.0,       # 20 min (one game day)
        "task_decompose": 600.0,    # 10 min (activity repetition)
        "town_prompt_parse": 0.0,   # forever (immutable)
        "reflection": 300.0,        # 5 min
        "reaction": 120.0,          # 2 min
        "conversation": 0.0,        # no caching — every convo is unique
        "general": 60.0,
    }

    def __init__(self, max_entries: int = 500):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_entries = max_entries
        self.hits: int = 0
        self.misses: int = 0

    def _make_key(
        self, system: str, messages: list[dict[str, str]],
        temperature: float, purpose: str,
    ) -> str:
        raw = json.dumps(
            {"s": system, "m": messages, "t": temperature, "p": purpose},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self, system: str, messages: list[dict[str, str]],
        temperature: float, purpose: str,
    ) -> str | None:
        ttl = self.DEFAULT_TTLS.get(purpose)
        if ttl == 0.0:
            # Purpose disabled from caching (conversation) or forever
            if purpose == "conversation":
                self.misses += 1
                return None
            # town_prompt_parse: ttl=0 means forever
        key = self._make_key(system, messages, temperature, purpose)
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            return None
        response, timestamp = entry
        if ttl and ttl > 0 and (time.monotonic() - timestamp) > ttl:
            del self._cache[key]
            self.misses += 1
            return None
        self._cache.move_to_end(key)
        self.hits += 1
        return response

    def put(
        self, system: str, messages: list[dict[str, str]],
        temperature: float, purpose: str, response: str,
    ) -> None:
        if purpose == "conversation":
            return  # Never cache conversations
        key = self._make_key(system, messages, temperature, purpose)
        self._cache[key] = (response, time.monotonic())
        if len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
        }


# Shared cache instance — all providers use the same cache
_response_cache = ResponseCache()


# ---------- Cost tracking ----------

@dataclass
class UsageRecord:
    """Single LLM call usage."""
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float
    purpose: str  # e.g. "daily_plan", "conversation", "reflection"


@dataclass
class CostTracker:
    """Tracks LLM API usage and estimated costs."""
    records: list[UsageRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Approximate costs per million tokens (USD)
    _COST_PER_M_INPUT: dict[str, float] = field(default_factory=lambda: {
        "claude-haiku-4-5-20251001": 0.80,
        "claude-sonnet-4-6": 3.0,
        "claude-opus-4-6": 15.0,
    })
    _COST_PER_M_OUTPUT: dict[str, float] = field(default_factory=lambda: {
        "claude-haiku-4-5-20251001": 4.0,
        "claude-sonnet-4-6": 15.0,
        "claude-opus-4-6": 75.0,
    })

    def record(self, model: str, input_tokens: int, output_tokens: int,
               purpose: str) -> None:
        self.records.append(UsageRecord(
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, timestamp=time.time(),
            purpose=purpose,
        ))
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def estimated_cost_usd(self) -> float:
        total = 0.0
        for r in self.records:
            in_cost = self._COST_PER_M_INPUT.get(r.model, 1.0)
            out_cost = self._COST_PER_M_OUTPUT.get(r.model, 5.0)
            total += (r.input_tokens * in_cost + r.output_tokens * out_cost) / 1_000_000
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
        }


# ---------- Rate limiter ----------

class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_calls_per_minute: int = 50):
        self.max_calls = max_calls_per_minute
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        """Wait until a call slot is available."""
        now = time.monotonic()
        # Purge timestamps older than 60 seconds
        self._timestamps = [t for t in self._timestamps if now - t < 60]

        if len(self._timestamps) >= self.max_calls:
            wait_time = 60 - (now - self._timestamps[0])
            if wait_time > 0:
                logger.debug("Rate limiter: waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)

        self._timestamps.append(time.monotonic())


# ---------- Provider interface ----------

class LLMProvider(ABC):
    """Abstract interface for LLM providers.

    Providers carry a `ThinkingProfile` that controls how deeply the
    LLM deliberates and how many tokens it's allowed per call. The
    profile is a knob users expose in the UI ("Fast / Balanced / Deep")
    and the architecture is layered so that future code can override
    it per-NPC via `set_npc_profile` for large-town performance:
    a handful of hero NPCs on DEEP, the crowd on FAST or deterministic.

    Concrete providers are expected to consult the profile in
    `complete()` — either by calling `_resolve_profile(npc_id)` or by
    reading `self.profile` directly. Providers that can't honour
    thinking mode (e.g. the mock provider) ignore it harmlessly.
    """

    def __init__(self) -> None:
        from core.npc.cognition.thinking import BALANCED, ThinkingProfile
        self.profile: ThinkingProfile = BALANCED
        self._npc_profiles: dict[str, ThinkingProfile] = {}

    def set_profile(self, profile) -> None:
        """Set the default thinking profile for all calls."""
        self.profile = profile

    def set_npc_profile(self, npc_id: str, profile) -> None:
        """Override the thinking profile for a specific NPC.

        Lets large towns mix qualities: call this with DEEP for the
        story-critical NPCs and leave the rest on the global default
        (which may itself be FAST for performance).
        """
        self._npc_profiles[npc_id] = profile

    def clear_npc_profile(self, npc_id: str) -> None:
        self._npc_profiles.pop(npc_id, None)

    def _resolve_profile(self, npc_id: str | None):
        """Return the profile that applies to this call."""
        if npc_id and npc_id in self._npc_profiles:
            return self._npc_profiles[npc_id]
        return self.profile

    @abstractmethod
    async def complete(self, system: str, messages: list[dict[str, str]],
                       max_tokens: int = 300, temperature: float = 0.7,
                       purpose: str = "general",
                       npc_id: str | None = None) -> str:
        """Send a completion request and return the text response."""
        ...


# ---------- Claude provider ----------

class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    # Model tiers
    NPC_MODEL = "claude-haiku-4-5-20251001"
    OVERSEER_MODEL = "claude-opus-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        npc_model: str | None = None,
        overseer_model: str | None = None,
        max_calls_per_minute: int = 50,
    ):
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.npc_model = npc_model or self.NPC_MODEL
        self.overseer_model = overseer_model or self.OVERSEER_MODEL
        self.cost_tracker = CostTracker()
        self._rate_limiter = RateLimiter(max_calls_per_minute)
        self._client = None

    def _get_client(self):
        """Lazy-initialise the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.7,
        purpose: str = "general",
        use_overseer_model: bool = False,
        npc_id: str | None = None,
    ) -> str:
        """Call Claude API and return the text response."""
        # Check cache first
        cached = _response_cache.get(system, messages, temperature, purpose)
        if cached is not None:
            logger.debug("LLM cache hit [%s]", purpose)
            return cached

        model = self.overseer_model if use_overseer_model else self.npc_model
        client = self._get_client()
        await self._rate_limiter.acquire()

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )

            text = response.content[0].text
            self.cost_tracker.record(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                purpose=purpose,
            )
            logger.debug(
                "LLM [%s] %s: %d in / %d out tokens",
                purpose, model,
                response.usage.input_tokens, response.usage.output_tokens,
            )

            # Store in cache
            _response_cache.put(system, messages, temperature, purpose, text)
            return text

        except Exception as e:
            logger.error("LLM call failed [%s]: %s", purpose, e)
            raise


# ---------- Mock provider ----------
# Full implementation in mock_provider.py; re-exported here for convenience.

from core.npc.mock_provider import MockProvider  # noqa: E402


# ---------- Prompt templates ----------

PROMPT_TEMPLATES: dict[str, str] = {
    "daily_plan": (
        "You are {name}, a {age}-year-old {occupation} living in Smallville.\n"
        "{backstory}\n\n"
        "Personality: {personality}\n"
        "{self_concept}\n"
        "Current goals: {goals}\n"
        "{town_agenda}\n"
        "Current state — Health: {health}, Energy: {energy}, Hunger: {hunger}\n"
        "Gold: {gold}\n"
        "{relationship_summary}\n\n"
        "It is the start of day {day}. The weather is fair.\n"
        "Create a realistic daily schedule with 5-8 activities.\n"
        "Each line: time range, activity, location.\n"
        "Consider your occupation, personality, needs, goals, and relationships."
    ),

    "task_decomposition": (
        "You are {name}, {occupation} in Smallville.\n"
        "Your current scheduled activity: {activity} at {location}\n"
        "Time slot: {slot}\n\n"
        "Break this into a specific, concrete action you should do right now.\n"
        "Reply with just the action in 5-10 words."
    ),

    "conversation_initiate": (
        "You are {name}, a {age}-year-old {occupation} in Smallville.\n"
        "Personality: {personality}\n"
        "{self_concept}\n"
        "{town_agenda}\n"
        "{shared_agenda}\n"
        "{unresolved_matters}\n"
        "{backstory}\n\n"
        "You have encountered {other_name} ({other_occupation}).\n"
        "{relationship_context}\n"
        "Recent observations: {recent_perceptions}\n\n"
        "Start a brief, natural conversation. Keep it to 1-2 sentences.\n"
        "Stay in character. If you have open matters with this "
        "person, this is a good time to raise them. If you share a "
        "town initiative with them — working on it together or "
        "recently completing it — mention it naturally; it's fresh "
        "common ground. If this feels like a new encounter and you "
        "know of a headline town matter, it's a natural opener. "
        "But if a town matter conflicts with how you see yourself, "
        "it is natural to voice your reservations rather than go "
        "along with it."
    ),

    "conversation_respond": (
        "You are {name}, a {age}-year-old {occupation} in Smallville.\n"
        "Personality: {personality}\n"
        "{self_concept}\n"
        "{town_agenda}\n"
        "{shared_agenda}\n"
        "{unresolved_matters}\n\n"
        "You are talking with {other_name} ({other_occupation}).\n"
        "{relationship_context}\n"
        "{recent_history}"
        "{other_name} just said: \"{other_message}\"\n\n"
        "Respond naturally in 1-2 sentences. Stay in character. "
        "Stay on topic — reference what was said earlier in this "
        "conversation when it's relevant; do NOT change subject "
        "unless it makes clear sense. If you have an open matter "
        "with this person that fits, work it into your reply. If "
        "you share a town initiative with them, referencing it is "
        "natural conversational glue. But if a town matter conflicts "
        "with how you see yourself, it is natural to voice your "
        "reservations rather than simply agree."
    ),

    "reaction": (
        "You are {name}, {occupation} in Smallville.\n"
        "You are currently: {current_activity}\n\n"
        "You notice: {observation}\n\n"
        "How do you react? Choose one:\n"
        "- continue_current (keep doing what you're doing)\n"
        "- approach (go investigate or interact)\n"
        "- avoid (move away)\n"
        "- observe (watch from a distance)\n\n"
        "Reply with just the action word."
    ),

    "importance": (
        "On a scale of 1 to 10, where 1 is mundane (e.g. brushing teeth) "
        "and 10 is life-changing (e.g. a death or marriage), rate the "
        "poignancy of the following event:\n\n"
        "{description}\n\n"
        "Reply with just the number."
    ),

    "task_decompose": (
        "You are {name}, a {occupation} in the town of Smallville.\n"
        "Personality: {personality}\n\n"
        "Your current schedule says: \"{activity}\" during the {slot} "
        "at {location}.\n"
        "Available objects at this location: {objects}\n\n"
        "Recent context:\n{memory_context}\n\n"
        "Break this into 3-5 specific, concrete sub-tasks you would "
        "actually do. For each, give:\n"
        "description | duration_minutes | activity_state\n\n"
        "Activity states: idle, working, eating, sleeping, talking, gathering\n"
        "Be specific to your occupation and personality. "
        "Reference the objects available at this location where relevant."
    ),

    "replan_schedule": (
        "You are {name}, a {age}-year-old {occupation} in Smallville.\n"
        "Personality: {personality}\n"
        "Current goals: {goals}\n\n"
        "It is currently {time}. Your remaining schedule for today:\n"
        "{remaining_schedule}\n\n"
        "Recent events:\n{recent_perceptions}\n\n"
        "Recent reflections:\n{recent_reflections}\n\n"
        "People you care about:\n{relationship_summary}\n\n"
        "Based on what you've seen, heard, and reflected on, should you "
        "change your remaining schedule? If yes, output a new schedule "
        "for the REST of today (same format: one line per activity with "
        "time, activity, location). If no changes needed, respond with: "
        "NO_CHANGE"
    ),

    "day_summary": (
        "You are {name}, a {occupation} in Smallville.\n"
        "Personality: {personality}\n"
        "{self_concept}\n"
        "These are the smaller events of day {day}, the ones not "
        "already lodged in your memory as promises, accusations, or "
        "town matters:\n"
        "{events}\n\n"
        "Write a 2-4 sentence first-person recollection, as {name}, "
        "covering:\n"
        "1. What actually happened — the thread of the day, not a "
        "list of every moment.\n"
        "2. How it made you feel, in your voice.\n"
        "3. Anything that shifted in how you see the people around "
        "you or in what you mean to do next.\n\n"
        "Skip trivial comings and goings. Do not restate events "
        "verbatim. If the day was truly uneventful, say so in one "
        "short sentence and stop."
    ),

    "week_summary": (
        "You are {name}, a {occupation} in Smallville.\n"
        "Personality: {personality}\n"
        "{self_concept}\n"
        "These are your own day-by-day recollections from week "
        "{week} (day {day_start} through day {day_end}):\n"
        "{day_summaries}\n\n"
        "Looking back on the week as a whole, write a 2-4 sentence "
        "first-person reflection as {name}, covering:\n"
        "1. The shape of the week — the through-line, not each day "
        "in turn.\n"
        "2. Which relationships or moods have hardened, softened, "
        "or changed.\n"
        "3. What you now mean to do, avoid, or watch for next.\n\n"
        "Write the character arc, not the diary. Keep it tight."
    ),

    "self_review": (
        "You are {name}, a {occupation} in Smallville.\n"
        "Personality: {personality}\n"
        "{self_concept}\n"
        "It is the end of day {day}. Before sleep, you take stock.\n\n"
        "Things you said you'd do and haven't finished:\n"
        "{commitments}\n\n"
        "Longer-term things you're working toward:\n"
        "{long_term_goals}\n\n"
        "Your own recollection of today:\n"
        "{day_summary}\n\n"
        "Review each open matter honestly in your own voice. For "
        "every one, decide whether it's `moving`, `stalled`, "
        "`abandoned`, or `done`. Be willing to call something "
        "stalled if it really is; don't pretend to progress that "
        "didn't happen. Reply in this exact block format:\n\n"
        "SUMMARY: <one or two sentences on the day as a whole>\n"
        "GOAL: <the matter, paraphrased>\n"
        "STATUS: moving|stalled|abandoned|done\n"
        "NOTE: <one short line on what shifted or what's blocking>\n"
        "GOAL: ...\n"
        "STATUS: ...\n"
        "NOTE: ...\n"
        "NEXT: <one concrete thing you mean to do tomorrow, or "
        "NO_ACTION if nothing specific follows>"
    ),

    "conversation_extract_facts": (
        "Here is a conversation between {npc_a_name} and {npc_b_name}:\n"
        "{conversation}\n\n"
        "List any important facts revealed in this conversation.\n"
        "Format each fact as: subject | predicate | object\n"
        "Examples: Bob | is_hungry | true\n"
        "Martha | wants_to_visit | the market\n"
        "The bridge | needs_repair | true\n\n"
        "Only list facts that are clearly stated or strongly implied.\n"
        "If no important facts, respond with: NO_FACTS"
    ),
}


def get_cache_stats() -> dict[str, Any]:
    """Return LLM response cache statistics."""
    return _response_cache.stats()


class _MissingEmpty(dict):
    """str.format_map backing that returns "" for any missing key.

    Used so we can add new placeholders (like `{town_agenda}`) to
    prompt templates without having to update every caller in
    lock-step. Callers that don't know about the new slot simply
    produce an empty line, which callers that do know about it can
    fill in.
    """

    def __missing__(self, key: str) -> str:
        return ""


def format_prompt(template_name: str, **kwargs) -> str:
    """Fill a prompt template with NPC-specific values.

    Missing keys are tolerated — they expand to empty strings. This
    lets new placeholders land in the shared template dict without
    forcing every legacy caller to pass the new value on the same
    commit.
    """
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown prompt template: {template_name}")
    return template.format_map(_MissingEmpty(kwargs))
