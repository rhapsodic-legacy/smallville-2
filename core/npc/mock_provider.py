"""
Context-aware mock LLM provider.

Cycles through varied canned responses per purpose, parses prompts for
occupation/sentiment/relationship signals, and adjusts tone accordingly.
Serves as the default experience for users without an API key.

Expandable: call ``register_responses()`` to add new pools at runtime.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.npc.llm_client import LLMProvider

logger = logging.getLogger(__name__)


# ---------- Response pools ----------

_DAILY_PLANS: dict[str, list[str]] = {
    "farmer": [
        (
            "1. Wake up at dawn and feed the chickens (06:00-06:30)\n"
            "2. Tend the wheat fields (06:30-12:00)\n"
            "3. Eat lunch at home (12:00-13:00)\n"
            "4. Repair fences and tools (13:00-15:00)\n"
            "5. Water the vegetable garden (15:00-17:00)\n"
            "6. Sell produce at the market (17:00-18:30)\n"
            "7. Supper and rest at home (18:30-06:00)"
        ),
        (
            "1. Early breakfast and check on livestock (06:00-07:00)\n"
            "2. Plough the south field (07:00-11:00)\n"
            "3. Quick lunch by the well (11:00-11:30)\n"
            "4. Harvest ripe crops (11:30-16:00)\n"
            "5. Visit the tavern for a drink (16:00-18:00)\n"
            "6. Return home and sleep (18:00-06:00)"
        ),
    ],
    "blacksmith": [
        (
            "1. Light the forge at sunrise (06:00-06:30)\n"
            "2. Fill orders — horseshoes, nails (06:30-12:00)\n"
            "3. Lunch at the tavern (12:00-13:00)\n"
            "4. Work on a special commission (13:00-17:00)\n"
            "5. Clean up the smithy (17:00-17:30)\n"
            "6. Socialise in the square (17:30-20:00)\n"
            "7. Home and sleep (20:00-06:00)"
        ),
        (
            "1. Breakfast and stoke the forge (06:00-07:00)\n"
            "2. Repair farming tools for villagers (07:00-11:00)\n"
            "3. Eat at home (11:00-12:00)\n"
            "4. Forge new blades (12:00-16:00)\n"
            "5. Trade at the market (16:00-18:00)\n"
            "6. Evening at the tavern (18:00-20:00)\n"
            "7. Sleep (20:00-06:00)"
        ),
    ],
    "merchant": [
        (
            "1. Open the shop early (06:00-06:30)\n"
            "2. Arrange stock and check inventory (06:30-08:00)\n"
            "3. Serve customers (08:00-12:00)\n"
            "4. Lunch at the tavern, listen for gossip (12:00-13:00)\n"
            "5. Negotiate with suppliers (13:00-15:00)\n"
            "6. Afternoon sales (15:00-18:00)\n"
            "7. Count the day's earnings and close (18:00-19:00)\n"
            "8. Home and sleep (19:00-06:00)"
        ),
        (
            "1. Breakfast and review ledgers (06:00-07:00)\n"
            "2. Travel to the market square (07:00-07:30)\n"
            "3. Sell wares and barter (07:30-12:00)\n"
            "4. Midday meal at home (12:00-13:00)\n"
            "5. Visit other merchants for trade deals (13:00-16:00)\n"
            "6. Restock shelves (16:00-18:00)\n"
            "7. Tavern for supper (18:00-20:00)\n"
            "8. Sleep (20:00-06:00)"
        ),
    ],
    "default": [
        (
            "1. Wake up and eat breakfast at home (06:00-07:00)\n"
            "2. Walk to workplace and begin work (07:00-12:00)\n"
            "3. Have lunch at the tavern (12:00-13:00)\n"
            "4. Continue working (13:00-17:00)\n"
            "5. Socialise at the tavern (17:00-20:00)\n"
            "6. Return home and sleep (20:00-06:00)"
        ),
        (
            "1. Morning meal and stretch (06:00-07:00)\n"
            "2. Head to the town square for errands (07:00-09:00)\n"
            "3. Work at my usual post (09:00-12:00)\n"
            "4. Lunch break by the well (12:00-13:00)\n"
            "5. Afternoon duties (13:00-16:00)\n"
            "6. Browse the market stalls (16:00-18:00)\n"
            "7. Supper and home (18:00-06:00)"
        ),
        (
            "1. Rise early and prepare for the day (06:00-06:30)\n"
            "2. Walk through town, greet neighbours (06:30-07:30)\n"
            "3. Work (07:30-12:00)\n"
            "4. Midday rest and food (12:00-13:30)\n"
            "5. Resume tasks (13:30-17:00)\n"
            "6. Evening at the tavern (17:00-20:00)\n"
            "7. Sleep (20:00-06:00)"
        ),
    ],
}

_CONVERSATION_POSITIVE: list[str] = [
    "Good day to you! Lovely weather we're having.",
    "Ah, good to see you! I was just thinking about how things are going in town.",
    "Hello there, friend! Have you heard any interesting news lately?",
    "Well met! I've been meaning to chat with you — how's your work going?",
    "Nice to run into you! Things have been quite busy on my end.",
    "What a pleasant surprise! Shall we walk together for a bit?",
    "Greetings! I hope all is well with you and yours.",
    "Ah, just the person I wanted to see. I could use your advice on something.",
]

_CONVERSATION_NEGATIVE: list[str] = [
    "Oh. It's you. What do you want?",
    "I'd rather not chat right now, if you don't mind.",
    "Hmph. I suppose we should talk, even if I'd prefer not to.",
    "Don't expect me to be friendly after what happened.",
    "I've got nothing pleasant to say to you today.",
    "Let's keep this brief. I have better things to do.",
    "You again. I was having such a nice day, too.",
    "I'll be civil, but don't mistake that for warmth.",
]

_CONVERSATION_NEUTRAL: list[str] = [
    "Good day to you! How are things?",
    "Hello. Busy day today, isn't it?",
    "Morning! Or is it afternoon already? I've lost track.",
    "Greetings. Anything interesting happening around town?",
    "Oh, hello there. Just going about my business.",
    "Hey. Seen anything unusual today?",
    "Afternoon. The town seems lively today.",
    "Hello — I don't think we've spoken in a while.",
]

_CONVERSATION_HELP: list[str] = [
    "Actually, I could use a hand with something if you're not too busy.",
    "Say, you wouldn't happen to know anything about fixing a broken fence?",
    "I've been struggling with my workload — any chance you could lend a hand?",
    "I hate to ask, but I'm in a bit of a bind. Could you help me out?",
    "Do you know anyone who might be able to help with a delivery?",
    "I've got more work than I can handle. Would you be willing to help?",
]

_REFLECTIONS: list[str] = [
    "The town square has been busier than usual — people seem restless.",
    "I've noticed the weather shifting. Could mean a change in the harvest.",
    "My relationships with neighbours have been evolving in unexpected ways.",
    "The economy seems to be picking up. More traders passing through.",
    "I should pay more attention to who I can trust around here.",
    "There's a tension in the air lately — factions forming, alliances shifting.",
    "I've been thinking about my place in this community and what I contribute.",
    "The nights are getting longer. Winter preparations should start soon.",
    "I overheard talk at the tavern about changes coming to the village.",
    "Some folk seem friendlier than others. I should be more observant.",
    "Resources have been a bit scarce. I need to plan more carefully.",
    "The market prices have shifted — supply and demand at work.",
]

_REACTIONS: list[str] = [
    "continue_current",
    "continue_current",
    "continue_current",
    "observe",
    "observe",
    "approach",
    "approach",
    "avoid",
]

_IMPORTANCE_SCORES: list[str] = [
    "2", "3", "3", "4", "5", "5", "5", "6", "6", "7", "8",
]

_TASK_DECOMPOSITION: dict[str, list[str]] = {
    "farmer": [
        "Pull weeds from the crop rows",
        "Carry water buckets to the field",
        "Check the fence for damage",
        "Feed the chickens and collect eggs",
        "Sharpen the scythe for harvesting",
    ],
    "blacksmith": [
        "Heat the iron in the forge",
        "Hammer out a set of nails",
        "Sharpen a dulled axe blade",
        "Polish and inspect finished work",
        "Restock coal for the furnace",
    ],
    "merchant": [
        "Arrange goods on the display",
        "Update prices on the ledger",
        "Greet a browsing customer",
        "Wrap a purchase for delivery",
        "Count coins in the strongbox",
    ],
    "default": [
        "Work at my station",
        "Tidy up the work area",
        "Check on today's progress",
        "Prepare materials for the next task",
        "Take a short break and stretch",
        "Review what still needs doing",
    ],
}


# ---------- Context detection ----------

_OCCUPATION_KEYWORDS: dict[str, list[str]] = {
    "farmer": ["farmer", "farming", "farm", "crops", "harvest", "fields"],
    "blacksmith": ["blacksmith", "smithy", "forge", "anvil", "metalwork"],
    "merchant": ["merchant", "trader", "shop", "trade", "goods", "sell"],
}

_NEGATIVE_SENTIMENT_PATTERN = re.compile(
    r"(trust:\s*-\d|fear:\s*\d|hostile|dislike|distrust|rival|enemy|"
    r"disposition:\s*(?:hostile|cold|wary|unfriendly))",
    re.IGNORECASE,
)

_POSITIVE_SENTIMENT_PATTERN = re.compile(
    r"(trust:\s*\d|affection:\s*\d|friendly|ally|allied|warm|"
    r"disposition:\s*(?:friendly|warm|trusting|close))",
    re.IGNORECASE,
)

_HELP_PATTERN = re.compile(
    r"(need help|lend a hand|assist|struggling|could use|request)",
    re.IGNORECASE,
)


def _detect_occupation(text: str) -> str:
    """Return the best-matching occupation key, or 'default'."""
    text_lower = text.lower()
    for occ, keywords in _OCCUPATION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return occ
    return "default"


def _detect_sentiment(text: str) -> str:
    """Return 'negative', 'positive', or 'neutral' from prompt signals."""
    neg = bool(_NEGATIVE_SENTIMENT_PATTERN.search(text))
    pos = bool(_POSITIVE_SENTIMENT_PATTERN.search(text))
    if neg and not pos:
        return "negative"
    if pos and not neg:
        return "positive"
    return "neutral"


def _detect_help_needed(text: str) -> bool:
    """Check if the prompt contains signals about needing help."""
    return bool(_HELP_PATTERN.search(text))


# ---------- MockProvider ----------

class MockProvider(LLMProvider):
    """Context-aware mock provider that cycles through varied responses.

    Parses prompt text for occupation, sentiment, and relationship signals
    to select appropriate response pools. Round-robins within each pool
    so repeated calls get different responses.

    Use ``register_responses()`` to add custom pools at runtime.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        super().__init__()
        # Legacy per-purpose overrides (exact string returned)
        self._overrides: dict[str, str] = responses or {}
        self.call_log: list[dict[str, Any]] = []
        # Cycling indices: (purpose, context_key) → current index
        self._indices: dict[tuple[str, str], int] = {}
        # Custom pools added via register_responses
        self._custom_pools: dict[str, dict[str, list[str]]] = {}

    def register_responses(
        self, purpose: str, context_key: str, responses: list[str],
    ) -> None:
        """Add or replace a response pool for a (purpose, context_key) pair.

        Example::

            provider.register_responses("conversation", "barter", [
                "I'll give you three gold for that.",
                "That price is far too high.",
            ])
        """
        self._custom_pools.setdefault(purpose, {})[context_key] = responses

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.7,
        purpose: str = "general",
        **kwargs: Any,
    ) -> str:
        self.call_log.append({
            "system": system, "messages": messages,
            "max_tokens": max_tokens, "purpose": purpose,
        })

        # Always yield to the event loop so batched mock calls from
        # asyncio.gather (e.g. 8 NPC schedules on startup) don't hog
        # the loop and starve the movement tick broadcaster. Without
        # this, the tick rate collapses during cognition_tick bursts.
        import asyncio as _asyncio
        import os as _os
        await _asyncio.sleep(0)

        # Optional artificial latency for tests that need to observe
        # behaviour during long-running LLM calls (e.g. verifying the
        # WS receive loop isn't blocked). Set SMALLVILLE_MOCK_DELAY_MS
        # to N to make every conversation call sleep N ms.
        if purpose in ("conversation", "conversation_initiate"):
            delay_ms = _os.environ.get("SMALLVILLE_MOCK_DELAY_MS", "")
            try:
                d = float(delay_ms) if delay_ms else 0.0
            except ValueError:
                d = 0.0
            if d > 0:
                await _asyncio.sleep(d / 1000.0)

        # Legacy exact-match overrides take priority
        if purpose in self._overrides:
            return self._overrides[purpose]

        # Build full prompt text for context detection
        prompt_text = system + " ".join(
            m.get("content", "") for m in messages
        )

        return self._contextual_response(purpose, prompt_text)

    # ------ internal ------

    def _contextual_response(self, purpose: str, prompt: str) -> str:
        """Select and cycle through the right pool for this call."""
        if purpose == "daily_plan":
            return self._pick_daily_plan(prompt)
        if purpose == "conversation":
            return self._pick_conversation(prompt)
        if purpose == "reflection":
            return self._pick_reflection(prompt)
        if purpose == "reaction":
            return self._pick_reaction(prompt)
        if purpose == "importance":
            return self._pick_importance(prompt)
        if purpose == "task_decomposition":
            return self._pick_task(prompt)
        # Unknown purpose — check custom pools, then fallback
        return self._cycle(purpose, "default", ["Acknowledged."])

    def _pick_daily_plan(self, prompt: str) -> str:
        # A mid-day REPLAN prompt asks whether to change the remaining
        # schedule and to reply NO_CHANGE if not (the initial daily-plan
        # prompt never mentions NO_CHANGE). A real NPC usually keeps its
        # plan — model that. Returning a freshly-rotated full-day schedule
        # on every replan was a stub artifact that made replan churn the
        # schedule every 60 game-minutes, preventing goal completion.
        if "NO_CHANGE" in prompt:
            return "NO_CHANGE"
        occ = _detect_occupation(prompt)
        pool = (
            self._custom_pool("daily_plan", occ)
            or _DAILY_PLANS.get(occ)
            or _DAILY_PLANS["default"]
        )
        return self._cycle("daily_plan", occ, pool)

    def _pick_conversation(self, prompt: str) -> str:
        sentiment = _detect_sentiment(prompt)
        help_needed = _detect_help_needed(prompt)

        # Help pool takes priority over sentiment
        if help_needed:
            pool = (
                self._custom_pool("conversation", "help")
                or _CONVERSATION_HELP
            )
            return self._cycle("conversation", "help", pool)

        pool_map = {
            "positive": _CONVERSATION_POSITIVE,
            "negative": _CONVERSATION_NEGATIVE,
            "neutral": _CONVERSATION_NEUTRAL,
        }
        pool = (
            self._custom_pool("conversation", sentiment)
            or pool_map[sentiment]
        )
        return self._cycle("conversation", sentiment, pool)

    def _pick_reflection(self, prompt: str) -> str:
        pool = self._custom_pool("reflection", "default") or _REFLECTIONS
        return self._cycle("reflection", "default", pool)

    def _pick_reaction(self, prompt: str) -> str:
        pool = self._custom_pool("reaction", "default") or _REACTIONS
        return self._cycle("reaction", "default", pool)

    def _pick_importance(self, prompt: str) -> str:
        pool = self._custom_pool("importance", "default") or _IMPORTANCE_SCORES
        return self._cycle("importance", "default", pool)

    def _pick_task(self, prompt: str) -> str:
        occ = _detect_occupation(prompt)
        pool = (
            self._custom_pool("task_decomposition", occ)
            or _TASK_DECOMPOSITION.get(occ)
            or _TASK_DECOMPOSITION["default"]
        )
        return self._cycle("task_decomposition", occ, pool)

    def _custom_pool(
        self, purpose: str, context_key: str,
    ) -> list[str] | None:
        """Return a custom-registered pool if one exists."""
        return self._custom_pools.get(purpose, {}).get(context_key)

    def _cycle(
        self, purpose: str, context_key: str, pool: list[str],
    ) -> str:
        """Round-robin through a pool of responses."""
        key = (purpose, context_key)
        idx = self._indices.get(key, 0)
        result = pool[idx % len(pool)]
        self._indices[key] = idx + 1
        return result
