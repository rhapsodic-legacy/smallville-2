"""
Mistral AI LLM provider.

Implements the LLMProvider interface using the Mistral API.
Designed for NPC cognition with cost-effective models (Mistral Small
for NPCs, Mistral Large for overseer).

Free tier: ~500K tokens/day — sufficient for 4-6 hours of simulation
with 10 Tier 1/2 NPCs using the cognition router.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from core.npc.llm_client import LLMProvider, CostTracker, RateLimiter

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):
    """
    Mistral AI API provider.

    Uses mistral-small for NPC cognition (fast, cheap) and
    mistral-large for overseer analysis (deeper reasoning).
    """

    NPC_MODEL = "mistral-small-latest"
    OVERSEER_MODEL = "mistral-large-latest"

    def __init__(
        self,
        api_key: str | None = None,
        npc_model: str | None = None,
        overseer_model: str | None = None,
        max_calls_per_minute: int = 50,
    ) -> None:
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.npc_model = npc_model or self.NPC_MODEL
        self.overseer_model = overseer_model or self.OVERSEER_MODEL
        self.cost_tracker = CostTracker()
        self._rate_limiter = RateLimiter(max_calls_per_minute)
        self._client = None

        if not self.api_key:
            logger.warning(
                "No MISTRAL_API_KEY found. Set it in .env or pass directly."
            )

    def _get_client(self):
        """Lazy-initialise the Mistral client."""
        if self._client is None:
            try:
                from mistralai.client import Mistral
                self._client = Mistral(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "mistralai package not installed. "
                    "Run: pip install mistralai"
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
    ) -> str:
        """Call Mistral API and return the text response."""
        model = self.overseer_model if use_overseer_model else self.npc_model
        client = self._get_client()
        await self._rate_limiter.acquire()

        # Build messages with system prompt
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        try:
            response = await client.chat.complete_async(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            self.cost_tracker.record(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                purpose=purpose,
            )

            logger.debug(
                "Mistral [%s] %s: %d in / %d out tokens",
                purpose, model, input_tokens, output_tokens,
            )
            return text

        except Exception as e:
            logger.error("Mistral call failed [%s]: %s", purpose, e)
            raise
