"""
Gemma local LLM provider via Ollama.

Implements the LLMProvider interface using a locally-running Gemma model
served by Ollama. Zero API cost, zero latency overhead, runs entirely
on-device. Designed for NPC cognition on Apple Silicon.

Default: gemma4:e2b for all NPC tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import partial
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from core.npc.llm_client import LLMProvider, CostTracker, _response_cache

logger = logging.getLogger(__name__)

# Ollama API defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def ollama_available(base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check if Ollama is running and reachable."""
    try:
        req = Request(f"{base_url}/api/tags", method="GET")
        with urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def ollama_has_model(
    model: str, base_url: str = DEFAULT_OLLAMA_URL,
) -> bool:
    """Check if a specific model is pulled in Ollama."""
    try:
        req = Request(f"{base_url}/api/tags", method="GET")
        with urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            names = [m.get("name", "") for m in data.get("models", [])]
            return any(model in n for n in names)
    except (URLError, OSError):
        return False


class GemmaProvider(LLMProvider):
    """
    Local Gemma model via Ollama HTTP API.

    Gemma 4 supports a thinking mode where it reasons internally before
    responding. Which purposes use thinking — and how many tokens they
    get — is now controlled by a user-selectable ThinkingProfile on the
    base provider. See `core/npc/cognition/thinking.py` for presets
    (FAST / BALANCED / DEEP) and per-NPC overrides.
    """

    NPC_MODEL = "gemma4:e2b"
    OVERSEER_MODEL = "gemma4:e2b"

    def __init__(
        self,
        base_url: str | None = None,
        npc_model: str | None = None,
        overseer_model: str | None = None,
    ) -> None:
        super().__init__()
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        )
        self.npc_model = npc_model or os.environ.get(
            "GEMMA_NPC_MODEL", self.NPC_MODEL,
        )
        self.overseer_model = overseer_model or os.environ.get(
            "GEMMA_OVERSEER_MODEL", self.OVERSEER_MODEL,
        )
        self.cost_tracker = CostTracker()

        if not ollama_available(self.base_url):
            logger.warning(
                "Ollama not reachable at %s. Start it with: "
                "brew services start ollama", self.base_url,
            )

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
        """Call local Ollama Gemma model and return the text response."""
        # Check cache first
        cached = _response_cache.get(system, messages, temperature, purpose)
        if cached is not None:
            logger.debug("Gemma cache hit [%s]", purpose)
            return cached

        model = self.overseer_model if use_overseer_model else self.npc_model

        # Resolve the thinking profile — global default unless this NPC
        # has an override set via set_npc_profile (hero characters, etc.).
        profile = self._resolve_profile(npc_id)
        use_thinking = profile.should_think(purpose)
        token_budget = profile.budget_for(purpose, max_tokens)
        effective_temp = temperature * profile.temperature_multiplier

        # Build chat messages for /api/chat endpoint
        chat_messages = self._build_chat_messages(system, messages)

        try:
            payload = json.dumps({
                "model": model,
                "messages": chat_messages,
                "stream": False,
                "think": use_thinking,
                "options": {
                    "num_predict": token_budget,
                    "temperature": effective_temp,
                },
            }).encode()

            # Run blocking HTTP call in a thread so it doesn't block the
            # asyncio event loop (urlopen is synchronous).
            data = await asyncio.to_thread(
                self._sync_chat, payload,
            )

            msg = data.get("message", {})
            text = msg.get("content", "").strip()
            thinking = msg.get("thinking", "")
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)

            self.cost_tracker.record(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                purpose=purpose,
            )

            duration_ms = data.get("total_duration", 0) / 1_000_000
            think_info = f" (thought {len(thinking)} chars)" if thinking else ""
            logger.debug(
                "Gemma [%s] %s: %d in / %d out tokens (%.0fms)%s",
                purpose, model, input_tokens, output_tokens, duration_ms,
                think_info,
            )

            _response_cache.put(system, messages, temperature, purpose, text)
            return text

        except Exception as e:
            logger.error("Gemma/Ollama call failed [%s]: %s", purpose, e)
            raise

    def _sync_chat(self, payload: bytes, attempts: int = 3) -> dict:
        """Synchronous Ollama /api/chat call — run via asyncio.to_thread.

        Each attempt has a hard 120s socket timeout so a dead connection
        (e.g. the TCP socket killed by a macOS sleep/wake cycle) can never
        block indefinitely. On timeout / connection error we retry a
        bounded number of times, then raise a clear error rather than
        stall — the caller (and the diagnostic's per-tick watchdog) then
        fail loudly instead of hanging silently.
        """
        last_err: Exception | None = None
        for i in range(attempts):
            req = Request(
                f"{self.base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except (TimeoutError, URLError, OSError) as e:
                last_err = e
                logger.warning(
                    "Ollama /api/chat attempt %d/%d failed: %s",
                    i + 1, attempts, e,
                )
        raise RuntimeError(
            f"Ollama /api/chat failed after {attempts} attempts: {last_err}"
        )

    @staticmethod
    def _build_chat_messages(
        system: str, messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Convert system + messages into Ollama chat message format."""
        chat_msgs: list[dict[str, str]] = []
        if system:
            chat_msgs.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chat_msgs.append({"role": role, "content": content})
        return chat_msgs
