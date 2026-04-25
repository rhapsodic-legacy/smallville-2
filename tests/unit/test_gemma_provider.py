"""Tests for the Gemma/Ollama LLM provider."""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from core.npc.gemma_provider import (
    GemmaProvider, ollama_available, ollama_has_model,
)


class TestOllamaDetection:
    def test_ollama_available_when_running(self):
        """Should detect a running Ollama instance."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("core.npc.gemma_provider.urlopen", return_value=mock_resp):
            assert ollama_available("http://localhost:11434") is True

    def test_ollama_unavailable_when_down(self):
        """Should return False when Ollama is not running."""
        from urllib.error import URLError
        with patch(
            "core.npc.gemma_provider.urlopen",
            side_effect=URLError("Connection refused"),
        ):
            assert ollama_available("http://localhost:11434") is False

    def test_ollama_has_model(self):
        """Should detect an installed model."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "models": [{"name": "gemma3:1b"}],
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("core.npc.gemma_provider.urlopen", return_value=mock_resp):
            assert ollama_has_model("gemma3:1b") is True
            assert ollama_has_model("gemma4:e2b") is False


class TestGemmaProvider:
    def test_provider_initialises(self):
        """GemmaProvider should initialise with defaults."""
        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider()
        assert provider.npc_model == "gemma4:e2b"
        assert provider.overseer_model == "gemma4:e2b"

    def test_custom_models(self):
        """Should accept custom model names."""
        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider(
                npc_model="gemma3:4b",
                overseer_model="gemma4:e4b",
            )
        assert provider.npc_model == "gemma3:4b"
        assert provider.overseer_model == "gemma4:e4b"

    def test_build_chat_messages(self):
        """Should format system + messages into Ollama chat format."""
        msgs = GemmaProvider._build_chat_messages(
            "You are a blacksmith.",
            [{"role": "user", "content": "What do you do at dawn?"}],
        )
        assert msgs[0] == {"role": "system", "content": "You are a blacksmith."}
        assert msgs[1] == {"role": "user", "content": "What do you do at dawn?"}

    def test_complete_calls_ollama(self):
        """Should call Ollama chat API and return response text."""
        ollama_response = {
            "message": {"role": "assistant", "content": "I stoke the forge."},
            "prompt_eval_count": 20,
            "eval_count": 5,
            "total_duration": 100_000_000,
        }

        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider()
        with patch.object(provider, "_sync_chat", return_value=ollama_response):
            result = asyncio.new_event_loop().run_until_complete(
                provider.complete(
                    system="You are a blacksmith.",
                    messages=[{"role": "user", "content": "What now?"}],
                    purpose="daily_plan",
                )
            )

        assert result == "I stoke the forge."

    def test_complete_uses_cache(self):
        """Second identical call should hit cache, not Ollama."""
        ollama_response = {
            "message": {"role": "assistant", "content": "Cached response."},
            "prompt_eval_count": 10,
            "eval_count": 3,
            "total_duration": 50_000_000,
        }

        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider()

        with patch.object(
            provider, "_sync_chat", return_value=ollama_response,
        ) as mock_chat:
            loop = asyncio.new_event_loop()
            result1 = loop.run_until_complete(
                provider.complete(
                    system="Test system",
                    messages=[{"role": "user", "content": "Test"}],
                    purpose="daily_plan",
                )
            )
            result2 = loop.run_until_complete(
                provider.complete(
                    system="Test system",
                    messages=[{"role": "user", "content": "Test"}],
                    purpose="daily_plan",
                )
            )

        # _sync_chat should only be called once — second call hits cache
        assert mock_chat.call_count == 1
        assert result1 == result2

    def test_complete_error_handling(self):
        """Should raise on Ollama failure."""
        from urllib.error import URLError

        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider()
        with patch.object(
            provider, "_sync_chat",
            side_effect=URLError("Connection refused"),
        ):
            with pytest.raises(URLError):
                asyncio.new_event_loop().run_until_complete(
                    provider.complete(
                        system="test",
                        messages=[{"role": "user", "content": "test"}],
                    )
                )


class TestLLMProviderInterface:
    """Verify GemmaProvider satisfies the LLMProvider contract."""

    def test_is_llm_provider(self):
        from core.npc.llm_client import LLMProvider
        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider()
        assert isinstance(provider, LLMProvider)

    def test_has_cost_tracker(self):
        from core.npc.llm_client import CostTracker
        with patch("core.npc.gemma_provider.ollama_available", return_value=True):
            provider = GemmaProvider()
        assert isinstance(provider.cost_tracker, CostTracker)
