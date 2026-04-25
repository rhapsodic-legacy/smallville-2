"""Tests for the context-aware MockProvider."""

import asyncio
import pytest

from core.npc.mock_provider import (
    MockProvider,
    _detect_occupation,
    _detect_sentiment,
    _detect_help_needed,
)


# ---------- Context detection ----------

class TestDetectOccupation:
    def test_farmer_detected(self):
        assert _detect_occupation("You are Bob, a farmer in Smallville.") == "farmer"

    def test_blacksmith_detected(self):
        assert _detect_occupation("You work at the forge as a blacksmith.") == "blacksmith"

    def test_merchant_detected(self):
        assert _detect_occupation("You are a merchant who runs a shop.") == "merchant"

    def test_unknown_defaults(self):
        assert _detect_occupation("You are a librarian.") == "default"

    def test_case_insensitive(self):
        assert _detect_occupation("FARMER Bob tends the FIELDS") == "farmer"


class TestDetectSentiment:
    def test_negative_trust(self):
        assert _detect_sentiment("trust: -20, fear: 5") == "negative"

    def test_positive_trust(self):
        assert _detect_sentiment("trust: 30, disposition: friendly") == "positive"

    def test_hostile_keyword(self):
        assert _detect_sentiment("They seem hostile towards you.") == "negative"

    def test_neutral_no_signals(self):
        assert _detect_sentiment("You are walking through town.") == "neutral"

    def test_mixed_signals_neutral(self):
        # Both positive and negative → neutral
        assert _detect_sentiment("trust: 10 but also hostile") == "neutral"


class TestDetectHelp:
    def test_need_help(self):
        assert _detect_help_needed("I need help carrying these supplies.")

    def test_no_help(self):
        assert not _detect_help_needed("Nice weather today.")

    def test_struggling(self):
        assert _detect_help_needed("I've been struggling with my tasks.")


# ---------- Response cycling ----------

class TestCycling:
    @pytest.fixture
    def provider(self):
        return MockProvider()

    def test_conversation_cycles(self, provider):
        """Same purpose should yield different responses on consecutive calls."""
        async def _run():
            results = []
            for _ in range(3):
                r = await provider.complete("system", [{"role": "user", "content": "Hi"}],
                                            purpose="conversation")
                results.append(r)
            # At least 2 of 3 should differ (cycling through neutral pool)
            assert len(set(results)) >= 2
        asyncio.new_event_loop().run_until_complete(_run())

    def test_reflection_cycles(self, provider):
        async def _run():
            results = set()
            for _ in range(5):
                r = await provider.complete("system", [{"role": "user", "content": "reflect"}],
                                            purpose="reflection")
                results.add(r)
            assert len(results) >= 3, "Reflections should vary"
        asyncio.new_event_loop().run_until_complete(_run())

    def test_importance_varies(self, provider):
        async def _run():
            results = set()
            for _ in range(6):
                r = await provider.complete("system", [{"role": "user", "content": "event"}],
                                            purpose="importance")
                results.add(r)
            assert len(results) >= 2, "Importance scores should vary"
        asyncio.new_event_loop().run_until_complete(_run())

    def test_reaction_varies(self, provider):
        async def _run():
            results = set()
            for _ in range(8):
                r = await provider.complete("system", [{"role": "user", "content": "obs"}],
                                            purpose="reaction")
                results.add(r)
            assert len(results) >= 2
        asyncio.new_event_loop().run_until_complete(_run())


# ---------- Context-aware responses ----------

class TestContextAware:
    @pytest.fixture
    def provider(self):
        return MockProvider()

    def test_negative_sentiment_conversation(self, provider):
        """Prompt with hostile signals should pick from negative pool."""
        async def _run():
            prompt = "You are talking with Bob. disposition: hostile, trust: -30"
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="conversation")
            from core.npc.mock_provider import _CONVERSATION_NEGATIVE
            assert r in _CONVERSATION_NEGATIVE
        asyncio.new_event_loop().run_until_complete(_run())

    def test_positive_sentiment_conversation(self, provider):
        async def _run():
            prompt = "You are talking with Alice. disposition: friendly, trust: 40"
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="conversation")
            from core.npc.mock_provider import _CONVERSATION_POSITIVE
            assert r in _CONVERSATION_POSITIVE
        asyncio.new_event_loop().run_until_complete(_run())

    def test_neutral_conversation(self, provider):
        async def _run():
            prompt = "You are talking with someone."
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="conversation")
            from core.npc.mock_provider import _CONVERSATION_NEUTRAL
            assert r in _CONVERSATION_NEUTRAL
        asyncio.new_event_loop().run_until_complete(_run())

    def test_help_conversation(self, provider):
        async def _run():
            prompt = "You need help carrying supplies."
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="conversation")
            from core.npc.mock_provider import _CONVERSATION_HELP
            assert r in _CONVERSATION_HELP
        asyncio.new_event_loop().run_until_complete(_run())

    def test_farmer_daily_plan(self, provider):
        async def _run():
            prompt = "You are Bob, a farmer in Smallville."
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="daily_plan")
            from core.npc.mock_provider import _DAILY_PLANS
            assert r in _DAILY_PLANS["farmer"]
        asyncio.new_event_loop().run_until_complete(_run())

    def test_blacksmith_task(self, provider):
        async def _run():
            prompt = "You are a blacksmith at the forge."
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="task_decomposition")
            from core.npc.mock_provider import _TASK_DECOMPOSITION
            assert r in _TASK_DECOMPOSITION["blacksmith"]
        asyncio.new_event_loop().run_until_complete(_run())

    def test_default_occupation_fallback(self, provider):
        async def _run():
            prompt = "You are a librarian."
            r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                        purpose="daily_plan")
            from core.npc.mock_provider import _DAILY_PLANS
            assert r in _DAILY_PLANS["default"]
        asyncio.new_event_loop().run_until_complete(_run())


# ---------- Legacy overrides ----------

class TestLegacyOverrides:
    def test_exact_override(self):
        async def _run():
            provider = MockProvider(responses={"conversation": "Custom response."})
            r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                        purpose="conversation")
            assert r == "Custom response."
        asyncio.new_event_loop().run_until_complete(_run())

    def test_override_does_not_affect_other_purposes(self):
        async def _run():
            provider = MockProvider(responses={"conversation": "Custom."})
            r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                        purpose="reflection")
            assert r != "Custom."
        asyncio.new_event_loop().run_until_complete(_run())


# ---------- register_responses ----------

class TestRegisterResponses:
    def test_custom_pool(self):
        async def _run():
            provider = MockProvider()
            provider.register_responses("conversation", "neutral", [
                "Custom neutral 1", "Custom neutral 2",
            ])
            r = await provider.complete(
                "Just a chat.", [{"role": "user", "content": ""}],
                purpose="conversation",
            )
            assert r in ("Custom neutral 1", "Custom neutral 2")
        asyncio.new_event_loop().run_until_complete(_run())

    def test_custom_pool_cycles(self):
        async def _run():
            provider = MockProvider()
            provider.register_responses("reflection", "default", ["A", "B", "C"])
            results = []
            for _ in range(3):
                r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                            purpose="reflection")
                results.append(r)
            assert results == ["A", "B", "C"]
        asyncio.new_event_loop().run_until_complete(_run())

    def test_unknown_purpose_fallback(self):
        async def _run():
            provider = MockProvider()
            r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                        purpose="some_new_purpose")
            assert r == "Acknowledged."
        asyncio.new_event_loop().run_until_complete(_run())


# ---------- Call log ----------

class TestCallLog:
    def test_calls_logged(self):
        async def _run():
            provider = MockProvider()
            await provider.complete("sys", [{"role": "user", "content": "hi"}],
                                    purpose="conversation")
            assert len(provider.call_log) == 1
            assert provider.call_log[0]["purpose"] == "conversation"
        asyncio.new_event_loop().run_until_complete(_run())

    def test_all_purposes_return_strings(self):
        async def _run():
            provider = MockProvider()
            purposes = [
                "daily_plan", "conversation", "reflection",
                "reaction", "importance", "task_decomposition",
            ]
            for p in purposes:
                r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                            purpose=p)
                assert isinstance(r, str)
                assert len(r) > 0, f"Empty response for purpose={p}"
        asyncio.new_event_loop().run_until_complete(_run())
