"""Tests for the context-aware MockProvider."""

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

    @pytest.mark.asyncio
    async def test_conversation_cycles(self, provider):
        """Same purpose should yield different responses on consecutive calls."""
        results = []
        for _ in range(3):
            r = await provider.complete("system", [{"role": "user", "content": "Hi"}],
                                        purpose="conversation")
            results.append(r)
        # At least 2 of 3 should differ (cycling through neutral pool)
        assert len(set(results)) >= 2

    @pytest.mark.asyncio
    async def test_reflection_cycles(self, provider):
        results = set()
        for _ in range(5):
            r = await provider.complete("system", [{"role": "user", "content": "reflect"}],
                                        purpose="reflection")
            results.add(r)
        assert len(results) >= 3, "Reflections should vary"

    @pytest.mark.asyncio
    async def test_importance_varies(self, provider):
        results = set()
        for _ in range(6):
            r = await provider.complete("system", [{"role": "user", "content": "event"}],
                                        purpose="importance")
            results.add(r)
        assert len(results) >= 2, "Importance scores should vary"

    @pytest.mark.asyncio
    async def test_reaction_varies(self, provider):
        results = set()
        for _ in range(8):
            r = await provider.complete("system", [{"role": "user", "content": "obs"}],
                                        purpose="reaction")
            results.add(r)
        assert len(results) >= 2


# ---------- Context-aware responses ----------

class TestContextAware:
    @pytest.fixture
    def provider(self):
        return MockProvider()

    @pytest.mark.asyncio
    async def test_negative_sentiment_conversation(self, provider):
        """Prompt with hostile signals should pick from negative pool."""
        prompt = "You are talking with Bob. disposition: hostile, trust: -30"
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="conversation")
        from core.npc.mock_provider import _CONVERSATION_NEGATIVE
        assert r in _CONVERSATION_NEGATIVE

    @pytest.mark.asyncio
    async def test_positive_sentiment_conversation(self, provider):
        prompt = "You are talking with Alice. disposition: friendly, trust: 40"
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="conversation")
        from core.npc.mock_provider import _CONVERSATION_POSITIVE
        assert r in _CONVERSATION_POSITIVE

    @pytest.mark.asyncio
    async def test_neutral_conversation(self, provider):
        prompt = "You are talking with someone."
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="conversation")
        from core.npc.mock_provider import _CONVERSATION_NEUTRAL
        assert r in _CONVERSATION_NEUTRAL

    @pytest.mark.asyncio
    async def test_help_conversation(self, provider):
        prompt = "You need help carrying supplies."
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="conversation")
        from core.npc.mock_provider import _CONVERSATION_HELP
        assert r in _CONVERSATION_HELP

    @pytest.mark.asyncio
    async def test_farmer_daily_plan(self, provider):
        prompt = "You are Bob, a farmer in Smallville."
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="daily_plan")
        from core.npc.mock_provider import _DAILY_PLANS
        assert r in _DAILY_PLANS["farmer"]

    @pytest.mark.asyncio
    async def test_blacksmith_task(self, provider):
        prompt = "You are a blacksmith at the forge."
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="task_decomposition")
        from core.npc.mock_provider import _TASK_DECOMPOSITION
        assert r in _TASK_DECOMPOSITION["blacksmith"]

    @pytest.mark.asyncio
    async def test_default_occupation_fallback(self, provider):
        prompt = "You are a librarian."
        r = await provider.complete(prompt, [{"role": "user", "content": ""}],
                                    purpose="daily_plan")
        from core.npc.mock_provider import _DAILY_PLANS
        assert r in _DAILY_PLANS["default"]


# ---------- Legacy overrides ----------

class TestLegacyOverrides:
    @pytest.mark.asyncio
    async def test_exact_override(self):
        provider = MockProvider(responses={"conversation": "Custom response."})
        r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                    purpose="conversation")
        assert r == "Custom response."

    @pytest.mark.asyncio
    async def test_override_does_not_affect_other_purposes(self):
        provider = MockProvider(responses={"conversation": "Custom."})
        r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                    purpose="reflection")
        assert r != "Custom."


# ---------- register_responses ----------

class TestRegisterResponses:
    @pytest.mark.asyncio
    async def test_custom_pool(self):
        provider = MockProvider()
        provider.register_responses("conversation", "neutral", [
            "Custom neutral 1", "Custom neutral 2",
        ])
        r = await provider.complete(
            "Just a chat.", [{"role": "user", "content": ""}],
            purpose="conversation",
        )
        assert r in ("Custom neutral 1", "Custom neutral 2")

    @pytest.mark.asyncio
    async def test_custom_pool_cycles(self):
        provider = MockProvider()
        provider.register_responses("reflection", "default", ["A", "B", "C"])
        results = []
        for _ in range(3):
            r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                        purpose="reflection")
            results.append(r)
        assert results == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_unknown_purpose_fallback(self):
        provider = MockProvider()
        r = await provider.complete("sys", [{"role": "user", "content": ""}],
                                    purpose="some_new_purpose")
        assert r == "Acknowledged."


# ---------- Call log ----------

class TestCallLog:
    @pytest.mark.asyncio
    async def test_calls_logged(self):
        provider = MockProvider()
        await provider.complete("sys", [{"role": "user", "content": "hi"}],
                                purpose="conversation")
        assert len(provider.call_log) == 1
        assert provider.call_log[0]["purpose"] == "conversation"

    @pytest.mark.asyncio
    async def test_all_purposes_return_strings(self):
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
