"""Tests for the LLM client — mock provider, cost tracking, prompt templates."""

import pytest
from core.npc.llm_client import (
    MockProvider, CostTracker, RateLimiter,
    format_prompt, PROMPT_TEMPLATES,
)


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_returns_default_daily_plan(self):
        provider = MockProvider()
        result = await provider.complete(
            system="test", messages=[{"role": "user", "content": "test"}],
            purpose="daily_plan",
        )
        assert "Wake up" in result

    @pytest.mark.asyncio
    async def test_returns_custom_response(self):
        provider = MockProvider(responses={"greeting": "Hello there!"})
        result = await provider.complete(
            system="test", messages=[], purpose="greeting",
        )
        assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_logs_calls(self):
        provider = MockProvider()
        await provider.complete(system="sys", messages=[], purpose="test")
        assert len(provider.call_log) == 1
        assert provider.call_log[0]["purpose"] == "test"

    @pytest.mark.asyncio
    async def test_unknown_purpose_returns_acknowledged(self):
        provider = MockProvider()
        result = await provider.complete(system="", messages=[], purpose="unknown")
        assert result == "Acknowledged."


class TestCostTracker:
    def test_records_usage(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5-20251001", 100, 50, "test")
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert len(tracker.records) == 1

    def test_accumulates_tokens(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5-20251001", 100, 50, "a")
        tracker.record("claude-haiku-4-5-20251001", 200, 100, "b")
        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150

    def test_cost_estimation(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5-20251001", 1_000_000, 0, "test")
        cost = tracker.estimated_cost_usd()
        assert cost > 0

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("claude-haiku-4-5-20251001", 100, 50, "test")
        summary = tracker.summary()
        assert summary["total_calls"] == 1
        assert "estimated_cost_usd" in summary


class TestPromptTemplates:
    def test_all_templates_exist(self):
        expected = [
            "daily_plan", "task_decomposition",
            "conversation_initiate", "conversation_respond", "reaction",
        ]
        for name in expected:
            assert name in PROMPT_TEMPLATES

    def test_format_daily_plan(self):
        result = format_prompt(
            "daily_plan",
            name="Thorin", age=45, occupation="blacksmith",
            backstory="A veteran smith.", personality="gruff, honest",
            goals="Master the forge", health="100%", energy="80%",
            hunger="10%", gold=150, day=1,
            relationship_summary="No notable relationships yet.",
        )
        assert "Thorin" in result
        assert "blacksmith" in result

    def test_format_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt template"):
            format_prompt("nonexistent")
