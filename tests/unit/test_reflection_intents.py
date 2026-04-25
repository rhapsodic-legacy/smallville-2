"""Tests for dynamic schedule injection from reflection action intents."""

import asyncio

import pytest

from core.memory.reflection import (
    ActionIntent,
    _parse_action_intent,
    _classify_insight_heuristic,
    classify_insight,
    run_reflection_with_intents,
)
from core.npc.models import NPC, PersonalityTraits, ScheduleEntry


# ---------- Fixtures ----------

def _make_npc(npc_id: str = "test_0", name: str = "Alice", tier: int = 3) -> NPC:
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory="A test NPC.",
        occupation="farmer",
        cognition_tier=tier,
    )


def _make_schedule(count: int = 4) -> list[ScheduleEntry]:
    """Build a simple 4-entry schedule totalling 1440 minutes."""
    activities = [
        ("work in the fields", "work", 360),
        ("eat at the tavern", "tavern", 120),
        ("rest in the square", "town_square", 480),
        ("sleep at home", "home", 480),
    ]
    return [
        ScheduleEntry(
            slot=f"slot_{i}",
            activity=act,
            location=loc,
            duration_minutes=dur,
        )
        for i, (act, loc, dur) in enumerate(activities[:count])
    ]


# ---------- _parse_action_intent ----------

class TestParseActionIntent:
    def test_valid_response(self):
        response = (
            "ACTION: bring lunch to Bob at the bridge\n"
            "LOCATION: tavern\n"
            "DURATION: 30"
        )
        intent = _parse_action_intent(response)
        assert intent is not None
        assert intent.activity == "bring lunch to Bob at the bridge"
        assert intent.location == "tavern"
        assert intent.duration_minutes == 30

    def test_no_action_response(self):
        assert _parse_action_intent("NO_ACTION") is None

    def test_missing_action_line(self):
        response = "LOCATION: tavern\nDURATION: 30"
        assert _parse_action_intent(response) is None

    def test_duration_clamped_low(self):
        response = "ACTION: check the forge\nLOCATION: work\nDURATION: 5"
        intent = _parse_action_intent(response)
        assert intent is not None
        assert intent.duration_minutes == 15  # clamped to minimum

    def test_duration_clamped_high(self):
        response = "ACTION: wander the hills\nLOCATION: outskirts\nDURATION: 120"
        intent = _parse_action_intent(response)
        assert intent is not None
        assert intent.duration_minutes == 60  # clamped to maximum

    def test_duration_with_extra_text(self):
        response = "ACTION: visit the priest\nLOCATION: church\nDURATION: 45 minutes"
        intent = _parse_action_intent(response)
        assert intent is not None
        assert intent.duration_minutes == 45

    def test_default_location(self):
        response = "ACTION: look around town\nDURATION: 20"
        intent = _parse_action_intent(response)
        assert intent is not None
        assert intent.location == "town_square"

    def test_default_duration(self):
        response = "ACTION: check on the crops\nLOCATION: farm"
        intent = _parse_action_intent(response)
        assert intent is not None
        assert intent.duration_minutes == 30  # default


# ---------- _classify_insight_heuristic ----------

class TestHeuristicClassification:
    def test_actionable_keyword(self):
        insight = "I should bring some food to Bob at the bridge."
        intent = _classify_insight_heuristic(insight)
        assert intent is not None
        assert intent.duration_minutes == 30
        assert intent.location == "town_square"

    def test_non_actionable_insight(self):
        insight = "Life in Smallville has been peaceful lately."
        intent = _classify_insight_heuristic(insight)
        assert intent is None

    def test_multiple_keywords_returns_one(self):
        insight = "I should visit the tavern and I need to check on my crops."
        intent = _classify_insight_heuristic(insight)
        assert intent is not None

    def test_case_insensitive(self):
        insight = "I SHOULD VISIT the merchant to discuss trade."
        intent = _classify_insight_heuristic(insight)
        assert intent is not None


# ---------- Schedule injection (via NPCManager._inject_reflection_entry) ----------

class TestScheduleInjection:
    """Test the injection mechanics directly on NPC schedule lists."""

    def test_inject_at_next_position(self):
        """Intent inserts at schedule_index + 1."""
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=42)
        mgr.spawn_population(1)

        npc = mgr.npcs[0]
        npc.daily_schedule = _make_schedule(4)
        npc.schedule_index = 1  # currently on entry 1

        intent = ActionIntent(
            activity="bring lunch to Bob",
            location="tavern",
            duration_minutes=30,
        )
        mgr._inject_reflection_entry(npc, intent)

        # Should be inserted at index 2
        assert len(npc.daily_schedule) == 5
        injected = npc.daily_schedule[2]
        assert injected.activity == "bring lunch to Bob"
        assert injected.location == "tavern"
        assert injected.duration_minutes == 30
        assert injected.slot == "reflection"
        assert injected.priority == 7

    def test_inject_preserves_current_entry(self):
        """Current schedule entry is NOT displaced."""
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=42)
        mgr.spawn_population(1)

        npc = mgr.npcs[0]
        npc.daily_schedule = _make_schedule(4)
        npc.schedule_index = 0

        original_first = npc.daily_schedule[0].activity
        intent = ActionIntent("check the forge", "work", 20)
        mgr._inject_reflection_entry(npc, intent)

        assert npc.daily_schedule[0].activity == original_first
        assert npc.daily_schedule[1].activity == "check the forge"

    def test_inject_at_end_of_schedule(self):
        """Intent inserted when NPC is on last entry."""
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=42)
        mgr.spawn_population(1)

        npc = mgr.npcs[0]
        npc.daily_schedule = _make_schedule(3)
        npc.schedule_index = 2  # last entry

        intent = ActionIntent("visit the priest", "church", 45)
        mgr._inject_reflection_entry(npc, intent)

        assert len(npc.daily_schedule) == 4
        assert npc.daily_schedule[3].activity == "visit the priest"

    def test_no_inject_on_empty_schedule(self):
        """Nothing happens if the NPC has no schedule."""
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world
        from core.npc.llm_client import MockProvider

        config = WorldConfig(population=2, terrain="riverside", seed=42)
        grid, buildings = generate_world(config)
        mgr = NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=42)
        mgr.spawn_population(1)

        npc = mgr.npcs[0]
        npc.daily_schedule = []

        intent = ActionIntent("wander", "outskirts", 30)
        mgr._inject_reflection_entry(npc, intent)

        assert npc.daily_schedule == []


# ---------- classify_insight (async, uses mock) ----------

class TestClassifyInsight:
    def test_heuristic_for_tier3(self):
        """Tier 3 NPCs use heuristic classification, no LLM."""
        async def _run():
            npc = _make_npc(tier=3)
            intent = await classify_insight(
                npc, "I should visit the tavern keeper.", None,
            )
            assert intent is not None
            assert "visit" in intent.activity.lower()
        asyncio.new_event_loop().run_until_complete(_run())

    def test_non_actionable_tier3(self):
        async def _run():
            npc = _make_npc(tier=3)
            intent = await classify_insight(
                npc, "Life is peaceful in Smallville.", None,
            )
            assert intent is None
        asyncio.new_event_loop().run_until_complete(_run())
