"""Tests for the NPC data model."""

import pytest
from core.npc.models import (
    NPC, PersonalityTraits, ActivityState, Direction, ScheduleEntry,
    OCCUPATION_DEFAULTS, FIRST_NAMES,
)


class TestPersonalityTraits:
    def test_default_values(self):
        p = PersonalityTraits()
        assert p.openness == 0.5
        assert p.conscientiousness == 0.5

    def test_description_low_values(self):
        p = PersonalityTraits(openness=0.1, extraversion=0.2)
        desc = p.to_description()
        assert "conventional" in desc
        assert "introverted" in desc

    def test_description_high_values(self):
        p = PersonalityTraits(agreeableness=0.9, neuroticism=0.8)
        desc = p.to_description()
        assert "cooperative" in desc
        assert "anxious" in desc

    def test_description_balanced(self):
        p = PersonalityTraits()
        desc = p.to_description()
        assert desc == "balanced temperament"

    def test_to_dict(self):
        p = PersonalityTraits(openness=0.3)
        d = p.to_dict()
        assert d["openness"] == 0.3
        assert len(d) == 5


class TestNPCCreation:
    def _make_npc(self, **overrides):
        defaults = {
            "npc_id": "test_1",
            "name": "Testman",
            "age": 30,
            "personality": PersonalityTraits(),
            "backstory": "A test NPC.",
            "occupation": "blacksmith",
        }
        defaults.update(overrides)
        return NPC(**defaults)

    def test_basic_creation(self):
        npc = self._make_npc()
        assert npc.npc_id == "test_1"
        assert npc.name == "Testman"
        assert npc.health == 1.0
        assert npc.energy == 1.0
        assert npc.hunger == 0.0

    def test_default_activity_is_idle(self):
        npc = self._make_npc()
        assert npc.activity == ActivityState.IDLE

    def test_default_direction_is_south(self):
        npc = self._make_npc()
        assert npc.direction == Direction.SOUTH

    def test_default_cognition_tier(self):
        npc = self._make_npc()
        assert npc.cognition_tier == 3


class TestNPCNeeds:
    def _make_npc(self):
        return NPC(
            npc_id="test_1", name="Test", age=30,
            personality=PersonalityTraits(), backstory="", occupation="farmer",
        )

    def test_hunger_increases_over_time(self):
        npc = self._make_npc()
        npc.hunger = 0.0
        npc.tick_needs(100)  # 100 game minutes
        assert npc.hunger > 0.0

    def test_energy_drains_while_working(self):
        npc = self._make_npc()
        npc.energy = 1.0
        npc.activity = ActivityState.WORKING
        npc.tick_needs(100)
        assert npc.energy < 1.0

    def test_energy_recovers_while_sleeping(self):
        npc = self._make_npc()
        npc.energy = 0.3
        npc.activity = ActivityState.SLEEPING
        npc.tick_needs(100)
        assert npc.energy > 0.3

    def test_hunger_decreases_while_eating(self):
        npc = self._make_npc()
        npc.hunger = 0.8
        npc.activity = ActivityState.EATING
        npc.tick_needs(50)
        assert npc.hunger < 0.8

    def test_hunger_capped_at_one(self):
        npc = self._make_npc()
        npc.hunger = 0.99
        npc.tick_needs(1000)
        assert npc.hunger <= 1.0

    def test_energy_capped_at_zero(self):
        npc = self._make_npc()
        npc.energy = 0.01
        npc.activity = ActivityState.WORKING
        npc.tick_needs(1000)
        assert npc.energy >= 0.0


class TestNPCSpatial:
    def _make_npc(self, x=0, z=0):
        return NPC(
            npc_id="test_1", name="Test", age=30,
            personality=PersonalityTraits(), backstory="", occupation="farmer",
            x=x, z=z,
        )

    def test_is_at(self):
        npc = self._make_npc(5, 10)
        assert npc.is_at(5, 10)
        assert not npc.is_at(5, 11)

    def test_distance_to(self):
        npc = self._make_npc(0, 0)
        assert npc.distance_to(3, 4) == 7  # Manhattan
        assert npc.distance_to(0, 0) == 0


class TestNPCSchedule:
    def _make_npc(self):
        npc = NPC(
            npc_id="test_1", name="Test", age=30,
            personality=PersonalityTraits(), backstory="", occupation="farmer",
        )
        npc.daily_schedule = [
            ScheduleEntry("morning", "work the fields", "work", 7),
            ScheduleEntry("evening", "eat at tavern", "tavern", 4),
        ]
        npc.schedule_day = 1
        return npc

    def test_needs_new_schedule_when_exhausted(self):
        """Duration model: needs schedule when daily_schedule is empty."""
        npc = self._make_npc()
        npc.daily_schedule = []
        assert npc.needs_new_schedule(2)

    def test_no_new_schedule_when_has_entries_same_day(self):
        """Within the same day, a non-empty schedule is reused."""
        npc = self._make_npc()  # schedule_day=1
        assert not npc.needs_new_schedule(1)

    def test_regen_on_new_day_even_with_entries(self):
        """New day forces regeneration — this guards against schedules
        bloated by replan across days, which in a prior regression
        caused NPCs to never cycle to a fresh day and end up parked
        at the map border after ~40 game days."""
        npc = self._make_npc()  # schedule_day=1
        assert npc.needs_new_schedule(2)
        assert npc.needs_new_schedule(5)

    def test_needs_schedule_when_empty(self):
        npc = self._make_npc()
        npc.daily_schedule = []
        assert npc.needs_new_schedule(1)

    def test_get_schedule_entry(self):
        npc = self._make_npc()
        entry = npc.get_current_schedule_entry("morning")
        assert entry is not None
        assert entry.activity == "work the fields"

    def test_get_missing_schedule_entry(self):
        npc = self._make_npc()
        entry = npc.get_current_schedule_entry("night")
        assert entry is None


class TestNPCSerialisation:
    def _make_npc(self):
        return NPC(
            npc_id="smith_1", name="Thorin", age=45,
            personality=PersonalityTraits(openness=0.7),
            backstory="A veteran blacksmith.",
            occupation="blacksmith",
            x=5, z=-3, gold=150,
        )

    def test_to_dict_has_required_fields(self):
        npc = self._make_npc()
        d = npc.to_dict()
        assert d["npc_id"] == "smith_1"
        assert d["name"] == "Thorin"
        assert d["x"] == 5
        assert d["z"] == -3
        assert d["activity"] == "idle"
        assert d["occupation"] == "blacksmith"

    def test_to_full_dict_includes_identity(self):
        npc = self._make_npc()
        d = npc.to_full_dict()
        assert d["age"] == 45
        assert d["backstory"] == "A veteran blacksmith."
        assert d["gold"] == 150
        assert "personality" in d

    def test_summary_for_prompt(self):
        npc = self._make_npc()
        summary = npc.summary_for_prompt()
        assert "Thorin" in summary
        assert "blacksmith" in summary
        assert "45" in summary


class TestOccupationDefaults:
    def test_all_occupations_have_skills(self):
        for occ, data in OCCUPATION_DEFAULTS.items():
            assert "skills" in data, f"{occ} missing skills"

    def test_all_occupations_have_goals(self):
        for occ, data in OCCUPATION_DEFAULTS.items():
            assert "goals" in data, f"{occ} missing goals"

    def test_first_names_pool_not_empty(self):
        assert len(FIRST_NAMES) >= 20
