"""Tests for the NPC cognition system — tiers, perception, planning, execution."""

import pytest
from core.npc.models import NPC, PersonalityTraits, ActivityState, Direction, ScheduleEntry
from core.npc.cognition.tiers import (
    assign_tier, should_perceive, should_plan,
    update_all_tiers, get_tier_config,
    TIER_1_RADIUS, TIER_2_RADIUS, TIER_3_RADIUS,
)
from core.npc.cognition.perceive import perceive, Observation, VISION_RADIUS
from core.npc.cognition.execute import (
    execute_tick, set_activity_for_location, navigate_to,
    _direction_towards,
)
from core.npc.cognition.plan import (
    _template_schedule, _parse_llm_schedule, resolve_schedule_location,
    DEFAULT_SCHEDULES,
)
from core.world.grid import Grid
from core.world.generator import PlacedBuilding


def _make_npc(npc_id="test_1", x=0, z=0, tier=3, **kwargs):
    defaults = {
        "name": "Test",
        "age": 30,
        "personality": PersonalityTraits(),
        "backstory": "",
        "occupation": "farmer",
    }
    defaults.update(kwargs)
    npc = NPC(npc_id=npc_id, x=x, z=z, **defaults)
    npc.cognition_tier = tier
    return npc


def _make_grid(width=20, height=20):
    return Grid(width, height)


# ---------- Tier Assignment ----------

class TestTierAssignment:
    def test_close_npc_gets_tier_1(self):
        npc = _make_npc(x=3, z=3)
        tier = assign_tier(npc, 0, 0)
        assert tier == 1

    def test_medium_npc_gets_tier_2(self):
        npc = _make_npc(x=15, z=0)
        tier = assign_tier(npc, 0, 0)
        assert tier == 2

    def test_far_npc_gets_tier_3(self):
        npc = _make_npc(x=25, z=0)
        tier = assign_tier(npc, 0, 0)
        assert tier == 3

    def test_very_far_npc_gets_tier_4(self):
        npc = _make_npc(x=100, z=100)
        tier = assign_tier(npc, 0, 0)
        assert tier == 4

    def test_relevance_boost_lowers_tier(self):
        npc = _make_npc(x=15, z=0)  # normally tier 2
        tier = assign_tier(npc, 0, 0, relevance_boost=1)
        assert tier == 1

    def test_conversation_forces_tier_2_max(self):
        npc = _make_npc(x=25, z=0)  # normally tier 3
        npc.conversation_partner = "other_1"
        tier = assign_tier(npc, 0, 0)
        assert tier == 2

    def test_tier_never_below_1(self):
        npc = _make_npc(x=0, z=0)
        tier = assign_tier(npc, 0, 0, relevance_boost=5)
        assert tier == 1


class TestTierScheduling:
    def test_tier_1_should_perceive(self):
        npc = _make_npc(tier=1)
        npc.last_perception_tick = 0
        assert should_perceive(npc, 5.0)

    def test_tier_1_no_perceive_too_soon(self):
        npc = _make_npc(tier=1)
        npc.last_perception_tick = 4.0
        assert not should_perceive(npc, 5.0)

    def test_tier_4_never_perceives(self):
        npc = _make_npc(tier=4)
        npc.last_perception_tick = 0
        assert not should_perceive(npc, 1000.0)

    def test_tier_3_no_llm_planning(self):
        npc = _make_npc(tier=3)
        assert not should_plan(npc, 1000.0)

    def test_tier_1_should_plan(self):
        npc = _make_npc(tier=1)
        npc.last_plan_tick = 0
        assert should_plan(npc, 20.0)


class TestUpdateAllTiers:
    def test_groups_npcs_by_tier(self):
        npcs = [
            _make_npc("close", x=2, z=0),
            _make_npc("mid", x=15, z=0),
            _make_npc("far", x=25, z=0),
            _make_npc("frozen", x=50, z=50),
        ]
        groups = update_all_tiers(npcs, 0, 0)
        assert "close" in groups[1]
        assert "mid" in groups[2]
        assert "far" in groups[3]
        assert "frozen" in groups[4]


# ---------- Perception ----------

class TestPerception:
    def test_perceives_nearby_npc(self):
        npc = _make_npc("observer", x=0, z=0, tier=1)
        other = _make_npc("target", x=2, z=0, occupation="blacksmith")
        grid = _make_grid()

        observations = perceive(npc, grid, [npc, other], 10.0)
        assert len(observations) >= 1
        assert any("blacksmith" in o.description for o in observations)

    def test_does_not_perceive_self(self):
        npc = _make_npc("observer", x=0, z=0, tier=1)
        grid = _make_grid()

        observations = perceive(npc, grid, [npc], 10.0)
        assert len(observations) == 0

    def test_does_not_perceive_far_npc(self):
        npc = _make_npc("observer", x=0, z=0, tier=1)
        other = _make_npc("distant", x=50, z=50)
        grid = _make_grid(width=60, height=60)

        observations = perceive(npc, grid, [npc, other], 10.0)
        npc_obs = [o for o in observations if o.category == "npc"]
        assert len(npc_obs) == 0

    def test_tier_4_perceives_nothing(self):
        npc = _make_npc("frozen", x=0, z=0, tier=4)
        other = _make_npc("target", x=1, z=0)
        grid = _make_grid()

        observations = perceive(npc, grid, [npc, other], 10.0)
        assert len(observations) == 0

    def test_updates_perception_tick(self):
        npc = _make_npc("observer", x=0, z=0, tier=1)
        npc.last_perception_tick = 0
        grid = _make_grid()

        perceive(npc, grid, [npc], 42.0)
        assert npc.last_perception_tick == 42.0

    def test_retention_window_prevents_duplicates(self):
        npc = _make_npc("observer", x=0, z=0, tier=1)
        other = _make_npc("target", x=2, z=0, occupation="merchant")
        grid = _make_grid()

        obs1 = perceive(npc, grid, [npc, other], 10.0)
        obs2 = perceive(npc, grid, [npc, other], 15.0)
        # Second perception should be filtered as duplicate
        assert len(obs2) == 0 or len(obs2) < len(obs1)


# ---------- Planning ----------

class TestTemplateSchedule:
    def test_blacksmith_has_schedule(self):
        npc = _make_npc(occupation="blacksmith")
        schedule = _template_schedule(npc)
        assert len(schedule) >= 4

    def test_schedule_includes_night(self):
        npc = _make_npc(occupation="farmer")
        schedule = _template_schedule(npc)
        slots = {e.slot for e in schedule}
        assert "night" in slots

    def test_unknown_occupation_gets_fallback(self):
        npc = _make_npc(occupation="wizard")
        schedule = _template_schedule(npc)
        assert len(schedule) >= 4

    def test_all_default_occupations_have_schedules(self):
        for occupation in DEFAULT_SCHEDULES:
            npc = _make_npc(occupation=occupation)
            schedule = _template_schedule(npc)
            assert len(schedule) >= 4, f"{occupation} schedule too short"


class TestParseLLMSchedule:
    def test_parses_numbered_list(self):
        response = (
            "1. Wake up at 6:00 AM and eat breakfast\n"
            "2. Open the forge at 8:00 AM\n"
            "3. Work until 12:00 PM\n"
            "4. Lunch at the tavern at 1:00 PM\n"
            "5. Socialise at 6:00 PM\n"
            "6. Sleep at 9:00 PM"
        )
        entries = _parse_llm_schedule(response)
        assert len(entries) >= 4
        slots = {e.slot for e in entries}
        assert "night" in slots

    def test_handles_empty_response(self):
        entries = _parse_llm_schedule("")
        assert len(entries) >= 1  # at least fallback


class TestResolveLocation:
    def test_resolves_home(self):
        npc = _make_npc(home_x=5, home_z=10)
        # Need to set attributes directly since _make_npc doesn't forward them
        npc.home_x = 5
        npc.home_z = 10
        entry = ScheduleEntry("morning", "eat breakfast", "home", 3)
        x, z = resolve_schedule_location(entry, npc, [])
        assert x == 5
        assert z == 10

    def test_resolves_work(self):
        npc = _make_npc()
        npc.work_x = 8
        npc.work_z = -4
        entry = ScheduleEntry("morning", "work", "work", 7)
        x, z = resolve_schedule_location(entry, npc, [])
        assert x == 8
        assert z == -4

    def test_resolves_building_type(self):
        building = PlacedBuilding(
            building_type="tavern", name="Tavern",
            x=3, z=5, width=4, height=4,
            sector="market", door_x=5, door_z=9,
        )
        npc = _make_npc()
        entry = ScheduleEntry("evening", "eat", "tavern", 4)
        x, z = resolve_schedule_location(entry, npc, [building])
        assert x == 5
        assert z == 9  # door tile — NPC appears inside the building

    def test_falls_back_to_town_square(self):
        npc = _make_npc()
        entry = ScheduleEntry("afternoon", "wander", "unknown_place", 2)
        x, z = resolve_schedule_location(entry, npc, [])
        assert x == 0
        assert z == 0


# ---------- Execution ----------

class TestExecution:
    def test_navigate_to_sets_path(self):
        npc = _make_npc(x=0, z=0)
        grid = _make_grid()
        result = navigate_to(npc, grid, 3, 0)
        assert result is True
        assert len(npc.current_path) > 0
        assert npc.activity == ActivityState.WALKING

    def test_navigate_to_self_returns_true(self):
        npc = _make_npc(x=5, z=5)
        grid = _make_grid()
        result = navigate_to(npc, grid, 5, 5)
        assert result is True

    def test_set_activity_at_home_night(self):
        npc = _make_npc(x=5, z=5)
        npc.home_x = 5
        npc.home_z = 5
        set_activity_for_location(npc, "night")
        assert npc.activity == ActivityState.SLEEPING

    def test_set_activity_at_work(self):
        npc = _make_npc(x=8, z=-4)
        npc.work_x = 8
        npc.work_z = -4
        set_activity_for_location(npc, "morning")
        assert npc.activity == ActivityState.WORKING


class TestDirectionTowards:
    def test_east(self):
        assert _direction_towards(0, 0, 1, 0) == Direction.EAST

    def test_west(self):
        assert _direction_towards(0, 0, -1, 0) == Direction.WEST

    def test_north(self):
        assert _direction_towards(0, 0, 0, 1) == Direction.NORTH

    def test_south(self):
        assert _direction_towards(0, 0, 0, -1) == Direction.SOUTH
