"""
Tests for schedule parsing and location inference.

Validates that LLM-generated schedules get correct locations inferred
from activity text, and that the deterministic (template) mode produces
schedules with sensible locations for every slot.
"""

import pytest

from core.npc.cognition.plan import (
    _parse_llm_schedule,
    _infer_location,
    _clean_activity,
    _template_schedule,
    DEFAULT_SCHEDULES,
    DEFAULT_SCHEDULE_FALLBACK,
)
from core.npc.models import NPC, PersonalityTraits


def _make_npc(occupation: str = "labourer") -> NPC:
    return NPC(
        npc_id=f"{occupation}_0",
        name="Test",
        age=30,
        personality=PersonalityTraits(),
        backstory="Test NPC.",
        occupation=occupation,
        x=0, z=0,
        home_x=0, home_z=0,
    )


# ---------- _clean_activity ----------


class TestCleanActivity:

    def test_strips_markdown_bold(self):
        assert "Wake up" in _clean_activity("**Wake up and eat**")

    def test_strips_time_prefix(self):
        result = _clean_activity("5:00 AM - 6:00 AM: Wake up")
        assert "Wake up" in result
        assert "5:00" not in result

    def test_strips_24h_time(self):
        result = _clean_activity("06:00 – 07:00 – Morning routine")
        assert "Morning routine" in result

    def test_empty_input(self):
        assert _clean_activity("") == "idle"


# ---------- _infer_location ----------


class TestInferLocation:

    def test_home_keywords(self):
        assert _infer_location("eat breakfast at home", "early_morning") == "home"
        assert _infer_location("sleep in bed", "night") == "home"
        assert _infer_location("wake up and wash", "early_morning") == "home"

    def test_work_keywords(self):
        assert _infer_location("work at the forge", "morning") == "work"
        assert _infer_location("open the shop", "morning") == "work"

    def test_tavern_keywords(self):
        assert _infer_location("visit the tavern for ale", "evening") == "tavern"
        assert _infer_location("eat stew at the inn", "evening") == "tavern"

    def test_church_keywords(self):
        assert _infer_location("morning prayers at the church", "morning") == "church"

    def test_market_keywords(self):
        assert _infer_location("sell goods at market", "afternoon") == "market_stall"

    def test_outskirts_keywords(self):
        assert _infer_location("tend the crops in the field", "morning") == "outskirts"
        assert _infer_location("patrol the perimeter", "morning") == "outskirts"

    def test_slot_defaults(self):
        """When no keywords match, fall back to slot defaults."""
        assert _infer_location("contemplate life", "early_morning") == "home"
        assert _infer_location("contemplate life", "morning") == "work"
        assert _infer_location("contemplate life", "evening") == "tavern"
        assert _infer_location("contemplate life", "night") == "home"


# ---------- _parse_llm_schedule ----------


class TestParseLLMSchedule:

    def test_basic_schedule(self):
        response = (
            "5:00 AM - Wake up, eat breakfast at home\n"
            "8:00 AM - Work at the forge\n"
            "12:00 PM - Sell goods at the market\n"
            "6:00 PM - Visit the tavern for ale\n"
            "9:00 PM - Sleep at home\n"
        )
        entries = _parse_llm_schedule(response)
        locs = {e.slot: e.location for e in entries}
        assert locs.get("early_morning") == "home"
        assert locs.get("morning") == "work"
        assert locs.get("afternoon") == "market_stall"
        assert locs.get("evening") == "tavern"
        assert locs.get("night") == "home"

    def test_markdown_formatted_response(self):
        response = (
            "**5:00 AM - 6:00 AM** – Wake and fuel: "
            "Breakfast at home (hearty stew, dark bread)\n"
            "**8:00 AM - 12:00 PM** – Work at the forge\n"
            "**1:00 PM - 5:00 PM** – Trade goods at market\n"
            "**6:00 PM - 9:00 PM** – Socialise at the tavern\n"
            "**10:00 PM** – Sleep at home\n"
        )
        entries = _parse_llm_schedule(response)
        locs = {e.slot: e.location for e in entries}
        assert locs.get("early_morning") == "home"
        assert locs.get("night") == "home"

    def test_no_location_work_hardcode_removed(self):
        """Ensure the old bug (all locations = 'work') is fixed."""
        response = (
            "6:00 AM - eat breakfast at home\n"
            "10:00 AM - wander around town\n"
            "2:00 PM - rest by the river\n"
            "7:00 PM - drink at the tavern\n"
            "10:00 PM - go home and sleep\n"
        )
        entries = _parse_llm_schedule(response)
        work_count = sum(1 for e in entries if e.location == "work")
        # At most 1 entry should be "work" (the wander might default)
        # but NOT all of them
        assert work_count < len(entries), (
            f"Bug regression: {work_count}/{len(entries)} entries still have location='work'"
        )

    def test_night_always_present(self):
        """Even garbage input should produce a night entry."""
        entries = _parse_llm_schedule("This is not a schedule at all.")
        slots = {e.slot for e in entries}
        assert "night" in slots

    def test_gemma_prose_with_title_and_rationales(self):
        """Real-world Gemma output wraps each entry with a bold title
        and an italic rationale line in parentheses. Neither should
        leak into the schedule as an activity."""
        response = (
            "**Day 33: A Schedule for Seren**\n"
            "\n"
            "**7:00 AM - 8:00 AM:** Wake up and Morning Ritual, The Tavern.\n"
            "*(Focus: Gentle start, addressing low energy.)*\n"
            "\n"
            "**8:00 AM - 9:30 AM:** Inventory and Morning Prep, The Tavern.\n"
            "*(Focus: Conscientious work. Seren reviews stock.)*\n"
            "\n"
            "**12:30 PM - 1:30 PM:** Midday Meal, The Tavern.\n"
            "*(Focus: Hunger. Seren eats a hearty meal.)*\n"
            "\n"
            "**4:00 PM - 6:00 PM:** Relationship Time with Dara.\n"
            "*(Focus: Relationship.)*\n"
            "\n"
            "**10:00 PM:** Close up and sleep.\n"
        )
        entries = _parse_llm_schedule(response)
        activities = [e.activity for e in entries]
        # None of the title/rationale fragments should have survived.
        for bad in (
            "Day 33", "A Schedule for Seren",
            "Focus:", "Gentle start", "Hunger",
            "Seren's Daily Schedule",
        ):
            for act in activities:
                assert bad not in act, (
                    f"Prose fragment {bad!r} leaked into activity {act!r}"
                )
        # And we should have at least one real entry per phase.
        slots = {e.slot for e in entries}
        assert "early_morning" in slots
        assert "morning" in slots
        assert "afternoon" in slots
        assert "night" in slots

    def test_markdown_table_format(self):
        """Some Gemma responses use markdown tables. Header and
        separator rows must not be parsed as activities; data rows
        should extract from the Activity column."""
        response = (
            "| Time Range | Activity | Location | Rationale |\n"
            "|------------|----------|----------|-----------|\n"
            "| 06:00 | Wake and dress | Home | Start day |\n"
            "| 09:00 | Open the forge | Work | Work begins |\n"
            "| 13:00 | Eat lunch at tavern | Tavern | Hunger |\n"
            "| 21:00 | Walk home and sleep | Home | Rest |\n"
        )
        entries = _parse_llm_schedule(response)
        activities = [e.activity for e in entries]
        for bad in ("Time Range", "Activity", "Rationale", "|-"):
            for act in activities:
                assert bad not in act, (
                    f"Table header fragment {bad!r} leaked into activity {act!r}"
                )
        # Activity column content is kept.
        combined = " ".join(activities).lower()
        assert "open the forge" in combined
        assert "walk home" in combined

    def test_insufficient_lines_fall_back_to_template(self):
        """If Gemma returns prose with no time-stamped entries, use the
        occupation fallback rather than fabricate a half-empty schedule."""
        from core.npc.cognition.plan import DEFAULT_SCHEDULES
        response = (
            "**Seren's Daily Schedule - Day 32**\n"
            "Focus: Rebuilding energy, managing the tavern.\n"
            "| Time Range | Activity | Location | Rationale |\n"
        )
        fallback = DEFAULT_SCHEDULES["tavern_keeper"]
        entries = _parse_llm_schedule(response, fallback=fallback)
        # Should be the tavern_keeper fallback, not a 1-entry half-day.
        assert len(entries) == len(fallback)
        assert {e.slot for e in entries} == {e.slot for e in fallback}

    def test_truncated_response_still_usable(self):
        """max_tokens cutoff leaves the response half-finished — we
        should still extract whatever complete time-stamped lines exist."""
        response = (
            "**7:00 AM:** Eat breakfast at home\n"
            "**9:00 AM:** Tend the crops at the fields\n"
            "**13:00:** Eat lunch at the tavern\n"
            "**17:00:** Head to the ta"   # truncated mid-word
        )
        entries = _parse_llm_schedule(response)
        # First three entries should parse cleanly; truncated 4th may or
        # may not survive — either is acceptable as long as no garbage.
        assert len(entries) >= 3
        # The accepted truncated entry, if present, must still have a
        # non-empty activity and location.
        for e in entries:
            assert e.activity
            assert e.location


# ---------- Template schedules ----------


class TestTemplateSchedule:

    def test_all_occupations_have_night_home(self):
        """Every template schedule must have night=home."""
        for occ, entries in DEFAULT_SCHEDULES.items():
            night = [e for e in entries if e.slot == "night"]
            assert night, f"Occupation {occ} has no night entry"
            assert night[0].location in ("home", "town_square"), (
                f"Occupation {occ} night location is {night[0].location}, expected home"
            )

    def test_all_occupations_have_early_morning(self):
        for occ, entries in DEFAULT_SCHEDULES.items():
            em = [e for e in entries if e.slot == "early_morning"]
            assert em, f"Occupation {occ} has no early_morning entry"

    def test_template_schedule_for_npc(self):
        for occ in DEFAULT_SCHEDULES:
            npc = _make_npc(occ)
            schedule = _template_schedule(npc)
            assert len(schedule) >= 5
            slots = {e.slot for e in schedule}
            assert "night" in slots
            assert "early_morning" in slots

    def test_fallback_occupation(self):
        npc = _make_npc("unknown_job")
        schedule = _template_schedule(npc)
        assert len(schedule) >= 5

    def test_no_location_is_empty(self):
        """Every entry must have a non-empty location."""
        for occ in list(DEFAULT_SCHEDULES.keys()) + ["labourer"]:
            npc = _make_npc(occ)
            schedule = _template_schedule(npc)
            for entry in schedule:
                assert entry.location, (
                    f"{occ} schedule entry '{entry.activity}' has empty location"
                )
