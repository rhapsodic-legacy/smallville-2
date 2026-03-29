"""
Planning module.

Generates daily schedules via LLM (tier 1-2) or templates (tier 3).
Decomposes schedule entries into concrete actions with target locations.
Handles reaction planning when NPCs encounter unexpected events.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from core.npc.models import ActivityState, ScheduleEntry

if TYPE_CHECKING:
    from core.npc.llm_client import LLMProvider
    from core.npc.models import NPC
    from core.world.grid import Grid
    from core.world.generator import PlacedBuilding

logger = logging.getLogger(__name__)


# ---------- Schedule slot mappings ----------

# Maps schedule slot names to default activities by occupation
DEFAULT_SCHEDULES: dict[str, list[ScheduleEntry]] = {
    "blacksmith": [
        ScheduleEntry("early_morning", "eat breakfast at home", "home", 3),
        ScheduleEntry("morning", "work at the forge", "work", 7),
        ScheduleEntry("afternoon", "work at the forge", "work", 7),
        ScheduleEntry("evening", "eat and socialise at the tavern", "tavern", 4),
        ScheduleEntry("night", "sleep at home", "home", 9),
    ],
    "farmer": [
        ScheduleEntry("early_morning", "tend the crops", "work", 7),
        ScheduleEntry("morning", "work the fields", "work", 7),
        ScheduleEntry("afternoon", "sell produce at the market", "market_stall", 5),
        ScheduleEntry("evening", "eat at the tavern", "tavern", 4),
        ScheduleEntry("night", "sleep at home", "home", 9),
    ],
    "merchant": [
        ScheduleEntry("early_morning", "eat breakfast at home", "home", 3),
        ScheduleEntry("morning", "open the market stall", "work", 7),
        ScheduleEntry("afternoon", "trade and negotiate", "work", 7),
        ScheduleEntry("evening", "socialise at the tavern", "tavern", 4),
        ScheduleEntry("night", "sleep at home", "home", 9),
    ],
    "tavern_keeper": [
        ScheduleEntry("early_morning", "prepare the tavern", "work", 6),
        ScheduleEntry("morning", "serve customers", "work", 7),
        ScheduleEntry("afternoon", "serve customers", "work", 7),
        ScheduleEntry("evening", "serve the evening crowd", "work", 8),
        ScheduleEntry("night", "close up and sleep", "home", 9),
    ],
    "priest": [
        ScheduleEntry("early_morning", "morning prayers", "work", 8),
        ScheduleEntry("morning", "counsel townsfolk", "work", 6),
        ScheduleEntry("afternoon", "walk through town", "town_square", 3),
        ScheduleEntry("evening", "evening service", "work", 6),
        ScheduleEntry("night", "sleep at home", "home", 9),
    ],
    "guard": [
        ScheduleEntry("early_morning", "patrol the perimeter", "outskirts", 7),
        ScheduleEntry("morning", "stand watch at the gate", "town_square", 7),
        ScheduleEntry("afternoon", "patrol the market", "market_stall", 6),
        ScheduleEntry("evening", "eat at the tavern", "tavern", 4),
        ScheduleEntry("night", "night watch", "town_square", 8),
    ],
}

# Fallback for unknown occupations
DEFAULT_SCHEDULE_FALLBACK = [
    ScheduleEntry("early_morning", "eat breakfast at home", "home", 3),
    ScheduleEntry("morning", "wander around town", "town_square", 2),
    ScheduleEntry("afternoon", "do odd jobs", "market_stall", 4),
    ScheduleEntry("evening", "visit the tavern", "tavern", 4),
    ScheduleEntry("night", "sleep at home", "home", 9),
]


async def generate_daily_schedule(
    npc: NPC,
    llm: LLMProvider,
    current_day: int,
    relationship_summary: str = "",
) -> list[ScheduleEntry]:
    """
    Generate a daily schedule for an NPC.

    Tier 1-2: Uses LLM to create a personalised schedule.
    Tier 3+: Uses occupation-based template with slight randomisation.
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config

    config = get_tier_config(npc.cognition_tier)

    if config.uses_llm:
        schedule = await _llm_schedule(npc, llm, current_day, relationship_summary)
    else:
        schedule = _template_schedule(npc)

    npc.daily_schedule = schedule
    npc.schedule_day = current_day

    logger.debug(
        "%s generated schedule for day %d (%d entries, tier %d)",
        npc.name, current_day, len(schedule), npc.cognition_tier,
    )
    return schedule


async def _llm_schedule(
    npc: NPC,
    llm: LLMProvider,
    current_day: int,
    relationship_summary: str = "",
) -> list[ScheduleEntry]:
    """Generate schedule via LLM, with fallback to template on failure."""
    from core.npc.llm_client import format_prompt

    try:
        prompt = format_prompt(
            "daily_plan",
            name=npc.name,
            age=npc.age,
            occupation=npc.occupation,
            backstory=npc.backstory,
            personality=npc.personality.to_description(),
            goals="; ".join(npc.long_term_goals[:3]),
            health=f"{npc.health:.0%}",
            energy=f"{npc.energy:.0%}",
            hunger=f"{npc.hunger:.0%}",
            gold=npc.gold,
            day=current_day,
            relationship_summary=relationship_summary or "No notable relationships yet.",
        )

        response = await llm.complete(
            system="You are a daily schedule planner for a medieval NPC.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.8,
            purpose="daily_plan",
        )

        return _parse_llm_schedule(response)

    except Exception as e:
        logger.warning(
            "LLM schedule failed for %s: %s — using template",
            npc.name, e,
        )
        return _template_schedule(npc)


def _parse_llm_schedule(response: str) -> list[ScheduleEntry]:
    """
    Parse LLM schedule response into ScheduleEntry list.

    Falls back to template-style entries if parsing fails.
    Maps time ranges to schedule slots.
    """
    entries: list[ScheduleEntry] = []
    slot_map = {
        range(5, 8): "early_morning",
        range(8, 12): "morning",
        range(12, 17): "afternoon",
        range(17, 21): "evening",
        range(21, 24): "night",
        range(0, 5): "night",
    }

    lines = response.strip().split("\n")
    assigned_slots: set[str] = set()

    for line in lines:
        line = line.strip().lstrip("0123456789.-) ")
        if not line:
            continue

        # Try to extract hour from common patterns like "6:00" or "06:00"
        hour = _extract_hour(line)
        if hour is not None:
            slot = _hour_to_slot(hour, slot_map)
        else:
            # Assign to first unassigned slot
            slot = _next_unassigned_slot(assigned_slots)

        if slot and slot not in assigned_slots:
            entries.append(ScheduleEntry(
                slot=slot,
                activity=line,
                location="work",  # resolved later by execution module
                priority=5,
            ))
            assigned_slots.add(slot)

    # Ensure we have at least sleep
    if "night" not in assigned_slots:
        entries.append(ScheduleEntry("night", "sleep at home", "home", 9))

    return entries if entries else DEFAULT_SCHEDULE_FALLBACK[:]


def _extract_hour(text: str) -> int | None:
    """Try to pull an hour number from text like '6:00 AM' or '14:00'."""
    import re
    match = re.search(r'(\d{1,2}):?\d{0,2}\s*(AM|PM|am|pm)?', text)
    if not match:
        return None
    hour = int(match.group(1))
    ampm = match.group(2)
    if ampm and ampm.upper() == "PM" and hour < 12:
        hour += 12
    if ampm and ampm.upper() == "AM" and hour == 12:
        hour = 0
    return hour if 0 <= hour < 24 else None


def _hour_to_slot(hour: int, slot_map: dict) -> str:
    for hours_range, slot in slot_map.items():
        if hour in hours_range:
            return slot
    return "morning"


def _next_unassigned_slot(assigned: set[str]) -> str | None:
    all_slots = ["early_morning", "morning", "afternoon", "evening", "night"]
    for slot in all_slots:
        if slot not in assigned:
            return slot
    return None


def _template_schedule(npc: NPC) -> list[ScheduleEntry]:
    """Generate a schedule from occupation templates with per-NPC variation."""
    rng = npc._rng
    base = DEFAULT_SCHEDULES.get(npc.occupation, DEFAULT_SCHEDULE_FALLBACK)
    # Deep copy so we don't mutate the template
    schedule = [
        ScheduleEntry(
            slot=e.slot,
            activity=e.activity,
            location=e.location,
            priority=e.priority,
        )
        for e in base
    ]

    # Slight randomisation: occasionally swap evening activity
    if rng.random() < 0.3:
        for entry in schedule:
            if entry.slot == "evening":
                entry.activity = rng.choice([
                    "walk through the town square",
                    "visit the tavern",
                    "sit by the road and rest",
                ])
                entry.location = rng.choice(["town_square", "tavern", "home"])
                break

    # Per-NPC personal entry: 30% chance to replace afternoon with a personal activity
    if rng.random() < 0.3:
        personal_activities = [
            ("visit a friend across town", "home"),
            ("take a stroll along the road", "outskirts"),
            ("sit in the town square and think", "town_square"),
            ("browse the market for something interesting", "market_stall"),
            ("wander by the fields and enjoy the air", "outskirts"),
        ]
        activity, location = rng.choice(personal_activities)
        for entry in schedule:
            if entry.slot == "afternoon":
                entry.activity = activity
                entry.location = location
                break

    return schedule


def resolve_schedule_location(
    entry: ScheduleEntry,
    npc: NPC,
    buildings: list[PlacedBuilding],
) -> tuple[int, int]:
    """
    Convert a schedule location name to grid coordinates.

    Resolves "home", "work", "tavern", building types, and "town_square"
    to the door tile of the relevant building. Always returns coordinates
    that should be passable (door tiles, not building interiors).
    """
    target = entry.location.lower()

    if target == "home":
        return (npc.home_x, npc.home_z)

    if target == "work":
        return (npc.work_x, npc.work_z)

    if target == "town_square":
        # Find the town hall or church door — the civic heart of town
        for b in buildings:
            if b.building_type in ("town_hall", "church"):
                return (b.door_x, b.door_z)
        # Fallback: use the first building's door as a gathering point
        if buildings:
            return (buildings[0].door_x, buildings[0].door_z)
        return (npc.home_x, npc.home_z)

    if target == "outskirts":
        # Use a farm door if available, otherwise wander near home
        for b in buildings:
            if b.building_type == "farm":
                return (b.door_x, b.door_z)
        return (npc.home_x, npc.home_z)

    # Try matching building type
    for b in buildings:
        if b.building_type == target or target in b.building_type:
            return (b.door_x, b.door_z)

    # Try matching building name
    for b in buildings:
        if target in b.name.lower():
            return (b.door_x, b.door_z)

    # Fallback: go home (always a valid door tile)
    return (npc.home_x, npc.home_z)


async def decide_reaction(
    npc: NPC,
    observation: str,
    llm: LLMProvider,
) -> str:
    """
    Decide how an NPC reacts to a new observation.

    Returns one of: "continue_current", "approach", "avoid", "observe".
    """
    from core.npc.llm_client import format_prompt
    from core.npc.cognition.tiers import get_tier_config

    config = get_tier_config(npc.cognition_tier)

    if not config.uses_llm:
        # Tier 3+: simple heuristic
        return "continue_current"

    try:
        prompt = format_prompt(
            "reaction",
            name=npc.name,
            occupation=npc.occupation,
            current_activity=npc.current_action_description or "idle",
            observation=observation,
        )

        response = await llm.complete(
            system="You are deciding how a medieval NPC reacts to an observation.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.5,
            purpose="reaction",
        )

        reaction = response.strip().lower()
        valid = {"continue_current", "approach", "avoid", "observe"}
        return reaction if reaction in valid else "continue_current"

    except Exception:
        return "continue_current"
