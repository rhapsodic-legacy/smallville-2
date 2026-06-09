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

# Stanford-style schedules: each entry has a duration in game-minutes.
# Entries start from 06:00 (dawn) and sum to 1440 (full day cycle).
# The NPC cycles through by duration, not by clock slot boundaries.
#
# Day structure: 06:00 wake → ... → 21:00 sleep (480 min sleep through to 06:00)
#                breakfast(60) + work_am(240) + lunch(60) + work_pm(180)
#                + evening(120) + sleep(480) + buffer(300) = 1440

def _sched(*args: tuple[str, str, str, int, float]) -> list[ScheduleEntry]:
    """Helper: build ScheduleEntry list from (slot, activity, location, priority, duration)."""
    return [
        ScheduleEntry(s, a, loc, p, duration_minutes=d)
        for s, a, loc, p, d in args
    ]

# Each schedule must sum to exactly 1440 minutes (06:00 to 06:00 next day).
# Typical day: breakfast(60) + travel(30) + work_am(270) + lunch(60)
#              + work_pm(240) + evening(240) + sleep(540) = 1440
DEFAULT_SCHEDULES: dict[str, list[ScheduleEntry]] = {
    "blacksmith": _sched(
        ("early_morning", "eat breakfast at home", "home", 3, 60),
        ("morning", "walk to the forge", "work", 5, 30),
        ("morning", "work at the forge", "work", 7, 270),
        ("afternoon", "eat lunch at the tavern", "tavern", 4, 60),
        ("afternoon", "work at the forge", "work", 7, 240),
        ("evening", "eat and socialise at the tavern", "tavern", 4, 240),
        ("night", "walk home and sleep", "home", 9, 540),
    ),
    "farmer": _sched(
        ("early_morning", "eat breakfast at home", "home", 3, 60),
        ("early_morning", "walk to the fields", "work", 5, 30),
        ("morning", "tend the crops", "work", 7, 270),
        ("afternoon", "eat lunch in the field", "work", 4, 60),
        ("afternoon", "sell produce at the market", "market_stall", 5, 240),
        ("evening", "eat at the tavern", "tavern", 4, 240),
        ("night", "walk home and sleep", "home", 9, 540),
    ),
    "merchant": _sched(
        ("early_morning", "eat breakfast at home", "home", 3, 60),
        ("morning", "open the market stall", "work", 7, 30),
        ("morning", "trade and negotiate", "work", 7, 270),
        ("afternoon", "eat lunch at home", "home", 4, 60),
        ("afternoon", "trade and negotiate", "work", 7, 240),
        ("evening", "socialise at the tavern", "tavern", 4, 240),
        ("night", "walk home and sleep", "home", 9, 540),
    ),
    "tavern_keeper": _sched(
        ("early_morning", "eat breakfast at home", "home", 3, 60),
        ("morning", "prepare the tavern", "work", 6, 30),
        ("morning", "serve customers", "work", 7, 270),
        ("afternoon", "eat lunch at the tavern", "work", 4, 60),
        ("afternoon", "serve customers", "work", 7, 240),
        ("evening", "serve the evening crowd", "work", 8, 240),
        ("night", "close up and walk home to sleep", "home", 9, 540),
    ),
    "priest": _sched(
        ("early_morning", "morning prayers at the church", "work", 8, 60),
        ("morning", "counsel townsfolk at the church", "work", 6, 300),
        ("afternoon", "eat lunch at the tavern", "tavern", 4, 60),
        ("afternoon", "walk through town", "town_square", 3, 240),
        ("evening", "evening service at the church", "work", 6, 240),
        ("night", "walk home and sleep", "home", 9, 540),
    ),
    "guard": _sched(
        ("early_morning", "eat breakfast at home", "home", 3, 60),
        ("early_morning", "patrol the perimeter", "outskirts", 7, 30),
        ("morning", "stand watch at the gate", "town_square", 7, 270),
        ("afternoon", "eat lunch at the tavern", "tavern", 4, 60),
        ("afternoon", "patrol the market", "market_stall", 6, 240),
        ("evening", "eat at the tavern", "tavern", 4, 240),
        ("night", "night watch at the town square", "town_square", 8, 540),
    ),
}

# Fallback for unknown occupations
DEFAULT_SCHEDULE_FALLBACK = _sched(
    ("early_morning", "eat breakfast at home", "home", 3, 60),
    ("morning", "wander around town", "town_square", 2, 300),
    ("afternoon", "eat lunch at the tavern", "tavern", 4, 60),
    ("afternoon", "do odd jobs", "market_stall", 4, 240),
    ("evening", "visit the tavern", "tavern", 4, 240),
    ("night", "walk home and sleep", "home", 9, 540),
)


async def generate_daily_schedule(
    npc: NPC,
    llm: LLMProvider,
    current_day: int,
    relationship_summary: str = "",
    town_agenda_summary: str = "",
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
        schedule = await _llm_schedule(
            npc, llm, current_day, relationship_summary,
            town_agenda_summary=town_agenda_summary,
        )
    else:
        schedule = _template_schedule(npc)

    npc.daily_schedule = schedule
    npc.schedule_day = current_day
    # Fresh schedule → fresh cursor. Without these resets, a new day's
    # schedule picks up mid-way through the old one's index, so the
    # NPC skips breakfast/work and never reaches their night "walk
    # home and sleep" entry — producing NPCs visibly wandering at 2 AM.
    npc.schedule_index = 0
    npc.action_start_minutes = 0.0

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
    town_agenda_summary: str = "",
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
            self_concept=npc.self_concept_summary(),
            goals="; ".join(npc.long_term_goals[:3]),
            town_agenda=town_agenda_summary,
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

        # Pass the occupation template as the parser's fallback so that
        # a low-quality Gemma response (e.g. truncated or prose-only)
        # still produces a coherent occupation-appropriate schedule
        # instead of the generic labourer default.
        return _parse_llm_schedule(response, fallback=_template_schedule(npc))

    except Exception as e:
        logger.warning(
            "LLM schedule failed for %s: %s — using template",
            npc.name, e,
        )
        return _template_schedule(npc)


_SENTINEL = object()


def _parse_llm_schedule(
    response: str,
    fallback: list[ScheduleEntry] | None | object = _SENTINEL,
) -> list[ScheduleEntry]:
    """
    Parse an LLM schedule response into a ScheduleEntry list.

    Only lines with an extractable time (e.g. "7:00 AM", "14:00") are
    accepted as activities. Lines without a time — titles like
    "Day 33: A Schedule for Seren", section headers like
    "Focus: ...", markdown table headers like
    "| Time Range | Activity | ... |", and rationale commentary —
    are silently skipped.

    Fallback semantics:
    - omitted:    use DEFAULT_SCHEDULE_FALLBACK when too few entries
    - a template: use that template when too few entries
    - None:       NO fallback — return the (possibly empty) raw parse
                  as-is. Used by replan to avoid appending a full-day
                  template to an already in-progress schedule.
    """
    slot_map = {
        range(5, 8): "early_morning",
        range(8, 12): "morning",
        range(12, 17): "afternoon",
        range(17, 21): "evening",
        range(21, 24): "night",
        range(0, 5): "night",
    }
    slot_defaults = {
        "early_morning": 60, "morning": 300, "afternoon": 300,
        "evening": 240, "night": 540,
    }

    # Faithful parse (Phase 3.5 rewrite): keep EVERY timed line as its
    # own entry — a real day has several morning/afternoon activities —
    # and derive each duration from its explicit time RANGE. The old
    # parser collapsed to one entry per coarse slot (dropping e.g. the
    # 07:00 work block because 06:00 breakfast already filled
    # early_morning) and discarded the LLM's times in favour of slot
    # defaults, producing truncated, mistimed days where NPCs reached
    # the sleep entry mid-morning.
    raw_entries: list[tuple] = []
    for raw_line in response.strip().split("\n"):
        raw = raw_line.strip()
        if not raw:
            continue
        tr = _extract_time_range(raw)
        if tr is None:
            # No time on this line — title, header, separator, prose.
            continue
        start, end = tr
        clean = _clean_activity(_extract_activity_from_line(raw))
        if not clean or clean == "idle":
            continue
        slot = _hour_to_slot(start // 60, slot_map)
        raw_entries.append(
            (start, end, slot, clean, _infer_location(clean, slot))
        )

    entries: list[ScheduleEntry] = []
    for i, (start, end, slot, clean, location) in enumerate(raw_entries):
        # Duration from the explicit range; else gap to the next entry;
        # else a slot default. % 1440 handles wrap (e.g. 20:00–06:00).
        if end is not None:
            dur = (end - start) % 1440
        elif i + 1 < len(raw_entries):
            dur = (raw_entries[i + 1][0] - start) % 1440
        else:
            dur = 0
        if dur <= 0:
            dur = slot_defaults.get(slot, 120)
        entries.append(ScheduleEntry(
            slot=slot, activity=clean, location=location, priority=5,
            duration_minutes=float(min(dur, 1440)),
        ))

    # Sanity gate: too little to form a coherent day -> occupation
    # template. Callers passing fallback=None (replan) want the raw
    # parse even if sparse and handle "too few" themselves.
    if len(entries) < 3:
        if fallback is None:
            return entries
        if fallback is _SENTINEL:
            return list(DEFAULT_SCHEDULE_FALLBACK)
        return list(fallback)

    # The day must end at home asleep.
    _append_sleep_entry_if_missing(entries)
    return entries


def _append_sleep_entry_if_missing(entries: list[ScheduleEntry]) -> None:
    """Guarantee the schedule's final entry is sleep-at-home.

    Mutates the list in place. If the last entry already goes home,
    no-op. Otherwise appends a 540-minute sleep entry.
    """
    last = entries[-1] if entries else None
    # Any final entry already at home is the day's wind-down (e.g.
    # "return home and sleep", "supper and home"). Appending another
    # sleep entry would push the day past 1440 minutes — only append
    # when the day does NOT already end at home.
    if last and last.location == "home":
        return
    entries.append(ScheduleEntry(
        "night", "walk home and sleep", "home", 9,
        duration_minutes=540,
    ))


def _extract_activity_from_line(line: str) -> str:
    """Extract the activity portion from a schedule line.

    Handles several formats:
      "7:00 AM - 8:00 AM: Wake up and eat"    → "Wake up and eat"
      "**7:00 AM - 8:00 AM:** Wake up, at home" → "Wake up, at home"
      "| 06:00 | Wake up | Home | notes |"   → "Wake up"
      "1. 6:00 AM Wake up"                   → "6:00 AM Wake up" (time stripped in clean)

    Markdown table rows take the second pipe-separated cell (the
    Activity column by convention). All other formats are returned
    as-is and _clean_activity removes the time prefix.
    """
    stripped = line.strip()
    # Markdown table row — extract activity column (second cell).
    if stripped.startswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) >= 2:
            return cells[1]
        return stripped
    # Strip bullet numbering / prefixes.
    return stripped.lstrip("0123456789.-) ")


def _clean_activity(text: str) -> str:
    """Strip markdown formatting and time prefixes from LLM output."""
    import re
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}', '', text)
    # Remove leading time patterns like "5:00 AM - 6:00 AM:" or "06:00 –"
    text = re.sub(
        r'^[\d:]+\s*(?:AM|PM|am|pm)?\s*[-–—]\s*'
        r'(?:[\d:]+\s*(?:AM|PM|am|pm)?\s*[-–—:]\s*)?',
        '', text,
    )
    text = text.strip(' -–—:')
    return text if text else "idle"


# Keywords that suggest each location type
_LOCATION_KEYWORDS: dict[str, list[str]] = {
    "home": [
        "home", "bed", "sleep", "breakfast", "wake", "rest at home",
        "morning ablution", "wash", "hygiene", "tent",
    ],
    "work": [
        "forge", "stall", "shop", "counter", "work", "craft",
        "anvil", "hammer", "tools",
    ],
    "tavern": [
        "tavern", "inn", "ale", "stew", "drink", "pub", "bar",
    ],
    "town_square": [
        "town square", "square", "town hall", "notice board",
        "fountain", "centre of town", "center of town",
    ],
    "outskirts": [
        "field", "farm", "crop", "harvest", "garden", "patrol",
        "perimeter", "outskirts", "woods", "forest", "river",
    ],
    "church": [
        "church", "pray", "prayer", "chapel", "temple", "service",
        "worship", "sermon",
    ],
    "market_stall": [
        "market", "trade", "sell", "buy", "merchant", "barter",
        "browse", "goods",
    ],
}

# Slots that default to home if no keyword match
_SLOT_DEFAULT_LOCATION: dict[str, str] = {
    "early_morning": "home",
    "morning": "work",
    "afternoon": "work",
    "evening": "tavern",
    "night": "home",
}


def _infer_location(activity: str, slot: str) -> str:
    """Infer a location name from the activity description and time slot."""
    lower = activity.lower()
    # Check keywords in priority order — first match wins
    for location, keywords in _LOCATION_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return location
    # Fall back to sensible defaults per time slot
    return _SLOT_DEFAULT_LOCATION.get(slot, "work")


def _extract_hour(text: str) -> int | None:
    """Try to pull an hour number from text like '6:00 AM' or '14:00'.

    Requires a colon (e.g. 6:00) or AM/PM suffix to distinguish from
    bullet numbers like '1.' or '2)'.
    """
    import re
    # Match "6:00", "06:00 AM", "14:00", "6 AM", "9 PM"
    match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', text)
    if not match:
        # Try bare hour with AM/PM: "6 AM", "9 PM"
        match = re.search(r'(\d{1,2})\s+(AM|PM|am|pm)', text)
        if not match:
            return None
        hour = int(match.group(1))
        ampm = match.group(2)
    else:
        hour = int(match.group(1))
        ampm = match.group(3)
    if ampm and ampm.upper() == "PM" and hour < 12:
        hour += 12
    if ampm and ampm.upper() == "AM" and hour == 12:
        hour = 0
    return hour if 0 <= hour < 24 else None


def _extract_time_range(text: str) -> tuple[int, int | None] | None:
    """Pull a start (and optional end) time from a schedule line as
    minutes-from-midnight: '06:00-07:00', '6:00 AM - 8:00 AM',
    '(20:00-06:00)'. Returns (start, end) — end may be None if only one
    time is present, or a next-day value (the caller wraps via % 1440).
    Returns None when no time is on the line (title/header/prose)."""
    import re
    mins: list[int] = []
    for m in re.finditer(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', text):
        h, mn, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if ap and ap.upper() == "PM" and h < 12:
            h += 12
        if ap and ap.upper() == "AM" and h == 12:
            h = 0
        if 0 <= h < 24 and 0 <= mn < 60:
            mins.append(h * 60 + mn)
    if not mins:
        for m in re.finditer(r'(\d{1,2})\s+(AM|PM|am|pm)', text):
            h, ap = int(m.group(1)), m.group(2)
            if ap.upper() == "PM" and h < 12:
                h += 12
            if ap.upper() == "AM" and h == 12:
                h = 0
            if 0 <= h < 24:
                mins.append(h * 60)
    if not mins:
        return None
    return mins[0], (mins[1] if len(mins) >= 2 else None)


def _hour_to_slot(hour: int, slot_map: dict) -> str:
    for hours_range, slot in slot_map.items():
        if hour in hours_range:
            return slot
    return "morning"


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
            duration_minutes=e.duration_minutes,
        )
        for e in base
    ]

    # Stagger wake/sleep times: offset first entry by -10 to +10 minutes
    # and compensate on the sleep entry. This spreads departures over ~20 min.
    wake_offset = rng.uniform(-10, 10)  # minutes
    if schedule:
        schedule[0] = ScheduleEntry(
            slot=schedule[0].slot,
            activity=schedule[0].activity,
            location=schedule[0].location,
            priority=schedule[0].priority,
            duration_minutes=max(30, schedule[0].duration_minutes + wake_offset),
        )
        # Compensate on the sleep entry so total stays at 1440
        if len(schedule) >= 2:
            last = schedule[-1]
            schedule[-1] = ScheduleEntry(
                slot=last.slot,
                activity=last.activity,
                location=last.location,
                priority=last.priority,
                duration_minutes=max(60, last.duration_minutes - wake_offset),
            )

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
    to the door tile of the relevant building. The door tile is part of
    the building footprint, so NPCs standing there appear inside the
    building in the 3D renderer.
    """
    target = entry.location.lower()

    if target == "home":
        return (npc.home_x, npc.home_z)

    if target == "work":
        return (npc.work_x, npc.work_z)

    if target == "town_square":
        # Find the town hall or church — the civic heart of town.
        for b in buildings:
            if b.building_type in ("town_hall", "church"):
                return (b.door_x, b.door_z)
        # No civic building: pick the building whose door is closest
        # to the geographic centroid of all buildings. This produces a
        # stable, sensible "centre of town" fallback instead of
        # arbitrary buildings[0] — which could be at the map edge and
        # would funnel every NPC to a corner.
        if buildings:
            cx = sum(b.door_x for b in buildings) / len(buildings)
            cz = sum(b.door_z for b in buildings) / len(buildings)
            central = min(
                buildings,
                key=lambda b: (b.door_x - cx) ** 2 + (b.door_z - cz) ** 2,
            )
            return (central.door_x, central.door_z)
        return (npc.home_x, npc.home_z)

    if target == "outskirts":
        # Use a farm if available, otherwise wander near home
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

    # Fallback: go home
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


# ---------- Mid-day replanning ----------

# Replan intervals per tier (game minutes). None = never replan.
REPLAN_INTERVALS: dict[int, float | None] = {
    1: 60.0,     # Tier 1: every 60 game-minutes (~50 seconds real)
    2: 120.0,    # Tier 2: every 120 game-minutes (~100 seconds real)
    3: None,     # Tier 3: never (template schedules)
    4: None,     # Tier 4: frozen
}


def should_replan(npc: NPC, current_minutes: float) -> bool:
    """Check whether an NPC is due for a mid-day replan."""
    if npc.has_custom_schedule:
        return False
    interval = REPLAN_INTERVALS.get(npc.cognition_tier)
    if interval is None:
        return False
    return (current_minutes - npc.last_replan_minutes) >= interval


async def replan_schedule(
    npc: NPC,
    llm: LLMProvider,
    current_minutes: float,
    recent_perceptions: list[str] | None = None,
    recent_reflections: list[str] | None = None,
    relationship_summary: str = "",
) -> bool:
    """
    Mid-day schedule replanning for Tier 1-2 NPCs.

    Sends the LLM the remaining schedule plus recent context, and
    lets it modify the remaining entries. Returns True if the schedule
    was actually changed.
    """
    from core.npc.llm_client import format_prompt

    if not npc.daily_schedule or npc.schedule_index >= len(npc.daily_schedule):
        return False

    # Build remaining schedule text
    remaining = npc.daily_schedule[npc.schedule_index:]
    schedule_text = "\n".join(
        f"- {e.activity} at {e.location} ({e.duration_minutes:.0f} min)"
        for e in remaining
    )

    perceptions_text = "\n".join(
        f"- {p}" for p in (recent_perceptions or [])[-5:]
    ) or "Nothing notable."
    reflections_text = "\n".join(
        f"- {r}" for r in (recent_reflections or [])[-3:]
    ) or "No recent reflections."

    # Calculate current time for the prompt
    day_minutes = current_minutes % 1440
    hours = int(day_minutes // 60)
    mins = int(day_minutes % 60)
    time_str = f"{hours:02d}:{mins:02d}"

    try:
        prompt = format_prompt(
            "replan_schedule",
            name=npc.name,
            age=npc.age,
            occupation=npc.occupation,
            personality=npc.personality.to_description(),
            goals="; ".join(npc.long_term_goals[:3]) or "live a good life",
            time=time_str,
            remaining_schedule=schedule_text,
            recent_perceptions=perceptions_text,
            recent_reflections=reflections_text,
            relationship_summary=relationship_summary or "No notable relationships.",
        )

        response = await llm.complete(
            system="You are a medieval NPC re-evaluating your daily plan.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            purpose="daily_plan",
        )

        response = response.strip()
        if "NO_CHANGE" in response:
            npc.last_replan_minutes = current_minutes
            return False

        # Parse the new remaining schedule. Crucially, we do NOT pass
        # a fallback template here — replan is for tweaking the tail
        # of an existing schedule, and defaulting to a full-day
        # template would APPEND a whole new day's worth of entries to
        # what's already there. Over many replans this bloats the
        # schedule (80+ entries observed after ~40 game days) and
        # NPCs never reach the end to regenerate. Prefer no-op.
        new_entries = _parse_llm_schedule(response, fallback=None)
        if not new_entries or len(new_entries) < 2:
            npc.last_replan_minutes = current_minutes
            return False

        # No-grow re-derivation: replace the remaining tail WITHOUT ever
        # increasing total length. The old `remaining_count + 2` cap
        # leaked ~2 entries per replan (~15 replans/day -> schedules
        # bloated to 20+, NPCs never reached the end to regenerate). A
        # day's schedule must never exceed the length it was generated
        # with — that single invariant kills the bloat structurally.
        # Re-plan the FUTURE, never the PRESENT. Preserve completed
        # entries AND the in-progress current entry (index unchanged);
        # only entries strictly after the cursor are re-derived. The old
        # code replaced `[:idx]` — i.e. the entry the NPC was actively
        # performing — which reset an in-progress town-goal entry every
        # 60 min so it never finished/credited (replan churn). Keeping
        # the current entry lets the NPC complete what it's doing; the
        # per-tick commitment projection re-adds any future goal entry.
        keep = min(npc.schedule_index + 1, len(npc.daily_schedule))
        remaining_count = max(0, len(npc.daily_schedule) - keep)
        if len(new_entries) > remaining_count:
            new_entries = new_entries[:remaining_count]
            # Truncation can drop the sleep-home entry that
            # `_parse_llm_schedule` guarantees last; restore it in place
            # (no growth) so the bedtime invariant still holds.
            last = new_entries[-1] if new_entries else None
            if not (last and last.location == "home" and (
                "sleep" in last.activity.lower()
                or "rest" in last.activity.lower()
            )):
                if new_entries:
                    new_entries[-1] = ScheduleEntry(
                        "night", "walk home and sleep", "home", 9,
                        duration_minutes=540,
                    )

        # Replace only the future tail. Invariant: total length never
        # grows beyond the previous length.
        npc.daily_schedule = npc.daily_schedule[:keep] + new_entries
        npc.last_replan_minutes = current_minutes

        logger.info(
            "%s replanned schedule: %d new future entries after index %d",
            npc.name, len(new_entries), npc.schedule_index,
        )
        return True

    except Exception as e:
        logger.warning("Replan failed for %s: %s", npc.name, e)
        npc.last_replan_minutes = current_minutes
        return False
