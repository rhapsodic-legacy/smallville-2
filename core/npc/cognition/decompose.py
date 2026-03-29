"""
Task decomposition �� Stanford level 2-3.

Breaks schedule entries ("work at the forge", 4 hours) into concrete
sub-tasks ("stoke the furnace", 30 min). NPCs always have a current
sub-task so they're never idle.

Two paths:
  - Template-based: occupation-specific sub-task templates (deterministic)
  - LLM-based: Mistral/Claude generates personalised sub-tasks from
    NPC identity + memory context (future, wired via cognition router)
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from core.npc.models import SubTask, ScheduleEntry

if TYPE_CHECKING:
    from core.npc.models import NPC
    from core.npc.llm_client import LLMProvider

logger = logging.getLogger(__name__)


# ---------- Template decomposition ----------

# Maps (occupation, schedule_keyword) -> list of possible sub-task sequences.
# Each sequence is a list of (description, duration_minutes, activity_state).
# The system picks one sequence and optionally shuffles middle tasks.

_WORK_TEMPLATES: dict[str, list[list[tuple[str, float, str]]]] = {
    "blacksmith": [
        [
            ("Stoke the furnace and lay out tools", 20, "working"),
            ("Hammer out iron ingots on the anvil", 60, "working"),
            ("Shape horseshoes for the merchant's order", 45, "working"),
            ("Quench finished pieces in the water trough", 15, "working"),
            ("Sharpen blades on the whetstone", 30, "working"),
            ("Sweep the workshop floor", 15, "working"),
        ],
        [
            ("Fire up the forge and check coal stores", 15, "working"),
            ("Repair a damaged ploughshare", 40, "working"),
            ("Forge nails and fittings for the builder", 50, "working"),
            ("Polish a commissioned sword", 30, "working"),
            ("Inspect and oil the bellows", 15, "working"),
        ],
    ],
    "farmer": [
        [
            ("Check the irrigation channels", 20, "working"),
            ("Hoe the rows and pull weeds", 45, "working"),
            ("Sow seeds in the prepared beds", 30, "working"),
            ("Haul water from the well", 25, "working"),
            ("Spread compost on the vegetable patch", 20, "working"),
            ("Mend a broken fence post", 15, "working"),
        ],
        [
            ("Feed the chickens and collect eggs", 15, "working"),
            ("Harvest ripe vegetables from the garden", 40, "gathering"),
            ("Bundle wheat sheaves for storage", 35, "working"),
            ("Sharpen the sickle", 10, "working"),
            ("Tend to the beehives", 20, "working"),
        ],
    ],
    "merchant": [
        [
            ("Unlock the stall and arrange wares", 15, "working"),
            ("Count yesterday's takings", 10, "working"),
            ("Negotiate with a supplier over fabric prices", 30, "talking"),
            ("Serve browsing customers", 45, "working"),
            ("Update the ledger with today's sales", 15, "working"),
            ("Restock shelves from the back room", 20, "working"),
        ],
        [
            ("Polish and display gemstones", 20, "working"),
            ("Appraise goods brought in by a traveller", 25, "talking"),
            ("Haggle over the price of rare spices", 35, "talking"),
            ("Wrap purchased items for customers", 20, "working"),
            ("Write a letter to a distant supplier", 25, "working"),
        ],
    ],
    "tavern_keeper": [
        [
            ("Wipe down the bar and tables", 15, "working"),
            ("Tap a new barrel of ale", 10, "working"),
            ("Serve drinks to the morning regulars", 40, "working"),
            ("Prepare a stew for the lunch crowd", 45, "working"),
            ("Chat with a regular about town gossip", 20, "talking"),
            ("Restock mugs and bowls from the kitchen", 15, "working"),
        ],
        [
            ("Sweep the tavern floor", 10, "working"),
            ("Roast a joint of meat over the hearth", 50, "working"),
            ("Pour rounds for a group of travellers", 25, "working"),
            ("Settle a tab dispute between patrons", 15, "talking"),
            ("Wash dishes and mugs in the basin", 20, "working"),
        ],
    ],
    "priest": [
        [
            ("Light candles on the altar", 10, "working"),
            ("Read morning scripture aloud", 25, "working"),
            ("Counsel a worried villager", 30, "talking"),
            ("Tend the church herb garden", 20, "working"),
            ("Transcribe a passage into the chronicle", 30, "working"),
            ("Offer a quiet prayer for the town", 15, "idle"),
        ],
    ],
    "guard": [
        [
            ("Inspect the armoury and sharpen weapons", 20, "working"),
            ("Walk the perimeter and check for intruders", 40, "working"),
            ("Stand watch at the main gate", 50, "working"),
            ("Drill sword forms in the yard", 25, "working"),
            ("Report the watch status to the captain", 10, "talking"),
        ],
    ],
}

_EAT_TEMPLATES: list[list[tuple[str, float, str]]] = [
    [
        ("Sit down and order a meal", 5, "idle"),
        ("Eat a bowl of hearty stew", 20, "eating"),
        ("Drink a mug of ale", 10, "eating"),
        ("Chat with whoever is sitting nearby", 15, "talking"),
    ],
    [
        ("Find a seat by the fire", 5, "idle"),
        ("Have some bread and cheese", 15, "eating"),
        ("Sip a warm cider", 10, "eating"),
    ],
]

_SLEEP_TEMPLATES: list[list[tuple[str, float, str]]] = [
    [
        ("Change into nightclothes", 5, "idle"),
        ("Sleep soundly", 120, "sleeping"),
    ],
]

_SOCIALISE_TEMPLATES: list[list[tuple[str, float, str]]] = [
    [
        ("Look around for familiar faces", 5, "idle"),
        ("Swap stories with a friend over drinks", 30, "talking"),
        ("Listen to a bard play a tune", 15, "idle"),
        ("Discuss the latest town news", 20, "talking"),
    ],
    [
        ("Wave hello to a neighbour", 5, "idle"),
        ("Share opinions on the weather and harvest", 20, "talking"),
        ("Laugh at a joke someone told", 10, "talking"),
        ("Say goodnight and head off", 5, "idle"),
    ],
]

_WANDER_TEMPLATES: list[list[tuple[str, float, str]]] = [
    [
        ("Stroll through the town square", 15, "idle"),
        ("Pause to watch the bustle of the market", 10, "idle"),
        ("Sit on a bench and people-watch", 20, "idle"),
        ("Stretch legs and walk to the outskirts", 15, "idle"),
    ],
]

_HOME_MORNING_TEMPLATES: list[list[tuple[str, float, str]]] = [
    [
        ("Wake up and stretch", 5, "idle"),
        ("Wash face in the basin", 5, "idle"),
        ("Prepare a simple breakfast", 10, "working"),
        ("Eat breakfast", 15, "eating"),
        ("Tidy the room before heading out", 10, "working"),
    ],
]


def decompose_schedule_entry(
    npc: NPC,
    entry: ScheduleEntry,
    rng: random.Random | None = None,
) -> list[SubTask]:
    """
    Decompose a schedule entry into concrete sub-tasks (template path).

    Returns 2-6 sub-tasks that fill the schedule slot's duration.
    The NPC will execute them in sequence, so they're never idle.
    """
    rng = rng or random.Random()
    keyword = entry.activity.lower()

    if _match(keyword, ["sleep"]):
        raw = _pick(rng, _SLEEP_TEMPLATES)
    elif _match(keyword, ["eat", "breakfast", "lunch", "meal", "dinner"]):
        raw = _pick(rng, _EAT_TEMPLATES)
    elif _match(keyword, ["socialise", "socialize", "tavern", "visit"]):
        raw = _pick(rng, _SOCIALISE_TEMPLATES)
    elif _match(keyword, ["wander", "walk", "stroll", "idle"]):
        raw = _pick(rng, _WANDER_TEMPLATES)
    elif _match(keyword, ["work", "forge", "farm", "trade", "serve",
                           "patrol", "pray", "craft", "open", "tend",
                           "sell", "counsel", "watch", "stand"]):
        raw = _pick_work(rng, npc.occupation)
    elif entry.slot == "early_morning" and entry.location == "home":
        raw = _pick(rng, _HOME_MORNING_TEMPLATES)
    else:
        # Generic fallback: make the activity description itself a sub-task
        raw = _generic_from_description(keyword, entry.slot)

    tasks = [
        SubTask(
            description=desc,
            duration_minutes=dur * rng.uniform(0.8, 1.2),  # ±20% jitter
            location=entry.location,
            activity_state=state,
        )
        for desc, dur, state in raw
    ]

    return tasks


async def decompose_schedule_entry_llm(
    npc: NPC,
    entry: ScheduleEntry,
    llm: LLMProvider,
    memory_context: str = "",
    location_objects: list[str] | None = None,
) -> list[SubTask]:
    """
    Decompose a schedule entry using the LLM (Mistral/Claude).

    Produces personalised sub-tasks informed by NPC identity, personality,
    and recent memory. Falls back to template on failure.
    """
    from core.npc.llm_client import format_prompt

    objects_str = ", ".join(location_objects) if location_objects else "none"

    try:
        prompt = format_prompt(
            "task_decompose",
            name=npc.name,
            occupation=npc.occupation,
            personality=npc.personality.to_description(),
            activity=entry.activity,
            slot=entry.slot,
            location=entry.location,
            objects=objects_str,
            memory_context=memory_context or "No recent memories.",
        )

        response = await llm.complete(
            system=(
                "You are decomposing a medieval NPC's scheduled activity into "
                "3-5 concrete sub-tasks. Each line: description | duration_minutes | "
                "activity_state (idle/working/eating/sleeping/talking/gathering). "
                "Be specific and grounded in the NPC's occupation and personality."
            ),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.8,
            purpose="task_decompose",
        )

        tasks = _parse_llm_subtasks(response, entry)
        if tasks:
            return tasks

    except Exception as e:
        logger.debug("LLM decomposition failed for %s: %s", npc.name, e)

    # Fallback to template
    return decompose_schedule_entry(npc, entry)


# ---------- Internal helpers ----------

def _match(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _pick(rng: random.Random, templates: list[list[tuple]]) -> list[tuple]:
    seq = rng.choice(templates)
    # Shuffle middle items for variety (keep first and last)
    if len(seq) > 3:
        middle = list(seq[1:-1])
        rng.shuffle(middle)
        seq = [seq[0]] + middle + [seq[-1]]
    return seq


def _pick_work(rng: random.Random, occupation: str) -> list[tuple]:
    templates = _WORK_TEMPLATES.get(occupation)
    if templates:
        return _pick(rng, templates)
    # Fallback for unknown occupations
    return [
        ("Prepare workspace", 10, "working"),
        ("Work steadily on the main task", 60, "working"),
        ("Take a short break", 10, "idle"),
        ("Continue working", 40, "working"),
        ("Tidy up", 10, "working"),
    ]


def _generic_from_description(keyword: str, slot: str) -> list[tuple]:
    """Make a generic sub-task sequence from a free-text description."""
    base_dur = 30 if slot in ("early_morning", "evening") else 60
    return [
        (f"Begin: {keyword[:50]}", 10, "idle"),
        (keyword[:60].capitalize(), base_dur, "working"),
        ("Finish up and move on", 10, "idle"),
    ]


_STAGGER_DESCRIPTIONS = [
    "pausing briefly",
    "taking a short break",
    "stretching and looking around",
    "catching breath",
    "adjusting clothes",
]


def make_staggered_subtasks(
    npc: NPC,
    entry: ScheduleEntry | None,
    rng: random.Random,
) -> list[SubTask]:
    """Decompose with a random idle pause prepended to prevent synchronisation.

    If entry is None, returns a single idle subtask. Otherwise, decomposes
    the entry and prepends a 2-15 game-minute pause so NPCs who exhaust
    their queue at similar times don't all resume simultaneously.
    """
    if entry is None:
        return [SubTask(
            description="taking a moment to think",
            duration_minutes=rng.uniform(10, 30),
            activity_state="idle",
        )]

    subtasks = decompose_schedule_entry(npc, entry, rng)
    pause = SubTask(
        description=rng.choice(_STAGGER_DESCRIPTIONS),
        duration_minutes=rng.uniform(5, 30),
        activity_state="idle",
    )
    return [pause] + subtasks


def _parse_llm_subtasks(
    response: str,
    entry: ScheduleEntry,
) -> list[SubTask]:
    """Parse LLM response into SubTask list."""
    valid_states = {"idle", "working", "eating", "sleeping", "talking", "gathering"}
    tasks: list[SubTask] = []

    for line in response.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        desc = parts[0]
        try:
            dur = float(parts[1])
        except ValueError:
            dur = 20.0

        state = parts[2].lower() if len(parts) > 2 else "working"
        if state not in valid_states:
            state = "working"

        tasks.append(SubTask(
            description=desc,
            duration_minutes=dur,
            location=entry.location,
            activity_state=state,
        ))

    return tasks
