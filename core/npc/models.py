"""
NPC data model.

Identity, physical state, goals, occupation, and schedule.
Designed to be serialisable for WebSocket transmission and compatible
with the tiered cognition system.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActivityState(Enum):
    """What the NPC is currently doing — drives animation on the client."""
    IDLE = "idle"
    WALKING = "walking"
    WORKING = "working"
    SLEEPING = "sleeping"
    TALKING = "talking"
    EATING = "eating"
    GATHERING = "gathering"


class Direction(Enum):
    """Facing direction for rendering."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


@dataclass
class PersonalityTraits:
    """Big Five personality dimensions, each 0.0–1.0."""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def to_description(self) -> str:
        """Natural language summary for LLM prompts."""
        parts = []
        for trait, low_label, high_label in [
            ("openness", "conventional", "curious and creative"),
            ("conscientiousness", "spontaneous", "disciplined and organised"),
            ("extraversion", "introverted and reserved", "outgoing and energetic"),
            ("agreeableness", "competitive and blunt", "cooperative and compassionate"),
            ("neuroticism", "emotionally stable", "anxious and sensitive"),
        ]:
            value = getattr(self, trait)
            if value < 0.3:
                parts.append(low_label)
            elif value > 0.7:
                parts.append(high_label)
        return ", ".join(parts) if parts else "balanced temperament"

    def to_dict(self) -> dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


@dataclass
class ScheduleEntry:
    """A single block in an NPC's daily schedule.

    Stanford style: each entry has a duration in game-minutes. The NPC
    cycles through entries by duration, not by clock slot boundaries.
    The 'slot' field is kept for backward compat and logging.
    """
    slot: str          # ScheduleSlot value: early_morning, morning, etc.
    activity: str      # e.g. "work at forge", "eat at tavern", "sleep"
    location: str      # Hierarchical address or building name
    priority: int = 5  # 1 (low) to 10 (high), for interruption decisions
    target_x: int | None = None  # Pre-resolved coordinates (from planner)
    target_z: int | None = None
    duration_minutes: float = 0.0  # How long this action lasts (game minutes)


@dataclass
class SubTask:
    """A concrete micro-activity within a schedule entry.

    Stanford's level-3 decomposition: "what am I doing RIGHT NOW?"
    Duration is in game minutes. target_object optionally specifies
    a specific object to interact with inside a building.
    """
    description: str
    duration_minutes: float       # how long this sub-task takes (game minutes)
    location: str = ""            # same format as ScheduleEntry.location
    target_object: str | None = None  # e.g. "anvil", "bed", "altar"
    activity_state: str = "idle"  # maps to ActivityState value


@dataclass
class NPC:
    """
    Complete NPC state.

    Holds identity, physical state, goals, schedule, and current action.
    The cognition system reads and writes to this model each tick.
    """

    # --- Identity (immutable after creation) ---
    npc_id: str
    name: str
    age: int
    personality: PersonalityTraits
    backstory: str
    occupation: str

    # --- Physical state ---
    x: float = 0.0
    z: float = 0.0
    home_x: int = 0
    home_z: int = 0
    work_x: int = 0
    work_z: int = 0
    # Stanford living_area: hierarchical address like "smallville:residential:home_3"
    living_area: str = ""
    health: float = 1.0
    energy: float = 1.0
    hunger: float = 0.0  # 0 = full, 1 = starving

    # --- Current action ---
    activity: ActivityState = ActivityState.IDLE
    direction: Direction = Direction.SOUTH
    current_path: list[tuple[int, int]] = field(default_factory=list)
    path_index: int = 0
    current_action_description: str = ""

    # --- Goals ---
    long_term_goals: list[str] = field(default_factory=list)
    short_term_goals: list[str] = field(default_factory=list)

    # --- Schedule (Stanford style: duration-based action list) ---
    daily_schedule: list[ScheduleEntry] = field(default_factory=list)
    schedule_day: int = 0  # which game day this schedule was generated for
    schedule_index: int = 0  # current position in daily_schedule
    action_start_minutes: float = 0.0  # game-minutes when current action started

    # --- Resources ---
    gold: int = 0
    inventory: dict[str, int] = field(default_factory=dict)
    skills: dict[str, float] = field(default_factory=dict)

    # --- Cognition state ---
    cognition_tier: int = 3          # 1=full LLM, 2=simplified, 3=state machine, 4=frozen
    last_perception_tick: float = 0  # game minutes when last perceived
    last_plan_tick: float = 0        # game minutes when last planned
    conversation_partner: str | None = None  # npc_id of current conversation partner
    recent_perceptions: list[str] = field(default_factory=list)

    # --- Conversation cooldown ---
    last_conversation_time: float = 0  # game minutes
    conversation_cooldown: float = 60  # game minutes before seeking new conversation

    # --- Visual archetype (for asset provider) ---
    archetype: str = ""  # e.g. "blacksmith_male" — derived from occupation if empty

    # --- Task decomposition (Stanford level 2–3) ---
    subtask_queue: list[SubTask] = field(default_factory=list)
    current_subtask: SubTask | None = None
    subtask_time_remaining: float = 0.0  # game minutes left on current sub-task

    # --- Per-NPC RNG (seeded from global seed + npc_id) ---
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    # --- Movement ---
    move_speed: float = 2.0   # tiles per real second
    _move_progress: float = 0.0  # fractional tile progress between path steps
    _last_path_index: int = 0    # tracks path progress for stuck detection
    _stuck_time: float = 0.0     # seconds since last path progress

    def tick_needs(self, game_minutes_elapsed: float) -> None:
        """Update hunger and energy based on time and activity."""
        hunger_rate = 0.002   # per game minute
        energy_drain = 0.001  # per game minute at rest

        self.hunger = min(1.0, self.hunger + hunger_rate * game_minutes_elapsed)

        if self.activity == ActivityState.SLEEPING:
            self.energy = min(1.0, self.energy + 0.005 * game_minutes_elapsed)
        elif self.activity == ActivityState.WORKING:
            self.energy = max(0.0, self.energy - energy_drain * 2 * game_minutes_elapsed)
        else:
            self.energy = max(0.0, self.energy - energy_drain * game_minutes_elapsed)

        if self.activity == ActivityState.EATING:
            self.hunger = max(0.0, self.hunger - 0.01 * game_minutes_elapsed)

    @property
    def tile_x(self) -> int:
        """Grid tile X (rounded from float position)."""
        return round(self.x)

    @property
    def tile_z(self) -> int:
        """Grid tile Z (rounded from float position)."""
        return round(self.z)

    def is_at(self, x: int, z: int) -> bool:
        return self.tile_x == x and self.tile_z == z

    def distance_to(self, x: int, z: int) -> float:
        """Manhattan distance to a point."""
        return abs(self.x - x) + abs(self.z - z)

    def needs_new_schedule(self, current_day: int) -> bool:
        """Check if this NPC needs a new schedule.

        Duration-based model: only regenerate when the schedule list is
        empty (exhausted). Day-based regeneration is handled by
        _advance_npc_action when the last entry expires.
        """
        return not self.daily_schedule

    def get_current_schedule_entry(self, slot: str) -> ScheduleEntry | None:
        """Get the schedule entry for the given time slot."""
        for entry in self.daily_schedule:
            if entry.slot == slot:
                return entry
        return None

    def summary_for_prompt(self) -> str:
        """Compact NPC summary for inclusion in LLM prompts."""
        personality_desc = self.personality.to_description()
        goals = "; ".join(self.long_term_goals[:2]) if self.long_term_goals else "none"
        return (
            f"{self.name}, age {self.age}, {self.occupation}. "
            f"Personality: {personality_desc}. "
            f"Goals: {goals}. "
            f"Health: {self.health:.0%}, Energy: {self.energy:.0%}, "
            f"Hunger: {self.hunger:.0%}. Gold: {self.gold}."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for WebSocket transmission to client.

        Stanford approach: send only the current tile position and activity.
        The client lerps smoothly toward the position each tick.
        No path data is sent — path exists only on the server.
        """
        return {
            "npc_id": self.npc_id,
            "name": self.name,
            "x": self.x,
            "z": self.z,
            "activity": self.activity.value,
            "direction": self.direction.value,
            "occupation": self.occupation,
            "archetype": self.archetype or self.occupation,
            "health": round(self.health, 2),
            "energy": round(self.energy, 2),
            "hunger": round(self.hunger, 2),
            "action": self.current_action_description,
            "subtask": self.current_subtask.description if self.current_subtask else "",
            "cognition_tier": self.cognition_tier,
            "conversation_partner": self.conversation_partner,
            "move_speed": self.move_speed,
            "trail": list(getattr(self, '_tick_trail', [])),
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Full serialisation including identity and goals (for save/load)."""
        return {
            **self.to_dict(),
            "age": self.age,
            "personality": self.personality.to_dict(),
            "backstory": self.backstory,
            "home_x": self.home_x,
            "home_z": self.home_z,
            "work_x": self.work_x,
            "work_z": self.work_z,
            "long_term_goals": self.long_term_goals,
            "short_term_goals": self.short_term_goals,
            "gold": self.gold,
            "inventory": dict(self.inventory),
            "skills": dict(self.skills),
            "daily_schedule": [
                {"slot": e.slot, "activity": e.activity,
                 "location": e.location, "priority": e.priority}
                for e in self.daily_schedule
            ],
        }


# ---------- NPC Templates ----------

# Occupation defaults: maps occupation -> (work_building_type, default_skills, goal_templates)
OCCUPATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "blacksmith": {
        "work_building": "blacksmith",
        "skills": {"smithing": 0.7, "trading": 0.3, "combat": 0.4},
        "goals": ["Master the craft of weapon-smithing", "Earn enough gold to expand the forge"],
    },
    "farmer": {
        "work_building": "farm",
        "skills": {"farming": 0.7, "cooking": 0.4, "trading": 0.2},
        "goals": ["Grow the finest crops in the region", "Save enough for a bigger plot"],
    },
    "merchant": {
        "work_building": "market_stall",
        "skills": {"trading": 0.8, "diplomacy": 0.5, "appraisal": 0.6},
        "goals": ["Build a trading empire", "Establish trade routes with neighbouring towns"],
    },
    "tavern_keeper": {
        "work_building": "tavern",
        "skills": {"cooking": 0.6, "diplomacy": 0.5, "trading": 0.4},
        "goals": ["Make the tavern the heart of the community", "Collect the best recipes"],
    },
    "priest": {
        "work_building": "church",
        "skills": {"diplomacy": 0.6, "medicine": 0.4, "teaching": 0.5},
        "goals": ["Guide the townsfolk spiritually", "Complete the church construction"],
    },
    "guard": {
        "work_building": "town_hall",
        "skills": {"combat": 0.7, "perception": 0.5, "discipline": 0.6},
        "goals": ["Keep the town safe", "Train new recruits"],
    },
    "labourer": {
        "work_building": None,
        "skills": {"strength": 0.6, "endurance": 0.5, "gathering": 0.4},
        "goals": ["Find steady work", "Save enough to learn a proper trade"],
    },
}

# First names pool — British/fantasy flavour
FIRST_NAMES = [
    "Aldric", "Bran", "Cedric", "Dara", "Edric", "Fiona", "Gwen",
    "Hilda", "Isolde", "Jasper", "Kira", "Leofric", "Mira", "Nessa",
    "Oswin", "Petra", "Quinn", "Rowan", "Seren", "Theron", "Una",
    "Voss", "Wren", "Xander", "Yara", "Zara", "Aelric", "Briar",
    "Calla", "Dorian", "Elara", "Finn", "Gareth", "Helena", "Idris",
]

BACKSTORY_TEMPLATES = [
    "Born and raised in Smallville, {name} has always dreamed of something greater.",
    "{name} arrived in Smallville {age_desc} seeking a fresh start after leaving {origin}.",
    "A {occupation} by trade, {name} learned the craft from a strict but fair mentor.",
    "{name} inherited the family {occupation} business and takes great pride in the work.",
    "Once a wanderer, {name} settled in Smallville when the road grew too lonely.",
    "{name} came to Smallville following rumours of opportunity and rich resources.",
]


# ---------- Seed memories ----------
# Foundational memories that give NPCs their initial "why".
# Each occupation has a set of memories covering: identity, craft knowledge,
# motivation, and social awareness. Formatted with {name} and {occupation}.

SEED_MEMORIES: dict[str, list[tuple[str, str, float]]] = {
    # (description, category, importance)
    "blacksmith": [
        ("I am {name}, the blacksmith of Smallville. The forge is my life's work.", "identity", 0.9),
        ("I learned smithing from old master Aldric. He taught me that the finest blades are forged with patience.", "knowledge", 0.7),
        ("The town relies on me for tools, nails, and horseshoes. Without a good smith, nothing gets built.", "motivation", 0.8),
        ("I want to forge a blade worthy of legend — something that will outlast me.", "aspiration", 0.9),
        ("Iron ore is scarce lately. I should trade with the merchants or find a new source.", "concern", 0.6),
    ],
    "farmer": [
        ("I am {name}, a farmer. The land feeds Smallville and I am its steward.", "identity", 0.9),
        ("My father taught me to read the soil — dark earth means good wheat, pale earth needs rest.", "knowledge", 0.7),
        ("The harvest this year must be better than the last. People are counting on me.", "motivation", 0.8),
        ("I dream of owning the largest farm in the region, with fields stretching to the river.", "aspiration", 0.9),
        ("I should sell my surplus at the market. The merchant always haggles, but fair trade keeps the town alive.", "concern", 0.6),
    ],
    "merchant": [
        ("I am {name}, a merchant. Every trade I make strengthens Smallville's economy.", "identity", 0.9),
        ("The secret to good trading is knowing what people need before they do.", "knowledge", 0.7),
        ("Gold is a means, not an end. Real wealth is in connections and reputation.", "motivation", 0.8),
        ("One day I will establish trade routes beyond Smallville — to the capital and the coastal towns.", "aspiration", 0.9),
        ("I should keep an eye on what the blacksmith and farmers produce. Supply drives opportunity.", "concern", 0.6),
    ],
    "tavern_keeper": [
        ("I am {name}, keeper of the tavern. This place is the heart of Smallville.", "identity", 0.9),
        ("A good tavern keeper listens more than they speak. Everyone shares their troubles over an ale.", "knowledge", 0.7),
        ("I hear every rumour, every grievance, every joy. The tavern connects this town.", "motivation", 0.8),
        ("I want the tavern to be famous — a place travellers speak of for leagues around.", "aspiration", 0.9),
        ("I need a steady supply of food and drink. The farmers and I should be close allies.", "concern", 0.6),
    ],
    "priest": [
        ("I am {name}, the priest of Smallville. I tend to the souls of this community.", "identity", 0.9),
        ("Faith is not found in grand cathedrals, but in the quiet moments of everyday life.", "knowledge", 0.7),
        ("When people quarrel, they come to me. When they mourn, they come to me. I must be steadfast.", "motivation", 0.8),
        ("I hope to build a proper church — one with stained glass and a bell that rings across the valley.", "aspiration", 0.9),
        ("The guard and I see the town differently, but we both want it to thrive.", "concern", 0.6),
    ],
    "guard": [
        ("I am {name}, a guard of Smallville. My duty is to keep these people safe.", "identity", 0.9),
        ("Watch the treeline at dusk. That is when trouble comes, if it comes at all.", "knowledge", 0.7),
        ("The townsfolk sleep soundly because I do not. That is enough.", "motivation", 0.8),
        ("I will train the next generation to defend this town. Strength must be passed on.", "aspiration", 0.9),
        ("The roads beyond Smallville are not safe. I should speak with the merchant about what travellers report.", "concern", 0.6),
    ],
    "labourer": [
        ("I am {name}, a labourer. I do whatever work needs doing in Smallville.", "identity", 0.9),
        ("Hard work is honest work. I may not have a craft yet, but I have strong hands.", "knowledge", 0.7),
        ("The blacksmith needs help at the forge. The farmers need help in the fields. I can be useful.", "motivation", 0.8),
        ("Someday I will learn a proper trade — perhaps smithing or carpentry. I just need the chance.", "aspiration", 0.9),
        ("I should talk to the other workers. Together we could take on bigger jobs.", "concern", 0.6),
    ],
}

# Universal seed memories that all NPCs receive
UNIVERSAL_SEED_MEMORIES: list[tuple[str, str, float]] = [
    ("I live in Smallville. It is a small but growing town with a tavern, a market, homes, and farms.", "knowledge", 0.7),
    ("The people of Smallville depend on each other. The blacksmith makes tools, the farmer grows food, the merchant trades goods.", "knowledge", 0.6),
    ("I know most of the townsfolk by sight. We are a close community.", "social", 0.5),
]
