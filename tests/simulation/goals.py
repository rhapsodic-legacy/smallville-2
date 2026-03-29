"""
Goal assignment system for the diagnostic experiment.

Each NPC gets a unique concrete goal with 3-5 sequential substeps.
Goals are achievable within the simulation's existing systems
(movement, subtasks, schedule entries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC


@dataclass
class GoalStep:
    """A single measurable substep within an NPC goal."""
    description: str
    location: str          # where this step should happen
    required_activity: str  # activity_state that counts as progress
    target_minutes: float   # how many game minutes of activity = completion
    minutes_accumulated: float = 0.0
    completed: bool = False
    started_tick: int | None = None
    completed_tick: int | None = None


@dataclass
class NPCGoal:
    """A concrete goal with sequential substeps."""
    npc_id: str
    description: str
    steps: list[GoalStep] = field(default_factory=list)
    current_step_index: int = 0
    completed: bool = False
    completed_tick: int | None = None

    @property
    def current_step(self) -> GoalStep | None:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress_fraction(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.completed)
        current = self.current_step
        partial = 0.0
        if current and current.target_minutes > 0:
            partial = min(
                current.minutes_accumulated / current.target_minutes, 1.0,
            )
        return (completed + partial) / len(self.steps)

    def tick(
        self,
        npc_activity: str,
        npc_location: str,
        game_minutes: float,
        tick_number: int,
    ) -> str | None:
        """Advance goal progress. Returns event type if something happened."""
        if self.completed:
            return None
        step = self.current_step
        if step is None:
            return None

        # Mark started
        if step.started_tick is None:
            step.started_tick = tick_number

        # Check if NPC is doing the right activity at the right place.
        # Activity matching is lenient: "working" matches working/gathering,
        # "any" matches everything, and being at the right location with
        # any non-idle activity counts as partial credit.
        work_activities = {"working", "gathering", "talking"}
        activity_match = (
            step.required_activity == "any"
            or npc_activity == step.required_activity
            or (step.required_activity in work_activities
                and npc_activity in work_activities)
        )
        location_match = (
            step.location == "any"
            or npc_location == step.location
            or npc_location == "any"  # NPC near no building counts loosely
        )

        if activity_match and location_match:
            step.minutes_accumulated += game_minutes

        if step.minutes_accumulated >= step.target_minutes:
            step.completed = True
            step.completed_tick = tick_number
            self.current_step_index += 1

            if self.current_step_index >= len(self.steps):
                self.completed = True
                self.completed_tick = tick_number
                return "GOAL_COMPLETE"
            return "GOAL_STEP_COMPLETE"

        return None


# ---------- Goal templates by occupation ----------

_GOAL_TEMPLATES: dict[str, tuple[str, list[tuple[str, str, str, float]]]] = {
    "blacksmith": (
        "Forge 3 iron tools",
        [
            ("Prepare the forge and lay out tools", "work", "working", 30),
            ("Hammer out the first tool", "work", "working", 60),
            ("Hammer out the second tool", "work", "working", 60),
            ("Hammer out the third tool", "work", "working", 60),
            ("Clean up and inspect all three tools", "work", "working", 20),
        ],
    ),
    "farmer": (
        "Harvest and sell 5 bushels",
        [
            ("Tend the crops in the morning", "work", "working", 40),
            ("Harvest the ripe field", "work", "gathering", 50),
            ("Carry produce to the market", "market_stall", "any", 20),
            ("Sell goods to the merchant", "market_stall", "working", 30),
            ("Return home with earnings", "home", "any", 10),
        ],
    ),
    "merchant": (
        "Accumulate 50 gold from trading",
        [
            ("Open the stall and arrange wares", "work", "working", 20),
            ("Buy goods from suppliers", "work", "working", 40),
            ("Mark up and display premium goods", "work", "working", 30),
            ("Sell to townsfolk throughout the day", "work", "working", 60),
            ("Count the day's earnings", "work", "working", 15),
        ],
    ),
    "tavern_keeper": (
        "Serve 20 customers",
        [
            ("Open the tavern and prepare", "work", "working", 20),
            ("Prepare food for the day", "work", "working", 40),
            ("Serve the lunch crowd", "work", "working", 50),
            ("Serve the evening crowd", "work", "working", 50),
            ("Close up and clean the tavern", "work", "working", 20),
        ],
    ),
    "priest": (
        "Hold 3 sermons this week",
        [
            ("Prepare sermon notes at the altar", "work", "working", 30),
            ("Deliver the first sermon", "work", "talking", 40),
            ("Counsel attendees afterwards", "work", "talking", 30),
            ("Deliver the second sermon", "work", "talking", 40),
            ("Deliver the third sermon and reflect", "work", "talking", 40),
        ],
    ),
    "guard": (
        "Complete 5 full patrols",
        [
            ("Inspect weapons and armour", "work", "working", 20),
            ("Patrol the north perimeter", "outskirts", "working", 40),
            ("Patrol the east side", "outskirts", "working", 40),
            ("Patrol the south road", "outskirts", "working", 40),
            ("Report patrol status at town hall", "work", "talking", 15),
        ],
    ),
    "labourer": (
        "Chop 10 trees for lumber",
        [
            ("Walk to the forest edge", "outskirts", "any", 15),
            ("Chop the first batch of trees", "outskirts", "working", 50),
            ("Carry logs to the lumber area", "work", "working", 30),
            ("Chop the second batch", "outskirts", "working", 50),
            ("Stack all logs neatly", "work", "working", 20),
        ],
    ),
}

# Alternate goals for duplicate occupations
_ALT_GOALS: dict[str, list[tuple[str, list[tuple[str, str, str, float]]]]] = {
    "farmer": [
        (
            "Build a fence around the farm",
            [
                ("Gather wood from the outskirts", "outskirts", "gathering", 30),
                ("Cut planks at the work area", "work", "working", 40),
                ("Dig post holes around the field", "work", "working", 40),
                ("Erect fence sections", "work", "working", 50),
                ("Complete and inspect the perimeter", "work", "working", 20),
            ],
        ),
        (
            "Deliver food to 3 homes",
            [
                ("Harvest vegetables for delivery", "work", "gathering", 30),
                ("Pack delivery baskets", "work", "working", 20),
                ("Deliver to the first home", "home", "any", 15),
                ("Deliver to the second home", "home", "any", 15),
                ("Deliver to the third home", "home", "any", 15),
            ],
        ),
    ],
    "labourer": [
        (
            "Dig a well for the town",
            [
                ("Choose the best location", "town_square", "any", 15),
                ("Dig the first layer", "town_square", "working", 40),
                ("Dig the second layer", "town_square", "working", 40),
                ("Line the well with stones", "town_square", "working", 30),
                ("Test the water flow", "town_square", "working", 15),
            ],
        ),
    ],
}


def assign_goals(npcs: list[NPC]) -> dict[str, NPCGoal]:
    """Assign a unique goal to each NPC based on their occupation."""
    goals: dict[str, NPCGoal] = {}
    occupation_count: dict[str, int] = {}

    for npc in npcs:
        occ = npc.occupation
        count = occupation_count.get(occ, 0)
        occupation_count[occ] = count + 1

        if count == 0 and occ in _GOAL_TEMPLATES:
            desc, step_defs = _GOAL_TEMPLATES[occ]
        elif occ in _ALT_GOALS and count - 1 < len(_ALT_GOALS[occ]):
            desc, step_defs = _ALT_GOALS[occ][count - 1]
        elif occ in _GOAL_TEMPLATES:
            # Reuse primary with modified description
            desc, step_defs = _GOAL_TEMPLATES[occ]
            desc = f"{desc} (attempt {count + 1})"
        else:
            desc = f"Work diligently as a {occ}"
            step_defs = [
                ("Start the day's work", "work", "working", 30),
                ("Work through the morning", "work", "working", 60),
                ("Take a midday break", "any", "any", 15),
                ("Finish the afternoon shift", "work", "working", 60),
                ("Wrap up for the day", "home", "any", 10),
            ]

        steps = [
            GoalStep(
                description=s[0],
                location=s[1],
                required_activity=s[2],
                target_minutes=s[3],
            )
            for s in step_defs
        ]

        goals[npc.npc_id] = NPCGoal(
            npc_id=npc.npc_id,
            description=desc,
            steps=steps,
        )

    return goals
