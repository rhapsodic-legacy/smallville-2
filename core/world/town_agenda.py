"""
TownAgenda — collective goals the town works toward.

Gives the simulation a *sense of direction* beyond individual NPC
schedules. The overseer proposes goals ("hold harvest festival",
"repair the bridge") each game-day; the scheduler reads the active
goals and injects matching schedule entries into NPCs whose
personality biases fit the goal. When enough NPCs contribute, the
goal completes and a town event fires — visible on the HUD, felt in
subsequent sentiment shifts.

Data-driven: goal templates live in this module and are loaded by
the overseer. Adding a new goal type is just adding a GoalTemplate
entry — no scheduler or planner changes needed.

Scope note: this is the first iteration. Three goal types for MVP
(festival, repair, council). Expandable via register_template().
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC, PersonalityTraits

logger = logging.getLogger(__name__)


# Sigmoid sharpness for participation_probability. At TEMPERATURE=3 a
# conscientious NPC (score ≈ +0.3) has p≈0.71 of joining; an objector
# with opposes:<goal>=0.9 against weak personality pull (score ≈ -0.7)
# has p≈0.11 — occasionally helps anyway, which is the point.
PARTICIPATION_TEMPERATURE: float = 3.0


class GoalStatus(str, Enum):
    PROPOSED = "proposed"   # Overseer has added it but nobody's contributed yet
    ACTIVE = "active"       # At least one NPC is contributing
    COMPLETED = "completed"
    EXPIRED = "expired"     # Deadline passed without completion


@dataclass
class GoalTemplate:
    """A recipe for a goal the overseer can propose.

    Templates are stateless configuration; each invocation of
    `create_goal()` produces a fresh `TownGoal` instance.
    """
    goal_id: str                    # Short id, e.g. "harvest_festival"
    title: str                      # "Prepare the harvest festival"
    description: str                # Player-facing summary
    activity_text: str              # Schedule activity that participants perform
    location_hint: str              # Logical location — "town_square", "tavern", etc.
    duration_minutes: int = 180     # Schedule entry duration when injected
    required_contributions: int = 3  # How many NPC-contributions complete it
    deadline_days: int = 2          # Days allowed before EXPIRED
    personality_bias: dict[str, float] = field(default_factory=dict)
    # Big-5 field → minimum value for this NPC to prefer this goal.
    # e.g. {"extraversion": 0.55} means extraverts will match.
    identity_key: str = "helped:town"
    # Phase I.4 — self_concept key reinforced on completion for every
    # contributor. `prefix:target` convention; prefix must be present
    # in `NPC.self_concept_summary()`'s phrase_map so the belief
    # renders in prompts. Default is the generic `helped:town` so a
    # template that omits the field still reinforces *something*.


@dataclass
class TownGoal:
    """An active goal on the town agenda.

    Tracks which NPCs have contributed and accumulated progress.
    Sentient status transitions from PROPOSED → ACTIVE (first
    contributor) → COMPLETED or EXPIRED.
    """
    goal_id: str
    title: str
    description: str
    activity_text: str
    location_hint: str
    duration_minutes: int
    required_contributions: int
    deadline_day: int               # Absolute day after which it expires
    personality_bias: dict[str, float]
    created_day: int
    progress: int = 0
    contributors: set[str] = field(default_factory=set)
    status: GoalStatus = GoalStatus.PROPOSED
    # Absolute day the goal completed. -1 until COMPLETED. Phase G.3
    # uses this + the current day to detect "recent shared victory"
    # so contributors get conversational colouring for ~1 game day.
    completed_day: int = -1
    # Copied from the template at instantiation so the completion
    # listener (Phase I.4) reads the reinforcement key off the goal
    # directly without walking back through TEMPLATES. Default matches
    # `GoalTemplate.identity_key` for back-compat with test-constructed
    # TownGoals that skip the field.
    identity_key: str = "helped:town"
    # Foundation rebuild (multi-town seam): which town owns this goal.
    # None = the single default town. Agenda queries filter on this so
    # cross-town goals become an additive layer, not single-town rework.
    town_id: str | None = None

    def participation_score(self, npc: NPC) -> float:
        """Weighted sum of pulls toward contributing to this goal.

        Zero is neutral; positive favours participation, negative opposes.
        Components:
        - Personality alignment: for each bias entry, add
          (npc.trait_value - threshold). A careful NPC matched against a
          `conscientiousness: 0.5` bias contributes +0.3.
        - `supports:<goal_id>` self_concept key — explicit pull toward
          this goal (e.g. "I championed this project").
        - `opposes:<goal_id>` self_concept key — explicit pull against
          (e.g. "I don't want the bridge repaired").

        No hard gates. Final decision falls out of the sampled sigmoid.
        """
        score = 0.0
        for trait, threshold in self.personality_bias.items():
            value = getattr(npc.personality, trait, None)
            if value is None:
                continue
            score += float(value) - float(threshold)
        score += float(npc.self_concept.get(f"supports:{self.goal_id}", 0.0))
        score -= float(npc.self_concept.get(f"opposes:{self.goal_id}", 0.0))
        return score

    def participation_probability(self, npc: NPC) -> float:
        """Sigmoid of `participation_score` — probability in (0, 1)."""
        return 1.0 / (1.0 + math.exp(
            -self.participation_score(npc) * PARTICIPATION_TEMPERATURE
        ))

    def should_participate(self, npc: NPC, rng: random.Random) -> bool:
        """Sampled decision: True with `participation_probability`."""
        return rng.random() < self.participation_probability(npc)

    def record_contribution(self, npc_id: str) -> bool:
        """Add a contribution from this NPC. Returns True if goal just
        transitioned to COMPLETED as a result.

        Idempotent per NPC: calling this twice for the same `npc_id`
        is a no-op on the second call. Without this dedup a single
        NPC calling from multiple code paths (e.g. eager-inject plus
        action-finish) would double-count, and `progress` could
        outrun `len(contributors)` and trigger early completion.
        """
        if self.status in (GoalStatus.COMPLETED, GoalStatus.EXPIRED):
            return False
        if npc_id in self.contributors:
            return False
        self.contributors.add(npc_id)
        self.progress += 1
        if self.status == GoalStatus.PROPOSED:
            self.status = GoalStatus.ACTIVE
        if self.progress >= self.required_contributions:
            self.status = GoalStatus.COMPLETED
            return True
        return False

    def check_expiry(self, current_day: int) -> bool:
        """Mark expired if past deadline. Returns True if state changed."""
        if self.status in (GoalStatus.COMPLETED, GoalStatus.EXPIRED):
            return False
        if current_day > self.deadline_day:
            self.status = GoalStatus.EXPIRED
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "activity_text": self.activity_text,
            "location_hint": self.location_hint,
            "duration_minutes": self.duration_minutes,
            "required_contributions": self.required_contributions,
            "deadline_day": self.deadline_day,
            "progress": self.progress,
            "contributors": sorted(self.contributors),
            "status": self.status.value,
            "created_day": self.created_day,
        }


# ---------- Templates ----------
#
# Starter set. Templates are data: add a new one here (or via
# register_template at runtime) and the overseer + scheduler pick it
# up automatically.

TEMPLATES: dict[str, GoalTemplate] = {
    "harvest_festival": GoalTemplate(
        goal_id="harvest_festival",
        title="Prepare the harvest festival",
        description=(
            "The town is organising a harvest festival. Townsfolk are "
            "needed to set up stalls, bring food, and welcome visitors."
        ),
        activity_text="help set up the harvest festival",
        location_hint="town_square",
        duration_minutes=180,
        required_contributions=3,
        deadline_days=2,
        # Extraverts and agreeable people show up first.
        personality_bias={"extraversion": 0.5},
        identity_key="helped:festival",
    ),
    "repair_bridge": GoalTemplate(
        goal_id="repair_bridge",
        title="Repair the old bridge",
        description=(
            "The bridge has sagged after the spring floods. Strong "
            "hands are needed to shore it up before the next cart gets stuck."
        ),
        activity_text="help repair the bridge",
        location_hint="outskirts",
        duration_minutes=240,
        required_contributions=4,
        deadline_days=3,
        personality_bias={"conscientiousness": 0.5},
        identity_key="built:bridge",
    ),
    "town_council": GoalTemplate(
        goal_id="town_council",
        title="Hold a town council",
        description=(
            "Tensions have been building. The townsfolk gather at the "
            "town hall to talk through grievances."
        ),
        activity_text="attend the town council",
        location_hint="town_square",
        duration_minutes=120,
        required_contributions=5,
        deadline_days=1,
        # Council needs engaged members — not highly neurotic ones
        # since they tend to avoid conflict. Lean agreeable + open.
        personality_bias={"agreeableness": 0.4, "openness": 0.4},
        identity_key="joined:council",
    ),
}


def register_template(template: GoalTemplate) -> None:
    """Add or replace a goal template at runtime (AI Game Studio hook)."""
    TEMPLATES[template.goal_id] = template


def create_goal_from_template(
    template_id: str, current_day: int,
) -> TownGoal | None:
    """Instantiate a goal from a template. Returns None if template unknown."""
    tmpl = TEMPLATES.get(template_id)
    if tmpl is None:
        return None
    return TownGoal(
        goal_id=tmpl.goal_id,
        title=tmpl.title,
        description=tmpl.description,
        activity_text=tmpl.activity_text,
        location_hint=tmpl.location_hint,
        duration_minutes=tmpl.duration_minutes,
        required_contributions=tmpl.required_contributions,
        deadline_day=current_day + tmpl.deadline_days,
        personality_bias=dict(tmpl.personality_bias),
        created_day=current_day,
        identity_key=tmpl.identity_key,
    )


# ---------- Agenda ----------

class TownAgenda:
    """
    The town's live list of collective goals.

    Owned by the NPCManager; mutated by the overseer (adds goals) and
    the planner (reads goals, records contributions). Broadcast to the
    client on every tick.
    """

    def __init__(self) -> None:
        self._goals: dict[str, TownGoal] = {}
        # Track which template ids are on cooldown (recently
        # completed/expired) so we don't spam the same goal.
        self._cooldown_until: dict[str, int] = {}
        # Lifecycle hooks — fired at each goal transition so the
        # NPCManager (or any other listener) can seed memories or
        # ripple side effects.
        self._on_complete: list[Callable[[TownGoal], None]] = []
        self._on_propose: list[Callable[[TownGoal], None]] = []
        self._on_expire: list[Callable[[TownGoal], None]] = []

    def add_completion_listener(self, fn: Callable[[TownGoal], None]) -> None:
        """Register a callback fired when a goal transitions to COMPLETED."""
        self._on_complete.append(fn)

    def add_propose_listener(self, fn: Callable[[TownGoal], None]) -> None:
        """Register a callback fired when a goal is newly PROPOSED."""
        self._on_propose.append(fn)

    def add_expire_listener(self, fn: Callable[[TownGoal], None]) -> None:
        """Register a callback fired when a goal transitions to EXPIRED."""
        self._on_expire.append(fn)

    def active_and_proposed(self) -> list[TownGoal]:
        return [g for g in self._goals.values()
                if g.status in (GoalStatus.PROPOSED, GoalStatus.ACTIVE)]

    def completed(self) -> list[TownGoal]:
        return [g for g in self._goals.values() if g.status == GoalStatus.COMPLETED]

    def get(self, goal_id: str) -> TownGoal | None:
        return self._goals.get(goal_id)

    def propose(self, goal: TownGoal, current_day: int) -> bool:
        """Add a proposed goal. Rejects duplicates and cooldown-blocked ids."""
        if goal.goal_id in self._goals:
            existing = self._goals[goal.goal_id]
            if existing.status in (GoalStatus.PROPOSED, GoalStatus.ACTIVE):
                return False
        cooldown_end = self._cooldown_until.get(goal.goal_id, 0)
        if current_day < cooldown_end:
            return False
        self._goals[goal.goal_id] = goal
        logger.info(
            "Town agenda: proposed '%s' (deadline day %d, needs %d contributions)",
            goal.title, goal.deadline_day, goal.required_contributions,
        )
        for fn in self._on_propose:
            try:
                fn(goal)
            except Exception:
                logger.exception("Goal propose listener failed")
        return True

    def record_contribution(
        self, goal_id: str, npc_id: str,
        current_day: int | None = None,
    ) -> bool:
        """Record a contribution; returns True if the goal completed.

        When the transition to COMPLETED fires, timestamp the goal
        with `current_day` so Phase G.3 can detect recent victories.
        If the caller doesn't provide a day, fall back to the goal's
        deadline_day — still bounded, just coarser.
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        completed = goal.record_contribution(npc_id)
        if completed:
            goal.completed_day = (
                current_day if current_day is not None else goal.deadline_day
            )
            # 3-day cooldown before the same template can be proposed again.
            self._cooldown_until[goal_id] = goal.deadline_day + 3
            for fn in self._on_complete:
                try:
                    fn(goal)
                except Exception:
                    logger.exception("Goal completion listener failed")
        return completed

    def expire_overdue(self, current_day: int) -> list[TownGoal]:
        """Sweep for expired goals. Returns the ones newly expired."""
        newly = []
        for goal in self._goals.values():
            if goal.check_expiry(current_day):
                self._cooldown_until[goal.goal_id] = current_day + 2
                newly.append(goal)
        for goal in newly:
            for fn in self._on_expire:
                try:
                    fn(goal)
                except Exception:
                    logger.exception("Goal expire listener failed")
        return newly

    def matching_goal_for(
        self, npc: NPC, rng: random.Random,
    ) -> TownGoal | None:
        """Pick an active/proposed goal this NPC will join today.

        Every eligible goal is sampled via `should_participate` — an NPC
        with strong personality pull or `supports:<goal>` is likely to
        pass; one with `opposes:<goal>` is likely to decline. Ties among
        survivors resolve active > proposed, nearer deadline first.
        Returns None if no goal's sample roll succeeds.
        """
        candidates = [g for g in self._goals.values()
                      if g.status in (GoalStatus.ACTIVE, GoalStatus.PROPOSED)
                      and npc.npc_id not in g.contributors
                      and g.should_participate(npc, rng)]
        if not candidates:
            return None
        candidates.sort(key=lambda g: (
            g.status != GoalStatus.ACTIVE,  # active before proposed
            g.deadline_day,                 # nearer deadline first
        ))
        return candidates[0]

    # Window (in game days) during which a completed goal still
    # colours conversations between its contributors. 1 day keeps the
    # "we just did it" chatter timely without lingering indefinitely.
    RECENT_VICTORY_DAYS: int = 1

    def shared_matters_for_prompt(
        self,
        npc_id: str,
        partner_id: str,
        current_day: int = 0,
    ) -> str:
        """Partner-aware agenda cue for Phase G.

        Surfaces three conversational shapes:
        - Both NPCs are currently committed to the same active goal
          ("You and X are both helping repair the old bridge").
        - Partner is committed; this NPC isn't — an invitation shape.
        - They share a completed goal within RECENT_VICTORY_DAYS —
          the post-completion chatter window.

        Returns the empty string when nothing applies so the prompt
        stays clean.
        """
        if not (npc_id and partner_id):
            return ""

        parts: list[str] = []

        # Active-goal co-contributors + partner-only mentions.
        for goal in self.active_and_proposed():
            in_self = npc_id in goal.contributors
            in_partner = partner_id in goal.contributors
            if in_self and in_partner:
                parts.append(
                    f"you and your partner are both helping to "
                    f"{goal.activity_text}"
                )
            elif in_partner and not in_self:
                parts.append(
                    f"your partner is helping to {goal.activity_text}"
                )

        # Recent shared victories.
        for goal in self._goals.values():
            if goal.status != GoalStatus.COMPLETED:
                continue
            if goal.completed_day < 0:
                continue
            if current_day - goal.completed_day > self.RECENT_VICTORY_DAYS:
                continue
            if npc_id in goal.contributors and partner_id in goal.contributors:
                parts.append(
                    f"you and your partner recently completed "
                    f"\"{goal.title}\" together"
                )

        if not parts:
            return ""
        return "Shared town matters: " + "; ".join(parts) + "."

    def summary_for_prompt(
        self, npc_id: str = "",
        self_concept: dict[str, float] | None = None,
    ) -> str:
        """One-line cue for LLM prompts describing active town matters.

        Returns the empty string when nothing is on the docket so the
        prompt stays clean. When a goal is on the list, highlights
        whether this NPC has already committed to it — that cue lets
        the conversation prompt drive commitment talk naturally.

        When ``self_concept`` is supplied, the NPC's stance toward a goal
        (an ``opposes:<goal_id>`` / ``supports:<goal_id>`` belief) is
        rendered alongside the title, so the salient town topic carries
        the NPC's own position rather than appearing stance-neutral. This
        couples the belief to the thing being discussed; without it the
        opposition is an isolated self-concept line the model tends to
        ignore in favour of the prompt's pro-engagement cues.
        """
        active = self.active_and_proposed()
        if not active:
            return ""
        parts: list[str] = []
        for goal in active[:3]:
            stance = ""
            if self_concept:
                if self_concept.get(f"opposes:{goal.goal_id}", 0.0) > 0:
                    stance = " (you oppose this)"
                elif self_concept.get(f"supports:{goal.goal_id}", 0.0) > 0:
                    stance = " (you support this)"
            if npc_id and npc_id in goal.contributors:
                parts.append(f"{goal.title} (you are helping){stance}")
            else:
                parts.append(f"{goal.title}{stance}")
        return "Town matters on your mind: " + "; ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": [g.to_dict() for g in self.active_and_proposed()],
            "completed_recent": [
                g.to_dict() for g in self.completed()
            ][-3:],
        }
