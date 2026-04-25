"""
Self-concept → long-term-goal mapper.

Closes the loop between identity beliefs (`NPC.self_concept`) and
behavioural drive (`NPC.long_term_goals`). When an NPC comes to
believe they are a king, an enemy of Bran, or someone who helped the
town, a matching long-term goal is proposed and an action-affinity
profile is attached so the deterministic planner's utility scorer
biases toward goal-relevant activities.

Design notes:
- Deterministic, rule-based. The overseer layer may later override or
  extend this, but the baseline loop must run without any LLM call.
- Each definition names both concrete action_ids and general action
  tags. Action-id matches give a larger utility bonus than tag matches
  so planners express the goal via the most on-theme action first.
- Derived goals are tracked in `NPC.goal_affinities`. When the source
  self-concept belief decays below `min_confidence`, the derived goal
  is removed on the next sync — identity and drive stay coupled.
- Hand-authored goals in `NPC.long_term_goals` that have no matching
  affinity entry are left untouched; occupational goals spawned at
  NPC creation keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC


# ---------- Definitions ----------

@dataclass(frozen=True)
class GoalDefinition:
    """Maps a self-concept key (or prefix) to a proposed goal.

    Either the full key ("role:king") or just the prefix ("enemy_of")
    may be used. Prefix matches fall through when no exact key match
    is present. The template is formatted with a `target` variable
    derived from the self-concept key after the colon.
    """
    template: str
    boost_actions: tuple[str, ...] = ()
    boost_tags: tuple[str, ...] = ()
    min_confidence: float = 0.4


# Ordering note: exact keys win over prefixes. Keep both views so the
# caller can opt into either. Prefix-based entries let brand-new claims
# (e.g. role:chancellor) pick up a sensible generic goal without any
# further configuration.
GOAL_DEFINITIONS: dict[str, GoalDefinition] = {
    # --- Exact role keys ---
    "role:king": GoalDefinition(
        template="Establish a royal court",
        boost_actions=("construct", "socialise", "patrol", "trade"),
        boost_tags=("community", "social"),
        min_confidence=0.5,
    ),
    "role:knight": GoalDefinition(
        template="Protect the realm from threats",
        boost_actions=("patrol", "construct"),
        boost_tags=("combat", "duty", "community"),
        min_confidence=0.5,
    ),
    "role:priest": GoalDefinition(
        template="Guide the faithful and expand the church",
        boost_actions=("pray", "socialise", "construct"),
        boost_tags=("social", "community"),
        min_confidence=0.5,
    ),
    "role:merchant": GoalDefinition(
        template="Build a trading empire",
        boost_actions=("trade", "craft"),
        boost_tags=("economy", "social"),
        min_confidence=0.5,
    ),
    "role:farmer": GoalDefinition(
        template="Cultivate a bountiful harvest",
        boost_actions=("work", "gather"),
        boost_tags=("economy", "outdoor"),
        min_confidence=0.5,
    ),
    "role:guard": GoalDefinition(
        template="Keep the town safe at all hours",
        boost_actions=("patrol",),
        boost_tags=("duty", "combat"),
        min_confidence=0.5,
    ),

    # --- Prefix fallbacks (applied when no exact-key definition exists) ---
    "role": GoalDefinition(
        template="Live into the role of {target}",
        boost_actions=("work", "socialise"),
        boost_tags=("social",),
        min_confidence=0.5,
    ),
    "enemy_of": GoalDefinition(
        template="Undermine {target}'s standing in the town",
        boost_actions=("socialise", "patrol"),
        boost_tags=("social", "combat"),
        min_confidence=0.5,
    ),
    "rival_of": GoalDefinition(
        template="Outshine {target} at their own craft",
        boost_actions=("craft", "work", "trade"),
        boost_tags=("economy",),
        min_confidence=0.5,
    ),
    "friend_of": GoalDefinition(
        template="Strengthen the bond with {target}",
        boost_actions=("socialise", "trade"),
        boost_tags=("social",),
        min_confidence=0.5,
    ),
    "helped": GoalDefinition(
        template="Defend {target} from future threats",
        boost_actions=("patrol", "construct"),
        boost_tags=("community", "duty"),
        min_confidence=0.5,
    ),
    "saved": GoalDefinition(
        template="Watch over {target} and their people",
        boost_actions=("patrol", "socialise"),
        boost_tags=("community", "duty"),
        min_confidence=0.5,
    ),
    "betrayed": GoalDefinition(
        template="Seek justice against {target}",
        boost_actions=("patrol", "socialise"),
        boost_tags=("combat", "social"),
        min_confidence=0.5,
    ),
}


# ---------- Proposal output ----------

@dataclass
class GoalAffinity:
    """Attached to a derived goal so the planner can bias toward it."""
    source_key: str
    boost_actions: set[str] = field(default_factory=set)
    boost_tags: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_key": self.source_key,
            "boost_actions": sorted(self.boost_actions),
            "boost_tags": sorted(self.boost_tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GoalAffinity":
        return cls(
            source_key=str(data.get("source_key", "")),
            boost_actions=set(data.get("boost_actions", []) or []),
            boost_tags=set(data.get("boost_tags", []) or []),
        )


@dataclass
class GoalProposal:
    """A single goal proposed from a self-concept belief."""
    text: str
    source_key: str
    boost_actions: set[str]
    boost_tags: set[str]

    def to_affinity(self) -> GoalAffinity:
        return GoalAffinity(
            source_key=self.source_key,
            boost_actions=set(self.boost_actions),
            boost_tags=set(self.boost_tags),
        )


# ---------- Pure-function API ----------

def propose_goals(self_concept: dict[str, float]) -> list[GoalProposal]:
    """Return one proposal per self-concept belief that clears its floor.

    Exact keys win over prefixes. Beliefs below the matching
    definition's `min_confidence` are skipped. Target placeholders in
    the template use the colon-suffix of the key with underscores
    replaced by spaces so `bran_1` → `bran 1`.
    """
    proposals: list[GoalProposal] = []
    for key, confidence in self_concept.items():
        definition = GOAL_DEFINITIONS.get(key)
        prefix, _, target = key.partition(":")
        if definition is None:
            definition = GOAL_DEFINITIONS.get(prefix)
        if definition is None:
            continue
        if confidence < definition.min_confidence:
            continue
        target_display = (target or prefix or key).replace("_", " ")
        text = definition.template.format(target=target_display)
        proposals.append(GoalProposal(
            text=text,
            source_key=key,
            boost_actions=set(definition.boost_actions),
            boost_tags=set(definition.boost_tags),
        ))
    return proposals


def sync_npc_goals(npc: "NPC") -> tuple[list[str], list[str]]:
    """Reconcile an NPC's long-term goals with their current self-concept.

    Returns `(added, removed)` — lists of goal texts that changed on
    this sync. Hand-authored goals (no entry in `goal_affinities`)
    are preserved. Derived goals whose source belief has fallen below
    the confidence floor are removed from both structures.
    """
    current_affinities: dict[str, GoalAffinity] = getattr(
        npc, "goal_affinities", {},
    ) or {}

    proposals = propose_goals(npc.self_concept)
    proposed_by_text = {p.text: p for p in proposals}

    added: list[str] = []
    removed: list[str] = []

    # Drop derived goals whose source belief no longer qualifies.
    for goal_text, affinity in list(current_affinities.items()):
        if goal_text not in proposed_by_text:
            current_affinities.pop(goal_text, None)
            if goal_text in npc.long_term_goals:
                npc.long_term_goals.remove(goal_text)
            removed.append(goal_text)

    # Add new derived goals; refresh existing ones so affinity tables
    # pick up any mapper edits between sessions.
    for goal_text, proposal in proposed_by_text.items():
        existed = goal_text in current_affinities
        current_affinities[goal_text] = proposal.to_affinity()
        if goal_text not in npc.long_term_goals:
            npc.long_term_goals.append(goal_text)
        if not existed:
            added.append(goal_text)

    npc.goal_affinities = current_affinities
    return added, removed


# ---------- Scorer helpers ----------

def aggregate_boost_actions(npc: "NPC") -> set[str]:
    """Union of boost_actions across all active derived goals."""
    affinities: dict[str, GoalAffinity] = getattr(
        npc, "goal_affinities", {},
    ) or {}
    result: set[str] = set()
    for affinity in affinities.values():
        result |= affinity.boost_actions
    return result


def aggregate_boost_tags(npc: "NPC") -> set[str]:
    """Union of boost_tags across all active derived goals."""
    affinities: dict[str, GoalAffinity] = getattr(
        npc, "goal_affinities", {},
    ) or {}
    result: set[str] = set()
    for affinity in affinities.values():
        result |= affinity.boost_tags
    return result


# ---------- Bonus magnitudes ----------

# Applied when the scored action's id is in the NPC's aggregated
# boost_actions set. A whole-number bonus puts the goal-aligned action
# clearly above neighbours with comparable base utility but without
# completely overriding urgent survival needs (hunger at 0.9 still
# beats a +1.5 goal nudge).
GOAL_ACTION_BONUS: float = 1.5

# Applied when the action's tags intersect the aggregated boost_tags
# set. Smaller than the action-id bonus so on-theme actions lead.
GOAL_TAG_BONUS: float = 0.6
