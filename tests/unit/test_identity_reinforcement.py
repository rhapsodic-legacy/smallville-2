"""
Phase I.4 — goal-completion reinforcement on self_concept.

When a town goal the NPC contributed to completes, the listener
applies `+REINFORCEMENT_DELTA` to the goal's `identity_key` (e.g.
`built:bridge`), writes a `reflection` memory tagged with the
goal_id + `town_agenda`, and returns an `IdentityReinforcementEvent`.

Covers:
- Template keys are copied through `create_goal_from_template`.
- Contributor gets the delta; bystander gets nothing.
- Strong-belief ceiling: `apply_self_concept_delta` clamps at 1.0.
- Reflection memory written with expected category/metadata/tags.
- Malformed / missing `identity_key` returns None without raising.
- `npc=None` returns None without raising.
- End-to-end through `NPCManager._on_goal_completed`.
"""

from __future__ import annotations

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.self_review import (
    REINFORCEMENT_DELTA, IdentityReinforcementEvent,
    apply_identity_reinforcement,
)
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.models import NPC, PersonalityTraits
from core.world.town_agenda import (
    GoalStatus, TownAgenda, TownGoal, create_goal_from_template,
)


# ---------- Helpers ----------


def _mgr() -> MemoryManager:
    mgr = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mgr.initialise()
    return mgr


def _npc(
    name: str = "Seren", self_concept: dict[str, float] | None = None,
) -> NPC:
    return NPC(
        npc_id=name.lower(), name=name, age=30, occupation="baker",
        backstory="", personality=PersonalityTraits(),
        self_concept=dict(self_concept or {}),
        cognition_tier=1,
    )


def _completed_goal(
    template_id: str = "repair_bridge",
    contributors: set[str] | None = None,
    identity_key: str | None = None,
) -> TownGoal:
    goal = create_goal_from_template(template_id, current_day=0)
    assert goal is not None
    if contributors:
        goal.contributors = set(contributors)
    if identity_key is not None:
        goal.identity_key = identity_key
    goal.status = GoalStatus.COMPLETED
    goal.completed_day = 1
    return goal


# ---------- Template field plumbing ----------


class TestTemplateIdentityKey:
    def test_harvest_festival_key(self):
        goal = create_goal_from_template("harvest_festival", current_day=0)
        assert goal.identity_key == "helped:festival"

    def test_repair_bridge_key(self):
        goal = create_goal_from_template("repair_bridge", current_day=0)
        assert goal.identity_key == "built:bridge"

    def test_town_council_key(self):
        goal = create_goal_from_template("town_council", current_day=0)
        assert goal.identity_key == "joined:council"

    def test_default_when_template_omits_field(self):
        """A TownGoal constructed without passing identity_key falls
        back to the dataclass default so older test code still works."""
        goal = TownGoal(
            goal_id="custom", title="Custom", description="",
            activity_text="do a thing", location_hint="town_square",
            duration_minutes=60, required_contributions=1,
            deadline_day=5, personality_bias={}, created_day=0,
        )
        assert goal.identity_key == "helped:town"


# ---------- apply_identity_reinforcement ----------


class TestApplyIdentityReinforcement:
    def test_contributor_receives_delta(self):
        mgr = _mgr()
        npc = _npc()
        goal = _completed_goal(contributors={npc.npc_id})

        event = apply_identity_reinforcement(
            mgr, npc, goal, game_time=1000.0,
        )
        assert event is not None
        assert isinstance(event, IdentityReinforcementEvent)
        assert event.goal_id == goal.goal_id
        assert event.self_concept_key == "built:bridge"
        assert event.delta == pytest.approx(+REINFORCEMENT_DELTA)
        assert npc.self_concept["built:bridge"] == pytest.approx(0.1)
        assert event.new_confidence == pytest.approx(0.1)

    def test_existing_belief_strengthens_not_reset(self):
        mgr = _mgr()
        npc = _npc(self_concept={"built:bridge": 0.5})
        goal = _completed_goal(contributors={npc.npc_id})

        event = apply_identity_reinforcement(
            mgr, npc, goal, game_time=1000.0,
        )
        assert event is not None
        assert npc.self_concept["built:bridge"] == pytest.approx(0.6)

    def test_ceiling_clamps_at_one(self):
        """A near-saturated belief doesn't exceed 1.0."""
        mgr = _mgr()
        npc = _npc(self_concept={"built:bridge": 0.95})
        goal = _completed_goal(contributors={npc.npc_id})

        event = apply_identity_reinforcement(
            mgr, npc, goal, game_time=1000.0,
        )
        assert event is not None
        assert npc.self_concept["built:bridge"] == pytest.approx(1.0)
        assert event.new_confidence == pytest.approx(1.0)

    def test_no_npc_is_noop(self):
        mgr = _mgr()
        goal = _completed_goal()
        assert apply_identity_reinforcement(
            mgr, None, goal, game_time=1000.0,
        ) is None

    def test_missing_identity_key_returns_none(self):
        """Guards against a template registered without identity_key
        (or a TownGoal constructed with identity_key='')."""
        mgr = _mgr()
        npc = _npc()
        goal = _completed_goal(contributors={npc.npc_id}, identity_key="")
        assert apply_identity_reinforcement(
            mgr, npc, goal, game_time=1000.0,
        ) is None
        assert npc.self_concept == {}

    def test_reflection_memory_written_with_expected_shape(self):
        mgr = _mgr()
        npc = _npc()
        goal = _completed_goal(contributors={npc.npc_id})

        event = apply_identity_reinforcement(
            mgr, npc, goal, game_time=1234.0,
        )
        assert event is not None
        assert event.reflection_memory_id

        ref = mgr.episodic.get_by_id(event.reflection_memory_id)
        assert ref is not None
        assert ref.category == "reflection"
        assert ref.npc_id == npc.npc_id
        assert "bridge" in (ref.description or "").lower()
        assert "town_agenda" in ref.tags
        assert goal.goal_id in ref.tags
        assert ref.metadata.get("outcome_kind") == "identity_reinforcement"
        assert ref.metadata.get("source_goal_id") == goal.goal_id
        assert ref.metadata.get("self_concept_key") == "built:bridge"

    def test_joined_council_key_renders(self):
        """Exercise the joined:council → 'someone who joined the council'
        phrase_map entry added for Phase I.4."""
        mgr = _mgr()
        npc = _npc()
        goal = _completed_goal("town_council", contributors={npc.npc_id})

        # Push confidence high enough that self_concept_summary uses
        # the strong-phrase branch.
        npc.apply_self_concept_delta("joined:council", 0.75)
        apply_identity_reinforcement(mgr, npc, goal, game_time=1000.0)

        summary = npc.self_concept_summary()
        assert "joined the council" in summary


# ---------- End-to-end through NPCManager._on_goal_completed ----------


class TestOnGoalCompletedIntegration:
    """The manager path exercises the real listener, not just the
    helper. Keeps us honest that the wiring in manager.py actually
    fires the delta for contributors and skips bystanders."""

    def _sim(self, count: int = 3):
        from core.npc.llm_client import MockProvider
        from core.npc.manager import NPCManager
        from core.world.generator import WorldConfig, generate_world

        config = WorldConfig(population=count, terrain="riverside", seed=1)
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=1,
        )
        mgr.spawn_population(count)
        return mgr

    def test_contributor_gets_delta_bystander_does_not(self):
        mgr = self._sim(count=3)
        contributors = mgr.npcs[:2]
        bystander = mgr.npcs[2]

        goal = create_goal_from_template("repair_bridge", current_day=0)
        goal.contributors = {n.npc_id for n in contributors}
        goal.status = GoalStatus.COMPLETED
        goal.completed_day = 0

        mgr._on_goal_completed(goal)

        for c in contributors:
            assert c.self_concept.get("built:bridge") == pytest.approx(
                REINFORCEMENT_DELTA,
            )
        # Bystander: no bump on the key.
        assert "built:bridge" not in bystander.self_concept
