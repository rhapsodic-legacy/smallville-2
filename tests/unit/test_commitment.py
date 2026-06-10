"""Phase 2 (foundation rebuild) — durable Commitment layer.

Taking on a town goal records a durable Commitment on the NPC that
survives schedule churn; goal completion/expiry prunes it so the list
stays bounded. The schedule injection is unchanged this phase (Phase 3
replaces it), so these tests cover only the commitment lifecycle.
"""

from __future__ import annotations

from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.npc.models import CommitmentStatus
from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.world.generator import WorldConfig, generate_world
from core.world.town_agenda import create_goal_from_template


def _make_manager(seed: int = 55) -> NPCManager:
    config = WorldConfig(population=3, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    memory = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    memory.initialise()
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=MockProvider(), seed=seed,
        memory=memory,
    )
    mgr.spawn_population(3)
    return mgr


class TestEnsureCommitment:
    def test_creates_pending_commitment(self):
        mgr = _make_manager()
        npc = mgr.npcs[0]
        goal = create_goal_from_template("repair_bridge", current_day=1)

        assert npc.commitments == []
        mgr._ensure_commitment(npc, goal, current_day=1)

        assert len(npc.commitments) == 1
        c = npc.commitments[0]
        assert c.goal_id == "repair_bridge"
        assert c.activity == goal.activity_text
        assert c.location == goal.location_hint
        assert c.deadline_day == goal.deadline_day
        assert c.status == CommitmentStatus.PENDING
        assert c.created_day == 1

    def test_idempotent_no_duplicate(self):
        mgr = _make_manager()
        npc = mgr.npcs[0]
        goal = create_goal_from_template("repair_bridge", current_day=1)

        mgr._ensure_commitment(npc, goal, current_day=1)
        mgr._ensure_commitment(npc, goal, current_day=2)
        mgr._ensure_commitment(npc, goal, current_day=3)

        assert len(npc.commitments) == 1  # one live commitment per goal

    def test_distinct_goals_get_distinct_commitments(self):
        mgr = _make_manager()
        npc = mgr.npcs[0]
        bridge = create_goal_from_template("repair_bridge", current_day=1)
        festival = create_goal_from_template("harvest_festival", current_day=1)

        mgr._ensure_commitment(npc, bridge, current_day=1)
        mgr._ensure_commitment(npc, festival, current_day=1)

        assert {c.goal_id for c in npc.commitments} == {
            "repair_bridge", "harvest_festival",
        }


class TestResolveCommitments:
    def test_prune_removes_only_that_goal(self):
        mgr = _make_manager()
        npc = mgr.npcs[0]
        bridge = create_goal_from_template("repair_bridge", current_day=1)
        festival = create_goal_from_template("harvest_festival", current_day=1)
        mgr._ensure_commitment(npc, bridge, current_day=1)
        mgr._ensure_commitment(npc, festival, current_day=1)

        mgr._resolve_commitments(bridge)

        assert [c.goal_id for c in npc.commitments] == ["harvest_festival"]

    def test_expiry_listener_clears_commitments(self):
        mgr = _make_manager()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        for npc in mgr.npcs:
            mgr._ensure_commitment(npc, goal, current_day=1)
        assert all(npc.commitments for npc in mgr.npcs)

        mgr._on_goal_expired(goal)

        assert all(npc.commitments == [] for npc in mgr.npcs)

    def test_completion_listener_clears_commitments(self):
        mgr = _make_manager()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        for npc in mgr.npcs:
            mgr._ensure_commitment(npc, goal, current_day=1)

        mgr._on_goal_completed(goal)

        assert all(npc.commitments == [] for npc in mgr.npcs)


class TestProjection:
    """Phase 3 — goal entries are a re-derived projection of commitments,
    so replanning can't permanently wipe them."""

    def _npc_with_commitment(self, mgr):
        from core.npc.models import ScheduleEntry
        npc = mgr.npcs[0]
        goal = create_goal_from_template("repair_bridge", current_day=1)
        mgr._ensure_commitment(npc, goal, current_day=1)
        npc.daily_schedule = [
            ScheduleEntry(slot="morning", activity="work",
                          location="work", duration_minutes=240),
            ScheduleEntry(slot="afternoon", activity="work",
                          location="work", duration_minutes=240),
            ScheduleEntry(slot="night", activity="walk home and sleep",
                          location="home", duration_minutes=540),
        ]
        npc.schedule_index = 0
        return npc

    def test_projects_goal_entry_into_reachable_slot(self):
        mgr = _make_manager()
        npc = self._npc_with_commitment(mgr)
        mgr._project_commitments(npc)
        assert any(e.goal_id == "repair_bridge" for e in npc.daily_schedule)
        # The sleep-home entry is preserved (never commandeered).
        assert npc.daily_schedule[-1].location == "home"

    def test_goal_entry_survives_replan_wipe(self):
        from core.npc.models import ScheduleEntry
        mgr = _make_manager()
        npc = self._npc_with_commitment(mgr)
        mgr._project_commitments(npc)
        assert any(e.goal_id == "repair_bridge" for e in npc.daily_schedule)

        # Simulate a mid-day replan replacing the remaining tail with
        # fresh entries that carry no goal_id (the old wipe bug).
        npc.daily_schedule = [
            npc.daily_schedule[0],
            ScheduleEntry(slot="afternoon", activity="something else",
                          location="work", duration_minutes=240),
            ScheduleEntry(slot="night", activity="sleep",
                          location="home", duration_minutes=540),
        ]
        assert not any(e.goal_id == "repair_bridge" for e in npc.daily_schedule)

        # Projection re-derives the goal from the durable commitment.
        mgr._project_commitments(npc)
        assert any(e.goal_id == "repair_bridge" for e in npc.daily_schedule), (
            "commitment must re-project after a replan wiped the goal entry"
        )

    def test_projection_is_idempotent(self):
        mgr = _make_manager()
        npc = self._npc_with_commitment(mgr)
        for _ in range(5):
            mgr._project_commitments(npc)
        n_goal = sum(1 for e in npc.daily_schedule
                     if e.goal_id == "repair_bridge")
        assert n_goal == 1, "projection must not add duplicate goal entries"
