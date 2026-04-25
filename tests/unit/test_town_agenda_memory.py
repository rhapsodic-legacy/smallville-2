"""
Phase F — town agenda memory integration.

Covers:
- TownAgenda propose/expire listeners fire at the right time.
- summary_for_prompt renders active goals and flags contributors.
- NPCManager listener seeds per-NPC memories on propose, commit,
  complete (with contributor split), and expire.
- `{town_agenda}` placeholder survives format_prompt missing-key
  tolerance (older callers keep working).
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.episodic import EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.llm_client import MockProvider, format_prompt
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world
from core.world.town_agenda import (
    GoalStatus, TownAgenda, create_goal_from_template,
)


def _make_manager(seed: int = 55) -> NPCManager:
    """Build an NPCManager backed by in-memory memory stores.

    ChromaDB's default Client() is a process-level singleton that
    leaks memories between tests in the same suite. Force the
    fallback in-memory store so each manager has its own isolated
    episodic collection.
    """
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


# ---------- TownAgenda listener hooks ----------

class TestAgendaListeners:
    def test_propose_listener_fires(self):
        agenda = TownAgenda()
        received = []
        agenda.add_propose_listener(lambda g: received.append(g.goal_id))

        goal = create_goal_from_template("repair_bridge", current_day=1)
        agenda.propose(goal, current_day=1)
        assert received == ["repair_bridge"]

    def test_propose_listener_skipped_on_cooldown(self):
        agenda = TownAgenda()
        received = []
        agenda.add_propose_listener(lambda g: received.append(g.goal_id))

        goal = create_goal_from_template("repair_bridge", current_day=1)
        agenda.propose(goal, current_day=1)
        # Propose again immediately — duplicate rejected before hook fires
        dupe = create_goal_from_template("repair_bridge", current_day=1)
        agenda.propose(dupe, current_day=1)
        assert received == ["repair_bridge"]

    def test_expire_listener_fires(self):
        agenda = TownAgenda()
        received = []
        agenda.add_expire_listener(lambda g: received.append(g.goal_id))

        goal = create_goal_from_template("repair_bridge", current_day=1)
        agenda.propose(goal, current_day=1)
        # deadline_days=3 → expired at day 5
        agenda.expire_overdue(current_day=5)
        assert received == ["repair_bridge"]


# ---------- summary_for_prompt ----------

class TestSummaryForPrompt:
    def test_empty_when_no_goals(self):
        agenda = TownAgenda()
        assert agenda.summary_for_prompt("alice") == ""

    def test_lists_active_titles(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        agenda.propose(goal, current_day=1)
        summary = agenda.summary_for_prompt("alice")
        assert "Town matters" in summary
        assert "Repair the old bridge" in summary

    def test_flags_self_contributor(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        agenda.propose(goal, current_day=1)
        agenda.record_contribution("repair_bridge", "alice")
        summary_alice = agenda.summary_for_prompt("alice")
        summary_other = agenda.summary_for_prompt("bran")
        assert "you are helping" in summary_alice
        assert "you are helping" not in summary_other


# ---------- NPCManager memory seeding ----------

class TestManagerMemorySeeding:
    def test_propose_writes_to_every_npc(self):
        mgr = _make_manager()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        mgr.town_agenda.propose(goal, current_day=1)

        # Every NPC should hold a town_agenda memory about it.
        for npc in mgr.npcs:
            agenda_mems = mgr.memory.episodic.get_recent(
                npc.npc_id, limit=20, category="town_agenda",
            )
            assert any(
                "Repair the old bridge" in m.description
                for m in agenda_mems
            ), f"{npc.name} missing agenda memory"

    def test_commit_writes_commitment_memory(self):
        mgr = _make_manager()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        mgr.town_agenda.propose(goal, current_day=1)

        npc = mgr.npcs[0]
        # Force conscientiousness high so the repair_bridge personality
        # bias matches and _inject_goal_entry proceeds.
        npc.personality.conscientiousness = 0.95
        # Ensure the NPC has a daily_schedule so _inject_goal_entry proceeds.
        from core.npc.models import ScheduleEntry
        npc.daily_schedule = [
            ScheduleEntry(slot="afternoon", activity="work", location="work",
                          priority=5, duration_minutes=240),
        ]
        mgr._inject_goal_entry(npc, current_day=1)

        commit_mems = mgr.memory.episodic.get_recent(
            npc.npc_id, limit=20, category="commitment",
        )
        assert commit_mems, "commitment memory not recorded"
        assert "repair the bridge" in commit_mems[0].description.lower()

    def test_completion_splits_contributors_from_bystanders(self):
        mgr = _make_manager()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        # Force a simpler contribution count so we can complete quickly
        goal.required_contributions = 1
        mgr.town_agenda.propose(goal, current_day=1)

        contributor = mgr.npcs[0]
        bystander = mgr.npcs[1]

        # Drain memory events so the completion ones are distinguishable
        mgr.memory.drain_memory_events()

        completed = mgr.town_agenda.record_contribution(
            goal.goal_id, contributor.npc_id,
        )
        assert completed

        contrib_mems = mgr.memory.episodic.get_recent(
            contributor.npc_id, limit=20, category="town_event",
        )
        bystander_mems = mgr.memory.episodic.get_recent(
            bystander.npc_id, limit=20, category="town_event",
        )
        assert contrib_mems and bystander_mems

        # Contributor phrasing is first-person plural; bystander is
        # third-person attribution.
        assert "We completed" in contrib_mems[0].description
        assert "was completed by" in bystander_mems[0].description
        # Contributor memory is higher importance than bystander's.
        assert contrib_mems[0].importance > bystander_mems[0].importance

    def test_expire_writes_failure_memory(self):
        mgr = _make_manager()
        goal = create_goal_from_template("repair_bridge", current_day=1)
        mgr.town_agenda.propose(goal, current_day=1)
        mgr.town_agenda.expire_overdue(current_day=5)

        for npc in mgr.npcs:
            fail_mems = mgr.memory.episodic.get_recent(
                npc.npc_id, limit=20, category="town_failure",
            )
            assert fail_mems, f"{npc.name} missing failure memory"
            assert "not completed in time" in fail_mems[0].description


# ---------- format_prompt missing-key tolerance ----------

class TestPromptMissingKey:
    def test_legacy_caller_still_works(self):
        # A daily_plan call without town_agenda should not throw.
        out = format_prompt(
            "daily_plan",
            name="A", age=30, occupation="farmer",
            backstory="tst", personality="balanced",
            self_concept="", goals="none",
            health="100%", energy="100%", hunger="0%",
            gold=10, day=1, relationship_summary="",
        )
        assert "A, a 30-year-old farmer" in out

    def test_with_town_agenda_slot_filled(self):
        out = format_prompt(
            "daily_plan",
            name="A", age=30, occupation="farmer",
            backstory="tst", personality="balanced",
            self_concept="", goals="none",
            town_agenda="Town matters on your mind: Repair the bridge.",
            health="100%", energy="100%", hunger="0%",
            gold=10, day=1, relationship_summary="",
        )
        assert "Repair the bridge" in out
