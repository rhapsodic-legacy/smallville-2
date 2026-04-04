"""Tests for the NPC Manager — spawning, population, state queries."""

import pytest
from core.npc.manager import NPCManager
from core.npc.llm_client import MockProvider
from core.world.generator import WorldConfig, generate_world


@pytest.fixture
def world():
    config = WorldConfig(population=5, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)
    return grid, buildings


@pytest.fixture
def manager(world):
    grid, buildings = world
    return NPCManager(grid=grid, buildings=buildings, llm=MockProvider(), seed=42)


class TestSpawning:
    def test_spawn_correct_count(self, manager):
        npcs = manager.spawn_population(5)
        assert len(npcs) == 5

    def test_npcs_have_unique_ids(self, manager):
        npcs = manager.spawn_population(5)
        ids = {n.npc_id for n in npcs}
        assert len(ids) == 5

    def test_npcs_have_names(self, manager):
        npcs = manager.spawn_population(5)
        for npc in npcs:
            assert len(npc.name) > 0

    def test_npcs_have_occupations(self, manager):
        npcs = manager.spawn_population(5)
        for npc in npcs:
            assert len(npc.occupation) > 0

    def test_npcs_have_valid_positions(self, manager):
        grid = manager.grid
        npcs = manager.spawn_population(5)
        for npc in npcs:
            assert grid.in_bounds(npc.x, npc.z)

    def test_npcs_have_personality(self, manager):
        npcs = manager.spawn_population(3)
        for npc in npcs:
            assert 0.0 <= npc.personality.openness <= 1.0
            assert 0.0 <= npc.personality.extraversion <= 1.0

    def test_npcs_have_starting_gold(self, manager):
        npcs = manager.spawn_population(3)
        for npc in npcs:
            assert npc.gold > 0

    def test_essential_occupations_filled(self, manager):
        npcs = manager.spawn_population(10)
        occupations = {n.occupation for n in npcs}
        # At minimum should have blacksmith and tavern_keeper
        # (since those buildings exist in the generated world)
        assert "blacksmith" in occupations or "tavern_keeper" in occupations


class TestNPCLookup:
    def test_get_npc_by_id(self, manager):
        npcs = manager.spawn_population(3)
        found = manager.get_npc(npcs[0].npc_id)
        assert found is not None
        assert found.name == npcs[0].name

    def test_get_nonexistent_npc(self, manager):
        manager.spawn_population(3)
        assert manager.get_npc("nonexistent") is None

    def test_get_npcs_near(self, manager):
        npcs = manager.spawn_population(5)
        # All NPCs start near their homes, so this should find at least some
        nearby = manager.get_npcs_near(0, 0, radius=30)
        assert len(nearby) >= 1


class TestMovementVariation:
    def test_npcs_have_varied_move_speed(self, manager):
        """NPCs should not all share the same move speed."""
        npcs = manager.spawn_population(10)
        speeds = {round(n.move_speed, 4) for n in npcs}
        assert len(speeds) > 1, "All NPCs have identical speed"

    def test_move_speed_within_range(self, manager):
        npcs = manager.spawn_population(10)
        for npc in npcs:
            assert 1.6 <= npc.move_speed <= 2.4

    @pytest.mark.asyncio
    async def test_action_advance_dispatches_immediately(self, manager):
        """Advancing to next action should dispatch NPCs immediately (Stanford model)."""
        npcs = manager.spawn_population(3)
        from core.npc.models import ScheduleEntry, ActivityState
        for npc in npcs:
            npc.daily_schedule = [
                ScheduleEntry("morning", "eat breakfast", "home", 3, duration_minutes=60),
                ScheduleEntry("morning", "work", "work", 5, duration_minutes=240),
            ]
            npc.schedule_index = 0
            npc.action_start_minutes = 0.0
        # Advance each NPC to the second entry
        for npc in npcs:
            await manager._advance_npc_action(npc, 60.0, "morning")
        # NPCs should be walking or at destination — no stagger queue
        for npc in npcs:
            assert npc.schedule_index == 1
            assert npc.action_start_minutes == 60.0

    @pytest.mark.asyncio
    async def test_action_advance_updates_schedule_index(self, manager):
        """Each advance should increment schedule_index and reset timer."""
        npcs = manager.spawn_population(1)
        from core.npc.models import ScheduleEntry
        npc = npcs[0]
        npc.daily_schedule = [
            ScheduleEntry("morning", "eat breakfast", "home", 3, duration_minutes=60),
            ScheduleEntry("morning", "work", "work", 5, duration_minutes=240),
            ScheduleEntry("afternoon", "eat lunch", "tavern", 4, duration_minutes=60),
        ]
        npc.schedule_index = 0
        npc.action_start_minutes = 0.0
        await manager._advance_npc_action(npc, 60.0, "morning")
        assert npc.schedule_index == 1
        await manager._advance_npc_action(npc, 300.0, "afternoon")
        assert npc.schedule_index == 2


class TestSeedMemories:
    def test_spawn_creates_episodic_memories(self, manager):
        """NPCs should have foundational memories after spawning."""
        npcs = manager.spawn_population(3)
        for npc in npcs:
            memories = manager.memory.episodic.get_recent(npc.npc_id, limit=50)
            # 5 occupation + 3 universal + 1 backstory = 9 minimum
            assert len(memories) >= 9, (
                f"{npc.name} ({npc.occupation}) has {len(memories)} memories"
            )

    def test_seed_memories_contain_identity(self, manager):
        """At least one memory should mention the NPC's name."""
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        memories = manager.memory.episodic.get_recent(npc.npc_id, limit=50)
        descriptions = [m.description for m in memories]
        assert any(npc.name in d for d in descriptions)

    def test_seed_memories_contain_occupation(self, manager):
        """At least one memory should mention the NPC's occupation."""
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        memories = manager.memory.episodic.get_recent(npc.npc_id, limit=50)
        descriptions = [m.description.lower() for m in memories]
        assert any(npc.occupation in d for d in descriptions)

    def test_structured_goals_seeded(self, manager):
        """NPC long-term goals should be stored in structured memory."""
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        goals = manager.memory.structured.get_active_goals(npc.npc_id)
        assert len(goals) >= 1
        goal_descs = {g.description for g in goals}
        # At least one of the NPC's long_term_goals should be stored
        assert any(g in goal_descs for g in npc.long_term_goals)

    def test_identity_fact_seeded(self, manager):
        """NPCs should have a structured 'is_a' fact for their occupation."""
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        facts = manager.memory.structured.get_facts(npc.npc_id, limit=50)
        is_a_facts = [f for f in facts if f.predicate == "is_a"]
        assert len(is_a_facts) >= 1
        assert any(f.obj == npc.occupation for f in is_a_facts)

    def test_universal_memories_present(self, manager):
        """All NPCs should know about Smallville."""
        npcs = manager.spawn_population(2)
        for npc in npcs:
            memories = manager.memory.episodic.get_recent(npc.npc_id, limit=50)
            descriptions = " ".join(m.description for m in memories)
            assert "Smallville" in descriptions


class TestEmergencyOverrides:
    def test_force_navigate_all_moves_npcs(self, manager):
        """force_navigate_all should start all NPCs walking."""
        from core.npc.models import ActivityState
        npcs = manager.spawn_population(5)
        count = manager.force_navigate_all(0, 0, "flee!")
        assert count >= 1
        walking = [n for n in npcs if n.activity == ActivityState.WALKING]
        assert len(walking) >= 1

    def test_force_navigate_all_clears_pending(self, manager):
        """Emergency movement should clear any pending departures."""
        npcs = manager.spawn_population(3)
        # Add pending departures using real NPC IDs
        for npc in npcs:
            manager._pending_departures[npc.npc_id] = (0.0, lambda: None)
        assert len(manager._pending_departures) == len(npcs)

        manager.force_navigate_all(0, 0, "emergency!")
        assert len(manager._pending_departures) == 0

    def test_force_navigate_all_with_filter(self, manager):
        """filter_fn should limit which NPCs are affected."""
        npcs = manager.spawn_population(5)
        # Only move guards
        count = manager.force_navigate_all(
            0, 0, "to your posts!",
            filter_fn=lambda n: n.occupation == "guard",
        )
        guards = [n for n in npcs if n.occupation == "guard"]
        assert count <= len(guards)

    def test_force_navigate_all_flee_from(self, manager):
        """flee_from=True should move NPCs away from the danger point."""
        from core.npc.models import ActivityState
        npcs = manager.spawn_population(3)
        # Place all NPCs at (5, 5)
        for npc in npcs:
            npc.x = 5
            npc.z = 5
        manager.force_navigate_all(5, 5, "run!", flee_from=True)
        # NPCs should be walking away
        for npc in npcs:
            if npc.activity == ActivityState.WALKING and npc.current_path:
                last = npc.current_path[-1]
                # Destination should be away from (5, 5)
                dist = abs(last[0] - 5) + abs(last[1] - 5)
                assert dist > 3

    def test_force_navigate_npc(self, manager):
        """force_navigate_npc should move a specific NPC."""
        from core.npc.models import ActivityState
        npcs = manager.spawn_population(3)
        target = npcs[0]
        result = manager.force_navigate_npc(
            target.npc_id, 10, 10, "go there",
        )
        assert result is True
        assert target.activity == ActivityState.WALKING
        assert target.current_action_description == "go there"


class TestState:
    def test_get_state_returns_npcs(self, manager):
        manager.spawn_population(3)
        state = manager.get_state()
        assert "npcs" in state
        assert len(state["npcs"]) == 3

    def test_state_npcs_have_required_fields(self, manager):
        manager.spawn_population(1)
        state = manager.get_state()
        npc_data = state["npcs"][0]
        required = {"npc_id", "name", "x", "z", "activity", "occupation"}
        assert required.issubset(npc_data.keys())

    def test_set_focus(self, manager):
        manager.set_focus(10, 20)
        assert manager.focus_x == 10
        assert manager.focus_z == 20


# ---------- Integration wiring ----------

class TestCognitionWiring:
    """Verify that the cognition router and planner are wired into the manager."""

    def test_manager_has_router(self, manager):
        from core.npc.cognition.router import CognitionRouter
        assert isinstance(manager.router, CognitionRouter)

    def test_manager_has_planner(self, manager):
        from core.npc.cognition.planner import DeterministicPlanner
        assert isinstance(manager.planner, DeterministicPlanner)

    def test_manager_has_economy(self, manager):
        from core.npc.economy_tick import EconomyTick
        assert isinstance(manager.economy, EconomyTick)

    def test_custom_router_accepted(self, world):
        from core.npc.cognition.router import (
            CognitionRouter, CognitionPolicy, policy_all_deterministic,
        )
        grid, buildings = world
        policy = policy_all_deterministic()
        router = CognitionRouter(policy=policy)
        mgr = NPCManager(
            grid=grid, buildings=buildings,
            llm=MockProvider(), seed=42, router=router,
        )
        assert mgr.router is router

    def test_deterministic_schedule_generation(self, manager):
        """When router says deterministic, planner generates a schedule."""
        from core.npc.cognition.router import policy_all_deterministic
        manager.router.set_policy(policy_all_deterministic())
        manager.spawn_population(3)
        for npc in manager.npcs:
            npc.daily_schedule = []
            npc.schedule_day = 0
        # Trigger deterministic schedule for each
        for npc in manager.npcs:
            manager._generate_deterministic_schedule(npc, "morning", 1)
        for npc in manager.npcs:
            assert len(npc.daily_schedule) >= 1
            assert npc.schedule_day == 1

    def test_get_state_includes_economy(self, manager):
        manager.spawn_population(2)
        state = manager.get_state()
        assert "economy" in state
        assert "resources" in state["economy"]

    def test_get_state_includes_cognition(self, manager):
        manager.spawn_population(2)
        state = manager.get_state()
        assert "cognition" in state
        assert "total_decisions" in state["cognition"]

    def test_build_tick_state_includes_economy(self, manager):
        manager.spawn_population(1)
        state = manager._build_tick_state()
        assert "economy" in state


class TestEconomyTick:
    """Verify the economy tick orchestrator works."""

    def test_economy_initialises_resources(self, manager):
        nodes = manager.economy.resources.get_all_nodes()
        # Generator places resource nodes, so there should be some
        assert isinstance(nodes, list)

    def test_economy_tick_runs(self, manager):
        manager.spawn_population(2)
        # Should not raise
        manager.economy.tick(manager.npcs, 1.0, 100.0)

    def test_resource_node_dicts(self, manager):
        dicts = manager.economy.get_resource_node_dicts()
        assert isinstance(dicts, list)
        for d in dicts:
            assert "node_id" in d
            assert "x" in d
            assert "z" in d

    def test_available_recipes(self, manager):
        recipes = manager.economy.get_available_recipes()
        assert isinstance(recipes, list)
