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
            assert 2.0 <= npc.move_speed <= 4.0

    def test_action_advance_dispatches_immediately(self, manager):
        """Advancing to next action should dispatch NPCs immediately (Stanford model)."""
        import asyncio
        npcs = manager.spawn_population(3)
        from core.npc.models import ScheduleEntry, ActivityState
        for npc in npcs:
            npc.daily_schedule = [
                ScheduleEntry("morning", "eat breakfast", "home", 3, duration_minutes=60),
                ScheduleEntry("morning", "work", "work", 5, duration_minutes=240),
            ]
            npc.schedule_index = 0
            npc.action_start_minutes = 0.0
        loop = asyncio.new_event_loop()
        # Advance each NPC to the second entry
        for npc in npcs:
            loop.run_until_complete(manager._advance_npc_action(npc, 60.0, "morning"))
        # NPCs should be walking or at destination — no stagger queue
        for npc in npcs:
            assert npc.schedule_index == 1
            assert npc.action_start_minutes == 60.0

    def test_action_advance_updates_schedule_index(self, manager):
        """Each advance should increment schedule_index and reset timer."""
        import asyncio
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
        loop = asyncio.new_event_loop()
        loop.run_until_complete(manager._advance_npc_action(npc, 60.0, "morning"))
        assert npc.schedule_index == 1
        loop.run_until_complete(manager._advance_npc_action(npc, 300.0, "afternoon"))
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


class TestRelationshipSeeding:
    """Verify inter-NPC relationships are seeded at spawn."""

    def test_npcs_have_relationship_facts(self, manager):
        """NPCs should know about other NPCs after spawning."""
        npcs = manager.spawn_population(10)
        # With 10 NPCs across various occupations, there should be
        # occupational bonds + neighbours + acquaintances
        total_rel_facts = 0
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            rel_facts = [
                f for f in facts
                if f.predicate in (
                    "trades_with", "supplies", "respects", "knows_well",
                    "works_for", "neighbour_of", "knows",
                )
            ]
            total_rel_facts += len(rel_facts)
        assert total_rel_facts >= 5, (
            f"Expected at least 5 relationship facts, got {total_rel_facts}"
        )

    def test_npcs_have_relationship_memories(self, manager):
        """NPCs should have episodic memories about relationships."""
        npcs = manager.spawn_population(10)
        rel_memory_count = 0
        for npc in npcs:
            memories = manager.memory.episodic.get_recent(npc.npc_id, limit=100)
            rel_memories = [
                m for m in memories if m.category == "relationship"
            ]
            rel_memory_count += len(rel_memories)
        assert rel_memory_count >= 5

    def test_npcs_know_others_occupations(self, manager):
        """NPCs with relationships should know the other's occupation."""
        npcs = manager.spawn_population(10)
        # Pick an NPC with relationship facts
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            rel_facts = [
                f for f in facts
                if f.predicate in (
                    "trades_with", "supplies", "knows_well",
                    "works_for", "neighbour_of", "knows",
                )
            ]
            if not rel_facts:
                continue
            # For each relationship, NPC should know the other's occupation
            other_name = rel_facts[0].obj
            about_facts = manager.memory.structured.get_facts_about(
                npc.npc_id, about=other_name,
            )
            is_a = [f for f in about_facts if f.predicate == "is_a"]
            assert len(is_a) >= 1, (
                f"{npc.name} knows {other_name} but doesn't know their occupation"
            )
            return  # One successful check is enough
        pytest.fail("No NPC had any relationship facts to test")

    def test_initial_sentiment_seeded(self, manager):
        """Occupational bonds should create initial sentiment values."""
        npcs = manager.spawn_population(10)
        # Find a pair with an occupational bond
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            rel_facts = [
                f for f in facts
                if f.predicate in ("trades_with", "supplies", "respects")
            ]
            if not rel_facts:
                continue
            other_name = rel_facts[0].obj
            other = next(
                (n for n in npcs if n.name == other_name), None,
            )
            if other is None:
                continue
            s = manager.sentiment.get(npc.npc_id, other.npc_id)
            # Should have non-zero trust or respect
            assert s.trust > 0 or s.respect > 0, (
                f"Expected positive sentiment from {npc.name} to {other.name}"
            )
            return
        # If no occupational bonds were found (unlikely with 10 NPCs), skip
        pytest.skip("No occupational bonds found in this seed")

    def test_acquaintances_seeded(self, manager):
        """Each NPC should know at least one other NPC."""
        npcs = manager.spawn_population(10)
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            knows_someone = any(
                f.predicate in (
                    "trades_with", "supplies", "respects", "knows_well",
                    "works_for", "neighbour_of", "knows",
                )
                for f in facts
            )
            assert knows_someone, (
                f"{npc.name} ({npc.occupation}) doesn't know anyone"
            )

    def test_neighbours_are_bidirectional(self, manager):
        """If A is neighbour_of B, then B should be neighbour_of A."""
        npcs = manager.spawn_population(10)
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            neighbour_facts = [f for f in facts if f.predicate == "neighbour_of"]
            for nf in neighbour_facts:
                other_name = nf.obj
                other = next((n for n in npcs if n.name == other_name), None)
                if other is None:
                    continue
                other_facts = manager.memory.structured.get_facts(
                    other.npc_id, limit=100,
                )
                reverse = [
                    f for f in other_facts
                    if f.predicate == "neighbour_of" and f.obj == npc.name
                ]
                assert len(reverse) >= 1, (
                    f"{npc.name} is neighbour_of {other_name} but not vice versa"
                )

    def test_neighbour_distance_rule(self, manager):
        """Neighbours should have homes within Manhattan distance 5."""
        npcs = manager.spawn_population(10)
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            neighbour_facts = [f for f in facts if f.predicate == "neighbour_of"]
            for nf in neighbour_facts:
                other = next(
                    (n for n in npcs if n.name == nf.obj), None,
                )
                if other is None:
                    continue
                dist = abs(npc.home_x - other.home_x) + abs(npc.home_z - other.home_z)
                assert dist <= 5, (
                    f"{npc.name} and {other.name} are neighbours but "
                    f"homes are {dist} tiles apart"
                )

    def test_initial_sentiment_values_match_bond(self, manager):
        """Occupational bond sentiment should match OCCUPATIONAL_BONDS values."""
        npcs = manager.spawn_population(10)
        # Find a pair with trades_with — should have trust=15, respect=10
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=100)
            trade_facts = [f for f in facts if f.predicate == "trades_with"]
            if not trade_facts:
                continue
            other = next(
                (n for n in npcs if n.name == trade_facts[0].obj), None,
            )
            if other is None:
                continue
            s = manager.sentiment.get(npc.npc_id, other.npc_id)
            # trades_with: {"trust": 15, "respect": 10}
            assert s.trust >= 10, f"Expected trust >= 10 for trade bond, got {s.trust}"
            assert s.respect >= 5, f"Expected respect >= 5 for trade bond, got {s.respect}"
            return
        pytest.skip("No trades_with bond found")

    def test_no_duplicate_relationships(self, manager):
        """An NPC should not have duplicate relationship predicates for the same person."""
        npcs = manager.spawn_population(10)
        for npc in npcs:
            facts = manager.memory.structured.get_facts(npc.npc_id, limit=200)
            rel_predicates = [
                "trades_with", "supplies", "respects", "knows_well",
                "works_for", "neighbour_of", "knows",
            ]
            pairs = [
                (f.predicate, f.obj) for f in facts if f.predicate in rel_predicates
            ]
            # Same (predicate, obj) should not appear twice
            assert len(pairs) == len(set(pairs)), (
                f"{npc.name} has duplicate relationship facts: {pairs}"
            )


class TestCustomSchedule:
    def test_assign_custom_schedule(self, manager):
        npcs = manager.spawn_population(3)
        npc = npcs[0]
        entries = [
            {"activity": "guard the bridge", "location": "work",
             "target_x": 15, "target_z": 0, "duration_minutes": 900},
            {"activity": "sleep at home", "location": "home",
             "duration_minutes": 540},
        ]
        ok, msg = manager.assign_custom_schedule(npc.npc_id, entries)
        assert ok, msg
        assert npc.has_custom_schedule is True
        assert len(npc.daily_schedule) == 2
        assert npc.schedule_index == 0

    def test_reject_wrong_total(self, manager):
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        entries = [
            {"activity": "guard", "location": "work", "duration_minutes": 500},
        ]
        ok, msg = manager.assign_custom_schedule(npc.npc_id, entries)
        assert not ok
        assert "1440" in msg

    def test_reject_missing_activity(self, manager):
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        entries = [
            {"location": "home", "duration_minutes": 1440},
        ]
        ok, msg = manager.assign_custom_schedule(npc.npc_id, entries)
        assert not ok
        assert "activity" in msg.lower()

    def test_reject_empty_entries(self, manager):
        npcs = manager.spawn_population(1)
        ok, msg = manager.assign_custom_schedule(npcs[0].npc_id, [])
        assert not ok

    def test_reject_unknown_npc(self, manager):
        manager.spawn_population(1)
        ok, msg = manager.assign_custom_schedule("fake_id", [
            {"activity": "idle", "duration_minutes": 1440},
        ])
        assert not ok

    def test_clear_custom_schedule(self, manager):
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        entries = [
            {"activity": "guard", "location": "work", "duration_minutes": 900},
            {"activity": "sleep", "location": "home", "duration_minutes": 540},
        ]
        manager.assign_custom_schedule(npc.npc_id, entries)
        ok, msg = manager.clear_custom_schedule(npc.npc_id)
        assert ok
        assert npc.has_custom_schedule is False
        assert len(npc.daily_schedule) > 0  # template regenerated

    def test_custom_schedule_loops(self, manager):
        """When a custom schedule is exhausted, it should loop back to index 0."""
        import asyncio
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        entries = [
            {"activity": "guard", "location": "work", "duration_minutes": 900},
            {"activity": "sleep", "location": "home", "duration_minutes": 540},
        ]
        manager.assign_custom_schedule(npc.npc_id, entries)
        original_schedule = list(npc.daily_schedule)

        # Walk to end of schedule
        npc.schedule_index = 1
        asyncio.new_event_loop().run_until_complete(
            manager._advance_npc_action(npc, 1440.0, "night")
        )

        # Should loop: index reset, same schedule kept
        assert npc.schedule_index == 0
        assert npc.daily_schedule == original_schedule
        assert npc.has_custom_schedule is True

    def test_target_coords_preserved(self, manager):
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        entries = [
            {"activity": "guard", "target_x": 15, "target_z": 7,
             "duration_minutes": 1440},
        ]
        ok, _ = manager.assign_custom_schedule(npc.npc_id, entries)
        assert ok
        assert npc.daily_schedule[0].target_x == 15
        assert npc.daily_schedule[0].target_z == 7


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


class TestScheduleRegenDispatch:
    """Regression: NPCs stuck in houses after schedule exhaustion.

    Root cause was action_start_minutes not reset to 0.0 on schedule
    regen, so the first-dispatch logic never fires for the new schedule.
    """

    def test_schedule_exhaust_resets_action_timer(self, manager):
        """After schedule exhaustion, action_start_minutes must be 0.0."""
        import asyncio
        from core.npc.models import ScheduleEntry
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        npc.daily_schedule = [
            ScheduleEntry("morning", "work", "work", 5, duration_minutes=120),
        ]
        npc.schedule_index = 0
        npc.action_start_minutes = 100.0

        # Advance past the single entry — exhausts the schedule
        asyncio.new_event_loop().run_until_complete(
            manager._advance_npc_action(npc, 220.0, "morning")
        )

        assert npc.schedule_index == 0, "Schedule index should reset to 0"
        assert npc.action_start_minutes == 0.0, (
            "action_start_minutes must be 0.0 after schedule exhaustion "
            "so first-dispatch logic fires on the next cognition tick"
        )

    def test_deterministic_regen_resets_action_timer(self, manager):
        """Deterministic schedule regen must also reset to 0.0."""
        import asyncio
        from core.npc.models import ScheduleEntry
        manager.deterministic = True
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        npc.daily_schedule = [
            ScheduleEntry("morning", "work", "work", 5, duration_minutes=120),
        ]
        npc.schedule_index = 0
        npc.action_start_minutes = 100.0

        asyncio.new_event_loop().run_until_complete(
            manager._advance_npc_action(npc, 220.0, "morning")
        )

        assert npc.action_start_minutes == 0.0, (
            "Deterministic regen must also reset action_start_minutes to 0.0"
        )

    def test_normal_advance_keeps_current_minutes(self, manager):
        """Normal advance (not exhaustion) should set current_minutes."""
        import asyncio
        from core.npc.models import ScheduleEntry
        npcs = manager.spawn_population(1)
        npc = npcs[0]
        npc.daily_schedule = [
            ScheduleEntry("morning", "eat", "home", 3, duration_minutes=60),
            ScheduleEntry("morning", "work", "work", 5, duration_minutes=240),
        ]
        npc.schedule_index = 0
        npc.action_start_minutes = 0.0

        asyncio.new_event_loop().run_until_complete(
            manager._advance_npc_action(npc, 60.0, "morning")
        )

        assert npc.schedule_index == 1
        assert npc.action_start_minutes == 60.0, (
            "Normal advance should anchor timer to current_minutes, not 0.0"
        )


class TestPlayerNPCFiltering:
    """Regression: chat targeted the player themselves.

    Root cause: player NPC in npc_manager.npcs is a plain NPC whose
    to_dict() lacks is_player. Client filter checked is_player but it
    was always undefined, so the player appeared as a nearby NPC.
    """

    def test_player_npc_identifiable_by_id(self, manager):
        """Player NPC must be identifiable by npc_id='player'."""
        npcs = manager.spawn_population(3)

        from core.player.player_agent import PlayerAgent
        player = PlayerAgent.create(name="Traveller", spawn_x=0, spawn_z=0)
        manager.npcs.append(player.npc)
        manager._npc_map[player.npc_id] = player.npc

        # Simulate what the client does: filter nearby NPCs
        all_npc_dicts = [npc.to_dict() for npc in manager.npcs]
        non_player = [
            d for d in all_npc_dicts
            if d["npc_id"] != "player" and not d.get("is_player")
        ]

        assert len(non_player) == 3, (
            f"Expected 3 non-player NPCs, got {len(non_player)}. "
            "Player NPC must be filterable by npc_id='player'."
        )

    def test_player_npc_id_is_player(self):
        """PlayerAgent always has npc_id='player'."""
        from core.player.player_agent import PlayerAgent
        player = PlayerAgent.create()
        assert player.npc_id == "player"
        assert player.npc.npc_id == "player"

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
