"""
Tests for the spatial awareness layer.

Enforces the invariant: no two resting NPCs may share a tile.
Covers rest-tile selection, conversation positioning, overlap
resolution, and a multi-NPC simulation stress test.
"""

import pytest

from core.world.grid import Grid
from core.world.spatial_awareness import (
    get_occupied_tiles,
    find_rest_tile,
    find_conversation_positions,
    resolve_overlaps,
)
from core.npc.models import NPC, PersonalityTraits, ActivityState


def _make_grid(size: int = 20) -> Grid:
    """Create a simple passable grid for testing."""
    return Grid(size, size)


def _make_npc(
    npc_id: str, x: int, z: int,
    activity: ActivityState = ActivityState.IDLE,
) -> NPC:
    """Create a minimal NPC at a given position."""
    npc = NPC(
        npc_id=npc_id,
        name=npc_id.capitalize(),
        age=30,
        personality=PersonalityTraits(),
        backstory="Test NPC.",
        occupation="labourer",
        x=x, z=z,
        home_x=x, home_z=z,
    )
    npc.activity = activity
    return npc


# ---------- Invariant helper ----------


def assert_no_resting_overlaps(npcs: list[NPC]) -> None:
    """Assert that no two resting NPCs share the same tile."""
    occupied: dict[tuple[int, int], str] = {}
    for npc in npcs:
        if npc.activity == ActivityState.WALKING:
            continue
        pos = (npc.x, npc.z)
        existing = occupied.get(pos)
        assert existing is None, (
            f"OVERLAP: {npc.npc_id} and {existing} both resting at {pos}"
        )
        occupied[pos] = npc.npc_id


# ---------- get_occupied_tiles ----------


class TestGetOccupiedTiles:

    def test_resting_npcs_included(self):
        npcs = [
            _make_npc("a", 0, 0, ActivityState.IDLE),
            _make_npc("b", 1, 1, ActivityState.WORKING),
        ]
        occ = get_occupied_tiles(npcs)
        assert (0, 0) in occ
        assert (1, 1) in occ

    def test_walking_npcs_excluded(self):
        npcs = [
            _make_npc("a", 0, 0, ActivityState.WALKING),
            _make_npc("b", 1, 1, ActivityState.IDLE),
        ]
        occ = get_occupied_tiles(npcs)
        assert (0, 0) not in occ
        assert (1, 1) in occ

    def test_empty_list(self):
        assert get_occupied_tiles([]) == set()

    def test_all_walking(self):
        npcs = [
            _make_npc("a", 0, 0, ActivityState.WALKING),
            _make_npc("b", 1, 1, ActivityState.WALKING),
        ]
        assert get_occupied_tiles(npcs) == set()

    def test_all_activity_states_except_walking_count(self):
        states = [
            ActivityState.IDLE, ActivityState.WORKING,
            ActivityState.SLEEPING, ActivityState.TALKING,
            ActivityState.EATING, ActivityState.GATHERING,
        ]
        npcs = [_make_npc(f"npc_{i}", i, 0, s) for i, s in enumerate(states)]
        occ = get_occupied_tiles(npcs)
        assert len(occ) == len(states)


# ---------- find_rest_tile ----------


class TestFindRestTile:

    def test_free_target_returned_directly(self):
        grid = _make_grid()
        pos = find_rest_tile(0, 0, grid, set())
        assert pos == (0, 0)

    def test_occupied_target_returns_neighbour(self):
        grid = _make_grid()
        occupied = {(0, 0)}
        pos = find_rest_tile(0, 0, grid, occupied)
        assert pos != (0, 0)
        assert pos not in occupied
        # Should be close
        assert abs(pos[0]) + abs(pos[1]) <= 2

    def test_heavily_occupied_still_finds_free(self):
        grid = _make_grid()
        # Occupy a 3x3 block around origin
        occupied = {(x, z) for x in range(-1, 2) for z in range(-1, 2)}
        pos = find_rest_tile(0, 0, grid, occupied)
        assert pos not in occupied
        tile = grid.get_tile(pos[0], pos[1])
        assert tile is not None and tile.is_passable

    def test_exclude_self_from_occupied(self):
        npcs = [_make_npc("a", 0, 0, ActivityState.IDLE)]
        occupied = get_occupied_tiles(npcs)
        # NPC "a" should be able to keep its own tile
        pos = find_rest_tile(0, 0, _make_grid(), occupied, exclude_npc_id="a", npcs=npcs)
        assert pos == (0, 0)

    def test_result_is_passable(self):
        grid = _make_grid()
        # Make origin impassable
        tile = grid.get_tile(0, 0)
        tile.walkable = False
        pos = find_rest_tile(0, 0, grid, set())
        assert pos != (0, 0)
        result_tile = grid.get_tile(pos[0], pos[1])
        assert result_tile.is_passable

    def test_never_returns_occupied(self):
        """Run 50 times with random occupancy patterns."""
        import random
        rng = random.Random(42)
        grid = _make_grid(30)
        for _ in range(50):
            occupied = {(rng.randint(-10, 10), rng.randint(-10, 10)) for _ in range(20)}
            pos = find_rest_tile(0, 0, grid, occupied)
            assert pos not in occupied


# ---------- find_conversation_positions ----------


class TestFindConversationPositions:

    def test_returns_adjacent_tiles(self):
        grid = _make_grid()
        a = _make_npc("a", 0, 0)
        b = _make_npc("b", 3, 3)
        pos_a, pos_b = find_conversation_positions(a, b, grid, set())
        dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
        assert dist == 1, f"Conversation positions {pos_a} and {pos_b} are not adjacent (dist={dist})"

    def test_already_adjacent_kept(self):
        grid = _make_grid()
        a = _make_npc("a", 0, 0)
        b = _make_npc("b", 1, 0)
        pos_a, pos_b = find_conversation_positions(a, b, grid, set())
        assert pos_a == (0, 0)
        assert pos_b == (1, 0)

    def test_avoids_occupied_tiles(self):
        grid = _make_grid()
        a = _make_npc("a", 0, 0)
        b = _make_npc("b", 3, 0)
        occupied = {(1, 0), (2, 0)}
        pos_a, pos_b = find_conversation_positions(a, b, grid, occupied)
        assert pos_a not in occupied
        assert pos_b not in occupied
        dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
        assert dist == 1

    def test_same_tile_gets_separated(self):
        grid = _make_grid()
        a = _make_npc("a", 0, 0)
        b = _make_npc("b", 0, 0)
        pos_a, pos_b = find_conversation_positions(a, b, grid, set())
        assert pos_a != pos_b
        dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
        assert dist == 1

    def test_positions_are_passable(self):
        grid = _make_grid()
        a = _make_npc("a", 0, 0)
        b = _make_npc("b", 2, 2)
        pos_a, pos_b = find_conversation_positions(a, b, grid, set())
        assert grid.get_tile(pos_a[0], pos_a[1]).is_passable
        assert grid.get_tile(pos_b[0], pos_b[1]).is_passable


# ---------- resolve_overlaps ----------


class TestResolveOverlaps:

    def test_no_overlaps_no_changes(self):
        grid = _make_grid()
        npcs = [
            _make_npc("a", 0, 0),
            _make_npc("b", 1, 1),
        ]
        moved = resolve_overlaps(npcs, grid)
        assert moved == 0

    def test_two_stacked_one_moves(self):
        grid = _make_grid()
        npcs = [
            _make_npc("a", 0, 0),
            _make_npc("b", 0, 0),
        ]
        moved = resolve_overlaps(npcs, grid)
        assert moved == 1
        assert_no_resting_overlaps(npcs)

    def test_three_stacked_two_move(self):
        grid = _make_grid()
        npcs = [
            _make_npc("a", 0, 0),
            _make_npc("b", 0, 0),
            _make_npc("c", 0, 0),
        ]
        moved = resolve_overlaps(npcs, grid)
        assert moved == 2
        assert_no_resting_overlaps(npcs)

    def test_walking_npcs_ignored(self):
        grid = _make_grid()
        npcs = [
            _make_npc("a", 0, 0, ActivityState.IDLE),
            _make_npc("b", 0, 0, ActivityState.WALKING),
        ]
        moved = resolve_overlaps(npcs, grid)
        assert moved == 0  # walker doesn't count as overlap

    def test_multiple_clusters_resolved(self):
        grid = _make_grid()
        npcs = [
            _make_npc("a", 0, 0),
            _make_npc("b", 0, 0),
            _make_npc("c", 5, 5),
            _make_npc("d", 5, 5),
        ]
        resolve_overlaps(npcs, grid)
        assert_no_resting_overlaps(npcs)


# ---------- Simulation stress test ----------


class TestSimulationInvariant:
    """
    Multi-NPC stress test: spawn NPCs, simulate arrivals and
    conversations, and verify the no-overlap invariant at every step.
    """

    def test_20_npcs_100_arrivals_no_overlaps(self):
        """Simulate 20 NPCs arriving at the same area. No overlaps allowed."""
        import random
        rng = random.Random(42)
        grid = _make_grid(40)

        npcs = [_make_npc(f"npc_{i}", rng.randint(-5, 5), rng.randint(-5, 5))
                for i in range(20)]

        # Each NPC "arrives" — resolve overlaps after each batch
        for _ in range(100):
            # Pick a random NPC and send it to a random spot
            npc = rng.choice(npcs)
            target_x = rng.randint(-3, 3)
            target_z = rng.randint(-3, 3)

            occupied = get_occupied_tiles(npcs)
            safe_pos = find_rest_tile(
                target_x, target_z, grid, occupied,
                exclude_npc_id=npc.npc_id, npcs=npcs,
            )
            npc.x, npc.z = safe_pos

            # Run safety net
            resolve_overlaps(npcs, grid)

            # INVARIANT: no resting overlaps
            assert_no_resting_overlaps(npcs)

    def test_nudge_sets_trail(self):
        """When resolve_overlaps nudges an NPC, it should set _tick_trail
        so the client can animate the separation."""
        grid = _make_grid()
        npcs = [
            _make_npc("a", 0, 0),
            _make_npc("b", 0, 0),
        ]
        npcs[0]._tick_trail = []
        npcs[1]._tick_trail = []

        resolve_overlaps(npcs, grid)

        # NPC b should have been nudged and have trail data
        assert hasattr(npcs[1], '_tick_trail')
        assert len(npcs[1]._tick_trail) > 0, (
            "Nudged NPC should have trail data for client animation"
        )

    def test_10_conversations_no_overlaps(self):
        """Simulate 10 conversation pairs positioning. No overlaps allowed."""
        import random
        rng = random.Random(99)
        grid = _make_grid(30)

        npcs = [_make_npc(f"npc_{i}", rng.randint(-5, 5), rng.randint(-5, 5))
                for i in range(20)]

        # Resolve initial overlaps
        resolve_overlaps(npcs, grid)

        for _ in range(10):
            # Pick two random non-conversing NPCs
            available = [n for n in npcs if n.conversation_partner is None]
            if len(available) < 2:
                break
            a, b = rng.sample(available, 2)

            occupied = get_occupied_tiles(npcs)
            pos_a, pos_b = find_conversation_positions(a, b, grid, occupied)
            a.x, a.z = pos_a
            b.x, b.z = pos_b
            a.activity = ActivityState.TALKING
            b.activity = ActivityState.TALKING
            a.conversation_partner = b.npc_id
            b.conversation_partner = a.npc_id

            # Verify adjacency
            dist = abs(a.x - b.x) + abs(a.z - b.z)
            assert dist == 1, f"{a.npc_id} and {b.npc_id} not adjacent after positioning (dist={dist})"

            # INVARIANT: no resting overlaps
            assert_no_resting_overlaps(npcs)
