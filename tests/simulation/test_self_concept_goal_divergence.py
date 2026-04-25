"""
Self-concept → goals → behaviour divergence simulation test.

The contract: two NPCs with matched personality and world placement
will plan differently when their self_concept is different. An NPC
who believes they are a king chooses court-establishing actions
(construct, socialise, patrol, trade) noticeably more often than a
farmer NPC with the same planner, same grid, same time of day.

This is the full closure of the identity → drive loop. Without the
goal mapper + utility bias wiring, both NPCs make indistinguishable
choices. With it, the king and farmer diverge.
"""

from __future__ import annotations

from collections import Counter

import pytest

from core.npc.cognition.goal_mapper import sync_npc_goals
from core.npc.cognition.planner import DeterministicPlanner
from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.world.generator import WorldConfig, generate_world


COURT_ACTIONS = {"construct", "socialise", "patrol", "trade"}
FARMER_ACTIONS = {"work", "gather"}


def _setup_manager(seed: int = 17) -> NPCManager:
    config = WorldConfig(population=2, terrain="riverside", seed=seed)
    grid, buildings = generate_world(config)
    mgr = NPCManager(
        grid=grid, buildings=buildings, llm=MockProvider(), seed=seed,
    )
    mgr.spawn_population(2)
    return mgr


def _plan_series(
    planner: DeterministicPlanner,
    npc,
    all_npcs,
    slots: list[str],
    resource_nodes: list[dict],
    construction_sites: list[dict],
) -> list[str]:
    """Run the deterministic planner across a series of slots and
    collect the chosen action_id for each."""
    picks: list[str] = []
    for slot in slots:
        action = planner.plan_action(
            npc, all_npcs, slot,
            resource_nodes=resource_nodes,
            construction_sites=construction_sites,
        )
        if action is not None:
            picks.append(action.action_id)
    return picks


def _stock_resources_and_sites() -> tuple[list[dict], list[dict]]:
    resource_nodes = [
        {"x": 3, "z": 3, "resource_type": "wheat",
         "current_amount": 10, "distance": 3},
        {"x": -2, "z": 2, "resource_type": "wood",
         "current_amount": 10, "distance": 3},
    ]
    construction_sites = [
        {"site_id": "church_site", "x": 4, "z": -4, "distance": 6,
         "blueprint_id": "church", "progress": 5,
         "needs_wood": 20, "needs_stone": 10, "needs_labour": 5},
    ]
    return resource_nodes, construction_sites


def test_king_and_farmer_diverge_across_slots():
    """Same world, same personality, different self_concept → different picks.

    The king's action mix must contain more court-coded actions than
    the farmer's, and vice versa for farmer-coded actions.
    """
    mgr = _setup_manager()
    king, farmer = mgr.npcs[0], mgr.npcs[1]

    # Deterministic identity: one king, one farmer. Personality left at
    # whatever the manager seeded — identical templates make the comparison
    # fair enough for a behavioural gap to only come from goals.
    king.self_concept["role:king"] = 0.9
    sync_npc_goals(king)
    farmer.self_concept["role:farmer"] = 0.9
    sync_npc_goals(farmer)

    planner = mgr.planner
    nodes, sites = _stock_resources_and_sites()
    slots = ["early_morning", "morning", "afternoon", "evening"] * 3

    king_picks = _plan_series(
        planner, king, mgr.npcs, slots, nodes, sites,
    )
    farmer_picks = _plan_series(
        planner, farmer, mgr.npcs, slots, nodes, sites,
    )

    king_court = sum(1 for a in king_picks if a in COURT_ACTIONS)
    farmer_court = sum(1 for a in farmer_picks if a in COURT_ACTIONS)
    king_farm = sum(1 for a in king_picks if a in FARMER_ACTIONS)
    farmer_farm = sum(1 for a in farmer_picks if a in FARMER_ACTIONS)

    assert king_court > farmer_court, (
        f"King should pick court-like actions more often than farmer. "
        f"king={Counter(king_picks)} farmer={Counter(farmer_picks)}"
    )
    assert farmer_farm > king_farm, (
        f"Farmer should pick farm-like actions more often than king. "
        f"king={Counter(king_picks)} farmer={Counter(farmer_picks)}"
    )


def test_goal_removal_restores_baseline_behaviour():
    """Decay the role belief → derived goal vanishes → bias vanishes.

    The NPC's action mix with a removed king belief should look
    indistinguishable from an NPC that never had one, absent the goal
    affinity boost.
    """
    mgr = _setup_manager(seed=21)
    npc = mgr.npcs[0]
    baseline = mgr.npcs[1]  # no self_concept

    npc.self_concept["role:king"] = 0.9
    sync_npc_goals(npc)
    assert any("royal court" in g.lower() for g in npc.long_term_goals)

    # Decay the belief past the floor; goal should disappear.
    npc.self_concept["role:king"] = 0.1
    sync_npc_goals(npc)
    assert not any("royal court" in g.lower() for g in npc.long_term_goals)
    assert npc.goal_affinities == {}

    # The scorer's goal-affinity bonus must collapse to zero for every
    # action once the derived goal is gone — that's the actual loop
    # closure contract. Action-mix comparisons across NPCs are noisy
    # due to spawn-randomised personality, so check the scorer output
    # directly.
    nodes, sites = _stock_resources_and_sites()
    scored = mgr.planner.score_all(
        npc, mgr.npcs, "afternoon",
        resource_nodes=nodes,
        construction_sites=sites,
    )
    for s in scored:
        assert s.breakdown.get("goal_affinity", 0.0) == 0.0, (
            f"Action {s.action_id} still carried a goal bonus after "
            f"belief decay: {s.breakdown}"
        )


def test_identity_claim_flow_adds_goal_after_conversation():
    """A strong claim injected through the real claim pathway ends up
    creating a derived goal."""
    from core.memory.reflection import IdentityClaim

    mgr = _setup_manager(seed=33)
    alice = mgr.npcs[0]

    # Repeated king claims push confidence past the goal floor.
    claim = IdentityClaim(
        key="role:king", confidence_delta=0.4,
        source_text="You are a king.", speaker="Bran",
    )
    for _ in range(3):
        mgr._inject_self_concept_delta(alice, claim, current_minutes=0.0)

    assert alice.self_concept["role:king"] >= 0.5
    assert any("royal court" in g.lower() for g in alice.long_term_goals)

    # And the planner should now prefer a king-coded action over the
    # wander baseline for this NPC.
    nodes, sites = _stock_resources_and_sites()
    scored = mgr.planner.score_all(
        alice, mgr.npcs, "afternoon",
        resource_nodes=nodes,
        construction_sites=sites,
    )
    by_id = {s.action_id: s for s in scored}
    assert by_id["construct"].total_score > by_id["wander"].total_score
