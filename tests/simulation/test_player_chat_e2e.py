"""
End-to-end player chat and sim responsiveness test.

Exercises the exact same code paths as the browser:
  1. Spawn world with player + NPCs
  2. Place player adjacent to an NPC
  3. Send a chat message via the server handler
  4. Assert NPC responds with actual text
  5. End the chat
  6. Run sim ticks and assert NPCs still move (sim not frozen)

This is NOT a unit test. It runs the real server handler code,
the real conversation system, and the real LLM provider (Gemma via
Ollama, falling back to MockProvider if Ollama is offline).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.npc.manager import NPCManager
from core.npc.models import ActivityState
from core.npc.llm_client import MockProvider
from core.memory.manager import MemoryManager
from core.memory.episodic import EpisodicStore
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world
from core.player.player_agent import PlayerAgent
from core.npc.cognition.converse import (
    initiate_conversation,
    continue_conversation,
    _active_conversations,
    end_conversation,
    clear_finished_conversations,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def make_sim():
    """Create a deterministic world with player and NPCs."""
    # Clear any leaked conversation state from previous tests
    _active_conversations.clear()

    config = WorldConfig(population=8, terrain="riverside", seed=42)
    grid, buildings = generate_world(config)

    # Try Gemma first, fall back to MockProvider
    try:
        from core.npc.gemma_provider import GemmaProvider
        import urllib.request
        import json
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if any("gemma" in m for m in models):
            llm = GemmaProvider()
            logger.info("Using GemmaProvider (Ollama online, models: %s)", models)
        else:
            llm = MockProvider()
            logger.info("Ollama online but no Gemma model — using MockProvider")
    except Exception as e:
        llm = MockProvider()
        logger.info("Ollama offline (%s) — using MockProvider", e)

    episodic = EpisodicStore(fallback_only=True)
    memory = MemoryManager(llm=llm, episodic=episodic)
    mgr = NPCManager(
        grid=grid,
        buildings=buildings,
        llm=llm,
        seed=42,
        memory=memory,
        deterministic=True,
    )
    mgr.spawn_population(config.population)
    clock = GameClock()
    player = PlayerAgent.create()
    return mgr, clock, grid, buildings, llm, memory, player


async def test_player_chat_gets_response():
    """
    CRITICAL TEST: When a player sends a chat message to a nearby NPC,
    the NPC MUST respond with non-empty text.
    """
    mgr, clock, grid, buildings, llm, memory, player = make_sim()

    # Pick first NPC
    target = mgr.npcs[0]
    logger.info("Target NPC: %s (%s) at (%s, %s)", target.name, target.npc_id, target.x, target.z)

    # Teleport player adjacent to target
    player.npc.x = float(target.x)
    player.npc.z = float(target.z) + 1.0
    logger.info("Player at (%s, %s), target at (%s, %s), distance=%s",
                player.npc.x, player.npc.z, target.x, target.z,
                player.npc.distance_to(target.x, target.z))

    # Verify distance is within chat range
    dist = player.npc.distance_to(target.x, target.z)
    assert dist <= player.interaction_radius, (
        f"Player distance {dist} > interaction radius {player.interaction_radius}"
    )

    # Step 1: Initiate conversation (same as _handle_player_chat first call)
    current_minutes = clock.day * MINUTES_PER_DAY + clock.minutes
    saved_x, saved_z = player.npc.x, player.npc.z

    conv = await initiate_conversation(
        player.npc, target, llm,
        current_minutes, memory,
    )
    # Restore player position
    player.npc.x, player.npc.z = saved_x, saved_z

    assert conv is not None, "initiate_conversation returned None — NPC refused to talk"
    logger.info("Conversation initiated. NPC greeting: '%s'", conv.exchanges[0].message if conv.exchanges else "NONE")
    assert len(conv.exchanges) >= 1, "No exchanges after initiation"
    assert conv.exchanges[0].message, "NPC greeting is empty"

    # Step 2: Add player message and get NPC response
    player_message = "Hello there! How are you today?"
    conv.add_exchange(player.npc_id, player.name, player_message)
    logger.info("Player said: '%s'", player_message)

    continues = await continue_conversation(
        target, player.npc, llm, memory,
    )

    # The last exchange should be from the NPC, not the player
    assert len(conv.exchanges) >= 3, (
        f"Expected >= 3 exchanges (greeting + player + response), got {len(conv.exchanges)}:\n"
        + "\n".join(f"  {e.speaker_name}: {e.message}" for e in conv.exchanges)
    )
    last = conv.exchanges[-1]
    logger.info("NPC responded: '%s' (speaker: %s)", last.message, last.speaker_name)

    assert last.speaker_id == target.npc_id, (
        f"Last exchange is from {last.speaker_name} ({last.speaker_id}), "
        f"expected {target.name} ({target.npc_id}). Exchanges:\n"
        + "\n".join(f"  {e.speaker_name}: {e.message}" for e in conv.exchanges)
    )
    assert last.message and last.message.strip(), (
        f"NPC response is empty! Exchanges:\n"
        + "\n".join(f"  {e.speaker_name}: {e.message}" for e in conv.exchanges)
    )

    # Step 3: Verify the response dict matches what the server sends to client
    response_dict = {
        "type": "chat_response",
        "npc_id": target.npc_id,
        "npc_name": target.name,
        "message": last.message,
        "ended": not continues,
    }
    assert response_dict["npc_name"] is not None, "npc_name is None in response"
    assert response_dict["message"] is not None, "message is None in response"
    assert response_dict["message"] != "", "message is empty string in response"
    logger.info("Response dict: %s", response_dict)

    print("\n=== CHAT TEST PASSED ===")
    print(f"  Player said: '{player_message}'")
    print(f"  {target.name} replied: '{last.message}'")
    print(f"  Conversation continues: {continues}")
    return True


async def test_sim_responsive_after_chat_close():
    """
    CRITICAL TEST: After a player chat ends, the sim must continue
    ticking. NPCs must still move and act. The sim must not freeze.
    """
    mgr, clock, grid, buildings, llm, memory, player = make_sim()

    target = mgr.npcs[0]

    # Place player adjacent
    player.npc.x = float(target.x)
    player.npc.z = float(target.z) + 1.0

    current_minutes = clock.day * MINUTES_PER_DAY + clock.minutes

    # Start and end a conversation
    conv = await initiate_conversation(
        player.npc, target, llm,
        current_minutes, memory,
    )
    assert conv is not None, "Failed to initiate conversation"

    # End it
    await end_conversation(player.npc, target, memory_manager=memory)
    logger.info("Conversation ended")

    # Clear player chat state and move player far away
    # (so cognition doesn't re-initiate conversation with adjacent player)
    player.is_chatting = False
    player.chat_target_id = None
    player.npc.x = 0.0
    player.npc.z = 0.0

    # Now run 100 ticks and check NPCs still function
    positions_before = {npc.npc_id: (npc.tile_x, npc.tile_z) for npc in mgr.npcs}
    activities_seen = set()

    for tick in range(100):
        clock.tick(1.0)
        mgr.movement_tick(clock, 1.0)
        await mgr.cognition_tick(clock, 1.0)

        for npc in mgr.npcs:
            activities_seen.add(npc.activity.value)

    positions_after = {npc.npc_id: (npc.tile_x, npc.tile_z) for npc in mgr.npcs}

    # At least some NPCs should have moved
    moved_count = sum(
        1 for npc_id in positions_before
        if positions_before[npc_id] != positions_after.get(npc_id, positions_before[npc_id])
    )

    # The target NPC should no longer be "talking"
    assert target.activity != ActivityState.TALKING, (
        f"{target.name} is still stuck in TALKING after conversation ended"
    )
    assert target.conversation_partner is None, (
        f"{target.name} still has conversation_partner={target.conversation_partner}"
    )

    logger.info("After 100 ticks: %d NPCs moved, activities seen: %s", moved_count, activities_seen)

    assert moved_count > 0, (
        f"NO NPCs moved after 100 ticks post-chat! Sim is frozen.\n"
        f"Activities seen: {activities_seen}\n"
        f"Positions: {positions_after}"
    )

    print("\n=== SIM RESPONSIVENESS TEST PASSED ===")
    print(f"  {moved_count}/{len(mgr.npcs)} NPCs moved after chat ended")
    print(f"  Activities seen: {activities_seen}")
    print(f"  Target NPC ({target.name}) activity: {target.activity.value}")
    return True


async def test_chat_response_not_player_echo():
    """
    Regression: The server must never echo the player's own message
    back as the NPC response. The last exchange after continue_conversation
    must be from the NPC, not the player.
    """
    mgr, clock, grid, buildings, llm, memory, player = make_sim()
    target = mgr.npcs[0]

    player.npc.x = float(target.x)
    player.npc.z = float(target.z) + 1.0

    current_minutes = clock.day * MINUTES_PER_DAY + clock.minutes

    conv = await initiate_conversation(
        player.npc, target, llm,
        current_minutes, memory,
    )
    assert conv is not None

    # Player says something unique
    player_msg = "UNIQUE_TEST_STRING_12345"
    conv.add_exchange(player.npc_id, player.name, player_msg)

    await continue_conversation(target, player.npc, llm, memory)

    last = conv.exchanges[-1]

    # The response must NOT be the player's message echoed back
    assert last.speaker_id != player.npc_id, (
        f"Last exchange is from PLAYER, not NPC! Server would echo player's own message.\n"
        f"Exchanges:\n"
        + "\n".join(f"  {e.speaker_name}: {e.message}" for e in conv.exchanges)
    )
    assert last.message != player_msg, (
        f"NPC response is identical to player message — echo bug"
    )

    print("\n=== NO-ECHO TEST PASSED ===")
    print(f"  Player said: '{player_msg}'")
    print(f"  NPC ({target.name}) said: '{last.message}'")
    return True


def main():
    """Run all tests and report results."""
    tests = [
        ("Player chat gets NPC response", test_player_chat_gets_response),
        ("Sim responsive after chat close", test_sim_responsive_after_chat_close),
        ("Chat response not player echo", test_chat_response_not_player_echo),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print(f"{'='*60}")
        try:
            passed = asyncio.new_event_loop().run_until_complete(test_fn())
            results.append((name, True, ""))
        except Exception as e:
            logger.error("FAILED: %s — %s", name, e)
            results.append((name, False, str(e)))

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    passed = 0
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if err:
            print(f"         {err[:200]}")
        if ok:
            passed += 1

    total = len(results)
    print(f"\n{passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
