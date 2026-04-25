"""
Full-stack player chat e2e test.

Boots the REAL FastAPI server in a subprocess, connects over a real
WebSocket, and exercises the complete browser flow. The previous
simulation test (test_player_chat_e2e.py) only called the conversation
module directly and passed even while production chat was broken —
because the bug lived in the WebSocket handler, not the core logic.

These tests use Gemma if available, otherwise fall back to MockProvider
so they run in any environment. Server log is captured to a temp file
for diagnostic output on failure.

Tests (all must pass):
  1. test_chat_responds_without_blocking_input
     — player gets a chat response, AND moves sent during the chat
       reach the server and advance the player's position (proves the
       WS receive loop isn't blocked by the LLM).
  2. test_sim_responsive_after_chat_close
     — after explicit chat close, player can move and start new chats.
  3. test_multiple_chat_exchanges_never_echo_player
     — the NPC's response is never the player's own message echoed back.
  4. test_tick_broadcast_continuous
     — tick broadcasts arrive continuously during chat (~4 Hz), proving
       the tick loop runs independently of chat handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import websockets

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_PORT = int(os.environ.get("SMALLVILLE_TEST_PORT", "8912"))
SERVER_URL = f"http://localhost:{TEST_PORT}"
WS_URL = f"ws://localhost:{TEST_PORT}/ws"


# ---------- Server lifecycle ----------

def _start_server() -> tuple[subprocess.Popen, Path]:
    """Boot the real server on a test port. Returns (process, logfile)."""
    log_path = Path("/tmp") / f"smallville_test_server_{TEST_PORT}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    # MockProvider is fastest for tests; Gemma is exercised by other tests.
    # Ollama can make chat take 30-60s which is fine for one test run but
    # we want deterministic timing here.
    env["SMALLVILLE_DISABLE_OLLAMA"] = "1"
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("MISTRAL_API_KEY", None)

    patched = _write_patched_main_for_test()

    # Unbuffer child stdio so proximity prints flush immediately.
    env["PYTHONUNBUFFERED"] = "1"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(patched)],
        cwd=str(REPO),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_path


def _write_patched_main_for_test() -> Path:
    """Produce a throwaway server entry that listens on TEST_PORT and
    forces MockProvider so LLM latency can't skew timing.

    We set the API-key env vars to empty strings BEFORE importing the
    server so that dotenv's default override=False behaviour leaves
    them empty (an empty-string key is falsy in server/main.py's
    provider selection chain). We also stub ollama_available.

    We inject a small artificial delay into conversation calls so that
    the non-blocking-receive-loop test has a measurable window to send
    movement commands that must be processed concurrently.
    """
    patched = Path("/tmp") / f"smallville_server_test_{TEST_PORT}.py"
    patched.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "os.environ['ANTHROPIC_API_KEY'] = ''\n"
        "os.environ['MISTRAL_API_KEY'] = ''\n"
        "os.environ.setdefault('SMALLVILLE_MOCK_DELAY_MS', '1500')\n"
        "import core.npc.gemma_provider as _gp\n"
        "_gp.ollama_available = lambda *a, **k: False\n"
        "from server import main as sm\n"
        "import uvicorn\n"
        f"uvicorn.run(sm.app, host='127.0.0.1', port={TEST_PORT}, log_level='warning')\n"
    )
    return patched


async def _wait_for_ready(timeout: float = 30.0) -> bool:
    """Poll /health until the server responds or timeout elapses."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        await asyncio.sleep(0.3)
    return False


# ---------- WebSocket helpers ----------

class WSClient:
    """Thin wrapper that collects typed messages from a WebSocket."""

    def __init__(self, ws):
        self.ws = ws
        self.ticks: list[dict] = []
        self.chat_responses: list[dict] = []
        self.other: list[dict] = []
        self.init: dict | None = None
        self._reader_task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._reader_task = asyncio.create_task(self._reader())
        # Wait for the init message before returning
        for _ in range(50):
            if self.init:
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("No init message received")

    async def stop(self) -> None:
        self._stop.set()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _reader(self) -> None:
        try:
            while not self._stop.is_set():
                raw = await self.ws.recv()
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "tick":
                    self.ticks.append({"t": time.monotonic(), "msg": msg})
                elif t == "chat_response":
                    self.chat_responses.append({"t": time.monotonic(), "msg": msg})
                elif t == "init":
                    self.init = msg
                else:
                    self.other.append({"t": time.monotonic(), "msg": msg})
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def send(self, msg: dict) -> None:
        await self.ws.send(json.dumps(msg))

    def latest_player(self) -> dict | None:
        if not self.ticks:
            return self.init.get("player") if self.init else None
        return self.ticks[-1]["msg"].get("player")


def _nearest_npc(npcs: list[dict], player: dict) -> dict:
    """Pick the NPC closest to the player."""
    real = [n for n in npcs if n.get("npc_id") != "player"]
    real.sort(key=lambda n: abs(n["x"] - player["x"]) + abs(n["z"] - player["z"]))
    return real[0]


def _latest_npc(client: WSClient, npc_id: str) -> dict | None:
    """Current position of an NPC from the last tick, or None."""
    if not client.ticks:
        if client.init:
            for n in client.init.get("npcs", []):
                if n.get("npc_id") == npc_id:
                    return n
        return None
    for n in client.ticks[-1]["msg"].get("npcs", []):
        if n.get("npc_id") == npc_id:
            return n
    return None


async def _pick_and_reach_stable_npc(client: WSClient, max_steps: int = 120) -> dict:
    """Pick an NPC and chase it until the player is within 2 tiles.

    NPCs move on their own schedule, so we re-read the target's live
    position on each step rather than using a stale snapshot. Falls
    back to the next-nearest NPC if the current one keeps moving away.
    """
    init_player = client.latest_player()
    assert init_player, "No player state from init"

    # Build a preference list of NPCs by distance (closest first).
    candidates = sorted(
        [n for n in client.init.get("npcs", []) if n.get("npc_id") != "player"],
        key=lambda n: abs(n["x"] - init_player["x"]) + abs(n["z"] - init_player["z"]),
    )

    for candidate in candidates[:4]:  # try up to 4 closest NPCs
        npc_id = candidate["npc_id"]
        logger.info("Trying to reach %s (%s)", candidate["name"], npc_id)

        for step in range(max_steps):
            player = client.latest_player()
            target = _latest_npc(client, npc_id) or candidate
            if player is None:
                await asyncio.sleep(0.1)
                continue
            dx = target["x"] - player["x"]
            dz = target["z"] - player["z"]
            manhattan = abs(dx) + abs(dz)
            # Stop within chat range (<=2 tiles) but not on top of the NPC:
            # overlapping tiles triggers resolve_overlaps which can fight
            # with the test's own movement commands. 1–2 tiles away is
            # the sweet spot.
            if 1 <= manhattan <= 2:
                return target

            if abs(dx) > abs(dz):
                direction = "east" if dx > 0 else "west"
            else:
                direction = "south" if dz > 0 else "north"
            await client.send({"type": "player_move", "direction": direction})
            await asyncio.sleep(0.12)

        logger.warning("Could not catch %s in %d steps — trying next candidate", npc_id, max_steps)

    raise AssertionError(
        f"Could not reach any of the {len(candidates[:4])} closest NPCs — "
        "player movement may be broken"
    )


# ---------- Tests ----------

async def test_chat_responds_without_blocking_input(client: WSClient) -> None:
    """The crucial regression test: moves sent during a chat call must
    still advance the player. If the WS receive loop is blocked by the
    LLM call, the player sits frozen until the chat response arrives."""
    target = await _pick_and_reach_stable_npc(client)
    logger.info("Target NPC %s at (%s, %s)", target["name"], target["x"], target["z"])
    start_player = client.latest_player()
    assert start_player, "No player state after movement"
    start_x = start_player["x"]
    start_z = start_player["z"]
    logger.info("Player adjacent at (%s, %s)", start_x, start_z)

    # Decide a move direction that takes us AWAY from the NPC, so we
    # don't bump into them and stall. Tile motion is integer-based, so
    # moving away puts us clearly at a different tile if the move lands.
    away_direction = _direction_away(start_player, target)

    # Snapshot chat start time.
    chat_start = time.monotonic()
    pre_ticks = len(client.ticks)
    pre_chat_responses = len(client.chat_responses)

    await client.send({
        "type": "player_chat",
        "npc_id": target["npc_id"],
        "message": "Hi",
    })

    # Immediately spam moves. With the fix these should ALL be processed
    # concurrently with the chat; without the fix none are processed
    # until the chat response comes back.
    for _ in range(15):
        await client.send({"type": "player_move", "direction": away_direction})
        await asyncio.sleep(0.05)

    # Wait up to 60s for the chat response. MockProvider responds in ms.
    deadline = chat_start + 60
    while time.monotonic() < deadline and len(client.chat_responses) == pre_chat_responses:
        await asyncio.sleep(0.1)

    assert len(client.chat_responses) > pre_chat_responses, (
        "No chat_response received within 60s — chat handler broken"
    )
    resp = client.chat_responses[-1]["msg"]
    # The response is either (a) a normal NPC reply, or (b) an
    # out-of-range error — which here is GOOD NEWS: it means the moves
    # we fired landed during the LLM call, proving the receive loop is
    # not blocked. Either way, we ensure it isn't an echo of our input.
    if resp.get("message"):
        assert resp["message"].strip().lower() != "hi", (
            f"NPC echoed player's message: {resp['message']}"
        )
    else:
        err = resp.get("error", "")
        assert "too far" in err.lower(), (
            f"Unexpected chat error: {err!r}"
        )

    # Player must have actually moved during the chat roundtrip.
    end_player = client.latest_player()
    assert end_player, "No player state at end"
    end_x = end_player["x"]
    end_z = end_player["z"]
    moved_during = (end_x, end_z) != (start_x, start_z)

    # Tick stream must have kept flowing.
    new_ticks = len(client.ticks) - pre_ticks
    elapsed = time.monotonic() - chat_start
    expected_ticks = int(elapsed * 2)  # conservative: 2 Hz minimum (actual is 4Hz)

    logger.info(
        "Chat roundtrip %.2fs, %d ticks (expected >= %d), player %s → %s (moved=%s)",
        elapsed, new_ticks, expected_ticks,
        (start_x, start_z), (end_x, end_z), moved_during,
    )

    assert new_ticks >= expected_ticks, (
        f"Only {new_ticks} ticks in {elapsed:.1f}s (expected >= {expected_ticks}) — "
        "tick broadcaster stalled during chat"
    )
    assert moved_during, (
        f"Player stayed at {(start_x, start_z)} during chat — "
        "WebSocket receive loop was blocked by the LLM call. "
        f"(Sent 15 moves toward {away_direction} during {elapsed:.1f}s chat.)"
    )


def _direction_away(player: dict, target: dict) -> str:
    dx = player["x"] - target["x"]
    dz = player["z"] - target["z"]
    if abs(dx) >= abs(dz):
        return "east" if dx >= 0 else "west"
    return "south" if dz >= 0 else "north"


def _directions_away(player: dict, target: dict) -> list[str]:
    """Return the four cardinal directions, best-first (most away from target).

    Walking continues until one of them actually clears range —
    defeats the "player is pinned against a wall / river / building
    in the chosen direction" class of flakiness."""
    primary = _direction_away(player, target)
    opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
    secondary = (
        "south" if primary in ("east", "west") and player["z"] >= target["z"]
        else "north" if primary in ("east", "west")
        else "east" if player["x"] >= target["x"]
        else "west"
    )
    rest = [d for d in ("north", "south", "east", "west")
            if d not in (primary, secondary, opposite.get(primary))]
    return [primary, secondary] + rest + [opposite[primary]]


async def test_sim_responsive_after_chat_close(client: WSClient) -> None:
    """After closing the chat, the player must be able to move again."""
    target = await _pick_and_reach_stable_npc(client)

    await client.send({
        "type": "player_chat",
        "npc_id": target["npc_id"],
        "message": "Hello",
    })
    # Wait for chat response to arrive.
    t0 = time.monotonic()
    pre_count = len(client.chat_responses)
    while time.monotonic() - t0 < 60 and len(client.chat_responses) == pre_count:
        await asyncio.sleep(0.1)
    assert len(client.chat_responses) > pre_count, "No response to Hello"

    # Close the chat.
    await client.send({"type": "player_chat_close", "npc_id": target["npc_id"]})
    await asyncio.sleep(0.3)

    # Snapshot player, then try to move. Try all four cardinal directions
    # so a single impassable tile (e.g. building wall, map edge) can't
    # produce a false "player can't move" failure.
    before = client.latest_player()
    assert before, "No player state"
    before_pos = (before["x"], before["z"])

    directions = ["north", "south", "east", "west"]
    moved = False
    for direction in directions:
        for _ in range(6):
            await client.send({"type": "player_move", "direction": direction})
            await asyncio.sleep(0.12)
        current = client.latest_player()
        if (current["x"], current["z"]) != before_pos:
            moved = True
            logger.info("Post-chat move %s: %s → %s", direction, before_pos, (current["x"], current["z"]))
            break

    assert moved, (
        f"Player could not move in any direction after chat close "
        f"(stayed at {before_pos} after trying {directions})"
    )


async def test_multiple_chat_exchanges_never_echo_player(client: WSClient) -> None:
    """Send several messages — NPC's replies must never be the player's own text."""
    target = await _pick_and_reach_stable_npc(client)

    unique_messages = [
        "UNIQUE_TEST_AAA",
        "UNIQUE_TEST_BBB",
        "UNIQUE_TEST_CCC",
    ]
    pre_count = len(client.chat_responses)
    for msg in unique_messages:
        await client.send({
            "type": "player_chat",
            "npc_id": target["npc_id"],
            "message": msg,
        })
        # Wait for a response.
        t0 = time.monotonic()
        target_count = pre_count + 1
        while time.monotonic() - t0 < 60 and len(client.chat_responses) < target_count:
            await asyncio.sleep(0.1)
        assert len(client.chat_responses) >= target_count, f"No response to {msg}"
        pre_count = target_count

        resp = client.chat_responses[-1]["msg"]
        reply = resp.get("message") or ""
        assert msg not in reply, (
            f"NPC echoed unique player message. Sent {msg!r}, received {reply!r}"
        )


async def test_tick_broadcast_continuous(client: WSClient) -> None:
    """Tick broadcasts must arrive at ~4 Hz regardless of chat activity."""
    # Measure tick rate over 2 seconds without any chat traffic.
    pre = len(client.ticks)
    await asyncio.sleep(2.0)
    baseline_ticks = len(client.ticks) - pre
    assert baseline_ticks >= 4, (
        f"Baseline tick rate too low: {baseline_ticks} ticks in 2s (expected >= 4)"
    )
    logger.info("Baseline: %d ticks in 2s", baseline_ticks)

    # Now fire a chat and measure tick rate during the 5-second window.
    target = await _pick_and_reach_stable_npc(client)
    await client.send({
        "type": "player_chat",
        "npc_id": target["npc_id"],
        "message": "Talk",
    })
    pre = len(client.ticks)
    await asyncio.sleep(5.0)
    during_ticks = len(client.ticks) - pre
    assert during_ticks >= 8, (
        f"Tick rate collapsed during chat: {during_ticks} ticks in 5s (expected >= 8)"
    )
    logger.info("During chat: %d ticks in 5s", during_ticks)


# ---------- Runner ----------

async def test_walking_out_of_range_ends_chat(client: WSClient) -> None:
    """If the player walks more than interaction_radius tiles away from
    their chat partner, the server must force-close the conversation
    and send a chat_response with ended=True. Mirrors the 3D sim rule:
    if you can't press E to start a chat, you can't sustain one either."""
    target = await _pick_and_reach_stable_npc(client)

    await client.send({
        "type": "player_chat",
        "npc_id": target["npc_id"],
        "message": "Hello",
    })
    # Wait for the initial reply.
    t0 = time.monotonic()
    pre_count = len(client.chat_responses)
    while time.monotonic() - t0 < 30 and len(client.chat_responses) == pre_count:
        await asyncio.sleep(0.1)
    assert len(client.chat_responses) > pre_count, "No reply to initial Hello"

    # Snapshot the response-count threshold BEFORE we start walking so
    # the proximity-end response (which may arrive mid-walk) isn't
    # counted as "prior".
    pre_responses = len(client.chat_responses)

    # Walk far away from the target. interaction_radius is 3 tiles —
    # move at least 4 tiles away to be unambiguously out of range.
    # Try every cardinal direction so the test isn't defeated by map
    # geometry (a wall, building or river in the chosen direction
    # would block the player and the proximity check would never fire).
    for direction in _directions_away(client.latest_player(), target):
        start_pos = client.latest_player()
        for _ in range(10):
            await client.send({"type": "player_move", "direction": direction})
            await asyncio.sleep(0.12)
        moved_pos = client.latest_player()
        dx = abs(moved_pos["x"] - target["x"]) + abs(moved_pos["z"] - target["z"])
        if dx > 4:
            logger.info("Walked %s from %s to (%s,%s), dist=%.1f",
                        direction, (start_pos["x"], start_pos["z"]),
                        moved_pos["x"], moved_pos["z"], dx)
            break

    # Within ~3 seconds after the walk, the server must have sent an
    # ended=True chat_response for the target NPC at some point after
    # pre_responses (could be during or just after the walk).
    deadline = time.monotonic() + 3
    end_response = None
    while time.monotonic() < deadline and end_response is None:
        for r in client.chat_responses[pre_responses:]:
            msg = r["msg"]
            if msg.get("ended") and msg.get("npc_id") == target["npc_id"]:
                end_response = msg
                break
        if end_response:
            break
        await asyncio.sleep(0.15)

    total_new = len(client.chat_responses) - pre_responses
    assert end_response is not None, (
        "Server did not send an ended=True chat_response after player "
        f"walked out of range. pre_responses={pre_responses} "
        f"total_now={len(client.chat_responses)} "
        f"all_responses={[r['msg'] for r in client.chat_responses]} "
        f"total_ticks={len(client.ticks)}"
    )
    logger.info("Proximity end received: %s", end_response.get("error") or end_response.get("message"))


async def test_player_trail_does_not_persist_across_ticks(client: WSClient) -> None:
    """After any overlap nudge populates the player's _tick_trail, the
    trail must be cleared on subsequent ticks so the client receives
    fresh positional updates.

    Regression: once _tick_trail was populated for the player (e.g. by
    resolve_overlaps nudging the avatar off an NPC's tile), it was
    never cleared on ticks where the player wasn't walking along a
    path. The client's trail branch kept appending the same stale
    waypoints forever, and the avatar+ring froze far behind the real
    server position."""
    # Collect ticks for ~3 seconds and check how many times the
    # player.trail field is non-empty. Since the player typically
    # isn't on top of an NPC, most ticks should have empty trail.
    # The key invariant: trail must NEVER be the SAME non-empty
    # sequence on consecutive ticks — that's the stale-trail bug.
    pre = len(client.ticks)
    await asyncio.sleep(3.0)
    new = client.ticks[pre:]
    assert len(new) >= 6, f"Too few ticks: {len(new)}"

    # Find the player entry in each tick and extract the trail
    trails = []
    for t in new:
        for npc in t["msg"].get("npcs", []):
            if npc.get("npc_id") == "player":
                trails.append(tuple(tuple(w) for w in (npc.get("trail") or [])))
                break

    # Consecutive identical non-empty trails indicate the stale-trail bug.
    max_repeat = 1
    run = 1
    prev = None
    for tr in trails:
        if tr and tr == prev:
            run += 1
            max_repeat = max(max_repeat, run)
        else:
            run = 1
        prev = tr
    assert max_repeat < 3, (
        f"Player trail stuck across {max_repeat} consecutive ticks "
        f"(stale-trail bug). Trails: {trails}"
    )


async def test_stale_response_does_not_leak_into_new_chat(client: WSClient) -> None:
    """Simulate the Dara-replying-in-Bran-window bug: start a chat with
    one NPC, immediately switch to a different NPC, and verify that the
    first NPC's reply never arrives (or arrives tagged so the client
    would filter it). Server-side guard in _handle_player_chat should
    drop the stale response before send."""
    first = await _pick_and_reach_stable_npc(client)
    logger.info("First target: %s", first["name"])

    # Send a message to the first NPC — don't wait for response.
    await client.send({
        "type": "player_chat",
        "npc_id": first["npc_id"],
        "message": "FIRST_CHAT_MARKER",
    })
    await asyncio.sleep(0.1)

    # Explicitly close the chat (like the user closing the panel).
    await client.send({"type": "player_chat_close", "npc_id": first["npc_id"]})
    await asyncio.sleep(0.2)

    # Find a different NPC and open a chat with them.
    init_npcs = client.init.get("npcs", [])
    others = [
        n for n in init_npcs
        if n.get("npc_id") not in ("player", first["npc_id"])
    ]
    # Move toward the nearest other NPC.
    player = client.latest_player()
    others.sort(key=lambda n: abs(n["x"] - player["x"]) + abs(n["z"] - player["z"]))

    # Chase one of them.
    second = None
    for candidate in others[:3]:
        npc_id = candidate["npc_id"]
        for _ in range(80):
            p = client.latest_player()
            live = _latest_npc(client, npc_id) or candidate
            dx = live["x"] - p["x"]
            dz = live["z"] - p["z"]
            if abs(dx) + abs(dz) <= 2:
                second = live
                break
            direction = "east" if dx > 0 and abs(dx) > abs(dz) else (
                "west" if dx < 0 and abs(dx) > abs(dz) else (
                    "south" if dz > 0 else "north"
                )
            )
            await client.send({"type": "player_move", "direction": direction})
            await asyncio.sleep(0.12)
        if second:
            break
    assert second is not None, "Could not reach a second NPC"
    logger.info("Second target: %s", second["name"])

    pre_responses = len(client.chat_responses)
    await client.send({
        "type": "player_chat",
        "npc_id": second["npc_id"],
        "message": "SECOND_CHAT_MARKER",
    })

    # Collect responses for 8 seconds.
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        await asyncio.sleep(0.2)

    new_responses = [r["msg"] for r in client.chat_responses[pre_responses:]]
    logger.info("Got %d responses after switch", len(new_responses))

    # ANY response tagged with the first NPC's id that arrives after the
    # switch is a leak.
    leaks = [r for r in new_responses if r.get("npc_id") == first["npc_id"]
             and not r.get("ended")]
    assert not leaks, (
        f"Stale chat_response leaked from {first['name']} into {second['name']}'s "
        f"conversation: {leaks}"
    )

    # Bonus: at least one response must be tagged with the second NPC's id.
    good = [r for r in new_responses if r.get("npc_id") == second["npc_id"]]
    assert good, (
        f"No response received from new target {second['name']} "
        f"within 8s. New responses: {new_responses}"
    )


async def test_thinking_level_toggle_roundtrips(client: WSClient) -> None:
    """Client sets the thinking level over WS; server confirms and applies.

    Exercises the new HUD dropdown end-to-end: the client sends
    {type: set_thinking_level, level: "fast"}; the server updates the
    provider's global ThinkingProfile and echoes a confirmation
    message with the resolved profile fields."""
    # Clear any prior 'other' (init etc.) so we can isolate the ack.
    pre_other = len(client.other)

    await client.send({"type": "set_thinking_level", "level": "fast"})

    deadline = time.monotonic() + 5
    ack = None
    while time.monotonic() < deadline:
        for entry in client.other[pre_other:]:
            msg = entry["msg"]
            if msg.get("type") == "thinking_level_set":
                ack = msg
                break
        if ack:
            break
        await asyncio.sleep(0.1)

    assert ack is not None, "Server did not acknowledge set_thinking_level"
    assert ack["level"] == "fast"
    # FAST never thinks — thinking_purposes list must be empty.
    assert ack["thinking_purposes"] == []
    # Budgets must match the FAST preset (keep the test coupled to
    # the presets so a refactor that silently changes them is flagged).
    from core.npc.cognition.thinking import FAST
    assert ack["thinking_budget"] == FAST.thinking_budget
    assert ack["quick_budget"] == FAST.quick_budget

    # Round-trip back to DEEP to confirm both directions work.
    pre_other = len(client.other)
    await client.send({"type": "set_thinking_level", "level": "deep"})
    deadline = time.monotonic() + 5
    ack2 = None
    while time.monotonic() < deadline and ack2 is None:
        for entry in client.other[pre_other:]:
            if entry["msg"].get("type") == "thinking_level_set":
                ack2 = entry["msg"]
                break
        await asyncio.sleep(0.1)
    assert ack2 is not None and ack2["level"] == "deep"
    assert "conversation" in ack2["thinking_purposes"]


ALL_TESTS = [
    ("chat responds without blocking input", test_chat_responds_without_blocking_input),
    ("sim responsive after chat close", test_sim_responsive_after_chat_close),
    ("multiple chats never echo player", test_multiple_chat_exchanges_never_echo_player),
    ("tick broadcast continuous", test_tick_broadcast_continuous),
    ("walking out of range ends chat", test_walking_out_of_range_ends_chat),
    ("player trail not stale across ticks", test_player_trail_does_not_persist_across_ticks),
    ("stale response does not leak", test_stale_response_does_not_leak_into_new_chat),
    ("thinking level toggle round-trips", test_thinking_level_toggle_roundtrips),
]


async def _run_one_test(name: str, test) -> tuple[bool, str]:
    """Boot a fresh server, run one test against it, shut it down.

    Isolates each test so state drift between them (NPCs wandering,
    schedules advancing) can't cause later tests to flake.
    """
    server_proc, log_path = _start_server()
    try:
        if not await _wait_for_ready():
            return False, f"Server did not become ready. Log tail:\n{log_path.read_text()[-1500:]}"
        try:
            async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
                client = WSClient(ws)
                await client.start()
                try:
                    await test(client)
                    return True, ""
                finally:
                    await client.stop()
        except AssertionError as e:
            return False, str(e)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        # Give the port a moment to release before the next test binds it.
        await asyncio.sleep(0.5)


async def run_all() -> int:
    results: list[tuple[str, bool, str]] = []
    for name, test in ALL_TESTS:
        print(f"\n{'='*60}\nRUNNING: {name}\n{'='*60}")
        ok, err = await _run_one_test(name, test)
        results.append((name, ok, err))
        if ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}\n    {err[:500]}")

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} passed")
    for name, ok, err in results:
        if not ok:
            print(f"  FAIL: {name}")
            print(f"       {err[:400]}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all()))
