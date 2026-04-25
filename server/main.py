"""
Smallville 2 — FastAPI server.

Server-authoritative architecture: all game logic lives in core/.
This server handles WebSocket connections, state sync, and static file serving.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from core.world.generator import WorldConfig, generate_world
from core.world.prompt_gen import TownPromptGenerator
from core.world.prompt_gen.features import TerrainFeature
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager
from core.npc.llm_client import ClaudeProvider, MockProvider
from core.npc.mistral_provider import MistralProvider
from core.npc.gemma_provider import GemmaProvider, ollama_available
from core.npc.cognition.router import CognitionRouter, CognitionPolicy
from core.npc.cognition.converse import (
    initiate_conversation, continue_conversation,
    start_player_conversation, end_conversation,
    _active_conversations,
)
from core.memory.manager import MemoryManager
from core.player.player_agent import PlayerAgent, find_player_spawn

logger = logging.getLogger(__name__)

app = FastAPI(title="Smallville 2", version="0.2.0")

# Serve client static files
CLIENT_DIR = Path(__file__).parent.parent / "client"
app.mount("/static", StaticFiles(directory=str(CLIENT_DIR)), name="static")


# ---------- Game State ----------

game_clock = GameClock()

# Town prompt system — generates world from natural language description
import asyncio as _asyncio

TOWN_DESCRIPTION = "A cozy riverside town with forest and two bridges"

_prompt_gen = TownPromptGenerator(seed=42)
try:
    _loop = _asyncio.get_running_loop()
except RuntimeError:
    _loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(_loop)
_spec = _loop.run_until_complete(
    _prompt_gen.generate_config(TOWN_DESCRIPTION)
)
world_config = _spec.config
town_name = _spec.town_name
grid, buildings = generate_world(world_config, features=_spec.features)
world_data = grid.to_dict()  # cached serialisation — rebuilt on world change
logger.info("Generated town '%s' from prompt: %s", town_name, TOWN_DESCRIPTION)

# LLM provider selection: Gemma (local) > Claude > Mistral > Mock
if ollama_available():
    llm_provider = GemmaProvider()
    logger.info("Using local Gemma via Ollama for NPC cognition")
elif os.environ.get("ANTHROPIC_API_KEY"):
    llm_provider = ClaudeProvider()
    logger.info("Using Claude API for NPC cognition")
elif os.environ.get("MISTRAL_API_KEY"):
    llm_provider = MistralProvider()
    logger.info("Using Mistral API for NPC cognition")
else:
    llm_provider = MockProvider()
    logger.info("No API key found — using mock LLM provider")

memory_manager = MemoryManager(llm=llm_provider)

# Set DETERMINISTIC=1 env var to use template schedules only (no LLM)
_deterministic_mode = os.environ.get("DETERMINISTIC", "").strip() in ("1", "true", "yes")
if _deterministic_mode:
    logger.info("Deterministic mode enabled — using template schedules only")

npc_manager = NPCManager(
    grid=grid,
    buildings=buildings,
    llm=llm_provider,
    seed=42,
    memory=memory_manager,
    deterministic=_deterministic_mode,
)
npc_manager.spawn_population(world_config.population)

# Player agent — spawns near town centre
_spawn_x, _spawn_z = find_player_spawn(grid, buildings)
player_agent = PlayerAgent.create(
    name="Traveller",
    spawn_x=_spawn_x,
    spawn_z=_spawn_z,
)
# Register player NPC with the NPC manager so NPCs can perceive/interact
# Player gets an empty schedule — movement is driven by input, not the planner
player_agent.npc.has_custom_schedule = True
player_agent.npc.daily_schedule = []
npc_manager.npcs.append(player_agent.npc)
npc_manager._npc_map[player_agent.npc_id] = player_agent.npc
# Wire player agent into manager for autonomy toggle
npc_manager.player_agent = player_agent
# Set focus to player position for tier assignment
npc_manager.set_focus(round(_spawn_x), round(_spawn_z))
logger.info("Player spawned at (%.0f, %.0f)", _spawn_x, _spawn_z)


# ---------- Connection manager ----------

class ConnectionManager:
    """Manages active WebSocket connections.

    Each connection gets its own asyncio.Lock for sends, so that
    concurrent writers (tick broadcaster, chat handler, trade handler)
    can't interleave WebSocket frames. Without this, the background
    chat task spawned per-player-chat could write partial frames while
    the tick broadcaster is mid-send — producing malformed traffic.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._send_locks: dict[int, asyncio.Lock] = {}
        # Per-connection chat lock: serialises chat handling so a new
        # player_chat message doesn't race with an in-flight LLM call.
        self._chat_locks: dict[int, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self._send_locks[id(websocket)] = asyncio.Lock()
        self._chat_locks[id(websocket)] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._send_locks.pop(id(websocket), None)
        self._chat_locks.pop(id(websocket), None)

    def chat_lock(self, websocket: WebSocket) -> asyncio.Lock:
        """Return the per-connection lock for chat processing."""
        lock = self._chat_locks.get(id(websocket))
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[id(websocket)] = lock
        return lock

    async def send_to(self, websocket: WebSocket, message: dict) -> bool:
        """Send a single message under the per-connection lock."""
        lock = self._send_locks.get(id(websocket))
        if lock is None:
            return False
        async with lock:
            try:
                await websocket.send_json(message)
                return True
            except Exception:
                return False

    async def broadcast(self, message: dict):
        dead = []
        for connection in list(self.active_connections):
            ok = await self.send_to(connection, message)
            if not ok:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


# ---------- Game Loop ----------

TICK_INTERVAL = 0.25  # real seconds between movement ticks (4 ticks/sec)


async def movement_loop():
    """Fast loop — movement, departures, overlaps at steady 4Hz.

    Never blocks on LLM calls. This is what clients see.
    """
    last_time = time.monotonic()
    while True:
        await asyncio.sleep(TICK_INTERVAL)
        now = time.monotonic()
        wall_delta = now - last_time
        last_time = now

        events = game_clock.tick(wall_delta)
        move_delta = min(wall_delta, TICK_INTERVAL)

        # Player movement (server-authoritative)
        player_agent.movement_tick(grid, move_delta)
        # Update focus to player position for tier assignment
        npc_manager.set_focus(player_agent.tile_x, player_agent.tile_z)

        npc_state = npc_manager.movement_tick(game_clock, move_delta)

        # Proximity check: if the player walked out of chat range, close
        # the conversation so responses can't arrive after the player has
        # moved on to someone else (or nobody).
        out_of_range = _player_chat_out_of_range()

        tick_msg = {
            "type": "tick",
            "time": game_clock.to_dict(),
            "npcs": npc_state.get("npcs", []),
            "player": player_agent.to_dict(),
        }
        if events:
            tick_msg["events"] = events
        if npc_state.get("conversations"):
            tick_msg["conversations"] = npc_state["conversations"]
        # Always include the town agenda — it's small (<= a few goals)
        # and the HUD needs every update so it can fade completed goals
        # and tick down progress bars smoothly.
        agenda = npc_state.get("town_agenda")
        if agenda:
            tick_msg["town_agenda"] = agenda

        # Phase A.5 — drain any memory_formed events accumulated since
        # the last tick. Only notable memories (importance ≥ threshold)
        # are here; the client renders a short-lived sparkle sprite
        # above each matching NPC.
        memory_events = memory_manager.drain_memory_events()
        if memory_events:
            tick_msg["memory_events"] = memory_events

        await manager.broadcast(tick_msg)
        if out_of_range:
            await manager.broadcast(out_of_range)


async def cognition_loop():
    """Slow loop — schedules, perception, conversations, reflections.

    Runs independently; may block for seconds on LLM calls.
    Movement continues uninterrupted via movement_loop.
    """
    while True:
        await asyncio.sleep(1.0)  # cognition checks every ~1s
        try:
            await npc_manager.cognition_tick(game_clock, 1.0)
        except Exception:
            logger.exception("Cognition tick error")


@app.on_event("startup")
async def startup():
    asyncio.create_task(movement_loop())
    asyncio.create_task(cognition_loop())


# ---------- Startup assertion ----------

assert len(npc_manager.npcs) > 0, (
    f"NPC population is empty after spawn_population({world_config.population}). "
    "Check building generation and occupation assignment."
)


# ---------- Routes ----------

@app.get("/")
async def root():
    """Serve the main client page."""
    return FileResponse(str(CLIENT_DIR / "index.html"))


@app.get("/health")
async def health():
    """System health check — verifies all subsystems are loaded."""
    return {
        "status": "ok",
        "version": app.version,
        "npcs": len(npc_manager.npcs),
        "buildings": len(buildings),
        "grid": f"{grid.width}x{grid.height}",
        "clock": game_clock.time_string,
        "day": game_clock.day,
        "phase": game_clock.phase.value,
        "llm_provider": type(llm_provider).__name__,
    }


@app.get("/api/memory/stats")
async def memory_stats():
    """Memory system overview stats."""
    return {
        "stats": memory_manager.get_stats(),
        "activity": memory_manager.get_recent_activity(limit=20),
    }


@app.get("/api/memory/npc/{npc_id}")
async def npc_memory(
    npc_id: str,
    limit: int = 20,
    include_compacted: bool = False,
):
    """Full memory dump for a specific NPC.

    Query params:
    - `limit=0` returns every memory (default 20).
    - `include_compacted=true` includes tombstoned raw memories
      alongside their summaries.
    """
    npc = npc_manager.get_npc(npc_id)
    if not npc:
        return {"error": f"NPC {npc_id} not found"}
    return memory_manager.get_npc_memory_summary(
        npc_id, limit=limit, include_compacted=include_compacted,
    )


@app.get("/api/memory/npcs")
async def memory_npc_list():
    """List all NPCs with their memory counts."""
    result = []
    for npc in npc_manager.npcs:
        result.append({
            "npc_id": npc.npc_id,
            "name": npc.name,
            "occupation": npc.occupation,
            "cognition_tier": npc.cognition_tier,
            "episodic_count": memory_manager.episodic.count(npc.npc_id),
        })
    return result


@app.get("/api/memory/dump")
async def memory_dump(
    limit: int = 0,
    include_compacted: bool = False,
):
    """Full memory dump for EVERY NPC (diagnostic).

    Returns the game clock, then each NPC's identity, physical
    state, goals, facts, and complete episodic history in the
    shape `{"day": N, "time": "HH:MM", "npcs": {npc_id: {...}}}`.

    Defaults: `limit=0` (no cap). Pass `limit=N` for a recent-N
    slice, `include_compacted=true` to see tombstoned originals.
    Large payload — meant for an out-of-band dump script, not
    the web UI.
    """
    dump: dict[str, Any] = {}
    for npc in npc_manager.npcs:
        summary = memory_manager.get_npc_memory_summary(
            npc.npc_id, limit=limit,
            include_compacted=include_compacted,
        )
        summary["name"] = npc.name
        summary["occupation"] = npc.occupation
        summary["age"] = getattr(npc, "age", None)
        summary["cognition_tier"] = npc.cognition_tier
        summary["position"] = {"x": npc.x, "z": npc.z}
        summary["home"] = {"x": npc.home_x, "z": npc.home_z}
        summary["activity"] = (
            npc.activity.value if npc.activity else "none"
        )
        summary["current_action"] = npc.current_action_description
        dump[npc.npc_id] = summary
    return {
        "day": game_clock.day,
        "time": game_clock.time_string,
        "phase": game_clock.phase.value,
        "npcs": dump,
    }


@app.get("/api/debug/npcs")
async def debug_npcs():
    """Full NPC diagnostic state — positions, paths, home/work, schedule."""
    result = []
    for npc in npc_manager.npcs:
        schedule_info = []
        for entry in (npc.daily_schedule or []):
            schedule_info.append({
                "slot": entry.slot,
                "activity": entry.activity,
                "location": entry.location,
                "priority": entry.priority,
            })
        result.append({
            "npc_id": npc.npc_id,
            "name": npc.name,
            "occupation": npc.occupation,
            "x": npc.x,
            "z": npc.z,
            "home_x": npc.home_x,
            "home_z": npc.home_z,
            "work_x": getattr(npc, "work_x", None),
            "work_z": getattr(npc, "work_z", None),
            "activity": npc.activity.value if npc.activity else "none",
            "current_action": npc.current_action_description,
            "cognition_tier": npc.cognition_tier,
            "path_length": len(npc.current_path) if npc.current_path else 0,
            "path_target": list(npc.current_path[-1]) if npc.current_path else None,
            "conversation_partner": npc.conversation_partner,
            "schedule": schedule_info,
            "schedule_day": getattr(npc, "schedule_day", None),
            "schedule_index": getattr(npc, "schedule_index", None),
            "action_start_minutes": getattr(npc, "action_start_minutes", None),
        })
    return {
        "time": game_clock.time_string,
        "day": game_clock.day,
        "phase": game_clock.phase.value,
        "npcs": result,
    }


@app.post("/api/npc/{npc_id}/schedule")
async def assign_npc_schedule(npc_id: str, body: dict):
    """Assign a custom schedule to an NPC."""
    entries = body.get("entries", [])
    ok, msg = npc_manager.assign_custom_schedule(npc_id, entries)
    status = 200 if ok else 400
    return JSONResponse(
        content={"success": ok, "message": msg, "npc_id": npc_id},
        status_code=status,
    )


@app.delete("/api/npc/{npc_id}/schedule")
async def clear_npc_schedule(npc_id: str):
    """Clear a custom schedule, reverting to template."""
    ok, msg = npc_manager.clear_custom_schedule(npc_id)
    status = 200 if ok else 400
    return JSONResponse(
        content={"success": ok, "message": msg, "npc_id": npc_id},
        status_code=status,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for game state sync."""
    await manager.connect(websocket)
    try:
        # Send initial state on connect (includes NPC + player data)
        npc_state = npc_manager.get_state()
        await websocket.send_json({
            "type": "init",
            "message": "Connected to Smallville 2",
            "world": world_data,
            "time": game_clock.to_dict(),
            "buildings": [
                {
                    "name": b.name,
                    "type": b.building_type,
                    "x": b.x, "z": b.z,
                    "width": b.width, "height": b.height,
                    "door_x": b.door_x, "door_z": b.door_z,
                }
                for b in buildings
            ],
            "npcs": npc_state.get("npcs", []),
            "player": player_agent.to_dict(),
        })

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            # Chat processing calls the LLM and can take 20–60s.
            # Spawning it as a task is CRITICAL: if we awaited inline,
            # the WebSocket receive loop would block and the player could
            # not move or send any other input for the duration of the
            # LLM call. This was the root cause of "sim becomes
            # unresponsive after chat" — the receive loop was stuck.
            if msg_type in ("player_chat", "chat"):
                asyncio.create_task(
                    _chat_task(websocket, message),
                    name=f"chat-{id(websocket)}",
                )
                continue

            try:
                response = await handle_message(message)
                if response:
                    await manager.send_to(websocket, response)
            except Exception:
                logger.exception("Error handling message: %s", msg_type)
                await manager.send_to(websocket, {
                    "type": "error",
                    "message": "Server error processing request",
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        logger.exception("WebSocket handler crashed")
        manager.disconnect(websocket)


async def _chat_task(websocket: WebSocket, message: dict) -> None:
    """Background task: handle a player chat without blocking the receive loop.

    Serialised per-connection via manager.chat_lock so rapid-fire messages
    from the same player are processed in order without overlapping LLM calls.
    """
    lock = manager.chat_lock(websocket)
    async with lock:
        try:
            response = await _handle_player_chat(message)
            if response:
                await manager.send_to(websocket, response)
        except Exception:
            logger.exception("Chat handler failed")
            await manager.send_to(websocket, {
                "type": "chat_response",
                "error": "Server error processing chat",
            })


async def handle_message(message: dict) -> dict | None:
    """Route incoming WebSocket messages to appropriate handlers."""
    msg_type = message.get("type")

    if msg_type == "ping":
        return {"type": "pong"}

    # --- Player movement (server-authoritative) ---
    if msg_type == "player_move":
        direction = message.get("direction")
        player_agent.set_move_direction(direction)
        return None  # Position sent via tick broadcast

    # --- Player trade with NPC ---
    if msg_type == "player_trade":
        return await _handle_player_trade(message)

    # --- Player closes chat window (lets server clean up conv state) ---
    if msg_type == "player_chat_close":
        _close_player_chat()
        return None

    # --- Thinking-level toggle: set the global LLM profile ---
    if msg_type == "set_thinking_level":
        return _set_thinking_level(message.get("level", "balanced"))

    # Legacy move stub
    if msg_type == "move":
        player_agent.set_move_direction(message.get("direction"))
        return None

    # Chat messages are handled in the outer WebSocket loop as a
    # background task and never reach this dispatcher.

    if msg_type == "get_state":
        npc_state = npc_manager.get_state()
        return {
            "type": "state",
            "world": world_data,
            "time": game_clock.to_dict(),
            "npcs": npc_state.get("npcs", []),
            "player": player_agent.to_dict(),
        }

    if msg_type == "set_focus":
        x = message.get("x", 0)
        z = message.get("z", 0)
        npc_manager.set_focus(x, z)
        return {"type": "ack", "message": "Focus updated"}

    if msg_type == "get_memory":
        npc_id = message.get("npc_id", "")
        if npc_id:
            summary = memory_manager.get_npc_memory_summary(npc_id)
            return {"type": "memory_data", "npc_id": npc_id, "data": summary}
        return {"type": "memory_data", "data": memory_manager.get_stats()}

    if msg_type == "get_memory_stats":
        return {
            "type": "memory_stats",
            "data": memory_manager.get_stats(),
            "activity": memory_manager.get_recent_activity(limit=20),
        }

    if msg_type == "assign_schedule":
        npc_id = message.get("npc_id", "")
        entries = message.get("entries", [])
        ok, msg = npc_manager.assign_custom_schedule(npc_id, entries)
        return {
            "type": "schedule_result",
            "success": ok,
            "message": msg,
            "npc_id": npc_id,
        }

    if msg_type == "clear_schedule":
        npc_id = message.get("npc_id", "")
        ok, msg = npc_manager.clear_custom_schedule(npc_id)
        return {
            "type": "schedule_result",
            "success": ok,
            "message": msg,
            "npc_id": npc_id,
        }

    return {"type": "error", "message": f"Unknown message type: {msg_type}"}


def _set_thinking_level(level: str) -> dict:
    """Apply a global thinking-level preset to the LLM provider.

    Layered so the same profile object is what per-NPC overrides
    would set via provider.set_npc_profile — a future UI can expose
    a per-NPC panel without touching this path.
    """
    from core.npc.cognition.thinking import profile_for_level
    profile = profile_for_level(level)
    llm_provider.set_profile(profile)
    logger.info("Thinking level set to %s", profile.level.value)
    return {
        "type": "thinking_level_set",
        "level": profile.level.value,
        "thinking_purposes": sorted(profile.thinking_purposes),
        "thinking_budget": profile.thinking_budget,
        "quick_budget": profile.quick_budget,
    }


async def _persist_new_exchanges(conv, npc_a, npc_b) -> list[str]:
    """Thin wrapper around MemoryManager.persist_new_exchanges.

    Adds the server's game-clock context so callers in the chat handler
    don't have to know how to build it.
    """
    minutes_now = game_clock.day * 1440 + game_clock.minutes
    loc_x = getattr(npc_a, "tile_x", int(getattr(npc_a, "x", 0)))
    loc_z = getattr(npc_a, "tile_z", int(getattr(npc_a, "z", 0)))
    return await memory_manager.persist_new_exchanges(
        conv, npc_a, npc_b,
        game_time=minutes_now,
        location_x=loc_x,
        location_z=loc_z,
    )


def _close_player_chat() -> str | None:
    """Tear down any active player-NPC conversation.

    Called on explicit chat_close, on target switch, or when the player
    walks out of interaction range. Returns the partner_id that was
    closed (for building an out-of-range notification), or None.
    """
    partner_id = player_agent.chat_target_id
    player_agent.is_chatting = False
    player_agent.chat_target_id = None
    player_agent.npc.conversation_partner = None

    if not partner_id:
        return None
    target = npc_manager.get_npc(partner_id)
    key = frozenset({player_agent.npc_id, partner_id})
    conv = _active_conversations.get(key)
    if conv and not conv.finished:
        conv.finished = True
    if target and target.conversation_partner == player_agent.npc_id:
        from core.npc.models import ActivityState
        target.conversation_partner = None
        target.activity = ActivityState.IDLE
        target.current_action_description = ""
        target._needs_post_convo_dispatch = True
    return partner_id


def _player_chat_out_of_range() -> dict | None:
    """If the player is in chat but has walked out of range, end it.

    Returns a chat_response payload to broadcast to the client, or None
    if nothing changed. The 3D sim already enforces proximity for
    initiating chat — the chat window must respect the same rule for
    ongoing conversations, otherwise the player gets NPC replies from
    anywhere on the map.
    """
    if not player_agent.is_chatting:
        return None
    partner_id = player_agent.chat_target_id
    if not partner_id:
        return None
    target = npc_manager.get_npc(partner_id)
    if not target:
        _close_player_chat()
        return None
    dist = player_agent.npc.distance_to(target.x, target.z)
    if dist <= player_agent.interaction_radius:
        return None
    # Out of range — tear down and tell the client to close the panel.
    target_name = target.name
    _close_player_chat()
    logger.info("Chat force-closed: player walked out of range of %s", target_name)
    return {
        "type": "chat_response",
        "npc_id": partner_id,
        "npc_name": target_name,
        "error": f"{target_name} is too far away now.",
        "ended": True,
    }


async def _handle_player_chat(message: dict) -> dict | None:
    """Handle one player chat message.

    Streamlined flow: one LLM call per message. The NPC responds
    directly to what the player said — no separate greeting step.
    The prior implementation did two LLM calls on the first message
    (initiate → greeting, then continue → response), which on Gemma's
    thinking mode took ~48 real seconds per message.
    """
    npc_id = message.get("npc_id", "")
    text = (message.get("message") or "").strip()

    if not npc_id or not text:
        return {"type": "chat_response", "error": "Missing npc_id or message"}

    target = npc_manager.get_npc(npc_id)
    if not target:
        return {"type": "chat_response", "error": f"NPC {npc_id} not found"}

    # Check distance (use tile distance — both sides are on a grid)
    if player_agent.npc.distance_to(target.x, target.z) > player_agent.interaction_radius:
        return {"type": "chat_response", "error": f"{target.name} is too far away"}

    # If the player switched targets mid-chat, close the old conversation first.
    if player_agent.is_chatting and player_agent.chat_target_id not in (None, npc_id):
        _close_player_chat()

    # Add the player's message to the conversation (starts one if none active).
    start_player_conversation(target, player_agent.npc, text)
    player_agent.is_chatting = True
    player_agent.chat_target_id = npc_id

    # Defensive tier pin: the player is ACTIVELY addressing this
    # NPC, so there's no world in which a canned tier-3 response
    # is the right answer. Bump the tier so any code path that
    # reads it (including a subsequent `continue_conversation`
    # call from the cognition tick) sees tier 1. A tier-update
    # race — focus had moved but tier assignment hadn't caught up
    # — was the root cause of the "Indeed, quite so." bug.
    target.cognition_tier = 1

    # NPC responds in a single LLM call (QUICK purpose, 150 tokens, no thinking).
    # Disable auto-end roll: random mid-chat endings are jarring when the
    # player is driving the conversation. Give a generous exchange cap so
    # one chat window stays usable for a real back-and-forth.
    # `force_llm=True` bypasses the tier-gated fallback and propagates
    # LLM errors rather than silently returning a canned string.
    llm_error: Exception | None = None
    try:
        matters_against_player = memory_manager.retrieve_unresolved_matters(
            target.npc_id,
            partner_id=player_agent.npc.npc_id,
            partner_name=player_agent.npc.name,
        )
        continues = await continue_conversation(
            target, player_agent.npc, llm_provider, memory_manager,
            allow_auto_end=False, max_exchanges=40,
            force_llm=True,
            town_agenda_summary=npc_manager.town_agenda.summary_for_prompt(
                target.npc_id,
            ),
            shared_agenda_summary=(
                npc_manager.town_agenda.shared_matters_for_prompt(
                    target.npc_id, player_agent.npc.npc_id,
                    current_day=game_clock.day,
                )
            ),
            unresolved_matters_summary=(
                memory_manager.format_unresolved_matters(
                    matters_against_player, player_agent.npc.name,
                )
            ),
        )
    except Exception as e:
        logger.exception("continue_conversation failed for %s", target.name)
        llm_error = e
        continues = False

    # Stale-response guard. Two shapes:
    #
    # 1. The player switched to a different NPC mid-LLM. We drop the
    #    reply silently because it'd land in the wrong chat window.
    # 2. The player walked out of range, or the NPC walked off as
    #    night fell. Here we DO want to deliver the reply — the
    #    conversation happened, it's in memory, and dropping it
    #    silently meant the player saw no closure even though the
    #    memory panel showed the line. Surface the reply with an
    #    `ended` flag so the chat UI prints it and closes gracefully.
    if player_agent.chat_target_id != npc_id:
        return None
    if not player_agent.is_chatting:
        return None

    key = frozenset({player_agent.npc_id, npc_id})
    conv = _active_conversations.get(key)
    stale_npc_line = ""
    if conv and conv.exchanges:
        for ex in reversed(conv.exchanges):
            if ex.speaker_id == target.npc_id:
                stale_npc_line = ex.message
                break

    dist_now = player_agent.npc.distance_to(target.x, target.z)
    if dist_now > player_agent.interaction_radius:
        _close_player_chat()
        payload = {
            "type": "chat_response",
            "npc_id": npc_id,
            "npc_name": target.name,
            "ended": True,
        }
        if stale_npc_line:
            payload["message"] = stale_npc_line
            payload["note"] = (
                f"{target.name} had already stepped away."
            )
        else:
            payload["error"] = f"{target.name} is too far away now."
        return payload

    key = frozenset({player_agent.npc_id, npc_id})
    conv = _active_conversations.get(key)
    npc_line = ""
    if conv and conv.exchanges:
        # The NPC's reply is the last exchange whose speaker is the target.
        for ex in reversed(conv.exchanges):
            if ex.speaker_id == target.npc_id:
                npc_line = ex.message
                break

        # Per-turn persistence (Phase A.2): every exchange added since
        # our last persistence pass gets written into both participants'
        # episodic memory. The player utterance and the NPC reply each
        # produce one memory per participant. Consolidation on chat
        # close dedupes them into a single summary.
        await _persist_new_exchanges(conv, player_agent.npc, target)

    if not continues:
        player_agent.is_chatting = False
        player_agent.chat_target_id = None

    if not npc_line:
        # LLM failed or generated nothing — surface the real cause
        # (network, timeout, empty response) so the player can act
        # on it instead of seeing a silent non-response. Prior
        # behaviour swallowed the exception and returned the canned
        # "Indeed, quite so." fallback, which made it impossible to
        # tell a real reply from an LLM outage.
        err_detail = (
            f" ({type(llm_error).__name__}: {llm_error})"
            if llm_error else ""
        )
        return {
            "type": "chat_response",
            "npc_id": npc_id,
            "npc_name": target.name,
            "error": (
                f"{target.name} didn't respond — the language model "
                f"didn't return a reply{err_detail}. Try again in a "
                f"moment."
            ),
            "ended": not continues,
        }

    return {
        "type": "chat_response",
        "npc_id": npc_id,
        "npc_name": target.name,
        "message": npc_line,
        "ended": not continues,
    }


async def _handle_player_trade(message: dict) -> dict:
    """Handle player trade proposal — route through economy system."""
    npc_id = message.get("npc_id", "")

    if not npc_id:
        return {"type": "trade_response", "error": "Missing npc_id"}

    target = npc_manager.get_npc(npc_id)
    if not target:
        return {"type": "trade_response", "error": f"NPC {npc_id} not found"}

    if player_agent.npc.distance_to(target.x, target.z) > player_agent.interaction_radius:
        return {"type": "trade_response", "error": f"{target.name} is too far away"}

    trade_mgr = npc_manager.economy.trade

    offer, reason = trade_mgr.propose_trade(
        proposer=player_agent.npc,
        recipient=target,
        items_offered=message.get("items_offered", {}),
        gold_offered=message.get("gold_offered", 0),
        items_requested=message.get("items_requested", {}),
        gold_requested=message.get("gold_requested", 0),
        game_time=game_clock.day * 1440 + game_clock.minutes,
    )

    if offer is None:
        return {
            "type": "trade_response",
            "accepted": False,
            "reason": reason,
        }

    # NPC evaluates the trade
    from core.economy.trading import evaluate_trade_heuristic
    prices = trade_mgr.price_engine.get_market_prices(
        npc_manager.economy.resources, npc_manager.npcs,
    )
    accepted, eval_reason = evaluate_trade_heuristic(target, offer, prices)

    if accepted:
        trade_mgr.accept_trade(
            offer.offer_id,
            game_clock.day * 1440 + game_clock.minutes,
        )
        return {
            "type": "trade_response",
            "accepted": True,
            "npc_name": target.name,
            "reason": eval_reason,
            "gold": player_agent.npc.gold,
            "inventory": dict(player_agent.npc.inventory),
        }
    else:
        trade_mgr.reject_trade(offer.offer_id, eval_reason)
        return {
            "type": "trade_response",
            "accepted": False,
            "npc_name": target.name,
            "reason": eval_reason,
        }


# ---------- Entry point ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
