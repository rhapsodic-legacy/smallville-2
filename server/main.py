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
from fastapi.responses import FileResponse

from core.world.generator import WorldConfig, generate_world
from core.world.prompt_gen import TownPromptGenerator
from core.world.prompt_gen.features import TerrainFeature
from core.time_system.clock import GameClock
from core.npc.manager import NPCManager
from core.npc.llm_client import ClaudeProvider, MockProvider
from core.npc.mistral_provider import MistralProvider
from core.npc.cognition.router import CognitionRouter, CognitionPolicy
from core.memory.manager import MemoryManager

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
_spec = _asyncio.get_event_loop().run_until_complete(
    _prompt_gen.generate_config(TOWN_DESCRIPTION)
)
world_config = _spec.config
town_name = _spec.town_name
grid, buildings = generate_world(world_config, features=_spec.features)
world_data = grid.to_dict()  # cached serialisation — rebuilt on world change
logger.info("Generated town '%s' from prompt: %s", town_name, TOWN_DESCRIPTION)

# LLM provider selection: Claude > Mistral > Mock
if os.environ.get("ANTHROPIC_API_KEY"):
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


# ---------- Connection manager ----------

class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.active_connections.remove(conn)


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

        npc_state = npc_manager.movement_tick(game_clock, move_delta)

        tick_msg = {
            "type": "tick",
            "time": game_clock.to_dict(),
            "npcs": npc_state.get("npcs", []),
        }
        if events:
            tick_msg["events"] = events
        if npc_state.get("conversations"):
            tick_msg["conversations"] = npc_state["conversations"]

        await manager.broadcast(tick_msg)


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
async def npc_memory(npc_id: str):
    """Full memory dump for a specific NPC."""
    npc = npc_manager.get_npc(npc_id)
    if not npc:
        return {"error": f"NPC {npc_id} not found"}
    return memory_manager.get_npc_memory_summary(npc_id)


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
        })
    return {
        "time": game_clock.time_string,
        "day": game_clock.day,
        "phase": game_clock.phase.value,
        "npcs": result,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for game state sync."""
    await manager.connect(websocket)
    try:
        # Send initial state on connect (includes NPC data)
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
        })

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            response = handle_message(message)
            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


def handle_message(message: dict) -> dict:
    """Route incoming WebSocket messages to appropriate handlers."""
    msg_type = message.get("type")

    if msg_type == "ping":
        return {"type": "pong"}

    if msg_type == "move":
        # TODO: Pass to core world.player_action() (Phase 8)
        return {"type": "state", "message": "Movement not yet implemented"}

    if msg_type == "chat":
        # TODO: Pass to core NPC conversation system (Phase 8)
        return {"type": "chat_response", "message": "Chat not yet implemented"}

    if msg_type == "get_state":
        npc_state = npc_manager.get_state()
        return {
            "type": "state",
            "world": world_data,
            "time": game_clock.to_dict(),
            "npcs": npc_state.get("npcs", []),
        }

    if msg_type == "set_focus":
        # Update camera focus point for tier assignment
        x = message.get("x", 0)
        z = message.get("z", 0)
        npc_manager.set_focus(x, z)
        return {"type": "ack", "message": "Focus updated"}

    if msg_type == "get_memory":
        # Memory inspector: get memory for a specific NPC
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

    return {"type": "error", "message": f"Unknown message type: {msg_type}"}


# ---------- Entry point ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
