# server/ — FastAPI Server

## Purpose
Thin server layer that wraps the core library and serves the client.
All game logic lives in core/ — the server handles WebSocket connections,
client state sync, and serves static files.

## Architecture
- **main.py** — FastAPI app, server startup, static file serving
- **api/** — WebSocket and REST endpoint handlers
- **config/** — Server configuration, environment variables

## WebSocket Protocol
Single JSON WebSocket connection per client.

### Client → Server Messages
```json
{"type": "move", "direction": "north"}
{"type": "chat", "npc_id": "blacksmith_1", "message": "Hello"}
{"type": "trade", "npc_id": "merchant_1", "offer": {...}}
{"type": "interact", "object_id": "notice_board"}
```

### Server → Client Messages
```json
{"type": "state", "world": {...}, "npcs": [...], "player": {...}}
{"type": "chat_response", "npc_id": "blacksmith_1", "message": "..."}
{"type": "event", "events": [...]}
{"type": "error", "message": "..."}
```

## Key Principles
- Server is authoritative — never trust client state
- Core library does all computation; server is a thin wrapper
- WebSocket sends full relevant state on every action (not deltas for MVP)
- REST endpoints for non-realtime operations (save, load, config)

## Configuration
- Port: 8002 (default; 8000 and 8001 are reserved for AI Game Studio)
- Environment variables via .env file
- ANTHROPIC_API_KEY required for LLM calls
