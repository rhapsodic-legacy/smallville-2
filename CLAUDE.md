# Smallville 2 — Self-Evolving NPC Ecosystem

## Overview
A browser-playable 3D world populated with AI-driven NPCs that have persistent memory,
goals, relationships, and resource needs. They form alliances, trade, compete, and adapt.
An overseer agent periodically evaluates population fitness and injects evolved behavioural
policies. Built on Stanford's Generative Agents research, upgraded with hybrid memory,
tiered cognition, and evolutionary dynamics.

**Dual purpose:** Standalone showcase + NPC backbone for the AI Game Design Studio
(see claude_agent_swarm project).

## Architecture

### Tech Stack
- **Backend:** Python 3.11+ / FastAPI / WebSocket
- **Frontend:** Three.js (procedural 3D geometry) / vanilla JS
- **Databases:** SQLite (structured NPC state) + ChromaDB (episodic memory embeddings)
- **LLM:** Claude API (Haiku for NPCs, Opus for overseer) via Anthropic SDK
- **Protocol:** Server-authoritative — all game logic in Python, browser is thin renderer

### Project Structure
```
core/           — smallville_core library (importable package)
  world/        — spatial grid, procedural town generator, pathfinding
  npc/          — NPC data model, tiered cognition (perceive/retrieve/plan/reflect/execute)
  memory/       — hybrid memory (SQLite knowledge graph + ChromaDB embeddings)
  relationships/— sentiment dimensions, factions, formal structures
  events/       — event impact system (hard coded, conditional, boolean triggers)
  economy/      — gold, resources, trading, construction, crafting
  evolution/    — overseer agent, fitness functions, policy injection
  time_system/  — game clock, day/night cycle, schedule slots
  player/       — player-as-NPC model, interaction handling
server/         — FastAPI server (WebSocket + REST endpoints)
client/         — Three.js frontend (renderer, UI, controls)
tests/          — unit, integration, simulation tests
```

### Key Design Patterns
- **Server-Authoritative:** All logic in Python. Client sends actions, server validates,
  returns new state. No client-side game logic.
- **Tiered Cognition:** NPCs get different levels of AI reasoning based on proximity
  and relevance: Tier 1 (full LLM), Tier 2 (simplified LLM), Tier 3 (state machine),
  Tier 4 (frozen).
- **Hybrid Memory:** Knowledge graph for hard facts + embedding retrieval for episodic
  memory. Upgraded from Stanford's pure-embedding approach.
- **Event Impact System:** Data-driven rules table mapping events to effects.
  Supports hard coded, conditional, and boolean/narrative triggers.
  Works at individual level (engagement ring) and population level (war = True).
- **Data-Driven Design:** NPC templates, world parameters, event rules, and fitness
  functions are all configurable data — not hard coded logic.

## Conventions

### File Size
- **Target:** 500 lines per file
- **Maximum:** 750 lines absolute limit
- If a file approaches 500 lines, split it into focused modules

### Code Style
- Python: follow PEP 8, type hints on public functions
- JavaScript: vanilla JS, no build step, ES modules
- British English in all text (comments, docs, strings facing users)

### Testing — Automated Pipelines (Never Manual)
- **All testing is automated.** Claude runs tests, fixes failures, and re-runs.
  The user never runs tests, observes behaviour, or manually verifies.
- **Every system must have a validation pipeline.** When building a new system,
  create its automated test as part of the work — not as a follow-up.
- **Movement/Pathfinding:** `python3 tests/simulation/test_npc_movement.py`
  — run after any change to pathfinding, generator, spatial_awareness, execute, models
- **Unit tests:** `pytest tests/unit/ -v` — mock LLM, fast
- **Integration tests:** `pytest tests/integration/ -v` — real Haiku API
- **Simulation tests:** `pytest tests/simulation/ -v --timeout=600` — headless multi-day runs
- **Workflow:** change code → run pipeline → fix failures → re-run → only then report to user

### Git
- Meaningful commit messages describing the "why"
- Feature branches for major additions
- Never commit .env or API keys

## Running
- **Server is not auto-started.** Start `python3 server/main.py` only
  when the browser client is actually needed this session. Before any
  Gemma-heavy sim, grep for straggler server processes from prior
  sessions (`ps aux | grep server/main`) — they steal Ollama
  throughput. See auto-memory "Server lifecycle".
- Server runs at http://localhost:8002
```bash
# Install dependencies
pip install -e ".[dev]"

# Start server (only when browser client needed)
python server/main.py

# Open browser to http://localhost:8002
```

## Deferred — run when hardware allows

The bridge-objector diagnostic is the next concrete experiment. It's a
logging-only sim (no pass/fail) that tests whether a weighted-
participation gate + an NPC carrying `opposes:repair_bridge = 0.9`
produces real emergent behaviour under non-deterministic Gemma. Phase
J of Memory v2 (persona snapshot) is parked until we've read those
logs. On the current Mac, Gemma-e2b produces ~1 sim day per 30
wall-minutes at 10 NPCs — so 30 days takes ~15 hours.

```bash
# Kill any straggler server first (frees Gemma throughput)
ps aux | grep server/main | grep -v grep

# Then run the diagnostic
python3 tests/simulation/diagnostic_bridge_objector.py --days=30
```

After the run, read the daily log for: does Jasper voice opposition in
dialogue, does the bridge goal succeed or fail around him, how does
sentiment toward him shift across cycles. Those signals decide
whether Phase J re-opens. Full rationale in MEMORY_V2_ROADMAP.md under
"Emergent-behaviour pivot".

## Reference Projects
- **Stanford Generative Agents:** ./generative_agents/ (cloned for reference)
- **AI Game Studio:** /Users/jessepassmore/Desktop/Programming_Pizazz/claude_agent_swarm/
- **Stanford Paper:** ./standford_smallville.pdf

## Tracking
- **Roadmap:** PROJECT_ROADMAP.md (phases, substeps, status)
- **Sub-roadmaps** for in-flight (or recently-shipped) feature arcs:
  - MEMORY_ROADMAP.md — holistic conversation memory (shipped
    2026-04-20; D.2/D.3 deferred tuning)
  - MEMORY_V2_ROADMAP.md — next-gen memory: tags, compaction,
    progress-aware objectives, unified persona snapshot (design
    phase, not started)
- Update roadmap status as work completes
- Read PROJECT_ROADMAP.md first after any interruption to understand current state,
  then any active sub-roadmap listed in its "Active Sub-Roadmaps" section
