# core/ — smallville_core Library

## Purpose
The core library contains all game logic, NPC cognition, memory, relationships,
and world simulation. It is designed to be importable as a Python package by both
the Smallville 2 server and the AI Game Studio.

## Public API (target)
```python
from smallville_core import World, NPCManager, Overseer

world = World.generate(population=30, terrain="riverside", ruler_type="good_king")
world.tick()  # advance one simulation step
world.player_action(player_id, action)
state = world.get_state()  # full serialisable state for client
```

## Module Responsibilities

### world/
- Spatial grid with tile data (terrain, objects, events, collision)
- Hierarchical addressing: world:sector:arena:object
- Procedural town generator with parameter shaping
- A* pathfinding
- **Spatial awareness** (`spatial_awareness.py`): foundational layer for NPC positioning
  - NPCs may pass through each other while WALKING — no movement collision
  - NPCs must NEVER share a tile when at rest (any non-WALKING state)
  - Conversations require adjacency (Manhattan distance = 1)
  - All rest-placement MUST go through `find_rest_tile()` / `find_conversation_positions()`
  - `resolve_overlaps()` runs every tick as a safety net

### npc/
- NPC data model (identity, physical state, goals, occupation)
- Tiered cognition system (4 tiers based on proximity/relevance)
- Cognitive cycle: perceive → retrieve → plan → reflect → execute
- Conversation system
- LLM integration layer (pluggable provider interface)

### memory/
- SQLite structured storage (facts, relationships, goals)
- ChromaDB episodic memory (embeddings with recency/importance/relevance scoring)
- Memory manager (unified retrieval combining both stores)
- Reflection system (importance accumulator, focal points, insight synthesis)
- Spatial memory (hierarchical world knowledge tree)

### relationships/
- Sentiment dimensions per NPC pair (trust, fear, respect, affection, debt)
- Sparse storage — only non-default relationships tracked
- Faction model (members, roles, allies, rivals)
- Formal structures (trade agreements, alliances, councils)

### events/
- Event impact system: data-driven rules table
- Three impact modes: hard coded, conditional, boolean/narrative
- World-level event propagation (war, famine, festival)
- Event rule API for AI Game Studio configuration

### economy/
- Resource types (wood, stone, gold, food)
- Gathering, trading, crafting systems
- Construction with progress tracking and build phases
- Supply and demand pricing

### evolution/
- Overseer agent (periodic evaluation, strategy analysis)
- Multi-objective fitness functions (configurable weights)
- Evolution mechanisms: parameter tuning, policy templates, prompt modifiers
- Behavioural guardrails and narrative consistency

### time_system/
- Game clock with configurable speed (default 1 day = 20 min)
- Day/night cycle (dawn, day, dusk, night)
- Schedule slots for NPC daily routines

### player/
- Player-as-NPC model (same data structure, human flag)
- Configurable NPC awareness (indistinguishable vs known human)
- Interaction priority boost for player encounters

## Conventions
- Each module has an __init__.py that exports the public interface
- Internal helpers prefixed with underscore
- Type hints on all public functions
- Docstrings only where logic is non-obvious
