# Smallville 2 — a self-evolving NPC ecosystem       

A browser-playable 3D world populated by AI-driven NPCs that have persistent
memory, goals, relationships, occupations and resource needs. They follow
daily routines, talk to one another, trade, form factions, take on town
initiatives, and adapt over time. An overseer agent periodically evaluates
the population and nudges its behaviour.

It's built on the ideas in Stanford's [*Generative Agents*](https://arxiv.org/abs/2304.03442),
extended with a hybrid memory system, tiered cognition (so hundreds of NPCs
can be simulated affordably), an economy, and evolutionary dynamics.

> **Dual purpose:** a standalone showcase, *and* the NPC backbone for a larger
> AI game-design project.

---

## What it is

The long-term goal is a **living world for an RPG** — one where the world isn't
shaped only by the player fighting through a story, but by **helping the peoples
of the land**, and where those societies **grow organically in response to what's
happening around them, including circumstances no designer scripted.**

Concretely, that means a world where, say, a human helping to build a mine near
a town doesn't trigger a hard-coded "mine → miners" rule — instead mining
becomes *economically valuable*, NPCs who choose their work by opportunity drift
toward it, gold flows in, prices shift, and the town reshapes itself. The same
general mechanisms have to handle a drought, a new trade road, or a war nobody
planned for.

Today it's a **single town simulated end-to-end** — a foundation you can run
headless or watch in the browser. The mechanisms that make organic, emergent
town behaviour possible are the focus of current work.

## How it's built

- **Server-authoritative.** All game logic lives in Python; the browser is a
  thin Three.js renderer. The client sends actions, the server validates and
  returns new state over WebSocket.
- **Tiered cognition.** NPCs get different levels of reasoning based on
  proximity and relevance — Tier 1 (full LLM) down to Tier 4 (frozen) — so a
  large population stays affordable. A cognition *router* decides, per
  NPC and per decision, whether to spend an LLM call or run a cheap
  deterministic planner.
- **Hybrid memory.** A SQLite knowledge graph for hard facts plus ChromaDB
  embeddings for episodic memory, with recency/importance/relevance retrieval,
  reflection, tag-based retention, and hierarchical compaction (so old days
  collapse into summaries instead of a growing firehose).
- **Rich NPCs.** Identity, a Big-Five personality vector, an evolving
  self-concept, long- and short-term goals, per-pair sentiment (trust, fear,
  respect, affection…), occupation and skills, and physical needs.
- **Collective behaviour.** A town agenda proposes goals (repair the bridge,
  hold the harvest festival); NPCs decide whether to take them on, perform
  them, and complete them together — and goal completion feeds back into
  identity and relationships.
- **Data-driven world reactions.** An event-impact system maps events to
  effects at both the individual and population level.
- **Economy.** Resources, gathering, NPC-to-NPC trade, construction, crafting,
  and supply/demand pricing.
- **Evolution layer.** An overseer scores population fitness and can inject
  behavioural policy nudges.

**Stack:** Python 3.11+ · FastAPI + WebSocket · Three.js (procedural geometry,
no build step) · SQLite + ChromaDB · LLMs via a pluggable provider interface
(Claude for the overseer; local Gemma or the Mistral API for NPCs; a
deterministic Mock provider for tests).

## Where we're at

The simulation runs: NPCs spawn, follow occupation-based daily schedules, move
and path around a procedurally generated town, converse, trade, form
relationships, and react to events. A human can join as a player and interact.

Most recently, the **scheduling / town-goal foundation was rebuilt** (see
[`FOUNDATION_REBUILD_ROADMAP.md`](./FOUNDATION_REBUILD_ROADMAP.md)). A long-
standing bug meant town goals could only ever *expire* — NPCs never actually
completed collective initiatives. It was root-caused through five layers (a
measurement bug, bedtime-preempted crediting, a schedule-execution stall, a
schedule parser that mangled days, and a test-double artifact) and fixed:
durable commitments, a faithful schedule parser, a bounded daily plan, and
robust crediting. Town goals now complete organically.

A few things define how the project is built:

- **Evals before mechanism.** Behaviour is steered by a deterministic
  behavioural eval (`tests/simulation/eval_foundation.py`) — a dashboard of
  goal-lifecycle, schedule-health and scalability metrics — not just unit
  tests. The recent rebuild was driven entirely by *measuring* what was
  breaking; several plausible-but-wrong fixes were caught and discarded
  because the numbers didn't move.
- **Automated test culture.** ~1,350 unit tests plus simulation tests and a
  dedicated movement/pathfinding harness; the suite is kept green.
- **Two kinds of evaluation.** Mechanics get binary pass/fail tests; emergent
  behaviour gets *observational* diagnostics with pre-registered criteria (you
  can't unit-test "something unforeseen and sensible happened").

## Where we're heading

The next major arc is the **living world** itself — turning the sound
foundation into a world that adapts:

- **General, signal-driven adaptation.** The core engine is utility/economy/
  needs driving behaviour, so towns respond sensibly to *any* circumstance
  rather than only the ones someone wrote a rule for. The first concrete
  capability is **utility-based dynamic professions** (an NPC's occupation
  becomes a choice driven by opportunity).
- **Most NPCs cheap, key NPCs deep.** The intent is for ~95–99% of NPCs to run
  on the lightweight utility layer, with important characters given togglable,
  triggerable LLM "bigger brains" — which the existing tier system and
  cognition-router priority mechanism already make possible.
- **Multiple interacting towns.** Distinct places that affect one another and
  can collaborate on shared goals; the `town_id` seam is already in the data
  model. Spatial partitioning by town doubles as the scaling strategy.
- **An architectural drift toward agents over machinery.** A captured design
  direction (`AGENT_DIRECTION.md`) leans toward a communal world-substrate +
  private per-NPC experience + information that propagates only through
  messages/perception — so emergent social dynamics arise from agent
  structure rather than hand-coded systems.

## Repository tour

```
core/            importable simulation library
  world/         spatial grid, procedural town generator, A* pathfinding
  npc/           NPC model, tiered cognition (perceive/plan/reflect/execute),
                 LLM providers, the deterministic planner + cognition router
  memory/        SQLite knowledge graph + ChromaDB episodic memory, reflection
  relationships/ sentiment dimensions, factions, formal structures
  events/        data-driven event-impact rules
  economy/       resources, trading, construction, crafting
  evolution/     overseer agent, fitness functions, policy injection
  time_system/   game clock, day/night cycle, schedule slots
  player/        player-as-NPC model and interaction handling
server/          FastAPI server (WebSocket + REST)
client/          Three.js front end (renderer, UI, controls)
tests/           unit, integration, and simulation tests + the behavioural eval

PROJECT_ROADMAP.md            master roadmap and status
FOUNDATION_REBUILD_ROADMAP.md the scheduling/town-goal rebuild, in detail
AGENT_DIRECTION.md            the longer-term architectural direction
```

## Running it

```bash
pip install -e ".[dev]"          # install dependencies
python server/main.py            # start the server (only when the browser
                                 # client is actually needed)
# open http://localhost:8002
```

Most development happens headless against the test/eval harnesses rather than
the browser. For example:

```bash
pytest tests/unit/ -v                            # fast, mocked
python3 tests/simulation/eval_foundation.py      # behavioural dashboard
python3 tests/simulation/test_npc_movement.py    # movement/pathfinding checks
```

## Status

This is an active, research-flavoured work in progress, not a finished game.
Expect rough edges, placeholder visuals, and systems in motion. The interesting
part is the simulation underneath — and whether simple, general mechanisms can
make a town feel alive.
