# Smallville 2 — Project Roadmap

> Master tracking document. Update status as work progresses.
> If restarting from a crash, read this first to understand where we left off.

## Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Phase 1: Foundation
> Goal: Repo structure, Claude Code scaffolding, dev tooling

- [x] 1.1 Create PROJECT_ROADMAP.md (this file)
- [x] 1.2 Git init + .gitignore
- [x] 1.3 Root CLAUDE.md (project overview, architecture, conventions)
- [x] 1.4 Directory structure (core/, server/, client/, tests/)
- [x] 1.5 Hierarchical CLAUDE.md files (core, server, client)
- [x] 1.6 Skills definitions (.claude/skills/)
  - [x] 1.6.1 npc_creation — NPC data model and initialization patterns
  - [x] 1.6.2 memory_ops — Memory system operations and retrieval
  - [x] 1.6.3 overseer_eval — Overseer evaluation and policy injection
  - [x] 1.6.4 test_runner — Test execution patterns
  - [x] 1.6.5 world_gen — Procedural world generation rules
  - [x] 1.6.6 event_system — Event impact system patterns
- [x] 1.7 Hooks configuration (.claude/settings.json)
  - [x] 1.7.1 Python syntax validation on file write
  - [x] 1.7.2 File size enforcement (500 line target, 750 max)
- [x] 1.8 Subagent definitions (.claude/agents/)
  - [x] 1.8.1 swarm_explorer — Reads claude_agent_swarm codebase
  - [x] 1.8.2 stanford_explorer — Reads Stanford generative_agents repo
  - [x] 1.8.3 self_explorer — Reads Smallville 2 codebase
  - [x] 1.8.4 npc_thinker — Parallel NPC cognition runner
  - [x] 1.8.5 overseer_analyst — Evolution layer analysis
  - [x] 1.8.6 world_simulator — Headless simulation runner
- [x] 1.9 Python project setup (pyproject.toml, requirements.txt)
- [x] 1.10 Client scaffolding (index.html, Three.js boilerplate, WebSocket)
- [x] 1.11 FastAPI server skeleton (main.py, WebSocket endpoint)
- [x] 1.12 Verify end to end: server starts, client connects, renders empty world

---

## Phase 2: World Core
> Goal: Procedural town, spatial grid, 3D rendering, time system

- [x] 2.1 Spatial grid system (core/world/grid.py)
  - [x] 2.1.1 Tile data model (terrain, objects, events, collision)
  - [x] 2.1.2 Hierarchical addressing (world:sector:arena:object)
  - [x] 2.1.3 Spatial queries (nearby tiles, pathfinding)
- [x] 2.2 Procedural town generator (core/world/generator.py)
  - [x] 2.2.1 Parameter driven layout (population, ruler, economy, terrain)
  - [x] 2.2.2 Building placement (tavern, blacksmith, church plot, homes, market)
  - [x] 2.2.3 Road and path network
  - [x] 2.2.4 Resource node placement (trees, mines, fields)
- [x] 2.3 A* pathfinding (core/world/pathfinding.py)
- [x] 2.4 Three.js world renderer (client/js/world_renderer.js)
  - [x] 2.4.1 Terrain mesh generation from grid
  - [x] 2.4.2 Building meshes (procedural geometry)
  - [x] 2.4.3 Camera controls (follow, free look)
  - [x] 2.4.4 Day/night lighting cycle
- [x] 2.5 Time system (core/time_system/)
  - [x] 2.5.1 Game clock (configurable speed, default 1 day = 20 min)
  - [x] 2.5.2 Day/night cycle with dawn, day, dusk, night phases
  - [x] 2.5.3 Schedule slots (morning, afternoon, evening, night)
- [x] 2.6 WebSocket state sync (server sends world state, client renders)
- [x] 2.7 Verify: 3D town renders, camera works, day/night cycles

---

## Phase 3: NPC Core
> Goal: NPCs that move, work, sleep, socialize with tiered cognition

- [x] 3.1 NPC data model (core/npc/models.py)
  - [x] 3.1.1 Identity (name, age, personality traits, backstory)
  - [x] 3.1.2 Physical state (location, health, energy, hunger)
  - [x] 3.1.3 Goals (short term, long term, daily schedule)
  - [x] 3.1.4 Occupation and skills
- [x] 3.2 LLM integration layer (core/npc/llm_client.py)
  - [x] 3.2.1 Claude API wrapper (Haiku for NPCs, Opus for overseer)
  - [x] 3.2.2 Pluggable provider interface (future: local models)
  - [x] 3.2.3 Rate limiting and cost tracking
  - [x] 3.2.4 Prompt template system
- [x] 3.3 Tiered cognition system (core/npc/cognition/)
  - [x] 3.3.1 Tier 1: Full LLM cycle (perceive, retrieve, plan, reflect, execute)
  - [x] 3.3.2 Tier 2: Simplified LLM (less frequent, shorter prompts)
  - [x] 3.3.3 Tier 3: Statistical state machine (no LLM calls)
  - [x] 3.3.4 Tier 4: Frozen (no updates until relevant)
  - [x] 3.3.5 Tier assignment logic (distance from player, relevance)
- [x] 3.4 Perception module (core/npc/cognition/perceive.py)
  - [x] 3.4.1 Vision radius and attention bandwidth
  - [x] 3.4.2 Event detection on nearby tiles
  - [x] 3.4.3 Retention window (avoid reperceiving)
- [x] 3.5 Planning module (core/npc/cognition/plan.py)
  - [x] 3.5.1 Daily schedule generation (LLM driven)
  - [x] 3.5.2 Task decomposition (hourly to minute level)
  - [x] 3.5.3 Action location selection
  - [x] 3.5.4 Reaction planning (encounter other NPCs)
- [x] 3.6 Execution module (core/npc/cognition/execute.py)
  - [x] 3.6.1 Path following
  - [x] 3.6.2 Action animation state (for client)
  - [x] 3.6.3 Object interaction
- [x] 3.7 Conversation system (core/npc/cognition/converse.py)
  - [x] 3.7.1 Decision to engage
  - [x] 3.7.2 Dialogue generation
  - [x] 3.7.3 Conversation memory recording
- [x] 3.8 NPC renderer (client/js/npc_renderer.js)
  - [x] 3.8.1 3D character meshes
  - [x] 3.8.2 Movement animation
  - [x] 3.8.3 Activity indicators (speech bubbles, status icons)
- [x] 3.9 Verify: NPCs spawn, follow schedules, move around, talk to each other

---

## Phase 4: Memory System
> Goal: Hybrid memory with knowledge graph + embeddings

- [x] 4.1 SQLite structured storage (core/memory/structured.py)
  - [x] 4.1.1 Schema: facts, relationships, events, goals
  - [x] 4.1.2 CRUD operations for NPC knowledge
  - [x] 4.1.3 Query interface (who owes me gold? who is allied with whom?)
- [x] 4.2 ChromaDB embeddings storage (core/memory/episodic.py)
  - [x] 4.2.1 Observation embedding and storage
  - [x] 4.2.2 Retrieval by similarity (recency, importance, relevance scoring)
  - [x] 4.2.3 Memory importance scoring (poignancy via LLM)
- [x] 4.3 Memory manager (core/memory/manager.py)
  - [x] 4.3.1 Unified retrieval (combines structured + episodic)
  - [x] 4.3.2 Memory formation pipeline (observe, score, store)
  - [x] 4.3.3 Memory decay and consolidation
- [x] 4.4 Reflection system (core/memory/reflection.py)
  - [x] 4.4.1 Importance accumulator trigger
  - [x] 4.4.2 Focal point generation
  - [x] 4.4.3 Insight synthesis via LLM
  - [x] 4.4.4 Post conversation reflection
- [x] 4.5 Spatial memory (core/memory/spatial.py)
  - [x] 4.5.1 Hierarchical world knowledge tree
  - [x] 4.5.2 Updates from perception
- [x] 4.6 Verify: NPCs remember interactions, retrieve relevant memories, reflect

---

## Phase 5: Relationships and Social
> Goal: Rich sentiment, factions, formal structures

- [x] 5.1 Sentiment dimensions (core/relationships/sentiment.py)
  - [x] 5.1.1 Per pair tracking: trust, fear, respect, affection, debt
  - [x] 5.1.2 Event driven updates (trade, conversation, conflict)
  - [x] 5.1.3 Sparse storage (only non default relationships tracked)
- [x] 5.2 Event impact system (core/events/impact.py)
  - [x] 5.2.1 Event rule table: (event_type, conditions, effects)
  - [x] 5.2.2 Hard coded impacts (ring = +500 affection)
  - [x] 5.2.3 Conditional impacts (ring IF would_accept)
  - [x] 5.2.4 Boolean/narrative triggers (ring sets waiting_for_proposal = False)
  - [x] 5.2.5 World level events (war = True modifies all NPCs)
- [x] 5.3 Formal structures (core/relationships/structures.py)
  - [x] 5.3.1 Faction model (name, members, roles, allies, rivals)
  - [x] 5.3.2 Trade agreements and alliances
  - [x] 5.3.3 Leadership and hierarchy
- [x] 5.4 Relationship driven decisions
  - [x] 5.4.1 NPCs consider relationships when planning
  - [x] 5.4.2 Alliance and betrayal logic
  - [x] 5.4.3 Group decision making (council votes)
- [x] 5.5 Verify: Relationships form, events trigger impacts, factions emerge

---

## Phase 6: Resource and Construction
> Goal: Gold economy, gathering, trading, building

- [x] 6.1 Resource system (core/economy/resources.py)
  - [x] 6.1.1 Resource types (wood, stone, gold, food)
  - [x] 6.1.2 Resource nodes on map (trees, mines, fields)
  - [x] 6.1.3 Gathering mechanics (time, skill, yield)
- [x] 6.2 Trading system (core/economy/trading.py)
  - [x] 6.2.1 NPC to NPC trade (negotiation via LLM)
  - [x] 6.2.2 Market/shop mechanics
  - [x] 6.2.3 Supply and demand pricing
- [x] 6.3 Construction system (core/economy/construction.py)
  - [x] 6.3.1 Building blueprints (resource requirements, build time)
  - [x] 6.3.2 Progress tracking (church: 60/100 wood, 30/50 stone)
  - [x] 6.3.3 NPC contribution decisions
  - [x] 6.3.4 Visual build phases (scaffolding, walls, complete)
- [x] 6.4 Crafting system (core/economy/crafting.py)
  - [x] 6.4.1 Recipe definitions
  - [x] 6.4.2 Skill requirements
  - [x] 6.4.3 Quality outcomes
- [x] 6.5 Verify: NPCs gather, trade, accumulate gold, build structures

---

## Phase 7: Evolution Layer
> Goal: Overseer agent that evaluates and evolves NPC behaviours

- [x] 7.1 Fitness functions (core/evolution/fitness.py)
  - [x] 7.1.1 Multi objective scoring (survival, prosperity, social, goals, engagement)
  - [x] 7.1.2 Configurable weights per world theme
  - [x] 7.1.3 Population level metrics
- [x] 7.2 Overseer agent (core/evolution/overseer.py)
  - [x] 7.2.1 Periodic evaluation cycle
  - [x] 7.2.2 Strategy analysis (what's working, what's failing)
  - [x] 7.2.3 Intervention triggers (stagnation, runaway behaviours, imbalance)
- [x] 7.3 Evolution mechanisms (core/evolution/mechanisms.py)
  - [x] 7.3.1 Parameter tuning (risk tolerance, cooperation, aggression)
  - [x] 7.3.2 Policy templates (merchant, hermit, politician strategies)
  - [x] 7.3.3 Prompt modifiers (behavioural directives injected into NPC prompts)
- [x] 7.4 Guardrails (core/evolution/guardrails.py)
  - [x] 7.4.1 Behavioural boundaries (prevent degenerate strategies)
  - [x] 7.4.2 Narrative consistency checks
  - [x] 7.4.3 Modular rule system (pluggable for AI Game Studio)
- [x] 7.5 Verify: Overseer runs, evaluates, injects policies, population adapts

---

## Phase 8: Player Integration
> Goal: Human player interacts with NPC world

- [x] 8.1 Player as NPC (core/player/player_agent.py)
  - [x] 8.1.1 Player data model (same as NPC, flagged as human)
  - [x] 8.1.2 Configurable NPC awareness of player (indistinguishable vs known)
  - [x] 8.1.3 NPC interaction priority boost for player
- [x] 8.2 Chat interface (client/js/chat_ui.js)
  - [x] 8.2.1 Text input for talking to nearby NPCs
  - [x] 8.2.2 Conversation history display
  - [x] 8.2.3 NPC response generation (LLM with memory context)
- [x] 8.3 Player movement and camera (client/js/player_controls.js)
  - [x] 8.3.1 WASD/arrow movement
  - [x] 8.3.2 Camera follow mode
  - [x] 8.3.3 Interaction radius display
- [x] 8.4 Trading UI (client/js/trade_ui.js)
  - [x] 8.4.1 Player inventory display
  - [x] 8.4.2 Trade proposal interface
  - [x] 8.4.3 NPC negotiation responses
- [x] 8.5 HUD (client/js/hud.js)
  - [x] 8.5.1 Time of day, gold, nearby NPCs
  - [x] 8.5.2 Minimap
  - [x] 8.5.3 Notification feed (events happening in town)
- [x] 8.6 Verify: Player walks around, chats, trades, NPCs remember player

---

## Phase 9: AI Game Studio Bridge
> Goal: Package as importable library + microservice API

- [ ] 9.1 Library packaging (core as pip installable package)
  - [ ] 9.1.1 Clean public API: create_world(), tick(), player_action(), get_state()
  - [ ] 9.1.2 Configuration objects for world params, NPC params, event rules
  - [ ] 9.1.3 Package metadata and documentation
- [ ] 9.2 FastAPI wrapper for AI Game Studio
  - [ ] 9.2.1 REST endpoints for world management
  - [ ] 9.2.2 WebSocket for real time state
  - [ ] 9.2.3 Event rule injection API
- [ ] 9.3 Parameter driven world configuration
  - [ ] 9.3.1 Town generation params (population, economy, ruler, terrain)
  - [ ] 9.3.2 NPC template system (archetypes, personality ranges)
  - [ ] 9.3.3 Event rule definitions (engagement ring, war, custom events)
  - [ ] 9.3.4 Fitness function configuration
- [ ] 9.4 Integration testing with claude_agent_swarm patterns
- [ ] 9.5 Verify: AI Game Studio can create and run an NPC world via API

---

## Phase 10: Multiplayer Prep
> Goal: Foundation for concurrent players

- [ ] 10.1 SQLite to PostgreSQL migration path
- [ ] 10.2 Session management (multiple WebSocket connections)
- [ ] 10.3 Player visibility and interaction rules
- [ ] 10.4 Concurrent action resolution
- [ ] 10.5 Verify: Two players can connect and interact in same world

---

## Active Sub-Roadmaps
- **FOUNDATION_REBUILD_ROADMAP.md** — Holistic rebuild of the NPC
  scheduling / planning / execution / town-goal-contribution
  foundation. **COMPLETE 2026-06-07** (single-town). The original bug —
  organic town goals could only ever expire — is fixed and validated;
  root-caused through five layers (eval measurement bug → bedtime
  preemption → schedule-advance stall → `_parse_llm_schedule` mangling
  → MockProvider replan-churn stub artifact), each disproven by the
  eval/probes, never guessed. Now: durable `Commitment`s → bounded
  derived plan → faithful schedule parser → bedtime-safe commitment-keyed
  crediting. Steered throughout by `tests/simulation/eval_foundation.py`
  (deterministic behavioural dashboard). Read this if revisiting the
  scheduling/agenda layer.
- **"Living world" arc (future, not started)** — the long-term vision:
  an RPG world where helping the peoples matters and towns grow
  *organically*, including under *unforeseen* circumstances. Design
  consequence (captured in conversation 2026-06-07): "unforeseen" rules
  out an event→effect rules table as the core engine — adaptation must
  emerge from GENERAL drivers (utility/economy/needs), with the LLM for
  novelty. ~95–99% of NPCs on the utility layer; important NPCs get
  togglable LLM "bigger brains" via the existing tier/router priority
  seam. First real capability = utility-based dynamic professions.
  Measured by emergence diagnostics (bridge-objector style), not binary
  tests. Ties into AGENT_DIRECTION (proximity/message propagation).
- **MEMORY_ROADMAP.md** — Holistic conversation memory (per-turn persistence,
  outcome extraction, cross-NPC propagation). Shipped 2026-04-20.
- **MEMORY_V2_ROADMAP.md** — Next-generation memory: tag-based specific
  retention, hierarchical compaction, progress-aware objectives,
  unified persona snapshot. K/H/I shipped; J parked pending evidence
  (now re-opened/upgraded by the vectorization arc — see below).
- **VECTORIZATION_ROADMAP.md** — NPC individuality. The measured problem:
  NPCs read as parrots (Layer-1 metrics → SYSTEMIC: self barely forms,
  drowned by 97% conversation volume, 0% sentiment friction). Research
  (NVIDIA ACE/Convai, Skyrim Mantella, roleplay-LLM, Stanford) → the
  foundation is a rich, CONCRETE, persistent persona (speech/behaviour
  rules, not Big-5 numbers) conditioning every cognition call = MEMORY_V2
  Phase J re-opened and upgraded. First step: concrete distinctive
  personas as the dominant conversation-prompt block, then re-measure
  with `npc_individuality.py`. The 30-day emergence run is GATED behind
  these metrics moving. Recommended as a fresh stronger-model arc.
- **AGENT_DIRECTION.md** — IoA-derived architectural philosophy
  (communal world-state + private experience + message-only
  propagation). Captured 2026-05-02. Not a roadmap to start now;
  the direction Smallville drifts toward if bridge-objector evidence
  confirms current mechanism layer compensates for missing agent
  properties. First experiment if reached: privatise sentiment.

## Current Status
**Active Phase:** Foundation rebuild COMPLETE (2026-06-07) — see
FOUNDATION_REBUILD_ROADMAP.md. The NPC scheduling/planning/town-goal
foundation was found broken (organic town goals could only ever expire)
and rebuilt: durable commitments, bounded derived plan, faithful
schedule parser, bedtime-safe crediting. Validated by a deterministic
behavioural eval (`tests/simulation/eval_foundation.py`) on both
schedule paths, plus full unit + movement suites. The single optional
remaining item is a real-Gemma bridge-objector confirmation run, which —
now that goals can actually complete — should finally answer the original
emergent-behaviour questions (does the bridge succeed around the objector,
does sentiment shift). The "living world" arc is the next major direction
(see Active Sub-Roadmaps).
**Last Updated:** 2026-06-07
**Notes:** Phase 8 complete. Memory v2 phases K, H, I shipped
(see MEMORY_V2_ROADMAP.md). Phase J parked. Foundation rebuild shipped
2026-06-07. Unit suite green (~1353).

**Hardware-constrained pause (2026-04-24):** The bridge-objector
diagnostic (`tests/simulation/diagnostic_bridge_objector.py`) uses
local Gemma-e2b to get non-deterministic NPC cognition — necessary
because MockProvider sims are deterministic and hide exactly the
emergent cases we need to see. Smoke-tested through day 1 on the
current Mac (conscientious objector Jasper, opposes:repair_bridge=0.9,
score=-0.60, p=14.1 %). Full 30-60 day run deferred: Gemma-e2b on
this hardware produces ~1 sim day per 30 wall-minutes with 10 NPCs.
Next concrete step when better hardware is available is a
`--days=30` run followed by reading the daily logs for voiced dissent,
bridge goal success/failure pattern, and sentiment shifts around the
objector. Decisions on Phase J and further Memory v2 work wait on
that evidence. Details + reasoning in MEMORY_V2_ROADMAP.md under
"Emergent-behaviour pivot" and "Phase J — PARKED".

**Phase 8 Player Integration (2026-04-11):**

**PlayerAgent** (`core/player/player_agent.py`): Composition-based player wrapping an
NPC. `PlayerAgent.create()` spawns a "Traveller" NPC with player-specific defaults
(gold=50, move_speed=3.0). Two awareness modes: `INDISTINGUISHABLE` (NPCs treat player
as another NPC) and `KNOWN_HUMAN` (NPCs acknowledge the player as a visitor).
Server-authoritative movement: `movement_tick()` validates direction against grid,
lerps smoothly, clamps to tile boundaries. `find_player_spawn()` finds passable tile
near grid centre (0,0). Player NPC registered in NPCManager — perceived by other NPCs,
included in conversations, participates in the economy.

**Server Wiring** (`server/main.py`): `handle_message()` now async. Three new message
types: `player_move` (sets direction, position broadcast via tick), `player_chat` (routes
through `initiate_conversation`/`continue_conversation` with full LLM + memory context),
`player_trade` (creates TradeOffer, NPC evaluates via `evaluate_trade_heuristic`). Player
data included in every tick broadcast and init message. Camera focus auto-tracks player
position for tier assignment.

**Client Modules:**
- **PlayerControls** (`player_controls.js`): WASD/arrow input, smooth camera follow
  with lerp, interaction radius ring indicator, Tab toggles camera mode (follow/orbit).
- **ChatUI** (`chat_ui.js`): Press E near NPC to open chat panel. Text input with Enter
  to send. Conversation history with player/NPC/system message styling. Auto-targets
  closest NPC. Escape to close.
- **TradeUI** (`trade_ui.js`): Press T to open trade panel. Item name/qty inputs for
  offer and request. Gold slider. NPC evaluates and responds. Inventory display.
- **HUD** (`hud.js`): Health/energy/hunger stat bars, minimap (canvas 2D with NPC dots
  and player marker), nearby NPC list, notification feed with auto-dismiss.

**main.js** rewritten to integrate all modules. OrbitControls disabled when player active.
Tick handler distributes data to player controls, chat, trade, HUD, and minimap.

24 new unit tests covering: creation, all 4 movement directions, blocked movement,
out-of-bounds, speed application, nearby NPC detection, awareness modes, spawn point
selection, and serialisation.

**Post-Phase 6 Fixes (2026-03-28):**

**Building Pathing Fix**: A* pathfinding no longer allows NPCs to walk into building
interiors. `_resolve_passable_goal()` redirects non-passable goals to nearest passable
neighbour. `resolve_schedule_location()` now maps "town_square" → civic building door,
"outskirts" → farm door, fallback → home (never raw coordinates that might be inside
buildings).

**Movement Variation**: NPCs spawn with randomized move_speed (1.6–2.4 tiles/sec).
Slot transitions now queue staggered departures (0–3 game-minute random delay per NPC)
instead of all NPCs navigating simultaneously. `_process_pending_departures()` dispatches
them as delays expire each tick.

**Emergency Movement Override**: `force_navigate_all()` bypasses stagger for emergencies
(kaiju attack, fire, rally). Supports flee_from mode (NPCs scatter away from danger),
filter_fn (target specific NPCs), and instantly clears pending departures. Single-NPC
override via `force_navigate_npc()`.

**Seed Memories**: NPCs now wake up with foundational memories — 5 occupation-specific
(identity, craft knowledge, motivation, aspiration, concern), 3 universal (town knowledge,
community awareness), plus backstory. Goals stored in structured memory. Identity facts
seeded. Based on Stanford Generative Agents' seed memory approach.

**Deterministic Planner** (core/npc/cognition/planner/): Modular Sims-style utility
scoring + Total War-style execution rules. Four independent, swappable components:

- **ActionRegistry** (actions.py): 12 default actions (eat, sleep, work, gather, trade,
  craft, socialise, wander, flee, rest, patrol, pray). Each ActionDef carries need_weights,
  personality_weights, time_weights, preconditions, target selectors, tags, metadata.
  Runtime extensible: register/remove/replace/by_tag.
- **UtilityScorer** (utility.py): Exponential need curves (Sims-style urgency), Big Five
  personality modifiers, time-of-day weighting. Pluggable need curves (exponential, linear,
  step) and per-action custom scorers. Pluggable need extractor for custom needs.
- **RuleRegistry** (rules.py): Per-action execution rules with fallthrough chains.
  PlannedAction output matches ScheduleEntry interface. 12 built-in rule sets + generic
  fallback. Custom rule sets registerable at runtime.
- **ContextBuilder** (context.py): Lightweight world snapshot (nearby NPCs, resources,
  buildings, threats, time). Subclassable for custom context fields. No LLM, no memory
  queries — pure data.
- **DeterministicPlanner** (__init__.py): Orchestrator. Every component injectable via
  constructor. plan_action() and score_all() as public API.

**Cognition Router** (core/npc/cognition/router/): Smart dispatcher between LLM
and deterministic. Per-decision-type routing with three modes (llm, deterministic,
auto). Auto mode scores importance based on proximity, novelty, budget pressure,
and scene pressure. Components:

- **TokenBudget** (budget.py): Daily token limit with reserve fraction for priority
  decisions. Throughput tracking (concurrent calls, rate limiting). Per-purpose usage
  stats. Auto-reset on configurable period.
- **CognitionPolicy** (policy.py): User-configurable routing rules per decision type.
  AutoConfig for threshold tuning. Priority NPCs (always LLM). Serialisable (to_dict/
  from_dict). Four preset policies: all_llm, all_deterministic, conversations_only,
  local_llm.
- **CognitionRouter** (__init__.py): Main dispatcher. route() and route_batch() API.
  Scene pressure awareness (auto_downgrade_threshold). Hot-swappable policy, budget,
  and importance scorer. Runtime route changes, priority NPC management. Full stats.

**Mistral Provider** (core/npc/mistral_provider.py): Implements LLMProvider for
Mistral API. mistral-small-latest for NPCs, mistral-large-latest for overseer.
Async, rate-limited, with cost tracking. API key in .env.

**COGNITION_GUIDE.md**: Full user documentation covering architecture, routing modes,
quick start, auto-mode tuning, planner components, budget management, priority NPCs,
emergency scenarios, runtime configuration, monitoring, and file reference.

**Integration Wiring (2026-03-28):** All systems now connected into the live game loop:

- **CognitionRouter** wired into NPCManager: schedule generation (step 2), perception
  reactions (step 4), post-conversation reflections, and periodic reflections all route
  through router.route(). When router returns DETERMINISTIC, planner generates the action.
- **DeterministicPlanner** wired as handler for deterministic-routed decisions.
  `_generate_deterministic_schedule()` converts PlannedAction to ScheduleEntry.
- **EconomyTick** (core/npc/economy_tick.py): Thin orchestrator holding ResourceManager,
  TradeManager, CraftingManager, ConstructionManager. Runs resource regen, auto-completes
  gathering/crafting sessions each tick (step 5c). Provides resource node dicts and recipe
  lists to planner context.
- **MistralProvider** wired as selectable LLM backend in server/main.py: Claude > Mistral
  > Mock priority chain based on available API keys.
- Economy and cognition stats included in get_state() and _build_tick_state() broadcasts.
- All components accept dependency injection via NPCManager constructor.

668 tests passing (12 new integration wiring tests).

**Diagnostic Experiment & Desync Fixes (2026-03-29):**

**NPC Synchronisation Fix**: Root cause identified — single shared `random.Random(seed)`
drove all NPC timing into lockstep after extended simulation. Three fixes applied:
- **Per-NPC RNG**: Each NPC gets its own `random.Random(hash((seed, npc_id)))`. All
  decomposition, stagger delays, and schedule variation use `npc._rng` instead of shared RNG.
- **Schedule variation**: `_template_schedule()` uses per-NPC RNG. 30% chance to replace
  afternoon with personal activity (visit friend, stroll, browse market).
- **Subtask duration jitter**: Every subtask gets ±20% duration from NPC's own RNG.
  Stagger pauses between queue refills widened to 5–30 game minutes.

**14-Day Diagnostic Experiment**: Instrumented simulation with structured JSONL logging,
per-NPC concrete goals, and post-run analysis. Results:
- Sync score: 0.24 (target <0.2 — slot transitions still cluster; needs reactive re-planning)
- Subtask variety: 57.5 unique/day (target ≥8 — PASS)
- Goal progress: 9/10 NPCs completed ≥3/5 substeps, 7/10 fully complete (PASS)
- Files: `tests/simulation/diagnostic_instrumented_sim.py`, `analyse_diagnostic.py`, `goals.py`

**Door Placement Fix**: Doors moved from one tile outside building to the building's south
wall (last row of footprint). Door tile is walkable; approach tile (one south) also walkable.
Road connections start from approach tile. Client renderer snaps door mesh to building's
south face. All 15 movement tests + 17 door validations pass.

657 unit tests + 15 simulation tests passing.
