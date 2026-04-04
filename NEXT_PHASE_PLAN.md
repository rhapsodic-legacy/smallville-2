# Next Phase: Per-NPC Commands & Emergent Behaviour

## Context

The Stanford-style action-duration model is complete and stable. NPCs cycle through
schedule entries by duration, walk home naturally for sleep, pathfind around buildings,
and avoid resting overlaps. The generator now routes roads around buildings with A*.

This document covers the next two features, ordered by complexity.

---

## Phase A: Per-NPC Custom Schedules (Player Commands)

### Goal
Allow the player (or game logic) to assign specific schedules to individual NPCs.
Example: "Bob guards the bridge all day, only leaving to sleep."

### What Already Exists
- `NPC.daily_schedule` is a list of `ScheduleEntry` objects with activity, location,
  duration, and optional `target_x`/`target_z` coordinates
- `_dispatch_to_entry()` in `core/npc/manager.py` resolves locations and navigates
- `_force_template_schedule()` in `core/npc/cognition/plan.py` generates schedules
  from occupation templates
- The schedule system is fully per-NPC — each NPC has independent timing

### Implementation Plan

#### 1. Custom Schedule API (core/npc/manager.py)
Add a method to NPCManager:
```python
def assign_custom_schedule(self, npc_id: str, entries: list[dict]) -> bool:
    """
    Assign a custom schedule to a specific NPC, overriding their template.

    entries format: [
        {"activity": "stand guard at the bridge", "location": "bridge",
         "target_x": 15, "target_z": 0, "duration_minutes": 900},
        {"activity": "walk home and sleep", "location": "home",
         "duration_minutes": 540},
    ]

    Entries must sum to 1440 minutes. Returns False if validation fails.
    """
```

This converts the dicts to ScheduleEntry objects, validates the 1440-minute sum,
sets `npc.daily_schedule`, resets `schedule_index` and `action_start_minutes`, and
marks the NPC so template regeneration doesn't overwrite the custom schedule.

#### 2. Custom Schedule Flag (core/npc/models.py)
Add a field to NPC:
```python
has_custom_schedule: bool = False
```
When True, `_advance_npc_action` regenerates the same custom schedule instead of
calling `_force_template_schedule`. The schedule loops rather than being replaced.

#### 3. Location Resolution for Custom Targets
Custom schedules can specify either:
- `target_x`/`target_z` — exact coordinates (e.g., "the bridge at 15,0")
- `location` — symbolic name resolved by `resolve_schedule_location` ("home", "work",
  "tavern", "town_square")

Add support for named landmarks (bridge, gate, well) by scanning buildings and
terrain features. Store in a lookup dict on NPCManager at init.

#### 4. REST/WebSocket API (server/main.py)
Expose the custom schedule through the existing WebSocket protocol:
```json
{
    "type": "assign_schedule",
    "npc_id": "guard_0",
    "entries": [
        {"activity": "stand guard at the bridge", "target_x": 15, "target_z": 0,
         "duration_minutes": 900},
        {"activity": "walk home and sleep", "location": "home",
         "duration_minutes": 540}
    ]
}
```

Also add a REST endpoint: `POST /api/npc/{npc_id}/schedule`

#### 5. Client UI (optional, can be deferred)
A simple panel: click an NPC, type a natural-language command ("guard the bridge"),
which gets sent to the LLM to generate schedule entries. This is a nice-to-have
that can come after the API works.

### Files to Modify
- `core/npc/models.py` — add `has_custom_schedule` field
- `core/npc/manager.py` — add `assign_custom_schedule()`, update `_advance_npc_action`
  to loop custom schedules instead of regenerating templates
- `core/npc/cognition/plan.py` — add landmark resolution helper
- `server/main.py` — add WebSocket handler and REST endpoint
- `tests/unit/test_npc_manager.py` — test custom schedule assignment and looping

### Estimated Scope
Small. The schedule plumbing exists — this is mostly wiring + validation + API.

---

## Phase B: Emergent Behaviour from Goals & Relationships

### Goal
NPCs make autonomous decisions based on personality, relationships, goals, and
memory — not just template schedules. The Martha/Bob example: Martha brings Bob
lunch not because it's programmed, but because she wants to make him happy, she
knows he's at the bridge, and at noon she reflects that he's probably hungry.

This is the core Stanford Generative Agents innovation — reflective planning.

### What Already Exists

#### Memory System (core/memory/)
- **Structured memory** (`structured.py`): SQLite-backed knowledge graph storing
  facts (subject, predicate, object), goals, and relationships
- **Episodic memory** (`episodic.py`): ChromaDB embeddings for experiences, with
  recency/importance/relevance scoring
- **Memory manager** (`manager.py`): Unified retrieval combining both stores,
  plus `get_relationship_context()` for conversation prompts
- **Reflection** (`reflection.py`): Importance accumulator, focal point extraction,
  and insight synthesis (scaffolded but not actively generating schedule changes)

#### Relationship System (core/relationships/)
- **Sentiment tracker** (`sentiment.py`): Per-pair dimensions (trust, fear, respect,
  affection, debt) with sparse storage
- **Faction model** (`factions.py`): Groups with roles, allies, rivals

#### Cognition System (core/npc/cognition/)
- **Tiered cognition** (`tiers.py`): Tier 1 (full LLM), Tier 2 (simplified LLM),
  Tier 3 (template/state machine), Tier 4 (frozen)
- **Router** (`router.py`): Decides which tier handles each cognitive task based
  on proximity to player/focus and NPC relevance
- **Planner** (`plan.py`): Has `_llm_schedule()` for Tier 1/2 NPCs (sends personality,
  goals, backstory to LLM to generate a daily schedule). Currently only used when
  `deterministic=False` and NPC is Tier 1/2.

### What Needs to Be Built

#### 1. Relationship Facts in Memory
When NPCs are spawned, seed relationship facts into structured memory:
```
("martha_0", "spouse_of", "bob_0")
("martha_0", "cares_about", "bob_0")
("bob_0", "assigned_to", "bridge_guard_post")
```

These facts get retrieved during LLM planning and reflection, informing schedule
decisions. The structured memory already supports `add_fact()` — this just needs
to be called during spawn with relationship data.

**Where:** `core/npc/manager.py` in `_seed_memories()`, using a new relationship
configuration (either from WorldConfig or a separate relationships definition).

#### 2. Dynamic Schedule Injection from Reflection
The reflection system (`core/memory/reflection.py`) already accumulates importance
scores and generates insights. What's missing is the final step: insights that
trigger schedule modifications.

The flow:
1. Martha perceives "Bob has been at the bridge since 06:00" (perception system)
2. Martha's episodic memory accumulates importance (> threshold)
3. Reflection triggers → LLM generates insight: "Bob must be hungry by now. I should
   bring him lunch."
4. **New step:** Insight is classified as "action_intent" and injected into Martha's
   schedule as a temporary entry: `("bring lunch to Bob", target=bridge, duration=30min)`

**Implementation:**
- Add an `action_intent` classification to reflection output
- When a reflection produces an action_intent, insert a ScheduleEntry into the NPC's
  `daily_schedule` at the current `schedule_index + 1` position
- The entry gets a short duration (15-60 min) and specific target coordinates
- After it completes, the NPC resumes their normal schedule

**Where:** `core/memory/reflection.py` (classify insights), `core/npc/manager.py`
(inject entries), new prompt template in `core/npc/llm_client.py`

#### 3. Enhanced Perception → Memory Pipeline
Currently, perceptions are stored as strings in `npc.recent_perceptions` (a short
rolling list). For emergent behaviour, perceptions need to flow into episodic memory
with proper importance scoring:

- "I see Bob standing at the bridge" → low importance, stored
- "Bob said 'I'm starving'" → high importance (hunger + relationship = concern)
- "The tavern is on fire" → very high importance (triggers reflection immediately)

**Implementation:**
- In `core/npc/cognition/perceive.py`, after generating perceptions, score importance
  (LLM for Tier 1, heuristic for Tier 2+) and store in episodic memory
- Add a `store_perception()` method to MemoryManager that handles importance scoring
  and storage in one call

**Where:** `core/npc/cognition/perceive.py`, `core/memory/manager.py`

#### 4. Goal-Directed Planning (Tier 1 Enhancement)
The existing `_llm_schedule()` in `plan.py` asks the LLM to generate a full day
schedule. For emergent behaviour, Tier 1 NPCs need **mid-day replanning**:

- Every N game-minutes (configurable, e.g. 60), Tier 1 NPCs re-evaluate their schedule
- The LLM receives: current schedule, recent perceptions, recent reflections, active
  goals, relationship state
- It can modify remaining entries (insert, remove, reorder)
- This is how Martha decides "actually, I'll skip gardening and bring Bob lunch instead"

**Implementation:**
- Add a `replan_interval_minutes` to tier config (Tier 1 = 60, Tier 2 = 120, others = never)
- Track `last_replan_minutes` on each NPC
- In `cognition_tick`, after the duration check, call `_maybe_replan(npc)` for eligible tiers
- The replan prompt includes recent memories and produces a modified schedule suffix

**Where:** `core/npc/cognition/plan.py` (replan logic), `core/npc/manager.py`
(replan trigger in cognition_tick), `core/npc/llm_client.py` (replan prompt template)

#### 5. Conversation → Memory → Behaviour Loop
Conversations already generate dialogue but the content doesn't persist meaningfully.
For Bob to tell Martha "I'm starving" and have it matter:

1. Conversation exchanges are already stored (in `Conversation.exchanges`)
2. **New:** After conversation ends, summarise key information and store in both
   participants' episodic memory with appropriate importance
3. **New:** Tag conversation memories with relationship context so retrieval
   during planning surfaces them
4. Bob's "I'm starving" becomes a high-importance memory for Martha
5. Martha's next reflection/replan picks it up and generates "bring lunch to Bob"

**Implementation:**
- In `end_conversation()` (`core/npc/cognition/converse.py`), after setting finished=True,
  call a new `_store_conversation_memories()` that:
  - Summarises the conversation (LLM for Tier 1, template for others)
  - Stores summary in both NPCs' episodic memory
  - Extracts facts ("Bob is hungry") into structured memory
- Wire this through MemoryManager

**Where:** `core/npc/cognition/converse.py`, `core/memory/manager.py`

### Architecture Diagram

```
Perception ──→ Episodic Memory ──→ Importance Accumulator
                    │                       │
                    ▼                       ▼
              Retrieval ◄──── Reflection (focal points → insights)
                    │                       │
                    ▼                       ▼
         Structured Memory          Action Intents
          (facts, goals,                   │
           relationships)                  ▼
                    │              Schedule Injection
                    ▼                       │
              LLM Planner ◄────────────────┘
                    │
                    ▼
            Daily Schedule
            (with dynamic
             entries)
                    │
                    ▼
              Execute Tick
         (duration-based cycling)
```

### Prompt Templates Needed

#### Replan Prompt (Tier 1)
```
You are {name}, a {age}-year-old {occupation} in Smallville.
Personality: {personality}
Current goals: {goals}

It is currently {time}. Your remaining schedule:
{remaining_entries}

Recent events:
{recent_perceptions}

Recent reflections:
{recent_insights}

People you care about:
{relationship_summary}

Based on these events and your goals, should you modify your remaining schedule?
If yes, output the new remaining entries. If no, output "NO_CHANGE".
```

#### Conversation Memory Extraction
```
Summarise this conversation between {npc_a} and {npc_b} in 1-2 sentences.
Then list any important facts learned (format: "subject | predicate | object").

Conversation:
{exchanges}
```

### Dependencies and Ordering

Build in this order — each step is independently useful:

1. **Relationship facts in memory** (foundation — no LLM cost, enables all later steps)
2. **Conversation → memory pipeline** (conversations become meaningful)
3. **Enhanced perception → memory** (NPCs notice and remember what they see)
4. **Dynamic schedule injection from reflection** (insights trigger actions)
5. **Goal-directed replanning** (full emergent behaviour loop)

Steps 1-3 work with the current template schedule system. Steps 4-5 require Tier 1
LLM calls and produce the Martha/Bob emergent behaviour.

### LLM Cost Considerations

- Tier 1 NPCs (near player/focus) get full LLM planning — expensive
- Tier 2 NPCs get simplified prompts — moderate
- Tier 3+ NPCs stay on template schedules — zero LLM cost
- The tiered system means only 2-5 NPCs use LLM planning at any time
- Replan calls are throttled (every 60 game-minutes = ~80 seconds real time)
- Reflection is gated by importance threshold (not every tick)

### Testing Strategy

- **Unit tests:** Mock LLM, verify schedule injection mechanics, memory storage
- **Integration tests:** Real Haiku API, verify replan produces valid schedules
- **Simulation tests:** Multi-day runs checking that Tier 1 NPCs deviate from
  templates meaningfully while Tier 3 NPCs stay on template

### Files Summary

| File | Changes |
|------|---------|
| `core/npc/models.py` | `has_custom_schedule`, `last_replan_minutes` fields |
| `core/npc/manager.py` | `assign_custom_schedule()`, replan trigger, relationship seeding |
| `core/npc/cognition/plan.py` | `_maybe_replan()`, landmark resolution |
| `core/npc/cognition/converse.py` | `_store_conversation_memories()` |
| `core/npc/cognition/perceive.py` | Importance scoring, memory storage |
| `core/memory/manager.py` | `store_perception()`, conversation memory helpers |
| `core/memory/reflection.py` | Action intent classification, schedule injection |
| `core/npc/llm_client.py` | Replan prompt, conversation extraction prompt |
| `server/main.py` | WebSocket + REST handlers for custom schedules |
| `tests/unit/test_npc_manager.py` | Custom schedule tests |
| `tests/integration/test_replan.py` | LLM replanning integration test |

---

## Quick Reference: Current Schedule Architecture

For context when implementing, here's how the current system works:

### Schedule Entry
```python
@dataclass
class ScheduleEntry:
    slot: str                    # "morning", "night", etc. (informational)
    activity: str                # "work at the forge", "walk home and sleep"
    location: str                # "home", "work", "tavern", "town_square"
    priority: int                # 1-10 (higher = harder to interrupt)
    target_x: int | None         # explicit coordinates (or None for resolution)
    target_z: int | None
    duration_minutes: int        # how long this entry lasts in game time
```

### Duration Cycling (core/npc/manager.py, cognition_tick step 3)
Each NPC independently tracks:
- `schedule_index` — current position in `daily_schedule`
- `action_start_minutes` — game time when current entry started
- Entry advances when `current_minutes - action_start_minutes >= entry.duration_minutes`
- When schedule exhausted, regenerates (template or LLM) and resets index to 0

### Location Resolution
`resolve_schedule_location()` maps symbolic locations to coordinates:
- "home" → `(npc.home_x, npc.home_z)`
- "work" → `(npc.work_x, npc.work_z)`
- "tavern" → nearest tavern door
- "town_square" → (0, 0) with spread
