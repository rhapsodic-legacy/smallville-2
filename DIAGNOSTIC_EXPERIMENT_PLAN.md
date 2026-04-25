# Diagnostic Experiment: Why NPCs Synchronise After Extended Simulation

## The Problem

After 43 simulated days, all NPCs move in synchronised back-and-forth patterns.
None go anywhere purposefully — just identical bursts of movement.

## Root Cause (Identified)

A single shared `random.Random(seed)` instance generates ALL NPC stagger delays,
subtask selections, and queue refills. After thousands of deterministic calls,
the RNG sequence stabilises. Combined with identical occupation-based schedule
templates, all NPCs of the same occupation receive identical timing, identical
subtasks, and identical destinations — locking into perfect synchronisation.

**Specific code paths:**
- `manager.py:79` — RNG seeded once, never reseeded
- `manager.py:283-285` — slot transitions fire for ALL NPCs in one tick
- `manager.py:396,422` — stagger delays drawn sequentially from shared RNG
- `manager.py:431-450` — empty queue refill uses same shared RNG
- `plan.py:29-72` — identical schedule templates per occupation
- `decompose.py:181-223` — subtask selection uses passed-in shared RNG

## What Stanford Did (Reference Implementation)

Stanford's system worked because of three key design choices that prevent
synchronisation. Our experiment must replicate these.

### Stanford's Cognitive Cycle (per tick)

```
1. PERCEIVE → 2. RETRIEVE → 3. PLAN → 4. REFLECT → 5. EXECUTE
```

### Stanford's 3-Level Plan Decomposition

| Level | Timescale | What It Produces | How |
|-------|-----------|-----------------|-----|
| **Daily Plan** | 24 hours | 4-6 broad activities ("have breakfast at 7am") | LLM generates from persona description + yesterday's summary |
| **Hourly Schedule** | 60 min slots | 24 time blocks ("eating breakfast: 60 min") | LLM decomposes daily plan into hour-by-hour |
| **5-Minute Actions** | 5 min each | ~12 concrete actions per hour ("pour orange juice") | LLM decomposes each hourly block on-the-fly |

### Why Stanford's System Doesn't Synchronise

1. **Per-NPC LLM calls**: Each NPC's daily plan, hourly schedule, and 5-minute
   decompositions are generated individually by the LLM. Two blacksmiths get
   DIFFERENT plans because LLM output is non-deterministic.

2. **Reactive re-planning**: When an NPC perceives something new (another NPC,
   an event), it can re-plan mid-action. This breaks lockstep because different
   NPCs perceive different things at different times.

3. **Memory-driven individuality**: Each NPC's plan is influenced by their
   unique accumulated memories. Isabella plans differently from Klaus because
   she remembers different things.

4. **No shared RNG**: Stanford doesn't use a global RNG for timing. Each NPC's
   action duration comes from their individual LLM-generated plan.

### What We're Missing vs Stanford

| Stanford Has | We Have | Gap |
|-------------|---------|-----|
| Per-NPC LLM daily plans | Shared occupation templates | Plans are identical |
| Per-NPC LLM 5-min decomposition | Template decomposition from shared RNG | Subtasks synchronise |
| Reactive re-planning on perception | No re-planning | NPCs can't break lockstep |
| Memory-influenced planning | Memory exists but doesn't affect plans | No individuality over time |
| Per-NPC timing from LLM | Shared RNG timing | Identical delays |

---

## The Experiment

### Goal

Run a multi-day simulation with **concrete, measurable NPC goals** where every
decision, action, memory, and metric is logged. Observe whether NPCs accomplish
their goals, and when/why behaviour degrades.

### Design Principles

1. **Every NPC gets a concrete, unique goal** with sequential substeps
2. **Every step is measurable** — did it happen? When? How long?
3. **Everything is logged** — decisions, memories, actions, locations, timing
4. **Run for 10+ simulated days** at accelerated speed
5. **Post-run analysis** reads the logs to identify exactly when and why
   behaviour breaks down

### Phase 1: Fix the Synchronisation Bug

Before the experiment, fix the identified root cause:

**Fix 1: Per-NPC RNG** (eliminates shared RNG synchronisation)
```python
# In NPC.__init__ or spawn:
npc._rng = random.Random(hash((seed, npc.npc_id)))
```
- Each NPC gets its own RNG seeded from (global_seed + npc_id)
- Deterministic per-NPC but independent between NPCs
- Pass `npc._rng` instead of `self.rng` to decompose/stagger functions

**Fix 2: Schedule variation** (eliminates identical templates)
- Add ±30 min random offset to each schedule entry's start time (per-NPC RNG)
- Randomise the order of same-slot activities
- Add 1-2 "personal" entries per NPC (visit a friend, take a walk)

**Fix 3: Subtask duration jitter** (eliminates identical timing)
- Each subtask duration gets ±20% jitter from the NPC's own RNG
- Ensures even identical subtask lists play out at different speeds

### Phase 2: Build the Instrumented Simulation

#### 2a: NPC Goal Assignment

Each NPC receives a **unique concrete goal** with 3-5 sequential substeps.
These goals are achievable within the simulation's systems.

**Example goals (10 NPCs):**

| NPC | Goal | Substeps |
|-----|------|----------|
| Blacksmith | Forge 3 iron tools | 1. Gather iron ore 2. Smelt at furnace 3. Forge tool 1 4. Forge tool 2 5. Forge tool 3 |
| Farmer 1 | Harvest and sell 5 bushels | 1. Tend crops (morning) 2. Harvest field 3. Carry to market 4. Sell to merchant 5. Return home |
| Farmer 2 | Build a fence around the farm | 1. Gather wood 2. Cut planks 3. Dig post holes 4. Erect fence sections 5. Complete perimeter |
| Merchant | Accumulate 50 gold from trading | 1. Open stall 2. Buy from farmer 3. Mark up goods 4. Sell to townsfolk 5. Count earnings |
| Tavern Keeper | Serve 20 customers | 1. Open tavern 2. Prepare food 3. Serve drinks 4. Clean up 5. Close for night |
| Priest | Hold 3 sermons this week | 1. Prepare sermon notes 2. Ring bell 3. Deliver sermon 4. Counsel attendees 5. Record in journal |
| Guard | Complete 5 full patrols | 1. Check north gate 2. Walk east perimeter 3. Check south road 4. Walk west side 5. Report to town hall |
| Labourer 1 | Chop 10 trees for lumber | 1. Walk to forest 2. Select tree 3. Chop tree 4. Carry logs 5. Stack at lumber yard |
| Labourer 2 | Dig a well | 1. Choose location 2. Dig first layer 3. Dig second layer 4. Line with stones 5. Test water flow |
| Farmer 3 | Deliver food to 3 homes | 1. Harvest vegetables 2. Pack basket 3. Deliver to home 1 4. Deliver to home 2 5. Deliver to home 3 |

#### 2b: The Logging System

Every tick, log to a structured JSON-lines file (`diagnostic_log.jsonl`):

```json
{
  "tick": 1423,
  "game_time": "Day 3, 14:22",
  "npc_id": "blacksmith_0",
  "npc_name": "Aldric",
  "event_type": "ACTION",
  "data": {
    "activity": "walking",
    "position": [12, 8],
    "destination": [6, 10],
    "subtask": "carry iron ore to furnace",
    "subtask_time_remaining": 4.2,
    "queue_depth": 3,
    "goal": "Forge 3 iron tools",
    "goal_step": 2,
    "goal_step_description": "Smelt at furnace",
    "schedule_slot": "morning",
    "schedule_entry": "work at the forge",
    "stagger_delay": 0.0,
    "rng_state_hash": "a3f2..."
  }
}
```

**Event types logged:**
- `TICK_STATE` — position, activity, subtask, queue depth (every tick)
- `SLOT_TRANSITION` — schedule slot changed, new entry, stagger delay assigned
- `SUBTASK_START` — new subtask began, description, duration
- `SUBTASK_COMPLETE` — subtask finished, time taken vs expected
- `QUEUE_REFILL` — subtask queue was empty, refilled with what
- `GOAL_STEP_COMPLETE` — a goal substep was achieved
- `GOAL_COMPLETE` — entire goal achieved
- `PERCEPTION` — NPC perceived something, what was it
- `MEMORY_STORED` — new memory added, content, importance
- `DECISION` — NPC made a decision, what influenced it
- `PATH_ASSIGNED` — new path computed, from/to, length
- `ARRIVAL` — NPC arrived at destination

#### 2c: Metrics Tracked

**Per-NPC metrics (computed from logs):**
- Goal progress: which substep, time per substep, completion %
- Distance travelled per day
- Time spent idle vs active vs walking
- Unique locations visited per day
- Subtask variety: unique subtask descriptions per day
- Synchronisation score: correlation of movement timing with other NPCs

**Population metrics (computed from logs):**
- Simultaneous departure count per tick (sync indicator)
- Simultaneous arrival count per tick (sync indicator)
- Position overlap count (stacking indicator)
- Activity distribution per tick (how many idle/walking/working)
- Goal completion rate across population
- Memory accumulation rate

#### 2d: Post-Run Analysis Script

`tests/simulation/analyse_diagnostic.py` reads the log file and produces:

1. **Timeline report**: Per-NPC goal progress over simulated days
2. **Sync detection**: Ticks where ≥3 NPCs change state simultaneously
3. **Behaviour degradation**: Does subtask variety decrease over time?
4. **Idle analysis**: When do NPCs go idle? Why? (queue empty? path stuck?)
5. **Memory report**: What memories accumulated? Did they influence plans?
6. **Heat map**: Where do NPCs spend time? Does it narrow over time?

### Phase 3: Run the Experiment

```
1. Apply Phase 1 fixes (per-NPC RNG, schedule variation, duration jitter)
2. Run instrumented simulation: 10 NPCs, 14 simulated days, accelerated
3. Analysis script processes the log file
4. Report identifies:
   - Did each NPC make progress on their goal?
   - When (if ever) did synchronisation appear?
   - What caused it?
   - Did memories influence behaviour?
```

**Runtime**: 14 simulated days at 1-second ticks = 20,160 ticks.
At accelerated speed (skip LLM, use templates), ~5 minutes real time.

### Phase 4: Iterate

Based on Phase 3 results:
- If sync reappears: the log will show exactly when and what triggered it
- If goals stall: the log will show which substep got stuck and why
- If memory doesn't influence behaviour: wire memory retrieval into planning
- If all works: extend to 43+ days and verify stability

---

## File Structure

```
tests/simulation/
  diagnostic_instrumented_sim.py    — The instrumented simulation runner
  analyse_diagnostic.py       — Post-run log analysis and reports
  diagnostic_log.jsonl        — Output: per-tick structured logs

core/npc/
  manager.py                  — Fix: per-NPC RNG, modified stagger
  cognition/
    decompose.py              — Fix: accept per-NPC RNG
    plan.py                   — Fix: schedule variation per NPC
    execute.py                — Already correct (uses trail)

.claude/skills/
  run_experiment.md           — Skill to run experiment + analysis
```

---

## Success Criteria

The experiment succeeds if:
1. **No synchronisation** at day 14 (sync score < 0.2)
2. **Goal progress** — ≥7/10 NPCs complete ≥3/5 substeps
3. **Subtask variety** — unique descriptions per day stays above 8 (of 10 NPCs)
4. **Logs are readable** — analysis script produces clear, actionable output

The experiment fails informatively if any criterion isn't met, because the
logs show exactly what went wrong and when.

---

## Estimated Work

| Task | Effort |
|------|--------|
| Phase 1: Per-NPC RNG fix | Small — swap `self.rng` for `npc._rng` |
| Phase 1: Schedule variation | Small — add jitter to entry times |
| Phase 2a: Goal assignment system | Medium — goal model + substep tracking |
| Phase 2b: Logging system | Medium — structured JSON logger |
| Phase 2c: Metrics | Small — computed from logs |
| Phase 2d: Analysis script | Medium — reads logs, produces report |
| Phase 3: Run + interpret | Small — run script, read output |
| **Total** | **One focused session** |
