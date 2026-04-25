---
name: validate_intelligence
description: Automated NPC intelligence validation -- unit tests + 300-tick simulation verifying the Stanford 3-level cognition hierarchy. Runs automatically via hook on file changes.
---

# Validate Intelligence Pipeline

## What It Does
Two-stage automated validation for the NPC sub-task and cognition system:

### Stage 1: Unit Tests
5 targeted checks on task decomposition:
1. **All occupations decompose** -- blacksmith, farmer, merchant, tavern_keeper, priest, guard all produce 2+ sub-tasks
2. **Description variety** -- 10 decompositions of the same entry yield 5+ unique descriptions
3. **All activity types** -- sleep, eat, socialise, work, wander all produce sub-tasks
4. **Valid activity states** -- all sub-tasks use valid states (idle/working/eating/sleeping/talking/gathering)
5. **Building objects populated** -- buildings have interior objects after generation

### Stage 2: Simulation (300 ticks, 10 NPCs)
3 behavioural checks:
1. **Never idle** -- <20% of resting snapshots are generic "idle" (target: 0%)
2. **Description variety** -- 10+ unique sub-task descriptions across the simulation
3. **Timer advances** -- sub-tasks rotate at least 5 times (timer system works)

## How It Runs
**Automatically** -- a PostToolUse hook triggers when these files are edited:
- `decompose.py`, `execute.py`, `manager.py`, `models.py`, `llm_client.py`

## Manual Run
```bash
python3 tests/simulation/test_npc_intelligence.py
```

## Architecture: Stanford 3-Level Hierarchy
- **Level 1**: Daily schedule (schedule entries like "work at the forge, 4 hours")
- **Level 2**: Hourly decomposition into 2-6 sub-tasks (template or LLM)
- **Level 3**: Moment-to-moment execution (sub-task timer, activity state, description)

Key files:
- `core/npc/cognition/decompose.py` -- template + LLM decomposition
- `core/npc/cognition/execute.py` -- sub-task timer, queue advancement
- `core/npc/manager.py` -- wires decomposition into tick loop
- `core/npc/models.py` -- SubTask dataclass, NPC queue fields
