# Cognition System Guide

> How NPC intelligence works in Smallville 2 — and how to tune it.

This guide covers the hybrid cognition system that powers NPC decision-making.
Every NPC decision can be handled by an LLM (deep, creative reasoning) or a
deterministic planner (instant, free, rule-based). The **cognition router**
decides which system handles each decision, based on your configuration.

---

## Architecture Overview

```
NPC needs to decide something
        │
        ▼
┌─────────────────┐
│ Cognition Router │ ← Policy (your configuration)
│                  │ ← Budget (token tracking)
│                  │ ← Scene pressure (NPC count)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────────┐
│  LLM  │ │ Deterministic │
│       │ │   Planner     │
└───┬───┘ └──────┬───────┘
    │            │
    ▼            ▼
  Same output format: PlannedAction
  (target coords, description, activity state)
```

Both paths produce the same output — the rest of the engine doesn't
care which system made the decision.

---

## The Three Routing Modes

Every decision type can be set to one of three modes:

| Mode | Behaviour | Token Cost |
|------|-----------|------------|
| `"llm"` | Always use the LLM | Uses tokens |
| `"deterministic"` | Always use the planner | Free |
| `"auto"` | Router decides per-decision | Variable |

### Default routing

```python
{
    "daily_schedule":    "auto",          # LLM when budget allows
    "reaction":          "auto",          # LLM for novel events
    "conversation":      "llm",           # Always LLM — must feel real
    "trade_evaluation":  "deterministic", # Heuristic is fine
    "craft_choice":      "deterministic", # Utility scoring handles this
    "flee":              "deterministic", # Must be instant
    "reflection":        "auto",          # LLM when budget allows
    "gather_choice":     "deterministic", # Simple decision
    "work_choice":       "deterministic", # Simple decision
}
```

---

## Quick Start

### Use a preset policy

```python
from core.npc.cognition.router import (
    CognitionRouter,
    policy_all_deterministic,
    policy_conversations_only,
    policy_local_llm,
)

# Zero-cost mode: no LLM at all
router = CognitionRouter(policy=policy_all_deterministic())

# Budget-friendly: only conversations use LLM
router = CognitionRouter(policy=policy_conversations_only())

# Local LLM (limited throughput)
router = CognitionRouter(policy=policy_local_llm(max_concurrent=2))
```

### Custom policy

```python
from core.npc.cognition.router import CognitionRouter, CognitionPolicy

policy = CognitionPolicy(
    routing={
        "conversation": "llm",
        "daily_schedule": "auto",
        "reaction": "auto",
        "reflection": "llm",
        # Everything else: deterministic
        "trade_evaluation": "deterministic",
        "craft_choice": "deterministic",
        "flee": "deterministic",
    },
    token_budget_daily=200_000,       # Conservative budget
    reserve_fraction=0.3,             # 30% reserved for conversations
    auto_downgrade_threshold=10,      # >10 pending decisions = force deterministic
    priority_npcs={"blacksmith_0"},   # This NPC always gets LLM
    max_calls_per_minute=30,          # Rate limit for API
)

router = CognitionRouter(policy=policy)
```

---

## How "Auto" Mode Works

When a decision type is set to `"auto"`, the router computes an
importance score and compares it to a threshold:

```
score = base_type_importance
      × proximity_to_player
      × novelty
      × importance_weight
      × (1.0 - budget_pressure)
      ÷ scene_pressure
```

| Factor | What it means |
|--------|---------------|
| Base importance | Conversations = 0.7, reactions = 0.5, schedules = 0.4 |
| Proximity | NPCs near the camera score higher (visible = interesting) |
| Budget pressure | 0.0 when flush, 1.0 when depleted — conserves tokens late in the day |
| Scene pressure | 100 NPCs deciding at once → everyone gets downgraded |

If `score >= llm_threshold` → LLM. Otherwise → deterministic.

### Tuning auto mode

```python
from core.npc.cognition.router import AutoConfig

policy.auto_config = AutoConfig(
    llm_threshold=0.3,          # Lower = more LLM calls
    novelty_weight=1.5,         # Boost novel situations
    proximity_weight=1.0,       # Standard proximity effect
    importance_weight=1.2,      # Slightly boost all decisions
    scene_pressure_divisor=2.0, # Less aggressive downgrade under pressure
)
```

---

## The Deterministic Planner

When the router chooses deterministic, the planner handles the decision.
It has four independent, swappable components:

### 1. Action Registry (`planner/actions.py`)

Defines what actions NPCs can take. Ships with 12 defaults:

| Action | Tags | When it scores high |
|--------|------|---------------------|
| eat | survival | High hunger |
| sleep | survival | Low energy + nighttime |
| work | economy | Morning/afternoon + conscientious personality |
| gather | economy, outdoor | Resources nearby |
| trade | economy, social | Has inventory + market exists |
| craft | economy | Has materials + recipes available |
| socialise | social | Extroverted + evening |
| wander | basic | Open personality + nothing urgent |
| flee | survival, combat | Active threat |
| rest | basic | Low energy |
| patrol | guard, duty | Guard occupation + morning/night |
| pray | social | Church exists |

**Adding a custom action:**

```python
from core.npc.cognition.planner import ActionDef

planner.actions.register(ActionDef(
    action_id="fish",
    display_name="Go fishing",
    need_weights={"hunger": 1.5},
    personality_weights={"openness": 0.4},
    time_weights={"morning": 1.5, "afternoon": 1.0},
    base_utility=0.4,
    precondition=lambda npc, ctx: ctx.has_river,  # custom check
    target_selector=lambda npc, ctx: ctx.river_spot,
    tags={"economy", "outdoor", "relaxation"},
))
```

### 2. Utility Scorer (`planner/utility.py`)

Sims-style scoring. Each action gets a utility score based on:

- **Need urgency**: Exponential curves — hunger at 0.9 is drastically more urgent
  than at 0.5. This mirrors The Sims' need system.
- **Personality**: Big Five traits modify scores. Extroverts score socialising higher;
  conscientious NPCs score work higher.
- **Time of day**: Sleep scores high at night, work at morning.
- **Base utility**: Some actions have inherent value (work = productive).

**Need curves:**

```
Urgency
  1.0 │                        ╱
      │                      ╱
      │                    ╱
  0.5 │                 ╱
      │              ╱
      │          ╱╱
  0.0 │─────╱╱──────────────
      0.0      0.5      1.0   Need value
         Exponential curve
```

**Custom scorer:**

```python
from core.npc.cognition.planner import UtilityScorer, linear_curve

scorer = UtilityScorer(
    need_curve=linear_curve,        # Replace exponential with linear
    personality_multiplier=2.0,     # Double personality effects
)

# Or override scoring for a specific action:
scorer.set_custom_scorer("work", lambda npc, ctx, action: (
    5.0 if npc.occupation == "guard" else 1.0
))
```

### 3. Rule Registry (`planner/rules.py`)

Total War-style execution logic. Once the scorer picks an action, rules
determine HOW it executes in the world:

- **Gather**: Find nearest resource node → path to it → begin session
- **Flee**: Calculate vector away from threat → find passable destination
- **Socialise**: Head to tavern, or approach nearest NPC

Rules can be chained with fallthrough — if the first rule fails
(e.g. no nodes available), the next rule in the chain is tried.

**Custom rule:**

```python
from core.npc.cognition.planner import RuleSet, PlannedAction

def guard_rally_rule(npc, ctx, scored):
    """Guards rally to the town hall during emergencies."""
    if ctx.threat_level > 0.5 and npc.occupation == "guard":
        return PlannedAction(
            action_id="patrol",
            description="rallying to the town hall!",
            target_x=ctx.town_hall_x,
            target_z=ctx.town_hall_z,
            activity_state="walking",
        )
    return None  # fallthrough to default patrol rule

planner.rules.register(RuleSet("patrol", [guard_rally_rule, default_patrol]))
```

### 4. Context Builder (`planner/context.py`)

Gathers world state for the planner. Cheap to build — no LLM calls,
no memory queries. Subclass to inject custom data:

```python
from core.npc.cognition.planner import ContextBuilder

class WeatherAwareContext(ContextBuilder):
    def build(self, npc, all_npcs, current_slot, **kwargs):
        ctx = super().build(npc, all_npcs, current_slot, **kwargs)
        ctx.metadata["weather"] = self.weather_system.current
        ctx.metadata["is_raining"] = self.weather_system.is_raining
        return ctx
```

---

## Token Budget

The budget system tracks consumption and enforces limits:

```python
router.budget.get_stats()
# {
#     "tokens_used": 42350,
#     "tokens_remaining": 457650,
#     "daily_limit": 500000,
#     "budget_pressure": 0.085,
#     "calls_total": 127,
#     "by_purpose": {"conversation": 28000, "schedule": 14350}
# }
```

### Budget reserve

By default, 20% of the daily budget is reserved for **priority decisions**
(conversations and reflections). This ensures routine scheduling decisions
can't starve the high-value interactions that make NPCs feel alive.

### Budget for local LLMs

Local models have unlimited tokens but limited throughput:

```python
policy = CognitionPolicy(
    token_budget_daily=0,           # Unlimited tokens
    max_concurrent_llm_calls=2,     # Hardware can handle 2 at once
    max_calls_per_minute=20,        # ~3 seconds per call
    auto_downgrade_threshold=5,     # Aggressive downgrade under load
)
```

---

## Priority NPCs

Story-critical NPCs can be guaranteed LLM access regardless of
budget or scene pressure:

```python
router.add_priority_npc("blacksmith_0")   # Always gets LLM
router.add_priority_npc("villain_0")

# Remove when no longer story-critical
router.remove_priority_npc("villain_0")
```

---

## Emergency Scenarios

### Kaiju attack (100+ NPCs need to flee)

```python
# 1. Emergency movement (instant, no LLM)
npc_manager.force_navigate_all(
    kaiju_x, kaiju_z,
    "fleeing from the kaiju!",
    flee_from=True,
)

# 2. Router automatically downgrades under scene pressure
# (auto_downgrade_threshold kicks in for reactions)

# 3. Two story-critical NPCs still get LLM for dramatic dialogue
router.add_priority_npc("hero_0")
router.add_priority_npc("elder_0")
```

### Market day (boost trading/socialising)

```python
# Temporarily increase social and trade utility
planner.scorer.set_custom_scorer("socialise", lambda n, c, a: 3.0)
planner.scorer.set_custom_scorer("trade", lambda n, c, a: 2.5)

# After the event
planner.scorer.remove_custom_scorer("socialise")
planner.scorer.remove_custom_scorer("trade")
```

---

## Runtime Configuration

Everything is hot-swappable:

```python
# Change a single decision type
router.set_route("daily_schedule", "deterministic")

# Swap the entire policy
router.set_policy(policy_conversations_only())

# Swap the importance scorer
router.set_importance_scorer(my_custom_scorer)

# Swap planner components
planner.actions = my_custom_registry
planner.scorer = my_custom_scorer
planner.rules = my_custom_rules
```

---

## LLM Providers

The system supports multiple LLM providers through the `LLMProvider` interface:

| Provider | Model | Use Case |
|----------|-------|----------|
| `ClaudeProvider` | Haiku/Opus | Highest quality NPC cognition |
| `MistralProvider` | Small/Large | Cost-effective, ~500K free tokens/day |
| `MockProvider` | — | Testing, no API calls |

```python
from core.npc.mistral_provider import MistralProvider
from core.npc.llm_client import ClaudeProvider

# Use Mistral for NPCs (free tier)
llm = MistralProvider()  # reads MISTRAL_API_KEY from .env

# Or Claude for highest quality
llm = ClaudeProvider()   # reads ANTHROPIC_API_KEY from .env
```

---

## Monitoring

### Router statistics

```python
stats = router.get_stats()
# {
#     "total_decisions": 1543,
#     "llm_decisions": 312,
#     "deterministic_decisions": 1231,
#     "llm_ratio": 0.202,
#     "by_type": {
#         "conversation": {"llm": 245, "deterministic": 12},
#         "daily_schedule": {"llm": 67, "deterministic": 430},
#         ...
#     },
#     "budget": { ... },
#     "policy": { ... }
# }
```

### Planner decision breakdown

```python
scores = planner.score_all(npc, all_npcs, "morning")
for s in scores[:5]:
    print(f"{s.action_id}: {s.total_score:.2f}")
    for k, v in s.breakdown.items():
        print(f"  {k}: {v}")
```

---

## File Reference

```
core/npc/cognition/
  router/
    __init__.py       — CognitionRouter, Route, RouteDecision
    budget.py         — TokenBudget, BudgetSnapshot
    policy.py         — CognitionPolicy, AutoConfig, presets
  planner/
    __init__.py       — DeterministicPlanner (orchestrator)
    actions.py        — ActionDef, ActionRegistry, defaults
    utility.py        — UtilityScorer, need curves
    rules.py          — RuleSet, RuleRegistry, PlannedAction
    context.py        — PlannerContext, ContextBuilder

core/npc/
  llm_client.py       — LLMProvider interface, ClaudeProvider
  mistral_provider.py — MistralProvider
  seed_memories.py    — Foundational NPC memories
```
