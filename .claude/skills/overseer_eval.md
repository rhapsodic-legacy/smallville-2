# Skill: Overseer Evaluation

## When to Use
When working on the evolution layer — fitness evaluation, policy injection,
or behavioural guardrails.

## Overseer Cycle
The overseer runs periodically (default: every 2 game days):
1. **Collect metrics** — Gather fitness scores for all NPCs
2. **Analyse trends** — Identify thriving vs struggling strategies
3. **Diagnose issues** — Detect stagnation, runaway behaviours, imbalance
4. **Synthesise policies** — Generate new behavioural guidelines
5. **Inject changes** — Apply policies to target NPCs or population

## Fitness Functions (Multiple Objectives)

    fitness = (
        weights["survival"] * survival_score +
        weights["prosperity"] * prosperity_score +
        weights["social"] * social_score +
        weights["goals"] * goal_progress_score +
        weights["engagement"] * engagement_score
    )

Weights are configurable per world theme.

## Evolution Mechanisms (Ordered by Risk)

### 1. Parameter Tuning (safest)
Adjust numerical weights: risk_tolerance, cooperation_tendency, aggression.

    npc.params["risk_tolerance"] += 0.1

### 2. Policy Templates (moderate)
Assign predefined strategy packages:
- "merchant_strategy" — prioritise trade, accumulate gold
- "builder_strategy" — gather resources, contribute to construction
- "social_strategy" — build relationships, form alliances
- "hermit_strategy" — remain self sufficient, avoid conflict

### 3. Prompt Modifiers (powerful)
Append behavioural directives to NPC system prompts:

    "You've noticed that cooperative trading leads to better outcomes.
     Consider proposing joint ventures with trusted allies."

## Guardrails
- No NPC should accumulate >50% of total world gold
- No faction should contain >60% of population
- Aggression levels must not cause >30% NPC deaths per game week
- Override thresholds configurable for AI Game Studio
