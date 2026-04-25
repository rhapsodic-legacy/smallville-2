# Skill: Event Impact System

## When to Use
When defining event rules, handling interactions between NPCs or between
the player and NPCs that should have specific mechanical outcomes.

## Event Rule Structure

    EventRule(
        event_type="ring_given",
        conditions=[...],       # optional conditions to check
        effects=[...],          # list of effects to apply
        scope="individual",     # "individual" or "world"
    )

## Three Impact Modes

### 1. Hard Coded Impact
Always fires when event occurs. No conditions.

    EventRule(
        event_type="ring_given",
        conditions=[],
        effects=[
            Effect("modify_sentiment", target="receiver", dimension="affection", delta=500),
        ],
    )

### 2. Conditional Impact
Evaluates conditions against NPC state before applying.

    EventRule(
        event_type="ring_given",
        conditions=[
            Condition("receiver.would_accept_proposal", op="==", value=True),
        ],
        effects=[
            Effect("modify_sentiment", target="receiver", dimension="affection", delta=500),
            Effect("set_flag", target="receiver", flag="engaged", value=True),
        ],
        else_effects=[
            Effect("modify_sentiment", target="receiver", dimension="affection", delta=-200),
            Effect("modify_sentiment", target="giver", dimension="trust", delta=-100),
        ],
    )

### 3. Boolean / Narrative Trigger
Flips flags that unlock narrative progression.

    EventRule(
        event_type="ring_given",
        conditions=[
            Condition("receiver.waiting_for_proposal", op="==", value=True),
        ],
        effects=[
            Effect("set_flag", target="receiver", flag="waiting_for_proposal", value=False),
            Effect("trigger_quest", quest_id="wedding_preparation"),
        ],
    )

## Events at the World Level
Scope "world" applies effects to all NPCs or modifies global parameters.

    EventRule(
        event_type="war_declared",
        conditions=[],
        effects=[
            Effect("modify_global", param="aggression_modifier", delta=30),
            Effect("modify_global", param="trust_strangers_modifier", delta=-50),
            Effect("set_world_flag", flag="at_war", value=True),
            Effect("overseer_weight", dimension="survival", delta=0.3),
        ],
        scope="world",
    )

## Effect Types
- modify_sentiment — Change relationship dimension between NPCs
- set_flag — Set boolean flag on NPC or world
- modify_param — Adjust NPC numerical parameter
- modify_global — Adjust parameter at the world level affecting all NPCs
- trigger_quest — Start a quest chain
- spawn_event — Create a subsequent event
- overseer_weight — Adjust overseer fitness weights

## Rules for Defining Events
- Keep conditions simple and evaluable from NPC state
- Use else_effects for conditional branches
- Events at the world level should be rare and high impact
- Document the narrative intent in a comment
- Event rules are data — defined in config, not hard coded in logic
