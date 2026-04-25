# Skill: NPC Creation

## When to Use
When creating new NPC definitions, adding NPCs to the world, or modifying NPC templates.

## NPC Data Model
Every NPC must have these fields:

### Identity
- npc_id: Unique string identifier (e.g. "blacksmith_1")
- name: Display name
- age: Integer
- personality_traits: List of 3 to 5 trait strings (e.g. ["cautious", "generous", "curious"])
- backstory: 2 to 3 sentence background
- occupation: String (e.g. "blacksmith", "farmer", "merchant")

### Physical State
- location: Current tile address (world:sector:arena:object format)
- home: Home tile address
- health: Float 0.0 to 1.0
- energy: Float 0.0 to 1.0
- hunger: Float 0.0 to 1.0

### Goals
- long_term_goals: List of 2 to 3 aspirational goals
- short_term_goals: List of 1 to 3 immediate goals
- daily_schedule: Generated each morning by cognition system

### Economy
- gold: Integer
- inventory: Dict of item_id to quantity
- skills: Dict of skill_name to proficiency (0.0 to 1.0)

## Cognition Tier Assignment
- Tier 1: Within player's vision radius or actively interacting
- Tier 2: In same sector as player, or recently interacted
- Tier 3: In world but far from player
- Tier 4: Sleeping or in irrelevant location

## Template System
NPCs can be created from archetype templates. Example archetypes:

    ARCHETYPES = {
        "merchant": {
            "personality_traits": ["shrewd", "sociable", "calculating"],
            "occupation": "merchant",
            "gold": 500,
            "skills": {"trading": 0.8, "persuasion": 0.6},
        },
        "farmer": {
            "personality_traits": ["patient", "hardworking", "honest"],
            "occupation": "farmer",
            "gold": 50,
            "skills": {"farming": 0.8, "crafting": 0.3},
        },
    }

Each archetype provides defaults; individual NPCs override specific fields.
