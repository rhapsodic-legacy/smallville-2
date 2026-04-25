# Narrative sim-tests

Opt-in, real-LLM end-to-end scenarios. Each test reads like a short
story: Traveller says X, sim advances, assert NPCs remembered /
decided / did Y.

## When to run

- When you introduce or change NPC reasoning (outcome extraction,
  reflection, planning, memory, tags, compaction, objectives).
- When you add a new "complex" feature and want a living
  demonstration that it works end-to-end.
- Before cutting a release.

NOT in the default `pytest` run. They're slow (real LLM calls,
tens of seconds per scenario).

## Run

```bash
# Every narrative scenario (slow):
pytest -m narrative tests/simulation/narrative/ -v

# One file:
pytest -m narrative tests/simulation/narrative/test_dara_gold_scenario.py -v

# One case:
pytest -m narrative -k "canned_fallback" -v
```

Tests auto-skip when Ollama is unreachable or no `gemma*` model is
installed, so the suite never generates spurious failures on a
laptop without LLM access.

## Writing a new scenario

Two moving pieces: (1) the `NarrativeSim` helper, which gives you
a fluent API over the sim, and (2) the `@narrative_scenario`
decorator, which wraps an async test in the right fixtures and
marker.

Minimum example:

```python
from tests.simulation.narrative.framework import (
    NarrativeSim, narrative_scenario,
)

@narrative_scenario
async def test_dara_responds_to_gold_claim(sim: NarrativeSim):
    reply = await sim.player_says(
        "Dara",
        "Bran said he wants to give you a thousand gold.",
    )
    assert "bran" in reply.lower() or "gold" in reply.lower()
```

### API cheat sheet

```python
sim.npc("Dara")                     # resolve by name substring or id
sim.memories("Dara", category="relayed_claim")

await sim.player_says("Dara", "text")     # one full chat exchange
await sim.player_closes_chat("Dara")      # end the chat window
await sim.advance(minutes=30)             # or days=2, or ticks=50

# Assertions (all fail with the full memory log attached):
sim.assert_has_memory("Dara", category="commitment", matches=("bran",))
sim.assert_tags_present("Dara", tags=("bran", "gold"))
sim.assert_schedule_contains("Dara", activity_substring="visit bran")

print(sim.dump_memories("Dara"))          # diagnostic dump
```

## Building more complex scenarios

The framework composes. Bigger narrative arcs are just chains of
the primitives above:

```python
@narrative_scenario
async def test_traveller_spreads_rumour_to_three_npcs(sim: NarrativeSim):
    for name in ("Dara", "Petra", "Kira"):
        await sim.player_says(
            name, "Bran wants to give you a thousand gold!",
        )
        await sim.player_closes_chat(name)
        await sim.advance(minutes=15)
    await sim.advance(days=1)
    # Bran should eventually be approached by multiple townsfolk
    # with the same story.
    sim.assert_has_memory(
        "Bran", category="relayed_claim", matches=("gold",), min_count=2,
    )
```

Aspirational scenarios (land as features ship):

- **"Repair the bridge"**: seed a town-agenda goal, run several
  game-days, assert ≥N NPCs actually visit the bridge tile and
  complete the activity (covers the insta-complete fix plus
  future physical-location gating).
- **"Bran figures out the Traveller is lying"**: Traveller spreads
  contradicting claims to 4 NPCs. After several days, Bran's
  self_concept gains a `distrusts:traveller` entry. This one
  waits for Phase J persona snapshot.
- **"Dara acts on her commitment"**: Dara commits to confront
  Bran. Assert her schedule gains an `action_intent` entry
  pointing at Bran's location within a game-day. Needs Phase I.

## Extending the framework

Add new assertions as features ship — keep the pattern
`assert_X(npc, ..., dump_on_fail=True)` so failures always carry
the memory log. Examples you might add:

- `assert_action_intent_fired(npc, target=...)` once Phase I lands.
- `assert_town_goal_progress(goal_id, contributors={...})`.
- `assert_persona_has(npc, tag_or_concept)` once Phase J lands.

The goal is a library of short-story-shaped tests that grows
alongside the simulation.
