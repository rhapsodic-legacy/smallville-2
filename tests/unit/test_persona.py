"""Persona generation invariants — determinism, distinctiveness,
serialisation, rendering, and spawn integration.

These are the failure-mode guards for the vectorization foundation:
a town where two NPCs share a voice, a forge that drifts between
runs of the same seed, or a persona that silently vanishes from a
save file would each quietly reproduce the parrot problem the arc
exists to fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.npc.llm_client import MockProvider
from core.npc.manager import NPCManager
from core.npc.models import NPC, PersonalityTraits
from core.npc.persona import (
    Persona,
    PersonaForge,
    persona_system_prompt,
    SPEECH_STYLES,
    TEMPERAMENTS,
)
from core.world.generator import WorldConfig, generate_world


def _make_npc(persona: Persona | None = None, **overrides) -> NPC:
    defaults = dict(
        npc_id="vex_1",
        name="Vex",
        age=40,
        personality=PersonalityTraits(),
        backstory="Vex has lived here always.",
        occupation="blacksmith",
        persona=persona,
    )
    defaults.update(overrides)
    return NPC(**defaults)


class TestPersonaForge:
    def test_same_seed_same_personas(self):
        a = PersonaForge.from_seed(123)
        b = PersonaForge.from_seed(123)
        for _ in range(20):
            assert a.forge("farmer").to_dict() == b.forge("farmer").to_dict()

    def test_different_seeds_differ(self):
        a = PersonaForge.from_seed(1)
        b = PersonaForge.from_seed(2)
        seq_a = [a.forge("farmer").speech_style for _ in range(10)]
        seq_b = [b.forge("farmer").speech_style for _ in range(10)]
        assert seq_a != seq_b

    def test_distinct_within_bank_size(self):
        forge = PersonaForge.from_seed(7)
        n = min(len(SPEECH_STYLES), len(TEMPERAMENTS))
        personas = [forge.forge("farmer") for _ in range(n)]
        assert len({p.speech_style for p in personas}) == n
        assert len({p.temperament for p in personas}) == n

    def test_over_bank_size_still_complete(self):
        forge = PersonaForge.from_seed(7)
        for _ in range(60):  # well past every bank size
            p = forge.forge("merchant")
            assert p.speech_style and p.verbal_tic and p.temperament
            assert len(p.behaviour_rules) == 2
            assert p.behaviour_rules[0] != p.behaviour_rules[1]
            assert p.core_value and p.fear and p.quirk and p.private_agenda

    def test_agenda_occupation_substituted(self):
        forge = PersonaForge.from_seed(11)
        for _ in range(40):
            agenda = forge.forge("priest").private_agenda
            assert "{occupation}" not in agenda

    def test_round_trip(self):
        p = PersonaForge.from_seed(3).forge("guard")
        assert Persona.from_dict(p.to_dict()).to_dict() == p.to_dict()


class TestPromptRendering:
    def test_block_contains_every_component(self):
        p = PersonaForge.from_seed(5).forge("farmer")
        block = p.to_prompt_block("Mira")
        for part in [
            p.speech_style, p.verbal_tic, p.temperament,
            *p.behaviour_rules, p.core_value, p.fear, p.quirk,
            p.private_agenda,
        ]:
            assert part in block
        assert "Mira" in block

    def test_system_prompt_structure_and_order(self):
        p = PersonaForge.from_seed(5).forge("blacksmith")
        npc = _make_npc(persona=p)
        npc.self_concept["opposes:repair_bridge"] = 0.9
        out = persona_system_prompt(npc, "Task line here.")
        assert out.startswith("You are Vex, a 40-year-old blacksmith")
        assert "character sheet" in out
        assert p.speech_style in out
        assert "opposed to repair bridge" in out
        # Dominance ordering: sheet before guardrail before task line.
        assert (
            out.index(p.speech_style)
            < out.index("Stay in this voice")
            < out.index("Task line here.")
        )

    def test_system_prompt_no_persona_falls_back(self):
        npc = _make_npc(persona=None)
        out = persona_system_prompt(npc, "Task line.")
        assert out == (
            "You are Vex, a 40-year-old blacksmith in Smallville.\n"
            "Task line."
        )

    def test_system_prompt_tolerates_none_and_partial(self):
        assert "someone" in persona_system_prompt(None, "Task.")

        class Partial:  # no age, no persona
            name = "Stub"
            occupation = "townsperson"

        out = persona_system_prompt(Partial(), "Task.")
        assert out.startswith("You are Stub, a townsperson in Smallville.")


class TestSpawnIntegration:
    def _spawn(self, seed: int = 55, population: int = 10):
        config = WorldConfig(
            population=population, terrain="riverside", seed=seed,
        )
        grid, buildings = generate_world(config)
        mgr = NPCManager(
            grid=grid, buildings=buildings, llm=MockProvider(), seed=seed,
        )
        return mgr.spawn_population(population)

    def test_every_npc_gets_distinct_persona(self):
        npcs = self._spawn()
        assert all(n.persona is not None for n in npcs)
        assert len({n.persona.speech_style for n in npcs}) == len(npcs)
        assert len({n.persona.temperament for n in npcs}) == len(npcs)

    def test_spawn_deterministic_across_managers(self):
        a = self._spawn(seed=99)
        b = self._spawn(seed=99)
        assert [n.persona.to_dict() for n in a] == [
            n.persona.to_dict() for n in b
        ]
        # Persona forging must not perturb the existing spawn RNG.
        assert [(n.name, n.age) for n in a] == [(n.name, n.age) for n in b]

    def test_full_dict_carries_persona(self):
        npc = self._spawn()[0]
        data = npc.to_full_dict()
        assert data["persona"] == npc.persona.to_dict()

    def test_player_agent_has_no_persona_and_survives(self):
        from core.player.player_agent import PlayerAgent

        player = PlayerAgent.create()
        assert player.npc.persona is None
        out = persona_system_prompt(player.npc, "Task.")
        assert "Traveller" in out
