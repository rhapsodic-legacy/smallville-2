"""
Narrative sim-test framework — opt-in, real-LLM scenarios.

Lets you write tests that look like short stories and assert on
what NPCs remember, decide, and do. These tests are:

- SLOW: each scenario makes real LLM calls (Gemma via Ollama by
  default). Expect 30s - a few minutes per scenario.
- OPT-IN: marked with `@pytest.mark.narrative`. `pytest` excludes
  them from the default run; `pytest -m narrative` includes them.
  Run them when you introduce or change NPC reasoning features.
- ROBUST: every scenario auto-skips when Ollama is unreachable or
  the expected Gemma model is missing — so a laptop without LLM
  access won't generate spurious failures.

Usage (see `test_dara_gold_scenario.py` for a worked example):

    from tests.simulation.narrative.framework import (
        NarrativeSim, narrative_scenario,
    )

    @narrative_scenario
    async def test_dara_hears_about_bran_gold(sim: NarrativeSim):
        await sim.player_says("Dara", "Bran said he has 1000 gold for you.")
        await sim.advance(minutes=30)
        sim.assert_has_memory(
            "Dara", category="relayed_claim",
            matches=("bran", "gold"),
        )

The `@narrative_scenario` decorator does three things:
  1. Marks the test `@pytest.mark.narrative` (opt-in).
  2. Supplies the `sim` fixture fresh per test.
  3. Teardown: close the LLM client / clear module state.

Extend the framework as new assertion shapes land. For example,
when Phase I progress-aware objectives ships you'll want
`sim.assert_action_intent_fired("Dara", "talk to Bran")`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import pytest

from core.memory.episodic import EpisodicMemory, EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.npc.cognition.converse import (
    _active_conversations,
    continue_conversation,
    end_conversation,
    start_player_conversation,
)
from core.npc.gemma_provider import GemmaProvider, ollama_available
from core.npc.llm_client import MockProvider, LLMProvider
from core.npc.manager import NPCManager
from core.player.player_agent import PlayerAgent
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.world.generator import WorldConfig, generate_world

logger = logging.getLogger(__name__)


# ---------- Environment gate ----------

def _gemma_available() -> tuple[bool, str]:
    """Return (ok, reason) — is Gemma reachable for a narrative run?"""
    if not ollama_available():
        return False, "Ollama not reachable on localhost:11434"
    try:
        import urllib.request, json
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=2,
        ) as resp:
            data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        if not any(m.startswith("gemma") for m in models):
            return False, (
                f"No gemma model installed in Ollama "
                f"(found: {models})"
            )
        return True, ""
    except Exception as e:
        return False, f"Ollama probe failed: {e}"


# ---------- Sim ----------

@dataclass
class _AssertionFailure(AssertionError):
    """Rich assertion failure with the actual memory log appended."""
    message: str
    dump: str

    def __str__(self) -> str:
        return f"{self.message}\n\n--- memory log ---\n{self.dump}"


class NarrativeSim:
    """A fresh Smallville world wrapped in a narrative-test-friendly API.

    One instance per scenario. Don't reuse across tests.
    """

    def __init__(
        self,
        *,
        population: int = 5,
        seed: int = 777,
        provider: LLMProvider | None = None,
    ):
        self.config = WorldConfig(
            population=population, terrain="riverside", seed=seed,
        )
        self.grid, self.buildings = generate_world(self.config)
        self.llm: LLMProvider = provider or GemmaProvider()
        # Fallback-only episodic keeps tests hermetic — no shared
        # ChromaDB state bleeds between scenarios.
        self.memory = MemoryManager(
            structured=StructuredMemory(":memory:"),
            episodic=EpisodicStore(fallback_only=True),
            spatial=SpatialMemory(),
            llm=self.llm,
        )
        self.memory.initialise()
        self.manager = NPCManager(
            grid=self.grid, buildings=self.buildings,
            llm=self.llm, seed=seed, memory=self.memory,
        )
        self.manager.spawn_population(population)
        self.clock = GameClock()
        self.player = PlayerAgent.create(
            name="Traveller", spawn_x=0.0, spawn_z=0.0,
        )
        # Player must NOT run autonomous cognition during advance()
        # — otherwise every tick burns LLM budget on the player's
        # "NPC reasoning" and narrative tests take forever.
        self.player.autonomous = False
        # Register the player NPC in the NPCManager so perception
        # and conversation loops see them as a townsfolk.
        self.manager.npcs.append(self.player.npc)
        self.manager._npc_map[self.player.npc.npc_id] = self.player.npc
        self.manager.player_agent = self.player

    # ---------- Casting ----------

    def cast(self, **roles: str) -> None:
        """Rename NPCs to fit a scenario's cast.

        Scenarios read better when NPCs have recognisable names
        ("Dara", "Bran", "Petra") — but spawned NPC names are a
        function of the random seed. Call `cast()` right after
        construction to map role names to specific spawn slots.

        Pass the role name ("dara", "bran", ...) as the keyword
        and either an occupation substring ("blacksmith") or an
        integer spawn-index as the value:

            sim.cast(dara="blacksmith", bran="merchant", petra=2)

        Unmatched roles raise immediately so a typo in the
        scenario doesn't silently run on the wrong NPC.
        """
        taken: set[int] = set()
        for role, selector in roles.items():
            idx: int | None = None
            if isinstance(selector, int):
                idx = selector
            else:
                needle = str(selector).lower()
                for i, n in enumerate(self.manager.npcs):
                    if i in taken:
                        continue
                    if needle in (n.occupation or "").lower():
                        idx = i
                        break
            if idx is None or idx in taken or idx >= len(self.manager.npcs):
                raise LookupError(
                    f"cast(): couldn't place role {role!r} via "
                    f"selector={selector!r}. "
                    f"Available (name, occupation): "
                    f"{[(n.name, n.occupation) for n in self.manager.npcs]}"
                )
            taken.add(idx)
            target = self.manager.npcs[idx]
            new_name = role.capitalize()
            target.name = new_name

    # ---------- Lookups ----------

    def npc(self, name_or_id: str):
        """Resolve an NPC by case-insensitive name substring OR id."""
        lower = name_or_id.lower()
        for n in self.manager.npcs:
            if n.npc_id == name_or_id or lower in n.name.lower():
                return n
        raise LookupError(
            f"No NPC matching {name_or_id!r}. "
            f"Available: {[n.name for n in self.manager.npcs]}"
        )

    def memories(
        self,
        name_or_id: str,
        category: str | None = None,
        include_compacted: bool = False,
    ) -> list[EpisodicMemory]:
        npc = self.npc(name_or_id)
        return self.memory.episodic.get_recent(
            npc.npc_id, limit=10_000,
            category=category,
            include_compacted=include_compacted,
        )

    # ---------- Player interaction ----------

    async def player_says(
        self, npc_name: str, text: str,
    ) -> str:
        """One player utterance + one NPC reply. Returns the reply.

        Mirrors `server/main.py::_handle_player_chat`: the NPC is
        pinned to tier 1 and force_llm=True so the response comes
        from the real LLM, not a canned fallback.
        """
        target = self.npc(npc_name)
        # Position player adjacent so distance gates pass.
        self.player.npc.x = float(target.x)
        self.player.npc.z = float(target.z) + 1.0
        target.cognition_tier = 1
        start_player_conversation(target, self.player.npc, text)
        self.player.is_chatting = True
        self.player.chat_target_id = target.npc_id
        current_minutes = (
            self.clock.day * MINUTES_PER_DAY + self.clock.minutes
        )
        try:
            matters = self.memory.retrieve_unresolved_matters(
                target.npc_id,
                partner_id=self.player.npc.npc_id,
                partner_name=self.player.npc.name,
            )
            await continue_conversation(
                target, self.player.npc, self.llm, self.memory,
                allow_auto_end=False, max_exchanges=40,
                force_llm=True,
                town_agenda_summary=self.manager.town_agenda.summary_for_prompt(
                    target.npc_id,
                ),
                unresolved_matters_summary=(
                    self.memory.format_unresolved_matters(
                        matters, self.player.npc.name,
                    )
                ),
            )
        finally:
            # Persist what just landed so downstream assertions see it.
            from core.npc.cognition.converse import _active_conversations
            key = frozenset({self.player.npc.npc_id, target.npc_id})
            conv = _active_conversations.get(key)
            if conv is not None and conv.exchanges:
                await self.memory.persist_new_exchanges(
                    conv, self.player.npc, target,
                    game_time=current_minutes,
                    location_x=int(target.x), location_z=int(target.z),
                )
        # Return the NPC's reply.
        key = frozenset({self.player.npc.npc_id, target.npc_id})
        conv = _active_conversations.get(key)
        if conv is not None:
            for ex in reversed(conv.exchanges):
                if ex.speaker_id == target.npc_id:
                    return ex.message
        return ""

    async def player_closes_chat(self, npc_name: str) -> None:
        target = self.npc(npc_name)
        await end_conversation(
            self.player.npc, target, memory_manager=self.memory,
        )
        self.player.is_chatting = False
        self.player.chat_target_id = None

    # ---------- Time advance ----------

    async def advance(
        self,
        *,
        minutes: int | None = None,
        days: int | None = None,
        ticks: int | None = None,
        real_delta: float = 8.0,
    ) -> None:
        """Advance the sim. Specify exactly one of
        `minutes`, `days`, `ticks`.

        The default `real_delta=8.0` yields roughly 9.6 game-minutes
        per cognition tick, mirroring the pattern used by
        `tests/simulation/test_multiday_invariants.py`.
        """
        if sum(x is not None for x in (minutes, days, ticks)) != 1:
            raise ValueError(
                "advance() needs exactly one of minutes/days/ticks"
            )
        if days is not None:
            minutes = days * MINUTES_PER_DAY
        if ticks is None:
            assert minutes is not None
            game_minutes_per_tick = 9.6
            ticks = max(1, int(minutes / game_minutes_per_tick) + 1)
        for _ in range(ticks):
            self.clock.tick(real_delta)
            self.manager.movement_tick(self.clock, real_delta)
            await self.manager.cognition_tick(self.clock, real_delta)

    # ---------- Assertions ----------

    def assert_has_memory(
        self,
        name_or_id: str,
        *,
        category: str | None = None,
        matches: tuple[str, ...] | str | None = None,
        within_days: int | None = None,
        min_count: int = 1,
    ) -> list[EpisodicMemory]:
        """Assert that an NPC holds ≥`min_count` memories matching
        the given filters. Returns the matched memories.

        - `category`: exact category name (e.g. "relayed_claim",
          "commitment", "accusation", "reflection", "day_summary").
        - `matches`: substring or tuple of substrings — all must
          appear (case-insensitive) in the memory's description.
        - `within_days`: only consider memories formed in the last
          N game-days from the current clock.

        Fails with the full memory dump appended so you can see
        what DID land when the assertion misses.
        """
        npc = self.npc(name_or_id)
        needle: tuple[str, ...] = ()
        if matches is not None:
            needle = (
                (matches,) if isinstance(matches, str) else tuple(matches)
            )
        current_minutes = (
            self.clock.day * MINUTES_PER_DAY + self.clock.minutes
        )
        cutoff = (
            current_minutes - within_days * MINUTES_PER_DAY
            if within_days is not None else None
        )
        hits: list[EpisodicMemory] = []
        for mem in self.memories(npc.npc_id, category=category):
            if cutoff is not None and mem.game_time < cutoff:
                continue
            desc = (mem.description or "").lower()
            if all(tok.lower() in desc for tok in needle):
                hits.append(mem)
        if len(hits) < min_count:
            raise _AssertionFailure(
                message=(
                    f"Expected {npc.name} to have ≥{min_count} memory "
                    f"(category={category!r}, matches={needle}, "
                    f"within_days={within_days}); "
                    f"found {len(hits)}."
                ),
                dump=self.dump_memories(npc.npc_id),
            )
        return hits

    def assert_tags_present(
        self,
        name_or_id: str,
        tags: tuple[str, ...] | str,
        min_count: int = 1,
    ) -> list[EpisodicMemory]:
        """Assert the NPC's tag index contains at least `min_count`
        memories carrying ANY of the listed tags.
        """
        npc = self.npc(name_or_id)
        tag_list = (tags,) if isinstance(tags, str) else tuple(tags)
        hits = self.memory.episodic.retrieve_by_tags(
            npc.npc_id, tag_list, limit=10_000,
        )
        if len(hits) < min_count:
            raise _AssertionFailure(
                message=(
                    f"Expected {npc.name} to have ≥{min_count} "
                    f"tagged memory matching tags={tag_list}; "
                    f"found {len(hits)}."
                ),
                dump=self.dump_memories(npc.npc_id),
            )
        return hits

    def assert_schedule_contains(
        self, name_or_id: str, *, activity_substring: str,
    ) -> None:
        """Assert the NPC's current daily_schedule has an entry
        whose activity text mentions `activity_substring`. Useful
        for verifying action-intent injection from a reflection."""
        npc = self.npc(name_or_id)
        needle = activity_substring.lower()
        for entry in (npc.daily_schedule or []):
            if needle in (entry.activity or "").lower():
                return
        schedule = "\n".join(
            f"  - [{e.slot}] {e.activity} @ {e.location}"
            for e in (npc.daily_schedule or [])
        ) or "(schedule empty)"
        raise _AssertionFailure(
            message=(
                f"Expected {npc.name}'s schedule to contain an "
                f"activity mentioning {activity_substring!r}."
            ),
            dump=f"schedule:\n{schedule}",
        )

    # ---------- Diagnostics ----------

    def dump_memories(
        self,
        name_or_id: str | None = None,
        limit: int | None = 20,
    ) -> str:
        """Format a readable memory log for one NPC or all NPCs.

        Empty `name_or_id` means every NPC. Useful to stick into a
        failure message so you can see what DID happen.
        """
        targets = [self.npc(name_or_id)] if name_or_id else self.manager.npcs
        lines: list[str] = []
        for npc in targets:
            mems = self.memory.episodic.get_recent(
                npc.npc_id, limit=limit or 10_000,
            )
            lines.append(f"=== {npc.name} ({npc.npc_id}) ===")
            for mem in mems:
                gt = mem.game_time
                d = int(gt // MINUTES_PER_DAY)
                mn = int(gt % MINUTES_PER_DAY)
                hh, mm = divmod(mn, 60)
                cat = (mem.category or "?").upper()
                desc = (mem.description or "")[:140]
                lines.append(
                    f"  {cat:<18} D{d} {hh:02d}:{mm:02d}  "
                    f"imp={mem.importance:.2f}  {desc}"
                )
            if not mems:
                lines.append("  (no memories)")
        return "\n".join(lines)


# ---------- Decorator / pytest plumbing ----------

_SkipIfNoGemma = pytest.mark.skipif(
    not _gemma_available()[0],
    reason=(
        f"Narrative tests need a local Gemma via Ollama "
        f"({_gemma_available()[1]}). Start Ollama and install "
        f"a gemma model to run."
    ),
)


@pytest.fixture
def sim():
    """Fresh `NarrativeSim` per test + conversation-registry hygiene.

    Scenarios pull this fixture by parameter name. The registry
    is cleared before and after the scenario so one bad chat can't
    contaminate the next test.
    """
    _active_conversations.clear()
    obj = NarrativeSim()
    try:
        yield obj
    finally:
        _active_conversations.clear()


def narrative_scenario(
    fn: Callable[[NarrativeSim], Awaitable[None]],
) -> Callable[..., None]:
    """Wrap an async narrative test with the narrative marker and
    Gemma-available skip — keeps the one-decorator ergonomics the
    scenarios were written against.

    The `sim` parameter is injected via pytest fixture.

    Usage:

        @narrative_scenario
        async def test_something(sim: NarrativeSim):
            await sim.player_says("Dara", "...")
            await sim.advance(minutes=30)
            sim.assert_has_memory("Dara", category="relayed_claim")
    """
    # 600s default timeout — post-chat reflection on Gemma fires
    # 5-8 LLM calls (extract_outcomes, record_conversation fact
    # extract, per-participant reflection + classify_insight +
    # extract_important_note), each with its own 10-15s bound.
    # On an overloaded local Gemma each can stretch.
    return pytest.mark.narrative(
        pytest.mark.timeout(600)(_SkipIfNoGemma(fn))
    )
