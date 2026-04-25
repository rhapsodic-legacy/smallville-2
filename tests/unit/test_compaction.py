"""
Phase H.1 — day-level compaction with tombstoning.

Covers:
- `EpisodicStore.get_memories_in_window` honours `[start, end)` and
  scopes to the requested NPC.
- `compaction.is_compactable` excludes tagged, preserved-category,
  and already-tombstoned memories.
- `MemoryManager.compact_day` writes a `day_summary` with the
  expected metadata shape (day, compacted_from, compacted_count,
  kept_tags) and tombstones the originals via `compacted_into`.
- Tagged memories survive compaction intact — neither rewritten nor
  tombstoned — so Phase K retention still holds.
- Preserved categories (reflection, commitment, town_event, …) are
  left alone even when untagged.
- No-op on empty days, no-op on a re-run of an already-compacted
  day, and fallback summariser kicks in when no LLM is provided.
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory import compaction
from core.memory.compaction import (
    DAYS_PER_WEEK, PRESERVED_CATEGORIES, compact_day, compact_week,
    is_compactable,
)
from core.memory.episodic import EpisodicMemory, EpisodicStore
from core.memory.manager import MemoryManager
from core.memory.spatial import SpatialMemory
from core.memory.structured import StructuredMemory
from core.time_system.clock import MINUTES_PER_DAY


def _mgr() -> MemoryManager:
    mgr = MemoryManager(
        structured=StructuredMemory(":memory:"),
        episodic=EpisodicStore(fallback_only=True),
        spatial=SpatialMemory(),
    )
    mgr.initialise()
    return mgr


def _day_time(day: int, minute: int = 0) -> float:
    return day * MINUTES_PER_DAY + minute


# ---------- Windowed fetch ----------

class TestGetMemoriesInWindow:
    def test_returns_memories_in_range(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        store.add_memory(
            npc_id="a", description="inside",
            game_time=_day_time(0, 100),
        )
        store.add_memory(
            npc_id="a", description="outside_future",
            game_time=_day_time(1, 0),
        )
        hits = store.get_memories_in_window(
            "a", _day_time(0), _day_time(1),
        )
        assert len(hits) == 1
        assert hits[0].description == "inside"

    def test_excludes_end_boundary(self):
        """End is exclusive so sequential day windows never overlap."""
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        store.add_memory(
            npc_id="a", description="boundary",
            game_time=_day_time(1, 0),
        )
        assert store.get_memories_in_window(
            "a", _day_time(0), _day_time(1),
        ) == []
        assert len(store.get_memories_in_window(
            "a", _day_time(1), _day_time(2),
        )) == 1

    def test_scoped_per_npc(self):
        store = EpisodicStore(fallback_only=True)
        store.initialise()
        store.add_memory(
            npc_id="a", description="mine", game_time=_day_time(0, 50),
        )
        store.add_memory(
            npc_id="b", description="yours", game_time=_day_time(0, 50),
        )
        hits = store.get_memories_in_window(
            "a", _day_time(0), _day_time(1),
        )
        assert len(hits) == 1
        assert hits[0].description == "mine"


# ---------- is_compactable ----------

class TestIsCompactable:
    def test_untagged_observation_is_compactable(self):
        m = EpisodicMemory(category="observation", tags=set())
        assert is_compactable(m)

    def test_tagged_memory_not_compactable(self):
        m = EpisodicMemory(category="observation", tags={"bread"})
        assert not is_compactable(m)

    def test_preserved_categories_skipped(self):
        for cat in PRESERVED_CATEGORIES:
            m = EpisodicMemory(category=cat, tags=set())
            assert not is_compactable(m), f"{cat} should be preserved"

    def test_already_tombstoned_skipped(self):
        m = EpisodicMemory(
            category="observation", tags=set(),
            metadata={"compacted_into": "summary_1"},
        )
        assert not is_compactable(m)


# ---------- compact_day ----------

class TestCompactDay:
    def _populate_day(self, mgr: MemoryManager, day: int = 0) -> dict:
        """Seed day `day` with one of each kind of memory.

        Returns the ids so tests can assert on exact rows.
        """
        ids: dict[str, str] = {}
        # Untagged observation — should be compacted.
        ids["obs"] = mgr.episodic.add_memory(
            npc_id="bran",
            description="I walked past the market and saw the baker open up.",
            category="observation",
            importance=0.4,
            game_time=_day_time(day, 30),
        )
        # A second untagged observation — also compacted.
        ids["obs2"] = mgr.episodic.add_memory(
            npc_id="bran",
            description="It rained briefly around midday.",
            category="observation",
            importance=0.3,
            game_time=_day_time(day, 180),
        )
        # Tagged outcome-style memory — must survive intact.
        ids["tagged"] = mgr.episodic.add_memory(
            npc_id="bran",
            description="Petra accused me of hoarding bread.",
            category="accusation",
            importance=0.8,
            game_time=_day_time(day, 300),
            tags={"accused:bran", "bread", "outcome:accusation"},
            extra_metadata={"unresolved": True},
        )
        # Preserved category, untagged — must still survive.
        ids["reflection"] = mgr.episodic.add_memory(
            npc_id="bran",
            description="Reflection: I feel the town is restless.",
            category="reflection",
            importance=0.8,
            game_time=_day_time(day, 500),
        )
        return ids

    def test_writes_day_summary_and_tombstones_originals(self):
        mgr = _mgr()
        ids = self._populate_day(mgr)
        summary_id = asyncio.run(
            compact_day(mgr, "bran", 0, llm=None)
        )
        assert summary_id is not None

        summary = mgr.episodic.get_by_id(summary_id)
        assert summary is not None
        assert summary.category == "day_summary"
        assert summary.metadata.get("day") == 0
        assert summary.metadata.get("compacted_count") == 2
        compacted_from = summary.metadata.get("compacted_from", "")
        assert ids["obs"] in compacted_from
        assert ids["obs2"] in compacted_from
        # Summary lands at end-of-day.
        assert summary.game_time == _day_time(1) - 1.0

        # Originals carry tombstone pointer.
        for key in ("obs", "obs2"):
            mem = mgr.episodic.get_by_id(ids[key])
            assert mem is not None
            assert mem.metadata.get("compacted_into") == summary_id

    def test_tagged_memory_survives_compaction_intact(self):
        mgr = _mgr()
        ids = self._populate_day(mgr)
        asyncio.run(compact_day(mgr, "bran", 0, llm=None))

        tagged = mgr.episodic.get_by_id(ids["tagged"])
        assert tagged is not None
        # Not tombstoned.
        assert "compacted_into" not in (tagged.metadata or {})
        # Original tags intact.
        assert "bread" in tagged.tags
        assert "accused:bran" in tagged.tags
        # Still discoverable via tag index.
        hits = mgr.episodic.retrieve_by_tags("bran", ["bread"])
        assert any(m.memory_id == ids["tagged"] for m in hits)

    def test_reflection_category_preserved(self):
        mgr = _mgr()
        ids = self._populate_day(mgr)
        asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        refl = mgr.episodic.get_by_id(ids["reflection"])
        assert refl is not None
        assert "compacted_into" not in (refl.metadata or {})

    def test_kept_tags_aggregate_from_tagged_memories(self):
        mgr = _mgr()
        self._populate_day(mgr)
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        summary = mgr.episodic.get_by_id(summary_id)
        kept = set((summary.metadata.get("kept_tags") or "").split())
        # Every tag on the surviving accusation is represented.
        assert {"bread", "accused:bran", "outcome:accusation"}.issubset(kept)

    def test_noop_on_empty_day(self):
        mgr = _mgr()
        # Seed a memory on day 1 only — day 0 is empty.
        mgr.episodic.add_memory(
            npc_id="bran", description="day 1 thing",
            game_time=_day_time(1, 10),
        )
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        assert summary_id is None

    def test_noop_when_only_tagged_memories(self):
        mgr = _mgr()
        mgr.episodic.add_memory(
            npc_id="bran", description="tagged",
            category="commitment",
            game_time=_day_time(0, 10),
            tags={"outcome:commitment"},
        )
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        assert summary_id is None

    def test_rerun_is_idempotent(self):
        """A second compaction on the same day finds nothing to do."""
        mgr = _mgr()
        self._populate_day(mgr)
        first = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        second = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        assert first is not None
        # Nothing new to compact — all originals already tombstoned.
        # The day_summary itself is also preserved-category, so it
        # won't be re-compacted.
        assert second is None

    def test_fallback_summary_used_when_no_llm(self):
        mgr = _mgr()
        self._populate_day(mgr)
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        summary = mgr.episodic.get_by_id(summary_id)
        # Heuristic fallback stamps the day prefix so it's obviously
        # not LLM output.
        assert "Day 0" in summary.description

    def test_day_summary_is_not_compactable(self):
        """Day summaries must never be re-compacted — they're
        already the compressed form.
        """
        mgr = _mgr()
        self._populate_day(mgr)
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        summary = mgr.episodic.get_by_id(summary_id)
        assert not is_compactable(summary)


# ---------- LLM path ----------

class _StubLLM:
    """Minimal LLM stub matching the `complete` surface used by
    `_summarise_with_llm`. Captures the prompt for assertion and
    returns a canned summary.
    """

    def __init__(self, response: str = "A quiet day. Nothing changed."):
        self.response = response
        self.last_prompt: str = ""

    async def complete(
        self, *, system, messages, max_tokens, temperature, purpose,
    ) -> str:
        self.last_prompt = messages[0]["content"]
        assert purpose in ("day_summary", "week_summary")
        self.last_purpose = purpose
        return self.response


class TestCompactDayWithLLM:
    def test_uses_llm_response_as_summary(self):
        mgr = _mgr()
        mgr.episodic.add_memory(
            npc_id="bran", description="I baked bread.",
            category="observation", game_time=_day_time(0, 20),
        )
        llm = _StubLLM(response="It was a warm, uneventful morning.")
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=llm))
        summary = mgr.episodic.get_by_id(summary_id)
        assert summary.description == "It was a warm, uneventful morning."
        # Prompt includes the compactable description.
        assert "I baked bread." in llm.last_prompt

    def test_llm_failure_falls_back_to_heuristic(self):
        class _BoomLLM:
            async def complete(self, **_kwargs):
                raise RuntimeError("network blew up")

        mgr = _mgr()
        mgr.episodic.add_memory(
            npc_id="bran", description="I baked bread.",
            category="observation", game_time=_day_time(0, 20),
        )
        summary_id = asyncio.run(
            compact_day(mgr, "bran", 0, llm=_BoomLLM())
        )
        summary = mgr.episodic.get_by_id(summary_id)
        assert "Day 0" in summary.description


# ---------- H.2: prompt shape ----------

class TestDaySummaryPrompt:
    """The prompt must cue the 3-point structure (events / feelings /
    shifts) and thread the NPC's voice through personality +
    self_concept."""

    def _run_with_npc(self, npc) -> str:
        from core.memory.manager import MemoryManager
        from core.memory.episodic import EpisodicStore
        from core.memory.spatial import SpatialMemory
        from core.memory.structured import StructuredMemory
        mgr = MemoryManager(
            structured=StructuredMemory(":memory:"),
            episodic=EpisodicStore(fallback_only=True),
            spatial=SpatialMemory(),
        )
        mgr.initialise()
        mgr.episodic.add_memory(
            npc_id="bran", description="I fetched water from the well.",
            category="observation", game_time=_day_time(0, 30),
        )
        llm = _StubLLM(response="A quiet day of chores.")
        asyncio.run(compact_day(mgr, "bran", 0, npc=npc, llm=llm))
        return llm.last_prompt

    def test_prompt_includes_three_point_structure(self):
        prompt = self._run_with_npc(npc=None)
        lowered = prompt.lower()
        assert "what actually happened" in lowered
        assert "how it made you feel" in lowered
        assert "shifted" in lowered  # relationships/plans line

    def test_prompt_discourages_verbatim_restatement(self):
        prompt = self._run_with_npc(npc=None)
        assert "do not restate events verbatim" in prompt.lower()

    def test_prompt_threads_personality_and_self_concept(self):
        from core.npc.models import NPC, PersonalityTraits
        npc = NPC(
            npc_id="bran", name="Bran", age=40,
            occupation="baker",
            backstory="A baker with too many opinions.",
            personality=PersonalityTraits(
                openness=0.8, conscientiousness=0.5,
                extraversion=0.2, agreeableness=0.4,
                neuroticism=0.6,
            ),
        )
        # Add a self-concept so the prompt has something to render.
        npc.self_concept["role"] = 0.9
        prompt = self._run_with_npc(npc=npc)
        # Name and occupation are inlined.
        assert "You are Bran" in prompt
        assert "baker" in prompt
        # Personality description makes it into the prompt
        # (PersonalityTraits.to_description returns a non-empty line
        # for any non-zero trait).
        assert "Personality:" in prompt
        # Self-concept line is non-empty when the NPC has beliefs.
        start = prompt.find("Personality:")
        after_personality = prompt[start:]
        # Something between personality and the events list (the
        # self_concept slot).
        assert "smaller events" in after_personality

    def test_prompt_renders_with_missing_persona_slots(self):
        """Legacy callers that don't pass `npc` must still produce a
        well-formed prompt thanks to `_MissingEmpty` tolerance."""
        prompt = self._run_with_npc(npc=None)
        # Empty personality / self_concept lines collapse cleanly.
        assert "Personality:" in prompt  # slot still labelled
        assert "smaller events of day 0" in prompt


# ---------- H.3: retrieval prefers summaries over raw ----------

class TestRetrievalHidesTombstoned:
    """Once compact_day tombstones originals, default retrieval
    paths should hide them in favour of the day_summary. Diagnostics
    can still ask for them with `include_compacted=True`."""

    def _seed_and_compact(self):
        mgr = _mgr()
        obs_id = mgr.episodic.add_memory(
            npc_id="bran",
            description="I chopped wood for an hour.",
            category="observation", importance=0.4,
            game_time=_day_time(0, 60),
        )
        tagged_id = mgr.episodic.add_memory(
            npc_id="bran",
            description="Petra accused me of hoarding bread.",
            category="accusation", importance=0.8,
            game_time=_day_time(0, 300),
            tags={"bread", "outcome:accusation"},
        )
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        return mgr, obs_id, tagged_id, summary_id

    def test_retrieve_hides_tombstoned_raw(self):
        mgr, obs_id, tagged_id, summary_id = self._seed_and_compact()
        results = mgr.episodic.retrieve(
            npc_id="bran", query="wood chopping",
            current_game_time=_day_time(1, 0),
        )
        ids = {r.memory.memory_id for r in results}
        assert obs_id not in ids
        # Summary + tagged survive and are retrievable.
        assert summary_id in ids or tagged_id in ids

    def test_retrieve_include_compacted_shows_provenance(self):
        mgr, obs_id, _, _ = self._seed_and_compact()
        results = mgr.episodic.retrieve(
            npc_id="bran", query="wood",
            current_game_time=_day_time(1, 0),
            include_compacted=True,
        )
        ids = {r.memory.memory_id for r in results}
        assert obs_id in ids

    def test_get_recent_hides_tombstoned(self):
        mgr, obs_id, _, summary_id = self._seed_and_compact()
        recents = mgr.episodic.get_recent("bran", limit=20)
        recent_ids = {m.memory_id for m in recents}
        assert obs_id not in recent_ids
        assert summary_id in recent_ids

    def test_get_recent_include_compacted_shows_tombstoned(self):
        mgr, obs_id, _, _ = self._seed_and_compact()
        recents = mgr.episodic.get_recent(
            "bran", limit=20, include_compacted=True,
        )
        assert obs_id in {m.memory_id for m in recents}

    def test_todays_raw_memories_still_visible(self):
        """The critical invariant: compaction runs on the PRIOR day,
        not today, so today's raw observations remain in retrieval."""
        mgr = _mgr()
        yesterday_id = mgr.episodic.add_memory(
            npc_id="bran", description="Yesterday I baked.",
            category="observation", game_time=_day_time(0, 60),
        )
        today_id = mgr.episodic.add_memory(
            npc_id="bran", description="Today I am kneading dough.",
            category="observation", game_time=_day_time(1, 60),
        )
        asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        recents = mgr.episodic.get_recent("bran", limit=20)
        recent_ids = {m.memory_id for m in recents}
        assert yesterday_id not in recent_ids  # compacted
        assert today_id in recent_ids          # still raw

    def test_tagged_memory_still_retrievable_by_tag(self):
        mgr, _, tagged_id, _ = self._seed_and_compact()
        hits = mgr.episodic.retrieve_by_tags("bran", ["bread"])
        assert any(m.memory_id == tagged_id for m in hits)

    def test_get_memories_in_window_hides_tombstoned_by_default(self):
        mgr, obs_id, _, summary_id = self._seed_and_compact()
        # Window covers day 0 (where the original observation lives).
        mems = mgr.episodic.get_memories_in_window(
            "bran", _day_time(0), _day_time(1),
        )
        ids = {m.memory_id for m in mems}
        assert obs_id not in ids

    def test_get_memories_in_window_include_compacted_restores(self):
        mgr, obs_id, _, _ = self._seed_and_compact()
        mems = mgr.episodic.get_memories_in_window(
            "bran", _day_time(0), _day_time(1),
            include_compacted=True,
        )
        ids = {m.memory_id for m in mems}
        assert obs_id in ids

    def test_retrieve_by_tags_hides_tombstoned_when_present(self):
        """Defensive: tagged memories bypass compaction today, but
        if a future rollup ever tombstones a tagged memory, the
        same filter should apply consistently."""
        mgr = _mgr()
        mid = mgr.episodic.add_memory(
            npc_id="bran", description="phantom",
            tags={"bread"}, game_time=_day_time(0, 10),
        )
        mgr.episodic.update_metadata(mid, {"compacted_into": "fake"})
        assert mgr.episodic.retrieve_by_tags("bran", ["bread"]) == []
        # Opt-in still returns it.
        hits = mgr.episodic.retrieve_by_tags(
            "bran", ["bread"], include_compacted=True,
        )
        assert len(hits) == 1


# ---------- H.4: week-level rollup ----------

class TestCompactWeek:
    """Week rollup collapses a 7-day window of day_summaries into a
    single `week_summary` memory, tombstoning each day_summary. The
    H.3 retrieval filter then demotes them in favour of the week.
    """

    def _seed_week(self, mgr, week: int = 0) -> list[str]:
        """Seed the 7 days of `week` each with one day_summary and
        some tagged + raw memories. Returns the day_summary ids in
        order.
        """
        day_summary_ids: list[str] = []
        base = week * DAYS_PER_WEEK
        for i in range(DAYS_PER_WEEK):
            day = base + i
            # Raw observation — tombstoned by compact_day in real
            # flow, but we skip that here and seed the summary
            # directly to test the week path in isolation.
            mgr.episodic.add_memory(
                npc_id="bran",
                description=f"Raw thing on day {day}",
                category="observation",
                game_time=_day_time(day, 100),
                extra_metadata={"compacted_into": f"ds_{day}_stub"},
            )
            # The day_summary itself.
            mid = mgr.episodic.add_memory(
                npc_id="bran",
                description=f"Day {day}: a quiet day of baking.",
                category="day_summary",
                importance=0.6,
                game_time=_day_time(day + 1) - 1.0,
                extra_metadata={
                    "day": day,
                    "kept_tags": (
                        f"bread day_{day}" if i % 2 == 0 else "traveller"
                    ),
                },
            )
            day_summary_ids.append(mid)
        # One tagged outcome mid-week that MUST survive.
        tagged_id = mgr.episodic.add_memory(
            npc_id="bran",
            description="Petra accused me of hoarding bread.",
            category="accusation", importance=0.8,
            game_time=_day_time(base + 3, 200),
            tags={"bread", "accused:bran", "outcome:accusation"},
        )
        day_summary_ids.append(f"TAGGED:{tagged_id}")
        return day_summary_ids

    def test_writes_week_summary_and_tombstones_day_summaries(self):
        mgr = _mgr()
        ids = self._seed_week(mgr, week=0)
        day_ids = [i for i in ids if not i.startswith("TAGGED:")]

        summary_id = asyncio.run(
            compact_week(mgr, "bran", 0, llm=None)
        )
        assert summary_id is not None

        summary = mgr.episodic.get_by_id(summary_id)
        assert summary is not None
        assert summary.category == "week_summary"
        assert summary.metadata.get("week") == 0
        assert summary.metadata.get("day_start") == 0
        assert summary.metadata.get("day_end") == 6
        assert summary.metadata.get("compacted_count") == 7

        compacted_from = summary.metadata.get("compacted_from", "")
        for did in day_ids:
            assert did in compacted_from

        # Each day_summary tombstoned to point at this week_summary.
        for did in day_ids:
            day_mem = mgr.episodic.get_by_id(did)
            assert day_mem.metadata.get("compacted_into") == summary_id

    def test_tagged_memory_survives_week_rollup(self):
        mgr = _mgr()
        ids = self._seed_week(mgr, week=0)
        tagged_id = [i for i in ids if i.startswith("TAGGED:")][0].split(":")[1]

        asyncio.run(compact_week(mgr, "bran", 0, llm=None))

        tagged = mgr.episodic.get_by_id(tagged_id)
        assert tagged is not None
        assert "compacted_into" not in (tagged.metadata or {})
        assert "bread" in tagged.tags
        # Still discoverable by tag after the rollup.
        hits = mgr.episodic.retrieve_by_tags("bran", ["bread"])
        assert any(m.memory_id == tagged_id for m in hits)

    def test_week_summary_aggregates_kept_tags_from_days_and_raws(self):
        mgr = _mgr()
        self._seed_week(mgr, week=0)
        summary_id = asyncio.run(compact_week(mgr, "bran", 0, llm=None))
        summary = mgr.episodic.get_by_id(summary_id)
        kept = set((summary.metadata.get("kept_tags") or "").split())
        # Both halves of the day-alternation appear.
        assert "bread" in kept
        assert "traveller" in kept
        # The surviving tagged accusation contributes its tags too.
        assert {"accused:bran", "outcome:accusation"}.issubset(kept)

    def test_week_summary_lands_at_end_of_week(self):
        mgr = _mgr()
        self._seed_week(mgr, week=0)
        summary_id = asyncio.run(compact_week(mgr, "bran", 0, llm=None))
        summary = mgr.episodic.get_by_id(summary_id)
        assert summary.game_time == _day_time(DAYS_PER_WEEK) - 1.0

    def test_week_summary_hides_day_summaries_in_retrieval(self):
        """After rollup, get_recent should surface the week_summary
        and hide the rolled-up day_summaries."""
        mgr = _mgr()
        ids = self._seed_week(mgr, week=0)
        day_ids = [i for i in ids if not i.startswith("TAGGED:")]
        summary_id = asyncio.run(compact_week(mgr, "bran", 0, llm=None))

        recents = mgr.episodic.get_recent("bran", limit=50)
        recent_ids = {m.memory_id for m in recents}
        for did in day_ids:
            assert did not in recent_ids
        assert summary_id in recent_ids

    def test_noop_when_no_day_summaries_in_window(self):
        mgr = _mgr()
        # Seed raw-only memories but no day_summaries.
        mgr.episodic.add_memory(
            npc_id="bran", description="orphan observation",
            category="observation", game_time=_day_time(3, 200),
        )
        assert asyncio.run(compact_week(mgr, "bran", 0, llm=None)) is None

    def test_noop_on_empty_week(self):
        mgr = _mgr()
        assert asyncio.run(compact_week(mgr, "bran", 5, llm=None)) is None

    def test_rerun_is_idempotent(self):
        mgr = _mgr()
        self._seed_week(mgr, week=0)
        first = asyncio.run(compact_week(mgr, "bran", 0, llm=None))
        second = asyncio.run(compact_week(mgr, "bran", 0, llm=None))
        assert first is not None
        assert second is None

    def test_scoped_to_requested_week(self):
        """Day_summaries outside the week window are left alone."""
        mgr = _mgr()
        self._seed_week(mgr, week=0)
        # Seed one day_summary for week 1 that must NOT be touched.
        outside_id = mgr.episodic.add_memory(
            npc_id="bran", description="Day 7 summary",
            category="day_summary",
            game_time=_day_time(8) - 1.0,
            extra_metadata={"day": 7},
        )
        asyncio.run(compact_week(mgr, "bran", 0, llm=None))
        outside = mgr.episodic.get_by_id(outside_id)
        assert "compacted_into" not in (outside.metadata or {})

    def test_uses_llm_response_as_week_summary(self):
        mgr = _mgr()
        self._seed_week(mgr, week=0)
        llm = _StubLLM(response="A week of small worries adding up.")
        summary_id = asyncio.run(
            compact_week(mgr, "bran", 0, llm=llm)
        )
        summary = mgr.episodic.get_by_id(summary_id)
        assert summary.description == "A week of small worries adding up."
        # Prompt lists each day_summary line.
        assert "Day 0:" in llm.last_prompt
        assert "Day 6:" in llm.last_prompt

    def test_fallback_summary_labels_the_week(self):
        mgr = _mgr()
        self._seed_week(mgr, week=2)  # days 14..20
        summary_id = asyncio.run(compact_week(mgr, "bran", 2, llm=None))
        summary = mgr.episodic.get_by_id(summary_id)
        assert "Week 2" in summary.description
        assert "14" in summary.description and "20" in summary.description


class TestWeekSummaryPrompt:
    """H.4 prompt must surface the character-arc framing, not the
    diary-entry framing of H.2."""

    def _run_with_npc(self, npc) -> str:
        mgr = _mgr()
        for day in range(DAYS_PER_WEEK):
            mgr.episodic.add_memory(
                npc_id="bran",
                description=f"Day {day}: quiet day of bread-baking.",
                category="day_summary", importance=0.6,
                game_time=_day_time(day + 1) - 1.0,
                extra_metadata={"day": day},
            )
        llm = _StubLLM(response="A slow week of becoming a baker again.")
        asyncio.run(compact_week(mgr, "bran", 0, npc=npc, llm=llm))
        return llm.last_prompt

    def test_prompt_frames_as_week_not_day(self):
        prompt = self._run_with_npc(npc=None)
        lowered = prompt.lower()
        assert "week 0" in lowered
        assert "day 0 through day 6" in lowered
        # Arc framing cue.
        assert "character arc" in lowered or "through-line" in lowered

    def test_prompt_threads_personality_and_self_concept(self):
        from core.npc.models import NPC, PersonalityTraits
        npc = NPC(
            npc_id="bran", name="Bran", age=40,
            occupation="baker",
            backstory="A baker with too many opinions.",
            personality=PersonalityTraits(
                openness=0.8, conscientiousness=0.5,
                extraversion=0.2, agreeableness=0.4,
                neuroticism=0.6,
            ),
        )
        npc.self_concept["role"] = 0.9
        prompt = self._run_with_npc(npc=npc)
        assert "You are Bran" in prompt
        assert "baker" in prompt
        assert "Personality:" in prompt


# ---------- H.5: provenance-chain access ----------

class TestProvenanceChain:
    """`get_raw_by_id` + `get_compacted_sources` let diagnostics walk
    the full provenance: week_summary → day_summaries → raw events.
    Default retrieval still hides tombstones; these methods explicitly
    don't."""

    def test_get_raw_by_id_returns_tombstoned_memory(self):
        mgr = _mgr()
        raw_id = mgr.episodic.add_memory(
            npc_id="bran", description="I chopped wood.",
            category="observation", game_time=_day_time(0, 60),
        )
        asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        # Default retrieval no longer returns it.
        recents = mgr.episodic.get_recent("bran", limit=20)
        assert raw_id not in {m.memory_id for m in recents}
        # But get_raw_by_id does.
        raw = mgr.episodic.get_raw_by_id(raw_id)
        assert raw is not None
        assert raw.description == "I chopped wood."
        assert raw.metadata.get("compacted_into")

    def test_get_raw_by_id_missing_returns_none(self):
        mgr = _mgr()
        assert mgr.episodic.get_raw_by_id("nope") is None

    def test_get_compacted_sources_on_day_summary(self):
        mgr = _mgr()
        raw_ids = [
            mgr.episodic.add_memory(
                npc_id="bran",
                description=f"Event {i}", category="observation",
                game_time=_day_time(0, 20 + i * 10),
            )
            for i in range(3)
        ]
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        sources = mgr.episodic.get_compacted_sources(summary_id)
        source_ids = {m.memory_id for m in sources}
        assert source_ids == set(raw_ids)

    def test_get_compacted_sources_on_week_summary(self):
        """A week_summary's sources are the day_summaries, not the
        original raws. Chain traversal is explicit — the caller
        walks one level at a time."""
        mgr = _mgr()
        day_ids: list[str] = []
        for d in range(DAYS_PER_WEEK):
            day_ids.append(
                mgr.episodic.add_memory(
                    npc_id="bran",
                    description=f"Day {d} summary",
                    category="day_summary",
                    importance=0.6,
                    game_time=_day_time(d + 1) - 1.0,
                    extra_metadata={"day": d},
                )
            )
        week_id = asyncio.run(compact_week(mgr, "bran", 0, llm=None))

        week_sources = mgr.episodic.get_compacted_sources(week_id)
        assert {m.memory_id for m in week_sources} == set(day_ids)
        # Each source carries day_summary category.
        assert all(m.category == "day_summary" for m in week_sources)

    def test_get_compacted_sources_full_chain_traversal(self):
        """From week_summary, walk one level to day_summaries, and
        from each day_summary another level to the raw memories."""
        mgr = _mgr()
        # Day 0: one raw that will be compacted into a day_summary.
        raw_id = mgr.episodic.add_memory(
            npc_id="bran", description="I baked bread.",
            category="observation",
            game_time=_day_time(0, 60),
        )
        day0_summary = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        # Seed stub day_summaries for the rest of the week.
        for d in range(1, DAYS_PER_WEEK):
            mgr.episodic.add_memory(
                npc_id="bran", description=f"Day {d} stub",
                category="day_summary", importance=0.6,
                game_time=_day_time(d + 1) - 1.0,
                extra_metadata={"day": d},
            )
        week_id = asyncio.run(compact_week(mgr, "bran", 0, llm=None))

        # Step 1: week → day summaries (includes day0_summary).
        day_level = mgr.episodic.get_compacted_sources(week_id)
        assert day0_summary in {m.memory_id for m in day_level}

        # Step 2: day0_summary → raws.
        raw_level = mgr.episodic.get_compacted_sources(day0_summary)
        assert raw_id in {m.memory_id for m in raw_level}

    def test_get_compacted_sources_on_non_summary_returns_empty(self):
        mgr = _mgr()
        raw_id = mgr.episodic.add_memory(
            npc_id="bran", description="lonely observation",
            category="observation", game_time=_day_time(0, 30),
        )
        assert mgr.episodic.get_compacted_sources(raw_id) == []

    def test_get_compacted_sources_missing_id_returns_empty(self):
        mgr = _mgr()
        assert mgr.episodic.get_compacted_sources("nope") == []

    def test_manager_passthroughs(self):
        """MemoryManager exposes both diagnostic methods so the
        memory panel doesn't need to reach into `.episodic`."""
        mgr = _mgr()
        raw_id = mgr.episodic.add_memory(
            npc_id="bran", description="I fetched water.",
            category="observation", game_time=_day_time(0, 30),
        )
        summary_id = asyncio.run(compact_day(mgr, "bran", 0, llm=None))
        # Raw still reachable via manager.
        assert mgr.get_raw_by_id(raw_id) is not None
        # Sources walkable via manager.
        sources = mgr.get_compacted_sources(summary_id)
        assert raw_id in {m.memory_id for m in sources}
