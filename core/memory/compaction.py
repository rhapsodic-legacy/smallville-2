"""
Memory compaction — Phase H of MEMORY_V2_ROADMAP.

Periodically collapses untagged day-to-day memories into a single
`day_summary` memory, tombstoning the originals with a
`compacted_into` metadata pointer. Tagged memories bypass compaction
entirely: Phase K's retention contract requires they remain findable
after this pass, so the summariser never touches them.

The module exposes `compact_day(manager, npc_id, game_day, ...)` which
the `MemoryManager` wraps as a thin async method. Keeping the logic
out of `manager.py` holds that already-oversized file below further
growth.

Day windows are `[day * MINUTES_PER_DAY, (day + 1) * MINUTES_PER_DAY)`
in game minutes, matching how `core.npc.manager` assembles
`current_minutes`.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from core.memory.episodic import EpisodicMemory
from core.time_system.clock import MINUTES_PER_DAY

if TYPE_CHECKING:
    from core.memory.manager import MemoryManager
    from core.npc.llm_client import LLMProvider
    from core.npc.models import NPC

logger = logging.getLogger(__name__)


# Categories whose memories carry structured meaning the untagged
# summariser would destroy. Belt-and-braces alongside the has-tags
# check — if a tag derivation bug ever slipped an outcome memory
# through without tags, this second filter still protects it.
PRESERVED_CATEGORIES: frozenset[str] = frozenset({
    "day_summary",
    "week_summary",
    "reflection",
    "commitment",
    "accusation",
    "relayed_claim",
    "town_event",
    "town_failure",
    "town_agenda",
    "note",
    # Phase I.1 — bedtime commitment reviews are structured self-
    # reflection the NPC needs to be able to surface weeks later
    # ("why have I been putting this off?"). Compaction would lose
    # the [status] tags that make the review queryable at all.
    "commitment_review",
})

# Category + importance for the summary memory itself.
DAY_SUMMARY_CATEGORY: str = "day_summary"
DAY_SUMMARY_IMPORTANCE: float = 0.6

# Week-level rollup (H.4). One call per NPC per 7 game days, acting
# on the day_summaries those days produced — NOT on raw memories.
# Importance slightly higher than a single day: a week survives
# retrieval-ranking longer because it represents more of the NPC's
# story.
WEEK_SUMMARY_CATEGORY: str = "week_summary"
WEEK_SUMMARY_IMPORTANCE: float = 0.65
DAYS_PER_WEEK: int = 7

# Hard cap on summary length; the LLM is already prompted for 2-4
# sentences but a runaway response shouldn't balloon the episodic
# store.
DAY_SUMMARY_MAX_CHARS: int = 800
WEEK_SUMMARY_MAX_CHARS: int = 1200

# Same delimiter used by EpisodicStore for tag lists in ChromaDB
# metadata (scalar-only). Reused here for `compacted_from` (list of
# ids) and `kept_tags` (list of tag strings) so the round-trip
# semantics match.
_LIST_METADATA_DELIM: str = " "


def is_compactable(mem: EpisodicMemory) -> bool:
    """Return True if `mem` is a candidate for day-level compaction.

    A memory is compactable when ALL of:
    - it carries no Phase K tags (tagged memories are anchored
      retention — they must survive compaction intact),
    - its category isn't on the preserved list (outcomes, reflections,
      already-compacted summaries),
    - it hasn't already been compacted into a summary on a prior pass
      (`compacted_into` metadata absent).
    """
    if mem.tags:
        return False
    if mem.category in PRESERVED_CATEGORIES:
        return False
    meta = getattr(mem, "metadata", None) or {}
    if meta.get("compacted_into"):
        return False
    return True


def _format_events_for_prompt(mems: list[EpisodicMemory]) -> str:
    """Bullet-list the compactable memories for the summariser prompt."""
    lines: list[str] = []
    for mem in mems:
        desc = (mem.description or "").strip()
        if not desc:
            continue
        lines.append(f"- {desc}")
    return "\n".join(lines)


def _fallback_summary(
    npc_name: str, day: int, mems: list[EpisodicMemory],
) -> str:
    """Heuristic summary when no LLM is available.

    Keeps the summary short and honest — lists the first few events
    and the count of the rest. Good enough to tombstone noise while
    leaving a breadcrumb for debugging.
    """
    if not mems:
        return f"Day {day}: nothing of note happened."
    previews: list[str] = []
    for mem in mems[:3]:
        desc = (mem.description or "").strip().rstrip(".")
        if desc:
            previews.append(desc)
    extra = len(mems) - len(previews)
    text = f"Day {day}: " + "; ".join(previews)
    if extra > 0:
        text += f"; and {extra} other small matter{'s' if extra != 1 else ''}"
    return text[:DAY_SUMMARY_MAX_CHARS].rstrip() + "."


def _persona_for_prompt(npc: NPC | None) -> tuple[str, str, str, str]:
    """Pull (name, occupation, personality, self_concept) off `npc`.

    Each field falls back to a neutral default so the prompt stays
    well-formed when compaction runs in contexts (tests, ad-hoc
    tools) that can't provide a full NPC. `personality.to_description`
    and `self_concept_summary` are existing accessors used by the
    other NPC-voiced prompts.
    """
    if npc is None:
        return ("I", "townsperson", "", "")
    name = getattr(npc, "name", "someone") or "someone"
    occupation = getattr(npc, "occupation", "townsperson") or "townsperson"
    personality_obj = getattr(npc, "personality", None)
    personality = (
        personality_obj.to_description()
        if personality_obj is not None
        and hasattr(personality_obj, "to_description")
        else ""
    )
    self_concept = ""
    if hasattr(npc, "self_concept_summary"):
        try:
            self_concept = npc.self_concept_summary() or ""
        except Exception:
            self_concept = ""
    return name, occupation, personality, self_concept


async def _summarise_with_llm(
    llm: LLMProvider,
    npc: NPC | None,
    day: int,
    mems: list[EpisodicMemory],
) -> str:
    """Call the LLM to produce the day summary. Raises on failure."""
    from core.npc.llm_client import format_prompt

    events_block = _format_events_for_prompt(mems)
    name, occupation, personality, self_concept = _persona_for_prompt(npc)
    prompt = format_prompt(
        "day_summary",
        name=name,
        occupation=occupation,
        personality=personality,
        self_concept=self_concept,
        day=day,
        events=events_block,
    )
    response = await llm.complete(
        system=(
            "You compress an NPC's day into a short first-person "
            "reflection. Stay in character. Prefer voice and feeling "
            "over event-listing."
        ),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4,
        purpose="day_summary",
    )
    text = (response or "").strip()
    if not text:
        raise RuntimeError("empty day_summary response")
    return text[:DAY_SUMMARY_MAX_CHARS]


def _aggregate_kept_tags(tagged_mems: list[EpisodicMemory]) -> list[str]:
    """Return a sorted list of every tag present in `tagged_mems`."""
    tags: set[str] = set()
    for mem in tagged_mems:
        tags.update(mem.tags)
    return sorted(tags)


async def compact_day(
    manager: MemoryManager,
    npc_id: str,
    game_day: int,
    *,
    npc: NPC | None = None,
    llm: LLMProvider | None = None,
) -> str | None:
    """Collapse a day's untagged memories into one `day_summary`.

    Returns the memory id of the summary written, or None when the
    day has nothing to compact (empty, all tagged, or all already
    compacted). Tombstones each compacted original with
    `{"compacted_into": summary_id}` so callers can still follow the
    provenance chain via `EpisodicStore.get_by_id`.

    `llm` defaults to `manager.llm` when available; pass an explicit
    provider (or `None`) to force the heuristic fallback. `npc` is
    used for name + occupation in the summariser prompt; omit it and
    the prompt falls back to generic pronouns.
    """
    start = game_day * MINUTES_PER_DAY
    end = start + MINUTES_PER_DAY
    day_memories = manager.episodic.get_memories_in_window(
        npc_id, start, end,
    )
    if not day_memories:
        return None

    compactable = [m for m in day_memories if is_compactable(m)]
    if not compactable:
        return None

    tagged = [m for m in day_memories if m.tags]
    kept_tags = _aggregate_kept_tags(tagged)

    effective_llm = llm if llm is not None else getattr(manager, "llm", None)
    summary_text: str
    if effective_llm is not None:
        try:
            summary_text = await _summarise_with_llm(
                effective_llm, npc, game_day, compactable,
            )
        except Exception as e:
            logger.warning(
                "Day-summary LLM call failed for %s day %d: %s — "
                "falling back to heuristic", npc_id, game_day, e,
            )
            summary_text = _fallback_summary(
                getattr(npc, "name", npc_id) if npc is not None else npc_id,
                game_day, compactable,
            )
    else:
        summary_text = _fallback_summary(
            getattr(npc, "name", npc_id) if npc is not None else npc_id,
            game_day, compactable,
        )

    compacted_ids = [m.memory_id for m in compactable]
    summary_id = manager.episodic.add_memory(
        npc_id=npc_id,
        description=summary_text,
        category=DAY_SUMMARY_CATEGORY,
        importance=DAY_SUMMARY_IMPORTANCE,
        # Stamp the summary at the last second of the day so
        # recency ordering places it just after the day's events.
        game_time=end - 1.0,
        extra_metadata={
            "day": game_day,
            "compacted_from": _LIST_METADATA_DELIM.join(compacted_ids),
            "compacted_count": len(compacted_ids),
            "kept_tags": _LIST_METADATA_DELIM.join(kept_tags),
        },
    )

    for mem in compactable:
        manager.episodic.update_metadata(
            mem.memory_id, {"compacted_into": summary_id},
        )

    logger.info(
        "Compacted %d memories for %s day %d → %s (%d tagged kept)",
        len(compactable), npc_id, game_day,
        summary_id, len(tagged),
    )
    return summary_id


# ---------- H.4: week-level rollup ----------

def _parse_kept_tags_from_summary(mem: EpisodicMemory) -> set[str]:
    """Extract the aggregated tag list off a day_summary's metadata.

    Day-summaries stamp their aggregated week-relevant tags into
    `kept_tags` (space-delimited, mirroring the tags-field convention).
    Week rollup unions those across the 7 days so downstream retrieval
    can still find "bread" in Bran's second week even after the
    day_summaries are themselves compacted.
    """
    meta = mem.metadata or {}
    raw = meta.get("kept_tags")
    if not raw:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(t) for t in raw if t}
    return {t for t in str(raw).split(_LIST_METADATA_DELIM) if t}


def _is_week_compactable(mem: EpisodicMemory) -> bool:
    """Eligible day_summaries only: untombstoned, right category,
    untagged (every normal day_summary is untagged; a test-injected
    tagged one would be anchored retention and skip rollup).
    """
    if mem.category != DAY_SUMMARY_CATEGORY:
        return False
    if mem.tags:
        return False
    meta = getattr(mem, "metadata", None) or {}
    if meta.get("compacted_into"):
        return False
    return True


def _format_day_summaries_for_prompt(mems: list[EpisodicMemory]) -> str:
    """Render each day_summary as a bullet with its day number.

    The day number comes from the metadata stamp `compact_day` lays
    down; if it's missing (manually-inserted test memory) fall back
    to the ordinal position of the memory in the window.
    """
    lines: list[str] = []
    for i, mem in enumerate(mems):
        meta = mem.metadata or {}
        day = meta.get("day")
        label = f"Day {day}" if day is not None else f"Day #{i + 1}"
        text = (mem.description or "").strip()
        if text:
            lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def _fallback_week_summary(
    npc_name: str, week: int, day_start: int, day_end: int,
    mems: list[EpisodicMemory],
) -> str:
    """Heuristic week summary — a short join of the day summaries."""
    if not mems:
        return f"Week {week} (days {day_start}-{day_end}): a quiet stretch."
    previews: list[str] = []
    for mem in mems[:3]:
        desc = (mem.description or "").strip().rstrip(".")
        if desc:
            previews.append(desc)
    extra = len(mems) - len(previews)
    text = f"Week {week} (days {day_start}-{day_end}): " + " | ".join(previews)
    if extra > 0:
        text += f"; plus {extra} other day{'s' if extra != 1 else ''}."
    return text[:WEEK_SUMMARY_MAX_CHARS].rstrip() + "."


async def _summarise_week_with_llm(
    llm: LLMProvider,
    npc: NPC | None,
    week: int,
    day_start: int,
    day_end: int,
    mems: list[EpisodicMemory],
) -> str:
    """Call the LLM to produce the week summary. Raises on failure."""
    from core.npc.llm_client import format_prompt

    name, occupation, personality, self_concept = _persona_for_prompt(npc)
    day_block = _format_day_summaries_for_prompt(mems)
    prompt = format_prompt(
        "week_summary",
        name=name,
        occupation=occupation,
        personality=personality,
        self_concept=self_concept,
        week=week,
        day_start=day_start,
        day_end=day_end,
        day_summaries=day_block,
    )
    response = await llm.complete(
        system=(
            "You compress a week of an NPC's life into a short "
            "first-person character-arc reflection. Favour feeling "
            "and shift over event-listing. Stay in character."
        ),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=280,
        temperature=0.4,
        purpose="week_summary",
    )
    text = (response or "").strip()
    if not text:
        raise RuntimeError("empty week_summary response")
    return text[:WEEK_SUMMARY_MAX_CHARS]


async def compact_week(
    manager: MemoryManager,
    npc_id: str,
    week_number: int,
    *,
    npc: NPC | None = None,
    llm: LLMProvider | None = None,
) -> str | None:
    """Collapse week `week_number`'s day_summaries into one
    `week_summary` memory.

    Operates on `day_summary` memories (NOT raw) in the window
    `[week * 7, (week + 1) * 7)` game-days. Tagged memories and any
    still-untouched raws from days that were never compacted are
    left alone. Returns the new summary id, or `None` when nothing
    compactable exists in the window (e.g. the week hasn't yet had
    its day-level compaction run).

    The week_summary metadata carries:
    - `week`: week number,
    - `day_start`, `day_end`: inclusive span,
    - `compacted_from`: space-delimited day_summary ids,
    - `compacted_count`,
    - `kept_tags`: union of every day_summary's own `kept_tags`
      field plus the tags on any raw tagged memory that survived
      in the window. This keeps Phase K's surgical-pointer chain
      intact across the week rollup.
    """
    day_start = week_number * DAYS_PER_WEEK
    day_end = day_start + DAYS_PER_WEEK  # exclusive
    start_time = day_start * MINUTES_PER_DAY
    end_time = day_end * MINUTES_PER_DAY
    window_memories = manager.episodic.get_memories_in_window(
        npc_id, start_time, end_time,
    )
    if not window_memories:
        return None

    compactable = [m for m in window_memories if _is_week_compactable(m)]
    if not compactable:
        return None

    # kept_tags: union of every day_summary's own kept_tags plus any
    # tagged memory still alive in the window.
    kept: set[str] = set()
    for mem in compactable:
        kept.update(_parse_kept_tags_from_summary(mem))
    for mem in window_memories:
        if mem.tags and not (mem.metadata or {}).get("compacted_into"):
            kept.update(mem.tags)
    kept_tags_sorted = sorted(kept)

    effective_llm = llm if llm is not None else getattr(manager, "llm", None)
    if effective_llm is not None:
        try:
            summary_text = await _summarise_week_with_llm(
                effective_llm, npc, week_number,
                day_start, day_end - 1, compactable,
            )
        except Exception as e:
            logger.warning(
                "Week-summary LLM call failed for %s week %d: %s — "
                "falling back to heuristic", npc_id, week_number, e,
            )
            summary_text = _fallback_week_summary(
                getattr(npc, "name", npc_id) if npc is not None else npc_id,
                week_number, day_start, day_end - 1, compactable,
            )
    else:
        summary_text = _fallback_week_summary(
            getattr(npc, "name", npc_id) if npc is not None else npc_id,
            week_number, day_start, day_end - 1, compactable,
        )

    compacted_ids = [m.memory_id for m in compactable]
    summary_id = manager.episodic.add_memory(
        npc_id=npc_id,
        description=summary_text,
        category=WEEK_SUMMARY_CATEGORY,
        importance=WEEK_SUMMARY_IMPORTANCE,
        # Stamp at the last second of the week for recency ordering.
        game_time=end_time - 1.0,
        extra_metadata={
            "week": week_number,
            "day_start": day_start,
            "day_end": day_end - 1,
            "compacted_from": _LIST_METADATA_DELIM.join(compacted_ids),
            "compacted_count": len(compacted_ids),
            "kept_tags": _LIST_METADATA_DELIM.join(kept_tags_sorted),
        },
    )

    # Tombstone the rolled-up day_summaries. Same mechanism as
    # day-level: the H.3 retrieval filter will then demote them
    # in favour of this week_summary without further code.
    for mem in compactable:
        manager.episodic.update_metadata(
            mem.memory_id, {"compacted_into": summary_id},
        )

    logger.info(
        "Compacted %d day_summaries for %s week %d → %s (%d kept tags)",
        len(compactable), npc_id, week_number,
        summary_id, len(kept_tags_sorted),
    )
    return summary_id
