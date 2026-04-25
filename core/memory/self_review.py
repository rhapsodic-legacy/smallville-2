"""
Bedtime self-review — Phase I.1 + I.2 of MEMORY_V2_ROADMAP.

Runs once per NPC per game day, on the same day-rollover tick that
fires `compact_day`, after the day_summary has been written. Reads:

- The fresh `day_summary` (what actually happened today, in the NPC's
  voice — output of H.1)
- Unresolved self-commitments (Phase B `category="commitment"` entries
  on the NPC's own store where `unresolved=True`)
- The NPC's open `long_term_goals`
- Personality + self_concept for voice-threading

Emits a single `commitment_review` memory: the NPC's own "what was I
trying to do, did it move, what next?" reflection with per-goal
progress labels (moving / stalled / abandoned / done) and, optionally,
an `ActionIntent` that the NPCManager queues onto tomorrow's schedule
via the same `_inject_reflection_entry` path reflection already uses.

Designed to run AFTER `compact_day`: the day_summary is the primary
input, so ordering matters. See `NPCManager._run_daily_self_review`
for the wiring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.memory.episodic import EpisodicMemory
from core.memory.reflection import ActionIntent, classify_insight
from core.time_system.clock import MINUTES_PER_DAY

if TYPE_CHECKING:
    from core.memory.manager import MemoryManager
    from core.npc.llm_client import LLMProvider
    from core.npc.models import NPC
    from core.world.town_agenda import TownGoal

logger = logging.getLogger(__name__)


# Category + importance of the review memory itself. Slightly higher
# than a day_summary (0.6) because this is structured self-reflection
# tied to active agenda items — the sort of thing the NPC should still
# surface on retrieval a week later.
REVIEW_CATEGORY: str = "commitment_review"
REVIEW_IMPORTANCE: float = 0.7

# Terminal and progress states the LLM is asked to choose between.
# Kept deliberately small so the parser stays trivial.
VALID_STATUSES: frozenset[str] = frozenset({
    "moving", "stalled", "abandoned", "done",
})

# Hard cap on the review memory's text — the LLM is prompted short
# but a runaway response shouldn't balloon the store.
REVIEW_MAX_CHARS: int = 900

# How many unresolved self-commitments the review considers. More than
# this and the bedtime prompt blows up. The remainder is handled next
# night (older entries drop to the front as their holder resolves or
# abandons the top ones).
MAX_COMMITMENTS_IN_REVIEW: int = 6

# Same list-metadata delimiter `compaction.py` uses, so any tooling
# that already parses `kept_tags` / `compacted_from` handles the
# `source_ids` field on review memories identically.
_LIST_METADATA_DELIM: str = " "

# Phase I.3 — per-commitment stagnation counter. Lives on the
# `commitment` memory's metadata, not on the review memory, because
# the retrieval boost (`MemoryManager.retrieve_unresolved_matters`)
# needs to read it off each candidate commitment directly. The
# counter is unbounded by design: I.5 reads the raw value to decide
# whether to emit a soft identity delta after prolonged stagnation,
# so we need the signal to keep accumulating even after the
# retrieval-side boost saturates (see `STAGNATION_BOOST_CAP`).
STAGNATION_METADATA_KEY: str = "stagnation_days"

# Phase I.5 — identity erosion threshold + delta. When a commitment
# crosses `STAGNATION_IDENTITY_THRESHOLD` stagnation days for the
# first time, the NPC internalises the failure: either a specific
# self_concept key matching the subject drops by IDENTITY_DELTA, or
# (fallback) `unreliable:self` is introduced/strengthened by the
# same magnitude. Fires exactly once per commitment, gated by the
# `identity_eroded` metadata flag.
#
# TUNING WATCHLIST: these numbers are initial guesses. 20-day
# threshold = 5 days past the retrieval cap (15), giving NPCs time
# to vocally raise the matter before identity starts sliding. The
# flat -0.1 delta is the simple choice; a proportional variant
# (`-0.05 * (stagnation_days - THRESHOLD + 1)` capped) is listed in
# MEMORY_V2_ROADMAP.md's "Tuning watchlist" section as a candidate
# if long sims show this is too gentle or too harsh.
STAGNATION_IDENTITY_THRESHOLD: int = 20
IDENTITY_ERODED_FLAG: str = "identity_eroded"
IDENTITY_DELTA: float = 0.1  # magnitude, applied as ±
IDENTITY_FALLBACK_KEY: str = "unreliable:self"

# Phase I.4 — reinforcement magnitude on town-goal completion. Every
# contributor to a completed goal gets `+REINFORCEMENT_DELTA` applied
# to the goal's `identity_key` (e.g. `built:bridge`). Chosen symmetric
# with IDENTITY_DELTA to isolate magnitude as the one variable we're
# tuning; revisit asymmetry only if 60-day sims show erosion
# dominating completion. See MEMORY_V2_ROADMAP.md "Tuning watchlist".
REINFORCEMENT_DELTA: float = 0.1

# Minimum word length for subject-match scanning. Words shorter than
# this are too generic ("the", "was") to indicate a meaningful
# identity anchor. 4 captures "bread", "town", "roof" etc. cleanly.
SUBJECT_MATCH_MIN_LENGTH: int = 4


@dataclass
class GoalProgress:
    """Per-goal verdict the LLM (or fallback) emits.

    `goal_text` is the commitment description or long-term goal as
    the review saw it, `status` is one of VALID_STATUSES, and `note`
    is the short free-text line the NPC would tell themselves about
    it ("the roof still leaks; I need to buy nails tomorrow").
    """
    goal_text: str
    status: str
    note: str = ""

    def is_valid(self) -> bool:
        return self.status in VALID_STATUSES


@dataclass
class IdentityErosionEvent:
    """One `self_concept` delta emitted by a stagnated commitment.

    `commitment_id` is the source commitment that triggered the
    erosion. `self_concept_key` is the `prefix:target` key whose
    confidence moved; `delta` is the signed amount applied (negative
    when a strong existing belief was weakened, positive when
    `unreliable:self` was introduced/strengthened). `new_confidence`
    is the post-apply value (0.0 when the belief dropped below the
    floor and was removed).
    """
    commitment_id: str
    self_concept_key: str
    delta: float
    new_confidence: float
    reflection_memory_id: str = ""


@dataclass
class IdentityReinforcementEvent:
    """One `self_concept` delta emitted by a completed town goal.

    Phase I.4 mirror of `IdentityErosionEvent`: `goal_id` is the town
    goal the NPC contributed to, `self_concept_key` is the reinforced
    `prefix:target` key (copied from the goal template), `delta` is
    always `+REINFORCEMENT_DELTA` (unlike erosion there is no
    fallback branch — every goal carries an explicit identity_key),
    and `new_confidence` is the post-apply value clamped to [0, 1].
    """
    goal_id: str
    self_concept_key: str
    delta: float
    new_confidence: float
    reflection_memory_id: str = ""


@dataclass
class SelfReviewResult:
    """Everything `daily_self_review` produced for this NPC/day.

    `memory_id` is the `commitment_review` memory that was persisted.
    `action_intent` is populated only when the final "what's next"
    line classifies as actionable via the existing Phase B path.
    `identity_erosions` lists any Phase I.5 events that fired when
    a commitment first crossed the stagnation identity threshold.
    """
    memory_id: str
    summary_text: str
    per_goal: list[GoalProgress]
    action_intent: ActionIntent | None = None
    source_ids: list[str] = field(default_factory=list)
    kept_tags: list[str] = field(default_factory=list)
    identity_erosions: list[IdentityErosionEvent] = field(default_factory=list)


# ---------- Commitment lookup ----------


def _unresolved_self_commitments(
    manager: MemoryManager, npc_id: str, limit: int,
) -> list[EpisodicMemory]:
    """Return the NPC's own open commitments, newest first.

    Phase B lands a `commitment` memory on the speaker's own store
    with `unresolved=True` in metadata and flips it to False once
    the matter is aired (see `resolve_matters_from_transcript`). So
    every `unresolved` commitment on an NPC's store is, by
    construction, something *they themselves* pledged to do.
    """
    fetched = manager.episodic.get_recent(
        npc_id, limit=max(limit * 2, 12), category="commitment",
    )
    open_only: list[EpisodicMemory] = []
    for mem in fetched:
        meta = getattr(mem, "metadata", None) or {}
        if meta.get("unresolved"):
            open_only.append(mem)
        if len(open_only) >= limit:
            break
    return open_only


def _latest_day_summary(
    manager: MemoryManager, npc_id: str, game_day: int,
) -> EpisodicMemory | None:
    """Fetch the `day_summary` compaction just wrote, if any.

    Day-summaries are stamped at `(day+1)*MINUTES_PER_DAY - 1.0`, so
    a window fetch pinned to the target day returns it. A retrieval
    with `include_compacted=True` would also work, but this avoids
    surfacing the future week_summary that rolls it up.
    """
    start = game_day * MINUTES_PER_DAY
    end = start + MINUTES_PER_DAY
    window = manager.episodic.get_memories_in_window(
        npc_id, start, end, include_compacted=True,
    )
    for mem in reversed(window):
        if mem.category == "day_summary":
            return mem
    return None


# ---------- Prompt plumbing ----------


def _persona_for_prompt(npc: NPC | None) -> tuple[str, str, str, str]:
    """Pull (name, occupation, personality, self_concept).

    Mirrors `compaction._persona_for_prompt` so the voice lines up
    across the day_summary and commitment_review the NPC emits the
    same night. Kept as a local copy rather than imported so
    `self_review` doesn't depend on a private name in `compaction`.
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


def _format_commitments(mems: list[EpisodicMemory]) -> str:
    """Render the unresolved self-commitments as a numbered block."""
    if not mems:
        return "(none that you made and haven't already seen through)"
    lines: list[str] = []
    for i, mem in enumerate(mems, start=1):
        desc = (mem.description or "").strip().rstrip(".")
        if desc:
            lines.append(f"{i}. {desc}.")
    return "\n".join(lines) if lines else "(none)"


def _format_long_term_goals(npc: NPC | None) -> str:
    """Render the NPC's long-term goals — pulled straight off the
    model — so the LLM can judge progress against them, not just
    against commitments. Falls through to a neutral line when the
    NPC has no long-term goals set."""
    goals = list(getattr(npc, "long_term_goals", []) or []) if npc else []
    if not goals:
        return "(you hold no long-term goals you're actively pursuing)"
    return "\n".join(f"- {g.strip().rstrip('.')}." for g in goals if g)


def _format_day_summary(summary: EpisodicMemory | None) -> str:
    """Show the LLM its own recollection of the day just ended. When
    compaction produced nothing — an empty day, or the NPC was tier-4
    all day — we degrade to a neutral marker the prompt can still
    reason over."""
    if summary is None:
        return "(you have no day_summary yet; consider today a quiet one)"
    return (summary.description or "").strip()


SELF_REVIEW_PROMPT_NAME: str = "self_review"


# ---------- Parsing ----------


def _parse_review_response(
    text: str, commitments: list[EpisodicMemory], long_term: list[str],
) -> tuple[list[GoalProgress], str, str]:
    """Parse the structured block-format LLM response.

    Expected shape (tolerant of extra whitespace / missing blocks):

        SUMMARY: <one-two sentence overall>
        GOAL: <goal text>
        STATUS: moving|stalled|abandoned|done
        NOTE: <what shifted, short>
        GOAL: ...
        STATUS: ...
        NOTE: ...
        NEXT: <one short line on what tomorrow means, or NO_ACTION>

    Returns (per_goal, summary_text, next_line). Keys beyond the
    recognised set are ignored; missing STATUS defaults to `stalled`
    so the review still records something rather than dropping the
    goal on the floor.
    """
    summary_text = ""
    next_line = ""
    per_goal: list[GoalProgress] = []
    current: dict[str, str] = {}

    def flush():
        goal = (current.get("GOAL") or "").strip()
        if not goal:
            return
        status = (current.get("STATUS") or "stalled").strip().lower()
        if status not in VALID_STATUSES:
            status = "stalled"
        note = (current.get("NOTE") or "").strip()
        per_goal.append(
            GoalProgress(goal_text=goal, status=status, note=note),
        )

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper, _, rest = line.partition(":")
        key = upper.strip().upper()
        value = rest.strip()
        if key == "SUMMARY":
            summary_text = value
        elif key == "NEXT":
            next_line = value
        elif key == "GOAL":
            flush()
            current = {"GOAL": value}
        elif key == "STATUS":
            current["STATUS"] = value
        elif key == "NOTE":
            current["NOTE"] = value
    flush()

    # Tolerant fallback: if the model never emitted GOAL blocks but
    # did produce a free-text review, synthesise `stalled` entries
    # for every known commitment + long-term goal so the structure
    # downstream code expects is always populated.
    if not per_goal:
        for mem in commitments:
            per_goal.append(
                GoalProgress(
                    goal_text=(mem.description or "").strip(),
                    status="stalled",
                ),
            )
        for g in long_term:
            per_goal.append(
                GoalProgress(goal_text=g.strip(), status="stalled"),
            )

    return per_goal, summary_text, next_line


def _fallback_review(
    npc_name: str,
    day: int,
    commitments: list[EpisodicMemory],
    long_term: list[str],
) -> tuple[list[GoalProgress], str, str]:
    """Produce a deterministic review when no LLM is available.

    Every known goal is marked `stalled` — we have no signal either
    way — and the summary text is a terse one-liner. This keeps the
    bedtime path useful under the deterministic router verdict or
    during unit tests without needing a mock LLM response.
    """
    per_goal: list[GoalProgress] = []
    for mem in commitments:
        per_goal.append(
            GoalProgress(
                goal_text=(mem.description or "").strip(),
                status="stalled",
                note="",
            ),
        )
    for g in long_term:
        per_goal.append(
            GoalProgress(goal_text=g.strip(), status="stalled"),
        )

    if not per_goal:
        summary = f"Day {day}: nothing I committed to moved today."
    else:
        summary = (
            f"Day {day}: {len(per_goal)} thing"
            f"{'s' if len(per_goal) != 1 else ''} I owe myself; "
            "none shifted today."
        )
    return per_goal, summary[:REVIEW_MAX_CHARS], ""


# ---------- LLM path ----------


async def _run_review_with_llm(
    llm: LLMProvider,
    npc: NPC | None,
    day: int,
    commitments: list[EpisodicMemory],
    long_term: list[str],
    day_summary: EpisodicMemory | None,
) -> str:
    """Call the LLM once; raise on empty/failure so the caller can
    fall back cleanly."""
    from core.npc.llm_client import format_prompt

    name, occupation, personality, self_concept = _persona_for_prompt(npc)
    prompt = format_prompt(
        SELF_REVIEW_PROMPT_NAME,
        name=name,
        occupation=occupation,
        personality=personality,
        self_concept=self_concept,
        day=day,
        commitments=_format_commitments(commitments),
        long_term_goals=_format_long_term_goals(npc),
        day_summary=_format_day_summary(day_summary),
    )
    response = await llm.complete(
        system=(
            "You run an NPC's bedtime self-review in first person. "
            "Be honest, specific, and short — this is them talking "
            "to themselves before sleep, not writing a diary."
        ),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=320,
        temperature=0.4,
        purpose="self_review",
    )
    text = (response or "").strip()
    if not text:
        raise RuntimeError("empty self_review response")
    return text


# ---------- Phase I.3: stagnation counter ----------


def _apply_stagnation_updates(
    manager: MemoryManager,
    commitments: list[EpisodicMemory],
    per_goal: list[GoalProgress],
) -> dict[str, int]:
    """Bump each commitment's `stagnation_days` counter per verdict.

    Matching rule: positional. Both `_parse_review_response` and
    `_fallback_review` enumerate commitments before long-term goals,
    so `per_goal[i]` is the verdict on `commitments[i]` for
    `i < len(commitments)`. If the LLM returned fewer structured
    blocks than we asked about, the trailing commitments are
    treated as `stalled` — a goal the NPC didn't even bring up in
    their own bedtime review is, for all intents and purposes,
    stagnating. Trailing `per_goal` entries are long-term goals or
    hallucinations; stagnation isn't tracked on those.

    Status → counter transition:
    - `moving`: reset to 0. The NPC reports progress — even if modest
      — and we want the boost to fall off immediately.
    - `stalled`: `+=1`. The classic escalation signal.
    - `abandoned`: frozen. The NPC has consciously dropped this;
      incrementing further would misread their decision. I.5 reads
      the frozen value to decide on a soft identity delta.
    - `done`: reset to 0. The NPC feels finished. Phase B resolution
      (flipping `unresolved=False`) is a conversation-level signal
      that follows separately when the topic is aired with a
      partner — the counter reset here is independent.
    - any other / missing: no change.

    Returns a `{memory_id: new_value}` map so callers (tests, the
    review's own return payload) can confirm the update shape
    without having to re-read the store.
    """
    updated: dict[str, int] = {}
    if not commitments:
        return updated

    for i, commitment in enumerate(commitments):
        meta = getattr(commitment, "metadata", None) or {}
        existing = int(meta.get(STAGNATION_METADATA_KEY, 0) or 0)

        status = per_goal[i].status if i < len(per_goal) else "stalled"

        if status == "moving" or status == "done":
            new_value = 0
        elif status == "stalled":
            new_value = existing + 1
        elif status == "abandoned":
            new_value = existing
        else:
            new_value = existing

        if new_value == existing:
            continue
        manager.episodic.update_metadata(
            commitment.memory_id,
            {STAGNATION_METADATA_KEY: new_value},
        )
        # Mirror the write into the in-memory object so callers
        # reading the same reference in this pass (e.g. tests that
        # fetched `commitments` once) see the new value without a
        # round-trip through the store.
        if isinstance(meta, dict):
            meta[STAGNATION_METADATA_KEY] = new_value
            commitment.metadata = meta
        updated[commitment.memory_id] = new_value

    return updated


# ---------- Phase I.5: soft identity erosion ----------


# Tokenisation split set for subject-match scanning. Keeps the
# regex dependency out and matches the rough whitespace+punctuation
# split Phase B's `_extract_topic_tokens` uses.
_SUBJECT_SPLIT_CHARS: str = " \t\n\r.,;:!?()[]{}\"'`—/\\"


def _tokenise_subject(text: str) -> list[str]:
    """Lowercase-word tokens ≥ `SUBJECT_MATCH_MIN_LENGTH` from `text`.

    Used to scan a commitment description for words that might
    match an existing self_concept key's target part. Keeps the
    list deterministic (preserves order, dedupe case-insensitive)
    so the match behaviour is reproducible across runs.
    """
    seen: set[str] = set()
    result: list[str] = []
    buffer: list[str] = []
    for ch in (text or "").lower():
        if ch in _SUBJECT_SPLIT_CHARS:
            if buffer:
                token = "".join(buffer)
                if (
                    len(token) >= SUBJECT_MATCH_MIN_LENGTH
                    and token not in seen
                ):
                    seen.add(token)
                    result.append(token)
                buffer = []
        else:
            buffer.append(ch)
    if buffer:
        token = "".join(buffer)
        if (
            len(token) >= SUBJECT_MATCH_MIN_LENGTH
            and token not in seen
        ):
            result.append(token)
    return result


def _find_matching_self_concept_key(
    npc: NPC, commitment_description: str,
) -> str | None:
    """Return the first self_concept key whose target overlaps the
    commitment's subject tokens.

    Walks the NPC's current self_concept dict; for each key, checks
    whether its `target` (the part after `prefix:`) contains or is
    contained by any token ≥ SUBJECT_MATCH_MIN_LENGTH pulled from
    the commitment description. Prefers the highest-confidence
    matching key so erosion hits the belief the NPC was most sure
    of. Returns None if no key matches.
    """
    self_concept = getattr(npc, "self_concept", {}) or {}
    if not self_concept:
        return None

    tokens = set(_tokenise_subject(commitment_description))
    if not tokens:
        return None

    candidates: list[tuple[str, float]] = []
    for key, confidence in self_concept.items():
        _, _, target = key.partition(":")
        target = target.lower().strip()
        if not target:
            continue
        # Normalise separators so "south_field" matches "south" or
        # "field" tokens from the commitment text.
        target_tokens = {
            part for part in target.replace("-", "_").split("_")
            if len(part) >= SUBJECT_MATCH_MIN_LENGTH
        }
        # Also consider the full target as a single word (covers
        # targets that are one word long already).
        if len(target) >= SUBJECT_MATCH_MIN_LENGTH:
            target_tokens.add(target)
        if target_tokens & tokens:
            candidates.append((key, confidence))

    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return candidates[0][0]


async def _record_erosion_reflection(
    manager: MemoryManager,
    npc_id: str,
    npc: NPC,
    commitment: EpisodicMemory,
    self_concept_key: str,
    new_confidence: float,
    game_time: float,
) -> str:
    """Write a `reflection` memory describing the erosion event.

    Tagged with the source commitment's tags so Phase K retention
    anchors still apply — an NPC scanning their `bread` tag bucket
    weeks later will find both the original commitment and the
    eventual identity-erosion reflection in one pass.
    """
    base_subject = (commitment.description or "").strip().rstrip(".")
    if base_subject.lower().startswith("i promised to "):
        base_subject = base_subject[len("I promised to "):]
    prefix, _, target = self_concept_key.partition(":")
    if prefix == IDENTITY_FALLBACK_KEY.split(":")[0]:
        tail = (
            "Perhaps I am not the sort of person who sees things "
            "through after all."
        )
    else:
        nice_target = (target or self_concept_key).replace("_", " ")
        tail = (
            f"Perhaps I am not as much of a {prefix.replace('_', ' ')} "
            f"of {nice_target} as I thought."
        )
    body = (
        f"I have been telling myself I would {base_subject} for weeks "
        f"now and nothing has moved. {tail}"
    )
    mem_id = manager.episodic.add_memory(
        npc_id=npc_id,
        description=body,
        category="reflection",
        importance=0.75,
        game_time=game_time,
        extra_metadata={
            "outcome_kind": "identity_erosion",
            "source_commitment_id": commitment.memory_id,
            "self_concept_key": self_concept_key,
            "new_confidence": new_confidence,
        },
        tags=set(commitment.tags),
    )
    return mem_id


async def _apply_identity_erosion(
    manager: MemoryManager,
    npc: NPC | None,
    commitments: list[EpisodicMemory],
    stagnation_updates: dict[str, int],
    game_time: float,
) -> list[IdentityErosionEvent]:
    """Fire at most one self_concept delta per commitment whose
    stagnation_days just crossed `STAGNATION_IDENTITY_THRESHOLD`.

    A crossing is detected by reading the post-update counter from
    `stagnation_updates[commitment.memory_id]` and checking whether
    the commitment's metadata already carries `identity_eroded=True`
    (set here on first fire to prevent repeats).

    Identity key selection:
    - Subject match: scan the commitment description for tokens that
      overlap any existing self_concept key's target. If found,
      apply `-IDENTITY_DELTA` to that key (weakens the specific
      belief).
    - Fallback: introduce or strengthen `unreliable:self` by
      `+IDENTITY_DELTA`. The phrase_map in `core/npc/models.py`
      renders it as "someone unreliable" in self_concept summaries.

    Returns the list of events fired this pass. When `npc` is None
    (e.g. a test that constructed only a MemoryManager) this is a
    no-op — the self_concept update needs the dataclass instance.
    """
    events: list[IdentityErosionEvent] = []
    if npc is None or not commitments:
        return events

    for commitment in commitments:
        new_days = stagnation_updates.get(commitment.memory_id)
        if new_days is None:
            # No update was recorded this pass (moving/done counter
            # reset to 0, or abandoned left frozen). Either way, not
            # a new crossing — skip.
            continue
        if new_days < STAGNATION_IDENTITY_THRESHOLD:
            continue
        meta = getattr(commitment, "metadata", None) or {}
        if meta.get(IDENTITY_ERODED_FLAG):
            continue

        matched_key = _find_matching_self_concept_key(
            npc, commitment.description or "",
        )
        if matched_key is not None:
            key = matched_key
            delta = -IDENTITY_DELTA
        else:
            key = IDENTITY_FALLBACK_KEY
            delta = +IDENTITY_DELTA

        new_confidence = npc.apply_self_concept_delta(key, delta)

        reflection_id = ""
        try:
            reflection_id = await _record_erosion_reflection(
                manager, commitment.npc_id, npc, commitment,
                key, new_confidence, game_time,
            )
        except Exception:
            logger.debug(
                "identity-erosion reflection write failed",
                exc_info=True,
            )

        manager.episodic.update_metadata(
            commitment.memory_id,
            {IDENTITY_ERODED_FLAG: True},
        )
        if isinstance(meta, dict):
            meta[IDENTITY_ERODED_FLAG] = True
            commitment.metadata = meta

        events.append(IdentityErosionEvent(
            commitment_id=commitment.memory_id,
            self_concept_key=key,
            delta=delta,
            new_confidence=new_confidence,
            reflection_memory_id=reflection_id,
        ))
        logger.info(
            "IDENTITY_EROSION %s: commitment %s crossed %d days → "
            "%s %+0.2f (new confidence %.2f)",
            commitment.npc_id, commitment.memory_id, new_days,
            key, delta, new_confidence,
        )

    return events


# ---------- Phase I.4 — goal-completion reinforcement ----------


def _record_reinforcement_reflection(
    manager: MemoryManager,
    npc_id: str,
    npc: NPC,
    goal: "TownGoal",
    self_concept_key: str,
    new_confidence: float,
    game_time: float,
) -> str:
    """Write a `reflection` memory describing the reinforcement event.

    Tagged `town_agenda` + the goal_id so Phase K retrieval can
    surface the reinforcement alongside related memories (the
    original town_event memory, any future commitment about the
    same thing) in one tag-bucket pass.
    """
    prefix, _, target = self_concept_key.partition(":")
    nice_target = (target or self_concept_key).replace("_", " ")
    if prefix:
        tail = (
            f"Perhaps I am more of a {prefix.replace('_', ' ')} of the "
            f"{nice_target} than I gave myself credit for."
        )
    else:
        # Defensive — shouldn't happen because identity_key is always
        # `prefix:target`, but if someone registers a malformed
        # template we don't want the listener to crash.
        tail = "Perhaps I contribute more than I give myself credit for."
    body = (
        f"We saw \"{goal.title}\" through together. {tail}"
    )
    return manager.episodic.add_memory(
        npc_id=npc_id,
        description=body,
        category="reflection",
        importance=0.75,
        game_time=game_time,
        extra_metadata={
            "outcome_kind": "identity_reinforcement",
            "source_goal_id": goal.goal_id,
            "self_concept_key": self_concept_key,
            "new_confidence": new_confidence,
        },
        tags={"town_agenda", goal.goal_id},
    )


def apply_identity_reinforcement(
    manager: MemoryManager,
    npc: NPC | None,
    goal: "TownGoal",
    game_time: float,
) -> IdentityReinforcementEvent | None:
    """Fire one reinforcement delta for an NPC who contributed to a
    completed town goal.

    Called from `NPCManager._on_goal_completed` for each contributor.
    Applies `+REINFORCEMENT_DELTA` to `goal.identity_key`, writes a
    `reflection` memory tagged with the goal_id + `town_agenda`, and
    returns the event. Idempotency is the caller's responsibility —
    `TownAgenda` fires the completion listener exactly once per goal,
    and `goal.contributors` is a set, so no guard is needed here.

    Returns None when `npc` is None (defensive, matches erosion
    helper) or when the goal has no `identity_key` attribute (old
    TownGoal instance constructed without the Phase I.4 field).
    """
    if npc is None:
        return None
    key = getattr(goal, "identity_key", None)
    if not key:
        return None

    delta = +REINFORCEMENT_DELTA
    new_confidence = npc.apply_self_concept_delta(key, delta)

    reflection_id = ""
    try:
        reflection_id = _record_reinforcement_reflection(
            manager, npc.npc_id, npc, goal, key, new_confidence, game_time,
        )
    except Exception:
        logger.debug(
            "identity-reinforcement reflection write failed",
            exc_info=True,
        )

    logger.info(
        "IDENTITY_REINFORCEMENT %s: goal %s completed → "
        "%s %+0.2f (new confidence %.2f)",
        npc.npc_id, goal.goal_id, key, delta, new_confidence,
    )

    return IdentityReinforcementEvent(
        goal_id=goal.goal_id,
        self_concept_key=key,
        delta=delta,
        new_confidence=new_confidence,
        reflection_memory_id=reflection_id,
    )


# ---------- Public entry point ----------


def _aggregate_tags(source_mems: list[EpisodicMemory]) -> list[str]:
    """Union the source commitments' tags so the review inherits
    their Phase K anchoring — a review about the bread accusation
    stays findable via the `bread` tag even after its sources are
    eventually resolved/archived."""
    tags: set[str] = set()
    for mem in source_mems:
        tags.update(mem.tags)
    return sorted(tags)


async def daily_self_review(
    manager: MemoryManager,
    npc_id: str,
    game_day: int,
    *,
    npc: NPC | None = None,
    llm: LLMProvider | None = None,
) -> SelfReviewResult | None:
    """Run the bedtime self-review for `npc` on `game_day`.

    Returns the `SelfReviewResult` describing what was written, or
    `None` when there's nothing to review (no open commitments, no
    long-term goals, and no day_summary — a genuinely blank slate).

    The review is stored as a `commitment_review` memory, stamped
    at the last second of the day so recency orders it alongside
    the day_summary the compaction pass just wrote. Tags are unioned
    from the source commitments so Phase K retention anchors still
    apply.

    `llm` defaults to `manager.llm` when available; pass an explicit
    provider to override. Pass `llm=None` with a manager that has
    no `.llm` attribute to force the heuristic fallback — useful in
    tests and when the cognition router routes the decision to
    DETERMINISTIC.
    """
    commitments = _unresolved_self_commitments(
        manager, npc_id, MAX_COMMITMENTS_IN_REVIEW,
    )
    long_term = list(getattr(npc, "long_term_goals", []) or []) if npc else []
    day_summary = _latest_day_summary(manager, npc_id, game_day)

    if not commitments and not long_term and day_summary is None:
        return None

    effective_llm = llm if llm is not None else getattr(manager, "llm", None)

    raw_response: str | None = None
    if effective_llm is not None:
        try:
            raw_response = await _run_review_with_llm(
                effective_llm, npc, game_day, commitments,
                long_term, day_summary,
            )
        except Exception as e:
            logger.warning(
                "Self-review LLM call failed for %s day %d: %s — "
                "falling back to heuristic", npc_id, game_day, e,
            )

    if raw_response is not None:
        per_goal, summary_text, next_line = _parse_review_response(
            raw_response, commitments, long_term,
        )
        if not summary_text:
            # Model skipped SUMMARY — use the raw response's first
            # non-empty line as the memory body so we still persist
            # usable first-person text.
            for candidate in raw_response.splitlines():
                candidate = candidate.strip()
                if candidate and ":" not in candidate.split(" ", 1)[0]:
                    summary_text = candidate
                    break
            if not summary_text:
                summary_text = raw_response.strip().splitlines()[0]
    else:
        per_goal, summary_text, next_line = _fallback_review(
            getattr(npc, "name", npc_id) if npc is not None else npc_id,
            game_day, commitments, long_term,
        )

    action_intent: ActionIntent | None = None
    if next_line and next_line.upper() != "NO_ACTION" and npc is not None:
        if effective_llm is not None:
            try:
                action_intent = await classify_insight(
                    npc, next_line, effective_llm,
                )
            except Exception:
                logger.debug(
                    "classify_insight raised during self-review; "
                    "skipping ActionIntent", exc_info=True,
                )
                action_intent = None

    source_ids = [mem.memory_id for mem in commitments]
    kept_tags = _aggregate_tags(commitments)

    # Phase I.3 — bump stagnation counters BEFORE writing the review
    # memory so the review's `stagnation_snapshot` metadata reflects
    # the post-update values. This happens every bedtime review: the
    # `moving` verdict resets to 0, `stalled` increments, others hold.
    stagnation_updates = _apply_stagnation_updates(
        manager, commitments, per_goal,
    )

    # Phase I.5 — any commitment whose counter just crossed the
    # identity threshold for the first time emits a soft self_concept
    # delta + a `reflection` memory. Must run AFTER the stagnation
    # update so the crossing check reads post-update values, and
    # BEFORE the review memory is written so the review metadata can
    # snapshot the erosion events.
    review_time = (game_day + 1) * MINUTES_PER_DAY - 1.0
    identity_erosions = await _apply_identity_erosion(
        manager, npc, commitments, stagnation_updates, review_time,
    )

    # Compose the memory body: summary + per-goal lines as a compact
    # block. Downstream retrieval works on the description text, so
    # the per-goal structure must be visible to keyword matching even
    # if the structured per_goal list isn't itself stored separately.
    body_parts: list[str] = [summary_text.strip().rstrip(".") + "."]
    for g in per_goal:
        line = f"[{g.status}] {g.goal_text}"
        if g.note:
            line += f" — {g.note}"
        body_parts.append(line)
    if next_line and next_line.upper() != "NO_ACTION":
        body_parts.append(f"Tomorrow: {next_line}")
    body = "\n".join(body_parts)[:REVIEW_MAX_CHARS]

    memory_id = manager.episodic.add_memory(
        npc_id=npc_id,
        description=body,
        category=REVIEW_CATEGORY,
        importance=REVIEW_IMPORTANCE,
        game_time=review_time,
        extra_metadata={
            "day": game_day,
            "source_ids": _LIST_METADATA_DELIM.join(source_ids),
            "source_count": len(source_ids),
            "kept_tags": _LIST_METADATA_DELIM.join(kept_tags),
            "status_counts": _LIST_METADATA_DELIM.join(
                f"{s}={sum(1 for g in per_goal if g.status == s)}"
                for s in sorted(VALID_STATUSES)
            ),
            # Phase I.3 — snapshot the post-update stagnation_days
            # per commitment so the review memory is self-contained
            # for diagnostics: you can read the review and see each
            # source commitment's current counter without following
            # provenance back to the commitment memory. Empty string
            # when no commitments were reviewed.
            "stagnation_snapshot": _LIST_METADATA_DELIM.join(
                f"{mid}={days}"
                for mid, days in stagnation_updates.items()
            ),
            # Phase I.5 — space-delimited log of identity-erosion
            # events that fired this bedtime review, each encoded
            # `<commitment_id>:<self_concept_key>` so tooling can
            # reconstruct what eroded from the review alone.
            "identity_erosions": _LIST_METADATA_DELIM.join(
                f"{e.commitment_id}:{e.self_concept_key}"
                for e in identity_erosions
            ),
        },
        tags=set(kept_tags),
    )

    logger.info(
        "Self-review for %s day %d → %s (%d goals, intent=%s)",
        npc_id, game_day, memory_id,
        len(per_goal), action_intent.activity if action_intent else None,
    )

    return SelfReviewResult(
        memory_id=memory_id,
        summary_text=summary_text,
        per_goal=per_goal,
        action_intent=action_intent,
        source_ids=source_ids,
        kept_tags=kept_tags,
        identity_erosions=identity_erosions,
    )
