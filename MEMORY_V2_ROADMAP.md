# Memory v2 — Compaction, Tags, Objectives, Persona

> The next-generation memory arc. Design discussions on 2026-04-20
> with Jesse: the current system records too much detail as raw
> transcripts; what NPCs need is compressed meaning that evolves over
> time, with surgically preserved specific details when the current
> objective or personality says those details matter. If you pick
> this up cold after a context reset, read this AFTER
> `MEMORY_ROADMAP.md` and `PROJECT_ROADMAP.md`.

## Goal

NPCs should remember the way you (Claude Code) remember: a short,
meaningful working set plus surgical specific notes plus layered
long-term compression. Today they get a flat firehose of transcripts
that ages out the *most important* bits first.

The four pieces:

1. **Tag-based specific retention (K)** — memories that name an
   active agenda item, or carry a Phase B outcome, or are explicitly
   marked "remember this" bypass compaction. NPC thinking first
   probes relevant tag buckets; no hit → fall through to general
   retrieval + persona.
2. **Hierarchical compaction (H)** — periodic LLM summaries collapse
   untagged day-to-day noise into day summaries → week summaries →
   month/biography. Raw originals tombstoned, not deleted.
3. **Progress-aware objectives (I)** — every NPC runs a short daily
   self-review on bedtime: "what was I trying to do, did it move,
   what next?" Frustration grows when things stall; completion
   reinforces self-concept.
4. **Unified persona snapshot (J)** — one cached object per NPC
   aggregating Big-5 + self_concept + goal_affinities + sentiment +
   dominant tags. Becomes the single conditioning signal for every
   prompt AND every retrieval ranker, so the NPC reads as one mind
   instead of six overlapping layers.

## Current state (2026-04-20, post-Phase-G)

- **Episodic memory** is flat: every exchange, observation, reflection,
  town event, commitment, accusation, and relayed claim lives as a
  sibling entry in `EpisodicStore`.
- **Retrieval** ranks by recency + importance + relevance. No
  category preference, no tag index, no persona alignment.
- **Compaction** doesn't exist. Turn-memories are swept on conversation
  close (Phase A.3) but no long-horizon compression.
- **Progress review** doesn't exist. Commitments are recorded as
  `commitment` memories (Phase B/F) but nothing loops back to ask
  "did I do what I said I'd do?"
- **Persona pieces are scattered**. Prompts assemble the persona
  from six independent threads each turn. Retrieval ignores persona
  entirely.

## The Seren/Bran example (why this matters)

Seren is told by Traveller that Bran plans to steal her bread. Bran,
separately, is told by Traveller that Seren plans to steal his. Neither
confronts the other directly; they just quietly cool on Traveller and
each other. Weeks later they refuse to help with the harvest festival.

Under today's system, by the time that refusal surfaces, the original
cause is buried in a pile of turn memories aged out of retrieval. Under
v2: the "bread" relayed_claim is tagged — because bread-related
accusations keep recurring and some town agenda item ("figure out
where all the missing bread is going") may eventually form. When Seren
hesitates on the festival, the planner's retrieval probes her
"bread" and "traveller:trust" tag buckets and surfaces the original
event. She can actually explain, in dialogue, why she doesn't want to
help — because she still has the memory.

The bar: Seren should be able to answer "why don't you trust
Traveller?" a month after the fact, without us archiving every raw
transcript forever.

## Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Phase K — Tag-based specific retention

- [x] K.1 `EpisodicMemory.tags: set[str]` + `add_memory(tags=...)` +
      per-NPC `_tag_index: dict[npc_id, dict[tag, set[memory_id]]]`
      populated on every add. `normalise_tag` / `normalise_tags`
      canonicalise raw strings to `[a-z0-9_:-]+`. Tags serialise
      into ChromaDB metadata as space-delimited.
- [x] K.2 `MemoryManager.tags_for_commitment / _accusation /
      _relayed_claim / _town_event` derive tag sets at memory-
      creation time. Phase B outcome persistence now auto-tags
      every record with subject/cited/from/accused/accuser markers
      plus topic keywords pulled from the claim text.
      `record_town_event_memory` auto-tags with `agenda:<goal_id>`.
- [x] K.3 `MemoryManager.retrieve_by_tags` + passthrough. O(t+k)
      via the in-memory index; rebuilt from ChromaDB metadata on
      `initialise`. Tag index stays in sync under
      `delete_by_metadata` and `update_metadata` via
      `_reindex_after_tag_change` / `_remove_from_tag_index`.
- [x] K.4 `MemoryManager.infer_tags_from_context(npc, partner_id,
      partner_name, active_agenda_titles, recent_text)` unions
      partner-name + agenda titles + self_concept keys + topic
      tokens. Pure set-building, no I/O.
- [x] K.5 `extract_important_note` in
      `core/memory/reflection.py` — a post-reflection LLM pass that
      asks "is there a specific fact worth remembering verbatim?"
      and returns `(note_text, tags)` or None. NPCManager persists
      accepted notes as `category="note"` with importance 0.85 and
      tags anchored to the partner. Skipped when insight is purely
      emotional (NO_NOTE reply). Timeout-bounded at 10s.
- [x] K.6 `retrieve_with_tag_boost(npc_id, query, context_tags, ...)`
      runs the normal semantic retrieval, boosts tag-hit composite
      scores by `TAG_RETRIEVAL_BOOST = 0.5`, and injects tag-only
      hits at the bottom so they can't be missed when relevant.
- [x] K.7 `tests/unit/test_memory_tags.py` (22 tests): normalisation,
      round-trip storage, per-NPC scoping, index sync under delete
      + update, tag derivation for each outcome shape, the Seren-
      by-cited-source bread scenario, `infer_tags_from_context`,
      and the tag-boost ranker.

## Phase H — Hierarchical compaction
> Depends on K. Tagged memories skip compaction; untagged flows into
> the summariser.

- [x] H.1 `MemoryManager.compact_day(npc_id, game_day)` (thin async
      delegator) + `core/memory/compaction.py`. Pulls memories in
      `[day*1440, (day+1)*1440)`, filters to untagged and
      non-preserved categories (commitment / accusation /
      relayed_claim / town_event / reflection / note / already-
      compacted are skipped), batches into one LLM call, writes a
      `day_summary` at importance 0.6 with metadata `{day,
      compacted_from, compacted_count, kept_tags}` (list fields
      space-delimited to satisfy ChromaDB's scalar-only constraint).
      Originals patched with `compacted_into: summary_id` via
      `update_metadata`.
- [x] H.2 Summariser prompt: 3-point structure (*what happened, how
      it made me feel, what shifted in how I see the people around
      me or in what I mean to do next*). Threads personality +
      self_concept through the prompt so the voice matches. Explicit
      "do not restate events verbatim" + "if the day was truly
      uneventful, say so in one short sentence" guards.
- [x] H.3 Tombstone-aware retrieval: `_is_tombstoned(meta)` helper
      plus `include_compacted: bool = False` on `retrieve`,
      `get_recent`, `retrieve_by_tags`, `get_memories_in_window`,
      and `_fallback_retrieve`. Tombstoned memories hidden by
      default; `include_compacted=True` restores them for
      diagnostics. Hierarchy extension is free — when H.4 rolls
      day_summaries, the same filter demotes them in favour of
      the week summary.
- [x] H.4 `compact_week(npc_id, week_number)` — weekly rollup that
      operates on day_summaries (category-filtered, not raw). Week
      window `[week*7, (week+1)*7)`. Produces `week_summary`
      (importance 0.65) with metadata `{week, day_start, day_end,
      compacted_from, compacted_count, kept_tags}` aggregating from
      each day_summary's own `kept_tags` plus any surviving tagged
      memory in the window. Each day_summary tombstoned to the
      week.
- [x] H.5 `EpisodicStore.get_raw_by_id` alias (semantically
      identical to `get_by_id` but documents intent) +
      `get_compacted_sources(memory_id)` that parses
      `compacted_from` and resolves originals. MemoryManager
      passthroughs (`get_raw_by_id`, `get_compacted_sources`) so
      the memory panel doesn't reach into `.episodic`. Chain
      traversal is explicit — one level at a time.
- [x] H.6 Compaction scheduling in `NPCManager.cognition_tick`
      step 0c: on day flip, `_run_daily_compaction(current_day-1)`
      routes each autonomous, non-frozen NPC through
      `router.route(npc, "compaction", ...)`. LLM verdict →
      proper summariser call; Deterministic verdict → heuristic
      fallback. Per-NPC cursors `_last_compacted_day` /
      `_last_compacted_week` make reruns free. Week rollup
      piggybacks on `day % 7 == 6`. Policy registers `"compaction"`
      as a decision type, `ROUTE_AUTO` by default, base
      importance 0.2 so only focus-area NPCs clear the threshold
      and get the LLM cost.
- [x] H.7 Tests shipped:
      - `tests/unit/test_compaction.py` — 52 tests across windowed
        fetch, is_compactable, compact_day metadata shape and
        tombstoning, kept-tags aggregation, idempotent re-run,
        fallback path, LLM path, H.2 prompt shape (three-point
        structure, personality/self_concept threading), H.3
        retrieval-filter (retrieve, get_recent, retrieve_by_tags,
        get_memories_in_window all hide tombstones by default),
        H.4 week rollup (all corresponding invariants), H.5
        provenance walking.
      - `tests/unit/test_compaction_wiring.py` — 10 tests for the
        NPCManager-level wiring: decision-type registered,
        no-op on negative day, all-NPCs-compacted, cursor guard,
        tier-4 skipped, week piggyback, router verdict gates
        LLM vs fallback.
      - `tests/simulation/test_compaction_preserves_tags.py` —
        the headline Phase K/H interaction: a tagged accusation
        memory seeded on day 0 survives the NPCManager's
        automatic day-rollover compaction over a 3-day sim, is
        still retrievable via `retrieve_by_tags` and via the
        Phase C `retrieve_unresolved_matters` path; originals are
        tombstoned; day_summary's `kept_tags` records the
        surviving bread topic.

## Phase I — Progress-aware objectives
> Depends on H (the daily summary is the natural place to compute
> progress deltas).

- [x] I.1 `daily_self_review(manager, npc_id, game_day, *, npc, llm)`
      in `core/memory/self_review.py`. Runs at bedtime per NPC
      immediately after `compact_day` completes, so the fresh
      day_summary is one of the prompt inputs. Pulls unresolved
      self-commitments (Phase B `category="commitment"` with
      `metadata.unresolved=True` — which by construction are
      always the NPC's own pledges) plus long-term goals and the
      day_summary, asks the LLM "what moved, what stalled, what's
      next?" in a structured block format. Thin async wrapper
      `MemoryManager.daily_self_review` keeps the module out of
      the already-oversized manager; new prompt template
      `self_review` in `core/npc/llm_client.py`; new decision type
      `"self_review"` in `router/policy.py` defaulting to
      ROUTE_LLM (voice-per-NPC is the feature's intent — cost
      difference vs ROUTE_AUTO is 2–3 calls per game day).
      Heuristic fallback marks every open matter `stalled` when
      no LLM is available, so the deterministic router verdict
      still produces a review.
- [x] I.2 Output shape: `commitment_review` memory at importance
      0.7, category added to `PRESERVED_CATEGORIES` so the next
      day's compaction leaves it alone. Description body is
      `SUMMARY` line + one `[status] goal — note` line per
      reviewed matter + optional `Tomorrow: …`. Metadata carries
      `day`, `source_ids` (space-delimited commitment memory ids),
      `source_count`, `kept_tags` (union of source-commitment
      tags for Phase K anchoring), and `status_counts`
      (`moving=N stalled=N abandoned=N done=N`). When the
      `NEXT:` line is actionable, existing `classify_insight`
      produces an `ActionIntent` that `NPCManager._run_daily_self_
      review` injects into tomorrow's schedule via the same
      `_inject_reflection_entry` path reflection-driven intents
      already use. Tests: `tests/unit/test_self_review.py`
      (28 tests — parser, fallback, commitment lookup, LLM happy
      path, tag inheritance, metadata shape, PRESERVED_CATEGORIES
      invariant), `tests/unit/test_self_review_wiring.py` (10
      tests — decision-type registration, tier-4 skip, cursor
      guard, DETERMINISTIC/LLM gating, ActionIntent injection),
      `tests/simulation/test_self_review_produces_review.py` (3-
      day headline invariant).
- [x] I.3 Stagnation escalation. `commitment` memories now carry
      a `stagnation_days` metadata counter that the bedtime review
      updates per verdict: `stalled` → `+=1`, `moving` / `done` →
      reset to 0, `abandoned` → frozen (so I.5 reads the terminal
      value). Counter is unbounded on the write side; I.5 needs the
      raw signal past the retrieval cap. `MemoryManager.retrieve_
      unresolved_matters` now ranks by composite score
      `importance + min(stagnation_days, CAP) * BOOST_PER_DAY`,
      with `STAGNATION_BOOST_PER_DAY = 0.04` and `STAGNATION_BOOST_
      CAP = 15`. Saturation dynamics chosen for 60+ day sims:
      - Day 5 stalled (base 0.75) → 0.95, matches a fresh critical
        accusation;
      - Day 15 saturates at +0.60 → 1.35, strong dominance but
        bounded;
      - Day 30, 60, 120 all score the same 1.35 — recency
        (game_time) breaks ties so the more-recent-but-also-stalled
        entry wins, preventing ancient baggage from permanent
        dominance.
      Positional matching between `GoalProgress` entries and source
      commitments; unmentioned commitments default to stalled (a
      goal the NPC didn't even raise at bedtime is functionally a
      stagnating one). Accusations and relayed_claims don't
      accumulate stagnation (`_stagnation_boost` returns 0 for
      non-commitment categories). Review memory gains
      `stagnation_snapshot` metadata so diagnostics can read the
      bedtime state without chasing provenance. Tests: 23 in
      `tests/unit/test_stagnation.py` (counter transitions, per-
      commitment independence, positional match with short/long
      per_goal, boost linearity + cap, non-commitment exclusion,
      malformed-value handling, end-to-end `daily_self_review`
      increments) + 2 sim tests in
      `tests/simulation/test_stagnation_escalation.py` (6-day ramp
      reorders matters; 18-day run verifies counter grows past the
      retrieval cap).
- [x] I.4 Goal completion → self_concept reinforcement. Every
      contributor to a completed `TownGoal` now receives
      `+REINFORCEMENT_DELTA` (= 0.1, symmetric with I.5's erosion
      magnitude) on the goal's `identity_key`. Keys live on
      `GoalTemplate` and are copied through to the instantiated
      `TownGoal`, keeping the reinforcement target data-driven
      alongside the rest of the template: `harvest_festival →
      helped:festival`, `repair_bridge → built:bridge`,
      `town_council → joined:council`. `built:` and `joined:`
      prefixes were added to `NPC.self_concept_summary()`'s
      phrase_map so the belief renders as "someone who built the
      bridge" in prompts. Fires synchronously inside
      `NPCManager._on_goal_completed` for each contributor;
      bystanders get the town_event news memory but no identity
      bump. Each fire writes a `reflection` memory tagged
      `{town_agenda, goal_id}` with `outcome_kind=
      identity_reinforcement` metadata pointing back to the source
      goal, and returns an `IdentityReinforcementEvent`
      (mirroring I.5's event shape). Idempotency is inherent —
      `TownAgenda` fires the completion listener exactly once per
      goal, and `goal.contributors` is a set. Tests: 12 in
      `tests/unit/test_identity_reinforcement.py` (template plumbing,
      contributor vs bystander, ceiling clamp, missing key guard,
      reflection memory shape, phrase rendering, manager
      integration) + 1 sim in `tests/simulation/
      test_identity_reinforcement.py` (4-contributor goal completion
      through the real agenda, reinforcement survives 3 days of
      subsequent ticks).
- [x] I.5 Soft identity erosion. When a commitment's
      `stagnation_days` crosses `STAGNATION_IDENTITY_THRESHOLD`
      (20) for the first time, `daily_self_review` emits a
      self_concept delta via `npc.apply_self_concept_delta`:
      - **Subject match** — if the commitment description contains
        a token ≥ 4 chars that matches the `target` part of any
        existing self_concept key (case/underscore-insensitive,
        prefers highest-confidence match), apply
        `-IDENTITY_DELTA` (= 0.1) to that key. E.g. commitment
        about "the bridge" with NPC's `helped:bridge = 0.8`
        drops to 0.7.
      - **Fallback** — no subject match → introduce or
        strengthen `unreliable:self` by `+IDENTITY_DELTA`. Three
        unmatched failures accumulate to `unreliable:self = 0.3`.
        `core/npc/models.py` phrase_map gained an entry so
        `self_concept_summary()` renders the belief as
        "someone unreliable".
      Fires exactly once per commitment via an `identity_eroded`
      metadata flag on the source `commitment` memory; later days
      past the threshold are no-ops. Each event writes a
      `reflection` memory ("I have been telling myself I would X
      for weeks now…") tagged with the source commitment's tags so
      Phase K retrieval surfaces it alongside the original.
      `SelfReviewResult` gains `identity_erosions:
      list[IdentityErosionEvent]` (commitment_id, self_concept_key,
      delta, new_confidence, reflection_memory_id); the review
      memory's metadata carries a space-delimited
      `identity_erosions` log of `<commitment_id>:<key>` pairs.
      Tests: 22 in `tests/unit/test_identity_erosion.py`
      (tokenisation, subject-match ranking, crossing detection,
      idempotency, fallback path, accumulation, reflection-memory
      provenance + tags, end-to-end 21-day run) + 2 sim tests in
      `tests/simulation/test_identity_erosion.py` (22-day
      crossing + matched-key delta; 35-day run confirms exactly
      one erosion per commitment).
- [x] I.6 Multi-day identity-arc sim. Two sim tests in
      `tests/simulation/test_identity_arc.py` braid the I.3/I.5 loop
      into an end-to-end story:
      - **Vocalness ramp → erosion**: across 23 fallback reviews, the
        bridge commitment's composite retrieval score (importance +
        I.3 stagnation boost — the proxy for "how prominently this
        matter sits in the NPC's prompt context when the relevant
        partner shows up") climbs strictly through day 14, plateaus
        at the `STAGNATION_BOOST_CAP` value of 1.35, and exactly one
        I.5 erosion event fires at the threshold crossing, dropping
        `helped:bridge` from 0.8 to 0.7. Provenance checked
        (commitment carries `identity_eroded`; reflection memory
        tagged `bridge`).
      - **Stagnated-then-abandoned**: a commitment that's already
        past-threshold and erosion-fired doesn't re-fire when the
        NPC subsequently marks it `abandoned`. Counter freezes,
        `identity_eroded` flag blocks re-fire, belief stays at 0.7
        through 4 more stalled days. Uses a narrow
        `_AbandonedVerdictProvider` LLMProvider stub to drive the
        one bedtime pass that needs an abandonment verdict.
      Completion path is covered by Phase I.4's
      `test_identity_reinforcement` sim — not duplicated here.
      Runtime: 0.09s for both tests combined.
      **Known hole (strict scope)**: a commitment that's abandoned
      *before* reaching the 20-day stagnation threshold currently
      incurs zero identity cost — the counter never hits threshold
      and I.5 never fires. Flagged as a narrative gap in the
      Tuning watchlist for future scope (separate feature phase,
      not a fix for I.6).

## Phase J — Unified persona snapshot (RE-OPENED, RESHAPED 2026-06-11)
> Polish. Ties the whole memory stack + personality + objectives into
> one coherent conditioning signal.
>
> **Parked 2026-04-24.** The audit-before-build pivot (see
> "Emergent-behaviour pivot" below) concluded that stacking another
> layer of persona plumbing on top of untested mechanics risks
> emergence-in-a-vacuum. J stays frozen until long-form sims on the
> weighted-participation gate produce evidence that personas are what
> the stack is actually missing.
>
> **Re-opened 2026-06-11** by the vectorization arc
> (VECTORIZATION_ROADMAP.md): the Layer-1 individuality metrics
> supplied the missing evidence (SYSTEMIC homogenisation; the persona
> signal thin AND drowned). Shipped in an upgraded form: instead of
> aggregating Big-5 + self_concept into a snapshot dataclass,
> `core/npc/persona.py` forges a CONCRETE character sheet at spawn
> (speech rules, temperament, behaviour rules, value, fear, quirk,
> private agenda) and `persona_system_prompt()` renders it + the live
> `self_concept_summary()` as the per-NPC SYSTEM prompt on every
> NPC-voiced cognition call. That delivers the intent of J.1/J.2 with
> concrete specificity rather than trait numbers. J.3 (persona-aware
> retrieval ranking) and J.4 (persona delta / character arc) remain
> open, gated on the post-change `npc_individuality.py` measurement.

- [ ] J.1 `NPC.persona_snapshot: PersonaSnapshot` — dataclass
      aggregating: Big-5 vector, top-k self_concept entries, active
      long_term_goals, aggregated goal_affinities, dominant sentiment
      toward 2-3 most-interacted NPCs, top-k tags in retrieval index.
      Cached on the NPC, refreshed each game day and on every
      self_concept delta.
- [ ] J.2 A `persona_snapshot_to_prompt()` method renders the
      persona as a single compact block that replaces (or augments)
      today's six independent slots.
- [ ] J.3 Retrieval ranker takes the persona as input: a memory's
      score includes a `persona_alignment` term (cosine-style overlap
      between memory tags + categories and persona tags + categories).
- [ ] J.4 Daily diff: the day_summary records `persona_delta` — the
      dimensions that changed today. Over a week, this is the NPC's
      character arc.
- [ ] J.5 Tests: a long-running NPC's persona snapshot reflects their
      dominant story (the king who built the bridge, the merchant
      burned by Traveller) even after raw memories have been
      compacted away.

## Emergent-behaviour pivot (2026-04-24)

Before starting J.1, Jesse pushed back: more mechanism without evidence
of where the current stack actually breaks under emergence is
upside-down. The concrete worry: mechanisms compound unpredictably at
scale, and a six-layer persona snapshot sitting on top of untested
weighted systems is the kind of thing that "just keeps breaking" as the
sim grows. Steering direction going forward: *very simple weighted
systems that can be iterated on, in the spirit of NVIDIA's vectorised
personalities + Stanford's Smallville — plus sims whose outcomes are
non-deterministic so we can witness real emergent behaviour.*

### What was built instead of J.1/J.2

A narrow, data-driven edge-case sim: **the bridge objector**. One NPC
carries `opposes:repair_bridge = 0.9` in their self_concept; the town's
`repair_bridge` goal stays on the docket; we read the logs to see
whether the opposition actually shapes behaviour.

Changes shipped:

- `core/world/town_agenda.py` — `matches_personality` (boolean gate)
  removed; replaced with a weighted-distribution gate following the
  image-classification confidence analogy:
    - `participation_score(npc)` — personality alignment sum + explicit
      `supports:<goal_id>` / `opposes:<goal_id>` self_concept pulls.
    - `participation_probability(npc)` — sigmoid of score.
    - `should_participate(npc, rng)` — sampled decision, *forces* the
      NPC to make a call on each eligible tick, non-deterministically.
  The objector's probability is ~0.14 — usually declines, occasionally
  begrudgingly helps. That 14% is the "human-like indecision" point.
- `core/npc/manager.py` — `_inject_goal_entry` now threads `self.rng`
  through `matching_goal_for`.
- `core/npc/models.py` — `self_concept_summary()` phrase_map gained
  `opposes:` → "opposed to {target}" and `supports:` → "a supporter of
  {target}". Belief now renders into every conversation prompt that
  runs `self_concept_summary()`, so the LLM can actually argue the
  position in dialogue without any new prompt-assembly machinery.
- `tests/unit/test_town_agenda.py` — `TestPersonalityMatching` rewritten
  as `TestParticipationScore` (8 tests: score additivity, opposes/supports
  pulls, sigmoid mapping, frequency-based sampling tolerance, the
  "objector occasionally helps" edge). `TestMatching` now threads a
  `_FixedRng(0.0|0.99)` stub so the ordering/skip invariants are
  isolated from the probabilistic eligibility.
- `tests/simulation/diagnostic_bridge_objector.py` — logging harness,
  no assertions. Uses `GemmaProvider` for real non-determinism. Per
  game-day, logs the bridge goal's status + the objector's score /
  probability / join outcome, plus any episodic memory mentioning the
  bridge. Ollama-availability gate at startup.

Full unit suite after the refactor: **1333 passing** (was 1329 pre-J
pivot — net +4 from the new `TestParticipationScore` cases).

### Smoke-test status

Day-1 run under local Gemma-e2b confirms the full wiring:

```
Objector: Jasper (farmer)
  conscientiousness = 0.80
  injected belief: opposes:repair_bridge = 0.9
  self_concept_summary: 'You see yourself as: opposed to repair bridge.'

[day 1] PROPOSED repair_bridge (cycle #1, deadline day 4)
[day 1] BRIDGE status=proposed  progress=0/4
        objector_score=-0.60 p=0.141 objector_joined=False
[day 1] OBJECTOR_MEM [town_agenda] 'The town has proposed a new
        initiative: "Repair the old bridge" — The bridge has sagged…'
```

Math checks out: 0.80 − 0.50 (conscientiousness threshold) − 0.90
(opposes) = −0.60 → sigmoid(−0.60 × 3.0) = 14.1 %. The phrase_map
renders "opposed to repair bridge" into the self_concept_summary, so
it will flow into conversation prompts whenever this NPC speaks.

**Runtime cost is the practical blocker.** 30 wall-minutes under
Gemma-e2b with no competing clients produced one simulated day. A
competing `server/main.py` process (since Wednesday, 37 min accumulated
CPU) was halving throughput — killed for the test, left off
afterwards. Without better hardware, 60-day runs are many hours.

### Open questions the diagnostic is meant to answer
Once a 30+ day run completes, read the logs for:

1. Does Jasper voice opposition in dialogue? Conversation prompts now
   carry "opposed to repair bridge"; the LLM is free to elaborate.
2. Does the bridge goal succeed around him, or fail at
   `deadline_days=3` because other NPCs' rolls fall short too often?
3. Do neighbouring NPCs' sentiment toward Jasper shift after a bridge
   cycle completes without him?
4. Does anyone pick up `supports:repair_bridge` organically from
   conversations / reflections, as a counter-pull?

These are the signals that tell us whether the weighted-gate shape is
producing real emergence, versus just randomised noise. Decision on
reviving J waits on that evidence.

---

## Cross-cutting concerns

### Cost budgeting
Compaction is the obvious spender: one LLM call per NPC per game day
for H.1, plus one per NPC per week for H.4, plus one per NPC per day
for I.1. At 10 NPCs that's ~30 extra LLM calls per game week. Offset
by shorter prompts at normal ticks (retrieval pulls 3 summaries vs 30
raw turns). Net should be lower or flat; instrument it.

### Tag noise
Jesse's worry: too many tags become noise. Mitigations:
- Tag creation is gated to Phase B outcomes + active agenda matches +
  explicit reflections. Not every observation gets tagged.
- Tag index is per-NPC, not global — one NPC's "bread" is not
  another's unless they both encountered it.
- Conditional importance from agenda: tags only surface strongly when
  they match the CURRENT agenda/persona context (K.6). A "bread" tag
  on Seren's memory is weak today when she's focused on the festival,
  but strong the day a "missing bread" agenda item gets proposed.

### Back-compat
All four phases extend `MemoryManager` with new methods; no existing
callers break. `EpisodicMemory.tags` defaults to empty, retrieval
degrades gracefully when no persona is cached, and compaction is
opt-in per NPCManager initialisation flag during rollout.

---

## Tuning watchlist

Parameters we've introduced with reasoned-but-untested initial values.
Revisit after long-form sims (60+ days, varied NPC archetypes) reveal
whether the dynamics land where we wanted.

### Identity erosion magnitude (Phase I.5)
- **Location**: `core/memory/self_review.py` — `IDENTITY_DELTA`
  (currently `0.1`).
- **Current behaviour**: flat -0.1 on the matched self_concept key
  per stagnation crossing event. Three stuck commitments take a
  belief from 0.8 to 0.5.
- **Alternative we've punted on**: proportional magnitude, e.g.
  `-0.05 * (stagnation_days - THRESHOLD + 1)` capped, so a commitment
  that stalls for 25 days erodes harder than one that stalls for 20.
  Richer dynamics but more to tune and harder to reason about under
  long sims.
- **Watch for**: NPCs either barely noticing long failures (flat
  delta too small against high-confidence beliefs) OR identity
  flipping after one abandonment (too large).
- **How to decide**: run a 60-day sim with 3+ unresolved commitments
  per NPC. Inspect final `self_concept` dicts. If strong beliefs
  (0.8+) never move meaningfully, bump the delta or adopt the
  proportional variant. If weak beliefs (0.3–0.5) collapse
  immediately, reduce or add a confidence-weighted dampener.

### Stagnation threshold (Phase I.5)
- **Location**: `STAGNATION_IDENTITY_THRESHOLD` (currently `20`).
- **Relation to I.3's cap**: 5 days past the retrieval-boost cap (15),
  giving the NPC time to raise the matter in conversation before
  identity starts sliding. If sims show NPCs vocally raising matters
  for weeks without ever crossing (because partners aren't around),
  this may need to drop — or the counter condition may need to be
  "stalled and not raised" rather than just "stalled".

### Early abandonment has no identity cost (Phase I.6, strict scope)
- **Location**: `core/memory/self_review.py` — `_apply_identity_
  erosion` gates entirely on `stagnation_days >=
  STAGNATION_IDENTITY_THRESHOLD`. Abandoned commitments freeze the
  counter but never write a separate identity signal.
- **Current behaviour**: an NPC who abandons a commitment on day 5
  (well before the 20-day threshold) pays nothing in self_concept.
  Only stagnation-threshold crossings move identity. I.6's sim
  asserts this strict interpretation.
- **Narrative hole**: "I give up on this" is arguably a stronger
  admission than "I haven't gotten around to it" yet today it
  carries less weight. A user watching an NPC cleanly abandon a
  goal on day 5 won't see any character consequence.
- **Alternatives if we revisit**: (a) dedicated `abandoned:self`
  self_concept key bumped on any `abandoned` verdict regardless of
  stagnation; (b) proportional — small delta on early abandon,
  larger if the commitment stagnated first; (c) leave strict and
  rely on LLM-driven narrative colouring in conversation prompts.
- **How to decide**: future scope, not I.6. Belongs in a Phase I.7
  (narrative-gap patch) or as a pre-req to Phase J's persona
  snapshot if we find personas feel hollow on abandonment.

### Participation temperature (bridge-objector pivot)
- **Location**: `core/world/town_agenda.py` — `PARTICIPATION_TEMPERATURE`
  (currently `3.0`).
- **Current behaviour**: sigmoid sharpness on `participation_score`. At
  3.0, score `+0.3` → p≈0.71, score `-0.6` → p≈0.14.
- **Watch for**: if the objector *never* begrudgingly helps across a
  60-day run (too sharp — drop toward 2.0), or if NPCs with clear
  personality mismatches still join town goals at ~35-40 % rates (too
  soft — push toward 5.0). Expect this to want retuning once we have
  real dialogue logs: the "14 % rare help" cadence is the intended
  feel; sims either confirm or falsify it.

### Reinforcement magnitude (Phase I.4)
- **Location**: `core/memory/self_review.py` — `REINFORCEMENT_DELTA`
  (currently `0.1`).
- **Current behaviour**: flat +0.1 on the goal template's
  `identity_key` for every contributor each time a town goal
  completes. Three completed festivals take `helped:festival` from 0.0
  to 0.3; ten take it to saturation at 1.0.
- **Why symmetric with erosion**: picked equal to `IDENTITY_DELTA`
  so the 60-day sim has one magnitude variable to reason about.
  Erosion fires at most once per commitment after 20 stalled days;
  reinforcement fires on every completion. Frequency already biases
  the loop positive — no need to stack magnitude asymmetry on top
  before we see data.
- **Alternatives we've punted on**: (a) contribution-weighted
  (`delta * own_progress / total`) so a 1-of-10 contributor moves less
  than a 1-of-3; (b) asymmetric (>0.1) if completions turn out to be
  rare enough that erosion dominates the steady state.
- **Watch for**: identity pegging at 1.0 across a population within
  a month of festivals (too large), or contributors never forming
  a distinct `built:bridge` identity because strong priors on other
  keys absorb all the retrieval attention (too small to matter).
- **How to decide**: run a 60-day sim with 5+ completed town goals
  per active NPC plus a mix of stalled commitments. If all
  contributors' `helped:*` / `built:*` keys saturate within 2-3
  weeks while the rest of their self_concept stays stable, reduce to
  0.05. If strong priors (e.g. `role:baker` at 0.9) still dominate
  prompts after every completion, consider the contribution-weighted
  variant or bump to 0.15.

---

## Current Status
**Phases K + H + I (complete):** memory now has tag-anchored
retention, day-level compaction with tombstoning, week-level
rollup, tombstone-aware retrieval, provenance walking,
auto-scheduled compaction + bedtime self-review on day rollover
routed through the cognition router, structured
`commitment_review` memories that survive subsequent compaction,
per-commitment stagnation tracking that lifts stale matters above
fresh ones in `retrieve_unresolved_matters`, one-shot soft
identity erosion when a commitment stagnates past 20 days,
one-shot identity reinforcement when a town goal the NPC
contributed to completes, and an end-to-end multi-day arc sim
tying vocalness ramp → erosion → abandonment-idempotency
together. Full unit suite: 1329 passing. Sim coverage:
`test_self_review_produces_review`, `test_stagnation_escalation`,
`test_identity_erosion`, `test_identity_reinforcement`,
`test_identity_arc`.
**Next step:** Phase J is PARKED (see "Emergent-behaviour pivot"
above). The weighted-participation gate + bridge-objector diagnostic
are in place and smoke-tested through day 1. The next concrete action
is a 30+ day local-Gemma run of `tests/simulation/
diagnostic_bridge_objector.py` on hardware that can sustain the
throughput — the daily logs answer whether Jasper's opposition shapes
the run (voiced dissent, failed/succeeded bridge cycles, sentiment
shifts) or gets absorbed as noise. That evidence decides whether J
stays parked or re-opens with a tighter scope.

**Architectural direction added 2026-05-02:** A second parallel reading
of the K/H/I/J phases emerged in conversation: each is mostly
*compensating* for missing agent properties that a different
architecture would produce for free. Captured in
`AGENT_DIRECTION.md` as the IoA-derived (Vijoy Pandey / AGNTCY)
communal-substrate + private-experience + message-only-propagation
philosophy. If the bridge-objector evidence reinforces the diagnosis,
the first experiment is privatising sentiment (move
`SentimentTracker` global table to per-NPC `beliefs_about_others`),
and Phase J in its current shape probably never reopens. Read
AGENT_DIRECTION.md for the full argument and dependency order.
**Last updated:** 2026-05-02
