# NPC Vectorization — finding the foundation

> Started 2026-06-11. The problem this arc exists to solve: our NPCs read
> as **parrots incapable of organised thought** (Jesse's player
> experience) — *less* individual than a 15-year-old rules-based Sims
> game, despite far more machinery. This doc captures (a) the measured
> diagnosis, (b) how other systems actually achieve distinct, vectorised
> personalities, (c) the absolute foundation we should start from.
>
> Intended as a cold-start anchor for a fresh (stronger-model) session.
> Read alongside MEMORY_V2_ROADMAP.md (Phase J) and AGENT_DIRECTION.md.

## The measured diagnosis (Layer-1 individuality metrics, 6-day Mistral run)

`tests/simulation/npc_individuality.py` on a run dump returned a
**SYSTEMIC** verdict — three independent homogenisation sources:

1. **The self barely forms.** Mean **1.1 self-concept keys per NPC**
   (2/10 empty); reflective "self" memories are only **~3.4%** of each
   NPC's memory.
2. **What self exists is drowned.** ~**97%** of memory is
   conversation/observation; **31%** of all memories are near-duplicates.
3. **No sentiment friction.** 87 relationships, **0% negative**, mean
   +45 — nobody dislikes anybody, not even the vocal objector.

These are independent: fixing the conversation volume alone leaves the
self anemic and the town frictionless.

## How others actually vectorise personality (research, 2026-06-11)

The consistent foundation across NVIDIA ACE/Convai, Skyrim's Mantella,
the roleplay-LLM research, and Stanford's Generative Agents:

1. **A strong, structured, PERSISTENT persona conditioning EVERY call.**
   - NVIDIA ACE/Convai: a detailed character-definition system prompt
     (personality, backstory, **speech patterns**, knowledge horizon),
     with scene/state injected per turn; guardrails keep it in character.
   - Mantella: per-NPC bios + memory — and they hit *our exact bug*:
     "previously mixed up all NPC bios and memories, confusing the LLM";
     fixed by structuring each NPC's own bio/memory distinctly.
   - Research consensus: "copy the character sheet into the system prompt
     every session" — don't rely on the model to stay in character.
2. **CONCRETE SPECIFICITY beats vague description — especially SPEECH.**
   The single highest-leverage finding: *"'Vex speaks in clipped
   sentences and never uses contractions' is more useful than a paragraph
   of backstory."* Concrete linguistic/behavioural RULES create
   differentiation; vague traits ("conscientious 0.8") produce generic
   output.
3. **DEEP traits, consistently exhibited** — not surface backstory.
   The failure mode named in the literature is exactly ours: characters
   "snap into FAQ-bot mode," "break persona," "neglect deeper personality
   traits" → bland sameness.
4. **Distinct per-character memory** (RAG), kept properly separate per
   NPC. (We share one ChromaDB — possible bleed; cf. AGENT_DIRECTION.)
5. **Sustained character is the hard part** — consistency over time via
   memory + guardrails.

## Our gap (why we get parrots)

We have the *scaffolding* (Big-5 vector, self-concept, backstory, a
ChromaDB memory, a perceive/retrieve/reflect/plan loop) but the **persona
conditioning is thin and vague, and it's drowned**:
- Personality is **numbers** (Big-5), not **concrete speech/behavioural
  rules** — so the LLM defaults to generic "medieval NPC". (Gap vs #2.)
- The self-concept that *should* condition is **near-empty (1.1 keys)**
  and **not the dominant prompt signal**. (Gap vs #1; this is the parked
  MEMORY_V2 **Phase J — unified persona snapshot**.)
- The thin persona signal is **buried** under generic instructions and
  97%-volume conversation context. (Gap vs #1.)
- Personality doesn't visibly **drive** distinct behaviour/dialogue, so
  experiences (→ memories → self) come out homogeneous. (Gap vs #3.)

## The absolute foundation — start here

**A rich, concrete, persistent PERSONA that strongly conditions every
cognition call.** Not vague trait numbers — concrete, distinctive
specifics: a speech style ("clipped, no contractions, curses by the
old gods"), 2-3 behavioural rules, core values, fears, quirks, a private
agenda. Built at spawn with real specificity, refreshed by the evolving
self-concept, and injected as the **dominant** block in every prompt
(conversation, reflection, planning) — never drowned.

This is **MEMORY_V2 Phase J, re-opened and upgraded** with the research:
the persona snapshot must carry **concrete speech/behaviour specificity**,
not just aggregated Big-5 + self-concept. It is the root the other two
sources hang off: a strong persona makes reflections self-distinctive
(self forms), gives the LLM a reason to disagree (sentiment friction),
and — by being the signal that matters — lets us justify cutting the
conversation volume that drowns it.

**First concrete step (for the fresh session):** generate a concrete,
distinctive persona per NPC (speech + behaviour + values + private
agenda) and make it the dominant conditioning block in the conversation
prompt; then re-measure with `npc_individuality.py`. Expected movement:
distinct *voices* immediately (the cheap, high-leverage win), then
self-concept richness as reflections become self-relevant.

## Measurement gate

No "Stanford-worthy" claim until `npc_individuality.py` numbers move:
signal-ratio up (from 3%), self-keys up (from 1.1), sentiment
differentiating (from 0% negative), near-dup churn down (from 31%),
voice similarity down (from 0.33 — section 6, added 2026-06-11). The
30-day emergence run is gated behind meaningful movement — a 30-day run
on homogenised NPCs is 250k memories of mush.

## Tooling already in place
- `tests/simulation/npc_individuality.py` — Layer-1 individuality metrics.
- `tests/simulation/run_memory.py` — dump + synopsis (bug/outlier flags).
- `tests/simulation/npc_metrics.py` — live activity/needs/life-balance.
- `tests/simulation/diagnostic_bridge_objector.py --dump` — run + harvest.

## Status
Research + foundation captured 2026-06-11. **Foundation implemented the
same day** (branch `npc-persona-foundation`):

- `core/npc/persona.py` — `Persona` (concrete speech style + verbal tic
  + temperament + 2 behaviour rules + value + fear + quirk + private
  agenda) and `PersonaForge` (seeded, deal-without-replacement: a town
  ≤ bank size shares no speech style or temperament; deliberately NOT
  drawn from the manager's RNG so existing spawn sequences are
  unperturbed). Personas serialise via `to_full_dict`.
- `persona_system_prompt()` — the per-NPC character sheet is now the
  SYSTEM prompt (the strongest conditioning slot, which previously
  carried the town-wide "You are a medieval NPC" string) on every
  NPC-voiced call: conversation initiate/respond, all three reflection
  calls (the self-formation loop was previously completely
  unconditioned), daily plan, replan, reaction, day/week summary,
  bedtime self-review. The evolving `self_concept_summary()` rides
  inside the sheet as the dynamic overlay (= Phase J.1/J.2, reshaped).
  Clerk calls (fact extraction, note extraction, action classification)
  intentionally stay unconditioned.
- Evals: `tests/unit/test_persona.py` (determinism, distinctiveness,
  round-trip, fallbacks), `tests/unit/test_persona_conditioning.py`
  (every call site audited against MockProvider.call_log), and
  `tests/simulation/eval_persona_conditioning.py` (whole-sim traffic
  audit with CRITERIA VERDICT — catches future call sites shipping
  unconditioned). At 8 days / pop 10: 2868/2868 NPC-voiced calls
  conditioned; persona ≈ 34% of conversation prompt chars (from
  ~5-10%). Unit suite 1370 green; `eval_foundation.py` and the
  movement pipeline unregressed.

**Measurement (2026-06-11, 6-day Mistral, baseline config, seed 42,
pop 10 → `runs/persona_foundation.json`):**

The original Layer-1 suite couldn't see the predicted first effect —
voice — so a section 6 (utterance-level voice distinctiveness:
token-trigram profiles over each NPC's OWN dialogue lines, pairwise
cosine) was added to `npc_individuality.py` and run on BOTH dumps.
Same instrument-first lesson as the foundation rebuild.

| metric | baseline | persona run |
|---|---|---|
| voice similarity (1.0 = one voice) | **0.33** | **0.09** (−73%) |
| near-duplicate churn | 31% | 27% |
| signal ratio | 3.4% | 3.9% |
| self-concept keys / NPC | 1.1 | 1.0 (flat) |
| negative sentiment | 0% | 0% |

Baseline top "signatures" were shared stage business ("brow with a",
"with a warm smile" — multiple NPCs' top trigrams identical). In the
persona run every NPC's emergent signature IS their forged tic:
Jasper "mark my words", Kira "saints preserve us", Voss "not lie to
you", Mira "as my mother used", Dorian "do you follow me", Helena
"wednesday it was". Voice held for all 6 days — the sustained-
character problem the research warns about did not appear.

The run's pre-registered bridge-objector verdict (first under
non-deterministic cognition on the rebuilt foundation):
**EMERGENCE-RICH** — C1 voiced dissent PASS (6 in-character
opposition lines), C3 social consequence PASS (objector sentiment
drift −5.9 relative to town), C4 organic belief formation PASS (3
NPCs formed `built:bridge`). C2 indecision OUT-OF-BAND (join rate
0.50 vs 0.05–0.30 band — only 2 cycles, small n; watch the sticky-
participation interaction). Meta-verdict: AGENT_DIRECTION rebuild NOT
indicated by this run.

**Remaining homogenisation sources, now attributable to specific
mechanisms (not to "the parrot problem"):**
1. *Self barely forms* (1.0 keys, flat) — persona conditioning can't
   move this directly: `self_concept` is only written by dialogue-
   claim extraction and Phase I.4/I.5 reinforcement/erosion. Next
   lever: the self-concept WRITE path (e.g. persona-relevant
   reflections feeding deltas), not more prompt conditioning.
2. *Zero negative sentiment* — relative differentiation now exists
   (C3) but absolute negativity never appears. Next lever:
   `SentimentTracker` update rules (positive-bias audit), per the
   standing watch item.
3. *Volume drowning* (3.9% signal) — untouched by design; the
   conversation-volume policy is its own decision.

**Gate status:** the 30-day emergence run STAYS GATED until at least
one of the above moves; voice alone is necessary but not sufficient.
Layer-2 (in-sim attribution instrumentation) remains deferred.

## Emergent write-paths arc (2026-06-12, branch `emergent-write-paths`)

Sources 1 and 2 above turned out to share one shape: **content-blind
heuristics sat between the LLM and durable state, discarding the
signal the persona now generates.** Sentiment was written ONLY by a
talking-is-bonding baseline (+2 trust/+1 affection/+1 respect per
conversation regardless of content — Jasper's six dissent speeches
could never register); self_concept was written ONLY by regexes over
other people's words (reflections never touched it).

Shipped (the fix is widened pipes, not scripted outcomes):
- **Tone pipe:** the post-conversation reflection (persona-
  conditioned, zero new LLM calls) emits a `TONE:
  warm|neutral|tense|hostile` verdict, parsed strictly and applied
  ONE-directionally via `CONVERSATION_TONE_DELTAS`
  (core/relationships/sentiment.py, data-driven) — asymmetric
  relationships by design.
- **Accusation pipe:** already-extracted accusations now apply
  trust/respect penalties between participants
  (`ACCUSATION_SENTIMENT_DELTAS`).
- **Baseline shrink:** mere-contact deltas cut ~4×, respect removed
  (earned via tone, never free); a personality clash can now net
  negative.
- **Self pipe:** the same reflection may emit `SELF:
  <prefix>:<target>` when the insight asserts identity; validated
  against an allow-list (hallucinated keys can never reach
  self_concept) and routed through the existing contradiction-damped
  `_apply_identity_claim` (+0.10/reflection — conviction comes from
  repetition).
- **Hardening:** Mistral 429 backoff-retry (a dropped reflection
  starves exactly the new signal); external MemoryManagers get the
  sentiment tracker attached; reflection max_tokens raised so
  truncation can't eat the trailer lines.

Evals: `tests/unit/test_write_paths.py` (22 — parser failure modes,
one-directionality, clash-vs-baseline arithmetic, accusation wiring,
end-to-end through `_persist_finished_conversations`). Suite 1392
green; foundation/persona-conditioning/movement gates pass.

**Measurement (2026-06-13, 6-day Mistral, baseline config →
`runs/write_paths.json`):**

| metric | persona run | write-paths run | read |
|---|---|---|---|
| self-concept keys / NPC | 1.0 (2/10 empty) | **9.5 (0/10 empty)** | Arc B: strong win |
| cross-NPC self overlap | 0.14 | **0.03** | selves more distinct, not just bigger |
| sentiment mean | +41.5 | **+14.5** | runaway warmth broken |
| sentiment min disposition | +6.2 | **−1.7** | first sub-zero relationships ever |
| cool/neutral band (−5..+5) | 0% | **26%** | friction registers… |
| negative (<−5) | 0% | **0%** | …as withheld warmth, not dislike |
| voice similarity | 0.09 | **0.07** | regression guard held |
| distinct long-term goals | 12 | **23** | richer selves → more varied goals |
| homogenisation verdict | SYSTEMIC (4) | **MULTI-FACTOR (2)** | two sources resolved |

Bridge-objector pre-registered verdict: **EMERGENCE-RICH** again —
C4 organic belief formation jumped from 3 NPCs to **7** (several with
multiple identity keys), C3 social consequence PASS (objector −3.4
relative). C1 PASS (3 dissent lines), C2 still OUT-OF-BAND (join rate
0.50, n=2 cycles).

**Arc B (self-formation): unambiguous win.** 1.0→9.5 keys, empty
selves gone, cross-NPC overlap *dropped* to 0.03 — fuller AND more
individuated. Routing reflection-asserted identity through the
contradiction-damped applier worked. *Watch-item:* NPCs accrue
near-synonymous keys (`role:bridge`/`role:bridgekeeper`/
`role:bridge_warden`), so 9.5 is partly within-NPC redundancy; key
canonicalisation is the obvious refinement (cross-NPC distinctness is
genuine, so it doesn't undermine the result).

**Arc A (sentiment): real but partial.** Pre-registered criterion
(negative % > 0 AND stdev up) was NOT literally met — negative stayed
0%, stdev fell (distribution compressed toward zero rather than
spreading). But the *intent* — break uniform warmth — substantially
landed: mean halved, a 26% cool band opened from nothing, min went
sub-zero, and the metric stopped flagging UNIFORM SENTIMENT.
Diagnosis (from `runs/write_paths.json`): **0/90 relationships hold
even one negative dimension.** Mechanism — positives apply on *every*
conversation (baseline + frequent warm tone) and dwarf the rare
bounded negatives; mere-contact bonding paints back over friction.
Decay (~2%/day) is negligible over 6 days, so the eroder is volume,
not time. → **Arc-A tuning pass** (branch `arc-a-negative-sentiment`):
larger hostile/tense magnitudes + fear; stop mere-contact bonding
from rebuilding a dimension that's currently negative; mild
asymmetric decay for the 30-day horizon.

**Gate status:** Arc B resolved source 1; Arc A partially resolved
source 2 (uniform warmth broken, genuine dislike not yet). 30-day run
still gated — pending the Arc-A tuning producing true negative
sentiment. Source 3 (volume drowning, 5%) untouched by design.
