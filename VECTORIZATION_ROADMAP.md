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
differentiating (from 0% negative), near-dup churn down (from 31%). The
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

**Measurement (in flight):** 6-day Mistral run matching the baseline
config (seed 42, pop 10) → `runs/persona_foundation.json` →
`npc_individuality.py` vs `runs/bridge_objector_retune.json`. Movement
expected first in voice/near-dup churn, then self-keys and signal
ratio. Layer-2 (in-sim attribution instrumentation) remains deferred.
