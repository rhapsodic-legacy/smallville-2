# Smallville 2 — State & Handoff (2026-06-17)

> Cold-start anchor. Assume the reader (a future model after a context
> wipe, or a human) knows nothing about the recent work. Read this first,
> then `PROJECT_ROADMAP.md` and `VECTORIZATION_ROADMAP.md` for depth.
>
> **One-line state:** the NPC-individuality ("vectorization") arc works
> and is validated; the memory store has been rebuilt to a simple,
> reliable text store (the old ChromaDB store is gone); the **only thing
> blocking a clean 30-day emergence run is the LLM provider** — the
> Mistral free tier can't sustain the call volume, so the next real run
> must use **local Gemma on a stronger machine**.

---

## TL;DR — where we are

1. **NPCs are no longer "parrots."** A concrete persona system + write-paths
   (conversation tone → sentiment, reflection → self-concept) turned
   undifferentiated NPCs into individuals with distinct voices, self-authored
   identities, and a web of individuated likes/dislikes. Validated by the
   Layer-1 metrics moving from **SYSTEMIC → LOCALISED** homogenisation.
2. **The memory store is fixed.** ChromaDB (vector DB + embeddings) was
   ripped out after it silently dropped a third of the town's memories at
   30-day scale and could not be diagnosed. It's replaced by a dead-simple
   in-memory text store. The latest 30-day run proved the fix: **0/10
   NPCs lost memory** (was 3/10 on ChromaDB). Unit suite runs in ~2s
   (was ~9min).
3. **The blocker is the provider, not the code.** The pipeline makes
   ~11,000 LLM calls over a 30-day / 10-NPC run. On the Mistral **free
   tier** this triggers mass rate-limiting (429s), and ~11k calls failed
   on the last run, collapsing much of the dialogue to canned fallback
   lines ("Indeed, quite so."). The decision: **move cognition to local
   Gemma** (free, no rate limit). This machine is too weak to run Gemma
   at that load, so the next run waits for a stronger machine.

---

## How to run (cold start)

```bash
# Install
python3 -m pip install -e ".[dev]"

# Fast automated checks (no LLM, deterministic) — should all pass:
python3 -m pytest tests/unit/ -q                       # ~1400 tests, ~2s
python3 tests/simulation/eval_foundation.py            # scheduling/town-goal health
python3 tests/simulation/eval_persona_conditioning.py --days 8   # persona on every call
python3 tests/simulation/test_npc_movement.py          # movement/pathfinding

# The emergence sim (the long run). On a STRONG machine, use Gemma:
#   - start Ollama first: brew services start ollama   (model: see GemmaProvider.NPC_MODEL)
#   - check no straggler procs steal throughput: ps aux | grep -E 'server/main|diagnostic'
caffeinate -i python3 -u tests/simulation/diagnostic_bridge_objector.py \
    --provider gemma --days=30 --dump runs/<name>.json > /tmp/run.log 2>&1 &
# Watch: tail -f /tmp/run.log   (look for [hb] heartbeats and [day N] SNAPSHOT lines)

# Read / compare a finished run (Stage 1.5 — the human-readable digest):
python3 tests/simulation/run_memory.py summary runs/<name>.json
#   -> writes runs/<name>_summary.txt

# Quantitative individuality metrics on a run dump:
python3 tests/simulation/npc_individuality.py runs/<name>.json
```

Mistral (`--provider mistral`) still works and is fast, **but only for
short runs (≤ ~6 sim-days) on the free tier** before rate-limits bite.
`--provider mock` is deterministic and LLM-free — use it to smoke-test
the harness/plumbing in seconds.

---

## The arc so far (what was built, and why)

The work was driven by a measured problem: NPCs read as "parrots incapable
of organised thought" — less individual than a 15-year-old rules-based Sims
game. `tests/simulation/npc_individuality.py` quantified it (a 6-day dump)
as **SYSTEMIC** homogenisation from three independent sources: the self
barely formed (~1.1 self-concept keys/NPC), what self existed was drowned
in 97%-volume conversation memory, and there was zero sentiment friction
(0% negative relationships).

The fixes, each validated by re-measuring before moving on:

1. **Persona foundation** (`core/npc/persona.py`, merged).
   Concrete per-NPC character sheets — speech style, verbal tic, temperament,
   behaviour rules, value, fear, quirk, private agenda — forged at spawn by
   a seeded deal-without-replacement sampler, rendered as the **system
   prompt** of every NPC-voiced LLM call (conversation, reflection, planning,
   day/week summary, self-review). Replaced the old shared
   "You are a medieval NPC" string. **Result: voice similarity 0.33 → 0.09**
   (NPCs stopped speaking alike); every NPC's emergent dialogue carries its
   forged tic.

2. **Emergent write-paths** (`core/memory/reflection.py`,
   `core/relationships/sentiment.py`, merged).
   The persona made the LLM *generate* friction and identity, but
   content-blind heuristics discarded it. Two pipes were opened:
   - **tone → sentiment**: the post-conversation reflection emits a
     `TONE: warm|neutral|tense|hostile` verdict (+ accusation penalties),
     applied one-directionally → asymmetric relationships.
   - **reflection → self-concept**: the reflection may emit
     `SELF: <prefix>:<target>`, routed through the existing
     contradiction-damped identity applier.
   **Result: self-concept keys 1.0 → 9.5**; uniform warmth broke.

3. **Arc-A tuning** (negativity bias, merged).
   Friction still only showed as *withheld* warmth (0% genuine dislike),
   because frequent positives swamped rare negatives and mere-contact
   bonding painted over grudges. Fixes: larger hostile/tense magnitudes +
   `fear`; mere-contact bonding can't rebuild an already-negative dimension
   (only a warm-*toned* conversation heals a grudge); mild asymmetric decay
   (grudges linger). **Result: negative sentiment 0% → 24%, individuated**
   (some NPCs well-liked, some genuinely disliked, an emergent town pariah),
   not uniform collapse.

4. **Temperament rebalance + trajectory instrumentation** (merged).
   The temperament bank was abrasive-heavy (seeded that way for the
   objector experiment), driving ~64% of conversations "tense." Added 14
   concrete *warm* temperaments → bank ~50/50 → a town of 10 draws ~6 warm
   / ~4 prickly. The diagnostic now takes **daily snapshots** (sentiment
   distribution, self-keys, tone mix, emergent-pariah, bridge events) to a
   `<dump>_timeseries.json` sidecar + an end-of-run trajectory table, so we
   can see *how* a town's relationships form over time (converge / oscillate
   / drift), not just the endpoint.

5. **Memory store rewrite** (`core/memory/episodic.py`, **branch
   `simple-memory-store`, NOT merged**). See the next section.

---

## What the 30-day runs showed

Two full 30-day Mistral runs were done (seed 42, pop 10).

**Both runs:** the town converges *smoothly* to a stable, individuated
equilibrium — roughly **75-80% positive / ~10-15% negative / single-digit
neutral**, mean disposition ~+20 — reached by ~day 10-13 and held. Not a
roller-coaster, not uniform warmth, not collapse. Bridge goals complete
repeatedly despite the objector. An emergent social structure forms:
well-liked figures, a slowly-souring "curmudgeon" (Xander, both runs),
a principled objector (Jasper), and a **shared culture** — most of the town
ends up identifying as bridge-builders (`built:bridge` etc.). The
individuality verdict is down to **LOCALISED** (only the deliberately-
untouched conversation-volume source remains).

**Run 1 (old ChromaDB store):** 3/10 NPCs silently returned **0 memories**
at the end — the bug that triggered the store rewrite.

**Run 2 (new simple store, 2026-06-17, 21h):** **0/10 memory loss — fix
proven.** Every NPC held its full 5,000 memories. This reliability let us
*disprove* the harness's auto-verdict (see "measurement-tool bugs"): with
Jasper's memory now readable, he demonstrably holds `opposes:repair_bridge
=0.9` AND voices it ("those bridge stones won't outlast the next thaw"), so
the objector mechanism works.

**BUT Run 2 was heavily throttle-degraded.** ~11k Mistral calls failed
(6,167 conversation, 3,272 reflection, …) on the free tier. The Stage 1.5
summary shows the damage at a glance — the most-repeated "thoughts" are the
**canned fallback strings** ("Indeed, quite so." ×228) emitted when an LLM
call fails. So a large fraction of dialogue (especially days 26-30, which
fully stalled) was canned, not persona-driven. The sentiment *trajectory*
held (it's computed from the SQLite store, not the dropped calls), but the
dialogue/self-formation *richness* is not a clean emergence sample.

---

## The blocker, and why Gemma needs a stronger machine

The pipeline issues ~11,000 LLM calls per 30-day / 10-NPC run. The volume
is ~85% **conversation** (an LLM call per conversational turn) + **reflection**
(one per participant after every conversation), across essentially all 10
NPCs — because the headless diagnostic pins the tier "focus" at town-centre
(0,0) and never moves it, and in a compact town nearly every NPC sits inside
the Tier-1/Tier-2 radius (both of which call the LLM). Nothing throttles the
call rate down to what a free API tier allows.

**Mistral free tier** cannot sustain this → mass 429s → canned dialogue.
The retry/back-off (`core/npc/mistral_provider.py`) helps but can't beat
sustained throttling.

**Local Gemma** (via Ollama, `core/npc/gemma_provider.py`, `--provider
gemma`) is the answer: free, no rate limit, runs entirely on-device. The
catch is throughput: on the current Mac, Gemma-e2b produces ~**1 sim-day per
~30 wall-minutes** at 10 NPCs → a 30-day run is ~15h *and* saturates the
machine. This box is too weak to carry that load comfortably. **Hence the
plan: copy this folder to a stronger machine and run Gemma there.**

---

## Next steps (on the stronger machine), in order

1. **Set up local Gemma.** Install Ollama, pull the model named in
   `GemmaProvider.NPC_MODEL` (`core/npc/gemma_provider.py`). Verify it's
   reachable (`ollama_available()` / `http://localhost:11434`). The
   CLAUDE.md "Ollama Critical" note: always confirm Ollama is online before
   a Gemma run.
2. **Smoke the harness on Gemma** with a short run
   (`--provider gemma --days=2`) to confirm cognition + the daily snapshots
   work end-to-end on that machine, and gauge wall-clock/day.
3. **Run the clean 30-day emergence sim** (`--provider gemma --days=30
   --dump runs/full30_gemma.json`) under `caffeinate -i python3 -u … > log
   2>&1 &`. Watch `[hb]` heartbeats; a per-tick watchdog aborts loudly if
   the backend hangs >20min.
4. **Read it with Stage 1.5** (`run_memory.py summary …`) and the metrics
   (`npc_individuality.py …`). Compare against the Mistral runs' summaries
   in `runs/*_summary.txt`. Because Gemma won't throttle, this is the first
   *clean* emergence read — check the trajectory is real (not canned) by
   confirming the Stage 1.5 "MOST-REPEATED THOUGHTS" are NOT the canned
   fallback strings.
5. **Then iterate, one small change at a time** (the standing method):
   if the NPCs are deterministic/idiotic or emergence is thin, make ONE
   change, re-run, compare summaries. The arc backlog of candidate changes
   lives in `VECTORIZATION_ROADMAP.md` and `AGENT_DIRECTION.md`.

If even Gemma is too heavy on the target machine, the call-volume levers
(declined for now) remain available: reflect only on *notable*
conversations, cap conversation turns, or gate most NPCs to the
deterministic Tier-3 and keep a few "hero" NPCs on the LLM (this last one
aligns with the project's "95-99% on the utility layer" direction in
`PROJECT_ROADMAP.md` / `AGENT_DIRECTION.md`).

---

## Open issues / measurement-tool bugs

> **Update 2026-06-17 (post-handoff audit, PR #7):** the first two issues
> below are FIXED, plus two more found and fixed:
> - C1 is now judged by a **deterministic speaker-attributed scan** over
>   ALL dialogue (the LLM judge's false zeros were mechanical: first-15
>   newest-first sampling atop a limit-400 read + 400-char truncation).
>   Validated: finds 29 dissent lines on the 30-day dump where the old
>   judge said 0.
> - Canned fallback lines now carry a **`fallback` provenance flag** into
>   memory, are marked `[canned]` in the journal files, and the Stage 1.5
>   summary prints a **run-validity canary** ("canned/fallback lines: N").
> - **Dumps were silently capped at 5000/NPC** — the 30-day dump was
>   missing days 1–15 entirely. Now uncapped.
> - **Daily-bucket journal files** now exist: with `--dump`, every memory
>   is appended on write to `<dump>_memories/<npc_id>.txt` (day-headed,
>   human-readable) — crash-safe and mid-run tailable.

- **ChromaDB retrieval-gap root cause was never found** — but it no longer
  matters: the store that exhibited it is deleted. The investigation
  (reproductions that ruled out scale / embeddings / tombstones / deletes)
  is recorded in `VECTORIZATION_ROADMAP.md` for posterity.
- **Unswept conversation turns** (~28k in the throttled run) — likely
  conversations that died mid-throttle before consolidation. Recheck the
  turn counts in the first clean Gemma run's summary before chasing.

---

## Git / branch state

- `main`: has the persona foundation, write-paths, Arc-A tuning, temperament
  rebalance + trajectory instrumentation (PRs #2–#5 merged).
- **`simple-memory-store` (current branch, NOT merged):** the ChromaDB →
  simple-text-store rewrite (Stage 1), the Stage 1.5 run-summary tool, the
  removal of obsolete ChromaDB-investigation artifacts, and this handoff.
  All ~1,400 unit tests + the behavioural eval gates pass on it. **Decide
  whether to merge before the Gemma run** (recommended — it's the reliable
  store and the validated state).
- Run dumps + summaries live under `runs/` (gitignored; large). The
  `runs/full30_simple_store_summary.txt` is the latest (throttled) run's
  digest.

---

## Key files reference

| Area | File |
|---|---|
| Persona system | `core/npc/persona.py` |
| Episodic memory (simple text store) | `core/memory/episodic.py` |
| Tone/SELF write-paths | `core/memory/reflection.py` |
| Sentiment + tone/accusation deltas | `core/relationships/sentiment.py` |
| Tiered cognition (LLM gating) | `core/npc/cognition/tiers.py` |
| Providers | `core/npc/{mistral,gemma,mock}_provider.py`, `llm_client.py` |
| The emergence sim harness | `tests/simulation/diagnostic_bridge_objector.py` |
| Stage 1.5 run summary | `tests/simulation/run_memory.py` (`summary` subcommand) |
| Individuality metrics | `tests/simulation/npc_individuality.py` |
| Foundation / persona / movement gates | `tests/simulation/eval_foundation.py`, `eval_persona_conditioning.py`, `test_npc_movement.py` |
| Depth / rationale | `VECTORIZATION_ROADMAP.md`, `MEMORY_V2_ROADMAP.md`, `AGENT_DIRECTION.md`, `PROJECT_ROADMAP.md` |
