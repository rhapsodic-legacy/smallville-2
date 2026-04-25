# Holistic Conversation Memory — Roadmap

> Phased plan for turning conversations (player↔NPC and NPC↔NPC) into
> durable, actionable memories that propagate across the population.
> If you pick this up cold after a context reset, read this file first,
> then `PROJECT_ROADMAP.md` for wider project state.

## Goal
The town should feel cognitively coherent:
1. A conversation forms an episodic memory on every meaningful turn,
   not just on close.
2. Conversations yield structured outcomes (commitments, accusations,
   relayed claims).
3. Prior outcomes are recalled when a later conversation makes them
   relevant.
4. Outcomes propagate between NPCs naturally — if Bran learns "Petra
   says I hoard bread" from the player, and Bran later talks to Petra,
   Petra should end up holding her own memory of being accused.
5. Town-level initiatives (festivals, repairs, councils) live in
   NPCs' memories as shared experiences, not only as HUD labels.

## Current State (2026-04-19)
- **NPC↔NPC**: `core/npc/manager.py::_persist_finished_conversations`
  calls `memory.record_conversation` once `conv.finished` is set (see
  `core/npc/cognition/converse.py::end_conversation`). Full transcript
  is stored in both participants' episodic memory; optional reflection
  runs via the router's LLM path. Works — but only on close, and only
  stores the raw transcript.
- **Player↔NPC**: `conv.finished` is only set in
  `server/main.py::_close_player_chat` (on explicit close or out-of-range
  walk). Mid-chat turns are never written to memory. This is the visible
  gap Jesse hit — Bran's memory list showed only perception entries from
  other NPCs, nothing from the chat itself.
- **Outcome extraction**: none. Bran "agreed to go ask Petra" is only
  implicit in the transcript string.
- **Recall during conversation**: `plan.py` / `converse.py` pull recent
  memories by recency + similarity, but do not specifically surface
  unresolved commitments or accusations about the current partner.
- **Visual feedback**: none. The player has no way to know an NPC formed
  a memory other than opening the memory panel.

## Legend
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Phase A — Persistence parity & visual feedback
> Cheapest win. Unblocks testing of everything downstream.

- [x] A.1 `persist_conversation_turn()` + `persist_new_exchanges()`
      + `consolidate_conversation_turns()` in
      `core/memory/manager.py`. Keyword-driven importance scoring.
- [x] A.2 `_handle_player_chat` calls `_persist_new_exchanges()`
      (server wrapper around the manager helper) after each NPC reply.
- [x] A.3 `_persist_finished_conversations` sweeps per-turn entries
      by `conv_id` after writing the consolidated summary.
- [x] A.4 NPC↔NPC path in `_run_conversations` uses the same
      `persist_new_exchanges` cursor helper. Player and NPC chats now
      emit structurally identical memories.
- [x] A.5 `MemoryManager._memory_events` queue + `drain_memory_events`;
      server broadcasts `memory_events` on every tick. Importance
      threshold set to 0.6 to keep perception noise out of the feed.
- [x] A.6 `NPCRenderer.flashMemory()` spawns a short-lived icosahedron
      sparkle above the NPC that rises, spins, and fades over 1.6s.
      Gold glyph for high-importance / accusation / commitment,
      silver otherwise.
- [x] A.7 `tests/unit/test_conversation_turn_memory.py` (13 tests) +
      `tests/simulation/test_conversation_turn_memory_e2e.py`
      (idempotent cursor, consolidation round-trip).

## Phase B — Outcome & claim extraction

- [x] B.1 `core/memory/conversation_outcomes.py` with
      `ConversationOutcome` + `Commitment`/`Accusation`/`RelayedClaim`.
- [x] B.2 LLM extractor (`extract_with_llm`) with tolerant JSON
      parsing (`_parse_llm_json` handles code fences, trailing prose,
      skeletal entries).
- [x] B.3 Heuristic extractor (`extract_heuristic`) with regex
      patterns for will/promise/should commitments, explicit
      "you are a liar"/"you stole"/"you hoard" accusations, and
      "X said Y is Z" relayed claims. Self-citation filtered.
- [x] B.4+B.5 `MemoryManager.store_conversation_outcomes` persists
      per-participant with correct framing (I accused vs accused me
      vs witness), importance 0.75-0.8, unresolved flag on
      metadata so Phase C retrieval can boost open matters.
- [x] B.6 Wired into `_persist_finished_conversations`; both
      player↔NPC and NPC↔NPC paths get the same treatment via the
      merged heuristic + LLM pipeline.
- [x] B.7 `tests/unit/test_conversation_outcomes.py` (20 tests) +
      `tests/simulation/test_conversation_outcome_e2e.py` (the
      bread-hoarding scenario end-to-end).

## Phase C — Topical recall at conversation start

- [x] C.1 `MemoryManager.retrieve_unresolved_matters(npc_id,
      partner_id, partner_name, limit)` filters outcome memories by
      partner via `accused` / `accuser` / `cited_source` / `subject`
      metadata fields plus a description-text fallback for
      commitments. Sorted by importance then recency.
- [x] C.2 `format_unresolved_matters(matters, partner_name)` renders
      "Matters you want to raise with X: …" — threaded into
      `generate_daily_schedule`'s conversation counterparts
      (`initiate_conversation`, `continue_conversation`) AND the
      player chat path in `server/main.py`. Prompt templates pick up
      the new `{unresolved_matters}` slot via
      `_MissingEmpty` tolerance.
- [x] C.3 `resolve_matters_from_transcript` flips `unresolved` to
      `False` when a distinctive claim token appears in the aired
      transcript; partner-name-only greetings do NOT resolve.
      `EpisodicStore.update_metadata` patches the original record.
- [x] C.4 `tests/unit/test_unresolved_matters.py` (14 tests) +
      `tests/simulation/test_unresolved_matters_e2e.py` (3 tests:
      surface, resolve-on-discussion, ignore incidental greeting).

## Phase D — Cross-NPC propagation

- [x] D.1 `tests/simulation/test_cross_npc_propagation.py` —
      the full chain (player → Bran → Petra → player) produces the
      expected records at each step: Bran's relayed_claim on turn 1,
      resolution on turn 2, Petra holds a Traveller-naming record
      that surfaces when the player next meets her. Negative control
      confirms an incidental meeting does not propagate phantom
      accusations.
- [~] D.2 Propagated claims are already distinguishable by
      structure — a chain-forwarded memory carries a full
      `{cited_source=Traveller, relayed_by=Bran}` chain in metadata
      which the prompt injection surfaces verbatim. Explicit
      "hearsay" tagging is deferred until a scenario forces it.
- [~] D.3 Retrieval already limits to 3 matters sorted by importance
      then recency, so the prompt stays bounded. An explicit decay
      curve on `unresolved` boost is deferred until a scenario shows
      propagated memories crowding other matters.

## Phase E — Polish & HUD

- [x] E.1 Nearby-NPC HUD rows carry a category-coloured dot when
      a recent `memory_formed` event is cached for that NPC. Row's
      `title` attribute holds `category + importance + summary` so
      hover reveals the memory without a 3D raycaster. Cache lives
      on `HUD._latestMemoryByNpc` and is updated per tick.
- [x] E.2 `NPCRenderer.flashMemory` now reads the shared
      `MEMORY_CATEGORY_COLOURS` map (exported from hud.js). Red for
      accusations, gold for commitments, purple for relayed_claims,
      green for completed town events, blue for agenda items, etc.
      Size scales with importance; high-importance bumps the scale
      an extra 30% so accusations visibly pop.
- [x] E.3 `HUD.recordMemoryEvent` routes events with
      importance ≥ 0.7 to the notification feed with the category
      prefixed (`[accusation] Alice accused Bran of …`). Feed entry
      uses a new `notif-memory` style (purple accent) to distinguish
      from town events.

## Phase F — Town agenda → NPC awareness

- [x] F.1 `add_propose_listener` + `_on_goal_proposed`: every NPC
      receives an episodic memory on propose (importance 0.6,
      category `town_agenda`).
- [x] F.2 `_inject_goal_entry` writes a personal commitment memory
      (importance 0.7, category `commitment`).
- [x] F.3 `_on_goal_completed` splits contributors (first-person
      plural, importance 0.8) from bystanders (third-person,
      importance 0.5) — both as category `town_event`.
- [x] F.4 `add_expire_listener` + `_on_goal_expired`: every NPC gets
      a `town_failure` memory summarising the missed deadline.
- [x] F.5 `TownAgenda.summary_for_prompt(npc_id)` renders "Town
      matters on your mind: …" with per-NPC contributor tagging.
      Threaded through `generate_daily_schedule`,
      `initiate_conversation`, `continue_conversation`, and the
      player chat path in `server/main.py::_handle_player_chat`.
      `format_prompt` now tolerates missing keys (`_MissingEmpty`)
      so legacy callers keep working.
- [x] F.6 `tests/unit/test_town_agenda_memory.py` (12 tests covering
      listeners, summary formatting, manager seeding splits, and
      legacy-prompt-caller compatibility).

## Phase G — Agenda-driven conversation hooks

- [x] G.1 `TownAgenda.shared_matters_for_prompt(npc_id, partner_id,
      current_day)` renders a "Shared town matters: …" line for
      three shapes: both contributing, partner-only (invitation),
      recent shared victory. Empty when nothing applies.
- [x] G.2 Prompt templates now nudge the LLM to use the shared
      cue as common ground on first meetings and as glue mid-chat.
      Placeholder `{shared_agenda}` threaded through
      `conversation_initiate` and `conversation_respond`.
- [x] G.3 `TownGoal.completed_day` + `RECENT_VICTORY_DAYS = 1`
      surfaces a "recently completed ... together" line for up to
      one full game day after completion, then goes quiet.
- [x] G.4 `tests/unit/test_shared_agenda.py` (10 tests) +
      `tests/simulation/test_shared_agenda_chat.py` (4 e2e tests
      inspecting the live conversation prompt through MockProvider).
- [x] Ancillary — fixed a pre-existing circular import by pushing
      `core.npc.models` into `TYPE_CHECKING` in `town_agenda.py`;
      threaded `current_day` into `record_contribution` so the
      completion timestamp reflects the actual game day.

---

## Current Status
**Phases A + B + C + D + E + F + G:** complete (D.2/D.3 deferred
and may stay that way until real gameplay data forces them).
**Next step:** roadmap complete — the holistic conversation-memory
arc Jesse outlined on 2026-04-18 is shipped. From here it's
tuning based on live play. See the deferred notes on D.2 (hearsay
tagging) and D.3 (retrieval decay curve) for the obvious next
improvements if play surfaces prompt-crowding or
cannot-distinguish-alleged-from-proven problems.
**Last updated:** 2026-04-20

## Testing protocol
Per `CLAUDE.md`, all validation is automated. For this arc:
- Unit: `pytest tests/unit/test_conversation_memory.py
  tests/unit/test_conversation_outcomes.py -v`
- Simulation: `pytest tests/simulation/test_player_chat_e2e.py
  tests/simulation/test_cross_npc_propagation.py -v`
  (the propagation test is created in Phase D).
- After every phase: run `pytest tests/unit/ -v` and
  `pytest tests/simulation/test_identity_claim_persistence.py
  tests/simulation/test_multiday_invariants.py
  tests/simulation/test_npc_dispatch_latency.py
  tests/simulation/test_self_concept_goal_divergence.py -v` to confirm
  no regression in the self-concept / dispatch / multi-day arcs.
