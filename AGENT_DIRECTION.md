# Agent Direction — IoA-derived architectural philosophy

> Captured 2026-05-02 with Jesse, sparked by Vijoy Pandey's "Internet
> of Agents" framing. This is not a roadmap and not a plan to start
> building. It is the architectural direction Smallville should drift
> toward as evidence from sims reveals where the current
> mechanism-heavy approach falls short. Read after `PROJECT_ROADMAP.md`
> and `MEMORY_V2_ROADMAP.md`. Cross-referenced from `CLAUDE.md`.

## Motivation — Vijoy Pandey via Jesse

Vijoy Pandey (Cisco / Outshift, public face of the AGNTCY collective's
Internet of Agents work) has framed today's multi-agent systems as
"geniuses who text each other." Each agent is individually capable,
they communicate via messages, and the *sum is weaker than the parts*.
The IoA pitch is that agents need shared substrate — communal memory,
alignment, intent — so the emergent organism becomes greater than the
sum of its parts.

Jesse's adaptation for Smallville: we don't want *full* communal
memory, because that voids the point of NPCs talking to each other —
why would Jasper tell Dara anything if she already knows what he
thinks? But there *should* be a communal substrate for **larger world
accomplishments**: the shared ground truth a town inhabitant would
naturally have. The bridge IS repaired. The festival DID happen. The
king IS dead. Everything else — opinions, beliefs, suspicions, plans —
stays private and propagates only through messages.

## Diagnosis — what the current architecture gets wrong

Smallville NPCs today have agent-shaped *data* (identity, goals,
self-concept, episodic memory, personality vector) but not agent-shaped
*architecture*. They are functions in a tick loop sharing global state
through a god-view manager. `NPCManager` can read every NPC's memory,
inspect any sentiment, force any conversation, inject schedule slots.
"NPC A talks to NPC B" is a Python method call mediated by a third
party with full access to both internals.

When you re-read MEMORY_V2_ROADMAP.md with that lens, **Phases K, H,
I, and J are mostly compensations for missing agent properties**:

- **K (tag-based retention)** simulates per-NPC information scoping
  that would fall out for free if each NPC owned its own memory store
  and didn't share one ChromaDB.
- **H (hierarchical compaction)** manages the shared episodic store's
  growth that wouldn't be a problem if each NPC's memory were
  privately bounded.
- **I (progress review + identity erosion)** scripts an autonomous
  "what am I doing, did it work?" loop that an actual agent would
  just *do* because that's what an agent IS.
- **J (persona snapshot)** assembles a coherent self-conditioning
  signal that would just BE the agent's own context window if the NPC
  ran as a service.

The Seren/Bran "Traveller tells lies about each one to the other"
example in MEMORY_V2_ROADMAP.md is the cleanest illustration. Today
we build specific machinery (`relayed_claim` memories, accusation
tags, `retrieve_by_tags`, tag-derivation rules) to simulate
information asymmetry that would *just exist* if each NPC only knew
what arrived in their inbox.

We are hand-coding scaffolding to fake properties that the right
architecture would produce for free.

## The three-layer model

### Communal layer — formal world ground truth
All NPCs share the same view. Bounded to outcomes and formal facts;
no opinions, no beliefs, no private intentions.

Belongs here:
- `TownAgenda` (active goals, contributors, status)
- Completed town goals and the events of completion
- Population facts (who lives where, occupations, who is married/
  related to whom)
- Public infrastructure state (the bridge IS / IS NOT repaired; the
  festival IS scheduled)
- Formal council decrees, public declarations, signed contracts

Does NOT belong here:
- "Jasper opposes the bridge repair" — that's an opinion until he
  declares it formally.
- "Dara distrusts Bran" — that's a belief.
- "The objector and three others quietly conspired" — emergent group
  state, not formal.

### Private experiential layer — per-NPC memory and belief
Each NPC's complete inner life. Readable only by that NPC's own
prompt-assembly path and that NPC's own decision logic. No global
table any code can query.

Belongs here:
- The NPC's `self_concept` (already correctly private in shape).
- The NPC's `episodic` memory of their own experiences.
- The NPC's beliefs about other NPCs (sentiment, suspicions,
  predictions). **This is the single biggest current violation —
  see "First experiment" below.**
- The NPC's plans, schedule preferences, secret intentions.

### Message-passing layer — the only legitimate way private becomes other-private
The exclusive channel through which one NPC's private state can enter
another NPC's private state. Conversations are messages. Observed
actions are messages (perception). Public declarations are messages
that *also* update the communal layer.

The architectural rule: there is no other path. Sentiment cannot leak
A → B by virtue of being in a shared table. Beliefs cannot leak by a
manager iterating both NPCs' internals.

## Mapping to current Smallville

What's already close:

- `TownAgenda` is functionally a communal layer. NPCs read its public
  state; contributors are tracked at the goal level, not in private
  belief.
- Episodic memory is per-NPC scoped at the data-model level — the
  index keys on `npc_id` — even though it lives in one ChromaDB.
  Tightening this to genuine privacy is mechanical.
- `self_concept` is already a per-NPC dict on `NPC.self_concept`.
- Conversations are already discrete events between two NPCs.
- The participation_score gate added 2026-04-24 is a tiny but
  philosophically aligned step: the NPC decides for itself whether to
  participate, rather than being forced by the manager.

What's the biggest offender:

- **`SentimentTracker` is global.** A's view of B lives in a SQLite
  table any code can query. Today
  `core/relationships/sentiment.py:get(npc_from, npc_to)` is callable
  from anywhere. Faction code, prompt assembly, conversation
  initiation, event impact rules — all reach into it freely. That's
  the textbook "fact exists about A's belief that A has not told
  anyone" failure pattern.

What also leaks:

- `NPCManager._inject_goal_entry` rewrites an NPC's daily schedule
  from outside, against the NPC's will. Today's participation_score
  gate gives the NPC a probabilistic veto, but the underlying pattern
  (manager mutates NPC schedule) is god-view.
- `record_town_event_memory` auto-injects memories into every NPC's
  store on town events, bypassing perception. Realistically these
  should arrive via a perception channel that an NPC could in
  principle miss (asleep, away, distracted).
- `seed_population_memories` writes seed beliefs and relationship
  facts directly to every NPC's store without any of them having
  experienced the events. Acceptable as bootstrap; just name it as
  a bootstrap rather than a runtime path.

## First experiment — privatise sentiment

The single highest-leverage test of this philosophy in Smallville is
to move `Sentiment(A → B)` out of `SentimentTracker` (global,
universally readable) and into `A.beliefs_about_others` (per-NPC,
accessible only via A's prompt-assembly path and A's own actions).
B has its own separate `B.beliefs_about_others[A]` which may diverge
arbitrarily.

What this changes:
- Asymmetric beliefs become natural. A genuinely fearing B while B
  feels neutral toward A is no longer a quirk of the data model — it
  is the default.
- Lies become possible. Traveller telling A "B said X" updates A's
  private belief about B; Traveller telling B the opposite updates B's
  private belief about Traveller; nothing reconciles them. The Seren/
  Bran example becomes native rather than a custom mechanism.
- Conversations become the only way beliefs propagate. A's distrust
  of B can only spread to C via A speaking to C about B. No global
  query route exists.
- The K-phase tag machinery becomes less necessary, because the
  filtering is structural rather than retrieval-time.

What this costs:
- Real test churn. Every system that reads sentiment globally —
  `shared_matters_for_prompt`, faction membership decisions, prompt
  assembly, conversation initiation, event impact rules — needs to
  route through a per-NPC accessor. This is not a one-evening edit.
- Some currently-working seed and bootstrap paths need rethinking.
  `seed_relationship_facts` writes the same sentiment to both sides
  of a relationship today; under the new model it writes to each
  NPC's private store independently (which is correct, but breaks
  any code that assumes symmetry).

Why this is the right first cut: the existing data shape is *almost*
already this. Sentiment is keyed `(npc_from, npc_to)` — directional —
and the only thing wrong is the storage location and access pattern.
The other architectural changes (private memory enforcement, message-
only propagation, bootstrap clean-up) can stage in incrementally
afterward.

## Why this isn't full Internet of Agents

Pure IoA includes cross-vendor identity (DIDs / verifiable
credentials), discovery directories, federated trust negotiation,
network-layer message envelopes, and observability infrastructure.
Smallville is a single-process simulation; none of that applies
directly. We're borrowing the *philosophy* (communal substrate +
private experience + message-only propagation) and applying it
internally to one process.

The cross-vendor / cross-system aspects of IoA do become relevant if
the AI Game Studio bridge ever materialises — Smallville-as-NPC-
backbone for `claude_agent_swarm` worlds where characters may come
from other studios. That's still future scope.

## Dependency order

This direction does not start now. The order:

1. **Hardware-permitting**: run the bridge-objector diagnostic
   (`python3 tests/simulation/diagnostic_bridge_objector.py
   --days=30`) and read the daily logs. That evidence either
   reinforces the agent-architecture diagnosis (the current mechanism
   layer can't produce the wanted emergence) or contradicts it.
2. **If the diagnosis is reinforced**: privatise sentiment as the
   architectural MVP. New per-NPC `beliefs_about_others` accessor;
   migrate every reader off the global `SentimentTracker`; rerun
   bridge-objector and read the new logs.
3. **If the privatised-sentiment sim shows richer emergence**: stage
   the rest — perception-mediated event memory, message-only
   propagation enforcement, persona snapshot rebuilt natively as the
   NPC's own context.
4. **Phase J** of MEMORY_V2_ROADMAP.md probably never reopens in its
   original shape — it was a compensation for a problem that goes
   away if the architecture changes. Re-evaluate when we get there.

## Open questions

- **What is "common knowledge"?** A town goal completes — does every
  NPC instantly know, or do they learn through perception/
  conversation? Probably tiered: some categories are communal-instant
  (the bridge IS repaired, anyone walking past sees it), others
  require message-mediated discovery (who specifically helped, what
  their motives were).
- **How does formal opinion become communal?** A council vote is
  formal; word-of-mouth is not. Where's the line? Probably best
  expressed as: opinions enter the communal layer only via explicit
  formal mechanisms (declarations, votes, signed contracts) — never
  via "enough NPCs ended up sharing the belief".
- **Bootstrap vs runtime.** Seed relationships at startup are
  inherently god-view. That's fine if they're named as bootstrap and
  no runtime code path mimics the pattern.
- **Asymmetric memory of the same conversation.** When A and B talk,
  each writes their own private memory of the exchange. They will
  diverge — different details remembered, different interpretations.
  That's the feature, not a bug. But it has costs (no shared "did
  this conversation actually happen?" oracle for debugging).
- **Performance.** Privatising sentiment turns one shared SQLite
  table into N per-NPC tables (or one table queried only via
  per-NPC accessors). Index design matters at 60+ NPCs.

## What this document is for

A landing page for the architectural direction so a future session
can pick it up cold without reconstructing the conversation. When
hardware allows running long sims and the bridge-objector evidence
lands, this document is the next reference point.
