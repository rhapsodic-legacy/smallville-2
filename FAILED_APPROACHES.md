# Failed Approaches Log

Record of what was tried and why it failed, so we don't repeat mistakes.

## Movement System Failures

### Attempt 1: Server-side sub-tile float interpolation
**What:** Server moved NPCs along paths using float positions (fractional tile progress).
Client also interpolated. Both systems animated simultaneously.
**Why it failed:** Dual movement drift — server and client positions diverged. NPCs
would "teleport" when the client snapped to the server's position on state changes.
Appeared as NPCs disappearing and reappearing far away.
**Lesson:** Only ONE system should animate movement. Server OR client, never both.

### Attempt 2: Client-side tick-locked linear interpolation
**What:** Server sent position snapshots each tick. Client linearly interpolated from
previous position to new position over the tick interval (1 second).
**Why it failed:** All NPCs received new positions in one batched WebSocket message.
Even with random stagger offsets (0–0.5s), movement looked synchronised because the
fundamental problem was server-side: all NPCs were processed and dispatched together.
**Lesson:** Client-side stagger cannot fix server-side synchronisation.

### Attempt 3: Client-side exponential lerp
**What:** `lerpSpeed * deltaTime` interpolation — NPCs raced to target in ~0.3s then
sat still for ~0.7s until next tick.
**Why it failed:** Created jerky burst-then-freeze pattern. All NPCs completed their
movement at the same time (fast lerp), then all sat still together.
**Lesson:** Exponential lerp is wrong for discrete position updates.

### Attempt 4: Path smoothing via line-of-sight string-pulling
**What:** Post-processed A* paths by removing intermediate waypoints when endpoints
had clear line of sight. Used Bresenham's algorithm for LOS checking.
**Why it failed:** Smoothed paths created diagonal shortcuts that clipped building
corners. NPCs walked through walls during float interpolation between non-adjacent
waypoints. Even with "conservative" adjacent-tile checks, building intrusions occurred.
**Lesson:** Path smoothing on a grid is dangerous unless the collision model is robust.
Raw A* on passable tiles is safe. Smoothing requires thick-line raycasting, not thin-line.

### Attempt 5: Stagger delay of 0–3 game minutes
**What:** When a schedule slot changed, each NPC got a random 0–3 game minute delay
before departing. Intended to stagger movement.
**Why it failed:** 3 game minutes = ~2.5 real seconds at default time scale. With 10
NPCs uniformly distributed, most departed within 1–2 ticks of each other. Visually
indistinguishable from simultaneous departure.
**Lesson:** Stagger delays must be measured in real seconds observed by the player,
not game time which compresses differently. Need 10–15+ real seconds of spread.

## NPC Intelligence Failures

### Problem: NPCs have no connection between identity and environment
**What:** NPCs have names, occupations, and backstories, but their moment-to-moment
behaviour is "go to building, sit there." A carpenter goes to the blacksmith shop and
idles. There's no concept of WHAT work means — no tasks, no micro-activities, no
environmental interaction.
**Why it's broken:** The schedule system generates slots like "working" but never
decomposes them into actual activities. Stanford Smallville uses a 3-level hierarchy:
daily plan → hourly blocks → 5-minute task decomposition. Our system only has level 1.
**Root cause:** Missing task decomposition layer. NPCs need "what am I doing RIGHT NOW"
not just "where should I be."

### Problem: NPCs congregate in one spot
**Why:** Multiple NPCs share the same destination (e.g., all workers go to the same
building door). Even with `find_rest_tile` spreading, they cluster within a few tiles.
Over time, schedule transitions funnel everyone to the same few locations.
**Stanford's approach:** Each NPC targets a specific OBJECT within a building (desk, easel,
anvil, bed), not just the building door. The spatial memory tree maps building → room →
object, giving each NPC a unique destination.

### Problem: NPCs are idle 90% of the time
**Why:** NPCs walk to a destination (10% of time), then sit in IDLE state until the next
schedule slot (90% of time). No intermediate activities, no task progression, no visible
"doing something."
**Stanford's approach:** Activities have durations (5–60 minutes). When one completes, the
next decomposed task begins immediately. NPCs are ALWAYS doing something — the description
updates continuously. The visual emoji/description layer shows constant activity.

## Testing Failures

### Problem: Unit tests passed but visuals were broken
**Why:** Tests checked pathfinding correctness, tile passability, overlap resolution — all
passed. But the actual visual experience (synchronized movement, teleportation, building
clipping) was never tested because tests operated on the server model, not the full
client-server loop.
**Lesson:** Need end-to-end simulation tests that observe the SAME data the client sees
(positions over time, departure/arrival timing, movement patterns) and flag anomalies.
Diagnostics must be built INTO the system, not bolted on as tests.

## Conversation Memory Failures

### Attempt 6: Async iteration of `_active_conversations` in persistence loop
**What:** `_persist_finished_conversations` iterated `_active_conversations.items()`
directly while awaiting LLM calls, memory writes, and reflection inside the loop body.
**Why it failed:** The chat task (`_handle_player_chat`) runs concurrently with the
cognition loop. When the chat task inserted a new `Conversation` into
`_active_conversations` during one of the awaits, Python raised
`RuntimeError: dictionary changed size during iteration`. The exception killed the
cognition tick. Because every tick tries to run persistence, the next tick raised again,
and the whole day wedged — every NPC stayed indoors at midday on day 38 after a heavy
conversational session on day 37.
**Lesson:** Any async iteration over shared mutable state must snapshot the keys
(`for x in list(d.items())`). A one-line fix; prevent the recurrence with a regression
test that inserts a new conversation mid-loop.

### Attempt 8: Unprotected `record_conversation` in persistence loop (house-staying v2)
**What:** `_persist_finished_conversations` called `self.memory.record_conversation(...)`
without a try/except. Inner `_extract_conversation_facts` makes an LLM call that can
raise on timeout/provider error.
**Why it failed:** Any exception propagated up through `_persist_finished_conversations`,
aborting the rest of `cognition_tick` (step 6b overlap resolution, step 7 reflection,
step 8 memory event drain). Worse: `clear_finished_conversations()` (called AFTER the
persistence step) never ran, so the bad conversation stayed in `_active_conversations`
and re-crashed every subsequent tick. Live result: the moment the player chatted with
any NPC, every NPC appeared stuck in their house and formed only sparse perceptions
for the rest of the game-day — and the behaviour persisted across day flips.
Reported three times by Jesse; tests caught none of it because no test exercised the
specific scenario "finished conversation whose `record_conversation` raises".
**Lesson:** Every async call inside a persistence-loop body needs try/except with
logger.exception. And every per-item body must guarantee a persistence-flag update
(success OR swallowed failure) BEFORE the loop moves on, otherwise a bad item
crash-loops forever. Fix (2026-04-22): added `Conversation.persisted` flag, flipped
to True before `record_conversation` runs, inner try/except around the call, bad
conv is swept by the cleanup pass same as a successful one. Regression tests:
`tests/simulation/test_chat_does_not_freeze_npcs.py`.

### Attempt 9: Eager town-goal contribution at schedule injection
**What:** `_inject_goal_entry(npc, day)` called `town_agenda.record_contribution(...)`
at injection time — before the NPC had moved, let alone completed the activity.
**Why it failed:** Goals with `required_contributions=N` plus N personality-matching
NPCs completed the same tick they were proposed. Dara's memory log showed both the
"Prepare the harvest festival" (day 78) and "Repair the old bridge" (day 83) agendas
propose AND complete at the same game-minute, with exactly the required number of
"contributors" listed who had never moved. Worse still: `TownGoal.record_contribution`
used `self.contributors.add(npc_id); self.progress += 1` — `contributors` is a set (dedups)
but `progress` is a counter (doesn't), so a double-call by the same NPC would inflate
`progress` beyond `len(contributors)`.
**Lesson:** "Eager" bookkeeping is deterministic but wrong when the book entry is
supposed to reflect physical activity. Fix (2026-04-22): moved the contribution call
to `_advance_npc_action` (fires when an entry with a `town_goal_id` finishes its
allotted duration) and made `record_contribution` dedup by npc_id.
Regression tests: `tests/simulation/test_agenda_not_insta_complete.py`.

### Attempt 7: Router-only gating for post-conversation reflection
**What:** After a conversation ended, reflection was triggered only when the cognition
router returned `Route.LLM`. For low-proximity or budget-pressured NPCs, reflection
was skipped entirely.
**Why it failed:** A player-driven accusation (high narrative weight) produced a
verbatim transcript + outcome records + sparkle — but no insight. NPCs never drew
conclusions. The router's scoring is general-purpose; it doesn't know a conversation
produced structured outcomes.
**Lesson:** Let the outcome-extraction step vote. If any Phase B outcome was
extracted (commitment, accusation, relayed claim), force the LLM reflection path
regardless of the router's default. Neutral chit-chat still defers to the router.

### Problem: No diagnostics in the simulation itself
**Why:** The server had no way to report what NPCs were actually doing, why they chose
their actions, or how their movement looked over time. Testing was blind — either look
at it manually or write post-hoc unit tests that check narrow invariants.
**Lesson:** The simulation needs a built-in diagnostic/telemetry layer that logs NPC
decisions, movement events, timing, and anomalies in a structured format. This feeds
both automated tests and human inspection.
