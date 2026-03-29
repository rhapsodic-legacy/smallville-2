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

### Problem: No diagnostics in the simulation itself
**Why:** The server had no way to report what NPCs were actually doing, why they chose
their actions, or how their movement looked over time. Testing was blind — either look
at it manually or write post-hoc unit tests that check narrow invariants.
**Lesson:** The simulation needs a built-in diagnostic/telemetry layer that logs NPC
decisions, movement events, timing, and anomalies in a structured format. This feeds
both automated tests and human inspection.
