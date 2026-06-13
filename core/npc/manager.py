"""
NPC Manager.

Orchestrates NPC spawning, the per-tick cognition cycle, and
schedule-driven movement. Acts as the main entry point for the
server to interact with the NPC population.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from core.npc.models import (
    NPC, ActivityState, PersonalityTraits, ScheduleEntry, SubTask,
    Commitment, CommitmentStatus,
    OCCUPATION_DEFAULTS, FIRST_NAMES, BACKSTORY_TEMPLATES,
)
from core.npc.persona import PersonaForge
from core.npc.seed_memories import seed_population_memories
from core.npc.llm_client import LLMProvider, MockProvider
from core.npc.cognition.tiers import (
    update_all_tiers, should_perceive, should_plan, get_tier_config,
)
from core.npc.cognition.perceive import perceive
from core.npc.cognition.plan import (
    generate_daily_schedule, resolve_schedule_location,
    _template_schedule, should_replan, replan_schedule,
)
from core.npc.cognition.execute import (
    execute_tick, set_activity_for_location, navigate_to,
    clear_arrival_claims,
)
from core.npc.cognition.decompose import (
    decompose_schedule_entry, decompose_schedule_entry_llm,
    make_staggered_subtasks,
)
from core.npc.cognition.converse import (
    should_initiate_conversation, initiate_conversation,
    continue_conversation, end_conversation,
    get_active_conversations, clear_finished_conversations,
    set_sentiment_tracker,
)
from core.npc.cognition.router import (
    CognitionRouter, Route, CognitionPolicy,
)
from core.npc.cognition.planner import DeterministicPlanner, PlannedAction
from core.npc.cognition.goal_mapper import sync_npc_goals
from core.npc.economy_tick import EconomyTick
from core.memory.manager import MemoryManager
from core.memory.reflection import (
    run_reflection, run_reflection_with_intents,
    reflect_on_conversation, ActionIntent, classify_insight,
    extract_important_note,
    apply_personality_drift, detect_identity_claims,
    IdentityClaim,
)
from core.memory.self_review import apply_identity_reinforcement
from core.relationships.sentiment import (
    SentimentTracker, ACCUSATION_SENTIMENT_DELTAS,
)
from core.relationships.structures import FactionManager
from core.events.impact import EventImpactSystem, GameEvent
from core.world.spatial_awareness import resolve_overlaps
from core.world.grid import Grid
from core.world.generator import PlacedBuilding
from core.world.town_agenda import TownAgenda, create_goal_from_template
from core.time_system.clock import GameClock, MINUTES_PER_DAY
from core.evolution.overseer import Overseer
from core.evolution.mechanisms import MechanismEngine
from core.evolution.guardrails import GuardrailEngine

logger = logging.getLogger(__name__)


def _force_template_schedule(npc: NPC, current_day: int) -> None:
    """Assign a template schedule — no LLM, fully deterministic.

    Resets schedule_index and action_start_minutes so the new day
    starts at entry 0 regardless of where the NPC's cursor was on
    the previous schedule.
    """
    npc.daily_schedule = _template_schedule(npc)
    npc.schedule_day = current_day
    npc.schedule_index = 0
    npc.action_start_minutes = 0.0
    logger.debug(
        "%s: template schedule assigned (deterministic mode)", npc.name,
    )


class NPCManager:
    """Manages the entire NPC population. Server calls tick() each game loop."""

    def __init__(
        self,
        grid: Grid,
        buildings: list[PlacedBuilding],
        llm: LLMProvider | None = None,
        seed: int | None = None,
        memory: MemoryManager | None = None,
        sentiment: SentimentTracker | None = None,
        factions: FactionManager | None = None,
        events: EventImpactSystem | None = None,
        router: CognitionRouter | None = None,
        planner: DeterministicPlanner | None = None,
        economy: EconomyTick | None = None,
        deterministic: bool = False,
    ):
        self.grid = grid
        self.buildings = buildings
        self.llm = llm or MockProvider()
        self.deterministic = deterministic
        self._seed = seed
        self.rng = random.Random(seed)
        # Separate seeded RNG: forging personas from self.rng would
        # shift every downstream draw and invalidate eval baselines.
        self._persona_forge = PersonaForge.from_seed(seed)
        self.npcs: list[NPC] = []
        self._npc_map: dict[str, NPC] = {}

        # Cognition router + deterministic planner
        self.router = router or CognitionRouter()
        self.planner = planner or DeterministicPlanner(
            grid, buildings, seed=seed,
        )

        # Economy systems
        self.economy = economy or EconomyTick(grid)

        # Relationships and events
        self.sentiment = sentiment or SentimentTracker()
        self.sentiment.initialise()
        self.factions = factions or FactionManager()
        self.events = events or EventImpactSystem(sentiment_tracker=self.sentiment)
        self.events.initialise()

        # Memory system (inject sentiment and factions for enriched context)
        self.memory = memory or MemoryManager(
            llm=self.llm,
            sentiment=self.sentiment,
            factions=self.factions,
        )
        # Externally-supplied MemoryManagers (tests, evals) often omit
        # the tracker; without this the tone→sentiment write path
        # no-ops silently in exactly the environments that audit it.
        if self.memory.sentiment is None:
            self.memory.sentiment = self.sentiment
        self.memory.initialise()

        # Wire sentiment into conversation module
        set_sentiment_tracker(self.sentiment)

        # Evolution layer: overseer evaluates once per game-day
        self.overseer = Overseer(llm=self.llm)
        self.mechanisms = MechanismEngine()
        self.guardrails = GuardrailEngine()
        self._last_eval_day: int = -1

        # Phase H.6 — per-NPC cursor for the most recent day/week
        # compacted. The cognition tick compacts `current_day - 1`
        # once per NPC at the first tick of a new day, then rolls up
        # the week when `day % 7 == 0`. Missing entries mean "never
        # compacted"; we do not backfill missed days — NPCs that
        # joined mid-run only lose a bit of provenance structure.
        self._last_compacted_day: dict[str, int] = {}
        self._last_compacted_week: dict[str, int] = {}

        # Phase I.1 — bedtime self-review cursor. Mirrors
        # `_last_compacted_day`: review runs once per NPC per game
        # day on the tick after compact_day succeeds for the
        # previous day. No backfill, same cursor-guard pattern.
        self._last_self_reviewed_day: dict[str, int] = {}

        # Collective town goals — overseer proposes, scheduler reads,
        # client renders. Gives the town a shared sense of direction
        # beyond individual NPC schedules.
        self.town_agenda = TownAgenda()
        self.town_agenda.add_completion_listener(self._on_goal_completed)
        self.town_agenda.add_propose_listener(self._on_goal_proposed)
        self.town_agenda.add_expire_listener(self._on_goal_expired)

        # Focus point for tier assignment (defaults to town centre)
        self.focus_x: int = 0
        self.focus_z: int = 0

        # Tick tracking
        self._last_slot: str = ""
        self._current_minutes: float = 0.0
        self._current_day: int = 0
        self._conversation_check_counter: int = 0
        self._reflection_check_counter: int = 0

        # Staggered departures: npc_id -> (delay_remaining_secs, target_x, target_z, description)
        # Delays are in REAL seconds (not game minutes) to guarantee visible spread
        self._pending_departures: dict[str, tuple[float, int, int, str]] = {}

    # Player agent reference — set by server after creation.
    # When set and autonomous=False, the player NPC is excluded from
    # automatic cognition loops (schedules, conversations, etc.).
    player_agent: object | None = None  # PlayerAgent, typed as object to avoid circular import

    def _skip_autonomous(self, npc: NPC) -> bool:
        """Check if this NPC should be skipped from autonomous cognition.

        Returns True only for the player NPC when autonomous mode is off.
        When autonomous=True, the player NPC acts like any other NPC.
        """
        if npc.npc_id != "player":
            return False
        if self.player_agent is None:
            return False
        return not self.player_agent.autonomous

    def get_npc(self, npc_id: str) -> NPC | None:
        return self._npc_map.get(npc_id)

    # ---------- Spawning ----------

    def spawn_population(self, count: int) -> list[NPC]:
        """Create the initial NPC population.

        Stanford pattern: each NPC gets a dedicated living_area. Homes
        are assigned round-robin with a max occupancy per building
        (interior tiles + door tile). No two NPCs share the same
        home tile.
        """
        occupations = self._assign_occupations(count)
        available_names = list(FIRST_NAMES)
        self.rng.shuffle(available_names)

        # Build home assignment table with max occupancy
        home_assignments = self._assign_homes(count)

        for i in range(count):
            occupation = occupations[i]
            name = available_names[i % len(available_names)]
            home_building, home_tile = home_assignments[i]

            npc = self._create_npc(name, occupation, home_building, i,
                                   home_tile=home_tile)
            self.npcs.append(npc)
            self._npc_map[npc.npc_id] = npc

        # Seed foundational memories and goals
        seed_population_memories(
            self.npcs, self.memory,
            sentiment=self.sentiment, rng=self.rng,
        )

        # Stanford: every NPC starts with a schedule and begins their
        # first action immediately. action_start_minutes = 0 so the
        # first cognition_tick will see the duration check and advance
        # once enough game time has elapsed.
        for npc in self.npcs:
            _force_template_schedule(npc, 1)
            npc.schedule_index = 0
            npc.action_start_minutes = 0.0

        logger.info("Spawned %d NPCs (with seed memories)", len(self.npcs))
        return self.npcs

    def _assign_homes(
        self, count: int,
    ) -> list[tuple[PlacedBuilding | None, tuple[int, int]]]:
        """Assign each NPC a unique home tile.

        Each home building has a capacity = interior tiles + door tile.
        NPCs are distributed round-robin across homes. If all homes
        are full, overflow NPCs get the door tile of the least-occupied
        home (resolve_overlaps will separate them on first tick).

        Returns list of (building, (tile_x, tile_z)) — one per NPC.
        """
        from core.world.generator import _building_interior_tiles

        homes = [b for b in self.buildings if b.building_type == "home"]
        if not homes:
            return [(None, (0, 0))] * count

        # Build tile pools per home: interior tiles + door tile
        home_tiles: list[list[tuple[int, int]]] = []
        for h in homes:
            tiles = sorted(_building_interior_tiles(
                h.x, h.z, h.width, h.height, h.door_x, h.door_z,
            ))
            tiles.append((h.door_x, h.door_z))  # door as overflow slot
            home_tiles.append(tiles)

        # Track which tiles have been assigned globally
        assigned_tiles: set[tuple[int, int]] = set()
        # Track occupancy per home
        home_occupancy: list[int] = [0] * len(homes)

        assignments: list[tuple[PlacedBuilding | None, tuple[int, int]]] = []

        for i in range(count):
            # Round-robin across homes
            home_idx = i % len(homes)
            home = homes[home_idx]
            tiles = home_tiles[home_idx]

            # Find the first unassigned tile in this home
            tile = None
            for t in tiles:
                if t not in assigned_tiles:
                    tile = t
                    break

            if tile is None:
                # Home is full — find the least occupied home with space
                for attempt in range(len(homes)):
                    alt_idx = (home_idx + attempt + 1) % len(homes)
                    for t in home_tiles[alt_idx]:
                        if t not in assigned_tiles:
                            home_idx = alt_idx
                            home = homes[alt_idx]
                            tile = t
                            break
                    if tile is not None:
                        break

            if tile is None:
                # All homes truly full — use door of least-occupied home
                min_idx = min(range(len(homes)), key=lambda j: home_occupancy[j])
                home = homes[min_idx]
                tile = (home.door_x, home.door_z)
                logger.warning(
                    "All home tiles occupied — NPC %d sharing door of %s",
                    i, home.name,
                )

            assigned_tiles.add(tile)
            home_occupancy[home_idx] += 1
            assignments.append((home, tile))

        return assignments

    def _assign_occupations(self, count: int) -> list[str]:
        """Distribute occupations based on available buildings."""
        occupations: list[str] = []

        # One per essential building
        building_types = {b.building_type for b in self.buildings}
        occupation_building_map = {
            "blacksmith": "blacksmith",
            "tavern_keeper": "tavern",
            "merchant": "market_stall",
            "priest": "church",
            "guard": "town_hall",
        }

        for occ, btype in occupation_building_map.items():
            if btype in building_types and len(occupations) < count:
                occupations.append(occ)

        # Fill with farmers and labourers
        farm_count = sum(1 for b in self.buildings if b.building_type == "farm")
        for _ in range(min(farm_count, count - len(occupations))):
            occupations.append("farmer")

        while len(occupations) < count:
            occupations.append(self.rng.choice(["labourer", "farmer", "merchant"]))

        self.rng.shuffle(occupations)
        return occupations[:count]

    def _create_npc(
        self,
        name: str,
        occupation: str,
        home: PlacedBuilding | None,
        index: int,
        home_tile: tuple[int, int] | None = None,
    ) -> NPC:
        """Create a single NPC with generated identity."""
        defaults = OCCUPATION_DEFAULTS.get(occupation, OCCUPATION_DEFAULTS["labourer"])

        # Find workplace
        work_building = self._find_work_building(defaults.get("work_building"))

        home_x, home_z = home_tile or self._pick_interior_tile(home, index)
        work_x, work_z = self._pick_interior_tile(work_building, index)

        age = self.rng.randint(18, 65)
        u = self.rng.uniform
        personality = PersonalityTraits(
            openness=u(0.2, 0.8), conscientiousness=u(0.2, 0.8),
            extraversion=u(0.2, 0.8), agreeableness=u(0.2, 0.8),
            neuroticism=u(0.1, 0.6),
        )
        backstory = self.rng.choice(BACKSTORY_TEMPLATES).format(
            name=name, age_desc="years ago" if age > 30 else "recently",
            occupation=occupation, origin=self.rng.choice(
                ["the northern highlands", "a coastal village", "the capital",
                 "a farming hamlet", "across the mountains"]),
        )
        npc_id = f"{occupation}_{index}"
        # Stanford living_area: hierarchical address of the NPC's home tile
        home_tile_obj = self.grid.get_tile(home_x, home_z)
        living_area = home_tile_obj.address if home_tile_obj else "smallville"

        # Snapshot the personality as the spawn baseline so future
        # drift from emotional reflections decays back toward it.
        spawn_baseline = personality.copy()

        npc = NPC(
            npc_id=npc_id,
            name=name,
            age=age,
            personality=personality,
            spawn_baseline=spawn_baseline,
            persona=self._persona_forge.forge(occupation),
            backstory=backstory,
            occupation=occupation,
            x=home_x,
            z=home_z,
            home_x=home_x,
            home_z=home_z,
            work_x=work_x,
            work_z=work_z,
            living_area=living_area,
            health=1.0,
            energy=self.rng.uniform(0.7, 1.0),
            hunger=self.rng.uniform(0.0, 0.2),
            long_term_goals=list(defaults.get("goals", [])),
            skills=dict(defaults.get("skills", {})),
            gold=self.rng.randint(10, 100),
            move_speed=self.rng.uniform(2.0, 4.0),
            _rng=random.Random(hash((self._seed, npc_id))),
        )

        return npc

    @staticmethod
    def _pick_interior_tile(
        building: PlacedBuilding | None, npc_index: int,
    ) -> tuple[int, int]:
        """Pick a passable interior tile inside a building for an NPC.

        Uses _building_interior_tiles to get only genuinely passable
        interior tiles (not walls). Falls back to door if no interior.
        """
        if building is None:
            return (0, 0)
        from core.world.generator import _building_interior_tiles
        interior = sorted(_building_interior_tiles(
            building.x, building.z, building.width, building.height,
            building.door_x, building.door_z,
        ))
        if not interior:
            return (building.door_x, building.door_z)
        return interior[npc_index % len(interior)]

    def _find_work_building(self, building_type: str | None) -> PlacedBuilding | None:
        """Find a building of the given type for NPC workplace."""
        if building_type is None:
            return None
        matches = [b for b in self.buildings if b.building_type == building_type]
        return self.rng.choice(matches) if matches else None

    # ---------- Tick cycle ----------

    def movement_tick(
        self,
        clock: GameClock,
        real_delta: float,
    ) -> dict[str, Any]:
        """Fast movement-only tick — runs at steady 4Hz, never blocks on LLM.

        Handles: departures, movement execution, overlap resolution,
        subtask timers, economy, and state broadcast.
        """
        current_slot = clock.schedule_slot.value
        game_minutes_elapsed = real_delta / clock._real_seconds_per_game_minute()

        # Reset per-tick arrival claims so _arrive() can track
        # which tiles have been claimed by earlier arrivals this tick.
        clear_arrival_claims()

        # Reset per-tick trail for every NPC. The trail is meant to
        # carry THIS tick's discrete movement (a step along a path, an
        # overlap nudge) to the client so it can animate smoothly.
        # Without this reset, NPCs that aren't currently walking — in
        # particular the player, whose trail is only populated when
        # resolve_overlaps nudges them off an NPC's tile — keep
        # broadcasting the same stale trail every tick. The client's
        # trail handling then appends those stale waypoints forever
        # and the avatar appears frozen far from the real server
        # position. Clearing here makes the trail field truly
        # per-tick.
        for npc in self.npcs:
            npc._tick_trail = []

        # Process pending departures (staggered in real seconds)
        if self._pending_departures:
            self._process_pending_departures(real_delta)

        # Execute movement and actions.
        #
        # INVARIANT: the player's position is input-authoritative. Any
        # code path in this manager that might write to an NPC's (x,z)
        # must short-circuit on npc_id == "player". The autonomous flag
        # is about COGNITION (should the player NPC think on its own
        # when idle?) — it must NEVER re-enable server-side writes to
        # the player's position. Two prior bugs landed because
        # different nudge paths (resolve_overlaps, _arrive inside
        # execute_tick) each needed to learn this separately; keeping
        # the player out of execute_tick entirely closes the whole
        # class of "avatar moves by itself" bugs at the source.
        for npc in self.npcs:
            if npc.npc_id == "player":
                continue
            if self._skip_autonomous(npc):
                continue
            if (npc.cognition_tier >= 4
                    and not npc.current_path
                    and npc.activity != ActivityState.WALKING):
                continue
            execute_tick(
                npc, self.grid, self.buildings, current_slot,
                real_delta, all_npcs=self.npcs,
            )
            if npc.cognition_tier < 4:
                npc.tick_needs(game_minutes_elapsed)

        # Safety net: resolve any resting NPCs that share a tile
        resolve_overlaps(self.npcs, self.grid)

        # Safety net #2: teleport NPCs back to their home if they've
        # somehow drifted absurdly far from any landmark they care
        # about (home, work, current schedule target). Over long runs
        # a small constant drift from resolve_overlaps' spread can
        # accumulate and park NPCs at the map border. Without this
        # we observed NPCs lined up at x=-30 after ~40 game days.
        self._reanchor_strays()

        # Economy tick
        current_minutes = clock.day * MINUTES_PER_DAY + clock.minutes
        self.economy.tick(self.npcs, game_minutes_elapsed, current_minutes)

        # Post-conversation dispatch
        self._dispatch_post_conversation(current_slot)

        # Final overlap safety net
        resolve_overlaps(self.npcs, self.grid)

        return self._build_tick_state()

    async def cognition_tick(
        self,
        clock: GameClock,
        real_delta: float,
    ) -> None:
        """Slow cognition tick — runs independently, may block on LLM calls.

        Handles: tier updates, schedule generation, slot transitions,
        perception, conversations, reflections.
        """
        current_minutes = clock.day * MINUTES_PER_DAY + clock.minutes
        self._current_minutes = current_minutes
        self._current_day = clock.day
        current_slot = clock.schedule_slot.value
        current_day = clock.day

        self._last_slot = current_slot

        # 0. Daily sentiment decay — drift all dimensions toward zero
        if not hasattr(self, "_last_decay_day"):
            self._last_decay_day = -1
        if current_day != self._last_decay_day:
            self._last_decay_day = current_day
            decayed = self.sentiment.decay_all(current_minutes)
            if decayed:
                logger.debug("Sentiment decay applied to %d relationships", decayed)
            # Personality drift decays back toward spawn baselines on the
            # same cadence. Keeps the Big-5 vector bounded while still
            # allowing strong recent events to be visible for days.
            self._decay_personalities()

        # 0b. Daily overseer evaluation — score population, detect issues, intervene
        if current_day != self._last_eval_day and current_day > 0:
            self._last_eval_day = current_day
            await self._run_overseer_eval(current_day, current_minutes)

        # 0c. Phase H.6 — daily memory compaction. Runs on the first
        # tick of a new day, collapsing the PREVIOUS day's untagged
        # firehose into a single day_summary per NPC. One LLM call
        # per NPC per game day, routed through the cognition router
        # so scene pressure / budget throttle it naturally. Week
        # rollup piggybacks every 7th day.
        if current_day > 0:
            await self._run_daily_compaction(current_day - 1)
            # Phase I.1 — bedtime self-review. Runs AFTER compaction
            # in the same tick so the review can read the fresh
            # day_summary. Router gate is independent per the policy
            # (self_review defaults to ROUTE_LLM; compaction to
            # ROUTE_AUTO) so a downgraded compaction doesn't silently
            # downgrade the review too.
            await self._run_daily_self_review(current_day - 1)

        # 1. Update tiers based on focus point
        update_all_tiers(self.npcs, self.focus_x, self.focus_z)

        # 1b. Schedule-cursor safety net. Over long runs we've observed
        #     NPCs end up with (schedule_index > 0, daily_schedule=[])
        #     — typically from an exception mid-_advance_npc_action or
        #     from an LLM/planner regen that left the schedule empty
        #     without resetting the cursor. That state is unrecoverable
        #     by the cycling loop (step 2 needs schedule_day to match
        #     current_day to skip regen; step 3 bails on empty). Repair
        #     here is cheap and closes the whole class.
        self._normalise_schedule_cursors()

        # 2. Schedule generation (router decides LLM vs deterministic)
        # Player NPC excluded — player movement is input-driven
        schedule_tasks = []
        for npc in self.npcs:
            if self._skip_autonomous(npc):
                continue
            if npc.cognition_tier < 4 and npc.needs_new_schedule(current_day):
                if self.deterministic:
                    # Template-only mode — bypass LLM entirely
                    _force_template_schedule(npc, current_day)
                    continue

                decision = self.router.route(
                    npc, "daily_schedule",
                    focus_x=self.focus_x, focus_z=self.focus_z,
                )
                if decision.route == Route.LLM:
                    rel_summary = self._build_relationship_summary(npc)
                    schedule_tasks.append(
                        generate_daily_schedule(
                            npc, self.llm, current_day,
                            relationship_summary=rel_summary,
                            town_agenda_summary=self.town_agenda.summary_for_prompt(
                                npc.npc_id, self_concept=npc.self_concept,
                            ),
                        )
                    )
                else:
                    self._generate_deterministic_schedule(
                        npc, current_slot, current_day,
                    )

        if schedule_tasks:
            import asyncio
            await asyncio.gather(*schedule_tasks)

        # 2b. Inject town-goal entries into NPCs whose schedules were
        #     just (re)generated today. This is what turns the town
        #     agenda from a piece of state into visible collective
        #     behaviour — matching NPCs will walk to the goal location
        #     and contribute during the goal's slot.
        for npc in self.npcs:
            if self._skip_autonomous(npc) or npc.cognition_tier >= 4:
                continue
            if getattr(npc, "schedule_day", None) != current_day:
                continue
            if npc.goal_injected_for_day == current_day:
                continue
            self._inject_goal_entry(npc, current_day)
            npc.goal_injected_for_day = current_day

        # 2b-proj. Re-derive goal entries from durable commitments every
        #     tick (Phase 3). Idempotent: only acts when a live
        #     commitment's goal entry is missing from the reachable
        #     schedule (e.g. the previous tick's replan wiped it), so the
        #     goal is always present before the action-advance in step 3.
        #     No-op for NPCs without live commitments.
        for npc in self.npcs:
            if self._skip_autonomous(npc) or npc.cognition_tier >= 4:
                continue
            self._project_commitments(npc)

        # 2c. Bed-time enforcement. Schedule entries are duration-
        #     based, so a badly-sized entry can keep an NPC at work
        #     past midnight if its own timer hasn't expired. Game
        #     design needs "at night, NPCs are home" to be a hard
        #     invariant: it's the most legible failure mode (players
        #     see NPCs wandering at 2 AM and lose all immersion).
        #     During the night phase, fast-forward any non-home NPC
        #     to their final sleep entry.
        if current_slot == "night":
            await self._enforce_bedtime(current_slot, current_minutes)

        # 3. Stanford action-duration cycling — each NPC independently
        #    advances to the next schedule entry when their current
        #    action's duration expires. No global slot transitions.
        for npc in self.npcs:
            if self._skip_autonomous(npc) or npc.cognition_tier >= 4:
                continue
            if not npc.daily_schedule:
                continue
            if npc.schedule_index >= len(npc.daily_schedule):
                continue

            # First tick after spawn/schedule reset: anchor the timer
            if npc.action_start_minutes == 0.0:
                npc.action_start_minutes = current_minutes
                # Also dispatch NPC to the first entry's location
                entry = npc.daily_schedule[npc.schedule_index]
                await self._dispatch_to_entry(npc, entry, current_slot)
                continue

            entry = npc.daily_schedule[npc.schedule_index]
            if entry.duration_minutes <= 0:
                continue
            elapsed = current_minutes - npc.action_start_minutes
            if elapsed >= entry.duration_minutes:
                await self._advance_npc_action(npc, current_minutes, current_slot)

        # 4. Perception cycle (tier-dependent intervals)
        #    Skip during night — NPCs should be sleeping, not reacting
        if current_slot != "night":
            for npc in self.npcs:
                if self._skip_autonomous(npc):
                    continue
                if should_perceive(npc, current_minutes):
                    observations = perceive(
                        npc, self.grid, self.npcs, current_minutes,
                        sentiment=self.sentiment,
                    )

                    for obs in observations:
                        tile = self.grid.get_tile(obs.x, obs.z)
                        await self.memory.store_perception(
                            npc_id=npc.npc_id,
                            description=obs.description,
                            category=obs.category,
                            importance=obs.importance,
                            game_time=current_minutes,
                            location_x=obs.x,
                            location_z=obs.z,
                            tile_sector=tile.sector if tile else "",
                            tile_arena=tile.arena if tile else "",
                            mentioned_npc_id=obs.subject_npc_id,
                        )

                    if observations:
                        obs = observations[0]
                        rd = self.router.route(
                            npc, "reaction",
                            focus_x=self.focus_x, focus_z=self.focus_z,
                            novelty=obs.importance,
                        )
                        if rd.route == Route.LLM:
                            from core.npc.cognition.plan import decide_reaction
                            reaction = await decide_reaction(npc, obs.description, self.llm)
                            if reaction == "approach":
                                navigate_to(npc, self.grid, obs.x, obs.z)

        # 5. Conversation system (check periodically)
        self._conversation_check_counter += 1
        if self._conversation_check_counter >= 3 and current_slot != "night":
            self._conversation_check_counter = 0
            await self._run_conversations(current_minutes)

        # 6. Persist finished conversations to memory, then clean up
        await self._persist_finished_conversations(current_minutes)
        clear_finished_conversations()

        # 6b. Conversations can reposition NPCs — resolve overlaps
        #     so movement_tick broadcasts clean state.
        resolve_overlaps(self.npcs, self.grid)

        # 7. Periodic reflection check
        self._reflection_check_counter += 1
        if self._reflection_check_counter >= 15:
            self._reflection_check_counter = 0
            await self._check_reflections(current_minutes)

        # 8. Mid-day replanning for Tier 1-2 NPCs
        if not self.deterministic and current_slot != "night":
            await self._check_replans(current_minutes, current_slot)

    async def tick(
        self,
        clock: GameClock,
        real_delta: float,
    ) -> dict[str, Any]:
        """Legacy combined tick — calls both movement and cognition.

        Used when the caller doesn't run separate loops.
        """
        await self.cognition_tick(clock, real_delta)
        return self.movement_tick(clock, real_delta)

    async def _advance_npc_action(
        self, npc: NPC, current_minutes: float, current_slot: str,
    ) -> None:
        """Stanford-style action cycling: advance to the next schedule entry.

        Called when the current action's duration has expired. Resolves
        the next entry's location, decomposes into subtasks, and queues
        a staggered departure. Sleep is just another action — no snap.
        """
        # Phase F town-goal contribution: if the entry that just
        # finished was a town-goal entry, credit the NPC now —
        # NOT at injection time. Eager injection-time crediting
        # caused every goal with `required=N` to complete in the
        # same tick it was proposed with N personality-matching
        # NPCs "contributing" without having moved at all.
        if 0 <= npc.schedule_index < len(npc.daily_schedule):
            finishing = npc.daily_schedule[npc.schedule_index]
            if finishing.goal_id:
                completed = self._credit_goal_entry(
                    npc, finishing.goal_id, npc.schedule_day,
                )
                logger.info(
                    "AGENDA %s: contributed to goal=%s (slot=%s%s)",
                    npc.name, finishing.goal_id, finishing.slot,
                    ", COMPLETED" if completed else "",
                )

        # Advance to next schedule entry
        npc.schedule_index += 1

        # Schedule exhausted — loop or regenerate
        if npc.schedule_index >= len(npc.daily_schedule):
            if npc.has_custom_schedule:
                # Custom schedules loop indefinitely
                pass
            elif self.deterministic:
                _force_template_schedule(npc, npc.schedule_day + 1)
            else:
                npc.daily_schedule = []  # triggers regeneration next tick
            npc.schedule_index = 0
            # Reset to 0.0 so the first-dispatch logic (action_start_minutes == 0.0)
            # fires on the next cognition tick and sends the NPC to their new location.
            npc.action_start_minutes = 0.0
            return

        entry = npc.daily_schedule[npc.schedule_index]
        npc.action_start_minutes = current_minutes

        # Wrap the dispatch side of the advance in try/except so that a
        # downstream failure (e.g. pathfinding throws on a weird target,
        # conversation end fails) can't strand the NPC mid-advance with
        # an incremented cursor and no dispatched action. On failure we
        # surrender the advance: keep the new cursor (the entry is
        # theoretically valid) but flag a post-convo dispatch so the
        # movement tick's fallback path redispatches next tick.
        try:
            # Sleep entries: force-end conversations so NPCs go home.
            # Stanford: sleep is just an action, but NPCs must actually
            # walk home — they can't chat all night at the tavern.
            is_sleep = entry.location == "home" and "sleep" in entry.activity.lower()
            if npc.conversation_partner and is_sleep:
                other = self.get_npc(npc.conversation_partner)
                if other:
                    await end_conversation(
                        npc, other, memory_manager=self.memory,
                    )
                    # end_conversation sets needs_post_convo_dispatch on both,
                    # but we handle the sleep NPC directly below — clear the flag
                    # to prevent double-dispatch in movement_tick.
                    npc.needs_post_convo_dispatch = False

            # NPCs in conversation (non-sleep): don't dispatch, pick up on end
            if npc.conversation_partner:
                return

            # Don't interrupt walking NPCs — they finish their path first
            if npc.activity == ActivityState.WALKING and npc.current_path:
                npc.needs_post_convo_dispatch = True
                return

            await self._dispatch_to_entry(npc, entry, current_slot)
        except Exception:
            logger.exception(
                "ADVANCE %s: dispatch failed on idx=%d (%s) — flagging "
                "post-convo redispatch",
                npc.name, npc.schedule_index, entry.activity,
            )
            npc.needs_post_convo_dispatch = True

    async def _dispatch_to_entry(
        self, npc: NPC, entry: ScheduleEntry, current_slot: str,
    ) -> None:
        """Resolve location, decompose, and queue staggered departure.

        Stanford model: subtasks only start AFTER the NPC arrives at the
        destination. If the NPC needs to walk, they walk first — subtasks
        are assigned on arrival via set_activity_for_location.
        """
        # Clear stale state
        npc.current_subtask = None
        npc.subtask_time_remaining = 0.0
        npc.subtask_queue = []
        self._pending_departures.pop(npc.npc_id, None)

        # Resolve target coordinates
        if entry.target_x is not None and entry.target_z is not None:
            target_x, target_z = entry.target_x, entry.target_z
        else:
            target_x, target_z = resolve_schedule_location(
                entry, npc, self.buildings,
            )
        target_x, target_z = self._spread_destination(npc, target_x, target_z)

        # Persist the resolved coordinates on the entry immediately.
        # Without this, post-conversation redispatch or schedule inspection
        # would re-resolve to the raw home/work tile and cause re-stacking.
        entry.target_x = target_x
        entry.target_z = target_z

        # If the NPC is already near this target (within 1 tile) and at
        # a free tile, don't re-navigate — stay put. Prevents vibration
        # from being sent 1 tile away and back every schedule cycle.
        if not npc.is_at(target_x, target_z) and npc.distance_to(target_x, target_z) <= 1.5:
            occupied = self._get_all_claimed_tiles(exclude_npc_id=npc.npc_id)
            if (npc.tile_x, npc.tile_z) not in occupied:
                target_x, target_z = npc.tile_x, npc.tile_z
                entry.target_x = target_x
                entry.target_z = target_z

        # Already at destination — decompose and start activity
        if npc.is_at(target_x, target_z):
            rd = self.router.route(
                npc, "task_decompose",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            if rd.route == Route.LLM:
                await self._decompose_llm(npc, entry)
            else:
                subtasks = decompose_schedule_entry(npc, entry, npc._rng)
                npc.subtask_queue = subtasks
            set_activity_for_location(npc, current_slot)
            logger.info(
                "ACTION %s: %s (already at (%d,%d))",
                npc.name, entry.activity, target_x, target_z,
            )
            return

        # Not at destination — walk there first, Stanford style.
        # Subtasks are pre-decomposed so they're ready on arrival.
        subtasks = decompose_schedule_entry(npc, entry, npc._rng)
        npc.subtask_queue = subtasks

        # Navigate immediately — no stagger delay needed because
        # duration-based cycling naturally desynchronizes NPCs.
        if navigate_to(npc, self.grid, target_x, target_z):
            npc.current_action_description = f"heading to {entry.activity}"
            logger.info(
                "ACTION %s: walking to %s at (%d,%d)",
                npc.name, entry.activity, target_x, target_z,
            )
        else:
            # No path — snap to nearest free tile at destination
            sx, sz = self._spread_destination(npc, target_x, target_z)
            npc.x = float(sx)
            npc.z = float(sz)
            entry.target_x = sx
            entry.target_z = sz
            set_activity_for_location(npc, current_slot)
            logger.warning(
                "ACTION %s: no path to (%d,%d), snapping",
                npc.name, target_x, target_z,
            )

    def _dispatch_post_conversation(self, current_slot: str) -> None:
        """One-time dispatch for NPCs that just finished a conversation.

        Uses the NPC's current schedule_index entry to dispatch them
        to the right location. Fires exactly once per conversation end.
        """
        for npc in self.npcs:
            if not npc.needs_post_convo_dispatch:
                continue
            npc.needs_post_convo_dispatch = False

            if npc.conversation_partner:
                continue

            if not npc.daily_schedule or npc.schedule_index >= len(npc.daily_schedule):
                continue

            entry = npc.daily_schedule[npc.schedule_index]

            subtasks = decompose_schedule_entry(npc, entry, npc._rng)
            npc.subtask_queue = subtasks
            npc.current_subtask = None

            if entry.target_x is not None and entry.target_z is not None:
                target_x, target_z = entry.target_x, entry.target_z
            else:
                target_x, target_z = resolve_schedule_location(
                    entry, npc, self.buildings,
                )
            target_x, target_z = self._spread_destination(npc, target_x, target_z)

            if not npc.is_at(target_x, target_z):
                delay = npc._rng.uniform(1.0, 5.0)
                self._pending_departures[npc.npc_id] = (
                    delay, target_x, target_z,
                    f"heading to {entry.activity}",
                )
                logger.info(
                    "INTENT %s: post-convo → %s at (%d,%d) delay=%.1fs",
                    npc.name, entry.activity, target_x, target_z, delay,
                )

    def _process_pending_departures(self, real_delta: float) -> None:
        """Tick down departure delays (real seconds) and dispatch NPCs when ready."""
        completed: list[str] = []

        for npc_id, (delay, tx, tz, desc) in self._pending_departures.items():
            remaining = delay - real_delta
            if remaining <= 0:
                npc = self.get_npc(npc_id)
                if npc and not npc.conversation_partner:
                    # Don't override NPCs already walking — let them
                    # finish their current path first (Stanford model).
                    if npc.activity == ActivityState.WALKING and npc.current_path:
                        completed.append(npc_id)  # discard stale departure
                        continue
                    if navigate_to(npc, self.grid, tx, tz):
                        npc.current_action_description = desc
                        logger.info(
                            "DEPART %s: now walking to (%d,%d) — %s",
                            npc.name, tx, tz, desc,
                        )
                    else:
                        # No path — snap to nearest free tile at
                        # destination rather than raw target (avoids stacking).
                        sx, sz = self._spread_destination(npc, tx, tz)
                        npc.x = float(sx)
                        npc.z = float(sz)
                        logger.info(
                            "DEPART %s: no path to (%d,%d), snapping to (%d,%d)",
                            npc.name, tx, tz, sx, sz,
                        )
                        set_activity_for_location(
                            npc, self._last_slot,
                        )
                completed.append(npc_id)
            else:
                self._pending_departures[npc_id] = (remaining, tx, tz, desc)

        for npc_id in completed:
            del self._pending_departures[npc_id]

    def _get_all_claimed_tiles(
        self, exclude_npc_id: str = "",
    ) -> set[tuple[int, int]]:
        """Build the full set of tiles that are occupied OR claimed.

        Includes:
        - Resting NPC positions (standard occupied set)
        - Destinations of walking NPCs (path endpoint)
        - Targets of pending staggered departures

        This prevents two NPCs from being dispatched to the same
        tile even when one is still walking there.
        """
        from core.world.spatial_awareness import get_occupied_tiles
        claimed = get_occupied_tiles(self.npcs)

        # Add destinations of walking NPCs
        for o in self.npcs:
            if o.npc_id == exclude_npc_id:
                continue
            if o.activity == ActivityState.WALKING and o.current_path:
                dest = o.current_path[-1]
                claimed.add(dest)

        # Add targets of pending departures
        for npc_id, (_, tx, tz, _) in self._pending_departures.items():
            if npc_id == exclude_npc_id:
                continue
            claimed.add((tx, tz))

        return claimed

    def _spread_destination(
        self, npc: NPC, target_x: int, target_z: int,
    ) -> tuple[int, int]:
        """
        If the target tile is already claimed by any NPC (resting,
        walking toward, or pending departure), find the nearest
        passable unclaimed neighbour instead.

        Door-aware: if the target is a building door, prefer the
        approach tile (one south of door) before spiralling randomly.
        This prevents NPCs entering buildings off-centre.
        """
        from core.world.spatial_awareness import find_rest_tile
        occupied = self._get_all_claimed_tiles(exclude_npc_id=npc.npc_id)

        # Check if target is free first (fast path)
        tile = self.grid.get_tile(target_x, target_z)
        if tile and tile.is_passable and (target_x, target_z) not in occupied:
            return (target_x, target_z)

        # Door-aware: if target is a building door, try approach tile
        # (one south) before spiralling randomly
        for b in self.buildings:
            if b.door_x == target_x and b.door_z == target_z:
                approach = (target_x, target_z + 1)
                at = self.grid.get_tile(approach[0], approach[1])
                if at and at.is_passable and approach not in occupied:
                    return approach
                break

        return find_rest_tile(
            target_x, target_z, self.grid, occupied,
            exclude_npc_id=npc.npc_id, npcs=self.npcs,
        )

    async def _run_conversations(self, current_minutes: float) -> None:
        """Check for conversation opportunities and advance active ones."""
        # Continue active conversations
        active = get_active_conversations()
        for conv in active:
            npc_a = self.get_npc(conv.npc_a_id)
            npc_b = self.get_npc(conv.npc_b_id)
            if not npc_a or not npc_b:
                continue

            # Determine who speaks next. Pass the responder's agenda
            # summary so their line can reference town matters they've
            # committed to. Also pass any unresolved matters this
            # responder holds about their partner (Phase C) so the
            # prompt can nudge them to raise it.
            last_speaker = conv.exchanges[-1].speaker_id if conv.exchanges else None
            current_day_for_prompts = getattr(
                self, "_current_day", 0,
            )
            if last_speaker == npc_a.npc_id:
                matters = self.memory.retrieve_unresolved_matters(
                    npc_b.npc_id,
                    partner_id=npc_a.npc_id,
                    partner_name=npc_a.name,
                )
                await continue_conversation(
                    npc_b, npc_a, self.llm, memory_manager=self.memory,
                    town_agenda_summary=self.town_agenda.summary_for_prompt(
                        npc_b.npc_id, self_concept=npc_b.self_concept,
                    ),
                    shared_agenda_summary=(
                        self.town_agenda.shared_matters_for_prompt(
                            npc_b.npc_id, npc_a.npc_id,
                            current_day=current_day_for_prompts,
                        )
                    ),
                    unresolved_matters_summary=(
                        self.memory.format_unresolved_matters(
                            matters, npc_a.name,
                        )
                    ),
                )
            else:
                matters = self.memory.retrieve_unresolved_matters(
                    npc_a.npc_id,
                    partner_id=npc_b.npc_id,
                    partner_name=npc_b.name,
                )
                await continue_conversation(
                    npc_a, npc_b, self.llm, memory_manager=self.memory,
                    town_agenda_summary=self.town_agenda.summary_for_prompt(
                        npc_a.npc_id, self_concept=npc_a.self_concept,
                    ),
                    shared_agenda_summary=(
                        self.town_agenda.shared_matters_for_prompt(
                            npc_a.npc_id, npc_b.npc_id,
                            current_day=current_day_for_prompts,
                        )
                    ),
                    unresolved_matters_summary=(
                        self.memory.format_unresolved_matters(
                            matters, npc_b.name,
                        )
                    ),
                )

            # Phase A.4 — per-turn persistence for NPC↔NPC chats.
            # Same cursor-based helper the player path uses. Each
            # tick, any newly-added exchanges land in both
            # participants' memory immediately.
            try:
                await self.memory.persist_new_exchanges(
                    conv, npc_a, npc_b,
                    game_time=current_minutes,
                    location_x=int(npc_a.x),
                    location_z=int(npc_a.z),
                )
            except Exception:
                logger.exception(
                    "NPC↔NPC per-turn persistence failed (conv=%s)",
                    getattr(conv, "conv_id", "?"),
                )

        # Check for new conversation opportunities (tier 1-3 only)
        # Player NPC is excluded — player conversations are initiated via chat input
        for npc in self.npcs:
            if self._skip_autonomous(npc):
                continue
            if npc.conversation_partner or npc.cognition_tier >= 4:
                continue
            for other in self.npcs:
                if other.npc_id == npc.npc_id or other.conversation_partner:
                    continue
                if self._skip_autonomous(other):
                    continue
                if should_initiate_conversation(npc, other, current_minutes):
                    initiator_matters = self.memory.retrieve_unresolved_matters(
                        npc.npc_id,
                        partner_id=other.npc_id,
                        partner_name=other.name,
                    )
                    await initiate_conversation(
                        npc, other, self.llm, current_minutes,
                        memory_manager=self.memory,
                        grid=self.grid,
                        all_npcs=self.npcs,
                        town_agenda_summary=self.town_agenda.summary_for_prompt(
                            npc.npc_id, self_concept=npc.self_concept,
                        ),
                        shared_agenda_summary=(
                            self.town_agenda.shared_matters_for_prompt(
                                npc.npc_id, other.npc_id,
                                current_day=self._current_day,
                            )
                        ),
                        unresolved_matters_summary=(
                            self.memory.format_unresolved_matters(
                                initiator_matters, other.name,
                            )
                        ),
                    )
                    break  # one conversation attempt per tick per NPC

    async def _persist_finished_conversations(
        self, current_minutes: float,
    ) -> None:
        """Save finished conversations to memory before they're cleaned up.

        Iterates a snapshot of `_active_conversations` rather than the
        live dict: the body awaits LLM / memory calls, during which
        the chat task (`_handle_player_chat` in server/main.py) can
        add new conversations, and we must not raise
        `RuntimeError: dictionary changed size during iteration`
        mid-tick — that took the whole cognition loop down once in
        production and froze every NPC indoors the following day.
        """
        from core.npc.cognition.converse import _active_conversations

        for key, conv in list(_active_conversations.items()):
            if not conv.finished:
                continue
            # Empty-but-finished conversations have nothing worth
            # persisting, but they still must flip to persisted=True
            # so `clear_finished_conversations()` sweeps them out of
            # the active registry. Without this line, a short
            # conversation that ended before any exchange stayed in
            # the registry forever and broke sim responsiveness
            # tests that expected the tick loop to eventually clear.
            if not conv.exchanges:
                conv.persisted = True
                continue
            # Idempotency: a persistence attempt has already run for
            # this conversation (success OR swallowed failure). Skip
            # to stop the retry-crash loop that used to freeze every
            # NPC indoors whenever any downstream step threw.
            if conv.persisted:
                continue

            npc_a = self.get_npc(conv.npc_a_id)
            npc_b = self.get_npc(conv.npc_b_id)
            if not npc_a or not npc_b:
                # Without participants we can't do anything with this
                # conversation; mark it persisted so the cleanup
                # sweep removes it instead of re-iterating forever.
                conv.persisted = True
                continue

            exchanges = [
                {"speaker": e.speaker_name, "message": e.message}
                for e in conv.exchanges
            ]

            # Claim persistence up front. Even if any step below
            # throws an uncaught exception and aborts the rest of
            # this tick, next tick skips this conv via the
            # `if conv.persisted: continue` guard above — so a
            # single bad conversation can never crash-loop the
            # cognition tick and freeze every NPC. That regression
            # took an entire game day out of a live session; see
            # FAILED_APPROACHES.md Attempt 6 for the prior variant
            # (dict-size-changed-during-iteration).
            conv.persisted = True

            # Defensive wrapper around `record_conversation` — the
            # LLM fact-extraction inside it can throw on timeout or
            # provider error. Phase A per-turn memories have already
            # landed by this point so the conversation isn't lost,
            # but we must NOT propagate the exception and skip the
            # rest of the cognition tick (step 6b+7+).
            try:
                await self.memory.record_conversation(
                    npc_a_id=conv.npc_a_id,
                    npc_b_id=conv.npc_b_id,
                    npc_a_name=npc_a.name,
                    npc_b_name=npc_b.name,
                    exchanges=exchanges,
                    game_time=current_minutes,
                    location_x=npc_a.x,
                    location_z=npc_a.z,
                )
            except Exception:
                logger.exception(
                    "record_conversation failed for conv=%s "
                    "(%s ↔ %s); per-turn memories remain, "
                    "consolidation + outcomes will be skipped",
                    conv.conv_id, npc_a.name, npc_b.name,
                )
                continue

            # Phase A.3 — consolidate: the summary above is the durable
            # memory; the per-turn entries we wrote mid-conversation
            # are now redundant. Sweep them by conv_id. If any were
            # missed (tier changes, exceptions), they simply weren't
            # there — the call is a no-op in that case.
            self.memory.consolidate_conversation_turns(conv.conv_id)

            # Phase C.3 — BEFORE writing new outcomes, flip
            # `unresolved` → False on matters each participant held
            # against the other that are actually aired in this
            # transcript. Ordering is deliberate: resolving first
            # means we inspect only PRIOR matters; a brand-new
            # relayed_claim about this exact topic (written by the
            # B step below) doesn't falsely resolve itself.
            try:
                transcript_text = " ".join(
                    e.get("message", "") for e in exchanges
                )
                self.memory.resolve_matters_from_transcript(
                    npc_id=npc_a.npc_id,
                    partner_id=npc_b.npc_id,
                    partner_name=npc_b.name,
                    transcript_text=transcript_text,
                )
                self.memory.resolve_matters_from_transcript(
                    npc_id=npc_b.npc_id,
                    partner_id=npc_a.npc_id,
                    partner_name=npc_a.name,
                    transcript_text=transcript_text,
                )
            except Exception:
                logger.exception(
                    "Matter resolution failed for conv=%s", conv.conv_id,
                )

            # Phase B.6 — extract structured outcomes (commitments,
            # accusations, relayed claims) and persist them. Runs the
            # heuristic pass unconditionally; LLM extractor attempted
            # only when the NPC manager has one available. Failures
            # are swallowed — the transcript memory from
            # record_conversation is still enough to keep the sim
            # coherent, structured records are an upgrade.
            from core.memory.conversation_outcomes import (
                ConversationOutcome, extract_outcomes,
            )
            outcome: ConversationOutcome = ConversationOutcome()
            try:
                outcome = await extract_outcomes(
                    exchanges,
                    llm=self.llm if self.llm is not None else None,
                )
                if not outcome.is_empty():
                    self.memory.store_conversation_outcomes(
                        outcome,
                        participants={
                            npc_a.npc_id: npc_a.name,
                            npc_b.npc_id: npc_b.name,
                        },
                        game_time=current_minutes,
                        location_x=int(npc_a.x),
                        location_z=int(npc_a.z),
                    )
                    self._apply_accusation_sentiment(
                        outcome,
                        participants={
                            npc_a.npc_id: npc_a.name,
                            npc_b.npc_id: npc_b.name,
                        },
                        current_minutes=current_minutes,
                    )
                    logger.info(
                        "OUTCOMES %s↔%s: +%d commitments, +%d accusations, "
                        "+%d relayed (conv=%s)",
                        npc_a.name, npc_b.name,
                        len(outcome.commitments),
                        len(outcome.accusations),
                        len(outcome.relayed_claims),
                        conv.conv_id,
                    )
            except Exception:
                logger.exception(
                    "Outcome extraction failed for conv=%s", conv.conv_id,
                )

            # Fire conversation event through impact system
            self.events.process_event(GameEvent(
                event_type="conversation",
                participants=[conv.npc_a_id, conv.npc_b_id],
                game_time=current_minutes,
                location_x=npc_a.x,
                location_z=npc_a.z,
            ))

            # Post-conversation reflection.
            #
            # When outcome extraction produced any structured record
            # the conversation was by definition notable — force the
            # LLM path regardless of the router's default decision.
            # Otherwise let the router decide. Neutral chit-chat thus
            # stays cheap.
            #
            # Bounded-latency: each reflection runs with a hard
            # timeout and the TWO participants reflect concurrently
            # (they're independent NPCs). A slow or stuck LLM call
            # used to freeze the whole cognition tick — Jesse lost
            # an entire game-day to that once (day 12+ frozen after
            # 4-5 chained accusation chats). Bounding each call at
            # 15s keeps the tick responsive even if the LLM is
            # struggling; we just drop the reflection for that turn.
            notable = not outcome.is_empty()
            identity_conversation = {
                "exchanges": list(exchanges),
                "current_minutes": current_minutes,
                "outcome": outcome,
            }
            # Always drift personality and detect identity claims
            # regardless of reflection route — those are cheap and
            # don't depend on LLM.
            for npc, other_id, other_name in [
                (npc_a, npc_b.npc_id, npc_b.name),
                (npc_b, npc_a.npc_id, npc_a.name),
            ]:
                convo_text = " ".join(
                    e.get("message", "") for e in exchanges
                )
                apply_personality_drift(npc, convo_text, importance=0.6)
                claims = detect_identity_claims(
                    exchanges, listener_name=npc.name, speaker_id=other_id,
                )
                for claim in claims:
                    self._inject_self_concept_delta(
                        npc, claim, current_minutes,
                    )

            # Concurrent reflection pass, bounded at 15s per NPC.
            async def _reflect_one(
                npc_: NPC, other_name_: str, other_id_: str = "",
            ) -> None:
                rd = self.router.route(
                    npc_, "reflection",
                    focus_x=self.focus_x, focus_z=self.focus_z,
                )
                if rd.route != Route.LLM and not notable:
                    return
                try:
                    insight = await asyncio.wait_for(
                        reflect_on_conversation(
                            npc_, other_name_, exchanges,
                            self.memory, self.llm, current_minutes,
                            outcome=outcome,
                            other_id=other_id_,
                            # Reflection-born identity goes through the
                            # same contradiction-damped applier as
                            # conversation claims (write-paths arc).
                            claim_sink=lambda claim, _n=npc_: (
                                self._inject_self_concept_delta(
                                    _n, claim, current_minutes,
                                )
                            ),
                        ),
                        timeout=15.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Reflection timed out for %s (conv=%s)",
                        npc_.name, conv.conv_id,
                    )
                    return
                except Exception:
                    logger.exception(
                        "Reflection failed for %s (conv=%s)",
                        npc_.name, conv.conv_id,
                    )
                    return

                if not insight:
                    return
                apply_personality_drift(npc_, insight, importance=0.8)

                # Classify the insight into an action intent. If
                # actionable, inject a temporary schedule entry so
                # the NPC physically acts on their conclusion
                # ("go talk to Dara") rather than just storing it.
                # A reflection like "I distrust Bob" classifies as
                # NO_ACTION and nothing is injected.
                try:
                    intent = await asyncio.wait_for(
                        classify_insight(npc_, insight, self.llm),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        "Action-intent classify timed out for %s",
                        npc_.name,
                    )
                    intent = None
                except Exception:
                    logger.exception(
                        "Action-intent classify failed for %s",
                        npc_.name,
                    )
                    intent = None
                if intent is not None:
                    self._inject_reflection_entry(npc_, intent)

                # Phase K.5 — surgical "important note" extraction.
                # Cost-bounded (timeout + tier gate inside helper)
                # and only worth running on notable conversations,
                # which is exactly when we're already here.
                try:
                    note = await asyncio.wait_for(
                        extract_important_note(
                            npc_, insight, other_name_, self.llm,
                        ),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        "Important-note extraction timed out for %s",
                        npc_.name,
                    )
                    note = None
                except Exception:
                    logger.exception(
                        "Important-note extraction failed for %s",
                        npc_.name,
                    )
                    note = None
                if note is not None:
                    note_text, note_tags = note
                    # Anchor at least one tag to the partner so retrieval
                    # by partner-name surfaces this note.
                    partner_tag_set = set(note_tags) | {
                        t for t in (other_name_.lower().replace(" ", "_"),)
                        if t
                    }
                    self.memory.episodic.add_memory(
                        npc_id=npc_.npc_id,
                        description=note_text,
                        category="note",
                        importance=0.85,
                        game_time=current_minutes,
                        extra_metadata={
                            "kind": "important_note",
                            "source": "post_conversation_reflection",
                            "partner_name": other_name_,
                        },
                        tags=partner_tag_set,
                    )
                    logger.info(
                        "NOTE %s: '%s' [tags=%s]",
                        npc_.name, note_text, sorted(partner_tag_set),
                    )

            await asyncio.gather(
                _reflect_one(npc_a, npc_b.name, npc_b.npc_id),
                _reflect_one(npc_b, npc_a.name, npc_a.npc_id),
                return_exceptions=True,
            )

    async def _check_reflections(self, current_minutes: float) -> None:
        """Check if any NPCs need a full reflection cycle (router decides).

        When a reflection produces an action intent, injects a temporary
        schedule entry so the NPC acts on their insight.
        """
        for npc in self.npcs:
            if self._skip_autonomous(npc) or npc.cognition_tier >= 4:
                continue
            if not self.memory.should_reflect(npc.npc_id, current_minutes):
                continue
            rd = self.router.route(
                npc, "reflection",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            if rd.route == Route.LLM:
                insights, intents = await run_reflection_with_intents(
                    npc, self.memory, self.llm, current_minutes,
                )
                for intent in intents:
                    self._inject_reflection_entry(npc, intent)
                # Personality drift — emotional reflections nudge the
                # Big-5 vector. A reflection is by definition high-
                # importance, so we use a fixed 0.8 unless the memory
                # manager tags a stronger one later.
                for insight in insights:
                    apply_personality_drift(npc, insight, importance=0.8)

    def _inject_reflection_entry(
        self, npc: NPC, intent: ActionIntent,
    ) -> None:
        """Insert a temporary schedule entry from a reflection action intent.

        The entry is placed at schedule_index + 1 so the NPC transitions
        to it after their current action. Once it completes, the NPC
        resumes their normal schedule — no permanent modification.
        """
        if not npc.daily_schedule:
            return

        entry = ScheduleEntry(
            slot="reflection",
            activity=intent.activity,
            location=intent.location,
            priority=7,  # high priority — reflection-driven
            duration_minutes=intent.duration_minutes,
        )

        insert_pos = min(
            npc.schedule_index + 1, len(npc.daily_schedule),
        )
        npc.daily_schedule.insert(insert_pos, entry)

        logger.info(
            "INJECT %s: '%s' at %s (%d min) inserted at index %d",
            npc.name, intent.activity, intent.location,
            intent.duration_minutes, insert_pos,
        )

    # How much of the gap to a contradicting belief must be crossed
    # before we refuse to apply a new identity claim. 0.5 means "if
    # the listener is already ≥0.5 sure of the opposite, reject".
    _IDENTITY_CONTRADICTION_FLOOR: float = 0.5

    # Map claim prefixes to the prefixes that would directly contradict
    # them. Used by _inject_self_concept_delta so that, for instance,
    # a strong friend_of:X belief dampens or rejects an enemy_of:X
    # claim arriving later in conversation.
    _CONTRADICTING_PREFIXES: dict[str, tuple[str, ...]] = {
        "friend_of": ("enemy_of", "rival_of", "betrayed"),
        "enemy_of": ("friend_of",),
        "rival_of": ("friend_of",),
        "helped": ("betrayed",),
        "betrayed": ("helped", "saved"),
        "saved": ("betrayed",),
    }

    def _apply_accusation_sentiment(
        self,
        outcome: Any,
        participants: dict[str, str],
        current_minutes: float,
    ) -> int:
        """Apply sentiment penalties for direct accusations.

        Emergent-write-paths arc: accusations were already extracted
        and stored but never touched the relationship table — an NPC
        could be accused of theft and walk away trusting the accuser
        MORE (via the contact baseline). Applies only when accuser
        and accused are both conversation participants; third-party
        (relayed) accusations stay memory-only for now. Deltas are
        one-directional per ACCUSATION_SENTIMENT_DELTAS. Returns the
        number of accusations applied.
        """
        accusations = getattr(outcome, "accusations", None) or []
        if not accusations:
            return 0
        name_to_id = {
            name.strip().lower(): npc_id
            for npc_id, name in participants.items()
        }
        applied = 0
        for acc in accusations:
            accuser_id = name_to_id.get((acc.accuser or "").strip().lower())
            accused_id = name_to_id.get((acc.accused or "").strip().lower())
            if not accuser_id or not accused_id or accuser_id == accused_id:
                continue
            for dim, delta in ACCUSATION_SENTIMENT_DELTAS[
                "accused_toward_accuser"
            ].items():
                self.sentiment.modify(
                    accused_id, accuser_id, dim, delta,
                    game_time=current_minutes,
                )
            for dim, delta in ACCUSATION_SENTIMENT_DELTAS[
                "accuser_toward_accused"
            ].items():
                self.sentiment.modify(
                    accuser_id, accused_id, dim, delta,
                    game_time=current_minutes,
                )
            applied += 1
            logger.info(
                "ACCUSATION_SENTIMENT %s accused %s: '%s'",
                acc.accuser, acc.accused, acc.claim[:60],
            )
        return applied

    def _inject_self_concept_delta(
        self, npc: NPC, claim: IdentityClaim,
        current_minutes: float,
    ) -> bool:
        """Apply an identity claim to the NPC's self_concept.

        Returns True if the claim was applied (in any amount), False if
        it was rejected for contradicting an existing strong belief.
        A faint contradicting memory dampens the delta rather than
        rejecting outright — identity is allowed to drift, just slowly.
        """
        prefix, _, target = claim.key.partition(":")

        # Check for a direct contradiction: do we already strongly
        # believe something opposite? e.g. friend_of:bran vs enemy_of:bran.
        contradict_prefixes = self._CONTRADICTING_PREFIXES.get(prefix, ())
        strongest_opposing = 0.0
        for other_key, other_conf in npc.self_concept.items():
            other_prefix, _, other_target = other_key.partition(":")
            if other_target != target:
                continue
            if other_prefix in contradict_prefixes:
                strongest_opposing = max(strongest_opposing, other_conf)

        if strongest_opposing >= self._IDENTITY_CONTRADICTION_FLOOR:
            logger.info(
                "IDENTITY %s: rejected '%s' (contradicts existing "
                "belief, strength=%.2f)",
                npc.name, claim.key, strongest_opposing,
            )
            return False

        # Dampen the delta by any weaker contradiction so the belief
        # shifts gradually instead of flipping on one line.
        damping = max(0.0, 1.0 - 2 * strongest_opposing)
        effective_delta = claim.confidence_delta * damping

        new_confidence = npc.apply_self_concept_delta(
            claim.key, effective_delta,
        )
        logger.info(
            "IDENTITY %s: '%s' +%.2f → %.2f (from: %r)",
            npc.name, claim.key, effective_delta, new_confidence,
            claim.source_text,
        )

        # Close the identity → drive loop: refresh derived long-term
        # goals so the planner's utility scorer starts biasing toward
        # the new identity on the next tick. A claim that fails to
        # clear its goal-floor is a no-op; a rising belief past the
        # floor adds a goal; a decaying belief below the floor
        # removes one.
        added, removed = sync_npc_goals(npc)
        if added or removed:
            logger.info(
                "GOALS %s: +%s / -%s (from self_concept)",
                npc.name, added, removed,
            )
        return True

    # Per-day fraction of the gap between current personality and
    # spawn baseline that decays back. Small enough that drift from
    # strong events still shows up for many days; large enough to
    # keep an NPC's character recognisable over a long run.
    _PERSONALITY_DECAY_RATE: float = 0.02

    def _normalise_schedule_cursors(self) -> None:
        """Repair invalid (schedule_index, daily_schedule) combinations.

        Three out-of-range states can arise from exceptions during
        `_advance_npc_action`, from LLM/planner regen that leaves the
        schedule empty, or from a replan that truncates below the
        current cursor:

        - daily_schedule == [] but schedule_index != 0: no entry to
          point at; reset cursor so regeneration restarts cleanly.
        - schedule_index >= len(daily_schedule) on a non-empty schedule:
          the cursor has walked off the end; wrap to 0 and force a
          fresh anchor.
        - action_start_minutes is non-zero on an empty schedule: would
          make the next first-dispatch path skip its anchor step.

        Fires each cognition tick before schedule generation (step 2).
        """
        for npc in self.npcs:
            if self._skip_autonomous(npc):
                continue
            if not npc.daily_schedule:
                if npc.schedule_index != 0 or npc.action_start_minutes != 0.0:
                    logger.info(
                        "SCHED-FIX %s: empty schedule with idx=%d start=%.1f "
                        "— resetting",
                        npc.name, npc.schedule_index, npc.action_start_minutes,
                    )
                    npc.schedule_index = 0
                    npc.action_start_minutes = 0.0
            elif npc.schedule_index >= len(npc.daily_schedule):
                logger.info(
                    "SCHED-FIX %s: idx=%d out of range (len=%d) — wrapping",
                    npc.name, npc.schedule_index, len(npc.daily_schedule),
                )
                npc.schedule_index = 0
                npc.action_start_minutes = 0.0

    def _decay_personalities(self) -> None:
        """Drift every NPC's personality back toward their spawn baseline.

        Runs once per game day alongside sentiment decay. NPCs with no
        baseline (e.g. custom-seeded test NPCs) are skipped silently.
        """
        for npc in self.npcs:
            if npc.spawn_baseline is None:
                continue
            npc.personality.nudge_toward(
                npc.spawn_baseline, self._PERSONALITY_DECAY_RATE,
            )

    # Maximum Manhattan distance an NPC may be from their nearest
    # landmark (home, work, or current schedule target) before the
    # stray-catcher teleports them back to their home. Generous enough
    # to allow long walks across the map; tight enough that ending up
    # at the grid border is caught.
    _STRAY_DISTANCE_LIMIT = 25

    async def _enforce_bedtime(
        self, current_slot: str, current_minutes: float,
    ) -> None:
        """Force every NPC toward their sleep-at-home entry at night.

        Walks through each NPC. If they are in conversation, end it
        (sleeping NPCs can't chat). If their current schedule entry
        isn't a home-sleep entry and they're not heading home, jump
        their schedule_index to the last entry (which is guaranteed
        to be a sleep-home by the parser/template invariant) and
        dispatch. This makes "NPCs are home at night" a property of
        the simulation rather than an emergent accident.
        """
        for npc in self.npcs:
            if self._skip_autonomous(npc) or npc.cognition_tier >= 4:
                continue
            if not npc.daily_schedule:
                continue

            idx = npc.schedule_index
            if 0 <= idx < len(npc.daily_schedule):
                entry = npc.daily_schedule[idx]
                is_sleep_home = (
                    entry.location == "home"
                    and ("sleep" in entry.activity.lower()
                         or "walk home" in entry.activity.lower())
                )
            else:
                is_sleep_home = False

            if is_sleep_home:
                continue  # already going or going home to sleep

            # Phase 4 — bedtime-safe crediting. A 240-min goal entry rarely
            # elapses before night, so the old "credit only on full-duration
            # finish" rule lost ~all contributions to the bedtime jump. If
            # the NPC REACHED its goal entry (is at the goal location) and
            # is now being sent home, it performed the goal today — credit
            # it. NPCs still en route (not near the target) are not credited.
            if (0 <= idx < len(npc.daily_schedule)
                    and entry.goal_id and entry.target_x is not None
                    and abs(npc.x - entry.target_x)
                        + abs(npc.z - entry.target_z) <= 3):
                self._credit_goal_entry(npc, entry.goal_id, npc.schedule_day)

            # End any conversation so the NPC can leave.
            if npc.conversation_partner:
                other = self.get_npc(npc.conversation_partner)
                if other:
                    await end_conversation(
                        npc, other, memory_manager=self.memory,
                    )
                npc.needs_post_convo_dispatch = False

            # Jump to the last entry — by invariant this is sleep-at-home.
            npc.schedule_index = max(0, len(npc.daily_schedule) - 1)
            npc.action_start_minutes = current_minutes
            entry = npc.daily_schedule[npc.schedule_index]
            await self._dispatch_to_entry(npc, entry, current_slot)

    def _reanchor_strays(self) -> None:
        """Teleport any NPC that's drifted past _STRAY_DISTANCE_LIMIT
        from every landmark they care about back to their home.

        Walking NPCs are exempt — they may legitimately be partway
        across the map. This only fires on resting (non-walking)
        NPCs, who have no reason to be 30 tiles from home.
        """
        for npc in self.npcs:
            if npc.npc_id == "player":
                continue
            if npc.activity == ActivityState.WALKING and npc.current_path:
                continue
            anchors: list[tuple[int, int]] = [
                (npc.home_x, npc.home_z),
            ]
            if getattr(npc, "work_x", None) is not None:
                anchors.append((npc.work_x, npc.work_z))
            # Current schedule target (if any)
            if npc.daily_schedule and 0 <= npc.schedule_index < len(npc.daily_schedule):
                entry = npc.daily_schedule[npc.schedule_index]
                if entry.target_x is not None and entry.target_z is not None:
                    anchors.append((entry.target_x, entry.target_z))

            nearest = min(
                abs(npc.x - ax) + abs(npc.z - az) for ax, az in anchors
            )
            if nearest <= self._STRAY_DISTANCE_LIMIT:
                continue

            logger.warning(
                "STRAY %s at (%.0f,%.0f) — %d tiles from nearest anchor, "
                "teleporting home to (%d,%d)",
                npc.name, npc.x, npc.z, nearest, npc.home_x, npc.home_z,
            )
            npc.x = float(npc.home_x)
            npc.z = float(npc.home_z)
            npc.current_path = []
            npc.path_index = 0
            npc._tick_trail = [(npc.home_x, npc.home_z)]

    def _ensure_commitment(self, npc: NPC, goal, current_day: int) -> None:
        """Foundation rebuild (Phase 2): record a DURABLE commitment when
        an NPC takes a town goal on.

        The commitment is the source of truth — it survives schedule
        regeneration and mid-day replanning, unlike the old injected
        schedule entry which replanning silently wiped. Phase 3 derives
        the schedule from these; Phase 4 credits on fulfilment. Idempotent:
        an NPC holds at most one live commitment per goal.
        """
        for c in npc.commitments:
            if c.goal_id == goal.goal_id and c.status in (
                CommitmentStatus.PENDING, CommitmentStatus.ACTIVE,
            ):
                return
        npc.commitments.append(Commitment(
            goal_id=goal.goal_id,
            activity=goal.activity_text,
            location=goal.location_hint,
            deadline_day=goal.deadline_day,
            duration_minutes=goal.duration_minutes,
            town_id=getattr(goal, "town_id", None),
            status=CommitmentStatus.PENDING,
            created_day=current_day,
        ))

    def _resolve_commitments(self, goal) -> None:
        """Prune every NPC's live commitment to a now-finished goal
        (completed or expired). Keeps `commitments` bounded to live goals
        only, so the list can't accumulate at population scale. Phase 4
        marks FULFILLED at contribution time before this prune runs; here
        a goal that's wrapped up simply has no live commitment left."""
        for npc in self.npcs:
            if not npc.commitments:
                continue
            npc.commitments = [
                c for c in npc.commitments if c.goal_id != goal.goal_id
            ]

    def _project_commitments(self, npc: NPC) -> None:
        """Re-derive goal entries from durable commitments (Phase 3).

        Idempotent safety net: for every live commitment whose goal entry
        is missing from the *reachable* part of the schedule (e.g. a
        mid-day replan replaced it, or the cursor walked past it),
        commandeer a future slot so it reappears. Because the schedule is
        a projection of commitments, replanning can no longer permanently
        wipe a goal — it returns before the NPC's next action advance.
        No-op for NPCs without live commitments, so it is cheap to run
        every tick at population scale.
        """
        if not npc.commitments or not npc.daily_schedule:
            return
        idx = max(0, npc.schedule_index)
        for c in npc.commitments:
            if c.status not in (
                CommitmentStatus.PENDING, CommitmentStatus.ACTIVE,
            ):
                continue
            if any(e.goal_id == c.goal_id for e in npc.daily_schedule[idx:]):
                c.status = CommitmentStatus.ACTIVE
                continue
            target = self._projectable_slot(npc, idx)
            if target is None:
                continue
            original = npc.daily_schedule[target]
            npc.daily_schedule[target] = ScheduleEntry(
                slot=original.slot,
                activity=c.activity,
                location=c.location,
                priority=8,
                duration_minutes=min(
                    c.duration_minutes,
                    original.duration_minutes or c.duration_minutes,
                ),
                goal_id=c.goal_id,
            )
            c.status = CommitmentStatus.ACTIVE

    def _credit_goal_entry(
        self, npc: NPC, goal_id: str, current_day: int,
    ) -> bool:
        """Credit one contribution for a goal the NPC has performed, mark
        its commitment FULFILLED, and fire exactly once (Phase 4).

        Idempotent on two levels: the agenda dedups per-NPC via the goal's
        `contributors` set, and the FULFILLED status stops the commitment
        re-projecting. Returns whether this contribution completed the goal.
        """
        for c in npc.commitments:
            if c.goal_id == goal_id and c.status != CommitmentStatus.FULFILLED:
                c.status = CommitmentStatus.FULFILLED
        try:
            return bool(self.town_agenda.record_contribution(
                goal_id, npc.npc_id, current_day=current_day,
            ))
        except Exception:
            logger.exception(
                "AGENDA contribution failed for %s goal=%s", npc.name, goal_id,
            )
            return False

    def _projectable_slot(self, npc: NPC, from_idx: int) -> int | None:
        """Pick a reachable slot (index >= from_idx) to host a goal entry:
        prefer afternoon, then evening, then morning; never the final
        sleep-home entry, never one already hosting a goal."""
        last_idx = len(npc.daily_schedule) - 1
        for slot in ("afternoon", "evening", "morning"):
            for i in range(from_idx, len(npc.daily_schedule)):
                if i == last_idx:
                    continue  # preserve the sleep-home entry
                e = npc.daily_schedule[i]
                if e.slot == slot and e.goal_id is None:
                    return i
        return None

    def _inject_goal_entry(self, npc: NPC, current_day: int) -> None:
        """If the NPC matches an active town goal, inject the goal entry.

        Places the entry in the afternoon slot (where possible) so
        breakfast and morning work still happen. Replaces an existing
        afternoon slot rather than appending, to keep schedule total
        duration intact. The contribution is NOT recorded here any
        more — eager crediting at injection caused every goal with
        `required_contributions == N` to complete on the same tick
        it was proposed, with N personality-matching NPCs listed as
        contributors who never actually moved. Credit now lands in
        `_advance_npc_action` when the NPC finishes the goal entry,
        so the agenda is tied to time-in-slot rather than intent.
        The goal's completion triggers a town-event broadcast.

        Phase 2 note: this still mutates the schedule (the fragile path
        Phase 3 replaces), but now ALSO records a durable Commitment via
        `_ensure_commitment` — the source of truth going forward.
        """
        goal = self.town_agenda.matching_goal_for(npc, self.rng)
        if goal is None or not npc.daily_schedule:
            return

        # Durable commitment first (survives replan); schedule projection
        # below is the to-be-replaced fragile mechanism.
        self._ensure_commitment(npc, goal, current_day)

        # Find an afternoon or evening slot to commandeer. If none,
        # bail out rather than break the schedule.
        preferred_slots = ("afternoon", "evening", "morning")
        target_idx = None
        for slot in preferred_slots:
            for i, e in enumerate(npc.daily_schedule):
                if e.slot == slot:
                    target_idx = i
                    break
            if target_idx is not None:
                break
        if target_idx is None:
            return

        original = npc.daily_schedule[target_idx]
        npc.daily_schedule[target_idx] = ScheduleEntry(
            slot=original.slot,
            activity=goal.activity_text,
            location=goal.location_hint,
            priority=8,
            duration_minutes=min(goal.duration_minutes, original.duration_minutes or goal.duration_minutes),
        )
        # Name the goal on the entry (declared field) so the execution
        # layer can credit the NPC when this entry finishes.
        npc.daily_schedule[target_idx].goal_id = goal.goal_id

        logger.info(
            "AGENDA %s: committed to '%s' (goal=%s, deadline day %d)",
            npc.name, goal.title, goal.goal_id, goal.deadline_day,
        )

        # Phase F.2 — write a personal commitment memory. Fires before
        # the completion listener so even for a goal that completes
        # with this NPC's contribution, the commitment memory still
        # lands first and the completion memory layers on top.
        try:
            self.memory.record_town_event_memory(
                npc_id=npc.npc_id,
                description=(
                    f"I have agreed to {goal.activity_text} "
                    f"(\"{goal.title}\") before day {goal.deadline_day}."
                ),
                category="commitment",
                importance=0.7,
                game_time=self._current_minutes,
                goal_id=goal.goal_id,
            )
        except Exception:
            logger.exception(
                "Failed to seed commitment memory for %s on %s",
                npc.name, goal.goal_id,
            )

    async def _check_replans(
        self, current_minutes: float, current_slot: str,
    ) -> None:
        """Mid-day replanning for Tier 1-2 NPCs.

        Asks the LLM whether the NPC wants to modify their remaining
        schedule based on recent perceptions and reflections. If the
        LLM returns new entries, the remaining schedule is replaced
        and the NPC is re-dispatched to the new current entry.
        """
        for npc in self.npcs:
            if self._skip_autonomous(npc):
                continue
            if not should_replan(npc, current_minutes):
                continue

            rd = self.router.route(
                npc, "daily_schedule",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            if rd.route != Route.LLM:
                npc.last_replan_minutes = current_minutes
                continue

            # Gather context for the replan prompt
            recent_episodic = self.memory.episodic.get_recent(
                npc.npc_id, limit=5,
            )
            perceptions = [m.description for m in recent_episodic
                           if m.category in ("observation", "conversation")]
            reflections = [m.description for m in recent_episodic
                           if m.category == "reflection"]
            rel_summary = self._build_relationship_summary(npc)

            changed = await replan_schedule(
                npc, self.llm, current_minutes,
                recent_perceptions=perceptions,
                recent_reflections=reflections,
                relationship_summary=rel_summary,
            )

            if changed:
                # Re-dispatch to the current entry under the new schedule
                if npc.schedule_index < len(npc.daily_schedule):
                    entry = npc.daily_schedule[npc.schedule_index]
                    await self._dispatch_to_entry(npc, entry, current_slot)

    async def _run_daily_compaction(self, day_to_compact: int) -> None:
        """Phase H.6 — collapse yesterday's untagged memories.

        Called once per NPC per game day, on the first tick of a
        new day (so `day_to_compact` is always `current_day - 1`
        and its memories are finalised). Routes each call through
        `CognitionRouter.route(npc, "compaction")` — LLM verdict
        yields a rich first-person summary, deterministic verdict
        falls through to the heuristic string in
        `core.memory.compaction`. One call max per (npc, day);
        re-runs no-op thanks to tombstone filtering.

        Week rollup hitches here too: when `day_to_compact` is the
        last day of a week (day % 7 == 6), the method also rolls
        up that week's 7 day_summaries into a week_summary. The
        router is consulted separately for the week call so the
        budget accounts for both.
        """
        if day_to_compact < 0:
            return

        from core.memory.compaction import DAYS_PER_WEEK
        from core.npc.cognition.router import Route

        for npc in self.npcs:
            if self._skip_autonomous(npc):
                continue
            # Skip NPCs whose cognition is entirely frozen this tick —
            # tier 4 means they're far from focus and we don't pay any
            # per-tick cost on them; compaction follows the same rule.
            if getattr(npc, "cognition_tier", 1) >= 4:
                continue
            if self._last_compacted_day.get(npc.npc_id, -1) >= day_to_compact:
                continue

            day_decision = self.router.route(
                npc, "compaction",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            day_llm = self.llm if day_decision.route == Route.LLM else None
            try:
                await self.memory.compact_day(
                    npc.npc_id, day_to_compact,
                    npc=npc, llm=day_llm,
                )
            except Exception:
                logger.exception(
                    "compact_day failed for %s day %d",
                    npc.npc_id, day_to_compact,
                )
            self._last_compacted_day[npc.npc_id] = day_to_compact

            # Week rollup when this is the tail day of a week (0-6,
            # 7-13, …). Index arithmetic: day % 7 == 6 → week ended.
            if day_to_compact % DAYS_PER_WEEK == DAYS_PER_WEEK - 1:
                week_to_compact = day_to_compact // DAYS_PER_WEEK
                if self._last_compacted_week.get(
                    npc.npc_id, -1,
                ) >= week_to_compact:
                    continue
                week_decision = self.router.route(
                    npc, "compaction",
                    focus_x=self.focus_x, focus_z=self.focus_z,
                )
                week_llm = (
                    self.llm if week_decision.route == Route.LLM else None
                )
                try:
                    await self.memory.compact_week(
                        npc.npc_id, week_to_compact,
                        npc=npc, llm=week_llm,
                    )
                except Exception:
                    logger.exception(
                        "compact_week failed for %s week %d",
                        npc.npc_id, week_to_compact,
                    )
                self._last_compacted_week[npc.npc_id] = week_to_compact

    async def _run_daily_self_review(self, day_to_review: int) -> None:
        """Phase I.1 — bedtime commitment review.

        Called once per NPC per game day, immediately after
        `_run_daily_compaction` on the first tick of a new day. For
        each autonomous non-frozen NPC:

        - Skip if the cursor says we've already reviewed this day.
        - Ask the router for a routing verdict on `self_review`. The
          default policy routes it to LLM unconditionally; users who
          swap in a budget-tight policy will land back in AUTO or
          DETERMINISTIC and fall through to the heuristic summary.
        - Run `memory.daily_self_review`; on success, inject any
          returned `ActionIntent` into tomorrow's schedule via the
          same path reflection-driven intents already use.

        Idempotent — the cursor guard makes re-runs a no-op.
        """
        if day_to_review < 0:
            return

        from core.npc.cognition.router import Route

        for npc in self.npcs:
            if self._skip_autonomous(npc):
                continue
            if getattr(npc, "cognition_tier", 1) >= 4:
                continue
            if self._last_self_reviewed_day.get(
                npc.npc_id, -1,
            ) >= day_to_review:
                continue

            decision = self.router.route(
                npc, "self_review",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            review_llm = self.llm if decision.route == Route.LLM else None
            try:
                result = await self.memory.daily_self_review(
                    npc.npc_id, day_to_review,
                    npc=npc, llm=review_llm,
                )
            except Exception:
                logger.exception(
                    "daily_self_review failed for %s day %d",
                    npc.npc_id, day_to_review,
                )
                result = None

            self._last_self_reviewed_day[npc.npc_id] = day_to_review

            if result is not None and result.action_intent is not None:
                self._inject_reflection_entry(npc, result.action_intent)

    async def _run_overseer_eval(
        self, current_day: int, current_minutes: float,
    ) -> None:
        """Run daily overseer evaluation: score, detect, intervene."""
        try:
            report = await self.overseer.evaluate(
                self.npcs, current_day,
                sentiment=self.sentiment,
                memory=self.memory,
            )

            # Expire old policies and modifiers
            self.mechanisms.expire_old(current_day)

            # Filter interventions through guardrails
            allowed = self.guardrails.filter_interventions(
                report.interventions, self.npcs, current_day,
            )

            # Apply allowed interventions
            for intervention in allowed:
                if self.mechanisms.apply_intervention(
                    intervention, self.npcs, current_day,
                ):
                    self.guardrails.record_applied(current_day)

            # Clamp any out-of-bounds NPC parameters
            self.guardrails.enforce_bounds(self.npcs)

            if allowed:
                logger.info(
                    "Overseer day %d: %d interventions applied (of %d proposed)",
                    current_day, len(allowed), len(report.interventions),
                )
        except Exception as e:
            logger.warning("Overseer evaluation failed on day %d: %s", current_day, e)

        # Town-level goal proposal. Rule-driven for now — the overseer
        # watches population state and adds goals that match the mood.
        # Future work: have the overseer LLM propose goals directly.
        try:
            self._propose_town_goals(current_day)
            expired = self.town_agenda.expire_overdue(current_day)
            for goal in expired:
                logger.info("Town goal expired: %s", goal.title)
        except Exception:
            logger.exception("Town agenda update failed")

    def _propose_town_goals(self, current_day: int) -> None:
        """Rule-based goal proposer. Cheap, deterministic, extensible.

        Each rule inspects the population and — if its trigger matches —
        asks the agenda to propose a goal. Cooldowns are enforced by
        TownAgenda so the same goal doesn't re-trigger every day.
        """
        # 1. Every 6th day-ish: hold a harvest festival. This is the
        #    'heartbeat' social event that gives the town a recurring
        #    collective activity.
        if current_day >= 2 and current_day % 6 in (0, 1):
            goal = create_goal_from_template("harvest_festival", current_day)
            if goal:
                self.town_agenda.propose(goal, current_day)

        # 2. When sentiment has drifted negative on average, call a
        #    town council so the NPCs convene and work it out.
        avg_disposition = self._average_disposition()
        if avg_disposition < -8.0:
            goal = create_goal_from_template("town_council", current_day)
            if goal:
                self.town_agenda.propose(goal, current_day)

        # 3. Every ~10 days, schedule bridge repair (placeholder for
        #    construction/damage integration).
        if current_day >= 3 and current_day % 10 == 3:
            goal = create_goal_from_template("repair_bridge", current_day)
            if goal:
                self.town_agenda.propose(goal, current_day)

    def _average_disposition(self) -> float:
        """Average dispositional sentiment across all tracked pairs.

        Walks every NPC's outgoing relationships. Cheap enough at
        our population scale; if the town grows, swap in a cached
        aggregate computed once per tick.
        """
        scores: list[float] = []
        for npc in self.npcs:
            if npc.npc_id == "player":
                continue
            try:
                rels = self.sentiment.get_all_for(npc.npc_id)
            except Exception:
                rels = []
            for r in rels:
                scores.append(r.overall_disposition())
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _on_goal_proposed(self, goal) -> None:
        """Phase F.1 — seed every NPC with awareness of a new town goal.

        The sidebar already shows the goal; now each NPC also holds
        an episodic memory of having heard about it, so retrieval and
        prompt injection can surface it in plan/converse/reflect.
        Importance 0.6 lands above perception noise but below personal
        commitments (0.7) and completion events (0.8).
        """
        description = (
            f"The town has proposed a new initiative: \"{goal.title}\" — "
            f"{goal.description} Needs {goal.required_contributions} "
            f"contributions by day {goal.deadline_day}."
        )
        for npc in self.npcs:
            try:
                self.memory.record_town_event_memory(
                    npc_id=npc.npc_id,
                    description=description,
                    category="town_agenda",
                    importance=0.6,
                    game_time=self._current_minutes,
                    goal_id=goal.goal_id,
                )
            except Exception:
                logger.exception(
                    "Failed to seed propose memory for %s on %s",
                    npc.name, goal.goal_id,
                )
        logger.info(
            "AGENDA propose: '%s' seeded to %d NPCs",
            goal.title, len(self.npcs),
        )

    def _on_goal_expired(self, goal) -> None:
        """Phase F.4 — record the town's disappointment when a goal
        lapses without completing. Lands for every NPC so the memory
        of a failed initiative colours subsequent planning/chat.
        """
        # Phase 2: the goal lapsed — clear any live commitments to it.
        self._resolve_commitments(goal)
        description = (
            f"The town initiative \"{goal.title}\" was not completed in "
            f"time — {goal.progress}/{goal.required_contributions} "
            f"contributions by day {goal.deadline_day}. "
            f"It has been shelved."
        )
        for npc in self.npcs:
            try:
                self.memory.record_town_event_memory(
                    npc_id=npc.npc_id,
                    description=description,
                    category="town_failure",
                    importance=0.6,
                    game_time=self._current_minutes,
                    goal_id=goal.goal_id,
                )
            except Exception:
                logger.exception(
                    "Failed to seed expire memory for %s on %s",
                    npc.name, goal.goal_id,
                )
        logger.info("AGENDA expire: '%s' recorded as failure", goal.title)

    def _on_goal_completed(self, goal) -> None:
        """Callback: record a town event when a goal completes.

        Fires a data-driven event through the existing impact system
        so sentiment and narrative ripples naturally, AND seeds
        per-NPC episodic memory (Phase F.3) so contributors remember
        the shared victory and bystanders remember the news.
        """
        # Phase 2: the goal is done — clear any live commitments to it.
        self._resolve_commitments(goal)
        try:
            self.events.process_event(GameEvent(
                event_type="town_goal_completed",
                participants=list(goal.contributors),
                data={"goal_id": goal.goal_id, "title": goal.title},
                game_time=self._current_minutes,
                location_x=0, location_z=0,
            ))
            logger.info(
                "Town goal completed: %s (%d contributors)",
                goal.title, len(goal.contributors),
            )
        except Exception:
            logger.exception("Failed to fire town_goal_completed event")

        # Build contributor name list so memories read naturally
        # ("with Alice, Bran, and Petra") instead of by npc_id.
        contributor_names = []
        for npc_id in goal.contributors:
            n = self.get_npc(npc_id)
            if n is not None:
                contributor_names.append(n.name)
        contributors_phrase = (
            ", ".join(contributor_names) if contributor_names else "others"
        )

        contributor_description = (
            f"We completed the town initiative \"{goal.title}\" together "
            f"with {contributors_phrase}."
        )
        bystander_description = (
            f"The town initiative \"{goal.title}\" was completed by "
            f"{contributors_phrase}."
        )

        for npc in self.npcs:
            was_contributor = npc.npc_id in goal.contributors
            try:
                self.memory.record_town_event_memory(
                    npc_id=npc.npc_id,
                    description=(
                        contributor_description if was_contributor
                        else bystander_description
                    ),
                    category="town_event",
                    importance=0.8 if was_contributor else 0.5,
                    game_time=self._current_minutes,
                    goal_id=goal.goal_id,
                )
            except Exception:
                logger.exception(
                    "Failed to seed completion memory for %s on %s",
                    npc.name, goal.goal_id,
                )

            # Phase I.4 — reinforce contributor self_concept. Bystanders
            # get the news (town_event memory above) but not the
            # identity bump; the delta is earned by showing up.
            if was_contributor:
                try:
                    apply_identity_reinforcement(
                        self.memory, npc, goal, self._current_minutes,
                    )
                except Exception:
                    logger.exception(
                        "Failed to apply identity reinforcement for %s on %s",
                        npc.name, goal.goal_id,
                    )

    def _build_tick_state(self) -> dict[str, Any]:
        """Build the NPC state update for WebSocket broadcast."""
        conversations = [
            c.to_dict() for c in get_active_conversations()
        ]

        return {
            "npcs": [npc.to_dict() for npc in self.npcs],
            "conversations": conversations,
            "world_state": self.events.get_world_state(),
            "economy": self.economy.get_state(),
            "town_agenda": self.town_agenda.to_dict(),
        }

    def _generate_deterministic_schedule(
        self, npc: NPC, current_slot: str, current_day: int,
    ) -> None:
        """Deterministic DAILY schedule = the sound occupation template.

        (Phase 3.5) `planner.plan_action` returns a single tactical
        action, not a coherent day — routing daily-schedule generation
        through it left NPCs with a 1-entry schedule, no afternoon slot
        to host a town-goal entry, and `projected=0`. The occupation
        template is the validated well-formed full-day source (7 entries,
        real durations, sleep at night), so route through it. The planner
        remains available for moment-to-moment action selection elsewhere.
        """
        _force_template_schedule(npc, current_day)

    async def _decompose_llm(self, npc: NPC, entry: ScheduleEntry) -> None:
        """Decompose a schedule entry using LLM with memory context."""
        memory_context = ""
        try:
            ctx = self.memory.retrieve_context(
                npc_id=npc.npc_id,
                query=entry.activity,
                cognition_tier=npc.cognition_tier,
            )
            memory_context = ctx.to_prompt_text()
        except Exception:
            pass

        # Find building objects at the destination
        location_objects: list[str] = []
        for b in self.buildings:
            if b.building_type == entry.location or b.name == entry.location:
                location_objects = b.interior_objects
                break

        subtasks = await decompose_schedule_entry_llm(
            npc, entry, self.llm,
            memory_context=memory_context,
            location_objects=location_objects or None,
        )
        npc.subtask_queue = subtasks
        npc.current_subtask = None
        npc.subtask_time_remaining = 0.0

    # ---------- Relationship helpers ----------

    def _build_relationship_summary(self, npc: NPC) -> str:
        """Build a brief relationship summary for schedule planning prompts."""
        parts: list[str] = []

        # Top relationships by intensity
        top_rels = self.sentiment.get_strongest_relationships(npc.npc_id, limit=3)
        for sent in top_rels:
            other = self.get_npc(sent.npc_to)
            name = other.name if other else sent.npc_to
            parts.append(f"{name}: {sent.to_description()}")

        # Faction context
        faction_ctx = self.factions.get_social_context(npc.npc_id)
        if faction_ctx and "no faction" not in faction_ctx.lower():
            parts.append(faction_ctx)

        if not parts:
            return ""
        return "Relationships: " + "; ".join(parts)

    def fire_event(
        self,
        event_type: str,
        participants: list[str] | None = None,
        data: dict | None = None,
        game_time: float = 0.0,
        location_x: int = 0,
        location_z: int = 0,
    ) -> list[dict]:
        """Public interface to fire a game event through the impact system."""
        event = GameEvent(
            event_type=event_type,
            participants=participants or [],
            data=data or {},
            game_time=game_time,
            location_x=location_x,
            location_z=location_z,
        )
        return self.events.process_event(event)

    # ---------- Emergency movement overrides ----------

    def force_navigate_all(
        self, target_x: int, target_z: int,
        description: str = "emergency movement",
        flee_from: bool = False, filter_fn: callable | None = None,
    ) -> int:
        """Force all (or filtered) NPCs to navigate immediately."""
        from core.npc.emergency import force_navigate_all as _force_all
        return _force_all(
            self.npcs, self.grid, self._pending_departures, self.rng,
            self.get_npc, target_x, target_z, description, flee_from,
            filter_fn,
        )

    def force_navigate_npc(
        self, npc_id: str, target_x: int, target_z: int,
        description: str = "",
    ) -> bool:
        """Force a single NPC to navigate immediately, bypassing stagger."""
        from core.npc.emergency import force_navigate_npc as _force_npc
        return _force_npc(
            npc_id, self.npcs, self.grid, self._pending_departures,
            self.get_npc, target_x, target_z, description,
        )

    # ---------- Public queries ----------

    def set_focus(self, x: int, z: int) -> None:
        """Update the camera/player focus point for tier assignment."""
        self.focus_x = x
        self.focus_z = z

    def assign_custom_schedule(
        self, npc_id: str, entries: list[dict],
    ) -> tuple[bool, str]:
        """Assign a custom schedule to a specific NPC.

        entries format: [
            {"activity": "stand guard at the bridge", "location": "bridge",
             "target_x": 15, "target_z": 0, "duration_minutes": 900},
            {"activity": "walk home and sleep", "location": "home",
             "duration_minutes": 540},
        ]

        Entries must sum to 1440 minutes. Returns (success, message).
        """
        npc = self.get_npc(npc_id)
        if not npc:
            return False, f"NPC '{npc_id}' not found"

        if not entries:
            return False, "Schedule must have at least one entry"

        total = sum(e.get("duration_minutes", 0) for e in entries)
        if abs(total - MINUTES_PER_DAY) > 1:
            return False, (
                f"Schedule entries must sum to {MINUTES_PER_DAY} minutes, "
                f"got {total}"
            )

        schedule = []
        for i, e in enumerate(entries):
            if not e.get("activity"):
                return False, f"Entry {i} missing 'activity'"
            if not e.get("duration_minutes"):
                return False, f"Entry {i} missing 'duration_minutes'"
            schedule.append(ScheduleEntry(
                slot=f"custom_{i}",
                activity=e["activity"],
                location=e.get("location", ""),
                priority=e.get("priority", 5),
                target_x=e.get("target_x"),
                target_z=e.get("target_z"),
                duration_minutes=e["duration_minutes"],
            ))

        npc.daily_schedule = schedule
        npc.schedule_index = 0
        npc.action_start_minutes = self._current_minutes
        npc.has_custom_schedule = True
        logger.info(
            "%s: custom schedule assigned (%d entries)", npc.name, len(schedule),
        )
        return True, f"Custom schedule assigned to {npc.name}"

    def clear_custom_schedule(self, npc_id: str) -> tuple[bool, str]:
        """Remove a custom schedule, reverting to template-based scheduling."""
        npc = self.get_npc(npc_id)
        if not npc:
            return False, f"NPC '{npc_id}' not found"

        npc.has_custom_schedule = False
        _force_template_schedule(npc, npc.schedule_day)
        npc.schedule_index = 0
        npc.action_start_minutes = self._current_minutes
        logger.info("%s: custom schedule cleared, reverted to template", npc.name)
        return True, f"Custom schedule cleared for {npc.name}"

    def get_npcs_near(self, x: int, z: int, radius: int = 5) -> list[NPC]:
        """Get NPCs within Manhattan distance of a point."""
        return [n for n in self.npcs if n.distance_to(x, z) <= radius]

    def get_state(self) -> dict[str, Any]:
        """Full NPC population state for client init."""
        return {
            "npcs": [npc.to_dict() for npc in self.npcs],
            "conversations": [
                c.to_dict() for c in get_active_conversations()
            ],
            "factions": self.factions.get_all_factions(),
            "world_state": self.events.get_world_state(),
            "economy": self.economy.get_state(),
            "cognition": self.router.get_stats(),
        }
