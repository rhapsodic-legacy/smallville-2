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
    OCCUPATION_DEFAULTS, FIRST_NAMES, BACKSTORY_TEMPLATES,
)
from core.npc.seed_memories import seed_population_memories
from core.npc.llm_client import LLMProvider, MockProvider
from core.npc.cognition.tiers import (
    update_all_tiers, should_perceive, should_plan, get_tier_config,
)
from core.npc.cognition.perceive import perceive
from core.npc.cognition.plan import (
    generate_daily_schedule, resolve_schedule_location,
    _template_schedule,
)
from core.npc.cognition.execute import (
    execute_tick, set_activity_for_location, navigate_to,
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
from core.npc.economy_tick import EconomyTick
from core.memory.manager import MemoryManager
from core.memory.reflection import run_reflection, reflect_on_conversation
from core.relationships.sentiment import SentimentTracker
from core.relationships.structures import FactionManager
from core.events.impact import EventImpactSystem, GameEvent
from core.world.spatial_awareness import resolve_overlaps
from core.world.grid import Grid
from core.world.generator import PlacedBuilding
from core.time_system.clock import GameClock, MINUTES_PER_DAY

logger = logging.getLogger(__name__)


def _force_template_schedule(npc: NPC, current_day: int) -> None:
    """Assign a template schedule — no LLM, fully deterministic."""
    npc.daily_schedule = _template_schedule(npc)
    npc.schedule_day = current_day
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
        self.memory.initialise()

        # Wire sentiment into conversation module
        set_sentiment_tracker(self.sentiment)

        # Focus point for tier assignment (defaults to town centre)
        self.focus_x: int = 0
        self.focus_z: int = 0

        # Tick tracking
        self._last_slot: str = ""
        self._current_minutes: float = 0.0
        self._conversation_check_counter: int = 0
        self._reflection_check_counter: int = 0

        # Staggered departures: npc_id -> (delay_remaining_secs, target_x, target_z, description)
        # Delays are in REAL seconds (not game minutes) to guarantee visible spread
        self._pending_departures: dict[str, tuple[float, int, int, str]] = {}

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
        seed_population_memories(self.npcs, self.memory)

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

        npc = NPC(
            npc_id=npc_id,
            name=name,
            age=age,
            personality=personality,
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

        # Process pending departures (staggered in real seconds)
        if self._pending_departures:
            self._process_pending_departures(real_delta)

        # Execute movement and actions
        for npc in self.npcs:
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
        current_slot = clock.schedule_slot.value
        current_day = clock.day

        self._last_slot = current_slot

        # 1. Update tiers based on focus point
        update_all_tiers(self.npcs, self.focus_x, self.focus_z)

        # 2. Schedule generation (router decides LLM vs deterministic)
        schedule_tasks = []
        for npc in self.npcs:
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
                        )
                    )
                else:
                    self._generate_deterministic_schedule(
                        npc, current_slot, current_day,
                    )

        if schedule_tasks:
            import asyncio
            await asyncio.gather(*schedule_tasks)

        # 3. Stanford action-duration cycling — each NPC independently
        #    advances to the next schedule entry when their current
        #    action's duration expires. No global slot transitions.
        for npc in self.npcs:
            if npc.cognition_tier >= 4:
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
                if should_perceive(npc, current_minutes):
                    observations = perceive(npc, self.grid, self.npcs, current_minutes)

                    for obs in observations:
                        tile = self.grid.get_tile(obs.x, obs.z)
                        await self.memory.record_observation(
                            npc_id=npc.npc_id,
                            description=obs.description,
                            category=obs.category,
                            importance=obs.importance,
                            game_time=current_minutes,
                            location_x=obs.x,
                            location_z=obs.z,
                            tile_sector=tile.sector if tile else "",
                            tile_arena=tile.arena if tile else "",
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
        # Advance to next schedule entry
        npc.schedule_index += 1

        # Schedule exhausted — regenerate for new cycle
        if npc.schedule_index >= len(npc.daily_schedule):
            if self.deterministic:
                _force_template_schedule(npc, npc.schedule_day + 1)
            else:
                npc.daily_schedule = []  # triggers regeneration next tick
            npc.schedule_index = 0
            npc.action_start_minutes = current_minutes
            return

        entry = npc.daily_schedule[npc.schedule_index]
        npc.action_start_minutes = current_minutes

        # Sleep entries: force-end conversations so NPCs go home.
        # Stanford: sleep is just an action, but NPCs must actually
        # walk home — they can't chat all night at the tavern.
        is_sleep = entry.location == "home" and "sleep" in entry.activity.lower()
        if npc.conversation_partner and is_sleep:
            other = self.get_npc(npc.conversation_partner)
            if other:
                await end_conversation(npc, other)
                # end_conversation sets _needs_post_convo_dispatch on both,
                # but we handle the sleep NPC directly below — clear the flag
                # to prevent double-dispatch in movement_tick.
                npc._needs_post_convo_dispatch = False

        # NPCs in conversation (non-sleep): don't dispatch, pick up on end
        if npc.conversation_partner:
            return

        # Don't interrupt walking NPCs — they finish their path first
        if npc.activity == ActivityState.WALKING and npc.current_path:
            npc._needs_post_convo_dispatch = True
            return

        await self._dispatch_to_entry(npc, entry, current_slot)

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
            # No path — snap to destination (pathfinding failure)
            npc.x = float(target_x)
            npc.z = float(target_z)
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
            if not getattr(npc, '_needs_post_convo_dispatch', False):
                continue
            npc._needs_post_convo_dispatch = False

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
                        # No path — snap to destination rather than
                        # sleeping on the road mid-journey.
                        npc.x = float(tx)
                        npc.z = float(tz)
                        logger.info(
                            "DEPART %s: no path to (%d,%d), snapping",
                            npc.name, tx, tz,
                        )
                        set_activity_for_location(
                            npc, self._last_slot,
                        )
                completed.append(npc_id)
            else:
                self._pending_departures[npc_id] = (remaining, tx, tz, desc)

        for npc_id in completed:
            del self._pending_departures[npc_id]

    def _spread_destination(
        self, npc: NPC, target_x: int, target_z: int,
    ) -> tuple[int, int]:
        """
        If the target tile is already claimed by a resting NPC,
        find the nearest passable unoccupied neighbour instead.

        Door-aware: if the target is a building door, prefer the
        approach tile (one south of door) before spiralling randomly.
        This prevents NPCs entering buildings off-centre.
        """
        from core.world.spatial_awareness import get_occupied_tiles, find_rest_tile
        occupied = get_occupied_tiles(self.npcs)

        # Exclude this NPC from occupied set
        npc_pos = (npc.tile_x, npc.tile_z)
        if npc_pos in occupied:
            others_on_tile = any(
                o.npc_id != npc.npc_id
                and not o.activity == ActivityState.WALKING
                and o.tile_x == npc.tile_x and o.tile_z == npc.tile_z
                for o in self.npcs
            )
            if not others_on_tile:
                occupied = occupied - {npc_pos}

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

            # Determine who speaks next
            last_speaker = conv.exchanges[-1].speaker_id if conv.exchanges else None
            if last_speaker == npc_a.npc_id:
                await continue_conversation(
                    npc_b, npc_a, self.llm, memory_manager=self.memory,
                )
            else:
                await continue_conversation(
                    npc_a, npc_b, self.llm, memory_manager=self.memory,
                )

        # Check for new conversation opportunities (tier 1-3 only)
        for npc in self.npcs:
            if npc.conversation_partner or npc.cognition_tier >= 4:
                continue
            for other in self.npcs:
                if other.npc_id == npc.npc_id or other.conversation_partner:
                    continue
                if should_initiate_conversation(npc, other, current_minutes):
                    await initiate_conversation(
                        npc, other, self.llm, current_minutes,
                        memory_manager=self.memory,
                        grid=self.grid,
                        all_npcs=self.npcs,
                    )
                    break  # one conversation attempt per tick per NPC

    async def _persist_finished_conversations(
        self, current_minutes: float,
    ) -> None:
        """Save finished conversations to memory before they're cleaned up."""
        from core.npc.cognition.converse import _active_conversations

        for key, conv in _active_conversations.items():
            if not conv.finished or not conv.exchanges:
                continue

            npc_a = self.get_npc(conv.npc_a_id)
            npc_b = self.get_npc(conv.npc_b_id)
            if not npc_a or not npc_b:
                continue

            exchanges = [
                {"speaker": e.speaker_name, "message": e.message}
                for e in conv.exchanges
            ]

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

            # Fire conversation event through impact system
            self.events.process_event(GameEvent(
                event_type="conversation",
                participants=[conv.npc_a_id, conv.npc_b_id],
                game_time=current_minutes,
                location_x=npc_a.x,
                location_z=npc_a.z,
            ))

            # Post-conversation reflection (router decides)
            for npc, other_name in [
                (npc_a, npc_b.name), (npc_b, npc_a.name),
            ]:
                rd = self.router.route(
                    npc, "reflection",
                    focus_x=self.focus_x, focus_z=self.focus_z,
                )
                if rd.route == Route.LLM:
                    await reflect_on_conversation(
                        npc, other_name, exchanges,
                        self.memory, self.llm, current_minutes,
                    )

    async def _check_reflections(self, current_minutes: float) -> None:
        """Check if any NPCs need a full reflection cycle (router decides)."""
        for npc in self.npcs:
            if npc.cognition_tier >= 4:
                continue
            if not self.memory.should_reflect(npc.npc_id, current_minutes):
                continue
            rd = self.router.route(
                npc, "reflection",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            if rd.route == Route.LLM:
                await run_reflection(
                    npc, self.memory, self.llm, current_minutes,
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
        }

    def _generate_deterministic_schedule(
        self, npc: NPC, current_slot: str, current_day: int,
    ) -> None:
        """Use the planner to build a schedule (deterministic path)."""
        action = self.planner.plan_action(
            npc, self.npcs, current_slot,
            resource_nodes=self.economy.get_resource_node_dicts(),
            available_recipes=self.economy.get_available_recipes(),
            construction_sites=self.economy.get_construction_site_dicts(),
        )
        if action is None:
            return
        entry = action.to_schedule_entry(current_slot)
        if entry:
            npc.daily_schedule = [entry]
            npc.schedule_day = current_day
            # Pre-decompose into sub-tasks so NPC is never idle on arrival
            subtasks = decompose_schedule_entry(npc, entry, npc._rng)
            npc.subtask_queue = subtasks
            npc.current_subtask = None
            npc.subtask_time_remaining = 0.0

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
