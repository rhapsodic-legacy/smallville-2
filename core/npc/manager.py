"""
NPC Manager.

Orchestrates NPC spawning, the per-tick cognition cycle, and
schedule-driven movement. Acts as the main entry point for the
server to interact with the NPC population.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from core.npc.models import (
    NPC, ActivityState, PersonalityTraits, ScheduleEntry,
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
    ):
        self.grid = grid
        self.buildings = buildings
        self.llm = llm or MockProvider()
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
        self._conversation_check_counter: int = 0
        self._reflection_check_counter: int = 0

        # Staggered departures: npc_id -> (delay_remaining_secs, target_x, target_z, description)
        # Delays are in REAL seconds (not game minutes) to guarantee visible spread
        self._pending_departures: dict[str, tuple[float, int, int, str]] = {}

    def get_npc(self, npc_id: str) -> NPC | None:
        return self._npc_map.get(npc_id)

    # ---------- Spawning ----------

    def spawn_population(self, count: int) -> list[NPC]:
        """Create the initial NPC population."""
        occupations = self._assign_occupations(count)
        homes = [b for b in self.buildings if b.building_type == "home"]
        available_names = list(FIRST_NAMES)
        self.rng.shuffle(available_names)

        for i in range(count):
            occupation = occupations[i]
            name = available_names[i % len(available_names)]
            home = homes[i % len(homes)] if homes else None

            npc = self._create_npc(name, occupation, home, i)
            self.npcs.append(npc)
            self._npc_map[npc.npc_id] = npc

        # Seed foundational memories and goals
        seed_population_memories(self.npcs, self.memory)

        logger.info("Spawned %d NPCs (with seed memories)", len(self.npcs))
        return self.npcs

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
    ) -> NPC:
        """Create a single NPC with generated identity."""
        defaults = OCCUPATION_DEFAULTS.get(occupation, OCCUPATION_DEFAULTS["labourer"])

        # Find workplace
        work_building = self._find_work_building(defaults.get("work_building"))

        home_x = home.door_x if home else 0
        home_z = home.door_z if home else 0
        work_x = work_building.door_x if work_building else 0
        work_z = work_building.door_z if work_building else 0

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
            health=1.0,
            energy=self.rng.uniform(0.7, 1.0),
            hunger=self.rng.uniform(0.0, 0.2),
            long_term_goals=list(defaults.get("goals", [])),
            skills=dict(defaults.get("skills", {})),
            gold=self.rng.randint(10, 100),
            move_speed=self.rng.uniform(1.6, 2.4),
            _rng=random.Random(hash((self._seed, npc_id))),
        )

        return npc

    def _find_work_building(self, building_type: str | None) -> PlacedBuilding | None:
        """Find a building of the given type for NPC workplace."""
        if building_type is None:
            return None
        matches = [b for b in self.buildings if b.building_type == building_type]
        return self.rng.choice(matches) if matches else None

    # ---------- Tick cycle ----------

    async def tick(
        self,
        clock: GameClock,
        real_delta: float,
    ) -> dict[str, Any]:
        """Run one simulation tick for all NPCs."""
        current_minutes = clock.day * MINUTES_PER_DAY + clock.minutes
        current_slot = clock.schedule_slot.value
        current_day = clock.day
        game_minutes_elapsed = real_delta / clock._real_seconds_per_game_minute()

        # 1. Update tiers based on focus point
        update_all_tiers(self.npcs, self.focus_x, self.focus_z)

        # 2. Schedule generation (router decides LLM vs deterministic)
        schedule_tasks = []
        for npc in self.npcs:
            if npc.cognition_tier < 4 and npc.needs_new_schedule(current_day):
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

        # 3. Slot transition — NPCs queue staggered departures
        if current_slot != self._last_slot:
            self._last_slot = current_slot
            await self._handle_slot_transition(current_slot)

        # 3b. Process pending departures (staggered in real seconds)
        if self._pending_departures:
            self._process_pending_departures(real_delta)

        # 4. Perception cycle (tier-dependent intervals)
        for npc in self.npcs:
            if should_perceive(npc, current_minutes):
                observations = perceive(npc, self.grid, self.npcs, current_minutes)

                # Store observations in memory
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

                # React to interesting observations (router decides LLM vs deterministic)
                if observations:
                    obs = observations[0]  # most important only
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

        # 5. Execute movement and actions
        for npc in self.npcs:
            # Tier 4 (frozen) NPCs still finish paths already in progress,
            # but don't get new plans or need updates.
            if npc.cognition_tier >= 4 and not npc.current_path:
                continue
            execute_tick(
                npc, self.grid, self.buildings, current_slot,
                real_delta, all_npcs=self.npcs,
            )
            if npc.cognition_tier < 4:
                npc.tick_needs(game_minutes_elapsed)

        # 5a. Re-decompose NPCs whose subtask queues are empty.
        # Adds a random "idle pause" subtask (2-15 game minutes) at the
        # front of the new queue so NPCs don't all resume simultaneously.
        self._refill_empty_queues(current_slot)

        # 5b. Safety net: resolve any resting NPCs that share a tile
        resolve_overlaps(self.npcs, self.grid)

        # 5c. Economy tick (resource regen, gathering/crafting completion)
        self.economy.tick(self.npcs, game_minutes_elapsed, current_minutes)

        # 6. Conversation system (check periodically, not every tick)
        self._conversation_check_counter += 1
        if self._conversation_check_counter >= 3:  # every 3 ticks
            self._conversation_check_counter = 0
            await self._run_conversations(current_minutes)

        # 7. Persist finished conversations to memory, then clean up
        await self._persist_finished_conversations(current_minutes)
        clear_finished_conversations()

        # 8. Periodic reflection check (every ~30 ticks)
        self._reflection_check_counter += 1
        if self._reflection_check_counter >= 30:
            self._reflection_check_counter = 0
            await self._check_reflections(current_minutes)

        # 9. Final overlap safety net — catches overlaps created by
        # conversations (step 6) or any other post-movement repositioning.
        resolve_overlaps(self.npcs, self.grid)

        # Build state for broadcast
        return self._build_tick_state()

    async def _handle_slot_transition(self, new_slot: str) -> None:
        """
        Queue staggered departures on slot change. Router decides LLM vs template.
        """
        import asyncio
        llm_tasks: list[asyncio.Task] = []

        for npc in self.npcs:
            if npc.cognition_tier >= 4:
                continue
            in_conversation = bool(npc.conversation_partner)

            entry = npc.get_current_schedule_entry(new_slot)
            if entry is None:
                if new_slot == "night":
                    if in_conversation:
                        # Still generate the schedule so it's ready post-convo
                        sleep_entry = ScheduleEntry(
                            slot="night", activity="sleep", location="home",
                        )
                        npc.daily_schedule = [sleep_entry]
                        continue
                    tx, tz = self._spread_destination(npc, npc.home_x, npc.home_z)
                    sleep_entry = ScheduleEntry(
                        slot="night", activity="sleep", location="home",
                    )
                    subtasks = decompose_schedule_entry(npc, sleep_entry, npc._rng)
                    npc.subtask_queue = subtasks
                    npc.current_subtask = None
                    npc.subtask_time_remaining = 0.0
                    delay = npc._rng.uniform(1.0, 40.0)
                    self._pending_departures[npc.npc_id] = (
                        delay, tx, tz, "heading home to sleep",
                    )
                    continue

                # No schedule entry for this slot — re-plan if deterministic
                rd = self.router.route(
                    npc, "daily_schedule",
                    focus_x=self.focus_x, focus_z=self.focus_z,
                )
                if rd.route != Route.LLM:
                    self._generate_deterministic_schedule(
                        npc, new_slot, npc.schedule_day,
                    )
                    entry = npc.get_current_schedule_entry(new_slot)
                if entry is None:
                    continue

            # NPCs in conversation get their schedule updated but
            # don't dispatch yet — they'll pick it up when the
            # conversation ends and _refill_empty_queues fires.
            if in_conversation:
                continue

            if entry.target_x is not None and entry.target_z is not None:
                target_x, target_z = entry.target_x, entry.target_z
            else:
                target_x, target_z = resolve_schedule_location(
                    entry, npc, self.buildings,
                )
            target_x, target_z = self._spread_destination(npc, target_x, target_z)

            # Router decides LLM vs template decomposition
            rd = self.router.route(
                npc, "task_decompose",
                focus_x=self.focus_x, focus_z=self.focus_z,
            )
            if rd.route == Route.LLM:
                llm_tasks.append(
                    self._decompose_llm(npc, entry)
                )
            else:
                subtasks = decompose_schedule_entry(npc, entry, npc._rng)
                npc.subtask_queue = subtasks
                npc.current_subtask = None
                npc.subtask_time_remaining = 0.0

            delay = npc._rng.uniform(1.0, 40.0)
            self._pending_departures[npc.npc_id] = (
                delay, target_x, target_z, f"heading to {entry.activity}",
            )

        # Await LLM decompositions in parallel
        if llm_tasks:
            await asyncio.gather(*llm_tasks)

    def _refill_empty_queues(self, current_slot: str) -> None:
        """Re-decompose for NPCs that exhausted their subtask queue.

        Uses make_staggered_subtasks to prepend a random idle pause
        so NPCs don't all resume activity simultaneously. Also dispatches
        NPCs to their schedule entry target if they have one (e.g. NPCs
        who had their schedule updated mid-conversation).
        """
        for npc in self.npcs:
            if npc.cognition_tier >= 4:
                continue
            if npc.activity == ActivityState.WALKING:
                continue
            if npc.current_subtask or npc.subtask_queue:
                continue
            if npc.conversation_partner:
                continue

            entry = npc.get_current_schedule_entry(current_slot)

            # If entry has a target and NPC isn't there yet, dispatch
            if entry is not None and entry.target_x is not None:
                dist = abs(npc.x - entry.target_x) + abs(npc.z - entry.target_z)
                if dist > 3 and npc.npc_id not in self._pending_departures:
                    tx, tz = self._spread_destination(
                        npc, entry.target_x, entry.target_z,
                    )
                    delay = npc._rng.uniform(0.5, 5.0)
                    self._pending_departures[npc.npc_id] = (
                        delay, tx, tz, f"heading to {entry.activity}",
                    )

            npc.subtask_queue = make_staggered_subtasks(npc, entry, npc._rng)
            npc.current_subtask = None
            npc.subtask_time_remaining = 0.0

    def _process_pending_departures(self, real_delta: float) -> None:
        """Tick down departure delays (real seconds) and dispatch NPCs when ready."""
        completed: list[str] = []

        for npc_id, (delay, tx, tz, desc) in self._pending_departures.items():
            remaining = delay - real_delta
            if remaining <= 0:
                npc = self.get_npc(npc_id)
                if npc and not npc.conversation_partner:
                    if navigate_to(npc, self.grid, tx, tz):
                        npc.current_action_description = desc
                    else:
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
        Delegates to the spatial awareness module.
        """
        from core.world.spatial_awareness import get_occupied_tiles, find_rest_tile
        occupied = get_occupied_tiles(self.npcs)
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
