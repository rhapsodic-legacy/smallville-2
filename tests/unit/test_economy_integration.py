"""
Phase 6.5 Verify: Economy integration tests.

End-to-end scenarios that wire all economy subsystems together —
resources, trading, crafting, construction — with the world grid,
NPC models, and event impact system. No LLM calls; exercises the
full gather → craft → trade → build loop.
"""

import pytest

from core.world.grid import Grid, WorldObject
from core.world.generator import TownGenerator, WorldConfig
from core.npc.models import NPC, PersonalityTraits, ActivityState
from core.economy.resources import ResourceManager, ResourceType
from core.economy.trading import TradeManager, PriceEngine
from core.economy.crafting import CraftingManager, ItemQuality
from core.economy.construction import ConstructionManager, BuildPhase
from core.events.impact import EventImpactSystem, GameEvent
from core.relationships.sentiment import SentimentTracker


# ---------- Helpers ----------


def _make_npc(
    npc_id: str,
    name: str,
    occupation: str = "labourer",
    x: int = 0,
    z: int = 0,
    gold: int = 50,
    skills: dict | None = None,
    inventory: dict | None = None,
) -> NPC:
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory=f"{name} is a test NPC.",
        occupation=occupation,
        x=x, z=z,
        gold=gold,
        skills=skills if skills is not None else {
            "gathering": 0.5, "farming": 0.3, "mining": 0.4,
            "smithing": 0.6, "crafting": 0.5, "cooking": 0.4, "trading": 0.5,
        },
        inventory=inventory if inventory is not None else {},
    )


def _setup_world():
    """Generate a world and initialise all economy managers."""
    config = WorldConfig(population=5, economy="mixed", seed=42)
    gen = TownGenerator(config)
    grid = gen.generate()

    sentiment = SentimentTracker()
    sentiment.initialise()
    events = EventImpactSystem(sentiment_tracker=sentiment)
    events.initialise()

    resource_mgr = ResourceManager()
    resource_mgr.initialise_from_grid(grid)

    def fire_event(event_type, participants, data):
        event = GameEvent(
            event_type=event_type,
            participants=participants,
            data=data,
        )
        events.process_event(event)

    trade_mgr = TradeManager(on_event=fire_event)
    craft_mgr = CraftingManager()
    construction_mgr = ConstructionManager(on_event=fire_event)

    return {
        "grid": grid,
        "buildings": gen.buildings,
        "sentiment": sentiment,
        "events": events,
        "resources": resource_mgr,
        "trading": trade_mgr,
        "crafting": craft_mgr,
        "construction": construction_mgr,
    }


# ---------- Scenario 1: Gather → Inventory ----------


class TestGatherToInventory:
    """NPC walks to a resource node and gathers materials."""

    def test_gather_wood_from_oak_tree(self):
        world = _setup_world()
        res = world["resources"]

        # Find an oak tree node
        oak = next(
            (n for n in res.get_all_nodes() if n.resource_name == "oak_tree"),
            None,
        )
        assert oak is not None, "World should have oak trees"

        npc = _make_npc("lumberjack", "Aldric", x=oak.x, z=oak.z)

        ok, msg = res.start_gathering(npc, oak, 100.0)
        assert ok, f"Should start gathering: {msg}"

        done, result = res.complete_gathering(npc, 100.0 + oak.gather_time)
        assert done, f"Should complete gathering: {result}"
        assert npc.inventory.get("wood", 0) > 0
        assert oak.current_amount < oak.capacity

    def test_gather_multiple_types(self):
        """NPC gathers from two different node types."""
        world = _setup_world()
        res = world["resources"]

        npc = _make_npc("worker", "Bran", skills={
            "gathering": 0.5, "farming": 0.3, "mining": 0.4,
        })

        # Gather wood
        oak = next(n for n in res.get_all_nodes() if n.resource_name == "oak_tree")
        npc.x, npc.z = oak.x, oak.z
        res.start_gathering(npc, oak, 0.0)
        res.complete_gathering(npc, 100.0)

        # Gather berries
        berry = next(
            (n for n in res.get_all_nodes() if n.resource_name == "berry_bush"),
            None,
        )
        if berry:
            npc.x, npc.z = berry.x, berry.z
            res.start_gathering(npc, berry, 200.0)
            res.complete_gathering(npc, 300.0)
            assert npc.inventory.get("berries", 0) > 0

        assert npc.inventory.get("wood", 0) > 0


# ---------- Scenario 2: Gather → Craft ----------


class TestGatherToCraft:
    """NPC gathers raw materials then crafts them into goods."""

    def test_gather_wood_then_craft_planks(self):
        world = _setup_world()
        res = world["resources"]
        craft = world["crafting"]

        npc = _make_npc("carpenter", "Cedric", skills={"gathering": 0.5, "crafting": 0.5})

        # Gather wood until we have enough for planks (need 1 wood)
        oak = next(n for n in res.get_all_nodes() if n.resource_name == "oak_tree")
        npc.x, npc.z = oak.x, oak.z
        res.start_gathering(npc, oak, 0.0)
        res.complete_gathering(npc, 100.0)
        assert npc.inventory.get("wood", 0) >= 1

        # Craft planks
        ok, msg = craft.start_crafting(npc, "plank", 200.0)
        assert ok, f"Should start crafting: {msg}"
        done, result = craft.complete_crafting(npc, 210.0)
        assert done
        assert npc.inventory.get("plank", 0) > 0

    def test_gather_berries_craft_jam(self):
        world = _setup_world()
        res = world["resources"]
        craft = world["crafting"]

        npc = _make_npc("cook", "Dara", skills={"gathering": 0.5, "cooking": 0.4})

        berry = next(
            (n for n in res.get_all_nodes() if n.resource_name == "berry_bush"),
            None,
        )
        if berry is None:
            pytest.skip("No berry bushes in this world seed")

        npc.x, npc.z = berry.x, berry.z

        # Gather berries multiple times to get enough (need 3 for jam)
        for t in range(3):
            res.start_gathering(npc, berry, t * 100.0)
            res.complete_gathering(npc, t * 100.0 + 100.0)

        if npc.inventory.get("berries", 0) >= 3:
            ok, _ = craft.start_crafting(npc, "berry_jam", 500.0)
            assert ok
            done, result = craft.complete_crafting(npc, 510.0)
            assert done
            assert "food" in result["item"]


# ---------- Scenario 3: Trade Between NPCs ----------


class TestTradeFlow:
    """Two NPCs negotiate and complete a trade."""

    def test_wood_for_gold(self):
        world = _setup_world()
        trade = world["trading"]
        sentiment = world["sentiment"]

        seller = _make_npc("seller", "Edric", inventory={"wood": 20}, gold=50)
        buyer = _make_npc("buyer", "Fiona", gold=200)

        # Propose: 10 wood for 40 gold
        offer, msg = trade.propose_trade(
            seller, buyer,
            items_offered={"wood": 10},
            gold_requested=40,
            game_time=100.0,
        )
        assert offer is not None

        # Buyer evaluates (heuristic — fair at base price wood=5, so 50 vs 40)
        prices = trade.price_engine.get_market_prices()
        accept, reason = trade.evaluate_heuristic(buyer, offer, prices)

        # Execute
        ok, _ = trade.accept_trade(offer.offer_id, seller, buyer, game_time=110.0)
        assert ok

        assert seller.inventory["wood"] == 10
        assert seller.gold == 90   # 50 + 40
        assert buyer.inventory.get("wood", 0) == 10
        assert buyer.gold == 160   # 200 - 40

        # Sentiment should have changed (trade_completed fires +5 trust, +3 respect)
        sent = sentiment.get("seller", "buyer")
        assert sent.get("trust") > 0

    def test_trade_with_items_both_ways(self):
        world = _setup_world()
        trade = world["trading"]

        npc_a = _make_npc("a", "Gareth", inventory={"wood": 20, "iron": 5}, gold=100)
        npc_b = _make_npc("b", "Helena", inventory={"food": 30, "stone": 10}, gold=100)

        offer, _ = trade.propose_trade(
            npc_a, npc_b,
            items_offered={"wood": 10, "iron": 3},
            items_requested={"food": 15, "stone": 5},
            game_time=0.0,
        )
        assert offer is not None
        ok, _ = trade.accept_trade(offer.offer_id, npc_a, npc_b, game_time=5.0)
        assert ok

        assert npc_a.inventory["wood"] == 10
        assert npc_a.inventory["iron"] == 2
        assert npc_a.inventory["food"] == 15
        assert npc_a.inventory["stone"] == 5
        assert npc_b.inventory["wood"] == 10
        assert npc_b.inventory["iron"] == 3
        assert npc_b.inventory["food"] == 15
        assert npc_b.inventory["stone"] == 5

    def test_rejected_trade_fires_event(self):
        world = _setup_world()
        trade = world["trading"]
        sentiment = world["sentiment"]

        seller = _make_npc("seller", "Idris", inventory={"wood": 10})
        buyer = _make_npc("buyer", "Jasper")

        offer, _ = trade.propose_trade(
            seller, buyer, items_offered={"wood": 5},
        )
        trade.reject_trade(offer.offer_id, game_time=50.0)

        # trade_refused fires -3 trust, -2 respect
        sent = sentiment.get("seller", "buyer")
        assert sent.get("trust") < 0


# ---------- Scenario 4: Construction With Multiple Contributors ----------


class TestConstructionFlow:
    """Multiple NPCs contribute resources and labour to build a structure."""

    def test_build_home(self):
        world = _setup_world()
        grid = world["grid"]
        con = world["construction"]

        # Find a clear spot
        site, msg = con.start_construction("home", -5, -5, grid, game_time=0.0)
        assert site is not None, f"Should start construction: {msg}"
        assert site.phase == BuildPhase.PLANNED

        # Two NPCs contribute (home needs wood:50, stone:20)
        npc_a = _make_npc("a", "Kira", inventory={"wood": 50, "stone": 20})
        npc_b = _make_npc("b", "Leofric", inventory={"wood": 10, "stone": 20})

        # NPC A contributes wood and stone
        accepted, _ = con.contribute_resources(npc_a, site.site_id, "wood", 50)
        assert accepted == 50
        accepted, _ = con.contribute_resources(npc_b, site.site_id, "stone", 20)
        assert accepted == 20

        # Both contribute labour
        con.contribute_labour(npc_a, site.site_id, 30.0)
        con.contribute_labour(npc_b, site.site_id, 30.0)

        assert site.is_complete
        ok, _ = con.check_and_complete(site.site_id, grid)
        assert ok

        # Verify building placed on grid
        tile = grid.get_tile(-5, -5)
        building_objs = [o for o in tile.objects if o.object_type == "building"]
        assert len(building_objs) == 1
        assert building_objs[0].name == "Home"

        # Both contributors credited
        assert "a" in building_objs[0].metadata["built_by"]
        assert "b" in building_objs[0].metadata["built_by"]

    def test_construction_phases_progress(self):
        world = _setup_world()
        grid = world["grid"]
        con = world["construction"]

        site, _ = con.start_construction("home", -8, -8, grid)
        npc = _make_npc("builder", "Mira", inventory={"wood": 60, "stone": 30})

        assert site.phase == BuildPhase.PLANNED

        # Contribute ~35% of resources (should reach FOUNDATION)
        # home needs wood:50, stone:20 = 70 total. 35% = ~25 units
        con.contribute_resources(npc, site.site_id, "wood", 20)
        con.contribute_resources(npc, site.site_id, "stone", 5)
        # 25/70 resources = ~36% resources, * 0.7 weight = ~25% progress
        assert site.phase == BuildPhase.FOUNDATION

        # Contribute to ~75% total
        con.contribute_resources(npc, site.site_id, "wood", 30)
        con.contribute_resources(npc, site.site_id, "stone", 15)
        # All resources done (100% * 0.7 = 70%), need some labour for 75%+
        con.contribute_labour(npc, site.site_id, 15.0)
        # labour progress = 15/60 = 25%, * 0.3 = 7.5%, total = 77.5%
        assert site.phase == BuildPhase.ROOFING


# ---------- Scenario 5: Full Economy Loop ----------


class TestFullEconomyLoop:
    """
    End-to-end: gather resources → craft goods → trade for gold →
    contribute resources to construction.
    """

    def test_gather_craft_trade_build(self):
        world = _setup_world()
        res = world["resources"]
        craft = world["crafting"]
        trade = world["trading"]
        con = world["construction"]
        grid = world["grid"]

        # --- Setup NPCs ---
        blacksmith = _make_npc(
            "smith", "Thorin", occupation="blacksmith",
            skills={"gathering": 0.5, "smithing": 0.8, "crafting": 0.6},
            gold=20,
        )
        merchant = _make_npc(
            "merch", "Isolde", occupation="merchant",
            skills={"trading": 0.8},
            inventory={"iron": 10},
            gold=200,
        )

        # --- Step 1: Blacksmith gathers wood ---
        oak = next(n for n in res.get_all_nodes() if n.resource_name == "oak_tree")
        blacksmith.x, blacksmith.z = oak.x, oak.z
        res.start_gathering(blacksmith, oak, 0.0)
        res.complete_gathering(blacksmith, 100.0)
        wood_gathered = blacksmith.inventory.get("wood", 0)
        assert wood_gathered > 0

        # --- Step 2: Blacksmith trades wood for iron from merchant ---
        offer, msg = trade.propose_trade(
            blacksmith, merchant,
            items_offered={"wood": wood_gathered},
            items_requested={"iron": 4},
            game_time=200.0,
        )
        assert offer is not None, f"Trade proposal failed: {msg}"
        trade.accept_trade(offer.offer_id, blacksmith, merchant, game_time=210.0)
        assert blacksmith.inventory.get("iron", 0) == 4

        # --- Step 3: Blacksmith crafts iron ingots (need 2 iron each) ---
        craft.start_crafting(blacksmith, "iron_ingot", 300.0)
        done, result = craft.complete_crafting(blacksmith, 320.0)
        assert done
        # Quality suffix may be appended (e.g. "iron_ingot (masterwork)")
        has_ingot = any("iron_ingot" in k and v > 0 for k, v in blacksmith.inventory.items())
        assert has_ingot, f"Expected iron_ingot in inventory: {blacksmith.inventory}"

        # --- Step 4: Blacksmith starts a construction project ---
        # Use remaining wood for building
        blacksmith.inventory["wood"] = blacksmith.inventory.get("wood", 0) + 50
        blacksmith.inventory["stone"] = 20
        site, msg = con.start_construction("home", -12, -12, grid, game_time=400.0)
        assert site is not None, f"Construction failed: {msg}"

        con.contribute_resources(blacksmith, site.site_id, "wood", 50)
        con.contribute_resources(blacksmith, site.site_id, "stone", 20)
        con.contribute_labour(blacksmith, site.site_id, 60.0)
        assert site.is_complete

        ok, _ = con.check_and_complete(site.site_id, grid)
        assert ok

        # Verify final state
        tile = grid.get_tile(-12, -12)
        assert any(o.object_type == "building" for o in tile.objects)


# ---------- Scenario 6: Supply and Demand Pricing ----------


class TestSupplyDemandIntegration:
    """Price engine responds to world state changes."""

    def test_prices_respond_to_scarcity(self):
        world = _setup_world()
        res = world["resources"]
        engine = world["trading"].price_engine

        npcs = [
            _make_npc(f"npc_{i}", f"NPC{i}", inventory={})
            for i in range(10)
        ]

        # Get prices with full resource nodes
        prices_full = engine.get_market_prices(res, npcs)

        # Deplete all oak trees
        for node in res.get_nodes_by_type(ResourceType.WOOD):
            node.current_amount = 0

        prices_scarce = engine.get_market_prices(res, npcs)

        # Wood should be more expensive when scarce
        assert prices_scarce["wood"] >= prices_full["wood"]

    def test_shop_spread(self):
        """Shops buy low, sell high."""
        trade = TradeManager()
        prices = {"wood": 10, "iron": 20}
        buy_price = trade.shop_buy_price("wood", prices)
        sell_price = trade.shop_sell_price("wood", prices)
        assert buy_price > sell_price


# ---------- Scenario 7: Resource Regeneration Over Time ----------


class TestRegenerationIntegration:
    """Depleted nodes regenerate when the resource manager ticks."""

    def test_gather_deplete_regenerate(self):
        world = _setup_world()
        res = world["resources"]

        # Find a berry bush (small capacity, fast regen)
        berry = next(
            (n for n in res.get_all_nodes() if n.resource_name == "berry_bush"),
            None,
        )
        if berry is None:
            pytest.skip("No berry bushes")

        npc = _make_npc("gatherer", "Nessa", x=berry.x, z=berry.z,
                        skills={"gathering": 1.0})

        # Deplete it
        while not berry.is_depleted:
            res.start_gathering(npc, berry, 0.0)
            res.complete_gathering(npc, 100.0)

        assert berry.is_depleted

        # Tick for several game days — should regenerate
        for _ in range(5):
            res.tick(1440.0)  # 1 game day per tick

        assert berry.current_amount > 0


# ---------- Scenario 8: Crafting Quality Affects Economy ----------


class TestQualityEconomy:
    """Higher-skill NPCs produce better goods."""

    def test_master_smith_double_output(self):
        craft = CraftingManager()
        master = _make_npc(
            "master", "Voss",
            skills={"smithing": 0.9},
            inventory={"iron": 10},
        )
        craft.start_crafting(master, "iron_ingot", 0.0)
        _, result = craft.complete_crafting(master, 100.0)

        # Skill 0.9 → MASTERWORK → 2x yield
        assert result["quality"] == "masterwork"
        assert result["quantity"] == 2  # base 1 * 2

    def test_novice_poor_quality(self):
        craft = CraftingManager()
        novice = _make_npc(
            "novice", "Quinn",
            skills={"smithing": 0.2},
            inventory={"iron": 10},
        )
        craft.start_crafting(novice, "iron_ingot", 0.0)
        _, result = craft.complete_crafting(novice, 100.0)

        assert result["quality"] == "poor"
        assert "(poor)" in result["item"]


# ---------- Scenario 9: Event System Integration ----------


class TestEventIntegration:
    """Economy actions fire events that modify sentiment and world state."""

    def test_trade_modifies_sentiment(self):
        world = _setup_world()
        trade = world["trading"]
        sentiment = world["sentiment"]

        a = _make_npc("a", "Aldric", inventory={"wood": 20}, gold=100)
        b = _make_npc("b", "Briar", inventory={"stone": 10}, gold=100)

        offer, _ = trade.propose_trade(
            a, b, items_offered={"wood": 5}, gold_requested=20,
        )
        trade.accept_trade(offer.offer_id, a, b, game_time=10.0)

        sent_ab = sentiment.get("a", "b")
        assert sent_ab.get("trust") > 0
        assert sent_ab.get("respect") > 0

    def test_construction_complete_boosts_morale(self):
        world = _setup_world()
        con = world["construction"]
        events = world["events"]
        grid = world["grid"]

        from core.economy.construction import Blueprint
        tiny = Blueprint(
            blueprint_id="shed", building_type="shed", name="Shed",
            width=1, height=1,
            required_resources={"wood": 1}, labour_required=0.0,
        )
        con.add_blueprint(tiny)
        site, _ = con.start_construction("shed", -14, -14, grid)

        npc = _make_npc("builder", "Finn", inventory={"wood": 5})
        con.contribute_resources(npc, site.site_id, "wood", 1)
        con.check_and_complete(site.site_id, grid)

        # construction_complete rule sets morale_modifier +5
        morale = events.get_world_param("morale_modifier")
        assert morale >= 5.0

        # NPC gets built_something flag
        assert events.get_npc_flag("builder", "built_something") is True
