"""
Tests for Phase 6.2: Trading System.

Covers trade offers, price engine, heuristic evaluation,
LLM prompt building, trade execution, shop mechanics, and history.
"""

import pytest

from core.economy.trading import (
    TradeOffer,
    TradeStatus,
    TradeManager,
    PriceEngine,
    BASE_PRICES,
    SHOP_BUY_MARKUP,
    SHOP_SELL_MARKDOWN,
    OFFER_EXPIRY_MINUTES,
    evaluate_trade_heuristic,
    build_trade_prompt,
)
from core.npc.models import NPC, PersonalityTraits


# ---------- Helpers ----------


def _make_npc(
    npc_id: str = "npc_1",
    name: str = "Aldric",
    occupation: str = "merchant",
    gold: int = 100,
    inventory: dict | None = None,
    skills: dict | None = None,
) -> NPC:
    return NPC(
        npc_id=npc_id,
        name=name,
        age=30,
        personality=PersonalityTraits(),
        backstory="A test NPC.",
        occupation=occupation,
        gold=gold,
        inventory=inventory if inventory is not None else {},
        skills=skills if skills is not None else {"trading": 0.5},
    )


# ---------- TradeOffer ----------


class TestTradeOffer:

    def test_offered_value(self):
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_offered={"wood": 10}, gold_offered=5,
        )
        prices = {"wood": 5}
        assert offer.offered_value(prices) == 55  # 10*5 + 5

    def test_requested_value(self):
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_requested={"iron": 3}, gold_requested=10,
        )
        prices = {"iron": 15}
        assert offer.requested_value(prices) == 55  # 3*15 + 10

    def test_fairness_ratio_fair(self):
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_offered={"wood": 10}, gold_offered=0,
            items_requested={"stone": 5}, gold_requested=0,
        )
        prices = {"wood": 5, "stone": 10}
        assert offer.fairness_ratio(prices) == pytest.approx(1.0)

    def test_fairness_ratio_generous(self):
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_offered={"wood": 20}, gold_offered=0,
            items_requested={"stone": 5}, gold_requested=0,
        )
        prices = {"wood": 5, "stone": 10}
        assert offer.fairness_ratio(prices) == pytest.approx(2.0)

    def test_fairness_ratio_zero_request(self):
        """Offering something for nothing = gift = inf."""
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_offered={"wood": 5}, gold_offered=0,
        )
        assert offer.fairness_ratio({"wood": 5}) == float("inf")

    def test_fairness_ratio_both_zero(self):
        offer = TradeOffer(offer_id="t1", proposer_id="a", recipient_id="b")
        assert offer.fairness_ratio({}) == 1.0

    def test_summary(self):
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_offered={"wood": 5}, gold_offered=10,
            items_requested={"iron": 2},
        )
        s = offer.summary()
        assert "wood" in s
        assert "gold" in s
        assert "iron" in s

    def test_summary_nothing(self):
        offer = TradeOffer(offer_id="t1", proposer_id="a", recipient_id="b")
        assert "nothing" in offer.summary()

    def test_to_dict(self):
        offer = TradeOffer(
            offer_id="t1", proposer_id="a", recipient_id="b",
            items_offered={"wood": 5}, gold_offered=10,
            status=TradeStatus.PENDING,
        )
        d = offer.to_dict()
        assert d["offer_id"] == "t1"
        assert d["status"] == "pending"
        assert d["items_offered"] == {"wood": 5}


# ---------- PriceEngine ----------


class TestPriceEngine:

    def test_base_prices(self):
        engine = PriceEngine()
        assert engine.get_base_price("wood") == 5
        assert engine.get_base_price("iron") == 15

    def test_unknown_resource_defaults_to_one(self):
        engine = PriceEngine()
        assert engine.get_base_price("mystery_ore") == 1

    def test_custom_base_prices(self):
        engine = PriceEngine(base_prices={"gems": 50})
        assert engine.get_base_price("gems") == 50

    def test_calculate_price_balanced(self):
        engine = PriceEngine()
        # comfortable supply (10 * 10 = 100), half population demands
        price = engine.calculate_price("wood", supply=100, demand=5, population=10)
        assert price >= 1

    def test_high_supply_lowers_price(self):
        engine = PriceEngine()
        low_supply = engine.calculate_price("wood", supply=10, demand=5, population=10)
        high_supply = engine.calculate_price("wood", supply=500, demand=5, population=10)
        assert high_supply <= low_supply

    def test_high_demand_raises_price(self):
        engine = PriceEngine()
        low_demand = engine.calculate_price("wood", supply=100, demand=1, population=10)
        high_demand = engine.calculate_price("wood", supply=100, demand=10, population=10)
        assert high_demand >= low_demand

    def test_price_never_below_one(self):
        engine = PriceEngine()
        price = engine.calculate_price("berries", supply=99999, demand=0, population=10)
        assert price >= 1

    def test_price_capped_at_ceiling(self):
        engine = PriceEngine()
        price = engine.calculate_price("wood", supply=0, demand=10, population=10)
        assert price <= BASE_PRICES["wood"] * 5

    def test_shop_markup(self):
        engine = PriceEngine()
        base = 10
        marked_up = engine.apply_shop_markup(base)
        assert marked_up == int(base * SHOP_BUY_MARKUP)

    def test_shop_markdown(self):
        engine = PriceEngine()
        base = 10
        marked_down = engine.apply_shop_markdown(base)
        assert marked_down == int(base * SHOP_SELL_MARKDOWN)

    def test_shop_markup_minimum_one(self):
        engine = PriceEngine()
        assert engine.apply_shop_markup(0) >= 1

    def test_get_market_prices_no_data(self):
        """Without resource_manager/npcs, returns base prices."""
        engine = PriceEngine()
        prices = engine.get_market_prices()
        assert prices == BASE_PRICES


# ---------- Heuristic Evaluation ----------


class TestHeuristicEvaluation:

    def test_fair_trade_accepted(self):
        npc = _make_npc(npc_id="buyer")
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id="buyer",
            items_offered={"wood": 10}, gold_offered=0,
            items_requested={"stone": 5}, gold_requested=0,
        )
        prices = {"wood": 5, "stone": 10}  # 50 vs 50 = fair
        accept, reason = evaluate_trade_heuristic(npc, offer, prices)
        assert accept

    def test_unfair_trade_rejected(self):
        npc = _make_npc(npc_id="buyer")
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id="buyer",
            items_offered={"berries": 2},
            items_requested={"iron": 5},
        )
        prices = {"berries": 2, "iron": 15}  # 4 vs 75 = very unfair
        accept, reason = evaluate_trade_heuristic(npc, offer, prices)
        assert not accept
        assert "unfair" in reason

    def test_free_goods_accepted(self):
        npc = _make_npc(npc_id="buyer")
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id="buyer",
            items_offered={"wood": 5},
        )
        accept, _ = evaluate_trade_heuristic(npc, offer, {"wood": 5})
        assert accept

    def test_slightly_unfair_still_accepted(self):
        """80% fairness threshold: 0.85 ratio should pass."""
        npc = _make_npc(npc_id="buyer")
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id="buyer",
            items_offered={"wood": 17}, gold_offered=0,
            items_requested={"wood": 20}, gold_requested=0,
        )
        prices = {"wood": 5}  # 85 vs 100 = 0.85 ratio
        accept, _ = evaluate_trade_heuristic(npc, offer, prices)
        assert accept


# ---------- LLM Prompt ----------


class TestBuildTradePrompt:

    def test_prompt_contains_npc_info(self):
        npc = _make_npc(name="Thorin", occupation="blacksmith")
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id=npc.npc_id,
            items_offered={"wood": 5}, items_requested={"iron": 2},
        )
        prompt = build_trade_prompt(npc, offer, {"wood": 5, "iron": 15})
        assert "Thorin" in prompt
        assert "blacksmith" in prompt

    def test_prompt_contains_value_estimates(self):
        npc = _make_npc()
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id=npc.npc_id,
            items_offered={"wood": 10}, gold_offered=5,
        )
        prompt = build_trade_prompt(npc, offer, {"wood": 5})
        assert "55" in prompt  # 10*5 + 5

    def test_prompt_includes_relationship(self):
        npc = _make_npc()
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id=npc.npc_id,
        )
        prompt = build_trade_prompt(
            npc, offer, {}, relationship_summary="Trusts the seller.",
        )
        assert "Trusts the seller" in prompt

    def test_prompt_ends_with_accept_reject_instruction(self):
        npc = _make_npc()
        offer = TradeOffer(
            offer_id="t1", proposer_id="seller", recipient_id=npc.npc_id,
        )
        prompt = build_trade_prompt(npc, offer, {})
        assert "ACCEPT" in prompt
        assert "REJECT" in prompt


# ---------- Parse Trade Response ----------


class TestParseTradeResponse:

    def test_accept(self):
        accept, reason = TradeManager.parse_trade_response("ACCEPT\nGood deal.")
        assert accept
        assert reason == "Good deal."

    def test_reject(self):
        accept, reason = TradeManager.parse_trade_response("REJECT\nToo expensive.")
        assert not accept
        assert reason == "Too expensive."

    def test_accept_case_insensitive(self):
        accept, _ = TradeManager.parse_trade_response("accept")
        assert accept

    def test_empty_response(self):
        accept, _ = TradeManager.parse_trade_response("")
        assert not accept

    def test_no_reason(self):
        accept, reason = TradeManager.parse_trade_response("ACCEPT")
        assert accept
        assert reason == "accepted"


# ---------- TradeManager — Proposal ----------


class TestTradeProposal:

    def test_propose_success(self):
        seller = _make_npc(npc_id="seller", inventory={"wood": 20}, gold=50)
        buyer = _make_npc(npc_id="buyer", inventory={"iron": 10}, gold=100)
        mgr = TradeManager()
        offer, msg = mgr.propose_trade(
            seller, buyer,
            items_offered={"wood": 5}, gold_offered=0,
            items_requested={"iron": 2}, gold_requested=10,
        )
        assert offer is not None
        assert msg == "ok"
        assert offer.status == TradeStatus.PENDING

    def test_propose_proposer_lacks_items(self):
        seller = _make_npc(npc_id="seller", inventory={"wood": 2})
        buyer = _make_npc(npc_id="buyer")
        mgr = TradeManager()
        offer, msg = mgr.propose_trade(
            seller, buyer, items_offered={"wood": 10},
        )
        assert offer is None
        assert "proposer lacks" in msg

    def test_propose_proposer_lacks_gold(self):
        seller = _make_npc(npc_id="seller", gold=5)
        buyer = _make_npc(npc_id="buyer")
        mgr = TradeManager()
        offer, msg = mgr.propose_trade(seller, buyer, gold_offered=50)
        assert offer is None
        assert "gold" in msg

    def test_propose_recipient_lacks_items(self):
        seller = _make_npc(npc_id="seller", inventory={"wood": 20})
        buyer = _make_npc(npc_id="buyer", inventory={"iron": 1})
        mgr = TradeManager()
        offer, msg = mgr.propose_trade(
            seller, buyer,
            items_offered={"wood": 5},
            items_requested={"iron": 5},
        )
        assert offer is None
        assert "recipient lacks" in msg

    def test_propose_recipient_lacks_gold(self):
        seller = _make_npc(npc_id="seller")
        buyer = _make_npc(npc_id="buyer", gold=0)
        mgr = TradeManager()
        offer, msg = mgr.propose_trade(
            seller, buyer, gold_requested=50,
        )
        assert offer is None
        assert "recipient lacks gold" in msg


# ---------- TradeManager — Accept ----------


class TestTradeAccept:

    def setup_method(self):
        self.events = []
        self.mgr = TradeManager(on_event=lambda t, p, d: self.events.append((t, p, d)))
        self.seller = _make_npc(npc_id="seller", inventory={"wood": 20}, gold=50)
        self.buyer = _make_npc(npc_id="buyer", inventory={"iron": 10}, gold=100)
        self.offer, _ = self.mgr.propose_trade(
            self.seller, self.buyer,
            items_offered={"wood": 5}, gold_offered=10,
            items_requested={"iron": 2}, gold_requested=20,
            game_time=100.0,
        )

    def test_accept_transfers_items(self):
        ok, _ = self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        assert ok
        assert self.seller.inventory["wood"] == 15   # gave 5
        assert self.buyer.inventory["wood"] == 5      # received 5
        assert self.buyer.inventory["iron"] == 8       # gave 2
        assert self.seller.inventory.get("iron", 0) == 2  # received 2

    def test_accept_transfers_gold(self):
        self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        assert self.seller.gold == 60   # 50 - 10 + 20
        assert self.buyer.gold == 90    # 100 + 10 - 20

    def test_accept_fires_event(self):
        self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        assert len(self.events) == 1
        assert self.events[0][0] == "trade_completed"
        assert "seller" in self.events[0][1]
        assert "buyer" in self.events[0][1]

    def test_accept_sets_status(self):
        self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        assert self.offer.status == TradeStatus.ACCEPTED

    def test_accept_removes_from_active(self):
        self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        assert self.mgr.get_offer(self.offer.offer_id) is None

    def test_accept_nonexistent_offer(self):
        ok, msg = self.mgr.accept_trade("nope", self.seller, self.buyer)
        assert not ok
        assert "not found" in msg

    def test_accept_already_resolved(self):
        self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        ok, msg = self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=120.0,
        )
        assert not ok

    def test_accept_proposer_lost_items(self):
        """Proposer's inventory changed between proposal and acceptance."""
        self.seller.inventory["wood"] = 2  # no longer has 5
        ok, msg = self.mgr.accept_trade(
            self.offer.offer_id, self.seller, self.buyer, game_time=110.0,
        )
        assert not ok
        assert "proposer no longer" in msg
        assert self.offer.status == TradeStatus.CANCELLED


# ---------- TradeManager — Reject ----------


class TestTradeReject:

    def setup_method(self):
        self.events = []
        self.mgr = TradeManager(on_event=lambda t, p, d: self.events.append((t, p, d)))
        self.seller = _make_npc(npc_id="seller", inventory={"wood": 20})
        self.buyer = _make_npc(npc_id="buyer")
        self.offer, _ = self.mgr.propose_trade(
            self.seller, self.buyer, items_offered={"wood": 5},
        )

    def test_reject_fires_event(self):
        ok, _ = self.mgr.reject_trade(self.offer.offer_id, game_time=110.0)
        assert ok
        assert self.events[0][0] == "trade_refused"

    def test_reject_sets_status(self):
        self.mgr.reject_trade(self.offer.offer_id)
        assert self.offer.status == TradeStatus.REJECTED

    def test_reject_nonexistent(self):
        ok, msg = self.mgr.reject_trade("nope")
        assert not ok


# ---------- TradeManager — Cancel & Expiry ----------


class TestTradeCancelExpiry:

    def test_cancel(self):
        mgr = TradeManager()
        seller = _make_npc(npc_id="s", inventory={"wood": 20})
        buyer = _make_npc(npc_id="b")
        offer, _ = mgr.propose_trade(seller, buyer, items_offered={"wood": 5})
        assert mgr.cancel_trade(offer.offer_id)
        assert offer.status == TradeStatus.CANCELLED
        assert mgr.get_offer(offer.offer_id) is None

    def test_cancel_nonexistent(self):
        mgr = TradeManager()
        assert not mgr.cancel_trade("nope")

    def test_expire_stale(self):
        mgr = TradeManager()
        seller = _make_npc(npc_id="s", inventory={"wood": 20})
        buyer = _make_npc(npc_id="b")
        offer, _ = mgr.propose_trade(
            seller, buyer, items_offered={"wood": 5}, game_time=0.0,
        )
        expired = mgr.expire_stale_offers(OFFER_EXPIRY_MINUTES + 1)
        assert expired == 1
        assert offer.status == TradeStatus.EXPIRED

    def test_expire_keeps_fresh(self):
        mgr = TradeManager()
        seller = _make_npc(npc_id="s", inventory={"wood": 20})
        buyer = _make_npc(npc_id="b")
        mgr.propose_trade(
            seller, buyer, items_offered={"wood": 5}, game_time=100.0,
        )
        expired = mgr.expire_stale_offers(110.0)  # only 10 min passed
        assert expired == 0


# ---------- TradeManager — Queries ----------


class TestTradeQueries:

    def test_get_offers_for(self):
        mgr = TradeManager()
        s = _make_npc(npc_id="s", inventory={"wood": 50})
        b = _make_npc(npc_id="b")
        mgr.propose_trade(s, b, items_offered={"wood": 5})
        mgr.propose_trade(s, b, items_offered={"wood": 3})
        assert len(mgr.get_offers_for("b")) == 2
        assert len(mgr.get_offers_for("s")) == 0

    def test_get_offers_by(self):
        mgr = TradeManager()
        s = _make_npc(npc_id="s", inventory={"wood": 50})
        b = _make_npc(npc_id="b")
        mgr.propose_trade(s, b, items_offered={"wood": 5})
        assert len(mgr.get_offers_by("s")) == 1

    def test_trade_history(self):
        events = []
        mgr = TradeManager(on_event=lambda t, p, d: events.append(t))
        s = _make_npc(npc_id="s", inventory={"wood": 50})
        b = _make_npc(npc_id="b", gold=100)
        offer, _ = mgr.propose_trade(
            s, b, items_offered={"wood": 5}, gold_requested=10,
        )
        mgr.accept_trade(offer.offer_id, s, b, game_time=10.0)
        assert len(mgr.get_trade_history("s")) == 1
        assert len(mgr.get_trade_history("b")) == 1


# ---------- TradeManager — Shop Helpers ----------


class TestShopHelpers:

    def test_shop_buy_price(self):
        mgr = TradeManager()
        price = mgr.shop_buy_price("wood", prices={"wood": 10})
        assert price == int(10 * SHOP_BUY_MARKUP)

    def test_shop_sell_price(self):
        mgr = TradeManager()
        price = mgr.shop_sell_price("wood", prices={"wood": 10})
        assert price == int(10 * SHOP_SELL_MARKDOWN)


# ---------- TradeManager — State ----------


class TestTradeState:

    def test_get_state(self):
        mgr = TradeManager()
        s = _make_npc(npc_id="s", inventory={"wood": 20})
        b = _make_npc(npc_id="b")
        mgr.propose_trade(s, b, items_offered={"wood": 5})
        state = mgr.get_state()
        assert state["active_offers"] == 1
        assert len(state["offers"]) == 1

    def test_get_stats(self):
        mgr = TradeManager()
        stats = mgr.get_stats()
        assert stats["active_offers"] == 0
        assert stats["total_trades_recorded"] == 0
