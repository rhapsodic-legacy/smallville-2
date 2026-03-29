"""
Trading system — NPC-to-NPC trade, market pricing, shop mechanics.

Trades flow through three stages:
  1. Proposal: one NPC builds an offer (items/gold for items/gold).
  2. Evaluation: the other NPC decides (LLM for Tier 1/2, heuristic for Tier 3).
  3. Resolution: items and gold transfer, events fire, memory records.

PriceEngine derives dynamic prices from supply (resource nodes + inventories)
and demand (NPC needs). Market stall merchants apply a markup/markdown spread.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.economy.resources import ResourceManager, ResourceType
    from core.npc.models import NPC

logger = logging.getLogger(__name__)


# ---------- Constants ----------

# Base gold value per unit of each resource type
BASE_PRICES: dict[str, int] = {
    "wood": 5,
    "stone": 8,
    "iron": 15,
    "gold_ore": 25,
    "food": 4,
    "wheat": 3,
    "berries": 2,
}

PRICE_FLOOR_MULTIPLIER = 0.2   # price never drops below 20% of base
PRICE_CEILING_MULTIPLIER = 5.0  # price never exceeds 500% of base

SHOP_BUY_MARKUP = 1.2   # shops charge 20% more when selling to NPCs
SHOP_SELL_MARKDOWN = 0.8  # shops pay 20% less when buying from NPCs

OFFER_EXPIRY_MINUTES = 30.0  # game minutes before an offer expires

TRADE_HISTORY_LIMIT = 20  # max recent trades kept per NPC


# ---------- Trade Status ----------

class TradeStatus(Enum):
    """Lifecycle states for a trade offer."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# ---------- Trade Offer ----------

@dataclass
class TradeOffer:
    """
    A proposed exchange between two NPCs.

    The proposer offers items_offered + gold_offered in exchange
    for items_requested + gold_requested from the recipient.
    """
    offer_id: str
    proposer_id: str
    recipient_id: str

    items_offered: dict[str, int] = field(default_factory=dict)
    gold_offered: int = 0

    items_requested: dict[str, int] = field(default_factory=dict)
    gold_requested: int = 0

    status: TradeStatus = TradeStatus.PENDING
    created_at: float = 0.0   # game minutes
    resolved_at: float = 0.0  # game minutes when accepted/rejected

    def offered_value(self, prices: dict[str, int]) -> int:
        """Total gold value of what the proposer offers."""
        total = self.gold_offered
        for item, qty in self.items_offered.items():
            total += prices.get(item, 0) * qty
        return total

    def requested_value(self, prices: dict[str, int]) -> int:
        """Total gold value of what the proposer requests."""
        total = self.gold_requested
        for item, qty in self.items_requested.items():
            total += prices.get(item, 0) * qty
        return total

    def fairness_ratio(self, prices: dict[str, int]) -> float:
        """
        Ratio of offered value to requested value.

        1.0 = perfectly fair, >1.0 = generous offer, <1.0 = unfair.
        Returns 0.0 if requested value is zero (pure gift).
        """
        req = self.requested_value(prices)
        if req == 0:
            return float("inf") if self.offered_value(prices) > 0 else 1.0
        return self.offered_value(prices) / req

    def to_dict(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "proposer_id": self.proposer_id,
            "recipient_id": self.recipient_id,
            "items_offered": dict(self.items_offered),
            "gold_offered": self.gold_offered,
            "items_requested": dict(self.items_requested),
            "gold_requested": self.gold_requested,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    def summary(self) -> str:
        """Human-readable summary for LLM prompts and logs."""
        offer_parts = []
        for item, qty in self.items_offered.items():
            offer_parts.append(f"{qty} {item}")
        if self.gold_offered:
            offer_parts.append(f"{self.gold_offered} gold")
        req_parts = []
        for item, qty in self.items_requested.items():
            req_parts.append(f"{qty} {item}")
        if self.gold_requested:
            req_parts.append(f"{self.gold_requested} gold")
        offer_str = ", ".join(offer_parts) or "nothing"
        req_str = ", ".join(req_parts) or "nothing"
        return f"Offers [{offer_str}] for [{req_str}]"


# ---------- Price Engine ----------

class PriceEngine:
    """
    Dynamic pricing based on supply and demand.

    Supply: total resource available (node amounts + NPC inventories).
    Demand: number of NPCs whose stock of a resource is below a threshold.
    Price = base * demand_factor / supply_factor, clamped to floor/ceiling.
    """

    def __init__(self, base_prices: dict[str, int] | None = None):
        self._base = dict(base_prices or BASE_PRICES)

    def get_base_price(self, resource: str) -> int:
        return self._base.get(resource, 1)

    def calculate_price(
        self,
        resource: str,
        supply: int,
        demand: int,
        population: int = 10,
    ) -> int:
        """
        Derive current price for a resource.

        supply: total units available in the world (nodes + inventories).
        demand: number of NPCs who want/need this resource.
        population: total NPC count (normalises demand).
        """
        base = self._base.get(resource, 1)

        # Supply factor: higher supply → lower price
        # Normalise against a "comfortable" supply (10 per NPC)
        comfortable_supply = population * 10
        if comfortable_supply <= 0:
            supply_factor = 1.0
        else:
            supply_factor = max(0.2, min(3.0, supply / comfortable_supply))

        # Demand factor: higher demand fraction → higher price
        if population <= 0:
            demand_factor = 1.0
        else:
            demand_ratio = demand / population
            demand_factor = max(0.5, min(2.5, 0.5 + demand_ratio * 2.0))

        raw_price = base * demand_factor / supply_factor
        floor = base * PRICE_FLOOR_MULTIPLIER
        ceiling = base * PRICE_CEILING_MULTIPLIER
        return max(1, int(max(floor, min(ceiling, raw_price))))

    def get_market_prices(
        self,
        resource_manager: ResourceManager | None = None,
        npcs: list[NPC] | None = None,
    ) -> dict[str, int]:
        """
        Calculate current market prices for all resource types.

        If resource_manager and npcs are provided, uses live supply/demand.
        Otherwise returns base prices.
        """
        if resource_manager is None or npcs is None:
            return dict(self._base)

        population = len(npcs)
        prices: dict[str, int] = {}

        for resource, base in self._base.items():
            # Supply: sum of node amounts + NPC inventories
            supply = sum(
                n.current_amount
                for n in resource_manager.get_all_nodes()
                if n.resource_type.value == resource
            )
            supply += sum(npc.inventory.get(resource, 0) for npc in npcs)

            # Demand: NPCs with less than 5 units of this resource
            demand = sum(
                1 for npc in npcs
                if npc.inventory.get(resource, 0) < 5
            )

            prices[resource] = self.calculate_price(
                resource, supply, demand, population,
            )

        return prices

    def apply_shop_markup(self, price: int) -> int:
        """Price an NPC pays when buying from a shop."""
        return max(1, int(price * SHOP_BUY_MARKUP))

    def apply_shop_markdown(self, price: int) -> int:
        """Price a shop pays when buying from an NPC."""
        return max(1, int(price * SHOP_SELL_MARKDOWN))


# ---------- Trade Evaluation (heuristic for Tier 3) ----------

def evaluate_trade_heuristic(
    npc: NPC,
    offer: TradeOffer,
    prices: dict[str, int],
) -> tuple[bool, str]:
    """
    Tier 3 heuristic trade evaluation.

    Accepts if the offer is at least 80% fair value.
    Also considers whether the NPC actually needs the offered items.
    Returns (accept, reason).
    """
    is_recipient = npc.npc_id == offer.recipient_id

    if is_recipient:
        # What we receive vs what we give
        receive_value = offer.offered_value(prices)
        give_value = offer.requested_value(prices)
    else:
        receive_value = offer.requested_value(prices)
        give_value = offer.offered_value(prices)

    if give_value == 0:
        return True, "free goods"

    ratio = receive_value / give_value

    if ratio >= 0.8:
        return True, f"fair trade (value ratio {ratio:.2f})"
    return False, f"unfair trade (value ratio {ratio:.2f})"


def build_trade_prompt(
    npc: NPC,
    offer: TradeOffer,
    prices: dict[str, int],
    relationship_summary: str = "",
) -> str:
    """
    Build an LLM prompt for Tier 1/2 trade evaluation.

    Returns the prompt text. The LLM should respond with ACCEPT or REJECT
    followed by a brief reason.
    """
    npc_summary = npc.summary_for_prompt()
    offer_summary = offer.summary()

    # Value context
    offered_val = offer.offered_value(prices)
    requested_val = offer.requested_value(prices)

    parts = [
        f"You are {npc.name}, {npc.occupation}.",
        f"Profile: {npc_summary}",
        "",
        f"Someone proposes a trade: {offer_summary}",
        f"Estimated value of what you receive: {offered_val} gold",
        f"Estimated value of what you give: {requested_val} gold",
        "",
        f"Your current inventory: {dict(npc.inventory)}",
        f"Your gold: {npc.gold}",
    ]

    if relationship_summary:
        parts.append(f"\nRelationship context: {relationship_summary}")

    parts.extend([
        "",
        "Consider: Is this trade fair? Do you need what's offered? "
        "Can you afford what's asked? Does your relationship with this "
        "person affect your willingness?",
        "",
        "Respond with exactly ACCEPT or REJECT on the first line, "
        "followed by a brief reason.",
    ])

    return "\n".join(parts)


# ---------- Trade Manager ----------

class TradeManager:
    """
    Orchestrates trade proposals, evaluations, and executions.

    Tracks active offers, enforces inventory/gold validation,
    fires events through a callback, and maintains trade history.
    """

    def __init__(
        self,
        price_engine: PriceEngine | None = None,
        on_event: Any | None = None,
    ):
        self._price_engine = price_engine or PriceEngine()
        self._on_event = on_event  # callback: (event_type, participants, data) -> None
        self._active_offers: dict[str, TradeOffer] = {}
        self._history: dict[str, list[TradeOffer]] = {}  # npc_id -> recent trades

    @property
    def price_engine(self) -> PriceEngine:
        return self._price_engine

    # ---------- Proposal ----------

    def propose_trade(
        self,
        proposer: NPC,
        recipient: NPC,
        items_offered: dict[str, int] | None = None,
        gold_offered: int = 0,
        items_requested: dict[str, int] | None = None,
        gold_requested: int = 0,
        game_time: float = 0.0,
    ) -> tuple[TradeOffer | None, str]:
        """
        Create a trade offer after validating the proposer can afford it.

        Returns (offer, message). Offer is None on validation failure.
        """
        items_offered = items_offered or {}
        items_requested = items_requested or {}

        # Validate proposer has the offered items
        for item, qty in items_offered.items():
            if proposer.inventory.get(item, 0) < qty:
                return None, f"proposer lacks {qty} {item}"

        if proposer.gold < gold_offered:
            return None, "proposer lacks gold"

        # Validate recipient has the requested items
        for item, qty in items_requested.items():
            if recipient.inventory.get(item, 0) < qty:
                return None, f"recipient lacks {qty} {item}"

        if recipient.gold < gold_requested:
            return None, "recipient lacks gold"

        offer = TradeOffer(
            offer_id=str(uuid.uuid4())[:8],
            proposer_id=proposer.npc_id,
            recipient_id=recipient.npc_id,
            items_offered=dict(items_offered),
            gold_offered=gold_offered,
            items_requested=dict(items_requested),
            gold_requested=gold_requested,
            created_at=game_time,
        )
        self._active_offers[offer.offer_id] = offer
        logger.debug("Trade proposed: %s", offer.summary())
        return offer, "ok"

    # ---------- Evaluation ----------

    def evaluate_heuristic(
        self,
        npc: NPC,
        offer: TradeOffer,
        prices: dict[str, int] | None = None,
    ) -> tuple[bool, str]:
        """Evaluate a trade using the Tier 3 heuristic."""
        if prices is None:
            prices = self._price_engine.get_market_prices()
        return evaluate_trade_heuristic(npc, offer, prices)

    def get_trade_prompt(
        self,
        npc: NPC,
        offer: TradeOffer,
        prices: dict[str, int] | None = None,
        relationship_summary: str = "",
    ) -> str:
        """Build an LLM prompt for Tier 1/2 trade evaluation."""
        if prices is None:
            prices = self._price_engine.get_market_prices()
        return build_trade_prompt(npc, offer, prices, relationship_summary)

    @staticmethod
    def parse_trade_response(response: str) -> tuple[bool, str]:
        """
        Parse an LLM trade evaluation response.

        Expects ACCEPT or REJECT on the first line, reason on subsequent lines.
        """
        lines = response.strip().splitlines()
        if not lines:
            return False, "no response"
        first = lines[0].strip().upper()
        reason = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
        if "ACCEPT" in first:
            return True, reason or "accepted"
        return False, reason or "rejected"

    # ---------- Resolution ----------

    def accept_trade(
        self,
        offer_id: str,
        proposer: NPC,
        recipient: NPC,
        game_time: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Execute an accepted trade — transfer items and gold.

        Re-validates both parties can still afford the trade.
        Fires trade_completed event on success.
        """
        offer = self._active_offers.get(offer_id)
        if offer is None:
            return False, "offer not found"

        if offer.status != TradeStatus.PENDING:
            return False, f"offer already {offer.status.value}"

        # Re-validate proposer
        for item, qty in offer.items_offered.items():
            if proposer.inventory.get(item, 0) < qty:
                offer.status = TradeStatus.CANCELLED
                return False, f"proposer no longer has {qty} {item}"
        if proposer.gold < offer.gold_offered:
            offer.status = TradeStatus.CANCELLED
            return False, "proposer no longer has enough gold"

        # Re-validate recipient
        for item, qty in offer.items_requested.items():
            if recipient.inventory.get(item, 0) < qty:
                offer.status = TradeStatus.CANCELLED
                return False, f"recipient no longer has {qty} {item}"
        if recipient.gold < offer.gold_requested:
            offer.status = TradeStatus.CANCELLED
            return False, "recipient no longer has enough gold"

        # Transfer: proposer -> recipient
        for item, qty in offer.items_offered.items():
            proposer.inventory[item] = proposer.inventory.get(item, 0) - qty
            recipient.inventory[item] = recipient.inventory.get(item, 0) + qty
        proposer.gold -= offer.gold_offered
        recipient.gold += offer.gold_offered

        # Transfer: recipient -> proposer
        for item, qty in offer.items_requested.items():
            recipient.inventory[item] = recipient.inventory.get(item, 0) - qty
            proposer.inventory[item] = proposer.inventory.get(item, 0) + qty
        recipient.gold -= offer.gold_requested
        proposer.gold += offer.gold_requested

        offer.status = TradeStatus.ACCEPTED
        offer.resolved_at = game_time
        del self._active_offers[offer_id]
        self._record_history(offer)

        if self._on_event:
            self._on_event(
                "trade_completed",
                [offer.proposer_id, offer.recipient_id],
                {"offer": offer.to_dict()},
            )

        logger.debug("Trade accepted: %s", offer.summary())
        return True, "ok"

    def reject_trade(
        self,
        offer_id: str,
        game_time: float = 0.0,
    ) -> tuple[bool, str]:
        """Reject a pending trade offer. Fires trade_refused event."""
        offer = self._active_offers.get(offer_id)
        if offer is None:
            return False, "offer not found"

        if offer.status != TradeStatus.PENDING:
            return False, f"offer already {offer.status.value}"

        offer.status = TradeStatus.REJECTED
        offer.resolved_at = game_time
        del self._active_offers[offer_id]
        self._record_history(offer)

        if self._on_event:
            self._on_event(
                "trade_refused",
                [offer.proposer_id, offer.recipient_id],
                {"offer": offer.to_dict()},
            )

        logger.debug("Trade rejected: %s", offer.summary())
        return True, "ok"

    def cancel_trade(self, offer_id: str) -> bool:
        """Cancel a pending offer (proposer changed their mind)."""
        offer = self._active_offers.get(offer_id)
        if offer is None or offer.status != TradeStatus.PENDING:
            return False
        offer.status = TradeStatus.CANCELLED
        del self._active_offers[offer_id]
        return True

    # ---------- Expiry ----------

    def expire_stale_offers(self, current_game_time: float) -> int:
        """Expire offers older than OFFER_EXPIRY_MINUTES. Returns count expired."""
        expired_ids = [
            oid for oid, offer in self._active_offers.items()
            if (current_game_time - offer.created_at) >= OFFER_EXPIRY_MINUTES
        ]
        for oid in expired_ids:
            self._active_offers[oid].status = TradeStatus.EXPIRED
            self._record_history(self._active_offers[oid])
            del self._active_offers[oid]
        return len(expired_ids)

    # ---------- Queries ----------

    def get_offer(self, offer_id: str) -> TradeOffer | None:
        return self._active_offers.get(offer_id)

    def get_offers_for(self, npc_id: str) -> list[TradeOffer]:
        """All pending offers where the NPC is the recipient."""
        return [
            o for o in self._active_offers.values()
            if o.recipient_id == npc_id and o.status == TradeStatus.PENDING
        ]

    def get_offers_by(self, npc_id: str) -> list[TradeOffer]:
        """All pending offers proposed by this NPC."""
        return [
            o for o in self._active_offers.values()
            if o.proposer_id == npc_id and o.status == TradeStatus.PENDING
        ]

    def get_trade_history(self, npc_id: str) -> list[TradeOffer]:
        """Recent resolved trades involving this NPC."""
        return list(self._history.get(npc_id, []))

    # ---------- Shop Helpers ----------

    def shop_buy_price(self, resource: str, prices: dict[str, int] | None = None) -> int:
        """Price an NPC pays to buy from a shop."""
        if prices is None:
            prices = self._price_engine.get_market_prices()
        base = prices.get(resource, self._price_engine.get_base_price(resource))
        return self._price_engine.apply_shop_markup(base)

    def shop_sell_price(self, resource: str, prices: dict[str, int] | None = None) -> int:
        """Price a shop pays an NPC for a resource."""
        if prices is None:
            prices = self._price_engine.get_market_prices()
        base = prices.get(resource, self._price_engine.get_base_price(resource))
        return self._price_engine.apply_shop_markdown(base)

    # ---------- State ----------

    def get_state(self) -> dict[str, Any]:
        return {
            "active_offers": len(self._active_offers),
            "offers": [o.to_dict() for o in self._active_offers.values()],
        }

    def get_stats(self) -> dict[str, Any]:
        total_history = sum(len(v) for v in self._history.values())
        return {
            "active_offers": len(self._active_offers),
            "total_trades_recorded": total_history,
        }

    # ---------- Internal ----------

    def _record_history(self, offer: TradeOffer) -> None:
        """Store a resolved offer in both participants' history."""
        for npc_id in (offer.proposer_id, offer.recipient_id):
            history = self._history.setdefault(npc_id, [])
            history.append(offer)
            if len(history) > TRADE_HISTORY_LIMIT:
                self._history[npc_id] = history[-TRADE_HISTORY_LIMIT:]
