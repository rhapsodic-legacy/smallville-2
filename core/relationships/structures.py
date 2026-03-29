"""
Formal structures — factions, trade agreements, alliances, councils.

Factions are the primary grouping mechanism. NPCs belong to factions
with specific roles. Factions can be allied, rival, or neutral.
Trade agreements and councils are modelled as formal agreements
between factions or individual NPCs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FactionRelation(Enum):
    """How two factions view each other."""
    ALLIED = "allied"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    RIVAL = "rival"
    HOSTILE = "hostile"


class FactionRole(Enum):
    """Standard roles within a faction."""
    LEADER = "leader"
    OFFICER = "officer"
    MEMBER = "member"
    RECRUIT = "recruit"


@dataclass
class FactionMember:
    """An NPC's membership in a faction."""
    npc_id: str
    role: FactionRole = FactionRole.MEMBER
    joined_at: float = 0.0    # game time

    def to_dict(self) -> dict[str, Any]:
        return {
            "npc_id": self.npc_id,
            "role": self.role.value,
            "joined_at": self.joined_at,
        }


@dataclass
class Agreement:
    """A formal agreement between two parties (factions or NPCs)."""
    agreement_id: str
    agreement_type: str      # "trade", "alliance", "non_aggression", "tribute"
    party_a: str             # faction_id or npc_id
    party_b: str
    terms: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    expires_at: float = 0.0  # 0 = no expiry
    active: bool = True

    def is_expired(self, current_time: float) -> bool:
        return self.expires_at > 0 and current_time >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement_id": self.agreement_id,
            "type": self.agreement_type,
            "party_a": self.party_a,
            "party_b": self.party_b,
            "terms": self.terms,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "active": self.active,
        }


@dataclass
class Faction:
    """A named group of NPCs with internal hierarchy and external relations."""
    faction_id: str
    name: str
    description: str = ""
    members: list[FactionMember] = field(default_factory=list)
    relations: dict[str, FactionRelation] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def leader(self) -> FactionMember | None:
        for m in self.members:
            if m.role == FactionRole.LEADER:
                return m
        return None

    @property
    def member_ids(self) -> list[str]:
        return [m.npc_id for m in self.members]

    def has_member(self, npc_id: str) -> bool:
        return any(m.npc_id == npc_id for m in self.members)

    def get_member(self, npc_id: str) -> FactionMember | None:
        for m in self.members:
            if m.npc_id == npc_id:
                return m
        return None

    def add_member(
        self, npc_id: str, role: FactionRole = FactionRole.MEMBER,
        game_time: float = 0.0,
    ) -> FactionMember:
        existing = self.get_member(npc_id)
        if existing:
            existing.role = role
            return existing
        member = FactionMember(npc_id=npc_id, role=role, joined_at=game_time)
        self.members.append(member)
        return member

    def remove_member(self, npc_id: str) -> bool:
        for i, m in enumerate(self.members):
            if m.npc_id == npc_id:
                self.members.pop(i)
                return True
        return False

    def set_relation(self, other_faction_id: str, relation: FactionRelation) -> None:
        self.relations[other_faction_id] = relation

    def get_relation(self, other_faction_id: str) -> FactionRelation:
        return self.relations.get(other_faction_id, FactionRelation.NEUTRAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faction_id": self.faction_id,
            "name": self.name,
            "description": self.description,
            "members": [m.to_dict() for m in self.members],
            "relations": {k: v.value for k, v in self.relations.items()},
            "leader": self.leader.npc_id if self.leader else None,
            "size": len(self.members),
            "created_at": self.created_at,
        }


class FactionManager:
    """
    Manages all factions, agreements, and council mechanics.

    Provides queries for the cognition system to consider faction
    membership and inter-faction relations when making decisions.
    """

    def __init__(self) -> None:
        self._factions: dict[str, Faction] = {}
        self._agreements: dict[str, Agreement] = {}
        self._npc_faction_cache: dict[str, str] = {}  # npc_id → faction_id
        self._next_agreement_id: int = 0

    def create_faction(
        self,
        faction_id: str,
        name: str,
        description: str = "",
        leader_id: str | None = None,
        game_time: float = 0.0,
    ) -> Faction:
        """Create a new faction, optionally with a founding leader."""
        faction = Faction(
            faction_id=faction_id,
            name=name,
            description=description,
            created_at=game_time,
        )
        if leader_id:
            faction.add_member(leader_id, FactionRole.LEADER, game_time)
            self._npc_faction_cache[leader_id] = faction_id
        self._factions[faction_id] = faction
        logger.info("Faction created: %s (%s)", name, faction_id)
        return faction

    def get_faction(self, faction_id: str) -> Faction | None:
        return self._factions.get(faction_id)

    def get_npc_faction(self, npc_id: str) -> Faction | None:
        """Get the faction an NPC belongs to (if any)."""
        fid = self._npc_faction_cache.get(npc_id)
        if fid:
            return self._factions.get(fid)
        # Cache miss — linear scan
        for faction in self._factions.values():
            if faction.has_member(npc_id):
                self._npc_faction_cache[npc_id] = faction.faction_id
                return faction
        return None

    def join_faction(
        self,
        npc_id: str,
        faction_id: str,
        role: FactionRole = FactionRole.MEMBER,
        game_time: float = 0.0,
    ) -> bool:
        """Add an NPC to a faction. Returns False if faction doesn't exist."""
        faction = self._factions.get(faction_id)
        if not faction:
            return False
        # Leave current faction first
        current = self.get_npc_faction(npc_id)
        if current and current.faction_id != faction_id:
            current.remove_member(npc_id)
        faction.add_member(npc_id, role, game_time)
        self._npc_faction_cache[npc_id] = faction_id
        return True

    def leave_faction(self, npc_id: str) -> bool:
        """Remove an NPC from their current faction."""
        faction = self.get_npc_faction(npc_id)
        if not faction:
            return False
        faction.remove_member(npc_id)
        self._npc_faction_cache.pop(npc_id, None)
        return True

    def are_allies(self, npc_a: str, npc_b: str) -> bool:
        """Check if two NPCs are in allied factions."""
        fa = self.get_npc_faction(npc_a)
        fb = self.get_npc_faction(npc_b)
        if not fa or not fb:
            return False
        if fa.faction_id == fb.faction_id:
            return True
        return fa.get_relation(fb.faction_id) in (
            FactionRelation.ALLIED, FactionRelation.FRIENDLY,
        )

    def are_rivals(self, npc_a: str, npc_b: str) -> bool:
        """Check if two NPCs are in rival/hostile factions."""
        fa = self.get_npc_faction(npc_a)
        fb = self.get_npc_faction(npc_b)
        if not fa or not fb:
            return False
        return fa.get_relation(fb.faction_id) in (
            FactionRelation.RIVAL, FactionRelation.HOSTILE,
        )

    def same_faction(self, npc_a: str, npc_b: str) -> bool:
        """Check if two NPCs are in the same faction."""
        fa = self.get_npc_faction(npc_a)
        fb = self.get_npc_faction(npc_b)
        if not fa or not fb:
            return False
        return fa.faction_id == fb.faction_id

    def set_faction_relation(
        self,
        faction_a_id: str,
        faction_b_id: str,
        relation: FactionRelation,
    ) -> bool:
        """Set the relation between two factions (bidirectional)."""
        fa = self._factions.get(faction_a_id)
        fb = self._factions.get(faction_b_id)
        if not fa or not fb:
            return False
        fa.set_relation(faction_b_id, relation)
        fb.set_relation(faction_a_id, relation)
        return True

    # ---------- Agreements ----------

    def create_agreement(
        self,
        agreement_type: str,
        party_a: str,
        party_b: str,
        terms: dict[str, Any] | None = None,
        game_time: float = 0.0,
        duration: float = 0.0,
    ) -> Agreement:
        """Create a formal agreement between two parties."""
        self._next_agreement_id += 1
        aid = f"agreement_{self._next_agreement_id}"
        agreement = Agreement(
            agreement_id=aid,
            agreement_type=agreement_type,
            party_a=party_a,
            party_b=party_b,
            terms=terms or {},
            created_at=game_time,
            expires_at=game_time + duration if duration > 0 else 0.0,
        )
        self._agreements[aid] = agreement
        return agreement

    def get_agreements_for(self, party_id: str) -> list[Agreement]:
        """Get all active agreements involving a party."""
        return [
            a for a in self._agreements.values()
            if a.active and (a.party_a == party_id or a.party_b == party_id)
        ]

    def expire_agreements(self, current_time: float) -> int:
        """Deactivate expired agreements. Returns count expired."""
        expired = 0
        for agreement in self._agreements.values():
            if agreement.active and agreement.is_expired(current_time):
                agreement.active = False
                expired += 1
        return expired

    # ---------- Council / group decisions ----------

    def get_faction_vote(
        self,
        faction_id: str,
        npc_opinions: dict[str, bool],
    ) -> tuple[bool, dict[str, int]]:
        """
        Simple majority vote among faction members.

        npc_opinions: {npc_id: True/False} for members who have an opinion.
        Returns (result, {"for": N, "against": M}).
        """
        faction = self._factions.get(faction_id)
        if not faction:
            return False, {"for": 0, "against": 0}

        votes_for = 0
        votes_against = 0
        for member in faction.members:
            vote = npc_opinions.get(member.npc_id)
            if vote is None:
                continue
            # Leaders get double weight
            weight = 2 if member.role == FactionRole.LEADER else 1
            if vote:
                votes_for += weight
            else:
                votes_against += weight

        result = votes_for > votes_against
        return result, {"for": votes_for, "against": votes_against}

    # ---------- Context for cognition ----------

    def get_social_context(self, npc_id: str) -> str:
        """
        Build a natural-language summary of an NPC's faction context.
        Used to enrich LLM prompts for planning and conversation.
        """
        faction = self.get_npc_faction(npc_id)
        if not faction:
            return "You belong to no faction or formal group."

        member = faction.get_member(npc_id)
        role_desc = member.role.value if member else "member"

        parts = [f"You are a {role_desc} of {faction.name}."]

        if faction.description:
            parts.append(faction.description)

        # Fellow members
        others = [m.npc_id for m in faction.members if m.npc_id != npc_id]
        if others:
            parts.append(f"Fellow members: {len(others)} others.")

        # External relations
        for other_id, rel in faction.relations.items():
            other = self._factions.get(other_id)
            if other:
                parts.append(f"{other.name} are {rel.value}.")

        # Active agreements
        agreements = self.get_agreements_for(faction.faction_id)
        for agr in agreements[:3]:
            other_party = agr.party_b if agr.party_a == faction.faction_id else agr.party_a
            parts.append(f"Active {agr.agreement_type} with {other_party}.")

        return " ".join(parts)

    # ---------- Inspection ----------

    def get_all_factions(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self._factions.values()]

    def get_stats(self) -> dict[str, Any]:
        return {
            "faction_count": len(self._factions),
            "total_members": sum(len(f.members) for f in self._factions.values()),
            "active_agreements": sum(
                1 for a in self._agreements.values() if a.active
            ),
        }
