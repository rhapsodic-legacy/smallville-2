"""
Conversation outcome extraction.

Turns raw transcripts into structured records — commitments,
accusations, and relayed claims — so the planner and conversation
layer can act on them later. Without this layer, conversations are
just strings and the only way an NPC "knows" something new is via
keyword matching against the episodic store.

Two extraction paths share the same dataclass shape:

- **LLM extractor** (default when `llm` is supplied): one small JSON
  prompt per closed conversation. Expensive but catches subtle
  phrasing ("Petra says you've been hoarding bread" is a
  relayed_claim even when the speaker doesn't say "claim").

- **Heuristic extractor** (always runs; merges with LLM output):
  regex patterns for the obvious shapes. Cheap, runs on every
  conversation regardless of LLM availability, and covers the
  high-signal cases end-to-end without any LLM cost.

The two are merged with `merge_outcomes` — LLM hits usually dominate
and the heuristic fills gaps. Duplicates are collapsed by subject +
claim.

Phase B of MEMORY_ROADMAP.md. Downstream phases (C, D) consume the
`ConversationOutcome` records from this module — the planner's
retrieval system surfaces unresolved accusations/commitments into
subsequent conversation prompts, and cross-NPC propagation falls out
naturally once those records are named in chat.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.llm_client import LLMProvider

logger = logging.getLogger(__name__)


# ---------- Dataclasses ----------

@dataclass
class Commitment:
    """Something a speaker promised to do."""
    speaker: str           # Name of who committed
    subject: str           # What they'll do, free-form
    about: str = ""        # Optional: person/target the commitment concerns
    source_line: str = ""  # Quote from the transcript, for provenance

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "commitment",
            "speaker": self.speaker,
            "subject": self.subject,
            "about": self.about,
            "source_line": self.source_line,
        }


@dataclass
class Accusation:
    """One speaker accused another of something."""
    accuser: str
    accused: str           # Name of the accused party
    claim: str             # The alleged wrongdoing
    source_line: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "accusation",
            "accuser": self.accuser,
            "accused": self.accused,
            "claim": self.claim,
            "source_line": self.source_line,
        }


@dataclass
class RelayedClaim:
    """The killer memory shape for propagation.

    Captures "Alice said Bob is a thief" as a first-class structured
    record. The listener can later confront Alice or check with Bob —
    and the `unresolved` flag lets Phase C's retrieval prioritise it
    until the matter is discussed.
    """
    subject: str           # Who the claim is about ("Bran")
    claim: str             # The alleged thing ("is hoarding bread")
    cited_source: str      # Who is said to have made the claim ("Petra")
    relayed_by: str        # Who told the listener ("Traveller")
    unresolved: bool = True
    source_line: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "relayed_claim",
            "subject": self.subject,
            "claim": self.claim,
            "cited_source": self.cited_source,
            "relayed_by": self.relayed_by,
            "unresolved": self.unresolved,
            "source_line": self.source_line,
        }


@dataclass
class ConversationOutcome:
    """Full extraction result for one conversation."""
    commitments: list[Commitment] = field(default_factory=list)
    accusations: list[Accusation] = field(default_factory=list)
    relayed_claims: list[RelayedClaim] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.commitments or self.accusations or self.relayed_claims)

    def to_dict(self) -> dict[str, list[dict]]:
        return {
            "commitments": [c.to_dict() for c in self.commitments],
            "accusations": [a.to_dict() for a in self.accusations],
            "relayed_claims": [r.to_dict() for r in self.relayed_claims],
        }


# ---------- Merging ----------

def _commit_key(c: Commitment) -> tuple[str, str]:
    return (c.speaker.lower(), c.subject.lower().strip())


def _accusation_key(a: Accusation) -> tuple[str, str, str]:
    return (a.accuser.lower(), a.accused.lower(), a.claim.lower().strip())


def _relayed_key(r: RelayedClaim) -> tuple[str, str, str]:
    return (r.subject.lower(), r.claim.lower().strip(), r.cited_source.lower())


def merge_outcomes(
    a: ConversationOutcome, b: ConversationOutcome,
) -> ConversationOutcome:
    """Union two outcomes, dropping duplicates by semantic key.

    Used to fold the heuristic pass into the LLM pass without double
    counting.
    """
    merged = ConversationOutcome()
    seen_c: set[tuple[str, str]] = set()
    for item in a.commitments + b.commitments:
        k = _commit_key(item)
        if k not in seen_c:
            merged.commitments.append(item)
            seen_c.add(k)
    seen_a: set[tuple[str, str, str]] = set()
    for item in a.accusations + b.accusations:
        k = _accusation_key(item)
        if k not in seen_a:
            merged.accusations.append(item)
            seen_a.add(k)
    seen_r: set[tuple[str, str, str]] = set()
    for item in a.relayed_claims + b.relayed_claims:
        k = _relayed_key(item)
        if k not in seen_r:
            merged.relayed_claims.append(item)
            seen_r.add(k)
    return merged


# ---------- Heuristic extractor ----------
#
# Deliberately conservative. False negatives are fine — the LLM pass
# picks up subtleties. False positives would poison the memory layer
# with phantom commitments, so patterns only match strong signals.

_COMMIT_PATTERNS = [
    # "I will <verb> ..." / "I shall <verb> ..." / "I promise to ..."
    # Separate pattern from the contraction so the regex engine
    # doesn't need `\s+` between `I` and the alternation — that's
    # the gap that swallowed "I'll" contractions in Dara's day-83
    # exchange with Traveller (every "I'll see ..." / "I'll do ..."
    # missed by the earlier pattern).
    re.compile(r"\bI\s+(?:will|shall|promise to|swear to)\s+([^.!?]+)", re.I),
    # Contraction: "I'll <verb> ..." (no space between I and 'll).
    re.compile(r"\bI'll\s+([^.!?]+)", re.I),
    # "I should / ought to / have to / must ..." — softer signals.
    re.compile(r"\bI\s+(?:ought to|have to|must|need to)\s+([^.!?]+)", re.I),
    # "I'll see / I'll do / I'll try ..." — soft commitment after
    # an ellipsis or mid-sentence. Captured separately so the
    # claim substring starts from "see what I can do ..." etc.
    # rather than the verb alone.
]

_ACCUSE_PATTERNS = [
    # "You are a liar / a thief / guilty of X"
    re.compile(
        r"\byou\s+(?:are|'re)\s+(?:a\s+)?"
        r"(liar|thief|traitor|coward|cheat|murderer|monster)",
        re.I,
    ),
    # "You stole / You killed / You betrayed …"
    re.compile(
        r"\byou\s+(stole|killed|betrayed|cheated|robbed|lied\s+to|attacked)\s+([^.!?]+)",
        re.I,
    ),
    # "You hoard / you have been hoarding …"
    re.compile(r"\byou\s+(?:have been\s+)?(hoard\w*|steal\w*|lying)\s+([^.!?]+)?", re.I),
    # "I accuse you of …" (+ "accusing me of …")
    re.compile(r"\bI\s+accuse\s+you\s+of\s+([^.!?]+)", re.I),
    # Third-person defamation: "<Name> is spreading lies / falsehoods / rumours"
    # also "<Name> is lying about / has been lying about".
    # Captures "<Name>" in group 1 so we can treat it as an
    # accusation *against* that named person with the accused
    # inferred from the captured name rather than the prior-speaker
    # heuristic (which assumes "you" framing).
    re.compile(
        r"\b([A-Z][a-zA-Z]+)\s+(?:is|was|has been|'s)\s+"
        r"(spread\w*\s+(?:lies|falsehoods|rumours|gossip)"
        r"|lying(?:\s+about\s+[^.!?]+)?)",
    ),
]

# Two shapes for "X told / said / says Y ..." attributions.
# NOTE: these patterns are case-sensitive — re.I would let common
# words like "never said ..." masquerade as proper-noun citations
# because `[A-Z]` collapses to `[A-Za-z]` under ignore-case.
_RELAYED_PATTERNS = [
    re.compile(
        r"\b([A-Z][a-zA-Z]+)\s+"
        r"(?:said|says|told me|mentioned|claimed|reckons?|reckoned)\s+"
        r"(?:that\s+)?(.+?)(?=[.!?]|$)",
    ),
    # "X has told <object> that Y" / "X told the town that Y". The
    # object phrase is anything short and non-sentence-ending.
    # Catches what the older `told me` pattern missed in the Dara
    # scenario: "Bran has told the whole town that you suck".
    re.compile(
        r"\b([A-Z][a-zA-Z]+)\s+(?:has\s+|had\s+)?told\s+"
        r"(?:the\s+[a-zA-Z ]+?|anyone|everyone|people|folks?)\s+"
        r"that\s+(.+?)(?=[.!?]|$)",
    ),
    # Inline attribution: "... — Petra said so."
    re.compile(r"([A-Z][a-zA-Z]+)\s+said\s+so\b"),
]


def _strip_trailing(s: str) -> str:
    return s.strip().rstrip(",;:.!?").strip()


def _extract_subject_from_relayed(
    body: str, listener_name: str = "",
) -> tuple[str, str]:
    """Split a relayed-claim body into (subject, claim).

    "Bran has been hoarding bread" → ("Bran", "has been hoarding bread")
    "you suck"                     → (listener_name, "suck")
    "he's a liar"                  → ("", "he's a liar")  (subject unknown)
    Returns empty subject only when no proper noun OR "you"-framing
    is present. `listener_name` is the conversation partner of the
    speaker who's doing the relaying — when the relayed claim starts
    with "you", it's about the listener.
    """
    body = body.strip()
    # "you <verb>" / "you <adj>" — the speaker is relaying a claim
    # about the listener. Subject = the listener.
    m = re.match(r"you\s+(.+)$", body, re.I)
    if m and listener_name:
        return (listener_name, m.group(1).strip())
    # Match leading proper-noun + verb, grab rest as claim.
    m = re.match(r"([A-Z][a-zA-Z]+)\s+(is|was|has|had|'s|does|did)\b(.*)", body)
    if m:
        subject = m.group(1)
        claim = f"{m.group(2)}{m.group(3)}".strip()
        return (subject, claim)
    # "that <subject> ..." forms occasionally slip through; try again.
    m = re.match(r"([A-Z][a-zA-Z]+)\s+(.*)", body)
    if m:
        return (m.group(1), m.group(2).strip())
    return ("", body)


def extract_heuristic(
    exchanges: list[dict[str, str]],
) -> ConversationOutcome:
    """Regex-only extraction. Safe to run on every conversation.

    Accepts the same exchange-dict shape `record_conversation` uses:
    ``[{"speaker": name, "message": text}, …]``. Returns a
    ConversationOutcome with whatever structure the patterns fire on.

    Phase B.3+ (2026-04-22): handles "I'll" contractions, third-
    person defamation ("Bran is spreading lies"), "X has told the
    town that Y", and "you"-framed relayed claims ("Bran said you
    suck" → listener is the subject).
    """
    outcome = ConversationOutcome()

    # Precompute the set of participants so each speaker's listener
    # (in a 2-person conversation) can be inferred from the rest of
    # the cast. For larger conversations we fall back to the most
    # recent non-self speaker, which matches the Stanford model.
    participants_ordered: list[str] = []
    for ex in exchanges:
        sp = (ex.get("speaker") or "").strip()
        if sp and sp not in participants_ordered:
            participants_ordered.append(sp)

    def _listener_for(speaker: str, idx: int) -> str:
        """Best-effort inference of who this speaker is addressing."""
        others = [p for p in participants_ordered if p != speaker]
        if len(others) == 1:
            return others[0]
        # Fall back to the most recent prior non-self speaker.
        for j in range(idx - 1, -1, -1):
            prev = (exchanges[j].get("speaker") or "").strip()
            if prev and prev != speaker:
                return prev
        return ""

    for idx, ex in enumerate(exchanges):
        speaker = (ex.get("speaker") or "").strip()
        message = (ex.get("message") or "").strip()
        if not message:
            continue

        listener = _listener_for(speaker, idx)

        # --- Commitments ---
        for pat in _COMMIT_PATTERNS:
            for m in pat.finditer(message):
                subject = _strip_trailing(m.group(1))
                # Drop obviously broken captures ("I'll..." alone is
                # 2 chars) but keep short real verbs like "I'll go"
                # (6 chars) when they make it through.
                if len(subject) < 3:
                    continue
                outcome.commitments.append(Commitment(
                    speaker=speaker,
                    subject=subject,
                    source_line=message,
                ))

        # --- Accusations ---
        # For "you"-framed patterns the accused is the prior speaker.
        # For "<Name> is spreading lies"-framed patterns (group 0 is
        # the name) the accused is whoever the pattern NAMED.
        prior_speaker = ""
        if idx > 0:
            prior_speaker = (exchanges[idx - 1].get("speaker") or "").strip()
        for pat in _ACCUSE_PATTERNS:
            for m in pat.finditer(message):
                groups = m.groups()
                # Defamation/third-person: leading group is a proper
                # noun (capitalised). Detect by peeking at the raw
                # text: if it starts with "<Name> ", treat group 0
                # as the accused.
                if (
                    groups and groups[0]
                    and groups[0][:1].isupper()
                    and groups[0].strip() != speaker
                ):
                    accused = groups[0].strip()
                    claim = " ".join(g for g in groups[1:] if g).strip()
                else:
                    accused = prior_speaker or "someone"
                    claim = " ".join(g for g in groups if g).strip()
                claim = _strip_trailing(claim)
                if not claim:
                    continue
                outcome.accusations.append(Accusation(
                    accuser=speaker,
                    accused=accused,
                    claim=claim,
                    source_line=message,
                ))

        # --- Relayed claims ---
        for pat in _RELAYED_PATTERNS:
            for m in pat.finditer(message):
                groups = m.groups()
                cited = groups[0].strip() if groups else ""
                # Skip if the cited source is the speaker themselves —
                # that's a commitment/declaration, not a relayed claim.
                if cited.lower() == speaker.lower():
                    continue
                body = groups[1].strip() if len(groups) > 1 else ""
                if not body:
                    continue
                subject, claim = _extract_subject_from_relayed(
                    body, listener_name=listener,
                )
                claim = _strip_trailing(claim)
                if not claim:
                    continue
                outcome.relayed_claims.append(RelayedClaim(
                    subject=subject,
                    claim=claim,
                    cited_source=cited,
                    relayed_by=speaker,
                    source_line=message,
                ))

    # A single utterance can hit multiple relayed-claim patterns
    # ("Bran said you suck" + "Bran has told the town that you suck"
    # share the same (subject, claim, cited_source) triple). Run the
    # usual dedup to keep each outcome unique by semantic key.
    return merge_outcomes(ConversationOutcome(), outcome)


# ---------- LLM extractor ----------

_LLM_EXTRACT_PROMPT = (
    "Extract structured outcomes from this medieval village conversation.\n\n"
    "Return ONLY valid JSON with these keys:\n"
    "- commitments: [{speaker, subject, about}] — what each speaker "
    "promised to do. `about` is the person/thing it concerns (may be empty).\n"
    "- accusations: [{accuser, accused, claim}] — explicit accusations.\n"
    "- relayed_claims: [{subject, claim, cited_source, relayed_by}] — "
    "second-hand claims of the form 'X said Y is/did Z'. subject=Y, "
    "claim=\"is/did Z\", cited_source=X, relayed_by=the speaker.\n\n"
    "Use empty arrays when nothing qualifies. Do not invent facts. "
    "Use the speaker names exactly as they appear.\n\n"
    "Transcript:\n{transcript}\n\n"
    "JSON:"
)


def _format_transcript(exchanges: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{e.get('speaker', '?')}: {e.get('message', '')}"
        for e in exchanges
        if (e.get("message") or "").strip()
    )


def _parse_llm_json(response: str) -> ConversationOutcome:
    """Parse the extractor's JSON response defensively.

    LLMs occasionally wrap JSON in code fences or add trailing prose.
    Strip the first balanced brace block we find and decode that.
    """
    text = response.strip()
    # Drop common wrappers.
    if text.startswith("```"):
        text = text.strip("`")
        # Remove an optional language tag on the first line.
        if "\n" in text:
            text = text.split("\n", 1)[1]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ConversationOutcome()

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.debug("Outcome JSON parse failed: %s", e)
        return ConversationOutcome()

    outcome = ConversationOutcome()
    for c in data.get("commitments", []) or []:
        if not isinstance(c, dict):
            continue
        outcome.commitments.append(Commitment(
            speaker=str(c.get("speaker", "")),
            subject=str(c.get("subject", "")),
            about=str(c.get("about", "") or ""),
            source_line="",
        ))
    for a in data.get("accusations", []) or []:
        if not isinstance(a, dict):
            continue
        outcome.accusations.append(Accusation(
            accuser=str(a.get("accuser", "")),
            accused=str(a.get("accused", "")),
            claim=str(a.get("claim", "")),
            source_line="",
        ))
    for r in data.get("relayed_claims", []) or []:
        if not isinstance(r, dict):
            continue
        outcome.relayed_claims.append(RelayedClaim(
            subject=str(r.get("subject", "")),
            claim=str(r.get("claim", "")),
            cited_source=str(r.get("cited_source", "")),
            relayed_by=str(r.get("relayed_by", "")),
            source_line="",
        ))

    # Drop entries with empty subject/claim/etc — the LLM sometimes
    # returns skeletal records.
    outcome.commitments = [c for c in outcome.commitments if c.speaker and c.subject]
    outcome.accusations = [a for a in outcome.accusations if a.accuser and a.claim]
    outcome.relayed_claims = [
        r for r in outcome.relayed_claims
        if r.claim and r.cited_source and r.relayed_by
    ]
    return outcome


async def extract_with_llm(
    exchanges: list[dict[str, str]],
    llm: "LLMProvider",
) -> ConversationOutcome:
    """LLM path. Returns an empty outcome on any failure — caller
    should already have a heuristic pass in its pocket."""
    transcript = _format_transcript(exchanges)
    if not transcript:
        return ConversationOutcome()

    try:
        response = await llm.complete(
            system=(
                "You extract structured social outcomes from medieval "
                "village conversations. You reply in valid JSON only."
            ),
            messages=[{
                "role": "user",
                "content": _LLM_EXTRACT_PROMPT.format(transcript=transcript),
            }],
            max_tokens=400,
            temperature=0.2,
            purpose="outcome_extraction",
        )
    except Exception as e:
        logger.debug("LLM outcome extraction failed: %s", e)
        return ConversationOutcome()

    return _parse_llm_json(response)


# ---------- Top-level orchestrator ----------

async def extract_outcomes(
    exchanges: list[dict[str, str]],
    llm: "LLMProvider | None" = None,
) -> ConversationOutcome:
    """Produce a merged heuristic + LLM ConversationOutcome.

    Callers pass `llm=None` when tight on budget or running a pure
    deterministic test — the heuristic path alone still covers the
    loud cases (explicit accusations, "I promise", "X said Y").
    """
    heuristic = extract_heuristic(exchanges)
    if llm is None:
        return heuristic
    llm_out = await extract_with_llm(exchanges, llm)
    return merge_outcomes(llm_out, heuristic)
