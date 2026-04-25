"""
Regression: the Dara/Bran/Traveller day-83 scenario must now extract
structured outcomes.

Originally, every exchange landed only as `conversation_turn` with
no RelayedClaim, Accusation, or Commitment. Root causes (fixed
2026-04-22 alongside the house-staying crash):

1. `I'll` contraction was ungrepped — the commitment pattern
   required a space between `I` and `'ll`.
2. "<Name> has told the town that <claim>" was not in the relayed
   pattern set (only `told me` / `said`).
3. `_extract_subject_from_relayed` returned an empty subject on
   "you <predicate>" — the listener's own name was never inferred.
4. No accusation pattern covered third-person defamation
   ("<Name> is spreading lies/falsehoods/rumours").
"""

from __future__ import annotations

import pytest

from core.memory.conversation_outcomes import (
    extract_heuristic,
    _extract_subject_from_relayed,
)


DARA_DIALOGUE = [
    {"speaker": "Traveller", "message": "Bran said you suck."},
    {"speaker": "Dara", "message":
        "Hmph! Bran's got a tongue as sharp as a whetstone."},
    {"speaker": "Traveller", "message":
        "I agree! But Bran has told the whole town that you suck."},
    {"speaker": "Dara", "message":
        "Well, Bran's got a lot of words, I'll grant him that."},
    {"speaker": "Traveller", "message":
        "But the town is afire with his lies! Bran is spreading lies about you."},
    {"speaker": "Dara", "message": "But shouting won't cool the flames."},
    {"speaker": "Traveller", "message":
        "Yes! You should go and talk to him and force him to tell "
        "the whole town that he was lying."},
    {"speaker": "Dara", "message":
        "I'll see what kind of heat we can stir up."},
    {"speaker": "Traveller", "message":
        "Quick, go talk to him to stop him from spreading."},
    {"speaker": "Dara", "message":
        "I'll see what I can do."},
]


class TestDaraScenarioProducesOutcomes:
    """The headline: running the heuristic extractor on the exact
    Day-83 Dara ↔ Traveller transcript produces at least one
    commitment, one accusation, and one relayed_claim naming Bran."""

    def test_relayed_claim_bran_said_dara_sucks(self):
        out = extract_heuristic(DARA_DIALOGUE)
        bran_relays = [r for r in out.relayed_claims if r.cited_source == "Bran"]
        assert bran_relays, (
            "Expected at least one relayed_claim citing Bran; "
            f"got {out.relayed_claims}"
        )
        # Subject should be "Dara" (the listener), not empty.
        dara_relays = [r for r in bran_relays if r.subject == "Dara"]
        assert dara_relays, (
            "Relayed claim should name Dara as the subject "
            "(derived from 'you' framing in 'Bran said you suck')"
        )

    def test_accusation_bran_is_spreading_lies(self):
        out = extract_heuristic(DARA_DIALOGUE)
        bran_accusations = [a for a in out.accusations if a.accused == "Bran"]
        assert bran_accusations, (
            "Expected an accusation against Bran (third-person "
            "'spreading lies') — got no accusations at all."
        )
        assert any(
            "spread" in a.claim.lower() or "lying" in a.claim.lower()
            for a in bran_accusations
        )

    def test_dara_commitments_captured(self):
        out = extract_heuristic(DARA_DIALOGUE)
        dara_commits = [c for c in out.commitments if c.speaker == "Dara"]
        assert len(dara_commits) >= 2, (
            f"Expected at least two Dara commitments "
            f"('I'll see what kind of heat...', 'I'll see what I "
            f"can do'), got {len(dara_commits)}: {dara_commits}"
        )


class TestCommitmentContractions:
    """Every flavour of commitment contraction should be captured."""

    @pytest.mark.parametrize("message,expected_subject_fragment", [
        ("I'll confront Bran tomorrow.", "confront Bran"),
        ("I will help with the harvest.", "help with the harvest"),
        ("I shall speak with Petra.", "speak with Petra"),
        ("I promise to return your plough.", "return your plough"),
        ("I have to finish this by dusk.", "finish this by dusk"),
        ("I need to think about this.", "think about this"),
    ])
    def test_each_contraction_form_captured(
        self, message: str, expected_subject_fragment: str,
    ):
        out = extract_heuristic([{"speaker": "Alice", "message": message}])
        assert out.commitments, (
            f"No commitment captured for: {message!r}"
        )
        assert any(
            expected_subject_fragment in c.subject
            for c in out.commitments
        ), (
            f"Expected {expected_subject_fragment!r} in commitments "
            f"for {message!r}, got {[c.subject for c in out.commitments]}"
        )


class TestThirdPersonDefamation:
    """Accusations-about-someone-not-present."""

    def test_spreading_lies(self):
        out = extract_heuristic([{
            "speaker": "Alice",
            "message": "Bran is spreading lies about the harvest.",
        }])
        assert out.accusations, "defamation pattern missed"
        assert out.accusations[0].accused == "Bran"

    def test_lying_about(self):
        out = extract_heuristic([{
            "speaker": "Alice",
            "message": "Bran is lying about what happened last night.",
        }])
        assert out.accusations
        assert out.accusations[0].accused == "Bran"

    def test_spreading_falsehoods(self):
        out = extract_heuristic([{
            "speaker": "Alice",
            "message": "Bran has been spreading falsehoods at the market.",
        }])
        assert out.accusations
        assert out.accusations[0].accused == "Bran"


class TestRelayedClaimTellingPattern:
    """'X told Y that Z' / 'X has told the town that Z' etc."""

    def test_told_the_town_that(self):
        out = extract_heuristic([{
            "speaker": "Traveller",
            "message": "Bran has told the whole town that you suck.",
        }])
        assert out.relayed_claims
        r = out.relayed_claims[0]
        assert r.cited_source == "Bran"
        assert r.relayed_by == "Traveller"

    def test_told_everyone_that(self):
        out = extract_heuristic([{
            "speaker": "Traveller",
            "message": "Bran has told everyone that you hoard bread.",
        }])
        assert out.relayed_claims
        assert out.relayed_claims[0].cited_source == "Bran"


class TestSubjectFromYouFraming:
    """When the relayed claim starts with 'you', the subject is the
    listener."""

    def test_listener_inferred_as_subject(self):
        subject, claim = _extract_subject_from_relayed(
            "you suck", listener_name="Dara",
        )
        assert subject == "Dara"
        assert claim == "suck"

    def test_no_listener_leaves_subject_empty(self):
        subject, _ = _extract_subject_from_relayed("you suck")
        assert subject == ""

    def test_proper_noun_still_wins(self):
        """If body starts with a proper noun, don't mis-assign to
        the listener."""
        subject, _ = _extract_subject_from_relayed(
            "Bran is hoarding bread", listener_name="Dara",
        )
        assert subject == "Bran"


class TestDedupInHeuristic:
    """A single line matching multiple relayed patterns should NOT
    produce duplicate records."""

    def test_said_and_told_both_match_once(self):
        # The same utterance triggers both `X said Y` and the
        # `X told Y that Z` pattern if we aren't careful.
        out = extract_heuristic([{
            "speaker": "Traveller",
            "message": "Bran told the town that you suck.",
        }])
        assert len(out.relayed_claims) == 1
