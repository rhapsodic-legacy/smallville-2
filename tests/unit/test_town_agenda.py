"""
Tests for the TownAgenda — the collective-goal system.

Covers: goal creation from templates, personality matching, contribution
progress, completion & cooldown, expiry, and serialisation.
"""

import pytest

from core.npc.models import NPC, PersonalityTraits
from core.world.town_agenda import (
    TownAgenda, TownGoal, GoalTemplate, GoalStatus,
    TEMPLATES, register_template, create_goal_from_template,
)


def _make_npc(
    npc_id: str = "test_0",
    extraversion: float = 0.5,
    conscientiousness: float = 0.5,
    agreeableness: float = 0.5,
    openness: float = 0.5,
    neuroticism: float = 0.3,
) -> NPC:
    return NPC(
        npc_id=npc_id,
        name=npc_id.capitalize(),
        age=30,
        personality=PersonalityTraits(
            extraversion=extraversion,
            conscientiousness=conscientiousness,
            agreeableness=agreeableness,
            openness=openness,
            neuroticism=neuroticism,
        ),
        backstory="Test.",
        occupation="labourer",
        x=0, z=0,
        home_x=0, home_z=0,
    )


class TestTemplates:

    def test_builtin_templates_exist(self):
        assert "harvest_festival" in TEMPLATES
        assert "repair_bridge" in TEMPLATES
        assert "town_council" in TEMPLATES

    def test_register_template(self):
        register_template(GoalTemplate(
            goal_id="test_custom",
            title="Custom goal",
            description="...",
            activity_text="do custom work",
            location_hint="town_square",
        ))
        assert "test_custom" in TEMPLATES
        # Cleanup
        TEMPLATES.pop("test_custom", None)

    def test_create_goal_from_template(self):
        g = create_goal_from_template("harvest_festival", current_day=5)
        assert g is not None
        assert g.goal_id == "harvest_festival"
        assert g.deadline_day == 5 + TEMPLATES["harvest_festival"].deadline_days
        assert g.created_day == 5
        assert g.status == GoalStatus.PROPOSED

    def test_unknown_template_returns_none(self):
        assert create_goal_from_template("no_such_goal", current_day=1) is None


class TestParticipationScore:
    """Scored eligibility — personality alignment + self_concept pulls.

    Replaces the old boolean `matches_personality`. Score is deterministic
    and continuous; `should_participate` layers RNG sampling on top.
    """

    def test_festival_favours_extraverts(self):
        goal = create_goal_from_template("harvest_festival", 1)
        extravert = _make_npc(extraversion=0.8)
        introvert = _make_npc(extraversion=0.2)
        assert goal.participation_score(extravert) > 0
        assert goal.participation_score(introvert) < 0
        # Sigmoid mapping preserves sign.
        assert goal.participation_probability(extravert) > 0.5
        assert goal.participation_probability(introvert) < 0.5

    def test_repair_favours_conscientious(self):
        goal = create_goal_from_template("repair_bridge", 1)
        careful = _make_npc(conscientiousness=0.8)
        careless = _make_npc(conscientiousness=0.2)
        assert goal.participation_score(careful) == pytest.approx(0.3)
        assert goal.participation_score(careless) == pytest.approx(-0.3)

    def test_council_sums_traits(self):
        goal = create_goal_from_template("town_council", 1)
        diplomat = _make_npc(agreeableness=0.8, openness=0.8)
        cynic = _make_npc(agreeableness=0.1, openness=0.1)
        # Two biased traits, each contributing (value - threshold).
        assert goal.participation_score(diplomat) == pytest.approx(0.8)
        assert goal.participation_score(cynic) == pytest.approx(-0.6)

    def test_empty_bias_neutral_score(self):
        goal = TownGoal(
            goal_id="universal",
            title="Universal",
            description="",
            activity_text="help",
            location_hint="town_square",
            duration_minutes=60,
            required_contributions=1,
            deadline_day=10,
            personality_bias={},
            created_day=1,
        )
        # Zero pull from personality; no supports/opposes; score = 0.
        for extraversion in (0.1, 0.5, 0.9):
            assert goal.participation_score(
                _make_npc(extraversion=extraversion)
            ) == pytest.approx(0.0)
            assert goal.participation_probability(
                _make_npc(extraversion=extraversion)
            ) == pytest.approx(0.5)

    def test_opposes_pulls_score_negative(self):
        goal = create_goal_from_template("repair_bridge", 1)
        npc = _make_npc(conscientiousness=0.7)
        base = goal.participation_score(npc)
        npc.self_concept["opposes:repair_bridge"] = 0.9
        opposed = goal.participation_score(npc)
        assert opposed == pytest.approx(base - 0.9)

    def test_supports_pulls_score_positive(self):
        goal = create_goal_from_template("repair_bridge", 1)
        npc = _make_npc(conscientiousness=0.7)
        base = goal.participation_score(npc)
        npc.self_concept["supports:repair_bridge"] = 0.6
        supported = goal.participation_score(npc)
        assert supported == pytest.approx(base + 0.6)

    def test_should_participate_samples_probability(self):
        """Frequency over N rolls stays within tolerance of probability."""
        import random as _random
        goal = create_goal_from_template("repair_bridge", 1)
        npc = _make_npc(conscientiousness=0.8)
        p = goal.participation_probability(npc)
        rng = _random.Random(12345)
        N = 2000
        hits = sum(goal.should_participate(npc, rng) for _ in range(N))
        observed = hits / N
        # 2000 rolls — binomial stderr bounded well under 0.02 for p in [0.5, 0.9].
        assert abs(observed - p) < 0.04

    def test_objector_occasionally_helps(self):
        """The defining edge case: opposes:=0.9 still yields rare True samples."""
        import random as _random
        goal = create_goal_from_template("repair_bridge", 1)
        npc = _make_npc(conscientiousness=0.7)
        npc.self_concept["opposes:repair_bridge"] = 0.9
        p = goal.participation_probability(npc)
        # Mostly declines but not impossible — the "begrudging help" case.
        assert 0.0 < p < 0.25
        rng = _random.Random(99)
        hits = sum(goal.should_participate(npc, rng) for _ in range(1000))
        assert 0 < hits < 300  # non-zero but clearly a minority


class TestContributions:

    def test_first_contribution_activates_goal(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)

        completed = agenda.record_contribution("harvest_festival", "alice")
        assert not completed
        assert goal.status == GoalStatus.ACTIVE
        assert goal.progress == 1

    def test_contributions_accumulate_and_complete(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)

        results = [
            agenda.record_contribution("harvest_festival", f"npc_{i}")
            for i in range(goal.required_contributions)
        ]
        # Only the LAST contribution should report completion.
        assert results[-1] is True
        assert all(not r for r in results[:-1])
        assert goal.status == GoalStatus.COMPLETED
        assert len(goal.contributors) == goal.required_contributions

    def test_contribution_after_completion_is_noop(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)
        for i in range(goal.required_contributions):
            agenda.record_contribution("harvest_festival", f"npc_{i}")

        # Extra contribution: status stays COMPLETED, no crash.
        assert not agenda.record_contribution("harvest_festival", "latecomer")
        assert goal.status == GoalStatus.COMPLETED

    def test_completion_fires_listener(self):
        calls = []
        agenda = TownAgenda()
        agenda.add_completion_listener(lambda g: calls.append(g.goal_id))

        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)
        for i in range(goal.required_contributions):
            agenda.record_contribution("harvest_festival", f"npc_{i}")

        assert calls == ["harvest_festival"]


class _FixedRng:
    """Minimal rng stub: always returns `value` from `random()`.

    `0.0` forces every sampled gate to succeed; `1.0` forces every one
    to fail. Lets the matching tests isolate the ordering/skip logic
    from the new probabilistic eligibility.
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


class TestMatching:

    def test_matching_goal_skips_already_contributed(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)
        npc = _make_npc("extravert", extraversion=0.8)
        rng = _FixedRng(0.0)  # force sample success

        first = agenda.matching_goal_for(npc, rng)
        assert first is goal
        agenda.record_contribution(goal.goal_id, npc.npc_id)
        assert agenda.matching_goal_for(npc, rng) is None

    def test_matching_goal_rarely_picks_mismatched_personality(self):
        """Strong personality mismatch → low probability → FixedRng(1.0) declines."""
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)
        introvert = _make_npc("introvert", extraversion=0.1)
        # Introvert's probability is low but non-zero; with rolls forced
        # to the high end of [0, 1), the sample fails and no goal matches.
        assert goal.participation_probability(introvert) < 0.3
        assert agenda.matching_goal_for(introvert, _FixedRng(0.99)) is None

    def test_urgent_deadline_wins(self):
        """When multiple goals match, the one closest to its deadline wins."""
        agenda = TownAgenda()
        far = create_goal_from_template("harvest_festival", 1)
        near = create_goal_from_template("repair_bridge", 1)
        near.deadline_day = 2  # manually set very near
        agenda.propose(far, 1)
        agenda.propose(near, 1)

        npc = _make_npc(extraversion=0.8, conscientiousness=0.8)
        pick = agenda.matching_goal_for(npc, _FixedRng(0.0))
        assert pick is near


class TestExpiry:

    def test_expire_overdue(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)

        # Before deadline, nothing expires.
        assert agenda.expire_overdue(goal.deadline_day) == []

        # Past deadline, marked EXPIRED.
        newly = agenda.expire_overdue(goal.deadline_day + 1)
        assert len(newly) == 1
        assert newly[0].status == GoalStatus.EXPIRED


class TestCooldown:

    def test_completed_goal_goes_on_cooldown(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)
        for i in range(goal.required_contributions):
            agenda.record_contribution("harvest_festival", f"npc_{i}")

        # Immediately after completion, same template can't be re-proposed.
        fresh = create_goal_from_template("harvest_festival", 2)
        assert not agenda.propose(fresh, 2)

        # Well past the cooldown window, it can.
        later = create_goal_from_template("harvest_festival", 2)
        assert agenda.propose(later, goal.deadline_day + 10)

    def test_active_goal_not_duplicated(self):
        """Proposing the same template while one is already active is a no-op."""
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        assert agenda.propose(goal, 1)
        duplicate = create_goal_from_template("harvest_festival", 2)
        assert not agenda.propose(duplicate, 2)


class TestSerialisation:

    def test_to_dict_shape(self):
        agenda = TownAgenda()
        goal = create_goal_from_template("harvest_festival", 1)
        agenda.propose(goal, 1)

        d = agenda.to_dict()
        assert "active" in d
        assert "completed_recent" in d
        assert len(d["active"]) == 1
        entry = d["active"][0]
        for field in ("goal_id", "title", "description", "progress",
                      "required_contributions", "status", "contributors",
                      "deadline_day", "activity_text", "location_hint"):
            assert field in entry, f"Missing field in serialised goal: {field}"
