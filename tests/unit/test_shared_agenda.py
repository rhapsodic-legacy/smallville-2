"""
Phase G — shared-agenda prompt cue.

Covers:
- TownAgenda.shared_matters_for_prompt surfaces a line when both
  NPCs are contributors to the same active goal.
- It surfaces a one-sided invitation when only the partner is
  contributing.
- It surfaces a recent-victory line within RECENT_VICTORY_DAYS and
  goes quiet after the window.
- The RECENT_VICTORY_DAYS window is 1 day.
- summary_for_prompt (no partner) is unchanged.
"""

from __future__ import annotations

import pytest

from core.world.town_agenda import (
    GoalStatus, TownAgenda, create_goal_from_template,
)


def _propose_bridge(agenda: TownAgenda, day: int = 1) -> None:
    goal = create_goal_from_template("repair_bridge", current_day=day)
    assert goal is not None
    agenda.propose(goal, current_day=day)


def _propose_festival(agenda: TownAgenda, day: int = 1) -> None:
    goal = create_goal_from_template("harvest_festival", current_day=day)
    assert goal is not None
    agenda.propose(goal, current_day=day)


class TestSharedMattersForPrompt:
    def test_empty_when_no_agenda(self):
        agenda = TownAgenda()
        assert agenda.shared_matters_for_prompt("a", "b", current_day=1) == ""

    def test_both_committed_surfaces_joint_line(self):
        agenda = TownAgenda()
        _propose_bridge(agenda)
        agenda.record_contribution("repair_bridge", "alice", current_day=1)
        agenda.record_contribution("repair_bridge", "bran", current_day=1)

        out = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=1,
        )
        assert "you and your partner are both helping" in out.lower()
        assert "bridge" in out.lower()

    def test_partner_only_is_invitation(self):
        agenda = TownAgenda()
        _propose_bridge(agenda)
        agenda.record_contribution("repair_bridge", "bran", current_day=1)

        out = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=1,
        )
        assert "your partner is helping" in out.lower()
        assert "both" not in out.lower()

    def test_self_only_is_silent(self):
        agenda = TownAgenda()
        _propose_bridge(agenda)
        agenda.record_contribution("repair_bridge", "alice", current_day=1)

        # Alice is committed but Bran isn't — no line.
        out = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=1,
        )
        assert out == ""

    def test_recent_shared_victory_surfaces(self):
        agenda = TownAgenda()
        _propose_bridge(agenda)
        # Force completion by filling contributions.
        goal = agenda.get("repair_bridge")
        goal.required_contributions = 2
        agenda.record_contribution("repair_bridge", "alice", current_day=2)
        agenda.record_contribution("repair_bridge", "bran", current_day=2)
        assert goal.status == GoalStatus.COMPLETED
        assert goal.completed_day == 2

        # Same day — surfaces.
        out_same = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=2,
        )
        assert "recently completed" in out_same.lower()

        # Within 1-day window — surfaces.
        out_next = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=3,
        )
        assert "recently completed" in out_next.lower()

        # Outside window — silent.
        out_later = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=10,
        )
        assert "recently completed" not in out_later.lower()

    def test_empty_when_either_id_missing(self):
        agenda = TownAgenda()
        _propose_bridge(agenda)
        agenda.record_contribution("repair_bridge", "alice", current_day=1)
        agenda.record_contribution("repair_bridge", "bran", current_day=1)

        assert agenda.shared_matters_for_prompt("", "bran", current_day=1) == ""
        assert agenda.shared_matters_for_prompt("alice", "", current_day=1) == ""

    def test_summary_for_prompt_unchanged(self):
        """summary_for_prompt still works when no partner is given."""
        agenda = TownAgenda()
        _propose_bridge(agenda)
        out = agenda.summary_for_prompt("alice")
        assert "Town matters" in out
        assert "Repair the old bridge" in out

    def test_completed_day_stamped_with_current_day(self):
        agenda = TownAgenda()
        _propose_bridge(agenda)
        goal = agenda.get("repair_bridge")
        goal.required_contributions = 1
        agenda.record_contribution("repair_bridge", "alice", current_day=7)
        assert goal.completed_day == 7

    def test_completed_day_fallback_when_day_not_passed(self):
        """Old callers that don't thread current_day still get a
        bounded completion timestamp."""
        agenda = TownAgenda()
        _propose_bridge(agenda, day=1)
        goal = agenda.get("repair_bridge")
        goal.required_contributions = 1
        agenda.record_contribution("repair_bridge", "alice")
        # Fallback used goal.deadline_day (1 + 3 = 4).
        assert goal.completed_day == goal.deadline_day


class TestMultipleGoalsInPrompt:
    def test_multiple_shared_goals_all_listed(self):
        agenda = TownAgenda()
        _propose_bridge(agenda, day=1)
        _propose_festival(agenda, day=1)
        for goal_id in ("repair_bridge", "harvest_festival"):
            agenda.record_contribution(goal_id, "alice", current_day=1)
            agenda.record_contribution(goal_id, "bran", current_day=1)

        out = agenda.shared_matters_for_prompt(
            "alice", "bran", current_day=1,
        )
        assert "bridge" in out.lower()
        assert "festival" in out.lower()
