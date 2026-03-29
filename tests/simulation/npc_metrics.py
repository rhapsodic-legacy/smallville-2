"""
NPC Metrics Tracker — rich per-NPC diagnostics for simulation tests.

Tracks every NPC's activity breakdown, movement, needs history,
contributions, and life balance. Designed to answer:
  - What was each NPC doing and for how long?
  - Were NPCs living balanced lives or robot-like grind?
  - Who was the outlier and why?
  - Did everyone eat, sleep, socialise, and contribute?

Usage:
    tracker = NPCMetricsTracker(npcs)
    # Each tick:
    tracker.sample(npcs, clock, construction_sites)
    # After simulation:
    tracker.print_report()
    outliers = tracker.get_outliers()
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.npc.models import NPC, ActivityState


@dataclass
class NPCProfile:
    """Accumulated metrics for a single NPC across the simulation."""
    npc_id: str
    name: str
    occupation: str

    # Activity time budget (samples in each state)
    activity_counts: Counter = field(default_factory=Counter)

    # Needs history (sampled periodically)
    energy_samples: list[float] = field(default_factory=list)
    hunger_samples: list[float] = field(default_factory=list)

    # Position history
    positions: list[tuple[float, float]] = field(default_factory=list)
    unique_tiles_visited: set[tuple[int, int]] = field(default_factory=set)

    # Construction contributions
    construction_samples: int = 0  # ticks spent constructing
    resources_contributed: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Action descriptions (for variety tracking)
    action_descriptions: Counter = field(default_factory=Counter)

    # Social interactions
    conversation_samples: int = 0

    # Life events
    times_exhausted: int = 0  # energy < 0.05
    times_starving: int = 0   # hunger > 0.95
    times_slept: int = 0      # sleeping state transitions
    times_ate: int = 0        # eating state transitions

    # Previous state (for transition detection)
    _prev_activity: str = ""

    @property
    def total_samples(self) -> int:
        return sum(self.activity_counts.values())

    @property
    def activity_pcts(self) -> dict[str, float]:
        """Activity percentages as 0-100 values."""
        total = self.total_samples
        if total == 0:
            return {}
        return {k: (v / total) * 100 for k, v in self.activity_counts.items()}

    @property
    def avg_energy(self) -> float:
        return statistics.mean(self.energy_samples) if self.energy_samples else 0

    @property
    def avg_hunger(self) -> float:
        return statistics.mean(self.hunger_samples) if self.hunger_samples else 0

    @property
    def movement_range(self) -> float:
        """Max Manhattan distance between any two sampled positions."""
        if len(self.positions) < 2:
            return 0.0
        xs = [p[0] for p in self.positions]
        zs = [p[1] for p in self.positions]
        return (max(xs) - min(xs)) + (max(zs) - min(zs))

    @property
    def life_balance_score(self) -> float:
        """
        0-100 score measuring how balanced the NPC's life is.

        Penalises extremes: all-work, all-idle, no-sleep, no-eating.
        A balanced NPC works ~40%, sleeps ~25%, eats ~10%, socialises ~10%,
        and has some idle/walking time.
        """
        pcts = self.activity_pcts
        score = 100.0

        # Penalise no sleep
        sleep_pct = pcts.get("sleeping", 0)
        if sleep_pct < 5:
            score -= 30
        elif sleep_pct < 15:
            score -= 10

        # Penalise no eating
        eat_pct = pcts.get("eating", 0)
        if eat_pct < 1:
            score -= 20
        elif eat_pct < 5:
            score -= 5

        # Penalise too much idle (>50% idle = problem)
        idle_pct = pcts.get("idle", 0)
        if idle_pct > 50:
            score -= 25
        elif idle_pct > 35:
            score -= 10

        # Penalise never moving (stuck)
        if self.movement_range < 3:
            score -= 15

        # Penalise constant exhaustion
        if self.avg_energy < 0.15:
            score -= 20

        return max(0, score)


class NPCMetricsTracker:
    """Tracks per-NPC metrics across a simulation run."""

    def __init__(self, npcs: list[NPC]) -> None:
        self.profiles: dict[str, NPCProfile] = {}
        for npc in npcs:
            self.profiles[npc.npc_id] = NPCProfile(
                npc_id=npc.npc_id,
                name=npc.name,
                occupation=npc.occupation,
            )
        self._sample_count = 0
        self._sync_scores: list[float] = []

    def sample(
        self,
        npcs: list[NPC],
        day: int = 0,
        game_minutes: float = 0,
        construction_sites: list[dict] | None = None,
    ) -> None:
        """Record one sample point for all NPCs."""
        self._sample_count += 1
        activities = []

        for npc in npcs:
            p = self.profiles.get(npc.npc_id)
            if p is None:
                continue

            state = npc.activity.value
            activities.append(state)

            # Activity budget
            p.activity_counts[state] += 1

            # Needs
            p.energy_samples.append(npc.energy)
            p.hunger_samples.append(npc.hunger)

            # Position
            p.positions.append((npc.x, npc.z))
            p.unique_tiles_visited.add((round(npc.x), round(npc.z)))

            # Description variety
            desc = getattr(npc, "current_action_description", "") or ""
            if desc:
                # Normalise to first few words for grouping
                key = " ".join(desc.split()[:4]).lower()
                p.action_descriptions[key] += 1

            # Construction detection
            if "construct" in desc.lower():
                p.construction_samples += 1

            # Social detection
            if npc.conversation_partner:
                p.conversation_samples += 1

            # Life event detection (state transitions)
            if state != p._prev_activity:
                if state == "sleeping":
                    p.times_slept += 1
                if state == "eating":
                    p.times_ate += 1
            p._prev_activity = state

            # Crisis detection
            if npc.energy < 0.05:
                p.times_exhausted += 1
            if npc.hunger > 0.95:
                p.times_starving += 1

        # Sync score
        if len(activities) >= 2:
            counts = Counter(activities)
            n = len(activities)
            pairs_same = sum(c * (c - 1) // 2 for c in counts.values())
            total_pairs = n * (n - 1) // 2
            self._sync_scores.append(
                pairs_same / total_pairs if total_pairs > 0 else 0
            )

    def track_resource_contribution(
        self, npc_id: str, resource: str, amount: int,
    ) -> None:
        """Track a resource contribution event."""
        p = self.profiles.get(npc_id)
        if p:
            p.resources_contributed[resource] += amount

    # ---------- Reports ----------

    def print_per_npc_report(self) -> None:
        """Detailed per-NPC activity breakdown."""
        print("\n--- PER-NPC ACTIVITY BREAKDOWN ---")
        header = (
            f"{'Name':12s} {'Occupation':12s} "
            f"{'Work%':>6s} {'Sleep%':>7s} {'Eat%':>5s} "
            f"{'Talk%':>6s} {'Walk%':>6s} {'Idle%':>6s} "
            f"{'AvgE':>5s} {'AvgH':>5s} {'Tiles':>6s} {'Balance':>8s}"
        )
        print(header)
        print("-" * len(header))

        for p in sorted(self.profiles.values(), key=lambda p: p.name):
            pcts = p.activity_pcts
            print(
                f"{p.name:12s} {p.occupation:12s} "
                f"{pcts.get('working', 0):5.1f}% {pcts.get('sleeping', 0):5.1f}% "
                f"{pcts.get('eating', 0):4.1f}% {pcts.get('talking', 0):5.1f}% "
                f"{pcts.get('walking', 0):5.1f}% {pcts.get('idle', 0):5.1f}% "
                f"{p.avg_energy:5.2f} {p.avg_hunger:5.2f} "
                f"{len(p.unique_tiles_visited):5d}  "
                f"{p.life_balance_score:5.0f}/100"
            )

    def print_life_events_report(self) -> None:
        """Report on eating, sleeping, exhaustion, starvation events."""
        print("\n--- LIFE EVENTS ---")
        header = (
            f"{'Name':12s} {'Slept':>6s} {'Ate':>5s} "
            f"{'Exhaust':>8s} {'Starving':>9s} {'Convos':>7s} {'Construct':>10s}"
        )
        print(header)
        print("-" * len(header))

        for p in sorted(self.profiles.values(), key=lambda p: p.name):
            print(
                f"{p.name:12s} "
                f"{p.times_slept:5d}x {p.times_ate:4d}x "
                f"{p.times_exhausted:7d}x {p.times_starving:8d}x "
                f"{p.conversation_samples:6d}x {p.construction_samples:9d}x"
            )

    def print_action_variety_report(self, top_n: int = 5) -> None:
        """Show the top action descriptions per NPC."""
        print("\n--- ACTION VARIETY (top descriptions per NPC) ---")
        for p in sorted(self.profiles.values(), key=lambda p: p.name):
            top = p.action_descriptions.most_common(top_n)
            variety = len(p.action_descriptions)
            descs = ", ".join(f"{k} ({v}x)" for k, v in top)
            print(f"  {p.name:12s} [{variety:2d} unique]: {descs}")

    def print_construction_report(self) -> None:
        """Per-NPC construction contribution details."""
        print("\n--- CONSTRUCTION CONTRIBUTIONS ---")
        any_construction = False
        for p in sorted(self.profiles.values(), key=lambda p: p.name):
            if p.construction_samples > 0 or p.resources_contributed:
                any_construction = True
                res_str = ", ".join(
                    f"{r}={a}" for r, a in p.resources_contributed.items()
                ) or "none"
                print(
                    f"  {p.name:12s}: {p.construction_samples:3d} ticks on-site, "
                    f"resources: {res_str}"
                )
        if not any_construction:
            print("  No construction activity observed")

    def print_sync_report(self) -> None:
        """Synchronisation analysis."""
        if not self._sync_scores:
            print("\n--- SYNC: no data ---")
            return
        avg = statistics.mean(self._sync_scores)
        late = self._sync_scores[len(self._sync_scores) // 2:]
        avg_late = statistics.mean(late) if late else 0
        print(f"\n--- SYNC ---")
        print(f"  Overall: {avg:.3f}  Late-sim: {avg_late:.3f}")

    def get_outliers(self, threshold: float = 2.0) -> list[dict[str, Any]]:
        """
        Find NPCs whose behaviour deviates significantly from the group.

        Returns list of {npc_id, name, metric, value, group_mean, deviation}
        sorted by severity.
        """
        outliers: list[dict[str, Any]] = []
        profiles = list(self.profiles.values())
        if len(profiles) < 3:
            return outliers

        # Metrics to check
        metrics = [
            ("life_balance", [p.life_balance_score for p in profiles]),
            ("avg_energy", [p.avg_energy for p in profiles]),
            ("movement_range", [p.movement_range for p in profiles]),
            ("idle_pct", [p.activity_pcts.get("idle", 0) for p in profiles]),
            ("work_pct", [p.activity_pcts.get("working", 0) for p in profiles]),
            ("unique_tiles", [float(len(p.unique_tiles_visited)) for p in profiles]),
        ]

        for metric_name, values in metrics:
            if not values or all(v == values[0] for v in values):
                continue
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            if stdev == 0:
                continue

            for i, p in enumerate(profiles):
                z_score = abs(values[i] - mean) / stdev
                if z_score >= threshold:
                    outliers.append({
                        "npc_id": p.npc_id,
                        "name": p.name,
                        "metric": metric_name,
                        "value": round(values[i], 2),
                        "group_mean": round(mean, 2),
                        "z_score": round(z_score, 2),
                        "direction": "high" if values[i] > mean else "low",
                    })

        outliers.sort(key=lambda o: o["z_score"], reverse=True)
        return outliers

    def print_outlier_report(self) -> None:
        """Flag NPCs with unusual behaviour."""
        outliers = self.get_outliers()
        print(f"\n--- OUTLIER DETECTION ({len(outliers)} flags) ---")
        if not outliers:
            print("  No significant outliers detected")
            return
        for o in outliers:
            print(
                f"  {o['name']:12s} {o['metric']:16s} = {o['value']:6.1f} "
                f"(group avg {o['group_mean']:6.1f}, z={o['z_score']:.1f} {o['direction']})"
            )

    def print_full_report(self) -> None:
        """Print the complete diagnostic report."""
        self.print_per_npc_report()
        self.print_life_events_report()
        self.print_construction_report()
        self.print_action_variety_report()
        self.print_sync_report()
        self.print_outlier_report()

    def get_summary(self) -> dict[str, Any]:
        """Machine-readable summary for multi-run analysis."""
        profiles = list(self.profiles.values())
        balance_scores = [p.life_balance_score for p in profiles]
        return {
            "samples": self._sample_count,
            "avg_sync": statistics.mean(self._sync_scores) if self._sync_scores else 0,
            "avg_balance": statistics.mean(balance_scores) if balance_scores else 0,
            "min_balance": min(balance_scores) if balance_scores else 0,
            "max_balance": max(balance_scores) if balance_scores else 0,
            "outlier_count": len(self.get_outliers()),
            "avg_energy": statistics.mean([p.avg_energy for p in profiles]),
            "avg_tiles_visited": statistics.mean(
                [len(p.unique_tiles_visited) for p in profiles]
            ),
            "npcs_never_slept": sum(
                1 for p in profiles if p.times_slept == 0
            ),
            "npcs_never_ate": sum(
                1 for p in profiles if p.times_ate == 0
            ),
        }
