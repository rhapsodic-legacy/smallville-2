"""
Game clock and day/night cycle.

Default: 1 game day = 20 real minutes (1200 real seconds).
A game day runs from 06:00 to 06:00 (next dawn).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DayPhase(Enum):
    """Phases of the day, each with distinct lighting and NPC behaviour."""
    DAWN = "dawn"       # 05:00 – 07:00
    DAY = "day"         # 07:00 – 17:00
    DUSK = "dusk"       # 17:00 – 19:00
    NIGHT = "night"     # 19:00 – 05:00


class ScheduleSlot(Enum):
    """Coarse time slots for NPC daily routines."""
    EARLY_MORNING = "early_morning"  # 05:00 – 08:00
    MORNING = "morning"              # 08:00 – 12:00
    AFTERNOON = "afternoon"          # 12:00 – 17:00
    EVENING = "evening"              # 17:00 – 21:00
    NIGHT = "night"                  # 21:00 – 05:00


# Boundaries in game-minutes from midnight
_PHASE_BOUNDARIES: list[tuple[int, DayPhase]] = [
    (5 * 60, DayPhase.DAWN),
    (7 * 60, DayPhase.DAY),
    (17 * 60, DayPhase.DUSK),
    (19 * 60, DayPhase.NIGHT),
]

_SLOT_BOUNDARIES: list[tuple[int, ScheduleSlot]] = [
    (5 * 60, ScheduleSlot.EARLY_MORNING),
    (8 * 60, ScheduleSlot.MORNING),
    (12 * 60, ScheduleSlot.AFTERNOON),
    (17 * 60, ScheduleSlot.EVENING),
    (21 * 60, ScheduleSlot.NIGHT),
]

MINUTES_PER_DAY = 24 * 60  # 1440


def _lookup(minutes: int, boundaries: list[tuple[int, any]], default):
    """Find the last boundary that is <= minutes."""
    result = default
    for threshold, value in boundaries:
        if minutes >= threshold:
            result = value
        else:
            break
    return result


@dataclass
class GameClock:
    """
    Tracks game time and converts between real and game time.

    Attributes:
        day: current day number (1-based)
        minutes: minutes elapsed in current day (0–1439)
        speed: real seconds per game day (default 1200 = 20 min)
        paused: whether the clock is frozen
    """
    day: int = 1
    minutes: int = 360  # start at 06:00
    speed: float = 1200.0
    paused: bool = False

    @property
    def hour(self) -> int:
        return int(self.minutes) // 60

    @property
    def minute(self) -> int:
        return int(self.minutes) % 60

    @property
    def time_string(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"

    @property
    def phase(self) -> DayPhase:
        return _lookup(self.minutes, _PHASE_BOUNDARIES, DayPhase.NIGHT)

    @property
    def schedule_slot(self) -> ScheduleSlot:
        return _lookup(self.minutes, _SLOT_BOUNDARIES, ScheduleSlot.NIGHT)

    @property
    def day_progress(self) -> float:
        """0.0 at midnight, 1.0 at next midnight."""
        return self.minutes / MINUTES_PER_DAY

    @property
    def sun_angle(self) -> float:
        """Sun angle in radians: 0 at dawn (06:00), pi at dusk (18:00)."""
        import math
        # Map 06:00–18:00 to 0–pi, outside that range clamp to 0 or pi
        daylight_start = 6 * 60
        daylight_end = 18 * 60
        if self.minutes <= daylight_start:
            return 0.0
        if self.minutes >= daylight_end:
            return math.pi
        t = (self.minutes - daylight_start) / (daylight_end - daylight_start)
        return t * math.pi

    def _real_seconds_per_game_minute(self) -> float:
        return self.speed / MINUTES_PER_DAY

    def tick(self, real_delta_seconds: float) -> list[str]:
        """
        Advance the clock by real_delta_seconds.

        Returns a list of event strings for phase/slot transitions that
        occurred during this tick (e.g. ["phase:day", "slot:morning"]).
        """
        if self.paused:
            return []

        old_phase = self.phase
        old_slot = self.schedule_slot
        old_day = self.day

        game_minutes = real_delta_seconds / self._real_seconds_per_game_minute()
        self.minutes += game_minutes

        events = []

        # Handle day rollover
        while self.minutes >= MINUTES_PER_DAY:
            self.minutes -= MINUTES_PER_DAY
            self.day += 1
            events.append(f"new_day:{self.day}")

        new_phase = self.phase
        new_slot = self.schedule_slot

        if new_phase != old_phase:
            events.append(f"phase:{new_phase.value}")
        if new_slot != old_slot:
            events.append(f"slot:{new_slot.value}")

        return events

    def to_dict(self) -> dict:
        """Serialise for WebSocket transmission."""
        return {
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "time": self.time_string,
            "phase": self.phase.value,
            "slot": self.schedule_slot.value,
            "day_progress": round(self.day_progress, 4),
            "sun_angle": round(self.sun_angle, 4),
            "paused": self.paused,
        }
