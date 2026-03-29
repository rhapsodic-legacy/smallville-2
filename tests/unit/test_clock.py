"""Tests for the game clock and time system."""

from core.time_system.clock import GameClock, DayPhase, ScheduleSlot


class TestGameClock:
    def test_default_starts_at_0600(self):
        clock = GameClock()
        assert clock.hour == 6
        assert clock.minute == 0
        assert clock.time_string == "06:00"

    def test_phase_at_dawn(self):
        clock = GameClock(minutes=5 * 60 + 30)  # 05:30
        assert clock.phase == DayPhase.DAWN

    def test_phase_at_midday(self):
        clock = GameClock(minutes=12 * 60)
        assert clock.phase == DayPhase.DAY

    def test_phase_at_dusk(self):
        clock = GameClock(minutes=18 * 60)
        assert clock.phase == DayPhase.DUSK

    def test_phase_at_night(self):
        clock = GameClock(minutes=22 * 60)
        assert clock.phase == DayPhase.NIGHT

    def test_phase_at_midnight(self):
        clock = GameClock(minutes=0)
        assert clock.phase == DayPhase.NIGHT

    def test_schedule_slot_morning(self):
        clock = GameClock(minutes=9 * 60)
        assert clock.schedule_slot == ScheduleSlot.MORNING

    def test_schedule_slot_afternoon(self):
        clock = GameClock(minutes=14 * 60)
        assert clock.schedule_slot == ScheduleSlot.AFTERNOON

    def test_schedule_slot_night(self):
        clock = GameClock(minutes=23 * 60)
        assert clock.schedule_slot == ScheduleSlot.NIGHT

    def test_tick_advances_time(self):
        clock = GameClock(minutes=360, speed=1200.0)  # 06:00
        # 1200 real seconds = 1 full game day (1440 minutes)
        # So 1 real second = 1440/1200 = 1.2 game minutes
        clock.tick(1.0)
        assert clock.minutes > 360

    def test_tick_rolls_over_day(self):
        clock = GameClock(day=1, minutes=1439, speed=1200.0)
        events = clock.tick(2.0)
        assert clock.day == 2
        assert any("new_day" in e for e in events)

    def test_tick_detects_phase_change(self):
        # Start just before dawn (04:59), tick into dawn (05:00+)
        clock = GameClock(minutes=4 * 60 + 59, speed=1200.0)
        events = clock.tick(2.0)
        assert any("phase:dawn" in e for e in events)

    def test_paused_clock_does_not_advance(self):
        clock = GameClock(minutes=360, paused=True)
        clock.tick(100.0)
        assert clock.minutes == 360

    def test_day_progress(self):
        clock = GameClock(minutes=720)  # noon
        assert abs(clock.day_progress - 0.5) < 0.01

    def test_to_dict_has_required_keys(self):
        clock = GameClock()
        data = clock.to_dict()
        assert "day" in data
        assert "hour" in data
        assert "phase" in data
        assert "slot" in data
        assert "sun_angle" in data
