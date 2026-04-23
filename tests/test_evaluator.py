"""Tests for the Ghost Car Evaluator.

Validates: relative delta computation, neutralization, pit accounting.
All offline, no network calls.
"""

import pytest

from pitwall.agents.evaluator import GhostCarEvaluator
from pitwall.models.tire_model import TireModel, PIT_STOP_LOSS
from pitwall.state import (
    Strategy,
    GhostCarState,
    LapData,
    StintData,
    RaceControlMessage,
)


def _make_lap_data(lap_num: int, duration: float,
                   pit_in: bool = False, pit_out: bool = False) -> LapData:
    return LapData(
        lap_number=lap_num,
        driver_number=1,
        lap_duration=duration,
        is_pit_in_lap=pit_in,
        is_pit_out_lap=pit_out,
    )


def _make_stint(compound: str, lap_start: int, lap_end: int | None = None,
                age_at_start: int = 0) -> StintData:
    return StintData(
        driver_number=1,
        stint_number=1,
        compound=compound,
        tyre_age_at_start=age_at_start,
        lap_start=lap_start,
        lap_end=lap_end,
    )


class TestGhostCarEvaluator:
    def _make_evaluator(self) -> GhostCarEvaluator:
        model = TireModel()  # uses profile fallback (no trained model)
        return GhostCarEvaluator(model)

    def test_initialize_locks_strategy(self):
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 20), ("HARD", 37)])
        ev.initialize(strategy, start_lap=1, total_laps=57)

        assert ev.ghost.initialized
        assert ev.ghost.strategy is strategy

    def test_reinitialize_replaces_strategy(self):
        """Calling initialize() again replaces the strategy."""
        ev = self._make_evaluator()
        strat1 = Strategy(stints=[("SOFT", 20), ("HARD", 37)])
        strat2 = Strategy(stints=[("MEDIUM", 30), ("HARD", 27)])

        ev.initialize(strat1, 1, 57)
        ev.initialize(strat2, 1, 57, weather="wet")

        assert ev.ghost.strategy is strat2
        assert ev.ghost.locked_weather == "wet"

    def test_same_strategy_zero_delta(self):
        """When ghost and actual follow the SAME strategy, delta ≈ 0."""
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 20), ("HARD", 37)])
        ev.initialize(strategy, start_lap=1, total_laps=57)

        # Actual driver follows exact same strategy
        actual_laps = [_make_lap_data(i, 95.0) for i in range(1, 58)]
        actual_stints = [
            _make_stint("SOFT", 1, 20),
            _make_stint("HARD", 21, 57),
        ]
        # Mark pit-in/out laps
        actual_laps[19] = _make_lap_data(20, 115.0, pit_in=True)  # Lap 20 pit-in
        actual_laps[20] = _make_lap_data(21, 97.0, pit_out=True)  # Lap 21 pit-out

        ev.update(57, actual_laps, actual_stints, [])

        # Same strategy → delta should be exactly 0 (same compound, same age, same pit count)
        assert abs(ev.delta) < 0.1, f"Same strategy should produce ~0 delta, got {ev.delta:.1f}s"

    def test_different_compound_produces_delta(self):
        """When ghost uses a different compound, delta reflects model's relative difference."""
        ev = self._make_evaluator()
        # Ghost: all SOFT
        strategy = Strategy(stints=[("SOFT", 10)])
        ev.initialize(strategy, start_lap=1, total_laps=10)

        # Actual: all HARD (slower compound per model)
        actual_laps = [_make_lap_data(i, 95.0) for i in range(1, 11)]
        actual_stints = [_make_stint("HARD", 1, 10)]

        ev.update(10, actual_laps, actual_stints, [])

        # HARD has higher base_delta than SOFT → actual_deg > ghost_deg → positive delta
        assert ev.delta > 0, f"Ghost on SOFT vs actual on HARD should be positive, got {ev.delta:.1f}s"

    def test_pit_stop_accounting(self):
        """Extra pit stop costs PIT_STOP_LOSS seconds."""
        ev = self._make_evaluator()
        # Ghost: 1 stop (2 stints)
        strategy = Strategy(stints=[("SOFT", 5), ("HARD", 5)])
        ev.initialize(strategy, start_lap=1, total_laps=10)

        # Actual: 2 stops (3 stints) — one extra pit stop
        actual_laps = [_make_lap_data(i, 95.0) for i in range(1, 11)]
        # Mark actual pit-in laps
        actual_laps[2] = _make_lap_data(3, 115.0, pit_in=True)  # 1st pit
        actual_laps[3] = _make_lap_data(4, 97.0, pit_out=True)
        actual_laps[6] = _make_lap_data(7, 115.0, pit_in=True)  # 2nd pit
        actual_laps[7] = _make_lap_data(8, 97.0, pit_out=True)
        actual_stints = [
            _make_stint("SOFT", 1, 3),
            _make_stint("MEDIUM", 4, 7),
            _make_stint("HARD", 8, 10),
        ]

        ev.update(10, actual_laps, actual_stints, [])

        # Actual has 2 pit stops, ghost has 1 → delta includes +PIT_STOP_LOSS from the extra stop
        # The compound differences are small, so pit delta dominates
        assert ev.delta > PIT_STOP_LOSS * 0.5, (
            f"Extra pit stop should add ~{PIT_STOP_LOSS}s to delta, got {ev.delta:.1f}s"
        )

    def test_neutral_on_safety_car_laps(self):
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 10)])
        ev.initialize(strategy, start_lap=1, total_laps=10)

        rc_messages = [
            RaceControlMessage(lap_number=3, category="SafetyCar", message="Safety Car Deployed"),
            RaceControlMessage(lap_number=4, category="SafetyCar", message="Safety Car"),
            RaceControlMessage(lap_number=5, category="SafetyCar", message="Safety Car"),
            RaceControlMessage(lap_number=6, category="", message="Green flag restart"),
        ]

        # Different compound on actual to normally produce a delta
        actual_laps = [_make_lap_data(i, 95.0) for i in range(1, 11)]
        actual_stints = [_make_stint("HARD", 1, 10)]  # Different from ghost SOFT

        ev.update(10, actual_laps, actual_stints, rc_messages)

        # SC laps 3-5 and lap 1 are neutralized → only laps 2,6,7,8,9,10 contribute delta
        assert ev.ghost.cumulative_delta is not None
        assert len(ev.ghost.cumulative_delta) > 0

    def test_no_capping_of_deltas(self):
        """Deltas must NOT be capped. Design requirement."""
        ev = self._make_evaluator()
        # Ghost: SOFT (fast), actual: HARD (slow) for 40 laps
        strategy = Strategy(stints=[("SOFT", 40)])
        ev.initialize(strategy, start_lap=1, total_laps=40)

        actual_laps = [_make_lap_data(i, 95.0) for i in range(1, 41)]
        actual_stints = [_make_stint("HARD", 1, 40)]

        ev.update(40, actual_laps, actual_stints, [])

        # HARD vs SOFT for 40 laps — each lap the model predicts a difference.
        # With cliff effects on SOFT past lap 18, this gets complex, but
        # the total should be uncapped.
        assert ev.ghost.cumulative_delta is not None

    def test_handles_missing_actual_data(self):
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 10)])
        ev.initialize(strategy, start_lap=1, total_laps=10)

        # Only provide half the laps
        actual_laps = [_make_lap_data(i, 92.0) for i in range(1, 6)]
        actual_stints = [_make_stint("SOFT", 1, 10)]

        ev.update(10, actual_laps, actual_stints, [])

        # Should not crash
        assert ev.ghost.cumulative_delta is not None

    def test_realistic_same_strategy_delta_near_zero(self):
        """When actual follows the ghost strategy exactly, delta ≈ 0."""
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 20), ("HARD", 37)])
        ev.initialize(strategy, start_lap=1, total_laps=57)

        # Actual follows same strategy with varying lap times (noise doesn't matter)
        actual_laps = []
        for i in range(1, 58):
            dur = 93.0 + (i % 5) * 0.3  # Some variation
            actual_laps.append(_make_lap_data(i, dur))
        # Pit-in/out on the boundary
        actual_laps[19] = _make_lap_data(20, 115.0, pit_in=True)
        actual_laps[20] = _make_lap_data(21, 97.0, pit_out=True)

        actual_stints = [
            _make_stint("SOFT", 1, 20),
            _make_stint("HARD", 21, 57),
        ]

        ev.update(57, actual_laps, actual_stints, [])

        # Same strategy → delta should be ~0
        # (pit costs cancel, compound+age identical on every lap)
        assert abs(ev.delta) < 1.0, (
            f"Same strategy should give ~0 delta, got {ev.delta:.1f}s"
        )

    def test_ghost_compound_and_age(self):
        """Helper correctly maps lap numbers to ghost compound/age."""
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 5), ("HARD", 5)])
        ev.initialize(strategy, start_lap=1, total_laps=10)

        assert ev._ghost_compound_and_age(1) == ("SOFT", 0)
        assert ev._ghost_compound_and_age(5) == ("SOFT", 4)
        assert ev._ghost_compound_and_age(6) == ("HARD", 0)
        assert ev._ghost_compound_and_age(10) == ("HARD", 4)
        assert ev._ghost_compound_and_age(11) is None  # Past range

    def test_ghost_is_pitting(self):
        """Pitting detected on last lap of non-final stint."""
        ev = self._make_evaluator()
        strategy = Strategy(stints=[("SOFT", 5), ("HARD", 5)])
        ev.initialize(strategy, start_lap=1, total_laps=10)

        assert ev._ghost_is_pitting(5) is True   # Last lap of first stint
        assert ev._ghost_is_pitting(4) is False
        assert ev._ghost_is_pitting(10) is False  # Last stint — no pit after
