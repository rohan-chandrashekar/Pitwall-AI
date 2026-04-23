"""Tests for the Spy agent.

Validates: probability bounds, pit prediction logic, compound transitions.
All offline, no network calls.
"""

import pytest

from pitwall.agents.spy import (
    pit_probability,
    predict_pit_lap,
    predict_next_compound,
)


class TestPitProbability:
    def test_probability_between_0_and_1(self):
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            for age in range(0, 50):
                prob = pit_probability(compound, age)
                assert 0.0 <= prob <= 1.0, f"{compound} age={age}: prob={prob}"

    def test_probability_increases_with_age(self):
        """Probability should generally increase with tire age."""
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            prob_5 = pit_probability(compound, 5, laps_remaining=40)
            prob_25 = pit_probability(compound, 25, laps_remaining=40)
            assert prob_25 >= prob_5, (
                f"{compound}: prob at age 25 ({prob_25:.3f}) "
                f"should be >= prob at age 5 ({prob_5:.3f})"
            )

    def test_soft_pits_earlier_than_hard(self):
        """SOFT tires should have higher pit probability at same age."""
        age = 15
        soft_prob = pit_probability("SOFT", age, laps_remaining=30)
        hard_prob = pit_probability("HARD", age, laps_remaining=30)
        assert soft_prob > hard_prob, (
            f"SOFT ({soft_prob:.3f}) should pit sooner than HARD ({hard_prob:.3f}) at age {age}"
        )

    def test_safety_car_increases_probability(self):
        prob_normal = pit_probability("MEDIUM", 15, safety_car=False, laps_remaining=30)
        prob_sc = pit_probability("MEDIUM", 15, safety_car=True, laps_remaining=30)
        assert prob_sc >= prob_normal

    def test_low_laps_remaining_reduces_probability(self):
        prob_many = pit_probability("MEDIUM", 15, laps_remaining=30)
        prob_few = pit_probability("MEDIUM", 15, laps_remaining=5)
        assert prob_few <= prob_many

    def test_late_stint_reduces_probability(self):
        prob_stint1 = pit_probability("MEDIUM", 15, stint_number=1, laps_remaining=30)
        prob_stint3 = pit_probability("MEDIUM", 15, stint_number=3, laps_remaining=30)
        assert prob_stint3 <= prob_stint1


class TestPredictPitLap:
    def test_returns_valid_lap_or_none(self):
        result = predict_pit_lap("SOFT", 0, 1, 57)
        assert result is None or (isinstance(result, int) and result >= 1)

    def test_soft_pits_before_hard(self):
        soft_lap = predict_pit_lap("SOFT", 0, 1, 57)
        hard_lap = predict_pit_lap("HARD", 0, 1, 57)
        if soft_lap and hard_lap:
            assert soft_lap < hard_lap

    def test_no_pit_near_end(self):
        """Should not predict pit with only 3 laps remaining."""
        result = predict_pit_lap("MEDIUM", 5, 54, 57)
        assert result is None


class TestPredictNextCompound:
    def test_returns_valid_compound(self):
        for current in ["SOFT", "MEDIUM", "HARD"]:
            next_c = predict_next_compound(current, 1, "dry", 30)
            assert next_c in ["SOFT", "MEDIUM", "HARD"]

    def test_wet_returns_wet_compound(self):
        next_c = predict_next_compound("INTERMEDIATE", 1, "wet", 30)
        assert next_c in ["INTERMEDIATE", "WET"]

    def test_changes_compound(self):
        """First stop should usually change compound."""
        next_c = predict_next_compound("SOFT", 1, "dry", 30)
        assert next_c != "SOFT"
