"""Tests for the tire degradation model.

Validates: compound differentiation, degradation increases with age,
cliff behavior, fuel effects, and neural net training.
All offline, no network calls. Must run in <10 seconds.
"""

import pytest
from pitwall.models.tire_model import (
    TireModel,
    CompoundProfile,
    COMPOUND_PROFILES,
    profile_predict,
    fuel_delta,
    generate_training_data_from_profiles,
)


class TestCompoundProfiles:
    """Test the physics-based compound profiles."""

    def test_soft_faster_than_medium_at_age_0(self):
        soft = profile_predict("SOFT", 0, 1, 57)
        medium = profile_predict("MEDIUM", 0, 1, 57)
        assert soft < medium, f"SOFT ({soft:.3f}) should be faster than MEDIUM ({medium:.3f}) at age 0"

    def test_medium_faster_than_hard_at_age_0(self):
        medium = profile_predict("MEDIUM", 0, 1, 57)
        hard = profile_predict("HARD", 0, 1, 57)
        assert medium < hard, f"MEDIUM ({medium:.3f}) should be faster than HARD ({hard:.3f}) at age 0"

    def test_compound_differentiation_significant(self):
        """CRITICAL: SOFT vs HARD difference must be >= 0.5s at age 0."""
        soft = COMPOUND_PROFILES["SOFT"].predict_delta(0)
        hard = COMPOUND_PROFILES["HARD"].predict_delta(0)
        diff = hard - soft
        assert diff >= 0.5, f"HARD-SOFT difference ({diff:.3f}s) must be >= 0.5s"

    def test_degradation_increases_with_age(self):
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            profile = COMPOUND_PROFILES[compound]
            prev = profile.predict_delta(0)
            for age in range(1, 30):
                current = profile.predict_delta(age)
                assert current >= prev, (
                    f"{compound} degradation should increase: age {age-1}={prev:.3f} > age {age}={current:.3f}"
                )
                prev = current

    def test_cliff_behavior(self):
        """After the cliff lap, degradation should increase faster."""
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            profile = COMPOUND_PROFILES[compound]
            pre_cliff = profile.predict_delta(profile.cliff_lap) - profile.predict_delta(profile.cliff_lap - 1)
            post_cliff = profile.predict_delta(profile.cliff_lap + 2) - profile.predict_delta(profile.cliff_lap + 1)
            assert post_cliff > pre_cliff, (
                f"{compound}: post-cliff degradation ({post_cliff:.4f}) "
                f"should be greater than pre-cliff ({pre_cliff:.4f})"
            )

    def test_soft_cliff_around_18_laps(self):
        assert COMPOUND_PROFILES["SOFT"].cliff_lap == 18

    def test_medium_cliff_around_33_laps(self):
        assert COMPOUND_PROFILES["MEDIUM"].cliff_lap == 33

    def test_hard_cliff_around_38_laps(self):
        assert COMPOUND_PROFILES["HARD"].cliff_lap == 38


class TestFuelEffect:
    def test_fuel_heavier_at_start(self):
        early = fuel_delta(1, 57)
        late = fuel_delta(50, 57)
        assert early > late, "Fuel delta should be higher at race start (heavier car)"

    def test_fuel_delta_non_negative(self):
        for lap in range(1, 60):
            assert fuel_delta(lap, 57) >= 0


class TestTireModel:
    """Test the TireModel wrapper."""

    def test_fallback_to_profiles_when_untrained(self):
        model = TireModel()
        assert not model.trained
        delta = model.predict("SOFT", 0, 1, 57)
        expected = profile_predict("SOFT", 0, 1, 57)
        assert abs(delta - expected) < 0.001

    def test_predict_lap_time(self):
        model = TireModel()
        base_pace = 90.0
        lap_time = model.predict_lap_time(base_pace, "SOFT", 0, 1, 57)
        assert lap_time > base_pace  # Always slower than base due to fuel + tire

    def test_compound_ordering_maintained(self):
        model = TireModel()
        soft = model.predict("SOFT", 5, 10, 57)
        medium = model.predict("MEDIUM", 5, 10, 57)
        hard = model.predict("HARD", 5, 10, 57)
        assert soft < medium < hard


class TestTraining:
    """Test model training on synthetic data."""

    def test_train_on_synthetic_data(self):
        data = generate_training_data_from_profiles(n_samples=2000, noise_std=0.2)
        assert len(data) == 2000

        model = TireModel()
        metrics = model.train_on_data(data, epochs=10, lr=1e-3, batch_size=128)

        assert model.trained
        assert "best_val_loss" in metrics
        assert metrics["best_val_loss"] < 10.0  # reasonable for 10 epochs with random init

    def test_trained_model_differentiates_compounds(self):
        data = generate_training_data_from_profiles(n_samples=5000, noise_std=0.2)
        model = TireModel()
        metrics = model.train_on_data(data, epochs=20, lr=1e-3, batch_size=128)

        diff = metrics.get("compound_differentiation", {})
        soft_hard_diff = abs(diff.get("HARD", 0) - diff.get("SOFT", 0))
        assert soft_hard_diff >= 0.3, (
            f"Trained model HARD-SOFT diff = {soft_hard_diff:.3f}s, expected >= 0.3s"
        )

    def test_empty_training_data(self):
        model = TireModel()
        result = model.train_on_data([])
        assert "error" in result
