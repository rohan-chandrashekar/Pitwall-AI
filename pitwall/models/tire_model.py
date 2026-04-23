"""PyTorch tire degradation model + calibrated compound profiles.

The model predicts lap time delta from base pace as a function of
compound, tire age, fuel load, and track temperature.

Compound profiles provide a physics-based fallback and sanity check.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}
NUM_COMPOUNDS = len(COMPOUNDS)

# Pit stop time loss (seconds) in normal conditions
PIT_STOP_LOSS = 22.0
# Reduced pit loss under safety car (~10s saved)
PIT_STOP_LOSS_SC = 12.0
# Reduced pit loss under VSC (~5s saved)
PIT_STOP_LOSS_VSC = 17.0

# Fuel parameters
FUEL_START_KG = 110.0
FUEL_BURN_PER_LAP_KG = 1.8
FUEL_EFFECT_PER_KG = 0.03  # seconds per kg per lap



# --- Calibrated Compound Profiles (physics-based fallback) ---

@dataclass
class CompoundProfile:
    """Physics-based tire compound model."""
    name: str
    base_delta: float       # seconds slower than SOFT at age 0
    deg_rate: float         # seconds per lap degradation
    cliff_lap: int          # lap number where cliff starts
    cliff_severity: float   # additional deg per lap after cliff

    def predict_delta(self, tire_age: int) -> float:
        """Predict lap time delta (seconds) for a given tire age."""
        delta = self.base_delta + self.deg_rate * tire_age
        if tire_age > self.cliff_lap:
            cliff_laps = tire_age - self.cliff_lap
            delta += self.cliff_severity * cliff_laps
        return delta


# Calibrated from real F1 stint data (2023-2025 seasons):
#   SOFT:   typically 12-18 laps, ~0.9s off pace at pit stop, cliff ~18 laps
#   MEDIUM: typically 18-30 laps, can extend to 40+ at low-deg tracks, cliff ~33 laps
#   HARD:   typically 25-38 laps, can extend to 42+ at low-deg tracks, cliff ~38 laps
#
# Degradation rate (deg_rate) is the linear per-lap time loss.  In real F1,
# this is subtle: 0.03-0.05 s/lap.  Previous values (0.05-0.08) were 50-100%
# too high, producing 4-8s deltas at normal pit ages.
#
# Cliff severity is the ADDITIONAL per-lap loss beyond the cliff point.
# Real cliff behaviour adds ~0.04-0.10 s/lap.  MEDIUM and HARD cliffs are
# gentler than SOFT — those compounds are designed for endurance.
#
# HARD base_delta: reduced from 1.2 to 0.9 — in practice the HARD vs SOFT
# gap is ~0.8-1.0s, not 1.2s.  The old value made the GA avoid HARD entirely
# in favor of MEDIUM, but then the cliff at 25 laps pushed it back to HARD
# for long stints.  With a correct cliff_lap (33 for MEDIUM), MEDIUM now
# wins for stints up to ~38 laps, matching real team behavior.
COMPOUND_PROFILES = {
    "SOFT": CompoundProfile("SOFT", base_delta=0.0, deg_rate=0.05, cliff_lap=18, cliff_severity=0.10),
    "MEDIUM": CompoundProfile("MEDIUM", base_delta=0.6, deg_rate=0.035, cliff_lap=33, cliff_severity=0.05),
    "HARD": CompoundProfile("HARD", base_delta=0.9, deg_rate=0.025, cliff_lap=38, cliff_severity=0.04),
    "INTERMEDIATE": CompoundProfile("INTERMEDIATE", base_delta=3.0, deg_rate=0.04, cliff_lap=25, cliff_severity=0.10),
    "WET": CompoundProfile("WET", base_delta=6.0, deg_rate=0.04, cliff_lap=22, cliff_severity=0.12),
}


def fuel_delta(lap_number: int, total_laps: int) -> float:
    """Compute fuel load time delta for a given lap."""
    fuel_remaining_kg = max(0, FUEL_START_KG - FUEL_BURN_PER_LAP_KG * (lap_number - 1))
    # Heavier car = slower; delta relative to empty car
    return fuel_remaining_kg * FUEL_EFFECT_PER_KG


def profile_predict(compound: str, tire_age: int, lap_number: int, total_laps: int) -> float:
    """Predict lap time delta using compound profiles (fallback model).

    Returns the delta to add to base pace.
    """
    profile = COMPOUND_PROFILES.get(compound, COMPOUND_PROFILES["MEDIUM"])
    return profile.predict_delta(tire_age) + fuel_delta(lap_number, total_laps)


# --- PyTorch Neural Network ---

FEATURE_DIM_V1 = 8   # Original: compound(5) + age + fuel + track_temp
FEATURE_DIM_V2 = 11  # V2: + rainfall + humidity + track_wetness


class TireDegNet(nn.Module):
    """Lightweight neural network for tire degradation prediction.

    V1 features (8): compound one-hot(5) + tire_age + fuel_fraction + track_temp
    V2 features (11): V1 + rainfall(0/1) + humidity(0-1) + track_wetness(0-1)

    Output (1): predicted lap time delta from base pace (seconds)
    """

    def __init__(self, input_dim: int = FEATURE_DIM_V1):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _encode_features(compound: str, tire_age: int, fuel_fraction: float,
                     track_temp: float = 30.0, rainfall: float = 0.0,
                     humidity: float = 0.5, track_wetness: float = 0.0,
                     feature_dim: int = FEATURE_DIM_V1) -> torch.Tensor:
    """Encode input features into a tensor.

    Args:
        compound: Tire compound name.
        tire_age: Tire age in laps.
        fuel_fraction: 0.0 (empty) to 1.0 (full tank).
        track_temp: Track temperature in Celsius.
        rainfall: 0.0 (dry) to 1.0 (heavy rain). Only used in V2.
        humidity: 0.0 to 1.0. Only used in V2.
        track_wetness: 0.0 (bone dry) to 1.0 (standing water). Only used in V2.
        feature_dim: 8 for V1 model, 11 for V2 model.
    """
    one_hot = [0.0] * NUM_COMPOUNDS
    idx = COMPOUND_TO_IDX.get(compound, 1)  # default MEDIUM
    one_hot[idx] = 1.0

    features = one_hot + [
        tire_age / 50.0,          # normalize age
        fuel_fraction,             # already 0-1
        track_temp / 50.0,         # normalize temp
    ]

    if feature_dim >= FEATURE_DIM_V2:
        features.extend([
            rainfall,              # 0-1 (already normalized)
            humidity,              # 0-1 (already normalized)
            track_wetness,         # 0-1 (already normalized)
        ])

    return torch.tensor(features, dtype=torch.float32)


class TireModel:
    """High-level tire degradation model with PyTorch backend + compound profile fallback.

    Supports two model versions:
      V1 (8 features): compound + tire_age + fuel + track_temp
      V2 (11 features): V1 + rainfall + humidity + track_wetness

    Backward compatible: loads V1 models and uses them with default weather values.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM_V1):
        self.feature_dim = feature_dim
        self.net = TireDegNet(input_dim=feature_dim).to(DEVICE)
        self.trained = False
        self._model_path = MODEL_DIR / "tire_deg.pt"

    def load(self) -> bool:
        """Load a trained model from disk. Auto-detects V1 vs V2."""
        if self._model_path.exists():
            try:
                state = torch.load(self._model_path, map_location=DEVICE, weights_only=True)

                # Auto-detect feature dimension from the first linear layer's weight shape
                first_key = [k for k in state if k.endswith(".weight")][0]
                saved_dim = state[first_key].shape[1]

                if saved_dim != self.feature_dim:
                    logger.info(
                        f"Model on disk has {saved_dim} features, "
                        f"re-creating network (was {self.feature_dim})"
                    )
                    self.feature_dim = saved_dim
                    self.net = TireDegNet(input_dim=saved_dim).to(DEVICE)

                # Reject models with NaN weights — fall back to profiles
                for name, param in state.items():
                    if torch.isnan(param).any():
                        logger.error(
                            f"Model on disk has NaN in '{name}' — "
                            f"corrupt model will NOT be loaded. Falling back to profiles."
                        )
                        return False

                self.net.load_state_dict(state)
                self.net.eval()
                self.trained = True
                version = "V2 (weather-aware)" if saved_dim >= FEATURE_DIM_V2 else "V1"
                logger.info(f"Loaded tire model {version} from {self._model_path} (device={DEVICE})")
                return True
            except Exception as e:
                logger.warning(f"Failed to load tire model: {e}")
        return False

    def save(self):
        """Save the trained model to disk. Refuses to save if weights contain NaN."""
        state = self.net.state_dict()
        for name, param in state.items():
            if torch.isnan(param).any():
                logger.error(
                    f"REFUSING TO SAVE: parameter '{name}' contains NaN values. "
                    f"Training produced a corrupt model — fix training data first."
                )
                return
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, self._model_path)
        version = "V2" if self.feature_dim >= FEATURE_DIM_V2 else "V1"
        logger.info(f"Saved tire model {version} ({self.feature_dim} features) to {self._model_path}")

    def predict(self, compound: str, tire_age: int, lap_number: int,
                total_laps: int, track_temp: float = 30.0,
                rainfall: float = 0.0, humidity: float = 0.5,
                track_wetness: float = 0.0) -> float:
        """Predict lap time delta from base pace.

        Uses the neural network if trained, otherwise falls back to compound profiles.
        V2 models use weather features; V1 models ignore them.

        IMPORTANT: The NN was trained on real data where teams always pit before
        severe tire degradation. It has NEVER seen age 25+ data for any compound,
        so it extrapolates too optimistically for long stints (it thinks lighter
        fuel = faster laps, ignoring that the rubber is destroyed). We enforce
        tire cliff degradation on top of the NN's output for high tire ages.
        """
        if not self.trained:
            return profile_predict(compound, tire_age, lap_number, total_laps)

        fuel_fraction = max(0, 1.0 - (lap_number - 1) / max(total_laps, 1))
        features = _encode_features(
            compound, tire_age, fuel_fraction, track_temp,
            rainfall=rainfall, humidity=humidity,
            track_wetness=track_wetness,
            feature_dim=self.feature_dim,
        ).to(DEVICE)

        with torch.no_grad():
            delta = self.net(features.unsqueeze(0)).item()

        # Sanity check: delta should be non-negative and bounded
        delta = max(0.0, delta)

        # --- Tire cliff enforcement ---
        # The NN cannot learn cliff behavior (no training data exists for it).
        # Add cliff degradation explicitly: past the cliff point, each extra lap
        # adds severe time loss from graining, blistering, or rubber loss.
        profile = COMPOUND_PROFILES.get(compound)
        if profile and tire_age > profile.cliff_lap:
            cliff_laps = tire_age - profile.cliff_lap
            delta += profile.cliff_severity * cliff_laps

        # --- Minimum tire degradation floor ---
        # The NN conflates fuel-lightening with tire wear, sometimes predicting
        # that old tires are FASTER (because fuel is lighter). Enforce that
        # tire-only degradation (from the compound profile) is a hard floor.
        if profile:
            tire_only_floor = profile.predict_delta(tire_age)
            delta = max(delta, tire_only_floor)

        return delta

    def predict_lap_time(self, base_pace: float, compound: str, tire_age: int,
                         lap_number: int, total_laps: int,
                         track_temp: float = 30.0) -> float:
        """Predict absolute lap time."""
        return base_pace + self.predict(compound, tire_age, lap_number, total_laps, track_temp)

    def train_on_data(self, training_data: list[dict], epochs: int = 50,
                      lr: float = 1e-3, batch_size: int = 256,
                      validation_split: float = 0.1) -> dict:
        """Train the model on historical lap data.

        Each training sample is a dict with:
            compound, tire_age, fuel_fraction, track_temp, delta (target)

        Returns training metrics.
        """
        if not training_data:
            logger.error("No training data provided")
            return {"error": "no data"}

        # Detect if training data has weather features → use V2
        has_weather = any("rainfall" in s for s in training_data[:100])
        if has_weather and self.feature_dim < FEATURE_DIM_V2:
            logger.info("Weather features detected in training data — upgrading to V2 model")
            self.feature_dim = FEATURE_DIM_V2
            self.net = TireDegNet(input_dim=FEATURE_DIM_V2).to(DEVICE)

        # Encode features and targets — skip any samples with NaN
        X_list, y_list = [], []
        skipped = 0
        for sample in training_data:
            delta = sample["delta"]
            if not isinstance(delta, (int, float)) or math.isnan(delta):
                skipped += 1
                continue
            features = _encode_features(
                sample["compound"],
                sample["tire_age"],
                sample["fuel_fraction"],
                sample.get("track_temp", 30.0),
                rainfall=sample.get("rainfall", 0.0),
                humidity=sample.get("humidity", 0.5),
                track_wetness=sample.get("track_wetness", 0.0),
                feature_dim=self.feature_dim,
            )
            if features.isnan().any():
                skipped += 1
                continue
            X_list.append(features)
            y_list.append(delta)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} samples with NaN values")

        X = torch.stack(X_list).to(DEVICE)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # Split into train/val
        n = len(X)
        n_val = max(1, int(n * validation_split))
        n_train = n - n_val

        indices = torch.randperm(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Training
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # --- Physics-informed compound ordering regularization ---
        # Precompute anchor features: each compound at age 0, mid-fuel, 30°C.
        # Using mid-fuel (0.5) avoids the confounding at full fuel where
        # SOFT → heavy fuel correlation in real data confuses the network.
        # We test at multiple fuel levels to make the constraint robust.
        anchor_sets = []
        for fuel_frac in [0.3, 0.5, 0.7]:
            anchors = {
                c: _encode_features(c, 0, fuel_frac, 30.0, feature_dim=self.feature_dim)
                    .unsqueeze(0).to(DEVICE)
                for c in ["SOFT", "MEDIUM", "HARD"]
            }
            anchor_sets.append(anchors)

        ORDERING_MARGIN = 0.3   # minimum seconds separation between compounds
        ORDERING_WEIGHT = 10.0  # how strongly to enforce the physics prior

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_ord_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.net(batch_X)
                data_loss = loss_fn(pred, batch_y)

                # Compound ordering loss: enforce SOFT < MEDIUM < HARD delta
                # at identical tire age and fuel, across multiple fuel levels.
                ordering_loss = torch.tensor(0.0, device=DEVICE)
                for anchors in anchor_sets:
                    soft_d = self.net(anchors["SOFT"])
                    med_d = self.net(anchors["MEDIUM"])
                    hard_d = self.net(anchors["HARD"])
                    # Hinge loss: penalize if ordering is violated or margin too small
                    ordering_loss = ordering_loss + (
                        torch.relu(soft_d - med_d + ORDERING_MARGIN) +
                        torch.relu(med_d - hard_d + ORDERING_MARGIN)
                    ).squeeze()
                ordering_loss = ordering_loss / len(anchor_sets)

                loss = data_loss + ORDERING_WEIGHT * ordering_loss
                loss.backward()
                optimizer.step()
                epoch_loss += data_loss.item() * len(batch_X)
                epoch_ord_loss += ordering_loss.item() * len(batch_X)

            epoch_loss /= n_train
            epoch_ord_loss /= n_train

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val)
                val_loss = loss_fn(val_pred, y_val).item()
                val_mae = (val_pred - y_val).abs().mean().item()
            self.net.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: data_loss={epoch_loss:.4f}, "
                    f"ordering_loss={epoch_ord_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_mae={val_mae:.3f}s"
                )

        # Restore best model
        if best_state:
            self.net.load_state_dict(best_state)

        self.net.eval()
        self.trained = True

        # Validate compound differentiation
        diff = self._validate_compound_differentiation()

        return {
            "epochs": epochs,
            "train_samples": n_train,
            "val_samples": n_val,
            "best_val_loss": best_val_loss,
            "val_mae": val_mae,
            "compound_differentiation": diff,
        }

    def _validate_compound_differentiation(self) -> dict:
        """Check that the model differentiates between compounds.

        Tests at mid-fuel (0.5) to avoid the confounding effect where SOFT
        is correlated with heavy fuel in real data. Returns predicted deltas.
        """
        results = {}
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            features = _encode_features(
                compound, 0, 0.5, 30.0, feature_dim=self.feature_dim
            ).to(DEVICE)
            with torch.no_grad():
                delta = self.net(features.unsqueeze(0)).item()
            results[compound] = delta

        soft_hard_diff = results["HARD"] - results["SOFT"]
        ordering_ok = results["SOFT"] < results["MEDIUM"] < results["HARD"]

        if not ordering_ok or soft_hard_diff < 0.5:
            logger.warning(
                f"COMPOUND DIFFERENTIATION IS WEAK: "
                f"SOFT={results['SOFT']:.3f}s, MEDIUM={results['MEDIUM']:.3f}s, "
                f"HARD={results['HARD']:.3f}s (diff={soft_hard_diff:.3f}s, "
                f"ordering={'OK' if ordering_ok else 'WRONG'}). "
                f"Expected SOFT < MEDIUM < HARD with >= 0.5s spread."
            )
        else:
            logger.info(
                f"Compound differentiation OK: SOFT={results['SOFT']:.3f}s, "
                f"MEDIUM={results['MEDIUM']:.3f}s, HARD={results['HARD']:.3f}s, "
                f"HARD-SOFT diff={soft_hard_diff:.3f}s"
            )
        return results


def generate_training_data_from_profiles(n_samples: int = 10000,
                                         noise_std: float = 0.3) -> list[dict]:
    """Generate synthetic training data from compound profiles.

    Useful for bootstrapping the model when no real OpenF1 data is available.
    The data matches the physics-based profiles with added noise to simulate
    real-world variability.
    """
    rng = np.random.default_rng(42)
    data = []

    for _ in range(n_samples):
        compound = rng.choice(["SOFT", "MEDIUM", "HARD"])
        tire_age = int(rng.integers(0, 45))
        fuel_fraction = rng.uniform(0.0, 1.0)
        track_temp = rng.uniform(20.0, 45.0)

        # Target: profile-based delta + fuel + noise
        profile = COMPOUND_PROFILES[compound]
        tire_delta = profile.predict_delta(tire_age)
        fuel_kg = fuel_fraction * FUEL_START_KG
        fuel_delta_val = fuel_kg * FUEL_EFFECT_PER_KG
        noise = rng.normal(0, noise_std)

        delta = max(0, tire_delta + fuel_delta_val + noise)

        data.append({
            "compound": compound,
            "tire_age": tire_age,
            "fuel_fraction": fuel_fraction,
            "track_temp": track_temp,
            "delta": delta,
        })

    return data


async def prepare_training_data_from_openf1(client, seasons: list[int]) -> list[dict]:
    """Fetch real training data from OpenF1 for given seasons.

    For each race session:
    1. Get all laps and stints
    2. Compute base pace (10th percentile of clean laps)
    3. For each lap with valid timing, create a training sample
    """
    from pitwall.data.openf1_client import OpenF1Client

    all_data = []

    for year in seasons:
        logger.info(f"Fetching training data for {year} season...")
        sessions = await client.get_sessions(year=year, session_type="Race")
        if not sessions:
            logger.warning(f"No race sessions found for {year}")
            continue

        for session in sessions:
            session_key = session["session_key"]
            meeting = session.get("meeting_name") or session.get("circuit_short_name") or session.get("location") or "Unknown"
            logger.info(f"  Processing {meeting} {year} (session_key={session_key})")

            try:
                laps = await client.get_laps(session_key)
                stints_raw = await client.get_stints(session_key)
            except Exception as e:
                logger.warning(f"  Failed to fetch data for {meeting}: {e}")
                continue

            if not laps or not stints_raw:
                continue

            # Get total laps for fuel calculation
            total_laps = max(l.get("lap_number", 0) for l in laps)
            if total_laps == 0:
                continue

            # Build stint lookup: (driver_number, stint_number) -> stint_info
            # Coalesce phantom stints per driver
            from pitwall.data.openf1_client import OpenF1Client as _OC
            driver_stints: dict[int, list[dict]] = {}
            for s in stints_raw:
                dn = s.get("driver_number")
                if dn not in driver_stints:
                    driver_stints[dn] = []
                driver_stints[dn].append(s)

            coalesced_stints: dict[int, list[dict]] = {}
            for dn, ss in driver_stints.items():
                coalesced_stints[dn] = _OC.coalesce_stints(ss)

            # Build lap -> stint mapping per driver
            def find_stint(driver_num: int, lap_num: int) -> dict | None:
                for stint in coalesced_stints.get(driver_num, []):
                    # Use `or` instead of dict defaults — keys may exist with None values
                    start = stint.get("lap_start") or stint.get("lap_number_start") or 0
                    end = stint.get("lap_end") or stint.get("lap_number_end") or total_laps
                    if start <= lap_num <= end:
                        return stint
                return None

            # Compute base pace per driver (10th percentile of valid laps)
            driver_times: dict[int, list[float]] = {}
            for lap in laps:
                dur = lap.get("lap_duration")
                dn = lap.get("driver_number")
                if dur and isinstance(dur, (int, float)) and dur > 30 and dur < 200 and dn:
                    if dn not in driver_times:
                        driver_times[dn] = []
                    driver_times[dn].append(dur)

            driver_base_pace: dict[int, float] = {}
            for dn, times in driver_times.items():
                if len(times) >= 5:
                    driver_base_pace[dn] = float(np.percentile(times, 10))

            # Create training samples
            for lap in laps:
                dur = lap.get("lap_duration")
                dn = lap.get("driver_number")
                lap_num = lap.get("lap_number")

                if not dur or not dn or not lap_num:
                    continue
                if dur < 30 or dur > 200:  # filter outliers (pit laps, SC laps, etc.)
                    continue
                if dn not in driver_base_pace:
                    continue

                stint = find_stint(dn, lap_num)
                if not stint:
                    continue

                compound = stint.get("compound", "UNKNOWN")
                if compound not in COMPOUND_TO_IDX:
                    continue

                tire_age_start = stint.get("tyre_age_at_start") or 0
                stint_start = stint.get("lap_start") or stint.get("lap_number_start") or 0
                tire_age = tire_age_start + (lap_num - stint_start)

                fuel_fraction = max(0.0, 1.0 - (lap_num - 1) / total_laps)

                delta = dur - driver_base_pace[dn]
                # Skip extreme outliers (likely SC laps, red flags, etc.)
                if delta > 15.0 or delta < -2.0:
                    continue

                all_data.append({
                    "compound": compound,
                    "tire_age": max(0, tire_age),
                    "fuel_fraction": fuel_fraction,
                    "track_temp": 30.0,  # OpenF1 doesn't reliably provide track temp
                    "delta": delta,
                })

    logger.info(f"Prepared {len(all_data)} training samples from {len(seasons)} seasons")
    return all_data
