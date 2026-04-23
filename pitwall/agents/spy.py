"""Spy agent: opponent strategy prediction.

Monitors drivers immediately ahead and behind the target driver.
Predicts when opponents will pit and what compound they'll switch to.
Uses a Bayesian hazard rate model fitted on historical pit stop data.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from pitwall.data.openf1_client import OpenF1Client
from pitwall.state import SpyIntel, StintData, PositionData

if TYPE_CHECKING:
    from pitwall.state import RaceState

logger = logging.getLogger(__name__)

# --- Pit Stop Hazard Model ---
# For each compound, models the probability of pitting as a function of tire age.
# Fitted as a logistic hazard function: P(pit at age t | survived to t) = sigmoid(a * t + b)
# Parameters derived from F1 pit stop patterns:
#   SOFT: teams typically pit around lap 15-20
#   MEDIUM: teams typically pit around lap 25-30
#   HARD: teams typically pit around lap 35-40

HAZARD_PARAMS = {
    "SOFT": {"a": 0.25, "b": -4.0, "peak_age": 18},
    "MEDIUM": {"a": 0.20, "b": -5.0, "peak_age": 28},
    "HARD": {"a": 0.15, "b": -5.5, "peak_age": 38},
    "INTERMEDIATE": {"a": 0.18, "b": -4.5, "peak_age": 30},
    "WET": {"a": 0.15, "b": -4.0, "peak_age": 25},
}


def _sigmoid(z: float) -> float:
    """Safe sigmoid with clamped exponent."""
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def pit_probability(compound: str, tire_age: int, stint_number: int = 1,
                    safety_car: bool = False, laps_remaining: int = 20) -> float:
    """Predict the probability of pitting in the next 1-3 laps.

    Args:
        compound: Current tire compound.
        tire_age: Current tire age in laps.
        stint_number: Which stint the driver is on (1-indexed).
        safety_car: Whether safety car is deployed (increases pit probability).
        laps_remaining: Laps remaining in the race.

    Returns:
        Probability [0, 1] of pitting soon.
    """
    params = HAZARD_PARAMS.get(compound, HAZARD_PARAMS["MEDIUM"])

    # Base hazard from logistic model
    z = params["a"] * tire_age + params["b"]
    base_prob = _sigmoid(z)

    # Modifiers
    # Near the end of the race, pitting is unlikely (would lose too much time)
    if laps_remaining < 8:
        base_prob *= 0.2
    elif laps_remaining < 15:
        base_prob *= 0.6

    # Safety car makes pitting very attractive (reduced time loss)
    if safety_car:
        base_prob = min(0.95, base_prob * 2.5)

    # If already on stint 3+, less likely to pit again
    if stint_number >= 3:
        base_prob *= 0.5

    return max(0.0, min(1.0, base_prob))


def predict_pit_lap(compound: str, tire_age: int, current_lap: int,
                    total_laps: int) -> int | None:
    """Predict the most likely pit lap for a driver.

    Returns the lap number where pit probability first exceeds 50%,
    or None if they're unlikely to pit.
    """
    laps_remaining = total_laps - current_lap + 1

    for future_age in range(tire_age, tire_age + 30):
        future_lap = current_lap + (future_age - tire_age)
        future_remaining = total_laps - future_lap + 1

        if future_remaining < 5:
            return None  # Too late to pit

        prob = pit_probability(compound, future_age, laps_remaining=future_remaining)
        if prob > 0.5:
            return future_lap

    return None


def predict_next_compound(current_compound: str, stint_number: int,
                          weather: str, laps_remaining: int) -> str:
    """Predict what compound a driver will switch to.

    Based on common F1 strategy patterns.
    """
    if weather == "wet":
        if current_compound == "INTERMEDIATE":
            return "WET" if laps_remaining > 20 else "INTERMEDIATE"
        return "INTERMEDIATE"

    # Dry race compound transitions
    if stint_number == 1:
        # First pit stop: usually go to a harder compound
        transitions = {
            "SOFT": "MEDIUM" if laps_remaining > 25 else "HARD",
            "MEDIUM": "HARD" if laps_remaining > 20 else "SOFT",
            "HARD": "MEDIUM",
        }
    else:
        # Later stints: go to the fastest remaining option
        transitions = {
            "SOFT": "MEDIUM",
            "MEDIUM": "SOFT" if laps_remaining < 20 else "HARD",
            "HARD": "SOFT" if laps_remaining < 20 else "MEDIUM",
        }

    return transitions.get(current_compound, "MEDIUM")


class Spy:
    """Opponent strategy prediction agent."""

    def __init__(self, client: OpenF1Client):
        self.client = client

    async def analyze_opponents(
        self,
        session_key: int,
        target_driver: int,
        target_position: int,
        all_drivers: dict[int, str],
        current_lap: int,
        total_laps: int,
        weather: str = "dry",
        safety_car: bool = False,
        window: int = 3,
    ) -> list[SpyIntel]:
        """Analyze drivers within a position window of the target.

        Args:
            session_key: OpenF1 session key.
            target_driver: Our driver's number.
            target_position: Our driver's current position.
            all_drivers: Dict of driver_number -> driver_code.
            current_lap: Current lap number.
            total_laps: Total race laps.
            weather: Weather condition.
            safety_car: Whether SC is deployed.
            window: How many positions ahead/behind to monitor.

        Returns:
            List of SpyIntel for nearby drivers.
        """
        intel: list[SpyIntel] = []

        # Get all positions to find nearby drivers
        all_positions = await self.client.get_position(session_key)

        # Build position -> driver mapping for the latest data
        latest_positions: dict[int, int] = {}  # driver_number -> position
        for p in all_positions:
            dn = p.get("driver_number")
            pos = p.get("position")
            if not dn or not pos:
                continue
            latest_positions[dn] = pos

        # Find drivers within the window
        nearby_drivers = []
        for dn, pos in latest_positions.items():
            if dn == target_driver:
                continue
            if abs(pos - target_position) <= window:
                nearby_drivers.append((dn, pos))

        # Analyze each nearby driver
        for dn, pos in sorted(nearby_drivers, key=lambda x: x[1]):
            driver_code = all_drivers.get(dn, str(dn))

            # Get their stint data
            raw_stints = await self.client.get_stints(session_key, dn)
            coalesced = OpenF1Client.coalesce_stints(raw_stints)

            if not coalesced:
                continue

            # Find current stint
            current_stint = None
            for s in reversed(coalesced):
                start = s.get("lap_start", s.get("lap_number_start", 0))
                end = s.get("lap_end", s.get("lap_number_end"))
                if start <= current_lap and (end is None or end >= current_lap):
                    current_stint = s
                    break
            if not current_stint:
                current_stint = coalesced[-1]

            compound = current_stint.get("compound", "UNKNOWN")
            tyre_age_start = current_stint.get("tyre_age_at_start", 0)
            stint_start = current_stint.get("lap_start", current_stint.get("lap_number_start", 0))
            # Full tire age (includes pre-race usage) — for display
            tire_age = tyre_age_start + (current_lap - stint_start)
            # Race stint age (laps on this set during the race only) — for predictions
            race_stint_age = current_lap - stint_start
            stint_number = len(coalesced)
            laps_remaining = total_laps - current_lap + 1

            # Predict pit stop using race stint age (not total age including practice)
            prob = pit_probability(
                compound, race_stint_age, stint_number,
                safety_car=safety_car,
                laps_remaining=laps_remaining,
            )
            predicted_lap = predict_pit_lap(compound, race_stint_age, current_lap, total_laps)
            predicted_compound = predict_next_compound(
                compound, stint_number, weather, laps_remaining
            )

            # Gap placeholder — computed by downstream logic when intervals are available.
            gap = None

            intel.append(SpyIntel(
                driver_number=dn,
                driver_code=driver_code,
                position=pos,
                current_compound=compound,
                tire_age=race_stint_age,
                predicted_pit_lap=predicted_lap,
                pit_probability=prob,
                predicted_compound=predicted_compound,
                gap_seconds=gap,
            ))

        return intel



async def run_spy(state: RaceState, client: OpenF1Client) -> dict:
    """LangGraph node: analyze opponent strategies."""
    spy = Spy(client)

    session_key = state["session_key"]
    driver_number = state["driver_number"]
    current_lap = state.get("current_lap", 1)
    total_laps = state.get("total_laps", 57)
    weather = state.get("weather", "dry")
    safety_car = state.get("safety_car", False)
    all_drivers = state.get("all_drivers", {})

    # Get current position
    positions = state.get("position_data", [])
    current_pos = 10  # default
    for p in reversed(positions):
        if p.lap_number <= current_lap:
            current_pos = p.position
            break

    intel = await spy.analyze_opponents(
        session_key=session_key,
        target_driver=driver_number,
        target_position=current_pos,
        all_drivers=all_drivers,
        current_lap=current_lap,
        total_laps=total_laps,
        weather=weather,
        safety_car=safety_car,
    )

    return {"spy_intelligence": intel}
