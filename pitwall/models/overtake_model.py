"""Overtake probability model for position projection.

Uses a logistic model: P(overtake) = sigmoid(pace_delta * k - drs_factor - track_difficulty)
"""

from __future__ import annotations

import math


def _sigmoid(z: float) -> float:
    """Safe sigmoid that clamps the exponent to prevent overflow."""
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# Track difficulty for overtaking (higher = harder to overtake)
# Based on real F1 overtaking stats and track characteristics.
# 0.95 = nearly impossible (Monaco), 0.30 = easy (Bahrain, China long straights)
TRACK_OVERTAKE_DIFFICULTY = {
    # Street circuits — narrow, no run-off, walls
    "monaco": 0.95,
    "singapore": 0.80,
    "lasvegas": 0.45,   # Long straights despite being a street circuit
    "jeddah": 0.50,     # Fast street circuit, DRS zones help
    "baku": 0.40,       # Very long main straight, good overtaking
    "miami": 0.55,      # Purpose-built, decent overtaking
    # Permanent circuits — hard to pass
    "hungary": 0.80,    # Narrow, twisty, very few overtaking spots
    "zandvoort": 0.80,  # Narrow, banked turns, almost no overtaking
    "imola": 0.70,      # Old-school, limited opportunities
    "melbourne": 0.65,  # Tight, but recent layout changes helped
    "spain": 0.65,      # Barcelona — hard mid-sector, one real chance at T1
    "suzuka": 0.65,     # Fast, flowing — hard to follow closely
    # Permanent circuits — moderate
    "spielberg": 0.50,  # Austria — short lap, DRS zones
    "silverstone": 0.50,# Long lap, some good braking zones
    "monza": 0.35,      # Temple of speed — long straights, easy DRS passes
    "montreal": 0.45,   # Long straight into heavy braking chicane
    "lusail": 0.55,     # Qatar — fast, medium difficulty
    "yasmarina": 0.55,  # Abu Dhabi — chicanes help, but tricky
    "yas": 0.55,        # Alias
    # Permanent circuits — easy to pass
    "bahrain": 0.35,    # Multiple heavy braking zones, good overtaking
    "sakhir": 0.35,     # Alias
    "shanghai": 0.40,   # China — back straight is very long
    "austin": 0.45,     # COTA — long straight into T1
    "mexico": 0.45,     # Long straight, but altitude affects braking
    "interlagos": 0.45, # Uphill into T1 — classic overtaking
    "saopaulo": 0.45,   # Alias
    "spa": 0.40,        # Kemmel straight, Eau Rouge — great racing
    "default": 0.50,
}


def overtake_probability(pace_delta: float, has_drs: bool = False,
                         track_name: str = "default") -> float:
    """Estimate probability of overtaking in a single lap.

    Args:
        pace_delta: Pace advantage of the overtaking car (positive = faster), in seconds.
        has_drs: Whether DRS is available.
        track_name: Circuit name for difficulty lookup.

    Returns:
        Probability [0, 1] of completing an overtake this lap.
    """
    # Base logistic model
    # pace_delta of ~1.0s gives reasonable overtake probability
    k = 2.0  # sensitivity to pace delta
    drs_bonus = 0.8 if has_drs else 0.0

    track_key = track_name.lower().replace(" ", "")
    for key, diff in TRACK_OVERTAKE_DIFFICULTY.items():
        if key in track_key:
            difficulty = diff
            break
    else:
        difficulty = TRACK_OVERTAKE_DIFFICULTY["default"]

    z = k * pace_delta + drs_bonus - difficulty * 2.0

    return _sigmoid(z)


def project_position(current_position: int, our_pace: float,
                     nearby_paces: list[tuple[int, float]],
                     laps_remaining: int, track_name: str = "default") -> int:
    """Project final position based on pace differentials.

    Args:
        current_position: Current race position (1-indexed).
        our_pace: Our predicted average lap time (seconds).
        nearby_paces: List of (position, predicted_avg_lap_time) for nearby drivers.
        laps_remaining: Number of laps remaining.
        track_name: Circuit name.

    Returns:
        Projected final position.
    """
    position = current_position

    for other_pos, other_pace in nearby_paces:
        pace_diff = other_pace - our_pace  # positive if we're faster

        if pace_diff <= 0:
            continue  # they're faster or same

        # Cumulative overtake probability over remaining laps
        single_lap_prob = overtake_probability(pace_diff, has_drs=True, track_name=track_name)
        # Probability of at least one successful overtake over N laps
        prob_no_overtake = (1 - single_lap_prob) ** min(laps_remaining, 20)
        prob_overtake = 1 - prob_no_overtake

        if prob_overtake > 0.5 and other_pos < position:
            position -= 1  # we pass them (gain a position)

    return max(1, position)
