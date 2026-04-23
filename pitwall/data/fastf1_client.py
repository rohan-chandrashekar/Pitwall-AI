"""FastF1 data fetcher for weather-enriched tire model training.

FastF1 provides what OpenF1 lacks:
  - Per-lap weather: track_temp, air_temp, humidity, rainfall, wind_speed
  - Per-lap tire life (TyreLife column)
  - Historical data from 2018+

Used ONLY for offline training. Live race pipeline still uses OpenF1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# FastF1 cache directory — avoids re-downloading sessions
CACHE_DIR = Path(__file__).parent.parent.parent / ".fastf1_cache"

COMPOUND_VALID = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


def _ensure_cache():
    """Create cache directory and enable FastF1 caching."""
    import fastf1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def _nearest_weather_row(weather_df, lap_time_stamp):
    """Find the weather row nearest to a lap timestamp. Returns None on failure."""
    import pandas as pd

    if weather_df is None or weather_df.empty:
        return None
    if lap_time_stamp is None or pd.isna(lap_time_stamp):
        return None
    try:
        if hasattr(weather_df.index, 'get_indexer'):
            idx = weather_df.index.get_indexer([lap_time_stamp], method="nearest")[0]
        else:
            idx = 0
        return weather_df.iloc[idx]
    except Exception:
        return None


def _rainfall_from_weather(weather_df, lap_time_stamp) -> float:
    """Extract rainfall value (0/1) closest to a lap's timestamp."""
    row = _nearest_weather_row(weather_df, lap_time_stamp)
    if row is None:
        return 0.0
    try:
        val = row.get("Rainfall", False)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return 1.0 if val else 0.0
    except Exception:
        return 0.0


def _humidity_from_weather(weather_df, lap_time_stamp) -> float:
    """Extract humidity (0-1) closest to a lap's timestamp."""
    row = _nearest_weather_row(weather_df, lap_time_stamp)
    if row is None:
        return 0.5
    try:
        h = row.get("Humidity", 50.0)
        if h is None or (isinstance(h, float) and np.isnan(h)):
            return 0.5
        return float(h) / 100.0
    except Exception:
        return 0.5


def _track_temp_from_weather(weather_df, lap_time_stamp) -> float:
    """Extract track temperature closest to a lap's timestamp."""
    row = _nearest_weather_row(weather_df, lap_time_stamp)
    if row is None:
        return 30.0
    try:
        t = row.get("TrackTemp", 30.0)
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return 30.0
        return float(t)
    except Exception:
        return 30.0


def _track_wetness(rainfall: float, compound: str) -> float:
    """Estimate track wetness (0-1) from rainfall and compound.

    Heuristic: if it's raining, wetness is high. If drivers are on slicks
    despite recent rain (drying track), wetness is moderate.
    """
    if rainfall > 0.5:
        return 0.8
    if compound in ("WET",):
        return 0.7
    if compound in ("INTERMEDIATE",):
        return 0.4
    return 0.0  # Dry


def prepare_training_data_from_fastf1(seasons: list[int]) -> list[dict]:
    """Fetch weather-enriched training data from FastF1.

    For each race session in the given seasons:
    1. Load session with weather data
    2. Compute per-driver base pace (10th percentile of clean laps)
    3. For each valid lap, extract compound, tire_age, fuel, weather → delta

    Returns list of training sample dicts with weather features.
    """
    import fastf1

    _ensure_cache()
    all_data = []

    for year in seasons:
        logger.info(f"FastF1: fetching {year} season schedule...")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            logger.warning(f"FastF1: failed to get {year} schedule: {e}")
            continue

        # Filter to race events (exclude testing)
        race_events = schedule[schedule["EventFormat"].notna()]

        for _, event in race_events.iterrows():
            event_name = event.get("EventName", "Unknown")
            round_num = event.get("RoundNumber", 0)
            if round_num == 0:
                continue

            logger.info(f"  FastF1: processing {event_name} {year} (Round {round_num})")

            try:
                session = fastf1.get_session(year, round_num, "Race")
                session.load(
                    laps=True,
                    weather=True,
                    telemetry=False,  # Don't need telemetry — too slow
                    messages=False,
                )
            except Exception as e:
                logger.warning(f"  FastF1: failed to load {event_name}: {e}")
                continue

            laps_df = session.laps
            weather_df = session.weather_data

            if laps_df is None or laps_df.empty:
                continue

            total_laps = int(laps_df["LapNumber"].max())
            if total_laps < 10:
                continue

            # Index weather by time for nearest-lookup
            if weather_df is not None and not weather_df.empty:
                if "Time" in weather_df.columns:
                    weather_df = weather_df.set_index("Time", drop=False)

            # Compute base pace per driver (10th percentile of valid laps)
            driver_base: dict[str, float] = {}
            for driver in laps_df["Driver"].unique():
                driver_laps = laps_df[laps_df["Driver"] == driver]
                valid = driver_laps["LapTime"].dropna()
                valid_secs = valid.dt.total_seconds()
                valid_secs = valid_secs[(valid_secs > 30) & (valid_secs < 200)]
                if len(valid_secs) >= 5:
                    driver_base[driver] = float(np.percentile(valid_secs, 10))

            # Extract training samples
            import math
            import pandas as pd

            for _, lap in laps_df.iterrows():
                driver = lap.get("Driver")
                if driver not in driver_base:
                    continue

                lap_time = lap.get("LapTime")
                if lap_time is None or pd.isna(lap_time):
                    continue

                try:
                    dur = lap_time.total_seconds()
                except Exception:
                    continue
                if math.isnan(dur) or dur < 30 or dur > 200:
                    continue

                compound = str(lap.get("Compound", "")).upper()
                if compound not in COMPOUND_VALID:
                    continue

                tire_life = lap.get("TyreLife")
                if tire_life is None or pd.isna(tire_life):
                    continue
                tire_age = int(tire_life)

                lap_num = lap.get("LapNumber")
                if lap_num is None or pd.isna(lap_num):
                    continue
                lap_num = int(lap_num)
                fuel_fraction = max(0.0, 1.0 - (lap_num - 1) / total_laps)

                # Weather features from the nearest weather sample
                lap_ts = lap.get("Time") or lap.get("LapStartTime")
                track_temp = _track_temp_from_weather(weather_df, lap_ts)
                rainfall = _rainfall_from_weather(weather_df, lap_ts)
                humidity = _humidity_from_weather(weather_df, lap_ts)
                wetness = _track_wetness(rainfall, compound)

                delta = dur - driver_base[driver]

                # Guard against any NaN leaking through
                if math.isnan(delta) or math.isnan(track_temp) or math.isnan(humidity):
                    continue

                # Filter extreme outliers (SC, pit laps, red flags)
                if delta > 15.0 or delta < -2.0:
                    continue

                all_data.append({
                    "compound": compound,
                    "tire_age": max(0, tire_age),
                    "fuel_fraction": fuel_fraction,
                    "track_temp": track_temp,
                    "rainfall": rainfall,
                    "humidity": humidity,
                    "track_wetness": wetness,
                    "delta": delta,
                })

            logger.info(f"    {event_name}: {len(all_data)} samples so far")

    logger.info(f"FastF1: total training samples = {len(all_data)} from {len(seasons)} seasons")
    return all_data
