"""Shared state types for the LangGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


# --- Domain data classes ---

@dataclass
class LapData:
    lap_number: int
    driver_number: int
    lap_duration: float | None  # seconds; None if incomplete
    sector_1: float | None = None
    sector_2: float | None = None
    sector_3: float | None = None
    is_pit_out_lap: bool = False
    is_pit_in_lap: bool = False


@dataclass
class StintData:
    driver_number: int
    stint_number: int
    compound: str  # SOFT, MEDIUM, HARD, INTERMEDIATE, WET
    tyre_age_at_start: int
    lap_start: int
    lap_end: int | None = None  # None if stint is ongoing


@dataclass
class PositionData:
    driver_number: int
    position: int
    lap_number: int


@dataclass
class RaceControlMessage:
    lap_number: int
    category: str  # SafetyCar, VirtualSafetyCar, Flag, etc.
    message: str
    flag: str | None = None


@dataclass
class SpyIntel:
    driver_number: int
    driver_code: str
    position: int
    current_compound: str
    tire_age: int
    predicted_pit_lap: int | None
    pit_probability: float  # 0-1
    predicted_compound: str | None
    gap_seconds: float | None = None  # Gap to our driver (positive = they're ahead of us)


@dataclass
class Strategy:
    """A race strategy: list of (compound, stint_length) tuples."""
    stints: list[tuple[str, int]]
    predicted_total_time: float = 0.0

    @property
    def total_laps(self) -> int:
        return sum(length for _, length in self.stints)

    @property
    def compounds_used(self) -> set[str]:
        return {c for c, _ in self.stints}

    @property
    def uses_multiple_dry_compounds(self) -> bool:
        dry = {"SOFT", "MEDIUM", "HARD"}
        used_dry = self.compounds_used & dry
        return len(used_dry) >= 2


@dataclass
class GhostCarState:
    strategy: Strategy | None = None
    lap_times: list[float] = field(default_factory=list)
    cumulative_delta: list[float] = field(default_factory=list)
    ghost_total_time: float = 0.0
    actual_total_time: float = 0.0
    initialized: bool = False
    # Persist these so the evaluator can be re-created across laps
    start_lap: int = 1
    base_pace: float = 90.0
    total_laps: int = 57
    first_stint_age: int = 0  # tire age at start of first ghost stint
    # Track weather at initialization — if conditions change (wet→dry, dry→wet),
    # the ghost car should be re-initialized with a strategy matching new conditions.
    locked_weather: str = "dry"


# --- LangGraph state ---

class RaceState(TypedDict, total=False):
    # Session metadata
    session_key: int
    meeting_key: int
    driver_number: int
    driver_code: str
    total_laps: int
    current_lap: int
    track_name: str

    # Scout outputs
    lap_data: list[LapData]
    stint_data: list[StintData]
    position_data: list[PositionData]
    race_control: list[RaceControlMessage]
    all_drivers: dict[int, str]  # driver_number -> driver_code
    weather: str  # "dry", "wet", "mixed"

    # Race flags (current lap)
    safety_car: bool
    virtual_safety_car: bool
    red_flag: bool

    # Spy outputs
    spy_intelligence: list[SpyIntel]

    # Tire availability (sets remaining per compound for race)
    available_compounds: dict[str, int]  # {"SOFT": 4, "MEDIUM": 2, "HARD": 1}

    # Strategist outputs
    strategy: Strategy | None
    strategy_changed: bool

    # Ghost car
    ghost: GhostCarState

    # Principal output
    briefing: str
    should_brief: bool  # whether to call LLM this lap
    used_llm: bool  # whether the briefing came from Groq LLM

    # Control
    race_finished: bool
    speed_multiplier: float
    replay_mode: bool
