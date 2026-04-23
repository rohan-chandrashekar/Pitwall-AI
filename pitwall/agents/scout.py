"""Scout agent: data ingestion from OpenF1 API.

Pulls telemetry, lap data, stint data, position data, and race control messages.
Supports both replay mode (historical data) and live mode (polling).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pitwall.data.openf1_client import OpenF1Client
from pitwall.state import LapData, StintData, PositionData, RaceControlMessage

if TYPE_CHECKING:
    from pitwall.state import RaceState

logger = logging.getLogger(__name__)


class Scout:
    """Data ingestion agent that fetches race data from OpenF1."""

    def __init__(self, client: OpenF1Client):
        self.client = client
        self._lap_cache: dict[int, LapData] = {}
        self._stint_cache: list[StintData] = []
        self._position_cache: list[PositionData] = []
        self._rc_cache: list[RaceControlMessage] = []
        self._all_drivers: dict[int, str] = {}

    async def fetch_lap(self, session_key: int, driver_number: int,
                        lap_number: int) -> LapData | None:
        """Fetch data for a specific lap."""
        laps = await self.client.get_laps(session_key, driver_number, lap_number)
        if not laps:
            return None

        lap = laps[0]
        ld = LapData(
            lap_number=lap.get("lap_number", lap_number),
            driver_number=driver_number,
            lap_duration=lap.get("lap_duration"),
            sector_1=lap.get("duration_sector_1"),
            sector_2=lap.get("duration_sector_2"),
            sector_3=lap.get("duration_sector_3"),
            is_pit_out_lap=bool(lap.get("is_pit_out_lap")),
            is_pit_in_lap=lap.get("pit_duration") is not None,
        )
        self._lap_cache[lap_number] = ld
        return ld

    async def fetch_all_laps_up_to(self, session_key: int, driver_number: int,
                                   up_to_lap: int) -> list[LapData]:
        """Fetch all laps up to a given lap number (for replay mode bulk loading)."""
        raw_laps = await self.client.get_laps(session_key, driver_number)
        results = []
        for lap in raw_laps:
            ln = lap.get("lap_number", 0)
            if ln > up_to_lap:
                continue
            ld = LapData(
                lap_number=ln,
                driver_number=driver_number,
                lap_duration=lap.get("lap_duration"),
                sector_1=lap.get("duration_sector_1"),
                sector_2=lap.get("duration_sector_2"),
                sector_3=lap.get("duration_sector_3"),
                is_pit_out_lap=bool(lap.get("is_pit_out_lap")),
                is_pit_in_lap=lap.get("pit_duration") is not None,
            )
            self._lap_cache[ln] = ld
            results.append(ld)
        return results

    async def fetch_stints(self, session_key: int, driver_number: int) -> list[StintData]:
        """Fetch and coalesce stints for a driver."""
        raw = await self.client.get_stints(session_key, driver_number)
        coalesced = OpenF1Client.coalesce_stints(raw)

        stints = []
        for s in coalesced:
            stint = StintData(
                driver_number=driver_number,
                stint_number=s.get("stint_number", 0),
                compound=s.get("compound", "UNKNOWN"),
                tyre_age_at_start=s.get("tyre_age_at_start", 0),
                lap_start=s.get("lap_start", s.get("lap_number_start", 0)),
                lap_end=s.get("lap_end", s.get("lap_number_end")),
            )
            stints.append(stint)

        self._stint_cache = stints
        return stints

    async def fetch_positions(self, session_key: int, driver_number: int) -> list[PositionData]:
        """Fetch position data for a driver."""
        raw = await self.client.get_position(session_key, driver_number)
        positions = []
        for p in raw:
            positions.append(PositionData(
                driver_number=driver_number,
                position=p.get("position", 0),
                lap_number=p.get("lap_number", 0),
            ))
        self._position_cache = positions
        return positions

    async def fetch_race_control(self, session_key: int) -> list[RaceControlMessage]:
        """Fetch race control messages (safety car, flags, etc.)."""
        raw = await self.client.get_race_control(session_key)
        messages = []
        for msg in raw:
            messages.append(RaceControlMessage(
                lap_number=msg.get("lap_number", 0),
                category=msg.get("category", ""),
                message=msg.get("message", ""),
                flag=msg.get("flag"),
            ))
        self._rc_cache = messages
        return messages

    async def fetch_all_drivers(self, session_key: int) -> dict[int, str]:
        """Fetch all driver mappings."""
        self._all_drivers = await self.client.get_all_driver_codes(session_key)
        return self._all_drivers

    def detect_weather(self, race_control: list[RaceControlMessage]) -> str:
        """Detect weather condition from race control messages.

        Ignores informational rain-risk messages (e.g., 'RISK OF RAIN IS 0%').
        Only triggers wet on actual weather changes or high rain probability.
        """
        import re

        for msg in reversed(race_control):
            msg_lower = msg.message.lower()

            # Handle rain-risk assessment messages specially:
            # "RISK OF RAIN FOR F1 RACE IS 0%" contains "rain" but is NOT wet
            if "risk of rain" in msg_lower:
                match = re.search(r'(\d+)\s*%', msg_lower)
                if match and int(match.group(1)) >= 40:
                    return "wet"
                continue  # Low/zero risk — skip this message

            # Actual wet-weather indicators
            if "wet" in msg_lower or "rain" in msg_lower:
                return "wet"
            if "intermediate" in msg_lower:
                return "wet"
            if "dry" in msg_lower:
                return "dry"
        return "dry"

    def detect_flags(self, race_control: list[RaceControlMessage],
                     current_lap: int) -> tuple[bool, bool, bool]:
        """Detect safety car, VSC, and red flag status for current lap.

        Returns (safety_car, virtual_safety_car, red_flag).
        """
        import re
        # Word-boundary match — "chequered flag" contains the substring
        # "red flag" (cheque-r-e-d + ' ' + flag), so naive `in` matching
        # falsely flags the final lap as a red-flag lap on every race.
        RED_FLAG_RE = re.compile(r"\bred\s+flag\b")

        sc = False
        vsc = False
        rf = False

        for msg in race_control:
            if msg.lap_number != current_lap:
                continue
            cat = msg.category.lower() if msg.category else ""
            text = msg.message.lower()
            flag_val = (msg.flag or "").upper()

            if "safety car" in cat or "safety car" in text:
                if "virtual" in cat or "virtual" in text:
                    vsc = True
                else:
                    sc = True
            if flag_val == "RED" or RED_FLAG_RE.search(text):
                rf = True
            # Check for SC/VSC ending
            if "green" in text or "clear" in text:
                sc = False
                vsc = False

        return sc, vsc, rf

    def get_current_stint(self, stints: list[StintData],
                          current_lap: int) -> StintData | None:
        """Get the stint active at the current lap."""
        for stint in reversed(stints):
            if stint.lap_start <= current_lap:
                if stint.lap_end is None or stint.lap_end >= current_lap:
                    return stint
        return stints[-1] if stints else None

    def get_current_position(self, positions: list[PositionData],
                             current_lap: int) -> int | None:
        """Get position at or closest to current lap."""
        best = None
        best_lap = -1
        for p in positions:
            if p.lap_number <= current_lap and p.lap_number > best_lap:
                best = p.position
                best_lap = p.lap_number
        return best


async def run_scout(state: RaceState, client: OpenF1Client) -> dict:
    """LangGraph node: fetch data for the current lap."""
    scout = Scout(client)

    session_key = state["session_key"]
    driver_number = state["driver_number"]
    current_lap = state.get("current_lap", 1)

    # Fetch data in parallel-ish fashion (sequential for simplicity)
    lap_data = await scout.fetch_all_laps_up_to(session_key, driver_number, current_lap)
    stint_data = await scout.fetch_stints(session_key, driver_number)
    position_data = await scout.fetch_positions(session_key, driver_number)
    race_control = await scout.fetch_race_control(session_key)
    all_drivers = await scout.fetch_all_drivers(session_key)

    # Detect conditions from race control messages
    weather = scout.detect_weather(race_control)
    sc, vsc, rf = scout.detect_flags(race_control, current_lap)

    # Filter stints up to current lap
    active_stints = [s for s in stint_data if s.lap_start <= current_lap]

    # Override weather based on current compound: if driver is on slicks,
    # conditions are effectively dry regardless of what race control says.
    # Conversely, if on inters/wets, it's effectively wet.
    # This handles races that transition (wet start → dry finish) where
    # race control may never explicitly declare "dry track".
    if active_stints:
        current_compound = active_stints[-1].compound
        DRY_TIRES = {"SOFT", "MEDIUM", "HARD"}
        WET_TIRES = {"INTERMEDIATE", "WET"}
        if current_compound in DRY_TIRES:
            weather = "dry"
        elif current_compound in WET_TIRES and weather == "dry":
            weather = "wet"

    # Check if race is finished
    total_laps = state.get("total_laps", 57)
    race_finished = current_lap >= total_laps

    # Check for driver retirement — match specific retirement phrases only.
    # "out" alone is too broad (matches "OUT OF POSITION", "PIT OUT", etc.)
    _RETIREMENT_PHRASES = ["retired", "retirement", "out of the race", "withdrawn", "dnf"]
    for msg in race_control:
        if msg.lap_number <= current_lap:
            driver_code = state.get("driver_code", "")
            msg_lower = msg.message.lower()
            if driver_code.lower() in msg_lower \
                    and any(phrase in msg_lower for phrase in _RETIREMENT_PHRASES):
                race_finished = True
                logger.info(f"Driver {driver_code} retired at lap {msg.lap_number}")

    return {
        "lap_data": lap_data,
        "stint_data": active_stints,
        "position_data": position_data,
        "race_control": race_control,
        "all_drivers": all_drivers,
        "weather": weather,
        "safety_car": sc,
        "virtual_safety_car": vsc,
        "red_flag": rf,
        "race_finished": race_finished,
    }
