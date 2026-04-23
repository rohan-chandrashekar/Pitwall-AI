"""OpenF1 API client with retry logic, rate limiting, and session resolution."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"

# Map of common race names to their official meeting names
RACE_NAME_MAP = {
    "bahrain": "Bahrain",
    "saudi": "Saudi Arabia", "jeddah": "Saudi Arabia",
    "australia": "Australia", "melbourne": "Australia",
    "japan": "Japan", "suzuka": "Japan",
    "china": "China", "shanghai": "China",
    "miami": "Miami",
    "emilia": "Emilia Romagna", "imola": "Emilia Romagna",
    "monaco": "Monaco",
    "canada": "Canada", "montreal": "Canada",
    "spain": "Spain", "barcelona": "Spain",
    "austria": "Austria", "spielberg": "Austria",
    "britain": "Great Britain", "silverstone": "Great Britain",
    "hungary": "Hungary", "budapest": "Hungary",
    "belgium": "Belgium", "spa": "Belgium",
    "netherlands": "Netherlands", "zandvoort": "Netherlands",
    "italy": "Italy", "monza": "Italy",
    "azerbaijan": "Azerbaijan", "baku": "Azerbaijan",
    "singapore": "Singapore",
    "usa": "United States", "austin": "United States", "cota": "United States",
    "mexico": "Mexico",
    "brazil": "Brazil", "interlagos": "Brazil", "sao paulo": "São Paulo",
    "las vegas": "Las Vegas", "vegas": "Las Vegas",
    "qatar": "Qatar", "lusail": "Qatar",
    "abu dhabi": "Abu Dhabi", "yas marina": "Abu Dhabi",
}


class OpenF1Client:
    """Async client for the OpenF1 API with retry and rate limit handling."""

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0,
                 cache_enabled: bool = False):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client: httpx.AsyncClient | None = None
        self.cache_enabled = cache_enabled
        self._cache: dict[str, list[dict]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _cache_key(self, endpoint: str, params: dict[str, Any] | None) -> str:
        """Build a deterministic cache key from endpoint and params."""
        sorted_params = sorted((params or {}).items())
        return f"{endpoint}|{sorted_params}"

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Make a GET request with exponential backoff retry."""
        # Check cache first (replay mode — historical data never changes)
        if self.cache_enabled:
            key = self._cache_key(endpoint, params)
            if key in self._cache:
                return self._cache[key]

        client = await self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await client.get(endpoint, params=params)

                if response.status_code == 200:
                    data = response.json()
                    if self.cache_enabled:
                        self._cache[self._cache_key(endpoint, params)] = data
                    return data

                if response.status_code == 429:
                    delay = self.base_delay * (2 ** attempt)
                    logger.debug(f"Rate limited on {endpoint}, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue

                if response.status_code == 422:
                    delay = self.base_delay * (2 ** attempt)
                    logger.debug(f"422 on {endpoint}, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue

                logger.warning(f"Unexpected status {response.status_code} on {endpoint}")
                return []

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                logger.debug(f"Connection error on {endpoint}: {e}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

        logger.warning(f"All retries exhausted for {endpoint}: {last_error}")
        return []

    # --- Endpoint methods ---

    async def get_sessions(self, year: int | None = None, **kwargs) -> list[dict]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        if year:
            params["year"] = year
        return await self._request("/sessions", params)

    async def get_laps(self, session_key: int, driver_number: int | None = None,
                       lap_number: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        if lap_number is not None:
            params["lap_number"] = lap_number
        return await self._request("/laps", params)

    async def get_stints(self, session_key: int, driver_number: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._request("/stints", params)

    async def get_position(self, session_key: int, driver_number: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._request("/position", params)

    async def get_race_control(self, session_key: int) -> list[dict]:
        return await self._request("/race_control", {"session_key": session_key})

    async def get_intervals(self, session_key: int, driver_number: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._request("/intervals", params)

    async def get_drivers(self, session_key: int) -> list[dict]:
        return await self._request("/drivers", {"session_key": session_key})

    async def get_car_data(self, session_key: int, driver_number: int,
                           speed: bool = False) -> list[dict]:
        params: dict[str, Any] = {
            "session_key": session_key,
            "driver_number": driver_number,
        }
        if speed:
            params["speed>"] = 0
        return await self._request("/car_data", params)

    # --- Session resolution ---

    async def resolve_session(self, race_name: str) -> dict | None:
        """Resolve a human-readable race name like 'bahrain-2024' to a session dict.

        Accepts formats: 'bahrain-2024', 'monaco-2024', 'qatar-2025'
        Returns the Race session for that meeting, or None.
        """
        match = re.match(r"^(.+)-(\d{4})$", race_name.strip().lower())
        if not match:
            logger.error(f"Invalid race name format: {race_name}. Expected format: 'bahrain-2024'")
            return None

        location = match.group(1).replace("-", " ")
        year = int(match.group(2))

        # Look up canonical meeting name
        meeting_name = RACE_NAME_MAP.get(location)

        # Fetch sessions for the year
        sessions = await self.get_sessions(year=year, session_type="Race")
        if not sessions:
            logger.error(f"No race sessions found for {year}")
            return None

        # Filter out Sprint sessions — OpenF1 marks both Sprint and Race as
        # session_type="Race", but session_name distinguishes them. We want
        # the main Race, not the Sprint (which has ~1/3 the laps).
        main_races = [s for s in sessions if s.get("session_name") != "Sprint"]
        if not main_races:
            main_races = sessions  # fallback if all are sprints somehow

        # Try to match by meeting name, country name, location, or circuit
        for session in main_races:
            sn = (session.get("meeting_name") or "").lower()
            cn = (session.get("country_name") or "").lower()
            ln = (session.get("location") or "").lower()
            csn = (session.get("circuit_short_name") or "").lower()

            if meeting_name:
                mn = meeting_name.lower()
                if mn in sn or mn in cn or mn in csn:
                    return session
            if location in sn or location in cn or location in ln or location in csn:
                return session

        # Fuzzy fallback: check if any part matches
        for session in main_races:
            sn = (session.get("meeting_name") or "").lower()
            cn = (session.get("country_name") or "").lower()
            csn = (session.get("circuit_short_name") or "").lower()
            for word in location.split():
                if word in sn or word in cn or word in csn:
                    return session

        logger.error(f"Could not resolve race: {race_name}")
        return None

    async def resolve_driver(self, session_key: int, driver_code: str) -> int | None:
        """Resolve a driver code (e.g., 'VER') to a driver number."""
        drivers = await self.get_drivers(session_key)
        code_upper = driver_code.upper()
        for d in drivers:
            if (d.get("name_acronym") or "").upper() == code_upper:
                return d.get("driver_number")
        logger.error(f"Driver '{driver_code}' not found in session {session_key}")
        return None

    async def get_total_laps(self, session_key: int) -> int:
        """Get the total number of laps in a race session."""
        laps = await self.get_laps(session_key)
        if not laps:
            return 57  # fallback default
        return max(l.get("lap_number", 0) for l in laps)

    async def get_all_driver_codes(self, session_key: int) -> dict[int, str]:
        """Return {driver_number: driver_code} for all drivers."""
        drivers = await self.get_drivers(session_key)
        return {
            d["driver_number"]: d.get("name_acronym", str(d["driver_number"]))
            for d in drivers
            if "driver_number" in d
        }

    async def get_tire_availability(self, meeting_key: int, session_key: int,
                                     driver_number: int) -> dict[str, int]:
        """Estimate how many tire sets of each compound are still available.

        Checks stints from ALL sessions of the same meeting (practice, qualifying)
        to count how many new sets have been deployed. Subtracts from the standard
        Pirelli allocation to estimate remaining sets for the race.

        Returns: {compound: remaining_sets}, e.g. {"SOFT": 4, "MEDIUM": 2, "HARD": 1}
        """
        # Standard F1 weekend allocation (most common split)
        allocation = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}

        # Get all sessions for this meeting
        sessions = await self.get_sessions(meeting_key=meeting_key)
        if not sessions:
            logger.debug("No meeting sessions found — using full allocation")
            return allocation

        # Count new tire sets used in sessions BEFORE or INCLUDING this one
        used_sets: dict[str, int] = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}

        for sess in sessions:
            sk = sess.get("session_key", 0)
            if sk == session_key:
                continue  # Don't count the current race session stints yet

            # Only count practice and qualifying sessions (before the race)
            stype = (sess.get("session_type") or "").lower()
            if stype not in ("practice", "qualifying", "sprint", "sprint_qualifying",
                             "sprint_shootout"):
                continue

            stints = await self.get_stints(sk, driver_number)
            if not stints:
                continue

            coalesced = self.coalesce_stints(stints)
            for stint in coalesced:
                compound = stint.get("compound", "").upper()
                age_at_start = stint.get("tyre_age_at_start", 0)
                if compound in used_sets and age_at_start == 0:
                    used_sets[compound] += 1

        # Calculate remaining
        remaining = {}
        for compound, total in allocation.items():
            left = max(0, total - used_sets.get(compound, 0))
            remaining[compound] = left
            if left == 0:
                logger.info(
                    f"Driver #{driver_number} has NO {compound} sets remaining "
                    f"(used {used_sets[compound]}/{total} before the race)"
                )

        logger.info(
            f"Tire availability for #{driver_number}: "
            f"SOFT={remaining['SOFT']}, MEDIUM={remaining['MEDIUM']}, "
            f"HARD={remaining['HARD']}"
        )
        return remaining

    # --- Data coalescing helpers ---

    @staticmethod
    def coalesce_stints(raw_stints: list[dict]) -> list[dict]:
        """Filter phantom stints from OpenF1 data.

        OpenF1 sometimes splits a single real stint into multiple phantom entries
        (same compound, continuous tire age). This merges consecutive same-compound
        stints into one. True same-compound pit stops (fresh set of the same tire)
        are extremely rare in F1 and the minor data loss is acceptable.
        """
        if not raw_stints:
            return []

        # Sort by stint_number
        sorted_stints = sorted(raw_stints, key=lambda s: s.get("stint_number", 0))

        coalesced = [sorted_stints[0].copy()]
        for stint in sorted_stints[1:]:
            prev = coalesced[-1]
            # Merge any consecutive same-compound stints (phantom data splits)
            if stint.get("compound") == prev.get("compound"):
                # Extend the previous stint's end lap
                new_end = stint.get("lap_end") or stint.get("lap_number_end")
                if new_end is not None:
                    prev["lap_end"] = new_end
                continue
            coalesced.append(stint.copy())

        return coalesced


# --- Synchronous wrapper for non-async contexts ---

def resolve_session_sync(race_name: str) -> dict | None:
    """Synchronous wrapper around session resolution."""
    client = OpenF1Client()
    try:
        return asyncio.run(client.resolve_session(race_name))
    finally:
        asyncio.run(client.close())
