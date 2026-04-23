"""InfluxDB writer for dashboard metrics.

Writes at DEBUG level — if InfluxDB is not running, logs are suppressed.

Uses wall-clock timestamps so Grafana's `now` window always picks the data up.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pitwall.state import RaceState, GhostCarState, SpyIntel, Strategy

logger = logging.getLogger(__name__)

# Suppress noisy InfluxDB/urllib3 warnings when InfluxDB is not running
logging.getLogger("influxdb_client").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class InfluxWriter:
    """Writes race metrics to InfluxDB."""

    def __init__(self, url: str | None = None, token: str | None = None,
                 org: str = "pitwall", bucket: str = "f1"):
        self._enabled = False
        self._client = None
        self._write_api = None
        self._url = url or os.environ.get("INFLUXDB_URL", "http://localhost:8086")
        self._token = token or os.environ.get("INFLUXDB_TOKEN", "pitwall-token")
        self._org = org
        self._bucket = bucket
        self._init_client()

    def _init_client(self):
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._client = InfluxDBClient(
                url=self._url, token=self._token, org=self._org
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._enabled = True
            logger.debug("InfluxDB writer initialized")
        except ImportError:
            logger.debug("influxdb-client not installed — metrics disabled")
        except Exception as e:
            logger.debug(f"InfluxDB init failed: {e}")

    def _lap_ts(self, lap: int) -> datetime:
        """Timestamp for a given lap.

        Uses wall-clock time so the Grafana `now` window always includes it.
        The `lap` argument is kept for API compatibility; the actual timestamp
        is just "right now" — data lands on the dashboard in real time.
        """
        return datetime.now(timezone.utc)

    def _write(self, point) -> None:
        if not self._enabled or not self._write_api:
            return
        try:
            self._write_api.write(bucket=self._bucket, record=point)
        except Exception as e:
            logger.debug(f"InfluxDB write failed: {e}")

    # ------------------------------------------------------------------
    # Session-level writes (called once per race)
    # ------------------------------------------------------------------

    def start_session(
        self,
        state: RaceState,
        session_date: str = "",
        meeting_name: str = "",
    ) -> None:
        """Write the session_info + tire_allocation metadata.

        Called once at the start of a race.
        """
        if not self._enabled:
            return

        try:
            from influxdb_client import Point

            driver_code = state.get("driver_code", "UNK")
            track = state.get("track_name", "unknown")
            total_laps = state.get("total_laps", 0)

            # session_info: one row per race, overwritten by tag uniqueness.
            p = (Point("session_info")
                 .tag("driver", driver_code)
                 .tag("track", track)
                 .tag("meeting", meeting_name or track)
                 .field("total_laps", int(total_laps))
                 .field("session_date", session_date or datetime.now().strftime("%Y-%m-%d"))
                 .field("driver_code", driver_code)
                 .time(self._lap_ts(0)))
            self._write(p)

            # tire_allocation: pre-race compounds available.
            avail = state.get("available_compounds", {}) or {}
            for compound in ("SOFT", "MEDIUM", "HARD"):
                sets = int(avail.get(compound, 0))
                p = (Point("tire_allocation")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .tag("compound", compound)
                     .field("sets", sets)
                     .time(self._lap_ts(0)))
                self._write(p)

            logger.debug(
                f"InfluxDB session_start: driver={driver_code} track={track} "
                f"total_laps={total_laps}"
            )
        except Exception as e:
            logger.debug(f"InfluxDB session_start failed: {e}")

    # ------------------------------------------------------------------
    # Per-lap writes
    # ------------------------------------------------------------------

    def write_lap(self, state: RaceState) -> None:
        """Write per-lap metrics to InfluxDB."""
        if not self._enabled:
            return

        try:
            from influxdb_client import Point

            driver_code = state.get("driver_code", "UNK")
            current_lap = state.get("current_lap", 0)
            track = state.get("track_name", "unknown")
            ts = self._lap_ts(current_lap)

            ghost: GhostCarState = state.get("ghost", GhostCarState())
            cum_delta: float | None = None
            if ghost.cumulative_delta:
                cum_delta = float(ghost.cumulative_delta[-1])
                p = (Point("ghost_delta")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .field("cumulative_delta", cum_delta)
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

            # Actual lap time for the driver
            actual_lap_time: float | None = None
            for ld in state.get("lap_data", []):
                if ld.lap_number == current_lap and ld.lap_duration and ld.lap_duration > 30:
                    actual_lap_time = float(ld.lap_duration)
                    p = (Point("lap_time")
                         .tag("driver", driver_code)
                         .tag("track", track)
                         .tag("source", "actual")
                         .field("duration", actual_lap_time)
                         .field("lap", current_lap)
                         .time(ts))
                    self._write(p)
                    break

            # AI ghost lap time (derived from per-lap delta so it stays grounded)
            if actual_lap_time is not None and ghost.initialized and len(ghost.cumulative_delta) >= 1:
                if len(ghost.cumulative_delta) >= 2:
                    prev_cum = ghost.cumulative_delta[-2]
                else:
                    prev_cum = 0.0
                per_lap_delta = (ghost.cumulative_delta[-1] - prev_cum)
                ai_lap_time = max(30.0, actual_lap_time - per_lap_delta)
                p = (Point("lap_time")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .tag("source", "ai")
                     .field("duration", float(ai_lap_time))
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

            # Current compound + tire age for the driver.
            # Age convention: 1-indexed count of laps completed on this tire,
            # matching how teams/broadcasts describe tire age ("15 laps on
            # these mediums") — the current lap is included in the count.
            stint_data = state.get("stint_data", [])
            if stint_data:
                s = stint_data[-1]
                tire_age = s.tyre_age_at_start + (current_lap - s.lap_start) + 1
                p = (Point("driver_state")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .field("compound", s.compound)
                     .field("tire_age", int(tire_age))
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

                # AI ghost compound for the same lap (same 1-indexed convention)
                if ghost.initialized and ghost.strategy:
                    cursor = ghost.start_lap
                    ghost_compound = None
                    ghost_age = None
                    for i, (compound, length) in enumerate(ghost.strategy.stints):
                        if cursor <= current_lap <= cursor + length - 1:
                            ghost_compound = compound
                            if i == 0:
                                ghost_age = ghost.first_stint_age + (current_lap - cursor) + 1
                            else:
                                ghost_age = (current_lap - cursor) + 1
                            break
                        cursor += length
                    if ghost_compound:
                        p = (Point("ghost_state")
                             .tag("driver", driver_code)
                             .tag("track", track)
                             .field("compound", ghost_compound)
                             .field("tire_age", int(ghost_age or 0))
                             .field("lap", current_lap)
                             .time(ts))
                        self._write(p)

            # Race flags
            sc = state.get("safety_car", False)
            vsc = state.get("virtual_safety_car", False)
            rf = state.get("red_flag", False)
            flag = "RED" if rf else "SC" if sc else "VSC" if vsc else "GREEN"
            p = (Point("race_flag")
                 .tag("driver", driver_code)
                 .tag("track", track)
                 .field("flag", flag)
                 .field("lap", current_lap)
                 .time(ts))
            self._write(p)

            # Strategy (AI plan) — keep latest snapshot
            strategy: Strategy | None = state.get("strategy")
            if strategy:
                for i, (compound, length) in enumerate(strategy.stints):
                    p = (Point("strategy")
                         .tag("driver", driver_code)
                         .tag("track", track)
                         .tag("stint_index", str(i))
                         .field("compound", compound)
                         .field("stint_length", length)
                         .field("lap", current_lap)
                         .time(ts))
                    self._write(p)

            # Driver position — find the latest known position at or before
            # current_lap.  position_data is sorted ascending by lap_number,
            # so we simply keep the last record whose lap_number <= current_lap.
            positions = state.get("position_data", [])
            latest_pos = None
            for pos in positions:
                if pos.lap_number <= current_lap:
                    latest_pos = pos.position
                else:
                    break
            if latest_pos is not None:
                p = (Point("position")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .field("position", int(latest_pos))
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

            # Rival positions & spy intel (display age as 1-indexed laps)
            spy_intel: list[SpyIntel] = state.get("spy_intelligence", [])
            for intel in spy_intel:
                display_age = int(intel.tire_age) + 1
                p = (Point("spy_intel")
                     .tag("driver", driver_code)
                     .tag("opponent", intel.driver_code)
                     .tag("track", track)
                     .field("position", int(intel.position))
                     .field("compound", intel.current_compound)
                     .field("tire_age", display_age)
                     .field("pit_probability", float(intel.pit_probability))
                     .field("predicted_pit_lap", int(intel.predicted_pit_lap or 0))
                     .field("gap_seconds", float(intel.gap_seconds or 0.0))
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

                # Position time-series for rivals (for the position chart)
                p = (Point("rival_position")
                     .tag("driver", driver_code)
                     .tag("opponent", intel.driver_code)
                     .tag("track", track)
                     .field("position", int(intel.position))
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

            # Briefing (only when a new one is issued)
            briefing = state.get("briefing", "")
            if briefing and state.get("should_brief", False):
                p = (Point("briefing")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .field("text", briefing)
                     .field("lap", current_lap)
                     .time(ts))
                self._write(p)

        except Exception as e:
            logger.debug(f"InfluxDB write_lap failed: {e}")

    def write_stint_timeline(self, state: RaceState) -> None:
        """Write stint timeline data for the strategy timeline panel."""
        if not self._enabled:
            return

        try:
            from influxdb_client import Point

            driver_code = state.get("driver_code", "UNK")
            track = state.get("track_name", "unknown")
            current_lap = state.get("current_lap", 0)
            ts = self._lap_ts(current_lap)

            # Actual stints
            for stint in state.get("stint_data", []):
                p = (Point("stint_timeline")
                     .tag("driver", driver_code)
                     .tag("track", track)
                     .tag("source", "actual")
                     .tag("stint_index", str(stint.stint_number))
                     .field("compound", stint.compound)
                     .field("lap_start", int(stint.lap_start))
                     .field("lap_end", int(stint.lap_end or current_lap))
                     .time(ts))
                self._write(p)

            # Ghost car stints
            ghost: GhostCarState = state.get("ghost", GhostCarState())
            if ghost.strategy:
                lap = ghost.start_lap
                for i, (compound, length) in enumerate(ghost.strategy.stints):
                    p = (Point("stint_timeline")
                         .tag("driver", driver_code)
                         .tag("track", track)
                         .tag("source", "ai")
                         .tag("stint_index", str(i + 1))
                         .field("compound", compound)
                         .field("lap_start", int(lap))
                         .field("lap_end", int(lap + length - 1))
                         .time(ts))
                    self._write(p)
                    lap += length

        except Exception as e:
            logger.debug(f"InfluxDB stint timeline write failed: {e}")

    # ------------------------------------------------------------------
    # End-of-race writes
    # ------------------------------------------------------------------

    def write_race_result(
        self,
        state: RaceState,
        finish_position: int | None,
        start_position: int | None,
        positions_gained: int,
        ai_positions_vs_actual: int,
        final_delta: float,
        pit_diff: int,
        estimated_ai_finish: int | None,
        verdict: str,
    ) -> None:
        """Write the final race result summary.

        `positions_gained` is the race positions gained by the driver
        (start_position - finish_position, standard F1 convention).
        `ai_positions_vs_actual` is the AI-vs-actual comparison count.
        """
        if not self._enabled:
            return

        try:
            from influxdb_client import Point

            driver_code = state.get("driver_code", "UNK")
            track = state.get("track_name", "unknown")
            total_laps = state.get("total_laps", 0)
            ts = self._lap_ts(total_laps + 1)

            p = (Point("race_result")
                 .tag("driver", driver_code)
                 .tag("track", track)
                 .field("finish_position", int(finish_position or 0))
                 .field("start_position", int(start_position or finish_position or 0))
                 .field("estimated_ai_finish", int(estimated_ai_finish or finish_position or 0))
                 .field("positions_gained", int(positions_gained))
                 .field("ai_positions_vs_actual", int(ai_positions_vs_actual))
                 .field("final_delta_seconds", float(final_delta))
                 .field("pit_diff", int(pit_diff))
                 .field("verdict", verdict)
                 .time(ts))
            self._write(p)
        except Exception as e:
            logger.debug(f"InfluxDB write_race_result failed: {e}")

    def write_debrief(self, state: RaceState, debrief_text: str) -> None:
        """Write the post-race debrief text (displayed at the bottom of the dashboard)."""
        if not self._enabled or not debrief_text:
            return

        try:
            from influxdb_client import Point

            driver_code = state.get("driver_code", "UNK")
            track = state.get("track_name", "unknown")
            total_laps = state.get("total_laps", 0)
            ts = self._lap_ts(total_laps + 2)

            p = (Point("debrief")
                 .tag("driver", driver_code)
                 .tag("track", track)
                 .field("text", debrief_text)
                 .time(ts))
            self._write(p)
        except Exception as e:
            logger.debug(f"InfluxDB write_debrief failed: {e}")

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
