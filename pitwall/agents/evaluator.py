"""Ghost Car Evaluator: validates whether the AI strategy is better than actual.

Design: RELATIVE DELTA comparison — immune to model prediction error.

Instead of comparing model-predicted ghost times vs actual times (which
accumulates ~1s/lap of model error over 57 laps = ~57s of fake delta),
we compute delta from STRATEGIC DIFFERENCES only:

1. Same compound on ghost & actual → delta = 0 (no model needed)
2. Different compound → delta = model's RELATIVE compound difference
   (predict both compounds, take the difference — errors cancel)
3. Pit stops → exact accounting of pit time loss
4. Neutral laps (SC/VSC/red flag) → delta = 0

This means the ghost delta reflects PURE STRATEGIC VALUE: compound choice,
stint length, and pit timing.  It does NOT accumulate model prediction noise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pitwall.models.tire_model import (
    TireModel,
    PIT_STOP_LOSS,
    PIT_STOP_LOSS_SC,
    PIT_STOP_LOSS_VSC,
)
from pitwall.state import GhostCarState, Strategy, LapData, StintData, RaceControlMessage

if TYPE_CHECKING:
    from pitwall.state import RaceState

logger = logging.getLogger(__name__)


class GhostCarEvaluator:
    """Evaluates AI strategy against actual driver performance.

    Uses RELATIVE compound deltas so model prediction error doesn't accumulate.
    The ghost delta only reflects strategic differences (compound choice + pit timing).
    """

    def __init__(self, tire_model: TireModel, ghost: GhostCarState | None = None):
        self.tire_model = tire_model
        self.ghost = ghost or GhostCarState()

    def initialize(self, strategy: Strategy, start_lap: int,
                   total_laps: int, weather: str = "dry",
                   first_stint_age: int = 0) -> None:
        """Lock in the ghost car strategy.

        Called once when we have a strategy and enough data.
        No base_pace needed — the new design doesn't use absolute predictions.
        """
        self.ghost.strategy = strategy
        self.ghost.initialized = True
        self.ghost.locked_weather = weather
        self.ghost.start_lap = start_lap
        self.ghost.total_laps = total_laps
        self.ghost.first_stint_age = first_stint_age
        self.ghost.cumulative_delta = []
        self.ghost.ghost_total_time = 0.0
        self.ghost.actual_total_time = 0.0
        self.ghost.lap_times = []  # Not used for predictions anymore, kept for state compat
        self.ghost.base_pace = 0.0  # Not used anymore

        logger.info(
            f"Ghost car initialized: strategy={strategy.stints}, "
            f"laps={start_lap}-{total_laps}, weather={weather}"
        )

    # Minimum laps for a ghost stint to survive — shorter stints are absorbed
    # into neighbors to avoid nonsensical 1-2 lap weather-change fragments.
    MIN_GHOST_STINT = 3
    # Maximum stints the ghost strategy should ever have — prevents
    # Frankenstein strategies from accumulating across multiple weather changes.
    MAX_GHOST_STINTS = 3

    def reoptimize(self, new_strategy: Strategy,
                   from_lap: int, weather: str) -> None:
        """Re-optimize remaining race after conditions change.

        Preserves ghost history before from_lap by building a combined strategy.
        With relative deltas, no need to lock past times — past deltas are
        already computed from actual data and don't change.

        Cleans up micro-stints (< MIN_GHOST_STINT laps) and caps total stints
        at MAX_GHOST_STINTS to prevent weather-change Frankenstein strategies.
        """
        # Build combined strategy: truncated old stints + new stints
        old_stints = []
        lap_cursor = self.ghost.start_lap
        for compound, length in self.ghost.strategy.stints:
            if lap_cursor >= from_lap:
                break
            used = min(length, from_lap - lap_cursor)
            if used > 0:
                old_stints.append((compound, used))
            lap_cursor += used

        raw_combined = old_stints + list(new_strategy.stints)

        # Merge consecutive same-compound stints
        combined: list[tuple[str, int]] = []
        for compound, length in raw_combined:
            if combined and combined[-1][0] == compound:
                combined[-1] = (compound, combined[-1][1] + length)
            else:
                combined.append((compound, length))

        # Absorb micro-stints: any stint < MIN_GHOST_STINT laps gets merged
        # into its longest neighbor (avoids 1-lap weather-change fragments)
        combined = self._cleanup_micro_stints(combined)

        # Cap total stints: merge the shortest inner stints until within limit
        combined = self._cap_stints(combined)

        self.ghost.strategy = Strategy(stints=combined)
        self.ghost.locked_weather = weather

        logger.info(
            f"Ghost re-optimized at lap {from_lap} ({weather}): "
            f"old={old_stints}, new={new_strategy.stints}, "
            f"combined={combined}"
        )

    def _cleanup_micro_stints(self, stints: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Absorb micro-stints into neighbors to remove weather-change fragments."""
        if len(stints) <= 2:
            return stints

        changed = True
        while changed:
            changed = False
            for i in range(1, len(stints) - 1):  # Skip first and last
                c, l = stints[i]
                if l < self.MIN_GHOST_STINT:
                    # Absorb into the longer neighbor
                    _, l_prev = stints[i - 1]
                    _, l_next = stints[i + 1]
                    if l_prev >= l_next:
                        stints[i - 1] = (stints[i - 1][0], l_prev + l)
                    else:
                        stints[i + 1] = (stints[i + 1][0], l_next + l)
                    stints.pop(i)
                    # Re-merge consecutive same-compound stints
                    stints = self._merge_consecutive(stints)
                    changed = True
                    break
        return stints

    def _cap_stints(self, stints: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Reduce stint count to MAX_GHOST_STINTS by merging shortest inner stints."""
        while len(stints) > self.MAX_GHOST_STINTS:
            if len(stints) <= 2:
                break
            inner = stints[1:-1]
            min_idx = min(range(len(inner)), key=lambda i: inner[i][1])
            merge_idx = min_idx + 1  # offset for the first stint

            c_short, l_short = stints[merge_idx]
            # Prefer merging with a same-compound neighbor
            if merge_idx > 0 and stints[merge_idx - 1][0] == c_short:
                c_prev, l_prev = stints[merge_idx - 1]
                stints[merge_idx - 1] = (c_prev, l_prev + l_short)
            elif merge_idx < len(stints) - 1 and stints[merge_idx + 1][0] == c_short:
                c_next, l_next = stints[merge_idx + 1]
                stints[merge_idx + 1] = (c_next, l_next + l_short)
            else:
                # Absorb into preceding stint
                c_prev, l_prev = stints[merge_idx - 1]
                stints[merge_idx - 1] = (c_prev, l_prev + l_short)
            stints.pop(merge_idx)
            stints = self._merge_consecutive(stints)
        return stints

    @staticmethod
    def _merge_consecutive(stints: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Merge consecutive stints with the same compound."""
        merged: list[tuple[str, int]] = []
        for compound, length in stints:
            if merged and merged[-1][0] == compound:
                merged[-1] = (compound, merged[-1][1] + length)
            else:
                merged.append((compound, length))
        return merged

    def _ghost_compound_and_age(self, lap: int) -> tuple[str, int] | None:
        """Get the ghost car's compound and tire age for a given lap.

        Returns None if the lap is outside the ghost's range.
        """
        if not self.ghost.strategy:
            return None

        cursor = self.ghost.start_lap
        for stint_idx, (compound, length) in enumerate(self.ghost.strategy.stints):
            stint_end = cursor + length - 1
            if cursor <= lap <= stint_end:
                if stint_idx == 0:
                    age = self.ghost.first_stint_age + (lap - cursor)
                else:
                    age = lap - cursor
                return compound, age
            cursor = stint_end + 1

        return None

    def _ghost_is_pitting(self, lap: int) -> bool:
        """Check if the ghost car pits at the end of this lap (last lap of a stint)."""
        if not self.ghost.strategy:
            return False

        cursor = self.ghost.start_lap
        for stint_idx, (compound, length) in enumerate(self.ghost.strategy.stints):
            stint_end = cursor + length - 1
            # Pit at end of this stint (except the last stint)
            if lap == stint_end and stint_idx < len(self.ghost.strategy.stints) - 1:
                return True
            cursor = stint_end + 1

        return False

    def update(self, current_lap: int, actual_lap_data: list[LapData],
               actual_stints: list[StintData],
               race_control: list[RaceControlMessage]) -> None:
        """Update ghost car evaluation using RELATIVE compound deltas.

        Delta only accumulates when strategies DIVERGE:
        - Same compound + same age → delta = 0 (no model error)
        - Different compound or age → delta = model's relative difference
          (errors cancel because both sides use the same model)
        - Pit stops → cost depends on conditions (SC/VSC/green)
        - Full SC / red flag → delta resets to 0 (gaps erased)
        - VSC → delta freezes (cars slow but maintain gaps)
        """
        if not self.ghost.initialized or not self.ghost.strategy:
            return

        sc_laps, vsc_laps, rf_laps, other_neutral = self._classify_neutral_laps(race_control)
        other_neutral |= self.detect_anomalous_laps(actual_lap_data)

        # SC and red flag both erase gaps — combine them for reset logic
        gap_reset_laps = sc_laps | rf_laps

        # Find the start of each gap-reset period (SC or red flag)
        reset_start_laps: set[int] = set()
        for lap in sorted(gap_reset_laps):
            if (lap - 1) not in gap_reset_laps:
                reset_start_laps.add(lap)

        # Build actual stint lookup: lap -> (compound, tire_age)
        actual_compound_at: dict[int, tuple[str, int]] = {}
        for stint in actual_stints:
            s_start = stint.lap_start
            s_end = stint.lap_end if stint.lap_end is not None else current_lap
            for lap_n in range(s_start, s_end + 1):
                age = stint.tyre_age_at_start + (lap_n - s_start)
                actual_compound_at[lap_n] = (stint.compound, age)

        # Build actual lap time lookup
        actual_times: dict[int, float] = {}
        for ld in actual_lap_data:
            if ld.lap_duration and ld.lap_duration > 30:
                actual_times[ld.lap_number] = ld.lap_duration

        # Detect actual pit laps from STINT BOUNDARIES (not LapData flags,
        # which rely on OpenF1's pit_duration field that isn't always present).
        # Pit-in = last lap of each non-final stint.
        # Pit-out = first lap of each non-first stint.
        pit_in_laps: set[int] = set()
        pit_out_laps: set[int] = set()
        for i, stint in enumerate(actual_stints):
            if i < len(actual_stints) - 1 and stint.lap_end is not None:
                pit_in_laps.add(stint.lap_end)
            if i > 0:
                pit_out_laps.add(stint.lap_start)

        # Per-lap delta accumulation
        cumulative = 0.0
        self.ghost.cumulative_delta = []
        actual_total = 0.0

        start = self.ghost.start_lap
        for lap in range(start, current_lap + 1):
            actual_time = actual_times.get(lap)
            ghost_info = self._ghost_compound_and_age(lap)

            if ghost_info is None:
                break  # Past ghost strategy range

            if actual_time is None:
                self.ghost.cumulative_delta.append(cumulative)
                continue

            actual_total += actual_time

            # --- SC / RED FLAG: reset delta to 0 ---
            # Full SC: field bunches up behind the safety car, gaps erased.
            # Red flag: race stopped, cars line up on grid, gaps erased.
            # Any strategic advantage from before is wiped out.
            # Reset at the START of each period, then freeze during it.
            if lap in reset_start_laps:
                if cumulative != 0.0:
                    logger.debug(
                        f"Gap reset at lap {lap}: delta "
                        f"{cumulative:+.1f}s → 0.0s"
                    )
                cumulative = 0.0

            # --- Pit stop cost: depends on track conditions ---
            # SC pitting is cheap (~12s), VSC is mid (~17s), green is full (22s)
            if lap in sc_laps:
                pit_cost = PIT_STOP_LOSS_SC
            elif lap in vsc_laps:
                pit_cost = PIT_STOP_LOSS_VSC
            else:
                pit_cost = PIT_STOP_LOSS

            if lap in pit_in_laps:
                cumulative += pit_cost  # Ghost saves this time
            if self._ghost_is_pitting(lap):
                cumulative -= pit_cost  # Ghost loses this time

            # Skip compound comparison on neutralized/distorted laps
            if (lap in gap_reset_laps or lap in vsc_laps or lap in other_neutral
                    or lap in pit_in_laps or lap in pit_out_laps):
                self.ghost.cumulative_delta.append(cumulative)
                continue

            actual_info = actual_compound_at.get(lap)
            if actual_info is None:
                self.ghost.cumulative_delta.append(cumulative)
                continue

            ghost_compound, ghost_age = ghost_info
            actual_compound, actual_age = actual_info

            # --- Core: delta only when strategies diverge ---
            if ghost_compound == actual_compound and ghost_age == actual_age:
                # Identical state → zero delta, zero model error
                pass
            else:
                # Strategies diverge: use model's RELATIVE prediction.
                # Both deltas share the same model bias, so the difference
                # reflects only the compound/age performance gap.
                ghost_deg = self.tire_model.predict(
                    ghost_compound, ghost_age, lap, self.ghost.total_laps
                )
                actual_deg = self.tire_model.predict(
                    actual_compound, actual_age, lap, self.ghost.total_laps
                )
                # positive = actual has more degradation = ghost is faster
                cumulative += (actual_deg - ghost_deg)

            self.ghost.cumulative_delta.append(cumulative)

        # Derive ghost_total from actual_total and cumulative delta.
        # This keeps both numbers grounded in real data.
        self.ghost.actual_total_time = actual_total
        self.ghost.ghost_total_time = actual_total - cumulative

    def _classify_neutral_laps(
        self, race_control: list[RaceControlMessage]
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        """Classify neutral laps by type: full SC, VSC, red flag, and other.

        Full SC:   field bunches up → delta resets to 0.
        Red flag:  race stopped, cars line up → delta resets to 0.
        VSC:       cars slow, gaps maintained → delta freezes.
        Other:     lap 1, etc. → delta freezes.

        Returns (sc_laps, vsc_laps, rf_laps, other_neutral).
        """
        sc_laps: set[int] = set()
        vsc_laps: set[int] = set()
        rf_laps: set[int] = set()
        other_neutral: set[int] = {1}  # Always neutralize lap 1 (standing start)

        sc_active = False
        vsc_active = False
        rf_active = False

        sorted_msgs = sorted(race_control, key=lambda m: m.lap_number)

        import re
        # Word-boundary match to avoid "chequered flag" matching "red flag"
        # (the substring "red flag" appears inside "chequered flag"!).
        RED_FLAG_RE = re.compile(r"\bred\s+flag\b")

        for msg in sorted_msgs:
            text = msg.message.lower()
            cat = (msg.category or "").lower()
            flag_val = (msg.flag or "").upper()

            if "safety car" in cat or "safety car" in text:
                if "virtual" in cat or "virtual" in text:
                    vsc_active = True
                else:
                    sc_active = True
            # Red flag: prefer the structured `flag` field ("RED"),
            # fall back to word-boundary text match.
            if flag_val == "RED" or RED_FLAG_RE.search(text):
                rf_active = True
                sc_active = False  # Red flag supersedes SC
            if "green" in text or "clear" in text or "restart" in text:
                sc_active = False
                vsc_active = False
                rf_active = False

            if rf_active:
                rf_laps.add(msg.lap_number)
            elif sc_active:
                sc_laps.add(msg.lap_number)
            elif vsc_active:
                vsc_laps.add(msg.lap_number)

        return sc_laps, vsc_laps, rf_laps, other_neutral

    @staticmethod
    def detect_anomalous_laps(lap_data: list[LapData]) -> set[int]:
        """Detect non-pit laps with anomalously slow times.

        Safety net for SC/VSC/red flag laps that race_control messages miss.
        A lap >30% slower than the median clean pace is clearly not racing.
        """
        import numpy as np

        clean_times = []
        for ld in lap_data:
            if (ld.lap_duration
                    and ld.lap_duration > 30
                    and ld.lap_number > 1
                    and not ld.is_pit_in_lap
                    and not ld.is_pit_out_lap):
                clean_times.append(ld.lap_duration)

        if len(clean_times) < 5:
            return set()

        median_pace = float(np.median(clean_times))
        threshold = median_pace * 1.3

        anomalous = set()
        for ld in lap_data:
            if (ld.lap_duration
                    and ld.lap_duration > threshold
                    and not ld.is_pit_in_lap
                    and not ld.is_pit_out_lap):
                anomalous.add(ld.lap_number)

        if anomalous:
            logger.debug(
                f"Pace-based anomaly detection: neutralizing laps {sorted(anomalous)} "
                f"(threshold={threshold:.1f}s, median={median_pace:.1f}s)"
            )

        return anomalous

    @property
    def delta(self) -> float:
        """Total time delta: positive = AI strategy is faster."""
        return self.ghost.actual_total_time - self.ghost.ghost_total_time

    @property
    def current_cumulative_delta(self) -> float:
        """Most recent cumulative delta value."""
        if self.ghost.cumulative_delta:
            return self.ghost.cumulative_delta[-1]
        return 0.0


MIN_LAPS_FOR_GHOST = 1


def run_evaluator(state: RaceState, tire_model: TireModel | None = None) -> dict:
    """LangGraph node: update ghost car evaluation.

    Simplified: no base_pace estimation or recalibration needed.
    The relative-delta design compares compounds directly, so model
    prediction error cancels out instead of accumulating.
    """
    if tire_model is None:
        tire_model = TireModel()
        tire_model.load()

    ghost = state.get("ghost", GhostCarState())
    strategy = state.get("strategy")
    current_lap = state.get("current_lap", 1)
    total_laps = state.get("total_laps", 57)
    lap_data = state.get("lap_data", [])
    stint_data = state.get("stint_data", [])
    race_control = state.get("race_control", [])
    weather = state.get("weather", "dry")

    evaluator = GhostCarEvaluator(tire_model, ghost)

    # --- Weather change detection ---
    if ghost.initialized and strategy and ghost.locked_weather != weather:
        logger.info(
            f"Weather changed ({ghost.locked_weather} → {weather}) at lap "
            f"{current_lap}. Re-optimizing ghost car for remaining laps."
        )
        evaluator.reoptimize(strategy, current_lap, weather)

    # Initialize on first strategy
    if not ghost.initialized and strategy:
        # Need at least MIN_LAPS_FOR_GHOST clean laps
        clean_count = sum(
            1 for ld in lap_data
            if ld.lap_duration and ld.lap_duration > 30
            and ld.lap_number > 1
            and not ld.is_pit_in_lap and not ld.is_pit_out_lap
        )
        if clean_count >= MIN_LAPS_FOR_GHOST:
            first_stint_age = 0
            if stint_data:
                s = stint_data[-1]
                first_stint_age = s.tyre_age_at_start + (current_lap - s.lap_start)
            evaluator.initialize(strategy, current_lap, total_laps,
                                 weather, first_stint_age)
        else:
            logger.debug(
                f"Lap {current_lap}: waiting for {MIN_LAPS_FOR_GHOST} clean laps "
                f"before initializing ghost car"
            )

    # Update with current lap data
    if ghost.initialized:
        evaluator.update(current_lap, lap_data, stint_data, race_control)

    return {"ghost": evaluator.ghost}
