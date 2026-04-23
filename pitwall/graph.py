"""LangGraph pipeline orchestration.

Sequential per-lap pipeline: Scout → Spy → Strategist → Principal
Ghost Car Evaluator runs after Strategist.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END

from pitwall.agents.evaluator import run_evaluator
from pitwall.agents.principal import run_principal, Principal
from pitwall.agents.scout import run_scout
from pitwall.agents.spy import run_spy
from pitwall.agents.strategist import run_strategist
from pitwall.data.openf1_client import OpenF1Client
from pitwall.influx import InfluxWriter
from pitwall.models.tire_model import TireModel
from pitwall.state import RaceState, GhostCarState

logger = logging.getLogger(__name__)


def build_graph(client: OpenF1Client, tire_model: TireModel) -> StateGraph:
    """Build the LangGraph state graph for the race pipeline."""
    graph = StateGraph(RaceState)

    # Node wrappers that capture client/model dependencies
    async def scout_node(state: RaceState) -> dict:
        return await run_scout(state, client)

    async def spy_node(state: RaceState) -> dict:
        return await run_spy(state, client)

    def strategist_node(state: RaceState) -> dict:
        return run_strategist(state)

    def evaluator_node(state: RaceState) -> dict:
        return run_evaluator(state, tire_model)

    def principal_node(state: RaceState) -> dict:
        return run_principal(state)

    # Add nodes
    graph.add_node("scout", scout_node)
    graph.add_node("spy", spy_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("principal", principal_node)

    # Define edges: Scout → Spy → Strategist → Evaluator → Principal → END
    graph.set_entry_point("scout")
    graph.add_edge("scout", "spy")
    graph.add_edge("spy", "strategist")
    graph.add_edge("strategist", "evaluator")
    graph.add_edge("evaluator", "principal")
    graph.add_edge("principal", END)

    return graph


class RaceRunner:
    """Runs the pipeline lap-by-lap for a complete race."""

    def __init__(self, race_name: str, driver_code: str,
                 speed_multiplier: float = 1000.0, live: bool = False):
        self.race_name = race_name
        self.driver_code = driver_code
        self.speed_multiplier = speed_multiplier
        self.live = live
        self.client = OpenF1Client(cache_enabled=not live)
        self.tire_model = TireModel()
        self.influx = InfluxWriter()

    async def run(self) -> None:
        """Execute the full race pipeline."""
        logger.info(f"Starting PitWall-AI: {self.race_name} / {self.driver_code}")
        logger.info(f"Device: {self.tire_model.net.net[0].weight.device}")

        # Load tire model
        if not self.tire_model.load():
            logger.info("No trained tire model found — using compound profile fallback")

        # Resolve session
        if self.live:
            session = await self._resolve_live_session()
        else:
            session = await self.client.resolve_session(self.race_name)

        if not session:
            logger.error(f"Could not resolve session for '{self.race_name}'")
            return

        session_key = session["session_key"]
        meeting_key = session.get("meeting_key", 0)
        track_name = session.get("circuit_short_name", session.get("location", session.get("country_name", "Unknown")))

        # Resolve driver
        driver_number = await self.client.resolve_driver(session_key, self.driver_code)
        if not driver_number:
            logger.error(f"Could not resolve driver '{self.driver_code}' in session {session_key}")
            return

        # Get total laps
        total_laps = await self.client.get_total_laps(session_key)

        logger.info(
            f"Session: {track_name} (key={session_key}), "
            f"Driver: {self.driver_code} (#{driver_number}), "
            f"Total laps: {total_laps}"
        )

        # Fetch tire availability — check practice/qualifying usage
        available_compounds = await self.client.get_tire_availability(
            meeting_key, session_key, driver_number
        )

        # Session date for the dashboard
        session_date = str(session.get("date_start", ""))[:10]
        meeting_name = session.get("meeting_name") or session.get("country_name") or track_name

        # Build graph
        graph = build_graph(self.client, self.tire_model)
        compiled = graph.compile()

        # Initial state
        state: RaceState = {
            "session_key": session_key,
            "meeting_key": meeting_key,
            "driver_number": driver_number,
            "driver_code": self.driver_code.upper(),
            "total_laps": total_laps,
            "current_lap": 1,
            "track_name": track_name,
            "lap_data": [],
            "stint_data": [],
            "position_data": [],
            "race_control": [],
            "all_drivers": {},
            "weather": "dry",
            "safety_car": False,
            "virtual_safety_car": False,
            "red_flag": False,
            "spy_intelligence": [],
            "available_compounds": available_compounds,
            "strategy": None,
            "strategy_changed": False,
            "ghost": GhostCarState(),
            "briefing": "",
            "should_brief": False,
            "race_finished": False,
            "speed_multiplier": self.speed_multiplier,
            "replay_mode": not self.live,
        }

        # Seed InfluxDB session metadata (track, driver, date, tire allocation)
        self.influx.start_session(state, session_date=session_date, meeting_name=meeting_name)

        # Lap-by-lap execution
        for lap in range(1, total_laps + 1):
            state["current_lap"] = lap
            lap_start = time.time()

            try:
                # Run the pipeline for this lap
                result = await compiled.ainvoke(state)
                state.update(result)

                # Write metrics to InfluxDB
                self.influx.write_lap(state)
                if lap % 5 == 0 or lap == total_laps:
                    self.influx.write_stint_timeline(state)

                # Print lap summary
                self._print_lap_summary(state)

                # Check if race is finished
                if state.get("race_finished", False):
                    logger.info("Race finished!")
                    break

                # Pace control for replay mode
                if not self.live:
                    elapsed = time.time() - lap_start
                    target_interval = 1.0 / self.speed_multiplier  # roughly
                    if elapsed < target_interval:
                        await asyncio.sleep(target_interval - elapsed)

            except KeyboardInterrupt:
                logger.info("Race interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error on lap {lap}: {e}", exc_info=True)
                continue  # Don't crash the whole race for one bad lap

        # Final summary
        self._print_final_summary(state)

        # Cleanup
        self.influx.close()
        await self.client.close()

    async def _resolve_live_session(self) -> dict | None:
        """Find the latest live race session."""
        sessions = await self.client.get_sessions(session_type="Race")
        if sessions:
            return sessions[-1]
        return None

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as M:SS.s or H:MM:SS.s"""
        if seconds < 0:
            return f"-{RaceRunner._fmt_time(-seconds)}"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:04.1f}"
        return f"{m}:{s:04.1f}"

    @staticmethod
    def _fmt_delta(seconds: float) -> str:
        """Format a time delta with sign."""
        sign = "+" if seconds >= 0 else "-"
        abs_s = abs(seconds)
        if abs_s >= 60:
            m = int(abs_s // 60)
            s = abs_s % 60
            return f"{sign}{m}:{s:04.1f}"
        return f"{sign}{abs_s:.1f}s"

    def _print_lap_summary(self, state: RaceState) -> None:
        """Print a rich per-lap summary with flags, gaps, rivals, and ghost delta."""
        lap = state.get("current_lap", 0)
        total = state.get("total_laps", 0)
        briefing = state.get("briefing", "")
        should_brief = state.get("should_brief", False)
        ghost: GhostCarState = state.get("ghost", GhostCarState())
        strategy = state.get("strategy")

        # --- Current stint info ---
        stints = state.get("stint_data", [])
        compound = stints[-1].compound if stints else "?"
        tire_age = 0
        if stints:
            current_stint = stints[-1]
            tire_age = current_stint.tyre_age_at_start + (lap - current_stint.lap_start)

        # --- Lap time ---
        lap_time_str = ""
        lap_data = state.get("lap_data", [])
        for ld in reversed(lap_data):
            if ld.lap_number == lap and ld.lap_duration and ld.lap_duration > 30:
                lap_time_str = f" | {self._fmt_time(ld.lap_duration)}"
                break

        # --- Position ---
        positions = state.get("position_data", [])
        current_pos = None
        pos_str = ""
        for p in reversed(positions):
            if p.lap_number <= lap:
                current_pos = p.position
                pos_str = f" | P{p.position}"
                break

        # --- Flags (SC / VSC / Red) ---
        flag_str = ""
        sc = state.get("safety_car", False)
        vsc = state.get("virtual_safety_car", False)
        rf = state.get("red_flag", False)
        if rf:
            flag_str = " | \033[1;31m🔴 RED FLAG\033[0m"
        elif sc:
            flag_str = " | \033[1;33m🟡 SC\033[0m"
        elif vsc:
            flag_str = " | \033[1;33m🟡 VSC\033[0m"

        # --- Ghost delta ---
        delta_str = ""
        if ghost.cumulative_delta:
            delta = ghost.cumulative_delta[-1]
            color = "\033[32m" if delta >= 0 else "\033[31m"
            delta_str = f" | Ghost: {color}{self._fmt_delta(delta)}\033[0m"

        # --- Strategy hint ---
        strat_str = ""
        if strategy and len(strategy.stints) > 1:
            next_compound = strategy.stints[1][0]
            strat_str = f" | Next: {next_compound}"

        # Main line
        line = (
            f"Lap {lap:>2}/{total} | {compound:<4} age {tire_age:>2}"
            f"{lap_time_str}{pos_str}{flag_str}{delta_str}{strat_str}"
        )
        print(line)

        # --- Nearest rivals (car ahead + car behind) with GAPS ---
        spy_intel = state.get("spy_intelligence", [])
        if spy_intel and current_pos:
            ahead = None
            behind = None
            for s in spy_intel:
                if s.position == current_pos - 1:
                    ahead = s
                if s.position == current_pos + 1:
                    behind = s

            for arrow, rival in [("\033[36m\u2191\033[0m", ahead), ("\033[35m\u2193\033[0m", behind)]:
                if rival:
                    gap_str = ""
                    if rival.gap_seconds is not None:
                        gap_str = f" gap {self._fmt_time(rival.gap_seconds)}"
                    pit_hint = ""
                    if rival.pit_probability > 0.5:
                        pit_hint = f" \033[33mPIT~L{rival.predicted_pit_lap or '?'}\033[0m"
                    print(
                        f"  {arrow} P{rival.position} {rival.driver_code:>3} "
                        f"{rival.current_compound:<4} age {rival.tire_age:>2}"
                        f"{gap_str}{pit_hint}"
                    )

        # --- Radio briefing ---
        if should_brief and briefing:
            used_llm = state.get("used_llm", False)
            tag = "\033[1;36m\U0001f4fb AI\033[0m" if used_llm else "\033[33m\U0001f4fb\033[0m"
            print(f"  {tag} {briefing}")

    def _print_final_summary(self, state: RaceState) -> None:
        """Print a detailed end-of-race summary with tabular strategy comparison."""
        ghost: GhostCarState = state.get("ghost", GhostCarState())
        strategy = state.get("strategy")
        driver_code = state.get("driver_code", "???")

        # Accumulators for dashboard race_result write
        final_delta_for_dash = 0.0
        pit_diff_for_dash = 0
        est_finish_for_dash: int | None = None
        ai_positions_vs_actual = 0  # AI-vs-actual comparison (informational)
        verdict_for_dash = "EQUAL"

        W = 72
        print()
        print("\033[1m" + "=" * W + "\033[0m")
        print("\033[1m  RACE COMPLETE\033[0m")
        print("\033[1m" + "=" * W + "\033[0m")

        # --- Final position ---
        positions = state.get("position_data", [])
        total_laps = state.get("total_laps", 0)
        final_pos = None
        for p in reversed(positions):
            if p.lap_number <= total_laps:
                final_pos = p.position
                break

        if final_pos:
            print(f"  {driver_code} finished \033[1mP{final_pos}\033[0m")
            print()

        # --- Strategy comparison table ---
        # Use the ghost car's strategy for display — it represents the full
        # AI-recommended path from ghost start to race end, and matches the
        # time comparison below.  Falls back to current strategy if ghost
        # wasn't initialized.
        display_strat = ghost.strategy if ghost.initialized and ghost.strategy else strategy
        actual_stints = state.get("stint_data", [])

        print("\033[1m  STRATEGY COMPARISON\033[0m")
        print("  " + "-" * (W - 4))

        ai_start = ghost.start_lap if ghost.initialized else 1
        print(f"\033[4m  {'AI Strategy':^{W - 4}}\033[0m")
        if display_strat:
            ai_pits = len(display_strat.stints) - 1
            lap_cursor = ai_start
            for i, (compound, length) in enumerate(display_strat.stints):
                lap_end = lap_cursor + length - 1
                pit = "\033[33m+22s\033[0m" if i < len(display_strat.stints) - 1 else ""
                print(f"    S{i+1}  {compound:<13} {length:>3} laps  (L{lap_cursor}-L{lap_end})  {pit}")
                lap_cursor = lap_end + 1
            if ai_start > 1:
                print(f"    (evaluation from lap {ai_start})")
            print(f"    Stops: {ai_pits}")
        print()

        print(f"\033[4m  {'Actual Strategy':^{W - 4}}\033[0m")
        if actual_stints:
            actual_pits = len(actual_stints) - 1
            for i, s in enumerate(actual_stints):
                end = s.lap_end or total_laps
                length = end - s.lap_start + 1
                pit = "\033[33m+22s\033[0m" if i < len(actual_stints) - 1 else ""
                print(f"    S{i+1}  {s.compound:<13} {length:>3} laps  (L{s.lap_start}-L{end})  {pit}")
            print(f"    Stops: {actual_pits}")
        print()

        # --- Time comparison ---
        print("\033[1m  TIME COMPARISON\033[0m")
        print("  " + "-" * (W - 4))

        if ghost.initialized:
            eval_laps = total_laps - ghost.start_lap + 1
            print(f"    Evaluation:         Laps {ghost.start_lap}-{total_laps} ({eval_laps} laps)")

            g_time = ghost.ghost_total_time
            a_time = ghost.actual_total_time
            delta = a_time - g_time  # positive = AI faster

            print(f"    AI ghost total:     \033[1m{self._fmt_time(g_time)}\033[0m")
            print(f"    Actual total:       \033[1m{self._fmt_time(a_time)}\033[0m")

            if delta > 0:
                color = "\033[32m"
                verdict = "AI FASTER"
            elif delta < 0:
                color = "\033[31m"
                verdict = "ACTUAL FASTER"
            else:
                color = ""
                verdict = "EQUAL"
            print(f"    Delta:              {color}\033[1m{self._fmt_delta(delta)}\033[0m ({verdict})")
            final_delta_for_dash = float(delta)
            verdict_for_dash = verdict

            # Pit stop accounting
            ai_pits = len(display_strat.stints) - 1 if display_strat else 0
            actual_pits = len(actual_stints) - 1 if actual_stints else 0
            pit_diff = actual_pits - ai_pits
            pit_diff_for_dash = int(pit_diff)
            if pit_diff != 0:
                pit_time = pit_diff * 22.0  # PIT_STOP_LOSS = 22s
                sign = "+" if pit_diff > 0 else ""
                print(f"    Pit stop diff:      {sign}{pit_diff} stop(s) = {sign}{pit_time:.0f}s")

            # Estimated position gain/loss using actual rival gaps and
            # circuit-specific passing difficulty.
            if abs(delta) > 1.0 and final_pos:
                from pitwall.models.overtake_model import TRACK_OVERTAKE_DIFFICULTY
                track_name = state.get("track_name", "")
                track_key = track_name.lower().replace(" ", "")
                difficulty = TRACK_OVERTAKE_DIFFICULTY.get("default", 0.5)
                for key, diff in TRACK_OVERTAKE_DIFFICULTY.items():
                    if key in track_key:
                        difficulty = diff
                        break

                # Use actual finish gaps to count how many positions
                # the time delta could realistically gain/lose.
                # In hard-to-pass circuits, you need a much larger gap.
                # Difficulty scales the required delta: 0.5 (easy) → 1x,
                # 0.95 (Monaco) → ~2x the gap needed.
                difficulty_mult = 1.0 + difficulty
                spy_intel = state.get("spy_intelligence", [])
                est_positions = 0
                remaining_delta = abs(delta)
                if delta > 0:
                    # AI faster — count rivals ahead we could have passed
                    rivals_ahead = sorted(
                        [s for s in spy_intel if s.position < final_pos],
                        key=lambda s: s.position, reverse=True
                    )
                    for rival in rivals_ahead:
                        gap = rival.gap_seconds if rival.gap_seconds else 5.0
                        needed = gap * difficulty_mult
                        if remaining_delta >= needed:
                            est_positions += 1
                            remaining_delta -= gap
                        else:
                            break
                else:
                    # Actual faster — count rivals behind who could have passed us
                    rivals_behind = sorted(
                        [s for s in spy_intel if s.position > final_pos],
                        key=lambda s: s.position
                    )
                    for rival in rivals_behind:
                        gap = rival.gap_seconds if rival.gap_seconds else 5.0
                        needed = gap * difficulty_mult
                        if remaining_delta >= needed:
                            est_positions += 1
                            remaining_delta -= gap
                        else:
                            break

                if est_positions == 0:
                    # Delta doesn't cover any real gap — don't fabricate positions
                    pass

                diff_label = f"{difficulty:.0%} passing difficulty"
                if est_positions > 0:
                    if delta > 0:
                        est_pos = max(1, final_pos - est_positions)
                        print(f"    Est. AI finish:     ~P{est_pos} (up to {est_positions} pos gained, {diff_label})")
                        est_finish_for_dash = est_pos
                        ai_positions_vs_actual = est_positions
                    else:
                        est_pos = min(20, final_pos + est_positions)
                        print(f"    Est. AI finish:     ~P{est_pos} (up to {est_positions} pos lost, {diff_label})")
                        est_finish_for_dash = est_pos
                        ai_positions_vs_actual = -est_positions
                else:
                    print(f"    Est. AI finish:     ~P{final_pos} (same position, {self._fmt_delta(delta)} gap insufficient to pass)")
                    est_finish_for_dash = final_pos
        else:
            print("    Ghost car was not initialized (not enough clean racing laps)")

        # --- Tire availability ---
        avail = state.get("available_compounds")
        if avail:
            print()
            print(f"  \033[1mTIRE ALLOCATION (pre-race)\033[0m")
            print("  " + "-" * (W - 4))
            for comp in ["SOFT", "MEDIUM", "HARD"]:
                sets = avail.get(comp, 0)
                bar = "\033[32m" + "█" * sets + "\033[0m" + "░" * (8 - sets)
                print(f"    {comp:<8} {bar} {sets} set(s)")

        # --- Post-race strategy debrief ---
        print()
        print("\033[1m  POST-RACE STRATEGY DEBRIEF\033[0m")
        print("  " + "-" * (W - 4))
        principal = Principal()
        debrief = principal.generate_debrief(state)
        if debrief:
            # Word-wrap to terminal width, indented
            import textwrap
            for paragraph in debrief.split("\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    wrapped = textwrap.fill(paragraph, width=W - 4, initial_indent="    ", subsequent_indent="    ")
                    print(wrapped)
                    print()
        else:
            print("    (Debrief unavailable — no LLM configured)")

        print("\033[1m" + "=" * W + "\033[0m")

        # --- Push end-of-race summary to the Grafana dashboard ---
        # Compute start position and positions_gained for the dashboard only.
        start_pos = None
        if positions:
            earliest = min(positions, key=lambda x: x.lap_number)
            start_pos = earliest.position
        race_positions_gained = 0
        if start_pos is not None and final_pos is not None:
            race_positions_gained = int(start_pos - final_pos)
        try:
            self.influx.write_race_result(
                state,
                finish_position=final_pos,
                start_position=start_pos,
                positions_gained=race_positions_gained,
                ai_positions_vs_actual=ai_positions_vs_actual,
                final_delta=final_delta_for_dash,
                pit_diff=pit_diff_for_dash,
                estimated_ai_finish=est_finish_for_dash,
                verdict=verdict_for_dash,
            )
            if debrief:
                self.influx.write_debrief(state, debrief)
        except Exception as e:
            logger.debug(f"Dashboard final write failed: {e}")
