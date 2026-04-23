"""Principal agent: LLM-powered strategy briefing generator.

Uses Groq (llama-3.3-70b-versatile) to generate natural-language strategy briefings.
Falls back to template-based briefings when rate-limited or when LLM is unavailable.

Rate limit handling: Groq free tier = 100k tokens/day. On first 429, set a class-level
flag and skip ALL future LLM calls for the session.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pitwall.state import Strategy, GhostCarState, SpyIntel

if TYPE_CHECKING:
    from pitwall.state import RaceState

logger = logging.getLogger(__name__)


# --- Singleton Groq client ---
# Created once per process, reused across all Principal instances.
# This avoids the class-level _llm_available flag getting poisoned
# if Principal is instantiated before the env var is set.

_groq_client = None
_groq_init_done = False
_groq_rate_limited = False


def _get_groq_client():
    """Lazy-init the Groq client singleton. Safe to call repeatedly."""
    global _groq_client, _groq_init_done

    if _groq_init_done:
        return _groq_client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.info("No GROQ_API_KEY — LLM briefings disabled, using templates")
        _groq_init_done = True
        return None

    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        _groq_init_done = True
        logger.info("Groq LLM client initialized (llama-3.3-70b-versatile)")
        return _groq_client
    except ImportError:
        logger.info("groq package not installed — using template-based briefings")
    except Exception as e:
        logger.warning(f"Failed to initialize Groq: {e}")

    _groq_init_done = True
    return None


class Principal:
    """Strategy briefing generator with LLM backend and template fallback."""

    def __init__(self):
        self._client = _get_groq_client()

    @property
    def llm_active(self) -> bool:
        return self._client is not None and not _groq_rate_limited

    def should_brief(self, state: RaceState) -> bool:
        """Determine if we should generate a briefing this lap.

        Call LLM on: lap 1, strategy changes, opponent pits, every 5 laps, race end.
        This keeps us at ~15-20 calls per race, well within Groq limits.
        """
        current_lap = state.get("current_lap", 1)
        total_laps = state.get("total_laps", 57)

        # Lap 1 — always brief
        if current_lap == 1:
            return True

        # Race end
        if current_lap >= total_laps:
            return True

        # Strategy changed
        if state.get("strategy_changed", False):
            return True

        # Opponent pitted (check spy intelligence for recent pit events)
        for intel in state.get("spy_intelligence", []):
            if intel.pit_probability > 0.8 and intel.tire_age <= 2:
                return True

        # Every 5 laps
        if current_lap % 5 == 0:
            return True

        return False

    def generate_briefing(self, state: RaceState) -> tuple[str, bool]:
        """Generate a strategy briefing.

        Returns (briefing_text, used_llm).
        """
        if self.llm_active:
            text = self._llm_briefing(state)
            if text is not None:
                return text, True
        return self._template_briefing(state), False

    def _llm_briefing(self, state: RaceState) -> str | None:
        """Generate briefing via Groq LLM. Returns None on failure."""
        prompt = self._build_prompt(state)
        return self._call_llm(prompt, max_tokens=200)

    # --- Context extraction ---

    def _extract_context(self, state: RaceState) -> dict:
        """Extract all actionable race context from state into a flat dict."""
        current_lap = state.get("current_lap", 1)
        total_laps = state.get("total_laps", 57)
        laps_remaining = total_laps - current_lap + 1
        strategy = state.get("strategy")
        ghost = state.get("ghost", GhostCarState())
        spy_intel = state.get("spy_intelligence", [])
        stints = state.get("stint_data", [])
        positions = state.get("position_data", [])
        lap_data = state.get("lap_data", [])

        # Current tire state
        current_compound = "UNKNOWN"
        tire_age = 0
        stint_number = 0
        if stints:
            current_stint = stints[-1]
            current_compound = current_stint.compound
            tire_age = current_stint.tyre_age_at_start + (current_lap - current_stint.lap_start)
            stint_number = len(stints)

        # Current position
        position = None
        for p in reversed(positions):
            if p.lap_number <= current_lap:
                position = p.position
                break

        # Last lap time
        last_lap_time = None
        for ld in reversed(lap_data):
            if ld.lap_number == current_lap - 1 and ld.lap_duration:
                last_lap_time = ld.lap_duration
                break

        # Next pit stop: from current strategy, compute the actual lap number
        pit_lap = None
        pit_compound = None
        laps_to_pit = None
        if strategy and len(strategy.stints) >= 2:
            first_stint_laps = strategy.stints[0][1]
            pit_lap = current_lap + first_stint_laps - 1
            pit_compound = strategy.stints[1][0]
            laps_to_pit = first_stint_laps

        # Ghost delta
        ghost_delta = None
        if ghost.initialized and ghost.cumulative_delta:
            ghost_delta = ghost.cumulative_delta[-1]

        # Nearby drivers with pit threat
        threats = []
        for s in spy_intel[:4]:
            threats.append({
                "code": s.driver_code,
                "pos": s.position,
                "compound": s.current_compound,
                "tire_age": s.tire_age,
                "pit_prob": s.pit_probability,
                "predicted_pit_lap": s.predicted_pit_lap,
                "predicted_compound": s.predicted_compound,
            })

        return {
            "current_lap": current_lap,
            "total_laps": total_laps,
            "laps_remaining": laps_remaining,
            "driver_code": state.get("driver_code", "driver"),
            "position": position,
            "current_compound": current_compound,
            "tire_age": tire_age,
            "stint_number": stint_number,
            "last_lap_time": last_lap_time,
            "weather": state.get("weather", "dry"),
            "safety_car": state.get("safety_car", False),
            "virtual_safety_car": state.get("virtual_safety_car", False),
            "strategy_changed": state.get("strategy_changed", False),
            "strategy_stints": strategy.stints if strategy else [],
            "pit_lap": pit_lap,
            "pit_compound": pit_compound,
            "laps_to_pit": laps_to_pit,
            "ghost_delta": ghost_delta,
            "threats": threats,
            "race_finished": current_lap >= total_laps,
        }

    # --- LLM prompt ---

    def _build_prompt(self, state: RaceState) -> str:
        """Build a structured, grounded prompt for the LLM."""
        ctx = self._extract_context(state)

        lines = [
            f"You are the F1 race engineer for driver {ctx['driver_code']}. "
            f"Give a 2-3 sentence strategy briefing based ONLY on the data below. "
            f"Refer to the driver ONLY as '{ctx['driver_code']}' — do NOT use first names or full names. "
            f"Focus on: tire strategy, pit windows, and nearby rival threats. Be direct.",
            "",
            "=== RACE STATE ===",
            f"Lap: {ctx['current_lap']}/{ctx['total_laps']} ({ctx['laps_remaining']} remaining)",
        ]

        if ctx["position"]:
            lines.append(f"Position: P{ctx['position']}")

        lines.append(
            f"Current tyres: {ctx['current_compound']} "
            f"(age: {ctx['tire_age']} laps, stint {ctx['stint_number']})"
        )

        if ctx["last_lap_time"]:
            lines.append(f"Last lap: {ctx['last_lap_time']:.3f}s")

        lines.append(f"Weather: {ctx['weather']}")

        if ctx["safety_car"]:
            lines.append("STATUS: SAFETY CAR DEPLOYED")
        elif ctx["virtual_safety_car"]:
            lines.append("STATUS: VIRTUAL SAFETY CAR")

        # Strategy section
        lines.append("")
        lines.append("=== PIT STRATEGY ===")
        if ctx["strategy_stints"]:
            for i, (compound, length) in enumerate(ctx["strategy_stints"]):
                label = "CURRENT" if i == 0 else f"STINT {i + 1}"
                lines.append(f"  {label}: {compound} for {length} laps")

        if ctx["pit_lap"] and ctx["pit_compound"]:
            lines.append(
                f"  >> NEXT STOP: Lap ~{ctx['pit_lap']} — switch to "
                f"{ctx['pit_compound']} (in {ctx['laps_to_pit']} laps)"
            )
        elif ctx["strategy_stints"] and len(ctx["strategy_stints"]) == 1:
            lines.append("  >> NO MORE STOPS — push to the end")

        if ctx["strategy_changed"]:
            lines.append("  NOTE: Strategy has CHANGED this lap.")

        # Opponents
        if ctx["threats"]:
            lines.append("")
            lines.append("=== NEARBY DRIVERS ===")
            for t in ctx["threats"]:
                pit_info = ""
                if t["pit_prob"] > 0.4:
                    pit_info = (
                        f" — likely pits lap ~{t['predicted_pit_lap'] or '?'} "
                        f"for {t['predicted_compound']} ({t['pit_prob']:.0%})"
                    )
                lines.append(
                    f"  P{t['pos']} {t['code']}: {t['compound']} age {t['tire_age']}{pit_info}"
                )

        # Ghost car
        if ctx["ghost_delta"] is not None:
            lines.append("")
            d = ctx["ghost_delta"]
            verdict = "AI strategy faster" if d > 0 else "actual faster"
            sign = "+" if d >= 0 else "-"
            ad = abs(d)
            if ad >= 60:
                dm, ds = int(ad // 60), ad % 60
                d_str = f"{sign}{dm}:{ds:04.1f}"
            else:
                d_str = f"{sign}{ad:.1f}s"
            lines.append(
                f"=== AI EVALUATION === Delta: {d_str} ({verdict})"
            )

        if ctx["race_finished"]:
            lines.append("")
            lines.append("CHEQUERED FLAG.")

        return "\n".join(lines)

    # --- Template fallback ---

    def _template_briefing(self, state: RaceState) -> str:
        """Generate a grounded template-based briefing (no LLM)."""
        ctx = self._extract_context(state)
        parts = []

        # Position and lap
        pos_str = f"P{ctx['position']}, " if ctx["position"] else ""
        parts.append(f"{pos_str}lap {ctx['current_lap']}/{ctx['total_laps']}.")

        # Safety car
        if ctx["safety_car"]:
            parts.append("Safety car — box this lap if window is open.")
        elif ctx["virtual_safety_car"]:
            parts.append("VSC — maintain delta.")

        # Current tires + strategy call
        parts.append(f"On {ctx['current_compound']}, age {ctx['tire_age']}.")

        if ctx["strategy_changed"]:
            parts.append("Strategy update:")

        if ctx["pit_lap"] and ctx["pit_compound"] and ctx["laps_to_pit"]:
            if ctx["laps_to_pit"] <= 3:
                parts.append(
                    f"Box box box in {ctx['laps_to_pit']} laps for {ctx['pit_compound']}."
                )
            elif ctx["laps_to_pit"] <= 8:
                parts.append(f"Plan to box lap {ctx['pit_lap']} for {ctx['pit_compound']}.")
            else:
                parts.append(
                    f"Next stop around lap {ctx['pit_lap']}, switch to {ctx['pit_compound']}."
                )
        elif not ctx["pit_lap"]:
            parts.append("No more stops, push to the end.")

        # Threats
        pit_threats = [t for t in ctx["threats"] if t["pit_prob"] > 0.5]
        if pit_threats:
            t = pit_threats[0]
            parts.append(
                f"P{t['pos']} {t['code']} likely pitting soon for {t['predicted_compound']}."
            )

        # Ghost delta
        if ctx["ghost_delta"] is not None:
            d = ctx["ghost_delta"]
            if abs(d) > 1.0:
                who = "AI strategy" if d > 0 else "Current strategy"
                ad = abs(d)
                if ad >= 60:
                    dm, ds = int(ad // 60), ad % 60
                    parts.append(f"{who} ahead by {dm}:{ds:04.1f}.")
                else:
                    parts.append(f"{who} ahead by {ad:.1f}s.")

        if ctx["race_finished"]:
            parts.append("Chequered flag!")

        return " ".join(parts)


    # --- Post-race debrief ---

    def generate_debrief(self, state: RaceState) -> str | None:
        """Generate a detailed post-race strategy debrief.

        This is called once after the race finishes. It uses the full race
        data (all stints, lap times, SC/VSC/RF events, ghost evaluation)
        to produce a grounded analysis of where time was gained/lost.

        The debrief ALWAYS attempts the LLM (even if mid-race briefings
        were rate-limited) — it's the single most important output of the
        race and worth one fresh attempt. Falls back to a rich template
        if the LLM is unavailable.
        """
        prompt = self._build_debrief_prompt(state)
        # Try the LLM even if briefings got rate-limited earlier —
        # token quotas often reset, and the debrief is worth the attempt.
        if self._client is not None:
            text = self._call_llm(prompt, max_tokens=700)
            if text is not None:
                return text
        return self._template_debrief(state)

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str | None:
        """Call the Groq LLM with a prompt. Returns None on failure."""
        global _groq_rate_limited
        try:
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.5,
            )
            text = response.choices[0].message.content.strip()
            text = text.strip('"').strip("'")
            return text
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg or "limit" in error_msg:
                logger.info(
                    "Groq rate limit hit — debrief will use template fallback"
                )
                _groq_rate_limited = True
            else:
                logger.warning(f"Groq API error during debrief: {e}")
            return None

    def _build_debrief_prompt(self, state: RaceState) -> str:
        """Build a detailed post-race prompt with all race data for the debrief."""
        driver_code = state.get("driver_code", "driver")
        total_laps = state.get("total_laps", 57)
        track_name = state.get("track_name", "Unknown")
        ghost: GhostCarState = state.get("ghost", GhostCarState())
        actual_stints = state.get("stint_data", [])
        lap_data = state.get("lap_data", [])
        race_control = state.get("race_control", [])
        spy_intel = state.get("spy_intelligence", [])
        positions = state.get("position_data", [])

        # Final position
        final_pos = None
        for p in reversed(positions):
            if p.lap_number <= total_laps:
                final_pos = p.position
                break

        # Build the prompt
        lines = [
            f"You are the head of F1 race strategy for {driver_code}. The race at {track_name} "
            f"({total_laps} laps) has just finished. Write a post-race strategy debrief.",
            "",
            "RULES:",
            "- Base EVERY statement on the data below. Do not invent facts.",
            f"- Refer to the driver ONLY as '{driver_code}'.",
            "- Structure: Overview → Stint-by-stint analysis → Key moments (SC/VSC/RF) → "
            "AI vs Actual comparison → Conclusion.",
            "- Be specific with lap numbers, time deltas, and compound choices.",
            "- Explain WHERE and WHY time was gained or lost.",
            "- Keep it under 250 words. Use short paragraphs, not bullet points.",
            "",
        ]

        # Final result
        lines.append("=== RESULT ===")
        if final_pos:
            lines.append(f"Finished: P{final_pos}")
        lines.append("")

        # Actual strategy
        lines.append("=== ACTUAL STRATEGY ===")
        for i, s in enumerate(actual_stints):
            end = s.lap_end or total_laps
            length = end - s.lap_start + 1
            lines.append(
                f"  Stint {i+1}: {s.compound} — Laps {s.lap_start}-{end} "
                f"({length} laps, tire age at start: {s.tyre_age_at_start})"
            )
        lines.append(f"  Total pit stops: {max(0, len(actual_stints) - 1)}")
        lines.append("")

        # AI ghost strategy
        if ghost.initialized and ghost.strategy:
            lines.append("=== AI STRATEGY ===")
            cursor = ghost.start_lap
            for i, (compound, length) in enumerate(ghost.strategy.stints):
                lap_end = cursor + length - 1
                lines.append(
                    f"  Stint {i+1}: {compound} — Laps {cursor}-{lap_end} ({length} laps)"
                )
                cursor = lap_end + 1
            lines.append(f"  Total pit stops: {max(0, len(ghost.strategy.stints) - 1)}")
            lines.append("")

            # Delta summary
            delta = ghost.actual_total_time - ghost.ghost_total_time
            lines.append("=== TIME COMPARISON ===")
            lines.append(f"  AI total:     {ghost.ghost_total_time:.1f}s")
            lines.append(f"  Actual total: {ghost.actual_total_time:.1f}s")
            sign = "+" if delta >= 0 else ""
            if abs(delta) < 0.1:
                verdict = "EQUAL"
            elif delta > 0:
                verdict = "AI faster"
            else:
                verdict = "Actual faster"
            lines.append(f"  Delta: {sign}{delta:.1f}s ({verdict})")
            lines.append("")

            # Delta progression — sample every ~10 laps for context
            if ghost.cumulative_delta:
                lines.append("=== DELTA PROGRESSION (sampled) ===")
                sample_interval = max(1, len(ghost.cumulative_delta) // 8)
                for idx in range(0, len(ghost.cumulative_delta), sample_interval):
                    lap_num = ghost.start_lap + idx
                    d = ghost.cumulative_delta[idx]
                    lines.append(f"  Lap {lap_num:>3}: {'+' if d >= 0 else ''}{d:.1f}s")
                # Always include the final value
                if (len(ghost.cumulative_delta) - 1) % sample_interval != 0:
                    d = ghost.cumulative_delta[-1]
                    lap_num = ghost.start_lap + len(ghost.cumulative_delta) - 1
                    lines.append(f"  Lap {lap_num:>3}: {'+' if d >= 0 else ''}{d:.1f}s (final)")
                lines.append("")

        # Race control events
        sc_events = []
        for msg in sorted(race_control, key=lambda m: m.lap_number):
            cat = (msg.category or "").lower()
            text = msg.message.lower()
            if any(kw in cat or kw in text for kw in ["safety car", "red flag", "virtual"]):
                sc_events.append(msg)

        if sc_events:
            lines.append("=== SAFETY CAR / RED FLAG EVENTS ===")
            for msg in sc_events:
                lines.append(f"  Lap {msg.lap_number}: [{msg.category}] {msg.message}")
            lines.append("")
            lines.append(
                "NOTE: Under full SC or red flag, gaps are neutralized (delta resets to 0). "
                "Under VSC, gaps are maintained (delta freezes)."
            )
            lines.append("")

        # Lap time samples (fastest, slowest non-pit, average)
        clean_times = []
        pit_times = []
        for ld in lap_data:
            if ld.lap_duration and ld.lap_duration > 30 and ld.lap_number > 1:
                if ld.is_pit_in_lap or ld.is_pit_out_lap:
                    pit_times.append((ld.lap_number, ld.lap_duration))
                elif ld.lap_duration < 200:
                    clean_times.append((ld.lap_number, ld.lap_duration))

        if clean_times:
            lines.append("=== PACE DATA ===")
            fastest = min(clean_times, key=lambda x: x[1])
            slowest = max(clean_times, key=lambda x: x[1])
            avg = sum(t for _, t in clean_times) / len(clean_times)
            lines.append(f"  Fastest clean lap: {fastest[1]:.3f}s (Lap {fastest[0]})")
            lines.append(f"  Slowest clean lap: {slowest[1]:.3f}s (Lap {slowest[0]})")
            lines.append(f"  Average clean lap: {avg:.3f}s")
            lines.append(f"  Clean laps counted: {len(clean_times)}")
            lines.append("")

        # Nearby final positions
        if spy_intel:
            lines.append("=== NEARBY RIVALS (end of race) ===")
            for s in spy_intel[:6]:
                lines.append(
                    f"  P{s.position} {s.driver_code}: {s.current_compound} age {s.tire_age}"
                    + (f", gap {s.gap_seconds:.1f}s" if s.gap_seconds else "")
                )
            lines.append("")

        return "\n".join(lines)

    def _template_debrief(self, state: RaceState) -> str:
        """Generate a template-based debrief when LLM is unavailable.

        Structured as multi-paragraph analysis so it still reads like a
        race engineer's debrief, not a one-line summary.
        """
        driver_code = state.get("driver_code", "driver")
        ghost: GhostCarState = state.get("ghost", GhostCarState())
        actual_stints = state.get("stint_data", [])
        total_laps = state.get("total_laps", 57)
        track_name = state.get("track_name", "the track")
        positions = state.get("position_data", [])
        race_control = state.get("race_control", [])
        lap_data = state.get("lap_data", [])

        final_pos = None
        for p in reversed(positions):
            if p.lap_number <= total_laps:
                final_pos = p.position
                break

        paragraphs: list[str] = []

        # --- Overview ---
        pos_str = f"P{final_pos}" if final_pos else "an unknown position"
        stops = max(0, len(actual_stints) - 1)
        paragraphs.append(
            f"OVERVIEW: {driver_code} finished {pos_str} at {track_name} "
            f"after {total_laps} laps and {stops} pit stop{'s' if stops != 1 else ''}."
        )

        # --- Stint-by-stint analysis ---
        if actual_stints:
            stint_lines = ["STINTS:"]
            for i, s in enumerate(actual_stints):
                end = s.lap_end or total_laps
                length = end - s.lap_start + 1
                # Compute avg pace for this stint
                stint_times = [
                    ld.lap_duration for ld in lap_data
                    if ld.lap_number
                    and s.lap_start <= ld.lap_number <= end
                    and ld.lap_duration and 30 < ld.lap_duration < 200
                    and not ld.is_pit_in_lap and not ld.is_pit_out_lap
                ]
                pace_str = ""
                if stint_times:
                    avg = sum(stint_times) / len(stint_times)
                    pace_str = f", avg {avg:.2f}s"
                stint_lines.append(
                    f"  S{i+1}: {s.compound} for {length} laps "
                    f"(L{s.lap_start}-L{end}, age {s.tyre_age_at_start} at start{pace_str})"
                )
            paragraphs.append("\n".join(stint_lines))

        # --- SC / VSC / Red Flag events ---
        sc_events = []
        for msg in sorted(race_control, key=lambda m: m.lap_number):
            cat = (msg.category or "").lower()
            text = msg.message.lower()
            if any(kw in cat or kw in text for kw in ["safety car", "red flag", "virtual"]):
                sc_events.append(msg)
        if sc_events:
            event_lines = ["KEY MOMENTS:"]
            for msg in sc_events[:8]:  # Cap at 8 to avoid spam
                event_lines.append(f"  L{msg.lap_number}: {msg.message}")
            event_lines.append(
                "  (Under full SC / red flag, time gaps are neutralized — "
                "the ghost delta resets to 0. Under VSC, gaps are maintained.)"
            )
            paragraphs.append("\n".join(event_lines))

        # --- AI vs Actual comparison ---
        if ghost.initialized and ghost.strategy:
            ai_cursor = ghost.start_lap
            ai_desc_parts = []
            for compound, length in ghost.strategy.stints:
                ai_desc_parts.append(f"{compound} L{ai_cursor}-L{ai_cursor + length - 1}")
                ai_cursor += length
            ai_stops = max(0, len(ghost.strategy.stints) - 1)

            delta = ghost.actual_total_time - ghost.ghost_total_time
            if abs(delta) < 0.1:
                verdict = "strategies effectively tied"
            elif delta > 0:
                verdict = f"AI strategy would have been {delta:.1f}s faster"
            else:
                verdict = f"actual strategy was {abs(delta):.1f}s faster"

            comparison = (
                f"AI vs ACTUAL: The AI recommended {', '.join(ai_desc_parts)} "
                f"({ai_stops} stop{'s' if ai_stops != 1 else ''}). "
                f"Over the evaluated window (laps {ghost.start_lap}-{total_laps}), "
                f"{verdict}."
            )
            paragraphs.append(comparison)

            # Delta progression highlights
            if ghost.cumulative_delta and len(ghost.cumulative_delta) >= 5:
                deltas = ghost.cumulative_delta
                # Find biggest swing point (where delta changed most)
                max_gain_idx = max(range(len(deltas)), key=lambda i: deltas[i])
                max_loss_idx = min(range(len(deltas)), key=lambda i: deltas[i])

                swing_lines = ["DELTA HIGHLIGHTS:"]
                swing_lines.append(
                    f"  Peak AI advantage: {deltas[max_gain_idx]:+.1f}s at lap {ghost.start_lap + max_gain_idx}"
                )
                swing_lines.append(
                    f"  Peak actual advantage: {deltas[max_loss_idx]:+.1f}s at lap {ghost.start_lap + max_loss_idx}"
                )
                swing_lines.append(f"  Final delta: {deltas[-1]:+.1f}s")
                paragraphs.append("\n".join(swing_lines))

        # --- Conclusion ---
        if ghost.initialized:
            delta = ghost.actual_total_time - ghost.ghost_total_time
            if abs(delta) < 0.5:
                conclusion = (
                    f"CONCLUSION: The actual strategy matched the AI's optimal "
                    f"strategy almost exactly. Solid execution by the team."
                )
            elif delta > 0:
                conclusion = (
                    f"CONCLUSION: The AI identified a faster path worth {delta:.1f}s. "
                    f"Key divergence was in compound choice and/or pit timing."
                )
            else:
                conclusion = (
                    f"CONCLUSION: The actual strategy outperformed the AI by {abs(delta):.1f}s. "
                    f"In-race conditions the AI couldn't foresee likely contributed."
                )
            paragraphs.append(conclusion)

        return "\n\n".join(paragraphs)


def run_principal(state: RaceState) -> dict:
    """LangGraph node: generate strategy briefing."""
    principal = Principal()

    should = principal.should_brief(state)
    if not should:
        return {"briefing": state.get("briefing", ""), "should_brief": False}

    briefing, used_llm = principal.generate_briefing(state)
    return {"briefing": briefing, "should_brief": True, "used_llm": used_llm}
