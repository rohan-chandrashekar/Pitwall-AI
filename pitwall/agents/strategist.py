"""Genetic Algorithm strategy optimizer using DEAP.

Evolves optimal stint plans (compound, stint_length) tuples that minimize
predicted total race time while enforcing FIA compound rules.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from deap import base, creator, tools, algorithms

from pitwall.models.tire_model import (
    TireModel,
    PIT_STOP_LOSS,
    PIT_STOP_LOSS_SC,
    PIT_STOP_LOSS_VSC,
)
from pitwall.state import Strategy

if TYPE_CHECKING:
    from pitwall.state import RaceState

logger = logging.getLogger(__name__)

# --- Constants ---

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]
MIN_STINT = 5
MAX_STINT = 35  # absolute max (overridden per-compound below)
MIN_STINTS_COUNT = 2  # must use at least 2 stints for compound diversity
MAX_STINTS_COUNT = 4  # practical maximum

# Per-compound stint limits — prevents the GA from creating fantasy strategies.
# Based on real F1 stint length data.  Previous limits (MEDIUM=30, HARD=35)
# were too restrictive — Bahrain 2025 showed PIA running MEDIUM for 43 laps
# and winning P1.  The tire model's cliff parameters now handle degradation
# penalties for excessively long stints, so hard caps can be more permissive.
COMPOUND_MAX_STINT = {
    "SOFT": 24,
    "MEDIUM": 40,
    "HARD": 45,
    "INTERMEDIATE": 30,
    "WET": 22,
}


def _max_stint_for(compound: str) -> int:
    """Get the maximum stint length for a compound."""
    return COMPOUND_MAX_STINT.get(compound, MAX_STINT)

# DEAP creator setup — done once at module level
# Guard against re-creation if module is reloaded
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def _available_compounds(weather: str) -> list[str]:
    """Get available compounds, filtered by actual tire set availability.

    Uses the module-level _tire_availability context (set by run_strategist)
    to exclude compounds the driver has no remaining sets of.
    """
    if weather == "wet":
        return WET_COMPOUNDS

    if _tire_availability:
        # Only include compounds the driver actually has sets of
        available = [c for c in DRY_COMPOUNDS if _tire_availability.get(c, 0) > 0]
        if len(available) >= 2:
            return available
        # Fallback: if somehow < 2 compounds available, use all dry
        # (shouldn't happen in practice — FIA requires 2+ compound use)
        logger.warning(
            f"Only {len(available)} compounds available from tire allocation: "
            f"{_tire_availability}. Using all dry compounds as fallback."
        )

    return DRY_COMPOUNDS


def _is_dry_race(weather: str) -> bool:
    return weather != "wet"


# --- Individual creation ---

def create_individual(remaining_laps: int, first_compound: str,
                      weather: str) -> list[tuple[str, int]]:
    """Create a random valid strategy individual.

    HARD CONSTRAINT: dry races use >= 2 different dry compounds.
    First stint compound is locked.
    """
    compounds = _available_compounds(weather)
    is_dry = _is_dry_race(weather)

    if remaining_laps < 2 * MIN_STINT:
        # Not enough laps for two legal stints — single stint only
        return [(first_compound, remaining_laps)]

    # Decide number of stints (2-5, but bounded by remaining laps)
    max_possible_stints = min(MAX_STINTS_COUNT, remaining_laps // MIN_STINT)
    max_possible_stints = max(MIN_STINTS_COUNT, max_possible_stints)
    n_stints = random.randint(MIN_STINTS_COUNT, max_possible_stints)

    stints: list[tuple[str, int]] = []

    # First stint: locked compound, respect per-compound limit
    first_max = min(_max_stint_for(first_compound), remaining_laps - (n_stints - 1) * MIN_STINT)
    first_length = random.randint(MIN_STINT, max(MIN_STINT, first_max))
    stints.append((first_compound, first_length))
    laps_left = remaining_laps - first_length

    # For dry races, ensure at least one stint uses a different compound
    other_compounds = [c for c in compounds if c != first_compound]
    forced_different = is_dry and len(other_compounds) > 0

    for i in range(1, n_stints):
        is_last = (i == n_stints - 1)
        remaining_stints_after = n_stints - i - 1

        # Compound selection (determine first so we know its max stint)
        if forced_different and i == 1:
            compound = random.choice(other_compounds)
            forced_different = False
        else:
            compound = random.choice(compounds)

        if is_last:
            length = laps_left
        else:
            max_this = min(_max_stint_for(compound), laps_left - remaining_stints_after * MIN_STINT)
            max_this = max(MIN_STINT, max_this)
            length = random.randint(MIN_STINT, max_this)

        stints.append((compound, length))
        laps_left -= length

    # Final safety: repair to guarantee validity
    stints = _repair(stints, remaining_laps, first_compound, weather)
    return stints


def _generate_baseline_strategies(remaining_laps: int, first_compound: str,
                                  weather: str,
                                  first_stint_age: int = 0) -> list[list[tuple[str, int]]]:
    """Generate baseline seed strategies for the GA initial population.

    Creates sensible 1-stop and 2-stop templates using available compounds.
    These guarantee the GA starts from reasonable strategies and can only
    improve from there — worst case, the best strategy IS one of these baselines.
    """
    compounds = _available_compounds(weather)
    is_dry = _is_dry_race(weather)
    baselines: list[list[tuple[str, int]]] = []

    if remaining_laps < 2 * MIN_STINT:
        return [[(first_compound, remaining_laps)]]

    other_compounds = [c for c in compounds if c != first_compound]
    if not other_compounds:
        other_compounds = compounds

    # --- 1-stop strategies ---
    for second_compound in other_compounds:
        # Proportional split: allocate laps based on each compound's max stint
        first_capacity = max(MIN_STINT, _max_stint_for(first_compound) - first_stint_age)
        second_capacity = _max_stint_for(second_compound)
        total_capacity = first_capacity + second_capacity
        split = int(remaining_laps * first_capacity / total_capacity)
        split = max(MIN_STINT, min(split, remaining_laps - MIN_STINT))

        baselines.append([
            (first_compound, split),
            (second_compound, remaining_laps - split),
        ])

        # Also try a half-and-half split (different perspective)
        half = remaining_laps // 2
        if abs(half - split) > 3:  # Only add if meaningfully different
            baselines.append([
                (first_compound, half),
                (second_compound, remaining_laps - half),
            ])

    # --- 2-stop strategies (only if enough laps) ---
    if remaining_laps >= 3 * MIN_STINT:
        for second_compound in other_compounds:
            for third_compound in compounds:
                if is_dry:
                    used = {first_compound, second_compound, third_compound} & set(DRY_COMPOUNDS)
                    if len(used) < 2:
                        continue
                # Equal thirds
                third_len = remaining_laps // 3
                first_len = remaining_laps // 3
                second_len = remaining_laps - first_len - third_len
                if first_len >= MIN_STINT and second_len >= MIN_STINT and third_len >= MIN_STINT:
                    baselines.append([
                        (first_compound, first_len),
                        (second_compound, second_len),
                        (third_compound, third_len),
                    ])

    # Repair all baselines to guarantee validity
    repaired = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for strat in baselines:
        fixed = _repair(strat, remaining_laps, first_compound, weather)
        key = tuple(tuple(s) for s in fixed)
        if key not in seen:
            seen.add(key)
            repaired.append(fixed)

    logger.debug(f"Generated {len(repaired)} baseline seed strategies")
    return repaired


# --- Repair function ---

def _repair(individual: list[tuple[str, int]], remaining_laps: int,
            first_compound: str, weather: str) -> list[tuple[str, int]]:
    """Repair an individual to satisfy all hard constraints.

    1. Lock first compound
    2. Enforce min/max stint lengths
    3. Ensure total laps == remaining_laps
    4. Ensure >= 2 different dry compounds (dry races)
    5. No zero-length stints
    """
    if not individual:
        # Emergency: create a minimal valid strategy
        return _emergency_strategy(remaining_laps, first_compound, weather)

    compounds = _available_compounds(weather)
    is_dry = _is_dry_race(weather)

    # 1. Lock first compound
    stints = list(individual)
    stints[0] = (first_compound, stints[0][1])

    # 2. Remove zero-length stints and enforce min/max (per-compound limits)
    cleaned = []
    for compound, length in stints:
        if length < MIN_STINT:
            length = MIN_STINT
        max_for_compound = _max_stint_for(compound)
        if length > max_for_compound:
            length = max_for_compound
        if compound not in compounds and compound not in DRY_COMPOUNDS + WET_COMPOUNDS:
            compound = random.choice(compounds)
        cleaned.append((compound, length))
    stints = cleaned

    # Ensure at least 2 stints
    if len(stints) < 2 and remaining_laps >= 2 * MIN_STINT:
        other = random.choice([c for c in compounds if c != first_compound] or compounds)
        stints.append((other, MIN_STINT))

    # 3. Fix total laps
    total = sum(l for _, l in stints)
    diff = remaining_laps - total

    if diff > 0:
        # Add laps to the last stint
        c, l = stints[-1]
        stints[-1] = (c, l + diff)
    elif diff < 0:
        # Remove laps from the longest stint
        for _ in range(abs(diff)):
            # Find the longest stint that can be shortened
            longest_idx = max(range(len(stints)), key=lambda i: stints[i][1])
            c, l = stints[longest_idx]
            if l > MIN_STINT:
                stints[longest_idx] = (c, l - 1)
            else:
                # Can't shorten further; remove a stint if possible
                if len(stints) > 2:
                    removed = stints.pop(longest_idx)
                    # Redistribute laps
                    if stints:
                        c2, l2 = stints[-1]
                        stints[-1] = (c2, l2 + removed[1] - 1)
                    break

    # Re-enforce per-compound max stint after adjustment
    for i, (c, l) in enumerate(stints):
        max_for_c = _max_stint_for(c)
        if l > max_for_c:
            overflow = l - max_for_c
            stints[i] = (c, max_for_c)
            # Create a new stint with the overflow
            new_compound = random.choice([x for x in compounds if x != c] or compounds)
            stints.insert(i + 1, (new_compound, max(MIN_STINT, overflow)))

    # 4. CRITICAL: Enforce compound diversity for dry races
    if is_dry:
        used_compounds = {c for c, _ in stints}
        dry_used = used_compounds & set(DRY_COMPOUNDS)
        if len(dry_used) < 2 and len(stints) >= 2:
            # Force the second stint to use a different compound
            current_first = stints[0][0]
            alternatives = [c for c in DRY_COMPOUNDS if c != current_first]
            if alternatives:
                new_compound = random.choice(alternatives)
                stints[1] = (new_compound, stints[1][1])

    # 5. Final lap count check
    total = sum(l for _, l in stints)
    if total != remaining_laps:
        diff = remaining_laps - total
        c, l = stints[-1]
        new_l = l + diff
        if new_l < MIN_STINT:
            new_l = MIN_STINT
        stints[-1] = (c, new_l)

    # Final final check
    total = sum(l for _, l in stints)
    if total != remaining_laps:
        return _emergency_strategy(remaining_laps, first_compound, weather)

    return stints


def _emergency_strategy(remaining_laps: int, first_compound: str,
                        weather: str) -> list[tuple[str, int]]:
    """Create a guaranteed-valid minimal strategy."""
    compounds = _available_compounds(weather)
    is_dry = _is_dry_race(weather)

    if remaining_laps < 2 * MIN_STINT:
        return [(first_compound, remaining_laps)]

    first_length = remaining_laps // 2
    second_length = remaining_laps - first_length

    if is_dry:
        other = [c for c in DRY_COMPOUNDS if c != first_compound]
        second_compound = other[0] if other else "MEDIUM"
    else:
        second_compound = random.choice(compounds)

    return [(first_compound, first_length), (second_compound, second_length)]


# --- Genetic operators ---

def mutate_strategy(individual, remaining_laps: int, first_compound: str,
                    weather: str) -> tuple:
    """Mutation operator: randomly modify compound or stint length."""
    if len(individual) < 2:
        return (individual,)

    compounds = _available_compounds(weather)
    mutation_type = random.choice(["compound", "length", "split", "merge"])

    stints = list(individual)

    if mutation_type == "compound":
        # Change a random stint's compound (not the first)
        idx = random.randint(1, len(stints) - 1)
        c, l = stints[idx]
        stints[idx] = (random.choice(compounds), l)

    elif mutation_type == "length":
        # Shift laps between two adjacent stints
        if len(stints) >= 2:
            idx = random.randint(0, len(stints) - 2)
            c1, l1 = stints[idx]
            c2, l2 = stints[idx + 1]
            shift = random.randint(-min(5, l1 - MIN_STINT), min(5, l2 - MIN_STINT))
            if l1 + shift >= MIN_STINT and l2 - shift >= MIN_STINT:
                stints[idx] = (c1, l1 + shift)
                stints[idx + 1] = (c2, l2 - shift)

    elif mutation_type == "split" and len(stints) < MAX_STINTS_COUNT:
        # Split a stint into two
        idx = random.randint(0, len(stints) - 1)
        c, l = stints[idx]
        if l >= 2 * MIN_STINT:
            split_at = random.randint(MIN_STINT, l - MIN_STINT)
            new_compound = random.choice(compounds) if idx > 0 else c
            stints[idx] = (c if idx == 0 else stints[idx][0], split_at)
            stints.insert(idx + 1, (new_compound, l - split_at))

    elif mutation_type == "merge" and len(stints) > 2:
        # Merge two adjacent stints
        idx = random.randint(1, len(stints) - 2) if len(stints) > 2 else 1
        c1, l1 = stints[idx]
        c2, l2 = stints[idx + 1]
        merged_length = l1 + l2
        if merged_length <= _max_stint_for(c1):
            stints[idx] = (c1, merged_length)
            stints.pop(idx + 1)

    # Repair after mutation
    individual[:] = _repair(stints, remaining_laps, first_compound, weather)
    return (individual,)


def crossover_strategy(ind1, ind2, remaining_laps: int, first_compound: str,
                       weather: str) -> tuple:
    """Crossover: take first half of one strategy and second half of another."""
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    # One-point crossover at stint boundaries
    cx1 = random.randint(1, len(ind1) - 1)
    cx2 = random.randint(1, len(ind2) - 1)

    new1 = list(ind1[:cx1]) + list(ind2[cx2:])
    new2 = list(ind2[:cx2]) + list(ind1[cx1:])

    # Repair both children
    ind1[:] = _repair(new1, remaining_laps, first_compound, weather)
    ind2[:] = _repair(new2, remaining_laps, first_compound, weather)

    return ind1, ind2


# --- Fitness evaluation ---

def evaluate_strategy(individual: list[tuple[str, int]], tire_model: TireModel,
                      base_pace: float, current_lap: int, total_laps: int,
                      weather: str, safety_car_laps: set[int] | None = None,
                      first_stint_age: int = 0) -> tuple[float]:
    """Evaluate a strategy's predicted total race time.

    Returns a tuple (total_time,) for DEAP compatibility.
    Illegal strategies (single-compound dry) get an enormous penalty
    so the GA can NEVER select them as optimal.

    first_stint_age: tire age at start of the first stint (0 for fresh tires,
    >0 for used tires already on the car).
    """
    # HARD CONSTRAINT: dry races must use 2+ compounds. Penalize immediately.
    # Only enforce when there are enough laps to physically pit and run 2 stints.
    total_individual_laps = sum(l for _, l in individual)
    if _is_dry_race(weather) and total_individual_laps >= 2 * MIN_STINT:
        used = {c for c, _ in individual}
        dry_used = used & set(DRY_COMPOUNDS)
        if len(dry_used) < 2:
            return (999999.0,)  # Death penalty — makes this individual unselectable

    total_time = 0.0
    lap = current_lap

    for stint_idx, (compound, stint_length) in enumerate(individual):
        # First stint continues on used tires; subsequent stints start fresh
        age_offset = first_stint_age if stint_idx == 0 else 0
        for i in range(stint_length):
            tire_age = age_offset + i
            if safety_car_laps and lap in safety_car_laps:
                total_time += base_pace + 20.0  # SC laps are much slower
            else:
                delta = tire_model.predict(compound, tire_age, lap, total_laps)
                total_time += base_pace + delta
            lap += 1

        # Add pit stop time (except after the last stint)
        if stint_idx < len(individual) - 1:
            total_time += PIT_STOP_LOSS

    return (total_time,)


# --- Main optimizer ---

class StrategyOptimizer:
    """GA-based strategy optimizer."""

    def __init__(self, tire_model: TireModel, population_size: int = 100,
                 generations: int = 50, cx_prob: float = 0.7,
                 mut_prob: float = 0.3):
        self.tire_model = tire_model
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

    def optimize(self, base_pace: float, current_lap: int, total_laps: int,
                 first_compound: str, weather: str = "dry",
                 safety_car_laps: set[int] | None = None,
                 first_stint_age: int = 0) -> Strategy:
        """Run the GA to find the optimal strategy.

        Args:
            base_pace: Estimated base lap time (seconds) for the driver/track.
            current_lap: Current lap number (1-indexed).
            total_laps: Total race laps.
            first_compound: Current tire compound (locked for first stint).
            weather: "dry", "wet", or "mixed".
            safety_car_laps: Set of lap numbers under safety car.
            first_stint_age: Current tire age (laps already done on current set).

        Returns:
            The best Strategy found.
        """
        # remaining_laps includes current lap (off-by-one fix)
        remaining_laps = total_laps - current_lap + 1
        if remaining_laps <= 0:
            return Strategy(stints=[(first_compound, 1)], predicted_total_time=base_pace)

        # Setup DEAP toolbox
        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            lambda: creator.Individual(create_individual(remaining_laps, first_compound, weather))
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register(
            "evaluate",
            evaluate_strategy,
            tire_model=self.tire_model,
            base_pace=base_pace,
            current_lap=current_lap,
            total_laps=total_laps,
            weather=weather,
            safety_car_laps=safety_car_laps,
            first_stint_age=first_stint_age,
        )
        toolbox.register(
            "mate",
            crossover_strategy,
            remaining_laps=remaining_laps,
            first_compound=first_compound,
            weather=weather,
        )
        toolbox.register(
            "mutate",
            mutate_strategy,
            remaining_laps=remaining_laps,
            first_compound=first_compound,
            weather=weather,
        )
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population — seed with baselines first
        baselines = _generate_baseline_strategies(
            remaining_laps, first_compound, weather, first_stint_age
        )

        # Fill remaining slots with random individuals
        n_random = max(0, self.population_size - len(baselines))
        pop = [creator.Individual(b) for b in baselines]
        pop += toolbox.population(n=n_random)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolve
        for gen in range(self.generations):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(lambda x: toolbox.clone(x), offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring with invalidated fitness
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalids))
            for ind, fit in zip(invalids, fitnesses):
                ind.fitness.values = fit

            # Replace population (elitism: keep best from previous gen)
            pop_combined = pop + offspring
            pop = tools.selBest(pop_combined, self.population_size)

        # Get the best individual
        best = tools.selBest(pop, 1)[0]
        best_stints = list(best)

        # POST-EVOLUTION VALIDATION: belt AND suspenders
        best_stints = _repair(best_stints, remaining_laps, first_compound, weather)

        # Final compound diversity check — absolute guarantee
        # Only enforce when there are enough remaining laps for 2 legal stints.
        # Late in the race (< 2*MIN_STINT laps left), a single-compound finish is fine —
        # the driver already used multiple compounds earlier.
        is_dry = _is_dry_race(weather)
        if is_dry and remaining_laps >= 2 * MIN_STINT:
            used = {c for c, _ in best_stints}
            dry_used = used & set(DRY_COMPOUNDS)
            if len(dry_used) < 2:
                logger.warning("Post-evolution: compound diversity violated! Forcing emergency strategy.")
                best_stints = _emergency_strategy(remaining_laps, first_compound, weather)

        # Re-evaluate the final strategy's time (may have been repaired)
        final_time = evaluate_strategy(
            best_stints, self.tire_model, base_pace,
            current_lap, total_laps, weather, safety_car_laps,
            first_stint_age=first_stint_age,
        )[0]

        strategy = Strategy(
            stints=best_stints,
            predicted_total_time=final_time,
        )

        pt = strategy.predicted_total_time
        pt_m, pt_s = int(pt // 60), pt % 60
        logger.info(
            f"Strategy optimized: {strategy.stints} "
            f"(predicted time: {pt_m}:{pt_s:04.1f}, "
            f"compounds: {strategy.compounds_used})"
        )

        return strategy


# Module-level singleton tire model for the strategist — avoids reloading
# the .pt file from disk on every lap.
_strategist_tire_model: TireModel | None = None

# Module-level tire availability context — set by run_strategist() each lap,
# read by _available_compounds() to filter compounds the driver actually has.
_tire_availability: dict[str, int] | None = None


def _get_tire_model() -> TireModel:
    global _strategist_tire_model
    if _strategist_tire_model is None:
        _strategist_tire_model = TireModel()
        _strategist_tire_model.load()
    return _strategist_tire_model


def _should_reoptimize(state: RaceState, current_lap: int, total_laps: int) -> bool:
    """Decide whether to re-run the GA optimizer this lap.

    Running the stochastic GA every lap causes strategy oscillation.
    Only re-optimize on meaningful triggers; otherwise tick the existing strategy.
    """
    if state.get("strategy") is None:
        return True  # First optimization
    if current_lap <= 3:
        return True  # Early race — still calibrating
    if current_lap % 5 == 0:
        return True  # Periodic review
    if state.get("safety_car", False) or state.get("virtual_safety_car", False):
        return True  # SC/VSC — potential cheap pit
    remaining = total_laps - current_lap + 1
    if remaining <= 10:
        return True  # End game — frequent re-evaluation
    # New stint started (driver just pitted)
    stint_data = state.get("stint_data", [])
    if stint_data and stint_data[-1].lap_start == current_lap:
        return True
    return False


def _tick_strategy(strategy: Strategy, remaining_laps: int) -> Strategy:
    """Advance existing strategy by 1 lap (reduce first stint by 1).

    Used on non-optimization laps to maintain stability.
    """
    if not strategy or not strategy.stints:
        return strategy
    stints = list(strategy.stints)
    c, l = stints[0]
    if l > 1:
        stints[0] = (c, l - 1)
    # Adjust last stint to ensure total matches remaining laps
    total = sum(length for _, length in stints)
    if total != remaining_laps:
        c_last, l_last = stints[-1]
        adjusted = l_last + (remaining_laps - total)
        if adjusted > 0:
            stints[-1] = (c_last, adjusted)
    return Strategy(stints=stints, predicted_total_time=strategy.predicted_total_time)


def run_strategist(state: RaceState) -> dict:
    """LangGraph node: run the strategy optimizer."""
    global _tire_availability

    tire_model = _get_tire_model()

    current_lap = state.get("current_lap", 1)
    total_laps = state.get("total_laps", 57)
    weather = state.get("weather", "dry")

    # Set tire availability context for this lap — _available_compounds() reads this
    _tire_availability = state.get("available_compounds")

    # Determine current compound and tire age from stint data
    stint_data = state.get("stint_data", [])
    if stint_data:
        current_stint = stint_data[-1]
        first_compound = current_stint.compound
        first_stint_age = current_stint.tyre_age_at_start + (current_lap - current_stint.lap_start)
    else:
        first_compound = "MEDIUM"
        first_stint_age = 0

    # --- Stability: only re-optimize on meaningful triggers ---
    remaining_laps = total_laps - current_lap + 1
    if not _should_reoptimize(state, current_lap, total_laps):
        old_strategy = state.get("strategy")
        ticked = _tick_strategy(old_strategy, remaining_laps)
        return {"strategy": ticked, "strategy_changed": False}

    # Estimate base pace: exclude lap 1 (standing start) and pit laps
    lap_data = state.get("lap_data", [])
    valid_times = [
        ld.lap_duration for ld in lap_data
        if ld.lap_duration and ld.lap_number > 1
        and not ld.is_pit_out_lap and not ld.is_pit_in_lap
        and 30 < ld.lap_duration < 200
    ]
    if valid_times:
        base_pace = min(valid_times)  # best clean lap as proxy
    else:
        base_pace = 90.0  # fallback

    # Collect safety car laps — track the full SC/VSC period, not just the
    # deployment message.  RC messages mark start ("SAFETY CAR DEPLOYED") and
    # end ("GREEN FLAG" / "CLEAR") so we must track the range.
    sc_laps: set[int] = set()
    _sc_on = False
    for msg in sorted(state.get("race_control", []), key=lambda m: m.lap_number):
        text = msg.message.lower() if msg.message else ""
        cat = (msg.category or "").lower()
        if "safety car" in cat or "safety car" in text or "safetycar" in cat:
            _sc_on = True
        if "green" in text or "clear" in text or "restart" in text:
            _sc_on = False
        if _sc_on:
            sc_laps.add(msg.lap_number)

    # Detect CURRENT SC/VSC status
    sc_active = state.get("safety_car", False)
    vsc_active = state.get("virtual_safety_car", False)

    optimizer = StrategyOptimizer(tire_model)
    new_strategy = optimizer.optimize(
        base_pace=base_pace,
        current_lap=current_lap,
        total_laps=total_laps,
        first_compound=first_compound,
        weather=weather,
        safety_car_laps=sc_laps if sc_laps else None,
        first_stint_age=first_stint_age,
    )

    # --- SC/VSC OPPORTUNISTIC PITTING ---
    # Under SC pit stop costs ~12s (saves 10s), under VSC ~17s (saves 5s).
    # If we're in a pit window (first stint done enough, enough laps left),
    # evaluate "pit THIS lap" vs the GA plan. Cheaper pit = free time gain.
    remaining_laps = total_laps - current_lap + 1
    if (sc_active or vsc_active) and remaining_laps > MIN_STINT:
        # Only consider if the current stint has run long enough
        first_stint_laps = new_strategy.stints[0][1] if new_strategy.stints else 0
        already_on_stint = first_stint_age  # laps already done on current tires

        # Pit window: already done at least MIN_STINT laps on these tires,
        # OR we're within 10 laps of the GA's planned pit stop
        in_window = already_on_stint >= MIN_STINT or first_stint_laps <= 10

        if in_window and len(new_strategy.stints) >= 1:
            # Build "pit now" alternative: 1 lap on current tires, then fresh
            reduced_loss = PIT_STOP_LOSS_SC if sc_active else PIT_STOP_LOSS_VSC
            saving = PIT_STOP_LOSS - reduced_loss

            # Pick the best compound for the remaining stint
            pit_now_remaining = remaining_laps - 1  # 1 lap to pit, rest on new tires
            if pit_now_remaining > 0:
                # Try each available compound for the second stint
                avail = _available_compounds(weather)
                best_pit_now_time = float("inf")
                best_pit_now_compound = None

                for comp in avail:
                    if comp == first_compound and _is_dry_race(weather):
                        # Need to use a different compound (regulation)
                        # Check if we've already used another compound
                        used_compounds = {s.compound for s in stint_data}
                        if len(used_compounds & set(DRY_COMPOUNDS)) < 2:
                            continue  # can't pit to same compound if haven't fulfilled rule
                    pit_now_stints = [(first_compound, 1), (comp, pit_now_remaining)]
                    t = evaluate_strategy(
                        pit_now_stints, tire_model, base_pace, current_lap,
                        total_laps, weather, safety_car_laps=sc_laps if sc_laps else None,
                        first_stint_age=first_stint_age,
                    )[0]
                    # Adjust: replace normal pit loss with reduced SC/VSC loss
                    t = t - PIT_STOP_LOSS + reduced_loss
                    if t < best_pit_now_time:
                        best_pit_now_time = t
                        best_pit_now_compound = comp

                if best_pit_now_compound and best_pit_now_time < new_strategy.predicted_total_time:
                    logger.info(
                        f"{'SC' if sc_active else 'VSC'} PIT OPPORTUNITY: "
                        f"pit now for {best_pit_now_compound} saves "
                        f"{new_strategy.predicted_total_time - best_pit_now_time:.1f}s "
                        f"(reduced pit loss: {reduced_loss:.0f}s vs {PIT_STOP_LOSS:.0f}s)"
                    )
                    new_strategy = Strategy(
                        stints=[(first_compound, 1), (best_pit_now_compound, pit_now_remaining)],
                        predicted_total_time=best_pit_now_time,
                    )

    # Check if strategy MEANINGFULLY changed (compound order, not just stint lengths)
    # A strategy going from [SOFT-20, MEDIUM-30] to [SOFT-19, MEDIUM-30]
    # is just a lap-tick adjustment — not a real change worth briefing about.
    old_strategy = state.get("strategy")
    if old_strategy is None:
        changed = True
    else:
        old_compounds = [c for c, _ in old_strategy.stints]
        new_compounds = [c for c, _ in new_strategy.stints]
        changed = old_compounds != new_compounds

    return {"strategy": new_strategy, "strategy_changed": changed}
