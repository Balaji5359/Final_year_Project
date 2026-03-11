from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import SIZE_ORDER
from .model import predict_delays_for_assignment


@dataclass
class OptimizationResult:
    assignment: pd.DataFrame
    score: float
    used_fallback: bool


def _overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    return a_start < b_end and b_start < a_end


def _compatible_gates_for_flight(flight_row: pd.Series, gates: pd.DataFrame) -> List[str]:
    match = gates[
        (gates["gate_type"] == flight_row["flight_type"])
        & (gates["gate_size"].map(SIZE_ORDER) >= SIZE_ORDER[flight_row["aircraft_type"]])
    ]
    if not match.empty:
        return match["gate_id"].tolist()
    looser = gates[gates["gate_size"].map(SIZE_ORDER) >= SIZE_ORDER[flight_row["aircraft_type"]]]
    if not looser.empty:
        return looser["gate_id"].tolist()
    return gates["gate_id"].tolist()


def _build_gate_lookup(gates: pd.DataFrame) -> Dict[str, Dict]:
    return gates.set_index("gate_id").to_dict(orient="index")


def _is_feasible(flights: pd.DataFrame, candidate_gates: Sequence[str], gates: pd.DataFrame) -> bool:
    gate_slots: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {g: [] for g in gates["gate_id"]}
    for i, gate_id in enumerate(candidate_gates):
        f = flights.iloc[i]
        g = gates[gates["gate_id"] == gate_id].iloc[0]
        if SIZE_ORDER[g["gate_size"]] < SIZE_ORDER[f["aircraft_type"]]:
            return False
        if g["gate_type"] != f["flight_type"]:
            return False
        arr = pd.to_datetime(f["arrival_time"])
        dep = pd.to_datetime(f["departure_time"])
        for occ_arr, occ_dep in gate_slots[gate_id]:
            if _overlap(arr, dep, occ_arr, occ_dep):
                return False
        gate_slots[gate_id].append((arr, dep))
    return True


def _first_fit_assignment(flights: pd.DataFrame, gates: pd.DataFrame) -> List[str]:
    gate_slots: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {g: [] for g in gates["gate_id"]}
    assigned: List[str] = []

    for _, f in flights.iterrows():
        compatible = _compatible_gates_for_flight(f, gates)
        arr = pd.to_datetime(f["arrival_time"])
        dep = pd.to_datetime(f["departure_time"])
        chosen = None
        for gate_id in compatible:
            overlaps = any(_overlap(arr, dep, occ_a, occ_d) for occ_a, occ_d in gate_slots[gate_id])
            if not overlaps:
                chosen = gate_id
                break
        if chosen is None:
            chosen = compatible[0]
        gate_slots[chosen].append((arr, dep))
        assigned.append(chosen)
    return assigned


def _compute_objectives(
    flights: pd.DataFrame, gates: pd.DataFrame, candidate_gates: Sequence[str], model
) -> Dict[str, float]:
    assigned = flights.copy()
    assigned["assigned_gate"] = list(candidate_gates)

    gate_lookup = _build_gate_lookup(gates)
    gate_terminal_distance = np.array([gate_lookup[g]["distance_from_terminal_center"] for g in candidate_gates], dtype=float)
    gate_runway_distance = np.array([gate_lookup[g]["distance_from_runway"] for g in candidate_gates], dtype=float)

    passengers = assigned["expected_passengers"].to_numpy(dtype=float)
    taxi_base = assigned["taxi_distance_to_runway"].to_numpy(dtype=float)
    fuel = assigned["fuel_burn_rate"].to_numpy(dtype=float)

    walking_total = float(np.sum(passengers * gate_terminal_distance))
    taxi_total = float(np.sum(taxi_base + gate_runway_distance))
    predicted_delay = predict_delays_for_assignment(model, assigned, gates=gates)
    delay_mean = float(np.mean(predicted_delay))
    co2_proxy = float(np.sum(fuel * (taxi_base + gate_runway_distance) * (1.0 + predicted_delay / 60.0)))

    return {
        "walking_total": walking_total,
        "taxi_total": taxi_total,
        "co2_total": co2_proxy,
        "pred_delay_avg": delay_mean,
    }


def _fitness(
    objectives: Dict[str, float],
    refs: Dict[str, float],
    weights: Dict[str, float],
    feasible: bool,
) -> float:
    penalty = 3.5 if not feasible else 0.0
    normalized = {
        k: objectives[k] / max(refs[k], 1e-9)
        for k in ["walking_total", "taxi_total", "co2_total", "pred_delay_avg"]
    }
    score = (
        weights["walking_total"] * normalized["walking_total"]
        + weights["taxi_total"] * normalized["taxi_total"]
        + weights["co2_total"] * normalized["co2_total"]
        + weights["pred_delay_avg"] * normalized["pred_delay_avg"]
        + penalty
    )
    return float(score)


def optimize_gate_assignment(
    flights: pd.DataFrame,
    gates: pd.DataFrame,
    model,
    generations: int = 28,
    population_size: int = 24,
    mutation_rate: float = 0.12,
    random_seed: int = 42,
) -> OptimizationResult:
    rng = np.random.default_rng(random_seed)
    n = len(flights)

    compatible_lists = [_compatible_gates_for_flight(flights.iloc[i], gates) for i in range(n)]
    baseline = _first_fit_assignment(flights, gates)
    refs = _compute_objectives(flights, gates, baseline, model)
    weights = {"walking_total": 0.35, "taxi_total": 0.2, "co2_total": 0.4, "pred_delay_avg": 0.05}

    def random_chromosome() -> List[str]:
        return [str(rng.choice(compatible_lists[i])) for i in range(n)]

    def crossover(a: List[str], b: List[str]) -> List[str]:
        point = int(rng.integers(1, n)) if n > 1 else 0
        return a[:point] + b[point:]

    def mutate(chrom: List[str]) -> List[str]:
        out = chrom[:]
        for i in range(n):
            if rng.random() < mutation_rate:
                out[i] = str(rng.choice(compatible_lists[i]))
        return out

    population: List[List[str]] = [baseline] + [random_chromosome() for _ in range(max(1, population_size - 1))]
    best = baseline
    best_score = _fitness(
        objectives=_compute_objectives(flights, gates, baseline, model),
        refs=refs,
        weights=weights,
        feasible=_is_feasible(flights, baseline, gates),
    )

    no_improve_rounds = 0
    for _ in range(generations):
        scored: List[Tuple[float, List[str]]] = []
        for chrom in population:
            feasible = _is_feasible(flights, chrom, gates)
            objectives = _compute_objectives(flights, gates, chrom, model)
            score = _fitness(objectives=objectives, refs=refs, weights=weights, feasible=feasible)
            scored.append((score, chrom))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_score:
            best_score = scored[0][0]
            best = scored[0][1]
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        elites = [chrom for _, chrom in scored[: max(2, population_size // 5)]]
        next_population = elites[:]
        while len(next_population) < population_size:
            p1 = elites[int(rng.integers(0, len(elites)))]
            p2 = elites[int(rng.integers(0, len(elites)))]
            child = mutate(crossover(p1, p2))
            next_population.append(child)
        population = next_population

        # Keep runtime bounded for dashboard interaction.
        if no_improve_rounds >= 7:
            break

    if not _is_feasible(flights, best, gates):
        best = _first_fit_assignment(flights, gates)
        used_fallback = True
    else:
        used_fallback = False

    out = flights.copy()
    out["optimized_gate"] = best
    return OptimizationResult(assignment=out, score=float(best_score), used_fallback=used_fallback)


def build_baseline_assignment(
    flights: pd.DataFrame, gates: pd.DataFrame, strategy: str = "random", random_seed: int = 42
) -> pd.DataFrame:
    out = flights.copy()
    if strategy == "first_fit":
        out["baseline_gate"] = _first_fit_assignment(flights, gates)
        return out

    rng = np.random.default_rng(random_seed)
    all_gates = gates["gate_id"].tolist()
    random_assignments: List[str] = []
    for _, f in flights.iterrows():
        _ = f
        random_assignments.append(str(rng.choice(all_gates)))
    out["baseline_gate"] = random_assignments
    return out
