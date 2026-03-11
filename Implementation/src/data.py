from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


AIRCRAFT_TYPES = ["Small", "Medium", "Large"]
FLIGHT_TYPES = ["Domestic", "International"]
SIZE_ORDER = {"Small": 1, "Medium": 2, "Large": 3}


@dataclass
class DataConfig:
    num_flights: int = 30
    num_gates: int = 8
    random_seed: int = 42
    start_hour: int = 6


def _gate_size_pool(num_gates: int) -> List[str]:
    # Keep a practical distribution for mixed airport operations.
    counts = {
        "Small": max(1, int(num_gates * 0.3)),
        "Medium": max(2, int(num_gates * 0.45)),
        "Large": max(1, num_gates - max(1, int(num_gates * 0.3)) - max(2, int(num_gates * 0.45))),
    }
    out = ["Small"] * counts["Small"] + ["Medium"] * counts["Medium"] + ["Large"] * counts["Large"]
    return out[:num_gates]


def generate_gates(num_gates: int = 8, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    sizes = _gate_size_pool(num_gates)
    rng.shuffle(sizes)

    rows: List[Dict] = []
    for i in range(num_gates):
        gate_id = f"G{i + 1:02d}"
        gate_size = sizes[i]
        gate_type = rng.choice(FLIGHT_TYPES, p=[0.65, 0.35])
        distance_from_runway = float(rng.uniform(350, 1800))
        distance_from_terminal_center = float(rng.uniform(80, 750))
        rows.append(
            {
                "gate_id": gate_id,
                "gate_type": gate_type,
                "gate_size": gate_size,
                "distance_from_runway": round(distance_from_runway, 2),
                "distance_from_terminal_center": round(distance_from_terminal_center, 2),
            }
        )

    gates = pd.DataFrame(rows)
    return gates


def _sample_aircraft(rng: np.random.Generator) -> str:
    return str(rng.choice(AIRCRAFT_TYPES, p=[0.35, 0.45, 0.2]))


def _sample_flight_type(rng: np.random.Generator) -> str:
    return str(rng.choice(FLIGHT_TYPES, p=[0.7, 0.3]))


def _passenger_count(aircraft_type: str, rng: np.random.Generator) -> int:
    if aircraft_type == "Small":
        return int(rng.integers(60, 130))
    if aircraft_type == "Medium":
        return int(rng.integers(120, 230))
    return int(rng.integers(220, 360))


def _fuel_burn_rate(aircraft_type: str, rng: np.random.Generator) -> float:
    base = {"Small": 18.0, "Medium": 28.0, "Large": 43.0}[aircraft_type]
    return float(round(base + rng.normal(0, 2.5), 2))


def _find_compatible_gates(
    gates: pd.DataFrame, aircraft_type: str, flight_type: str
) -> pd.DataFrame:
    return gates[
        (gates["gate_type"] == flight_type)
        & (gates["gate_size"].map(SIZE_ORDER) >= SIZE_ORDER[aircraft_type])
    ]


def _time_overlap(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a < end_b and start_b < end_a


def _assign_baseline_gate(
    arrival: datetime,
    departure: datetime,
    aircraft_type: str,
    flight_type: str,
    gates: pd.DataFrame,
    occupancy: Dict[str, List[Tuple[datetime, datetime]]],
    rng: np.random.Generator,
) -> str:
    compatible = _find_compatible_gates(gates, aircraft_type, flight_type)
    if compatible.empty:
        compatible = gates[gates["gate_size"].map(SIZE_ORDER) >= SIZE_ORDER[aircraft_type]]
    if compatible.empty:
        compatible = gates

    for gate_id in compatible["gate_id"]:
        overlaps = any(
            _time_overlap(arrival, departure, occupied_arrival, occupied_departure)
            for occupied_arrival, occupied_departure in occupancy[gate_id]
        )
        if not overlaps:
            occupancy[gate_id].append((arrival, departure))
            return str(gate_id)

    # Fall back to random compatible gate if all overlap. This keeps some imperfect baseline behavior.
    gate_id = str(rng.choice(compatible["gate_id"].tolist()))
    occupancy[gate_id].append((arrival, departure))
    return gate_id


def generate_flights(
    gates: pd.DataFrame, num_flights: int = 30, random_seed: int = 42, start_hour: int = 6
) -> pd.DataFrame:
    if num_flights < 20 or num_flights > 60:
        raise ValueError("num_flights must be between 20 and 60 for this project setup.")

    rng = np.random.default_rng(random_seed)
    base_date = datetime(2025, 1, 1, start_hour, 0, 0)
    occupancy: Dict[str, List[Tuple[datetime, datetime]]] = {g: [] for g in gates["gate_id"]}
    rows: List[Dict] = []

    for i in range(num_flights):
        flight_id = f"FL{i + 1:03d}"
        aircraft_type = _sample_aircraft(rng)
        flight_type = _sample_flight_type(rng)

        # Keep clustered arrivals so overlaps naturally occur.
        arrival_offset = int(rng.integers(0, 18 * 60))
        arrival = base_date + timedelta(minutes=arrival_offset)
        turnaround = int(rng.integers(45, 185))
        departure = arrival + timedelta(minutes=turnaround)

        expected_passengers = _passenger_count(aircraft_type, rng)
        taxi_distance_to_gate = float(round(rng.uniform(300, 1700), 2))
        taxi_distance_to_runway = float(round(rng.uniform(300, 1800), 2))
        fuel_burn_rate = _fuel_burn_rate(aircraft_type, rng)

        historical_delay_minutes = max(
            0,
            int(
                0.02 * expected_passengers
                + 0.007 * taxi_distance_to_gate
                + 0.005 * taxi_distance_to_runway
                + (4 if flight_type == "International" else 0)
                + rng.normal(0, 7)
            ),
        )

        assigned_gate = _assign_baseline_gate(
            arrival=arrival,
            departure=departure,
            aircraft_type=aircraft_type,
            flight_type=flight_type,
            gates=gates,
            occupancy=occupancy,
            rng=rng,
        )

        rows.append(
            {
                "flight_id": flight_id,
                "arrival_time": arrival.strftime("%Y-%m-%d %H:%M"),
                "departure_time": departure.strftime("%Y-%m-%d %H:%M"),
                "aircraft_type": aircraft_type,
                "flight_type": flight_type,
                "expected_passengers": expected_passengers,
                "historical_delay_minutes": historical_delay_minutes,
                "taxi_distance_to_gate": taxi_distance_to_gate,
                "taxi_distance_to_runway": taxi_distance_to_runway,
                "fuel_burn_rate": fuel_burn_rate,
                "assigned_gate": assigned_gate,
            }
        )

    flights = pd.DataFrame(rows).sort_values("arrival_time").reset_index(drop=True)
    return flights


def generate_synthetic_data(config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gates = generate_gates(num_gates=config.num_gates, random_seed=config.random_seed)
    flights = generate_flights(
        gates=gates,
        num_flights=config.num_flights,
        random_seed=config.random_seed,
        start_hour=config.start_hour,
    )
    return flights, gates

