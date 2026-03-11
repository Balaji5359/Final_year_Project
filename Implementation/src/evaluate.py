from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .model import predict_delays_for_assignment


def _utilization_percent(assignment: pd.DataFrame, gate_col: str, total_gates: int) -> float:
    used = assignment[gate_col].nunique()
    return float(100.0 * used / max(total_gates, 1))


def compute_assignment_metrics(
    flights_with_gate: pd.DataFrame, gate_col: str, gates: pd.DataFrame, model
) -> Dict[str, float]:
    data = flights_with_gate.copy()
    data["assigned_gate"] = data[gate_col]
    merged = data.merge(
        gates[["gate_id", "distance_from_runway", "distance_from_terminal_center"]],
        left_on=gate_col,
        right_on="gate_id",
        how="left",
    )
    taxi = merged["taxi_distance_to_runway"] + merged["distance_from_runway"]
    predicted_delay = predict_delays_for_assignment(model, data, gates=gates, assigned_gate_col="assigned_gate")
    co2_total = np.sum(merged["fuel_burn_rate"] * taxi * (1.0 + predicted_delay / 60.0))
    avg_walk = np.mean(merged["distance_from_terminal_center"])
    avg_delay = float(np.mean(predicted_delay))
    util = _utilization_percent(data, gate_col=gate_col, total_gates=len(gates))

    return {
        "total_co2_proxy": float(co2_total),
        "avg_passenger_walking_distance": float(avg_walk),
        "avg_predicted_delay_minutes": avg_delay,
        "gate_utilization_percent": util,
    }


def compare_metrics(baseline: Dict[str, float], optimized: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, baseline_value in baseline.items():
        optimized_value = optimized[metric]
        if baseline_value == 0:
            improvement_pct = 0.0
        else:
            # Lower-is-better metrics; utilization is shown as change, not "improvement".
            if metric == "gate_utilization_percent":
                improvement_pct = 100.0 * (optimized_value - baseline_value) / abs(baseline_value)
            else:
                improvement_pct = 100.0 * (baseline_value - optimized_value) / abs(baseline_value)
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "optimized": optimized_value,
                "change_percent": improvement_pct,
            }
        )
    return pd.DataFrame(rows)

