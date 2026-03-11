from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from .data import DataConfig, generate_synthetic_data
from .evaluate import compare_metrics, compute_assignment_metrics
from .model import ModelArtifacts, train_and_select_model
from .optimize import build_baseline_assignment, optimize_gate_assignment


@dataclass
class RunArtifacts:
    flights: pd.DataFrame
    gates: pd.DataFrame
    model: ModelArtifacts
    baseline_assignment: pd.DataFrame
    optimized_assignment: pd.DataFrame
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    comparison: pd.DataFrame
    used_fallback: bool


def run_experiment(
    num_flights: int = 30,
    num_gates: int = 8,
    random_seed: int = 42,
    save_outputs: bool = True,
    output_dir: str = "outputs",
) -> RunArtifacts:
    config = DataConfig(num_flights=num_flights, num_gates=num_gates, random_seed=random_seed)
    flights, gates = generate_synthetic_data(config)
    model_artifacts = train_and_select_model(flights=flights, gates=gates)

    baseline_assignment = build_baseline_assignment(
        flights=flights, gates=gates, strategy="random", random_seed=random_seed
    )
    opt = optimize_gate_assignment(flights=flights, gates=gates, model=model_artifacts.model, random_seed=random_seed)
    optimized_assignment = opt.assignment.copy()

    baseline_metrics = compute_assignment_metrics(
        flights_with_gate=baseline_assignment.rename(columns={"baseline_gate": "gate"}),
        gate_col="gate",
        gates=gates,
        model=model_artifacts.model,
    )
    optimized_metrics = compute_assignment_metrics(
        flights_with_gate=optimized_assignment.rename(columns={"optimized_gate": "gate"}),
        gate_col="gate",
        gates=gates,
        model=model_artifacts.model,
    )

    comparison = compare_metrics(baseline=baseline_metrics, optimized=optimized_metrics)

    if save_outputs:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        flights.to_csv(out / "flights.csv", index=False)
        gates.to_csv(out / "gates.csv", index=False)
        baseline_assignment.to_csv(out / "baseline_assignment.csv", index=False)
        optimized_assignment.to_csv(out / "optimized_assignment.csv", index=False)
        model_artifacts.metrics.to_csv(out / "model_metrics.csv", index=False)
        model_artifacts.feature_importance.to_csv(out / "feature_importance.csv", index=False)
        comparison.to_csv(out / "comparison.csv", index=False)

    return RunArtifacts(
        flights=flights,
        gates=gates,
        model=model_artifacts,
        baseline_assignment=baseline_assignment,
        optimized_assignment=optimized_assignment,
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        comparison=comparison,
        used_fallback=opt.used_fallback,
    )
