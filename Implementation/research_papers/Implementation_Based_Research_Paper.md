# Intelligent Airport Gate Assignment Using Machine Learning and Multi-Objective Optimization

**Author:** R. Balaji  
**Project Type:** Final Year Project Prototype  
**Implementation Basis:** `Implementation/` repository artifacts and exported results

## Abstract

Airport gate assignment affects turnaround coordination, passenger movement, and airside efficiency. This paper documents an implementation-driven prototype for the Airport Gate Assignment Problem (AGAP) built from a synthetic airport operations pipeline, a delay-prediction model selection layer, and a constrained optimization engine. The system generates realistic flight and gate records, trains multiple regression models to estimate delay, and uses the selected model inside a genetic-algorithm-based search to optimize gate allocation under gate-type, gate-size, and time-overlap constraints. The implemented prototype is exposed through both a command-line runner and a Streamlit dashboard. In the exported experiment with 30 flights, 8 gates, and random seed 42, the best predictive model was Random Forest with RMSE 8.855 and MAE 7.993. Compared with the random baseline assignment used in evaluation, the optimized assignment reduced CO2 proxy by 5.628 percent and reduced average predicted delay by 0.829 percent, while increasing average passenger walking distance by 9.760 percent. The study demonstrates a transparent academic prototype for combining prediction and optimization in airport operations research.

**Keywords:** Airport Gate Assignment Problem, AGAP, machine learning, genetic algorithm, Streamlit, multi-objective optimization

## 1. Introduction

Airport gate assignment is a constrained decision problem in which flights must be matched to limited gate resources while respecting aircraft compatibility, terminal service requirements, and schedule conflicts. Poor assignments can increase taxiing effort, passenger walking burden, and disruption propagation. Because these objectives often conflict, AGAP is commonly treated as a multi-objective optimization problem.

This project implements a compact research prototype that connects prediction and optimization in one pipeline. Instead of optimizing purely static cost terms, the system estimates delay behavior from generated operational features and uses those predictions as part of the gate-assignment fitness function. The result is an explainable end-to-end framework suitable for final-year project demonstration and academic reporting.

## 2. Problem Statement

The implemented system solves a constrained AGAP variant in which each flight must be assigned to exactly one gate. A feasible assignment must satisfy the following rules:

1. A flight can only use a gate whose size can accommodate the aircraft type.
2. A flight should use a gate whose operational type matches the flight type, such as domestic or international.
3. Two flights with overlapping arrival-departure windows cannot occupy the same gate.
4. The system must return a complete assignment even when the search procedure fails to improve further.

The optimization objective combines four lower-is-better terms:

- passenger walking cost,
- taxiing cost,
- CO2 proxy cost, and
- predicted delay penalty.

The optimizer also applies a penalty to infeasible chromosomes during search.

## 3. Repository-Driven System Design

The implementation is organized into modular components:

- `src/data.py` generates synthetic gates and flights.
- `src/model.py` builds features, trains candidate regressors, compares RMSE and MAE, and extracts feature importance.
- `src/optimize.py` defines compatibility checks, the fitness function, a genetic-algorithm-style search process, and a first-fit fallback.
- `src/evaluate.py` computes assignment metrics and baseline-versus-optimized comparisons.
- `src/pipeline.py` orchestrates one complete experiment and exports CSV outputs.
- `app.py` exposes the workflow through a Streamlit dashboard.
- `run_pipeline.py` supports command-line execution.

This design keeps the project reproducible and aligned with academic demonstration needs.

## 4. Methodology

### 4.1 Synthetic Data Generation

The project generates a mixed airport scenario using configurable numbers of flights and gates. The flight records include:

- `flight_id`
- `arrival_time`
- `departure_time`
- `aircraft_type`
- `flight_type`
- `expected_passengers`
- `historical_delay_minutes`
- `taxi_distance_to_gate`
- `taxi_distance_to_runway`
- `fuel_burn_rate`
- `assigned_gate`

The gate records include:

- `gate_id`
- `gate_type`
- `gate_size`
- `distance_from_runway`
- `distance_from_terminal_center`

Flight generation introduces clustered arrivals and realistic turnaround durations so that gate conflicts emerge naturally. Delay values are synthesized from passenger demand, taxi distance, flight type, and random noise.

### 4.2 Delay Prediction Layer

The learning target is `historical_delay_minutes`. The implementation constructs a feature matrix using:

- aircraft type,
- flight type,
- arrival hour,
- expected passengers,
- taxi distance feature, and
- fuel burn rate.

The candidate models are:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor, when the dependency is available

The system performs train-test splitting, evaluates RMSE and MAE, and keeps the best model for downstream optimization.

### 4.3 Optimization Layer

The optimizer encodes one chromosome as a complete gate assignment across all flights. A chromosome is scored using normalized objective terms with the following implemented weights:

- walking cost: 0.35
- taxi cost: 0.20
- CO2 proxy cost: 0.40
- predicted delay cost: 0.05

The search loop includes:

1. initialization from one heuristic assignment plus random compatible assignments,
2. fitness evaluation,
3. elite retention,
4. one-point crossover,
5. mutation using compatible gate lists, and
6. early stopping after stagnation.

If the best final chromosome is infeasible, the implementation falls back to a first-fit assignment. This design guarantees a usable output for the dashboard and exported analysis.

## 5. Experimental Setup

The current exported results correspond to the default configuration used by the repository pipeline:

- number of flights: 30
- number of gates: 8
- random seed: 42

The evaluation baseline in `src/pipeline.py` is a random gate assignment, while the optimizer internally uses a feasible first-fit assignment as its initial reference chromosome.

## 6. Results

### 6.1 Model Performance

| Model | RMSE | MAE |
| --- | ---: | ---: |
| RandomForest | 8.855 | 7.993 |
| LinearRegression | 10.355 | 9.076 |

Random Forest achieved the best error profile and was therefore selected for optimization-time delay prediction.

### 6.2 Top Feature Importance from the Selected Model

| Feature | Importance |
| --- | ---: |
| `numeric__fuel_burn_rate` | 0.3775 |
| `numeric__arrival_hour` | 0.3623 |
| `numeric__expected_passengers` | 0.1829 |
| `numeric__taxi_distance_feature` | 0.0650 |
| `categorical__aircraft_type_Small` | 0.0054 |
| `categorical__flight_type_Domestic` | 0.0036 |

The model is dominated by operational load and timing features rather than categorical flight labels.

### 6.3 Assignment Comparison

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| `total_co2_proxy` | 2312285.060 | 2182149.026 | 5.628% improvement |
| `avg_passenger_walking_distance` | 438.094 | 480.852 | -9.760% |
| `avg_predicted_delay_minutes` | 17.185 | 17.042 | 0.829% improvement |
| `gate_utilization_percent` | 100.000 | 100.000 | 0.000% |

The optimized assignment reduced the environmental proxy and slightly reduced predicted delay. The penalty is a higher average walking distance, which is consistent with the weighted multi-objective design. Gate utilization remained unchanged because the scenario already used all available gates.

## 7. Discussion

Three implementation-level observations are important.

First, the prototype succeeds as an integrated pipeline: data generation, model selection, optimization, evaluation, dashboard presentation, and CSV export all work through a single experiment runner.

Second, the results show a realistic multi-objective trade-off rather than a universal improvement across every metric. This is desirable in an academic prototype because it makes the weighting strategy explicit and interpretable.

Third, the current evaluation is bounded by its synthetic scenario design. The reported gains should therefore be treated as controlled proof-of-concept evidence rather than as deployment-ready airport performance estimates.

## 8. Limitations

The current implementation has several research limitations:

1. The dataset is synthetic and does not capture full airport operational variability.
2. The environmental metric is a comparative CO2 proxy, not a validated emissions inventory.
3. The evaluation baseline is random, so stronger baselines could narrow the apparent gains.
4. The experiments are based on single exported runs rather than repeated trials with confidence intervals.
5. No exact optimization benchmark, such as MILP, is included for comparison.

## 9. Conclusion and Future Work

This implementation demonstrates that a delay-aware gate assignment workflow can be built as a coherent academic prototype using synthetic data, regression-based delay prediction, and multi-objective evolutionary search. In the current exported run, the system improved CO2 proxy and predicted delay relative to the evaluation baseline, while exposing a measurable walking-distance trade-off. That outcome supports the usefulness of combined prediction and optimization for AGAP research at student-project scale.

Future work should focus on real airport datasets, repeated multi-seed experiments, stronger baselines, exact-solver comparisons, and richer objective terms such as passenger connections, fairness across gates, and disruption recovery.

## 10. References

1. `README.md`
2. `docs/methodology.md`
3. `src/data.py`
4. `src/model.py`
5. `src/optimize.py`
6. `src/evaluate.py`
7. `src/pipeline.py`
8. `app.py`
9. `outputs/model_metrics.csv`
10. `outputs/feature_importance.csv`
11. `outputs/comparison.csv`
