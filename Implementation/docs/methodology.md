# Methodology (Research-Oriented Summary)

## Problem Formulation

The Airport Gate Assignment Problem is treated as a constrained multi-objective optimization problem.

### Objectives

1. Minimize passenger walking distance.
2. Minimize taxiing distance.
3. Minimize CO2 proxy emissions.
4. Minimize predicted operational delay penalty.

### Constraints

1. One flight is assigned to exactly one gate.
2. Aircraft size must be compatible with gate size.
3. Flight type should match gate type (Domestic/International).
4. Two flights with overlapping occupancy time cannot use the same gate.

## Synthetic Data Design

- A configurable dataset (20-60 flights) is generated to match project scale.
- Features include timing, aircraft/flight type, passengers, taxi distances, and fuel burn proxy.
- Some arrivals are intentionally clustered to create realistic temporal overlaps.
- Gate data includes type, size, runway distance, and terminal-center distance.

## Machine Learning Component

- Target: `historical_delay_minutes`.
- Candidate models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor (if available in environment)
- Selection criteria:
  - RMSE
  - MAE
- The best model is used to provide delay penalty estimates inside optimization.

## Optimization Component

- Primary algorithm: Genetic Algorithm (GA).
- Chromosome encoding: index = flight, value = gate assignment.
- Fitness function uses normalized weighted objectives:
  - passenger walking distance
  - taxi distance
  - CO2 proxy
  - predicted delay
- Constraint violations are penalized heavily; if final solution is infeasible, a heuristic first-fit fallback is applied.

## Evaluation

The optimized assignment is compared against baseline assignment using:

1. Total CO2 proxy emissions.
2. Average passenger walking distance.
3. Average predicted delay.
4. Gate utilization percentage.

Percentage change is reported to show directional improvement.

## Assumptions and Limitations

1. Data is synthetic and does not represent real airport operations.
2. CO2 is a proxy metric derived from fuel burn and taxi-like distance.
3. Small dataset size is intended for explainability, not deployment performance.
4. Time handling is simplified to gate occupancy windows.

## Future Scope

1. Integrate real ADS-B/AODB operational data.
2. Add stochastic delays and disruption recovery logic.
3. Extend to multi-terminal and towing constraints.
4. Evaluate exact optimization baselines (MILP) for benchmark comparison.

