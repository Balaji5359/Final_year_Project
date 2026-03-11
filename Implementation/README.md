# Intelligent Airport Gate Assignment (Proof of Concept)

This project is a final-year research proof-of-concept for intelligent airport gate assignment.

## What it does

- Generates a realistic synthetic airport operations dataset (20-60 flights configurable).
- Trains and compares machine learning regressors for delay prediction.
- Optimizes gate assignment with a Genetic Algorithm style search and a fallback heuristic.
- Evaluates baseline vs optimized assignment on walking distance, taxi distance, CO2 proxy, delay proxy, and gate utilization.
- Presents results in a Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional CLI run:

```bash
python run_pipeline.py --flights 30 --gates 8 --seed 42
```

## Structure

- `src/data.py`: synthetic data generation.
- `src/model.py`: ML training and prediction utilities.
- `src/optimize.py`: baseline and GA optimization.
- `src/evaluate.py`: metrics and comparison summaries.
- `src/pipeline.py`: one-call experiment runner.
- `app.py`: Streamlit dashboard.
- `run_pipeline.py`: CLI runner that exports CSV files into `outputs/`.

## Notes

- This is not a deployment-grade system.
- It is intentionally small-scale and explainable for academic demonstration.
