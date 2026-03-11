from __future__ import annotations

import streamlit as st

from src.pipeline import run_experiment


st.set_page_config(page_title="Intelligent Airport Gate Assignment", layout="wide")
st.title("Intelligent Airport Gate Assignment System")
st.caption("Final-year research proof-of-concept (synthetic data, ML + optimization).")

st.sidebar.header("Configuration")
num_flights = st.sidebar.slider("Number of Flights", min_value=20, max_value=60, value=30, step=1)
num_gates = st.sidebar.slider("Number of Gates", min_value=6, max_value=16, value=8, step=1)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

if st.button("Run Optimization", type="primary"):
    with st.spinner("Running synthetic generation, ML training, and optimization..."):
        artifacts = run_experiment(
            num_flights=int(num_flights),
            num_gates=int(num_gates),
            random_seed=int(seed),
            save_outputs=True,
            output_dir="outputs",
        )

    st.subheader("Project Overview")
    st.write(
        "This system generates a synthetic airport operations dataset, predicts delay with machine learning, "
        "and solves gate assignment using a GA-based search under compatibility and overlap constraints."
    )

    st.subheader("Dataset Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("`Flights`")
        st.dataframe(artifacts.flights, use_container_width=True)
    with c2:
        st.markdown("`Gates`")
        st.dataframe(artifacts.gates, use_container_width=True)

    st.subheader("ML Model Performance")
    st.write(f"Best model selected: `{artifacts.model.best_model_name}`")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("`Model Metrics (RMSE/MAE)`")
        st.dataframe(artifacts.model.metrics, use_container_width=True)
    with m2:
        st.markdown("`Feature Importance`")
        st.dataframe(artifacts.model.feature_importance.head(12), use_container_width=True)

    st.subheader("Optimization Results")
    if artifacts.used_fallback:
        st.warning("GA returned an infeasible solution; fallback heuristic was used.")
    else:
        st.success("GA optimization produced a feasible assignment.")

    st.markdown("`Before vs After Comparison`")
    st.dataframe(artifacts.comparison, use_container_width=True)

    st.subheader("CO2 and Distance Visualizations")
    metric_df = artifacts.comparison.copy()
    chart_df = metric_df.set_index("metric")[["baseline", "optimized"]]
    st.bar_chart(chart_df)

    st.subheader("Assignment Samples")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("`Baseline Assignment`")
        st.dataframe(artifacts.baseline_assignment[["flight_id", "arrival_time", "departure_time", "baseline_gate"]], use_container_width=True)
    with a2:
        st.markdown("`Optimized Assignment`")
        st.dataframe(artifacts.optimized_assignment[["flight_id", "arrival_time", "departure_time", "optimized_gate"]], use_container_width=True)
else:
    st.info("Set parameters in the sidebar and click `Run Optimization`.")

