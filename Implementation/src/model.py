from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "historical_delay_minutes"
FEATURES = [
    "aircraft_type",
    "flight_type",
    "arrival_hour",
    "expected_passengers",
    "taxi_distance_feature",
    "fuel_burn_rate",
]


@dataclass
class ModelArtifacts:
    best_model_name: str
    model: Pipeline
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def _build_x_matrix(flights: pd.DataFrame, gates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = flights.copy()
    arr_dt = pd.to_datetime(df["arrival_time"])
    df["arrival_hour"] = arr_dt.dt.hour + arr_dt.dt.minute / 60.0
    df["taxi_distance_feature"] = df["taxi_distance_to_gate"]

    if gates is not None and "assigned_gate" in df.columns:
        g = gates[["gate_id", "distance_from_runway"]].rename(columns={"gate_id": "assigned_gate"})
        df = df.merge(g, on="assigned_gate", how="left")
        df["taxi_distance_feature"] = (
            0.6 * df["taxi_distance_to_runway"] + 0.4 * df["distance_from_runway"].fillna(df["taxi_distance_to_gate"])
        )

    return df


def _make_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[("categorical", cat_pipe, cat_cols), ("numeric", num_pipe, num_cols)],
        remainder="drop",
    )


def _get_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2),
    }
    try:
        from xgboost import XGBRegressor  # type: ignore

        models["XGBoost"] = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    except Exception:
        pass
    return models


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _extract_feature_names(model: Pipeline) -> List[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    names = _extract_feature_names(model)
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        raw = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        raw = np.abs(estimator.coef_)
    else:
        raw = np.zeros(len(names))

    fi = pd.DataFrame({"feature": names, "importance": raw}).sort_values("importance", ascending=False)
    return fi.reset_index(drop=True)


def train_and_select_model(flights: pd.DataFrame, gates: pd.DataFrame) -> ModelArtifacts:
    df = _build_x_matrix(flights, gates=gates)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cat_cols = ["aircraft_type", "flight_type"]
    num_cols = ["arrival_hour", "expected_passengers", "taxi_distance_feature", "fuel_burn_rate"]
    preprocessor = _make_preprocessor(cat_cols=cat_cols, num_cols=num_cols)

    rows = []
    fitted_models: Dict[str, Pipeline] = {}
    for name, estimator in _get_models().items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = _rmse(y_test.to_numpy(), pred)
        mae = float(mean_absolute_error(y_test, pred))
        rows.append({"model": name, "rmse": rmse, "mae": mae})
        fitted_models[name] = pipe

    metrics = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_name]
    importance = _extract_feature_importance(best_model)

    return ModelArtifacts(
        best_model_name=best_name,
        model=best_model,
        metrics=metrics,
        feature_importance=importance,
    )


def predict_delays_for_assignment(
    model: Pipeline, flights: pd.DataFrame, gates: pd.DataFrame, assigned_gate_col: str = "assigned_gate"
) -> np.ndarray:
    df = flights.copy()
    df["assigned_gate"] = df[assigned_gate_col]
    feature_df = _build_x_matrix(df, gates=gates)
    X = feature_df[FEATURES]
    pred = model.predict(X)
    return np.clip(pred, a_min=0, a_max=None)
