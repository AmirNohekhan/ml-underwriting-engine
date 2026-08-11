from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_tweedie_deviance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from underwriting_engine.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

try:
    import mlflow
except Exception:  # pragma: no cover - exercised only in minimal local envs
    mlflow = None


@dataclass
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    model_path: Path
    calibration_path: Path


def _preprocess() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", encoder, CATEGORICAL_FEATURES),
        ]
    )


def build_models(config: dict[str, Any]) -> dict[str, Pipeline]:
    power = config["model"]["tweedie_power"]
    models = {
        "tweedie_glm": Pipeline(
            [
                ("preprocess", _preprocess()),
                ("model", TweedieRegressor(power=power, link="log", alpha=0.015, max_iter=700)),
            ]
        )
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost_tweedie"] = Pipeline(
            [
                ("preprocess", _preprocess()),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:tweedie",
                        tweedie_variance_power=power,
                        n_estimators=config["model"]["xgboost_estimators"],
                        max_depth=config["model"]["xgboost_max_depth"],
                        learning_rate=config["model"]["xgboost_learning_rate"],
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        n_jobs=2,
                        random_state=config["seed"],
                    ),
                ),
            ]
        )
    except Exception:
        pass
    return models


def _metrics(y_true: np.ndarray, pred: np.ndarray, exposure: np.ndarray, power: float) -> dict[str, float]:
    pred = np.maximum(pred, 1e-6)
    return {
        "weighted_mae": float(mean_absolute_error(y_true, pred, sample_weight=exposure)),
        "tweedie_deviance": float(mean_tweedie_deviance(y_true, pred, power=power, sample_weight=exposure)),
        "bias_ratio": float(np.average(pred, weights=exposure) / np.average(y_true, weights=exposure)),
    }


def cross_validate_models(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    target = config["model"]["target"]
    exposure_col = config["model"]["exposure_col"]
    group_col = config["model"]["group_col"]
    power = config["model"]["tweedie_power"]
    splitter = GroupKFold(n_splits=config["model"]["cv_folds"])
    rows: list[dict[str, float | str | int]] = []

    for model_name, model in build_models(config).items():
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(df, groups=df[group_col]), start=1):
            train, valid = df.iloc[tr_idx], df.iloc[va_idx]
            model.fit(
                train[FEATURES],
                train[target],
                model__sample_weight=train[exposure_col],
            )
            pred = model.predict(valid[FEATURES])
            rows.append({"model": model_name, "fold": fold, **_metrics(valid[target].to_numpy(), pred, valid[exposure_col].to_numpy(), power)})
    return pd.DataFrame(rows)


def fit_calibrator(y_true: np.ndarray, pred: np.ndarray, exposure: np.ndarray) -> IsotonicRegression:
    order = np.argsort(pred)
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(pred[order], y_true[order], sample_weight=exposure[order])
    return calibrator


def train_and_select(df: pd.DataFrame, holdout: pd.DataFrame, config: dict[str, Any]) -> TrainingResult:
    Path(config["artifacts"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["artifacts"]["report_dir"]).mkdir(parents=True, exist_ok=True)
    if mlflow is not None:
        mlflow.set_experiment("ml-underwriting-engine")

    cv = cross_validate_models(df, config)
    cv_path = Path(config["artifacts"]["report_dir"]) / "cv_metrics.csv"
    cv.to_csv(cv_path, index=False)
    ranking = cv.groupby("model")["tweedie_deviance"].mean().sort_values()
    best_name = ranking.index[0]
    best_model = build_models(config)[best_name]
    target = config["model"]["target"]
    exposure_col = config["model"]["exposure_col"]
    power = config["model"]["tweedie_power"]

    if mlflow is None:
        best_model.fit(df[FEATURES], df[target], model__sample_weight=df[exposure_col])
        raw_pred = best_model.predict(holdout[FEATURES])
        calibrator = fit_calibrator(holdout[target].to_numpy(), raw_pred, holdout[exposure_col].to_numpy())
        metrics = _metrics(
            holdout[target].to_numpy(),
            calibrator.predict(raw_pred),
            holdout[exposure_col].to_numpy(),
            power,
        )
        model_path = Path(config["artifacts"]["model_dir"]) / "risk_model.joblib"
        calibration_path = Path(config["artifacts"]["model_dir"]) / "calibrator.joblib"
        joblib.dump(best_model, model_path)
        joblib.dump(calibrator, calibration_path)
        return TrainingResult(best_name, metrics, model_path, calibration_path)

    with mlflow.start_run(run_name=f"train-{best_name}"):
        mlflow.log_params({"selected_model": best_name, "tweedie_power": power, "cv_folds": config["model"]["cv_folds"]})
        for model_name, value in ranking.items():
            mlflow.log_metric(f"cv_deviance_{model_name}", float(value))
        best_model.fit(df[FEATURES], df[target], model__sample_weight=df[exposure_col])
        raw_pred = best_model.predict(holdout[FEATURES])
        calibrator = fit_calibrator(holdout[target].to_numpy(), raw_pred, holdout[exposure_col].to_numpy())
        calibrated = calibrator.predict(raw_pred)
        metrics = _metrics(holdout[target].to_numpy(), calibrated, holdout[exposure_col].to_numpy(), power)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cv_path))

    model_path = Path(config["artifacts"]["model_dir"]) / "risk_model.joblib"
    calibration_path = Path(config["artifacts"]["model_dir"]) / "calibrator.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(calibrator, calibration_path)
    return TrainingResult(best_name, metrics, model_path, calibration_path)


def load_model(model_dir: str | Path = "models") -> tuple[Pipeline, IsotonicRegression]:
    model_dir = Path(model_dir)
    return joblib.load(model_dir / "risk_model.joblib"), joblib.load(model_dir / "calibrator.joblib")
