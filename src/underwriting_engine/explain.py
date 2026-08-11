from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from underwriting_engine.features import FEATURES


def global_explanations(model, sample: pd.DataFrame, out_path: str | Path) -> pd.DataFrame:
    """Write a compact global importance report.

    Uses SHAP when available. Falls back to model feature importances or GLM
    coefficients so the project remains runnable in constrained environments.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shap

        transformed = model.named_steps["preprocess"].transform(sample[FEATURES])
        names = model.named_steps["preprocess"].get_feature_names_out()
        explainer = shap.Explainer(model.named_steps["model"], transformed)
        values = explainer(transformed[: min(500, len(sample))])
        importance = np.abs(values.values).mean(axis=0)
        report = pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        estimator = model.named_steps["model"]
        names = model.named_steps["preprocess"].get_feature_names_out()
        if hasattr(estimator, "feature_importances_"):
            scores = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            scores = np.abs(estimator.coef_)
        else:
            scores = np.zeros(len(names))
        report = pd.DataFrame({"feature": names, "importance": scores}).sort_values("importance", ascending=False)
    report.to_csv(out_path, index=False)
    return report
