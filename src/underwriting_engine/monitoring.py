from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    # Open the outer edges so actual values outside the expected range land in
    # the extreme bin instead of being dropped as NaN, which would otherwise
    # mask the most severe kind of drift (new out-of-range values).
    edges = quantiles.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts = pd.cut(expected, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    act_counts = pd.cut(actual, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    exp = np.maximum(exp_counts.to_numpy(), 1e-6)
    act = np.maximum(act_counts.reindex(exp_counts.index, fill_value=0).to_numpy(), 1e-6)
    return float(np.sum((act - exp) * np.log(act / exp)))


def drift_report(baseline: pd.DataFrame, current: pd.DataFrame, features: list[str], warning: float, alert: float) -> pd.DataFrame:
    rows = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(baseline[feature]):
            psi = population_stability_index(baseline[feature], current[feature])
        else:
            base = baseline[feature].value_counts(normalize=True)
            cur = current[feature].value_counts(normalize=True)
            idx = sorted(set(base.index) | set(cur.index))
            b = np.maximum(base.reindex(idx, fill_value=0).to_numpy(), 1e-6)
            c = np.maximum(cur.reindex(idx, fill_value=0).to_numpy(), 1e-6)
            psi = float(np.sum((c - b) * np.log(c / b)))
        status = "alert" if psi >= alert else "warning" if psi >= warning else "ok"
        rows.append({"feature": feature, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
