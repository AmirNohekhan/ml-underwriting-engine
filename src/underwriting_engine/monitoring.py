from __future__ import annotations

import numpy as np
import pandas as pd


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    exp = np.maximum(expected, 1e-6)
    act = np.maximum(actual, 1e-6)
    return float(np.sum((act - exp) * np.log(act / exp)))


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    # Open the outer edges so values outside the baseline range land in the
    # first/last bucket instead of being dropped as NaN (which hid drift).
    edges = np.concatenate([[-np.inf], quantiles[1:-1], [np.inf]])
    exp_counts = pd.cut(expected.dropna(), bins=edges).value_counts(normalize=True, sort=False)
    act_counts = pd.cut(actual.dropna(), bins=edges).value_counts(normalize=True, sort=False)
    return _psi_from_proportions(exp_counts.to_numpy(), act_counts.reindex(exp_counts.index, fill_value=0).to_numpy())


def drift_report(baseline: pd.DataFrame, current: pd.DataFrame, features: list[str], warning: float, alert: float) -> pd.DataFrame:
    rows = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(baseline[feature]):
            psi = population_stability_index(baseline[feature], current[feature])
        else:
            base = baseline[feature].value_counts(normalize=True)
            cur = current[feature].value_counts(normalize=True)
            idx = sorted(set(base.index) | set(cur.index))
            psi = _psi_from_proportions(
                base.reindex(idx, fill_value=0).to_numpy(), cur.reindex(idx, fill_value=0).to_numpy()
            )
        status = "alert" if psi >= alert else "warning" if psi >= warning else "ok"
        rows.append({"feature": feature, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
