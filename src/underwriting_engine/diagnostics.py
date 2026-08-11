from __future__ import annotations

import numpy as np
import pandas as pd

from underwriting_engine.features import SENSITIVE_FEATURES


def fairness_report(scored: pd.DataFrame, target: str = "expected_loss") -> pd.DataFrame:
    rows = []
    overall_error = np.average(scored["predicted_expected_loss"] - scored[target], weights=scored["exposure_years"])
    for feature in SENSITIVE_FEATURES:
        for value, group in scored.groupby(feature):
            err = np.average(group["predicted_expected_loss"] - group[target], weights=group["exposure_years"])
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "n": int(len(group)),
                    "avg_predicted_loss": float(np.average(group["predicted_expected_loss"], weights=group["exposure_years"])),
                    "avg_actual_loss": float(np.average(group[target], weights=group["exposure_years"])),
                    "weighted_error": float(err),
                    "error_vs_overall": float(err - overall_error),
                }
            )
    return pd.DataFrame(rows)


def stability_report(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, group in scored.groupby("risk_tier"):
        actual = np.average(group["expected_loss"], weights=group["exposure_years"])
        pred = np.average(group["predicted_expected_loss"], weights=group["exposure_years"])
        rows.append(
            {
                "risk_tier": tier,
                "n": int(len(group)),
                "avg_predicted_loss": float(pred),
                "avg_actual_loss": float(actual),
                "loss_ratio": float(actual / max(pred, 1e-6)),
            }
        )
    return pd.DataFrame(rows).sort_values("risk_tier")
