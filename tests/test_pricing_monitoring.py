import numpy as np
import pandas as pd
import pytest

from underwriting_engine.monitoring import drift_report, population_stability_index
from underwriting_engine.pricing import assign_tiers, price_policies


def test_price_policies_adds_tiers_and_minimum_premium():
    df = pd.DataFrame({"policy_id": ["a", "b", "c", "d", "e"]})
    scored = price_policies(df, [10, 200, 400, 900, 1500], 0.18, 0.07, 120, [0.2, 0.5, 0.8, 0.95])
    assert "risk_tier" in scored
    assert scored["technical_premium"].min() >= 120


def test_drift_report_flags_shift():
    baseline = pd.DataFrame({"x": range(100), "cat": ["a"] * 80 + ["b"] * 20})
    current = pd.DataFrame({"x": range(50, 150), "cat": ["a"] * 20 + ["b"] * 80})
    report = drift_report(baseline, current, ["x", "cat"], warning=0.01, alert=0.1)
    assert set(report["feature"]) == {"x", "cat"}
    assert report["psi"].max() > 0


def test_psi_is_zero_for_identical_samples():
    x = pd.Series(range(100))
    assert population_stability_index(x, x.copy()) == 0.0


def test_psi_counts_values_outside_baseline_range():
    baseline = pd.Series(range(100))
    shifted = pd.Series(range(500, 600))
    assert population_stability_index(baseline, shifted) > 1.0


def test_assign_tiers_rejects_wrong_quantile_count():
    with pytest.raises(ValueError):
        assign_tiers(np.arange(10), [0.5])
