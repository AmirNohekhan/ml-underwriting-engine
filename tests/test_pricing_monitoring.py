import numpy as np
import pandas as pd

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


def test_psi_is_zero_for_identical_distributions():
    series = pd.Series(range(200))
    assert population_stability_index(series, series) < 1e-6


def test_psi_returns_zero_for_constant_baseline():
    assert population_stability_index(pd.Series([1.0] * 50), pd.Series(range(50))) == 0.0


def test_psi_grows_with_shift():
    baseline = pd.Series(range(200))
    small = population_stability_index(baseline, baseline + 10)
    large = population_stability_index(baseline, baseline + 100)
    assert 0 < small < large


def test_assign_tiers_orders_losses():
    tiers, cuts = assign_tiers(np.arange(100), [0.2, 0.5, 0.8, 0.95])
    assert len(cuts) == 4
    assert tiers[0] == "A_preferred"
    assert tiers[-1] == "E_decline_review"
