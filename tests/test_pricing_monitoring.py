import numpy as np
import pytest
import pandas as pd

from underwriting_engine.monitoring import drift_report
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


def test_price_policies_rejects_length_mismatch():
    df = pd.DataFrame({"policy_id": ["a", "b"]})
    with pytest.raises(ValueError, match="rows"):
        price_policies(df, [10, 20, 30], 0.18, 0.07, 120, [0.2, 0.5, 0.8, 0.95])


@pytest.mark.parametrize(
    "quantiles",
    [[0.2, 0.5, 0.8], [0.5, 0.2, 0.8, 0.95], [0.2, 0.5, 0.8, 1.5]],
)
def test_assign_tiers_rejects_bad_quantiles(quantiles):
    with pytest.raises(ValueError, match="quantiles"):
        assign_tiers(np.array([1.0, 2.0, 3.0]), quantiles)


def test_assign_tiers_rejects_non_finite_losses():
    with pytest.raises(ValueError, match="finite"):
        assign_tiers(np.array([1.0, np.nan]), [0.2, 0.5, 0.8, 0.95])
