from __future__ import annotations

import numpy as np
import pandas as pd


TIER_LABELS = ["A_preferred", "B_standard", "C_watch", "D_high", "E_decline_review"]


def assign_tiers(predicted_loss: np.ndarray, quantiles: list[float]) -> tuple[np.ndarray, list[float]]:
    predicted_loss = np.asarray(predicted_loss, dtype=float)
    if predicted_loss.size == 0:
        raise ValueError("predicted_loss must not be empty")
    if not np.isfinite(predicted_loss).all():
        raise ValueError("predicted_loss must contain only finite values")
    if len(quantiles) != len(TIER_LABELS) - 1:
        raise ValueError(f"quantiles must contain {len(TIER_LABELS) - 1} cut points")
    if any(not 0.0 <= q <= 1.0 for q in quantiles) or any(a > b for a, b in zip(quantiles, quantiles[1:])):
        raise ValueError("quantiles must be within [0, 1] and non-decreasing")
    cuts = np.quantile(predicted_loss, quantiles).tolist()
    tiers = pd.cut(pd.Series(predicted_loss), bins=[-np.inf, *cuts, np.inf], labels=TIER_LABELS, include_lowest=True)
    return tiers.astype(str).to_numpy(), cuts


def price_policies(
    df: pd.DataFrame,
    predicted_loss: np.ndarray,
    expense_load: float,
    profit_load: float,
    min_premium: float,
    tier_quantiles: list[float],
) -> pd.DataFrame:
    predicted_loss = np.asarray(predicted_loss, dtype=float)
    if len(df) != len(predicted_loss):
        raise ValueError(f"df has {len(df)} rows but predicted_loss has {len(predicted_loss)} values")
    tiers, cuts = assign_tiers(predicted_loss, tier_quantiles)
    premium = predicted_loss * (1.0 + expense_load + profit_load)
    premium = np.maximum(premium, min_premium)
    out = df.copy()
    out["predicted_expected_loss"] = np.round(predicted_loss, 2)
    out["risk_tier"] = tiers
    out["technical_premium"] = np.round(premium, 2)
    out.attrs["tier_cutpoints"] = cuts
    return out
