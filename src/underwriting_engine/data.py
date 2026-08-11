from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_portfolio(
    n_rows: int = 12000,
    n_geographies: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Create public-safe synthetic insurance/rental underwriting records.

    The target is expected loss per policy term. It is generated from a compound
    frequency-severity process so Tweedie-like models have a defensible role.
    """
    rng = np.random.default_rng(seed)
    geo_ids = np.arange(n_geographies)
    regions = rng.choice(["Northeast", "South", "Midwest", "West"], size=n_geographies)
    geo_base_income = rng.normal(72000, 14000, n_geographies).clip(38000, 130000)
    geo_cat = rng.beta(2.0, 5.0, n_geographies)
    geo_crime = rng.gamma(2.2, 22, n_geographies).clip(5, 150)
    geo_unemp = rng.normal(0.045, 0.013, n_geographies).clip(0.015, 0.11)
    geo_rent_growth = rng.normal(0.035, 0.018, n_geographies).clip(-0.03, 0.11)

    geography_id = rng.choice(geo_ids, size=n_rows)
    property_type = rng.choice(
        ["single_family", "condo", "small_multifamily", "large_multifamily"],
        p=[0.42, 0.26, 0.22, 0.10],
        size=n_rows,
    )
    occupancy_type = rng.choice(["owner", "tenant", "vacant"], p=[0.54, 0.40, 0.06], size=n_rows)
    applicant_segment = rng.choice(["standard", "thin_file", "preferred"], p=[0.66, 0.16, 0.18], size=n_rows)

    region = regions[geography_id]
    median_income = geo_base_income[geography_id] + rng.normal(0, 3500, n_rows)
    catastrophe_risk = geo_cat[geography_id] + rng.normal(0, 0.035, n_rows)
    crime_index = geo_crime[geography_id] + rng.normal(0, 7, n_rows)
    local_unemployment = geo_unemp[geography_id] + rng.normal(0, 0.004, n_rows)
    rent_growth = geo_rent_growth[geography_id] + rng.normal(0, 0.008, n_rows)
    inflation = rng.normal(0.032, 0.006, n_rows)

    property_value = rng.lognormal(mean=12.4, sigma=0.42, size=n_rows).clip(85000, 1_250_000)
    square_feet = (property_value / rng.normal(210, 32, n_rows)).clip(450, 5500)
    building_age = rng.gamma(4.0, 9.0, n_rows).clip(0, 120)
    prior_claims = rng.poisson(
        0.18
        + 0.55 * (catastrophe_risk > 0.45)
        + 0.25 * (crime_index > 70)
        + 0.18 * (occupancy_type == "vacant")
    )
    credit_mu = np.select(
        [applicant_segment == "preferred", applicant_segment == "thin_file"],
        [740, 635],
        default=695,
    )
    credit_score = rng.normal(credit_mu, 42, n_rows).clip(480, 820)
    income_to_rent = rng.normal(3.1, 0.85, n_rows).clip(0.8, 8.0)
    deductible = rng.choice([500, 1000, 2500, 5000], p=[0.30, 0.42, 0.20, 0.08], size=n_rows)
    coverage_limit = property_value * rng.uniform(0.55, 0.95, n_rows)
    exposure_years = rng.uniform(0.25, 1.0, n_rows)

    type_factor = pd.Series(property_type).map(
        {"single_family": 0.00, "condo": -0.10, "small_multifamily": 0.16, "large_multifamily": 0.30}
    ).to_numpy()
    occ_factor = pd.Series(occupancy_type).map({"owner": -0.08, "tenant": 0.05, "vacant": 0.45}).to_numpy()
    segment_factor = pd.Series(applicant_segment).map(
        {"preferred": -0.16, "standard": 0.0, "thin_file": 0.18}
    ).to_numpy()
    region_factor = pd.Series(region).map({"Northeast": 0.06, "South": 0.12, "Midwest": -0.03, "West": 0.10}).to_numpy()

    log_frequency = (
        -2.95
        + type_factor
        + occ_factor
        + segment_factor
        + region_factor
        + 0.013 * prior_claims
        + 0.95 * catastrophe_risk
        + 0.005 * crime_index
        + 2.2 * local_unemployment
        + 0.010 * np.maximum(building_age - 35, 0)
        - 0.0023 * (credit_score - 680)
        - 0.045 * np.log1p(deductible / 500)
    )
    claim_count = rng.poisson(np.exp(log_frequency) * exposure_years)
    severity_mean = (
        1100
        + 0.010 * coverage_limit
        + 13 * building_age
        + 1800 * catastrophe_risk
        + 280 * prior_claims
        + 3000 * inflation
    )
    severity = rng.gamma(shape=2.1, scale=severity_mean / 2.1)
    pure_premium = claim_count * severity / exposure_years
    process_noise = rng.gamma(shape=7.0, scale=1 / 7.0, size=n_rows)
    expected_loss = (0.35 * severity_mean * np.exp(log_frequency) + 0.65 * pure_premium) * process_noise
    expected_loss = np.maximum(expected_loss, 1.0)

    df = pd.DataFrame(
        {
            "policy_id": [f"P{i:07d}" for i in range(n_rows)],
            "geography_id": geography_id.astype(str),
            "region": region,
            "property_type": property_type,
            "occupancy_type": occupancy_type,
            "applicant_segment": applicant_segment,
            "property_value": property_value.round(2),
            "building_age": building_age.round(1),
            "square_feet": square_feet.round(0),
            "prior_claims": prior_claims,
            "credit_score": credit_score.round(0),
            "income_to_rent": income_to_rent.round(2),
            "local_unemployment": local_unemployment.round(4),
            "median_income": median_income.round(0),
            "crime_index": crime_index.round(2),
            "catastrophe_risk": catastrophe_risk.clip(0, 1).round(4),
            "rent_growth": rent_growth.round(4),
            "inflation": inflation.round(4),
            "deductible": deductible,
            "coverage_limit": coverage_limit.round(2),
            "exposure_years": exposure_years.round(4),
            "expected_loss": expected_loss.round(2),
        }
    )
    return df


def split_train_holdout(df: pd.DataFrame, seed: int = 42, holdout_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    geos = df["geography_id"].unique()
    holdout_geos = set(rng.choice(geos, size=max(1, int(len(geos) * holdout_frac)), replace=False))
    holdout = df[df["geography_id"].isin(holdout_geos)].copy()
    train = df[~df["geography_id"].isin(holdout_geos)].copy()
    return train, holdout


def write_dataset(train: pd.DataFrame, holdout: pd.DataFrame, train_path: str, holdout_path: str) -> None:
    for path, frame in [(train_path, train), (holdout_path, holdout)]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)
