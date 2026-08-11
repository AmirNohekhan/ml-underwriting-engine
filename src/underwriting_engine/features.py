from __future__ import annotations

NUMERIC_FEATURES = [
    "property_value",
    "building_age",
    "square_feet",
    "prior_claims",
    "credit_score",
    "income_to_rent",
    "local_unemployment",
    "median_income",
    "crime_index",
    "catastrophe_risk",
    "rent_growth",
    "inflation",
    "deductible",
    "coverage_limit",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "occupancy_type",
    "applicant_segment",
    "region",
]

SENSITIVE_FEATURES = [
    "applicant_segment",
    "region",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
