from underwriting_engine.data import generate_synthetic_portfolio, split_train_holdout


def test_generate_synthetic_portfolio_schema_and_target():
    df = generate_synthetic_portfolio(n_rows=500, n_geographies=20, seed=7)
    assert len(df) == 500
    assert df["expected_loss"].min() > 0
    assert df["exposure_years"].between(0.25, 1.0).all()
    assert {"property_value", "credit_score", "geography_id", "expected_loss"}.issubset(df.columns)


def test_group_holdout_has_distinct_geographies():
    df = generate_synthetic_portfolio(n_rows=500, n_geographies=20, seed=7)
    train, holdout = split_train_holdout(df, seed=7)
    assert set(train["geography_id"]).isdisjoint(set(holdout["geography_id"]))
    assert len(train) > 0
    assert len(holdout) > 0
