import pandas as pd

from underwriting_engine.data import generate_synthetic_portfolio, split_train_holdout
from underwriting_engine.features import FEATURES
from underwriting_engine.modeling import build_models, fit_calibrator


def test_tweedie_glm_smoke_fit_predict():
    config = {
        "seed": 1,
        "model": {
            "tweedie_power": 1.5,
            "xgboost_estimators": 5,
            "xgboost_max_depth": 2,
            "xgboost_learning_rate": 0.1,
        },
    }
    df = generate_synthetic_portfolio(n_rows=250, n_geographies=12, seed=1)
    train, holdout = split_train_holdout(df, seed=1)
    model = build_models(config)["tweedie_glm"]
    model.fit(train[FEATURES], train["expected_loss"], model__sample_weight=train["exposure_years"])
    pred = model.predict(holdout[FEATURES])
    cal = fit_calibrator(holdout["expected_loss"].to_numpy(), pred, holdout["exposure_years"].to_numpy())
    calibrated = cal.predict(pred)
    assert pd.Series(calibrated).notna().all()
    assert min(calibrated) > 0
