from fastapi.testclient import TestClient

from underwriting_engine import api


def test_health_reports_degraded_when_model_not_loaded():
    api.model = None
    api.calibrator = None
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_predict_returns_503_when_model_not_loaded():
    api.model = None
    api.calibrator = None
    client = TestClient(api.app)
    payload = {
        "property_value": 325000,
        "building_age": 28,
        "square_feet": 1600,
        "prior_claims": 1,
        "credit_score": 705,
        "income_to_rent": 3.2,
        "local_unemployment": 0.045,
        "median_income": 76000,
        "crime_index": 42,
        "catastrophe_risk": 0.22,
        "rent_growth": 0.035,
        "inflation": 0.032,
        "deductible": 1000,
        "coverage_limit": 260000,
        "property_type": "single_family",
        "occupancy_type": "owner",
        "applicant_segment": "standard",
        "region": "Midwest",
        "exposure_years": 1.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 503
