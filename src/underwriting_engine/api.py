from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from underwriting_engine.config import load_config
from underwriting_engine.features import FEATURES
from underwriting_engine.modeling import load_model
from underwriting_engine.pricing import price_policies

logger = logging.getLogger(__name__)


class RiskRequest(BaseModel):
    property_value: float = Field(gt=0)
    building_age: float = Field(ge=0)
    square_feet: float = Field(gt=0)
    prior_claims: int = Field(ge=0)
    credit_score: float = Field(ge=300, le=850)
    income_to_rent: float = Field(gt=0)
    local_unemployment: float = Field(ge=0, le=1)
    median_income: float = Field(gt=0)
    crime_index: float = Field(ge=0)
    catastrophe_risk: float = Field(ge=0, le=1)
    rent_growth: float
    inflation: float
    deductible: float = Field(gt=0)
    coverage_limit: float = Field(gt=0)
    property_type: Literal["single_family", "condo", "small_multifamily", "large_multifamily"]
    occupancy_type: Literal["owner", "tenant", "vacant"]
    applicant_segment: Literal["standard", "thin_file", "preferred"]
    region: Literal["Northeast", "South", "Midwest", "West"]
    exposure_years: float = Field(gt=0, le=1)


app = FastAPI(title="ML Underwriting Engine", version="0.1.0")
config = load_config()
model = None
calibrator = None


@app.on_event("startup")
def _load() -> None:
    global model, calibrator
    try:
        model, calibrator = load_model(config["artifacts"]["model_dir"])
    except FileNotFoundError:
        logger.warning(
            "No trained model artifacts found in %s; /predict will return 503 until "
            "the pipeline is trained.",
            config["artifacts"]["model_dir"],
        )
        model, calibrator = None, None


@app.get("/health")
def health() -> dict[str, object]:
    model_loaded = model is not None and calibrator is not None
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}


@app.post("/predict")
def predict(request: RiskRequest) -> dict[str, object]:
    if model is None or calibrator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Train the pipeline first.")
    frame = pd.DataFrame([request.model_dump()])
    pred = calibrator.predict(model.predict(frame[FEATURES]))
    scored = price_policies(frame, pred, **config["pricing"]).iloc[0]
    return {
        "predicted_expected_loss": float(scored["predicted_expected_loss"]),
        "risk_tier": scored["risk_tier"],
        "technical_premium": float(scored["technical_premium"]),
    }
