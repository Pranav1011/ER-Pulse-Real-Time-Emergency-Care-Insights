"""API routes for ML models and predictions."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from pathlib import Path

from ..services.multi_predictor import MultiPredictor, HospitalLoadForecaster

router = APIRouter(prefix="/api/models", tags=["models"])

# Initialize services
predictor = MultiPredictor()
forecaster = HospitalLoadForecaster()


class PatientInput(BaseModel):
    admission_type: str = "EMERGENCY"
    admission_location: str = "EMERGENCY ROOM ADMIT"
    ethnicity: str = "WHITE"
    hour: int = 12
    day_of_week: int = 2
    age: int = 55
    prev_admissions: int = 1
    num_diagnoses: int = 3


class ForecastInput(BaseModel):
    current_load: float = 75.0
    history: List[float] = []
    hours_ahead: int = 24


class ExplainRequest(BaseModel):
    patient: PatientInput
    target: str = "ed_wait_time"


@router.post("/predict")
async def predict_outcomes(patient: PatientInput) -> Dict:
    """Predict all patient outcomes (wait time, LOS, mortality risk)."""
    try:
        results = predictor.predict_all(patient.model_dump())
        return {
            "status": "success",
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_prediction(request: ExplainRequest) -> Dict:
    """Get SHAP explanation for a specific prediction."""
    try:
        explanation = predictor.explain(
            request.patient.model_dump(),
            request.target
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast")
async def forecast_load(request: ForecastInput) -> Dict:
    """Forecast hospital load for next N hours."""
    try:
        forecasts = forecaster.forecast(
            request.current_load,
            request.history,
            request.hours_ahead
        )
        return {
            "status": "success",
            "forecasts": forecasts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_model_metrics() -> Dict:
    """Get performance metrics for all models."""
    try:
        metrics = predictor.get_model_metrics()

        # Add feature importances
        feature_importance = {}
        for name, model in predictor.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = predictor.metadata.get('feature_names', [])
                top_features = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                feature_importance[name] = [
                    {'feature': f, 'importance': round(float(i), 4)}
                    for f, i in top_features
                ]

        return {
            "status": "success",
            "metrics": metrics,
            "feature_importance": feature_importance,
            "models_loaded": list(predictor.models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_model_info() -> Dict:
    """Get information about available models."""
    return {
        "models": {
            "ed_wait_time": {
                "description": "Predicts ED wait time in minutes",
                "type": "regression",
                "target_unit": "minutes"
            },
            "length_of_stay": {
                "description": "Predicts hospital length of stay",
                "type": "regression",
                "target_unit": "days"
            },
            "mortality_risk": {
                "description": "Predicts in-hospital mortality probability",
                "type": "classification",
                "target_unit": "probability"
            },
            "hospital_load": {
                "description": "Forecasts hospital occupancy load",
                "type": "timeseries",
                "target_unit": "percentage"
            }
        },
        "features": predictor.metadata.get('feature_names', []),
        "cat_features": predictor.metadata.get('cat_features', []),
        "num_features": predictor.metadata.get('num_features', [])
    }
