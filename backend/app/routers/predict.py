from fastapi import APIRouter, HTTPException
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path

router = APIRouter(prefix="/api/predict", tags=["Predictions"])

class PredictionRequest(BaseModel):
    admission_type: str = Field(..., example="EMERGENCY")
    hour: int = Field(..., ge=0, le=23, example=14)
    admission_location: str = Field(..., example="EMERGENCY ROOM ADMIT")
    ethnicity: str = Field(..., example="WHITE")

MODEL_PATH = Path("src/models/ed_wait_time_model.pkl")
ENCODER_PATH = Path("src/models/encoder.pkl")

if not MODEL_PATH.exists():
    MODEL_PATH = Path("../src/models/ed_wait_time_model.pkl")
    ENCODER_PATH = Path("../src/models/encoder.pkl")

@router.post("/ed-wait")
def predict_ed_wait(request: PredictionRequest):
    """Predict ED wait time for given patient parameters."""
    try:
        from ..services.predictor import PredictorService
        predictor = PredictorService(str(MODEL_PATH), str(ENCODER_PATH))

        prediction, confidence = predictor.predict(
            request.admission_type,
            request.hour,
            request.admission_location,
            request.ethnicity
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "predicted_wait_time": round(prediction, 2),
            "confidence_interval": {
                "lower": round(confidence[0], 2),
                "upper": round(confidence[1], 2)
            },
            "unit": "minutes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain")
def explain_prediction(request: PredictionRequest):
    """Get SHAP explanation for a prediction."""
    try:
        from ..services.predictor import PredictorService
        predictor = PredictorService(str(MODEL_PATH), str(ENCODER_PATH))

        explanation = predictor.explain(
            request.admission_type,
            request.hour,
            request.admission_location,
            request.ethnicity
        )

        return {
            "timestamp": datetime.now().isoformat(),
            **explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
