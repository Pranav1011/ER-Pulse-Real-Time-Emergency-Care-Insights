from fastapi import APIRouter, Query
from datetime import datetime
from typing import Optional
from pathlib import Path

router = APIRouter(prefix="/api/anomalies", tags=["Anomalies"])

DATA_PATH = Path("data/processed")
if not DATA_PATH.exists():
    DATA_PATH = Path("../data/processed")

@router.get("/active")
def get_active_anomalies():
    """Get currently active anomalies."""
    from ..services.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(str(DATA_PATH))
    anomalies = detector.get_active_anomalies()
    scores = detector.get_anomaly_scores()

    return {
        "timestamp": datetime.now().isoformat(),
        "active_anomalies": anomalies,
        "multi_dimensional_score": scores['combined_score'],
        "is_system_anomalous": scores['is_anomalous']
    }

@router.get("/score")
def get_anomaly_scores():
    """Get real-time anomaly scores for all metrics."""
    from ..services.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(str(DATA_PATH))
    scores = detector.get_anomaly_scores()

    return {
        "timestamp": datetime.now().isoformat(),
        **scores
    }

@router.get("/history")
def get_anomaly_history(hours: int = Query(default=24, ge=1, le=168)):
    """Get historical anomalies (simulated from static data)."""
    from ..services.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(str(DATA_PATH))

    return {
        "hours_requested": hours,
        "note": "Historical data simulated from static dataset",
        "anomalies": detector.get_active_anomalies()
    }
