from fastapi import APIRouter, Query
from datetime import datetime
from typing import Optional
import pandas as pd
from pathlib import Path

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])

DATA_PATH = Path("data/processed")
if not DATA_PATH.exists():
    DATA_PATH = Path("../data/processed")

@router.get("/current")
def get_current_metrics():
    """Get latest values for all tracked metrics."""
    from ..services.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(str(DATA_PATH))
    metrics = detector.get_current_metrics()

    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }

@router.get("/history")
def get_metrics_history(hours: int = Query(default=24, ge=1, le=168)):
    """Get historical metric values."""
    admissions = pd.read_csv(DATA_PATH / "admissions.csv")

    # Aggregate by hour
    hourly = admissions.groupby('hour').agg({
        'ed_wait_time': ['mean', 'std', 'count'],
        'admission_duration': 'mean'
    }).reset_index()

    hourly.columns = ['hour', 'ed_wait_mean', 'ed_wait_std', 'count', 'los_mean']

    return {
        "hours_requested": hours,
        "data": hourly.to_dict('records')
    }

@router.get("/baselines")
def get_baselines():
    """Get seasonal baseline values for comparison."""
    from ..services.baseline_calculator import BaselineCalculator
    calculator = BaselineCalculator(str(DATA_PATH / "admissions.csv"))

    return {
        "timestamp": datetime.now().isoformat(),
        "baselines": calculator.get_all_baselines()
    }
