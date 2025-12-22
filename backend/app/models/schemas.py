from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class SeverityLevel(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"

class MetricValue(BaseModel):
    name: str
    current: float
    baseline: float
    z_score: float
    status: SeverityLevel

class MetricsResponse(BaseModel):
    timestamp: datetime
    ed_wait_time: MetricValue
    admission_rate: MetricValue
    transfer_delay: MetricValue
    department_load: MetricValue
    length_of_stay: MetricValue

class AnomalyAlert(BaseModel):
    id: str
    timestamp: datetime
    metric: str
    current: float
    baseline: float
    z_score: float
    severity: SeverityLevel
    context: str

class AnomaliesResponse(BaseModel):
    active_anomalies: List[AnomalyAlert]
    multi_dimensional_score: float
    timestamp: datetime

class AnomalyScoreResponse(BaseModel):
    scores: dict
    combined_score: float
    is_anomalous: bool
    timestamp: datetime

class PredictionRequest(BaseModel):
    admission_type: str = Field(..., example="EMERGENCY")
    hour: int = Field(..., ge=0, le=23, example=14)
    admission_location: str = Field(..., example="EMERGENCY ROOM ADMIT")
    ethnicity: str = Field(..., example="WHITE")

class PredictionResponse(BaseModel):
    predicted_wait_time: float
    confidence_interval: tuple
    timestamp: datetime

class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str

class ExplanationResponse(BaseModel):
    prediction: float
    base_value: float
    top_contributors: List[FeatureContribution]
    timestamp: datetime

class HistoryParams(BaseModel):
    hours: int = Field(default=24, ge=1, le=168)

class BaselineResponse(BaseModel):
    hour: int
    day_of_week: int
    ed_wait_time_mean: float
    ed_wait_time_std: float
    admission_rate_mean: float
    admission_rate_std: float
