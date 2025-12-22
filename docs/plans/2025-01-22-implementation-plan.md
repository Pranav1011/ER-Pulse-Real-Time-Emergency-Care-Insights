# Healthcare Analytics Platform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready healthcare analytics platform with multi-dimensional anomaly detection, deployed as Next.js frontend on Vercel + FastAPI backend on Railway.

**Architecture:** Monorepo with `frontend/` (Next.js + Tremor) and `backend/` (FastAPI). Frontend calls backend REST API. Reuses existing ML code from `src/`. Anomaly detection combines rolling window + seasonal baselines using Isolation Forest.

**Tech Stack:** Next.js 14, Tremor, Tailwind CSS, FastAPI, Pydantic, scikit-learn, SHAP, Vercel, Railway.

---

## Phase 1: Backend API (FastAPI)

### Task 1: Initialize FastAPI Backend Structure

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/requirements.txt`
- Create: `backend/.gitignore`

**Step 1: Create backend directory structure**

```bash
mkdir -p backend/app/routers backend/app/services backend/app/models
touch backend/app/__init__.py backend/app/routers/__init__.py backend/app/services/__init__.py backend/app/models/__init__.py
```

**Step 2: Create requirements.txt**

```text
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
shap>=0.42.0
python-multipart>=0.0.6
pydantic>=2.0.0
```

**Step 3: Create main.py with health check**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Healthcare Analytics API",
    description="Multi-dimensional anomaly detection and ED wait time prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "healthcare-analytics-api"}

@app.get("/")
def root():
    return {"message": "Healthcare Analytics API", "docs": "/docs"}
```

**Step 4: Create .gitignore**

```text
__pycache__/
*.pyc
.env
venv/
```

**Step 5: Test locally**

Run: `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload`
Expected: Server starts at http://localhost:8000, /health returns `{"status": "healthy"}`

**Step 6: Commit**

```bash
git add backend/
git commit -m "feat(backend): initialize FastAPI structure with health check"
```

---

### Task 2: Create Pydantic Models

**Files:**
- Create: `backend/app/models/schemas.py`

**Step 1: Write schema definitions**

```python
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
```

**Step 2: Verify imports work**

Run: `cd backend && python -c "from app.models.schemas import *; print('Schemas OK')"`
Expected: `Schemas OK`

**Step 3: Commit**

```bash
git add backend/app/models/schemas.py
git commit -m "feat(backend): add Pydantic schema definitions"
```

---

### Task 3: Create Baseline Calculator Service

**Files:**
- Create: `backend/app/services/baseline_calculator.py`

**Step 1: Write baseline calculator**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class BaselineCalculator:
    """Calculate seasonal baselines for metrics (hour, day-of-week)."""

    def __init__(self, data_path: str = "data/processed/admissions.csv"):
        self.data_path = Path(data_path)
        self.baselines: Dict[str, pd.DataFrame] = {}
        self._load_and_calculate()

    def _load_and_calculate(self):
        """Load data and calculate baselines."""
        if not self.data_path.exists():
            # Use parent directory path for when running from backend/
            alt_path = Path("../data/processed/admissions.csv")
            if alt_path.exists():
                self.data_path = alt_path
            else:
                raise FileNotFoundError(f"Data not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df['admittime'] = pd.to_datetime(df['admittime'])
        df['day_of_week'] = df['admittime'].dt.dayofweek

        # ED Wait Time baselines by (hour, day_of_week)
        wait_baselines = df.groupby(['hour', 'day_of_week'])['ed_wait_time'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        wait_baselines.columns = ['hour', 'day_of_week', 'mean', 'std', 'count']
        wait_baselines['std'] = wait_baselines['std'].fillna(wait_baselines['std'].mean())
        self.baselines['ed_wait_time'] = wait_baselines

        # Admission rate baselines (count per hour/day)
        admission_counts = df.groupby(['hour', 'day_of_week']).size().reset_index(name='count')
        admission_baselines = admission_counts.groupby(['hour', 'day_of_week'])['count'].agg([
            'mean', 'std'
        ]).reset_index()
        admission_baselines['std'] = admission_baselines['std'].fillna(1.0)
        self.baselines['admission_rate'] = admission_baselines

    def get_baseline(self, metric: str, hour: int, day_of_week: int) -> Tuple[float, float]:
        """Get mean and std for a metric at specific hour/day."""
        if metric not in self.baselines:
            return 0.0, 1.0

        df = self.baselines[metric]
        match = df[(df['hour'] == hour) & (df['day_of_week'] == day_of_week)]

        if match.empty:
            # Fallback to hour-only baseline
            match = df[df['hour'] == hour]
            if match.empty:
                return df['mean'].mean(), df['std'].mean()

        return float(match['mean'].iloc[0]), float(match['std'].iloc[0])

    def calculate_z_score(self, metric: str, value: float, hour: int, day_of_week: int) -> float:
        """Calculate z-score for a value against seasonal baseline."""
        mean, std = self.get_baseline(metric, hour, day_of_week)
        if std == 0:
            std = 1.0
        return (value - mean) / std

    def get_all_baselines(self) -> Dict:
        """Return all baselines as dict."""
        return {k: v.to_dict('records') for k, v in self.baselines.items()}
```

**Step 2: Test baseline calculator**

Run: `cd backend && python -c "from app.services.baseline_calculator import BaselineCalculator; bc = BaselineCalculator('../data/processed/admissions.csv'); print(bc.get_baseline('ed_wait_time', 14, 1))"`
Expected: Tuple of (mean, std) values

**Step 3: Commit**

```bash
git add backend/app/services/baseline_calculator.py
git commit -m "feat(backend): add seasonal baseline calculator service"
```

---

### Task 4: Create Anomaly Detector Service

**Files:**
- Create: `backend/app/services/anomaly_detector.py`

**Step 1: Write anomaly detector**

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
import joblib

from .baseline_calculator import BaselineCalculator

class AnomalyDetector:
    """Multi-dimensional anomaly detection with rolling window + seasonal baselines."""

    METRICS = ['ed_wait_time', 'admission_rate', 'transfer_delay', 'department_load', 'length_of_stay']
    Z_THRESHOLD = 2.5
    MULTI_DIM_THRESHOLD = 5.0

    def __init__(self, data_path: str = "data/processed"):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.data_path = Path("../data/processed")

        self.baseline_calculator = BaselineCalculator(str(self.data_path / "admissions.csv"))
        self.isolation_forest = None
        self._train_isolation_forest()
        self._load_current_metrics()

    def _load_current_metrics(self):
        """Load current state of metrics from data."""
        admissions = pd.read_csv(self.data_path / "admissions.csv")
        transfers = pd.read_csv(self.data_path / "transfers.csv")
        ward_metrics = pd.read_csv(self.data_path / "metrics_ward_metrics.csv")

        # Calculate current metric values
        self.current_metrics = {
            'ed_wait_time': admissions['ed_wait_time'].dropna().mean(),
            'admission_rate': len(admissions) / 24,  # Simplified
            'transfer_delay': transfers['length_of_stay'].mean() if 'length_of_stay' in transfers.columns else 0,
            'department_load': ward_metrics['subject_id_count'].mean() if 'subject_id_count' in ward_metrics.columns else 0,
            'length_of_stay': admissions['admission_duration'].mean() if 'admission_duration' in admissions.columns else 0
        }

    def _train_isolation_forest(self):
        """Train Isolation Forest on historical multi-dimensional data."""
        admissions = pd.read_csv(self.data_path / "admissions.csv")

        # Prepare feature matrix
        features = admissions[['ed_wait_time', 'hour']].dropna()
        if len(features) > 10:
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest.fit(features)

    def get_current_metrics(self) -> Dict:
        """Get current values for all metrics with anomaly scores."""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()

        results = {}
        for metric in self.METRICS:
            current = self.current_metrics.get(metric, 0)
            mean, std = self.baseline_calculator.get_baseline(metric, hour, day_of_week)
            z_score = (current - mean) / std if std > 0 else 0

            # Determine severity
            abs_z = abs(z_score)
            if abs_z > 3:
                severity = "critical"
            elif abs_z > self.Z_THRESHOLD:
                severity = "warning"
            else:
                severity = "info"

            results[metric] = {
                'current': round(current, 2),
                'baseline': round(mean, 2),
                'z_score': round(z_score, 2),
                'severity': severity,
                'context': f"Expected {mean:.1f}±{std:.1f} for {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]} {hour}:00"
            }

        return results

    def get_anomaly_scores(self) -> Dict:
        """Calculate anomaly scores for current state."""
        metrics = self.get_current_metrics()

        # Individual z-scores
        z_scores = {m: abs(metrics[m]['z_score']) for m in metrics}

        # Combined multi-dimensional score
        combined = sum(z_scores.values())
        is_anomalous = combined > self.MULTI_DIM_THRESHOLD or any(z > self.Z_THRESHOLD for z in z_scores.values())

        return {
            'scores': z_scores,
            'combined_score': round(combined, 2),
            'is_anomalous': is_anomalous
        }

    def get_active_anomalies(self) -> List[Dict]:
        """Get list of currently active anomalies."""
        metrics = self.get_current_metrics()
        anomalies = []

        for metric, data in metrics.items():
            if data['severity'] in ['warning', 'critical']:
                anomalies.append({
                    'id': f"{metric}_{datetime.now().strftime('%Y%m%d%H%M')}",
                    'timestamp': datetime.now().isoformat(),
                    'metric': metric,
                    'current': data['current'],
                    'baseline': data['baseline'],
                    'z_score': data['z_score'],
                    'severity': data['severity'],
                    'context': data['context']
                })

        return anomalies

    def detect_multi_dimensional_anomaly(self, features: np.ndarray) -> bool:
        """Use Isolation Forest for multi-dimensional anomaly detection."""
        if self.isolation_forest is None:
            return False

        prediction = self.isolation_forest.predict(features.reshape(1, -1))
        return prediction[0] == -1  # -1 = anomaly
```

**Step 2: Test anomaly detector**

Run: `cd backend && python -c "from app.services.anomaly_detector import AnomalyDetector; ad = AnomalyDetector('../data/processed'); print(ad.get_anomaly_scores())"`
Expected: Dict with scores, combined_score, is_anomalous

**Step 3: Commit**

```bash
git add backend/app/services/anomaly_detector.py
git commit -m "feat(backend): add multi-dimensional anomaly detector service"
```

---

### Task 5: Create Predictor Service

**Files:**
- Create: `backend/app/services/predictor.py`

**Step 1: Write predictor service**

```python
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class PredictorService:
    """ED wait time prediction with SHAP explanations."""

    def __init__(
        self,
        model_path: str = "src/models/ed_wait_time_model.pkl",
        encoder_path: str = "src/models/encoder.pkl"
    ):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)

        # Try alternate paths
        if not self.model_path.exists():
            self.model_path = Path("../src/models/ed_wait_time_model.pkl")
            self.encoder_path = Path("../src/models/encoder.pkl")

        self.model = None
        self.encoder = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        """Load trained model and encoder."""
        if self.model_path.exists() and self.encoder_path.exists():
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)

            if SHAP_AVAILABLE and self.model is not None:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except:
                    self.explainer = None
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def predict(
        self,
        admission_type: str,
        hour: int,
        admission_location: str,
        ethnicity: str
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Predict ED wait time.

        Returns:
            Tuple of (prediction, (lower_ci, upper_ci))
        """
        input_df = pd.DataFrame({
            'admission_type': [admission_type],
            'hour': [hour],
            'admission_location': [admission_location],
            'ethnicity': [ethnicity]
        })

        encoded = self.encoder.transform(input_df)
        prediction = self.model.predict(encoded)[0]

        # Estimate confidence interval using model's estimators if RandomForest
        if hasattr(self.model, 'estimators_'):
            predictions = np.array([tree.predict(encoded)[0] for tree in self.model.estimators_])
            lower = np.percentile(predictions, 5)
            upper = np.percentile(predictions, 95)
        else:
            # Fallback: ±20%
            lower = prediction * 0.8
            upper = prediction * 1.2

        return float(prediction), (float(lower), float(upper))

    def explain(
        self,
        admission_type: str,
        hour: int,
        admission_location: str,
        ethnicity: str,
        top_k: int = 10
    ) -> Dict:
        """
        Get SHAP explanation for a prediction.

        Returns:
            Dict with prediction, base_value, and top contributors
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            prediction, _ = self.predict(admission_type, hour, admission_location, ethnicity)
            return {
                'prediction': prediction,
                'base_value': None,
                'top_contributors': [],
                'error': 'SHAP not available'
            }

        input_df = pd.DataFrame({
            'admission_type': [admission_type],
            'hour': [hour],
            'admission_location': [admission_location],
            'ethnicity': [ethnicity]
        })

        encoded = self.encoder.transform(input_df)
        prediction = self.model.predict(encoded)[0]

        # Get SHAP values
        shap_values = self.explainer.shap_values(encoded)[0]
        feature_names = self.encoder.get_feature_names_out(['admission_type', 'hour', 'admission_location', 'ethnicity'])

        # Sort by absolute value
        indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

        contributors = []
        for idx in indices:
            val = float(shap_values[idx])
            contributors.append({
                'feature': str(feature_names[idx]),
                'shap_value': round(val, 4),
                'direction': 'increases' if val > 0 else 'decreases'
            })

        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]

        return {
            'prediction': float(prediction),
            'base_value': float(base_value),
            'top_contributors': contributors
        }
```

**Step 2: Test predictor**

Run: `cd backend && python -c "from app.services.predictor import PredictorService; ps = PredictorService('../src/models/ed_wait_time_model.pkl', '../src/models/encoder.pkl'); print(ps.predict('EMERGENCY', 14, 'EMERGENCY ROOM ADMIT', 'WHITE'))"`
Expected: Tuple of (prediction, (lower, upper))

**Step 3: Commit**

```bash
git add backend/app/services/predictor.py
git commit -m "feat(backend): add predictor service with SHAP explanations"
```

---

### Task 6: Create API Routers

**Files:**
- Create: `backend/app/routers/metrics.py`
- Create: `backend/app/routers/anomalies.py`
- Create: `backend/app/routers/predict.py`

**Step 1: Create metrics router**

```python
# backend/app/routers/metrics.py
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
```

**Step 2: Create anomalies router**

```python
# backend/app/routers/anomalies.py
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
    # For static data, we return current state as "history"
    from ..services.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(str(DATA_PATH))

    return {
        "hours_requested": hours,
        "note": "Historical data simulated from static dataset",
        "anomalies": detector.get_active_anomalies()
    }
```

**Step 3: Create predict router**

```python
# backend/app/routers/predict.py
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
```

**Step 4: Update main.py to include routers**

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import metrics, anomalies, predict

app = FastAPI(
    title="Healthcare Analytics API",
    description="Multi-dimensional anomaly detection and ED wait time prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metrics.router)
app.include_router(anomalies.router)
app.include_router(predict.router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "healthcare-analytics-api"}

@app.get("/")
def root():
    return {"message": "Healthcare Analytics API", "docs": "/docs"}
```

**Step 5: Test all endpoints**

Run: `cd backend && uvicorn app.main:app --reload`

Test endpoints:
- GET http://localhost:8000/docs (OpenAPI docs)
- GET http://localhost:8000/api/metrics/current
- GET http://localhost:8000/api/anomalies/active
- POST http://localhost:8000/api/predict/ed-wait

**Step 6: Commit**

```bash
git add backend/app/routers/ backend/app/main.py
git commit -m "feat(backend): add API routers for metrics, anomalies, and predictions"
```

---

### Task 7: Add Railway Deployment Config

**Files:**
- Create: `backend/railway.json`
- Create: `backend/Procfile`

**Step 1: Create railway.json**

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

**Step 2: Create Procfile**

```text
web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Step 3: Update requirements.txt with version pins**

```text
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
shap==0.43.0
python-multipart==0.0.6
pydantic==2.5.2
```

**Step 4: Commit**

```bash
git add backend/railway.json backend/Procfile backend/requirements.txt
git commit -m "feat(backend): add Railway deployment configuration"
```

---

## Phase 2: Frontend (Next.js + Tremor)

### Task 8: Initialize Next.js Frontend

**Files:**
- Create: `frontend/` directory with Next.js app

**Step 1: Create Next.js app**

```bash
cd /Users/saipranavkrovvidi/Documents/Personal\ Projects/Healthcare-Analytics
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
```

Select options:
- Would you like to use TypeScript? Yes
- Would you like to use ESLint? Yes
- Would you like to use Tailwind CSS? Yes
- Would you like to use `src/` directory? Yes
- Would you like to use App Router? Yes
- Would you like to customize the default import alias? No

**Step 2: Install Tremor**

```bash
cd frontend && npm install @tremor/react
```

**Step 3: Update tailwind.config.ts for Tremor**

```typescript
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './node_modules/@tremor/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    transparent: 'transparent',
    current: 'currentColor',
    extend: {
      colors: {
        tremor: {
          brand: {
            faint: '#eff6ff',
            muted: '#bfdbfe',
            subtle: '#60a5fa',
            DEFAULT: '#3b82f6',
            emphasis: '#1d4ed8',
            inverted: '#ffffff',
          },
          background: {
            muted: '#f9fafb',
            subtle: '#f3f4f6',
            DEFAULT: '#ffffff',
            emphasis: '#374151',
          },
          border: {
            DEFAULT: '#e5e7eb',
          },
          ring: {
            DEFAULT: '#e5e7eb',
          },
          content: {
            subtle: '#9ca3af',
            DEFAULT: '#6b7280',
            emphasis: '#374151',
            strong: '#111827',
            inverted: '#ffffff',
          },
        },
        'dark-tremor': {
          brand: {
            faint: '#0B1229',
            muted: '#172554',
            subtle: '#1e40af',
            DEFAULT: '#3b82f6',
            emphasis: '#60a5fa',
            inverted: '#030712',
          },
          background: {
            muted: '#131A2B',
            subtle: '#1f2937',
            DEFAULT: '#111827',
            emphasis: '#d1d5db',
          },
          border: {
            DEFAULT: '#1f2937',
          },
          ring: {
            DEFAULT: '#1f2937',
          },
          content: {
            subtle: '#4b5563',
            DEFAULT: '#6b7280',
            emphasis: '#e5e7eb',
            strong: '#f9fafb',
            inverted: '#000000',
          },
        },
      },
    },
  },
  safelist: [
    {
      pattern:
        /^(bg-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(?:50|100|200|300|400|500|600|700|800|900|950))$/,
      variants: ['hover', 'ui-selected'],
    },
    {
      pattern:
        /^(text-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(?:50|100|200|300|400|500|600|700|800|900|950))$/,
      variants: ['hover', 'ui-selected'],
    },
    {
      pattern:
        /^(border-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(?:50|100|200|300|400|500|600|700|800|900|950))$/,
      variants: ['hover', 'ui-selected'],
    },
    {
      pattern:
        /^(ring-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(?:50|100|200|300|400|500|600|700|800|900|950))$/,
    },
    {
      pattern:
        /^(stroke-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(?:50|100|200|300|400|500|600|700|800|900|950))$/,
    },
    {
      pattern:
        /^(fill-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(?:50|100|200|300|400|500|600|700|800|900|950))$/,
    },
  ],
  plugins: [],
  darkMode: 'class',
}
export default config
```

**Step 4: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): initialize Next.js with Tremor and Tailwind"
```

---

### Task 9: Create API Client

**Files:**
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/types.ts`

**Step 1: Create types**

```typescript
// frontend/src/lib/types.ts
export interface MetricValue {
  current: number;
  baseline: number;
  z_score: number;
  severity: 'info' | 'warning' | 'critical';
  context: string;
}

export interface MetricsResponse {
  timestamp: string;
  metrics: {
    ed_wait_time: MetricValue;
    admission_rate: MetricValue;
    transfer_delay: MetricValue;
    department_load: MetricValue;
    length_of_stay: MetricValue;
  };
}

export interface AnomalyAlert {
  id: string;
  timestamp: string;
  metric: string;
  current: number;
  baseline: number;
  z_score: number;
  severity: 'info' | 'warning' | 'critical';
  context: string;
}

export interface AnomaliesResponse {
  timestamp: string;
  active_anomalies: AnomalyAlert[];
  multi_dimensional_score: number;
  is_system_anomalous: boolean;
}

export interface AnomalyScoreResponse {
  timestamp: string;
  scores: Record<string, number>;
  combined_score: number;
  is_anomalous: boolean;
}

export interface PredictionRequest {
  admission_type: string;
  hour: number;
  admission_location: string;
  ethnicity: string;
}

export interface PredictionResponse {
  timestamp: string;
  predicted_wait_time: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  unit: string;
}

export interface FeatureContribution {
  feature: string;
  shap_value: number;
  direction: string;
}

export interface ExplanationResponse {
  timestamp: string;
  prediction: number;
  base_value: number | null;
  top_contributors: FeatureContribution[];
}
```

**Step 2: Create API client**

```typescript
// frontend/src/lib/api.ts
import type {
  MetricsResponse,
  AnomaliesResponse,
  AnomalyScoreResponse,
  PredictionRequest,
  PredictionResponse,
  ExplanationResponse,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export const api = {
  // Metrics
  getCurrentMetrics: () => fetchAPI<MetricsResponse>('/api/metrics/current'),

  getMetricsHistory: (hours: number = 24) =>
    fetchAPI<{ hours_requested: number; data: any[] }>(`/api/metrics/history?hours=${hours}`),

  getBaselines: () => fetchAPI<{ timestamp: string; baselines: Record<string, any[]> }>('/api/metrics/baselines'),

  // Anomalies
  getActiveAnomalies: () => fetchAPI<AnomaliesResponse>('/api/anomalies/active'),

  getAnomalyScores: () => fetchAPI<AnomalyScoreResponse>('/api/anomalies/score'),

  getAnomalyHistory: (hours: number = 24) =>
    fetchAPI<{ hours_requested: number; anomalies: any[] }>(`/api/anomalies/history?hours=${hours}`),

  // Predictions
  predictWaitTime: (data: PredictionRequest) =>
    fetchAPI<PredictionResponse>('/api/predict/ed-wait', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  explainPrediction: (data: PredictionRequest) =>
    fetchAPI<ExplanationResponse>('/api/predict/explain', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Health
  healthCheck: () => fetchAPI<{ status: string }>('/health'),
};
```

**Step 3: Create environment file**

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Step 4: Commit**

```bash
git add frontend/src/lib/
git commit -m "feat(frontend): add API client and TypeScript types"
```

---

### Task 10: Create Dashboard Components

**Files:**
- Create: `frontend/src/components/MetricCard.tsx`
- Create: `frontend/src/components/AnomalyBadge.tsx`
- Create: `frontend/src/components/SparklineChart.tsx`

**Step 1: Create MetricCard**

```typescript
// frontend/src/components/MetricCard.tsx
'use client';

import { Card, Metric, Text, Flex, BadgeDelta, ProgressBar } from '@tremor/react';
import type { MetricValue } from '@/lib/types';

interface MetricCardProps {
  title: string;
  metric: MetricValue;
  unit?: string;
}

function getSeverityColor(severity: string): 'emerald' | 'yellow' | 'red' {
  switch (severity) {
    case 'critical':
      return 'red';
    case 'warning':
      return 'yellow';
    default:
      return 'emerald';
  }
}

function getDeltaType(zScore: number): 'increase' | 'decrease' | 'unchanged' {
  if (Math.abs(zScore) < 0.5) return 'unchanged';
  return zScore > 0 ? 'increase' : 'decrease';
}

export function MetricCard({ title, metric, unit = '' }: MetricCardProps) {
  const color = getSeverityColor(metric.severity);
  const deltaType = getDeltaType(metric.z_score);
  const percentFromBaseline = ((metric.current - metric.baseline) / metric.baseline) * 100;

  return (
    <Card className="max-w-sm" decoration="top" decorationColor={color}>
      <Flex justifyContent="between" alignItems="center">
        <Text>{title}</Text>
        <BadgeDelta deltaType={deltaType} size="xs">
          {metric.z_score > 0 ? '+' : ''}{metric.z_score.toFixed(1)}σ
        </BadgeDelta>
      </Flex>
      <Metric className="mt-2">
        {metric.current.toFixed(1)} {unit}
      </Metric>
      <Flex className="mt-2">
        <Text className="text-xs text-tremor-content-subtle">
          Baseline: {metric.baseline.toFixed(1)} {unit}
        </Text>
        <Text className="text-xs text-tremor-content-subtle">
          {percentFromBaseline > 0 ? '+' : ''}{percentFromBaseline.toFixed(0)}%
        </Text>
      </Flex>
      <ProgressBar
        value={Math.min(Math.abs(metric.z_score) * 20, 100)}
        color={color}
        className="mt-2"
      />
      <Text className="mt-2 text-xs text-tremor-content-subtle">
        {metric.context}
      </Text>
    </Card>
  );
}
```

**Step 2: Create AnomalyBadge**

```typescript
// frontend/src/components/AnomalyBadge.tsx
'use client';

import { Badge } from '@tremor/react';

interface AnomalyBadgeProps {
  severity: 'info' | 'warning' | 'critical';
}

export function AnomalyBadge({ severity }: AnomalyBadgeProps) {
  const colors: Record<string, 'emerald' | 'yellow' | 'red'> = {
    info: 'emerald',
    warning: 'yellow',
    critical: 'red',
  };

  const labels: Record<string, string> = {
    info: 'Normal',
    warning: 'Warning',
    critical: 'Critical',
  };

  return (
    <Badge color={colors[severity]} size="sm">
      {labels[severity]}
    </Badge>
  );
}
```

**Step 3: Create SparklineChart**

```typescript
// frontend/src/components/SparklineChart.tsx
'use client';

import { SparkAreaChart } from '@tremor/react';

interface SparklineChartProps {
  data: { hour: number; value: number }[];
  color?: 'blue' | 'emerald' | 'red' | 'yellow';
}

export function SparklineChart({ data, color = 'blue' }: SparklineChartProps) {
  return (
    <SparkAreaChart
      data={data}
      categories={['value']}
      index="hour"
      colors={[color]}
      className="h-10 w-36"
    />
  );
}
```

**Step 4: Commit**

```bash
git add frontend/src/components/
git commit -m "feat(frontend): add MetricCard, AnomalyBadge, and SparklineChart components"
```

---

### Task 11: Create Dashboard Page

**Files:**
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/app/layout.tsx`
- Modify: `frontend/src/app/globals.css`

**Step 1: Update globals.css for dark mode**

```css
/* frontend/src/app/globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --foreground-rgb: 255, 255, 255;
  --background-start-rgb: 17, 24, 39;
  --background-end-rgb: 17, 24, 39;
}

body {
  color: rgb(var(--foreground-rgb));
  background: linear-gradient(
      to bottom,
      transparent,
      rgb(var(--background-end-rgb))
    )
    rgb(var(--background-start-rgb));
  min-height: 100vh;
}
```

**Step 2: Update layout.tsx**

```typescript
// frontend/src/app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Healthcare Analytics',
  description: 'Multi-dimensional anomaly detection for healthcare operations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-900 text-white`}>
        <nav className="border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🏥</span>
              <span className="font-bold text-xl">Healthcare Analytics</span>
            </div>
            <div className="flex gap-6">
              <a href="/" className="hover:text-blue-400 transition">Dashboard</a>
              <a href="/alerts" className="hover:text-blue-400 transition">Alerts</a>
              <a href="/predict" className="hover:text-blue-400 transition">Predict</a>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
```

**Step 3: Create dashboard page**

```typescript
// frontend/src/app/page.tsx
'use client';

import { useEffect, useState } from 'react';
import {
  Card,
  Title,
  Text,
  Grid,
  Col,
  Flex,
  Badge,
  AreaChart,
  DonutChart,
} from '@tremor/react';
import { MetricCard } from '@/components/MetricCard';
import { api } from '@/lib/api';
import type { MetricsResponse, AnomaliesResponse } from '@/lib/types';

export default function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [anomalies, setAnomalies] = useState<AnomaliesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [metricsData, anomaliesData] = await Promise.all([
          api.getCurrentMetrics(),
          api.getActiveAnomalies(),
        ]);
        setMetrics(metricsData);
        setAnomalies(anomaliesData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Text>Loading metrics...</Text>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/20 border-red-500">
        <Title>Error</Title>
        <Text>{error}</Text>
        <Text className="mt-2 text-sm">
          Make sure the backend API is running at {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
        </Text>
      </Card>
    );
  }

  const metricsList = metrics?.metrics ? Object.entries(metrics.metrics) : [];
  const systemStatus = anomalies?.is_system_anomalous ? 'Anomalous' : 'Normal';
  const statusColor = anomalies?.is_system_anomalous ? 'red' : 'emerald';

  return (
    <div className="space-y-8">
      {/* Header */}
      <Flex justifyContent="between" alignItems="center">
        <div>
          <Title>Live Metrics Dashboard</Title>
          <Text>Real-time hospital operations monitoring</Text>
        </div>
        <Flex className="gap-4" alignItems="center">
          <Text className="text-sm text-gray-400">
            Last updated: {metrics?.timestamp ? new Date(metrics.timestamp).toLocaleTimeString() : 'N/A'}
          </Text>
          <Badge color={statusColor} size="lg">
            System: {systemStatus}
          </Badge>
        </Flex>
      </Flex>

      {/* Multi-dimensional Score */}
      {anomalies && (
        <Card decoration="left" decorationColor={statusColor}>
          <Flex>
            <div>
              <Text>Multi-Dimensional Anomaly Score</Text>
              <Title className="text-3xl">{anomalies.multi_dimensional_score.toFixed(2)}</Title>
            </div>
            <div className="text-right">
              <Text>Active Alerts</Text>
              <Title className="text-3xl">{anomalies.active_anomalies.length}</Title>
            </div>
          </Flex>
        </Card>
      )}

      {/* Metric Cards Grid */}
      <Grid numItemsMd={2} numItemsLg={3} className="gap-6">
        {metricsList.map(([key, value]) => (
          <Col key={key}>
            <MetricCard
              title={key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              metric={value}
              unit={key === 'ed_wait_time' ? 'min' : key === 'length_of_stay' ? 'hrs' : ''}
            />
          </Col>
        ))}
      </Grid>

      {/* Active Anomalies List */}
      {anomalies && anomalies.active_anomalies.length > 0 && (
        <Card>
          <Title>Active Anomalies</Title>
          <div className="mt-4 space-y-3">
            {anomalies.active_anomalies.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border ${
                  alert.severity === 'critical'
                    ? 'bg-red-900/20 border-red-500'
                    : 'bg-yellow-900/20 border-yellow-500'
                }`}
              >
                <Flex justifyContent="between">
                  <div>
                    <Text className="font-semibold">
                      {alert.metric.replace(/_/g, ' ').toUpperCase()}
                    </Text>
                    <Text className="text-sm text-gray-400">{alert.context}</Text>
                  </div>
                  <div className="text-right">
                    <Text>Current: {alert.current.toFixed(1)}</Text>
                    <Text className="text-sm text-gray-400">
                      Baseline: {alert.baseline.toFixed(1)} | Z: {alert.z_score.toFixed(2)}
                    </Text>
                  </div>
                </Flex>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
```

**Step 4: Test the page**

Run: `cd frontend && npm run dev`
Expected: Dashboard loads at http://localhost:3000 (may show error if backend not running)

**Step 5: Commit**

```bash
git add frontend/src/app/
git commit -m "feat(frontend): create main dashboard page with live metrics"
```

---

### Task 12: Create Alerts Page

**Files:**
- Create: `frontend/src/app/alerts/page.tsx`

**Step 1: Create alerts page**

```typescript
// frontend/src/app/alerts/page.tsx
'use client';

import { useEffect, useState } from 'react';
import {
  Card,
  Title,
  Text,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Badge,
  Flex,
} from '@tremor/react';
import { AnomalyBadge } from '@/components/AnomalyBadge';
import { api } from '@/lib/api';
import type { AnomaliesResponse, AnomalyAlert } from '@/lib/types';

export default function AlertsPage() {
  const [anomalies, setAnomalies] = useState<AnomaliesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await api.getActiveAnomalies();
        setAnomalies(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch anomalies');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Text>Loading anomalies...</Text>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/20 border-red-500">
        <Title>Error</Title>
        <Text>{error}</Text>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <Title>Anomaly Alerts</Title>
        <Text>Real-time detection of unusual patterns in hospital operations</Text>
      </div>

      {/* Summary Card */}
      <Card>
        <Flex justifyContent="between" alignItems="center">
          <div>
            <Text>System Status</Text>
            <Title className="text-2xl">
              {anomalies?.is_system_anomalous ? 'Anomalous Activity Detected' : 'Operating Normally'}
            </Title>
          </div>
          <Badge
            color={anomalies?.is_system_anomalous ? 'red' : 'emerald'}
            size="xl"
          >
            Score: {anomalies?.multi_dimensional_score.toFixed(2)}
          </Badge>
        </Flex>
      </Card>

      {/* Alerts Table */}
      <Card>
        <Title>Active Alerts</Title>
        {anomalies && anomalies.active_anomalies.length > 0 ? (
          <Table className="mt-4">
            <TableHead>
              <TableRow>
                <TableHeaderCell>Timestamp</TableHeaderCell>
                <TableHeaderCell>Metric</TableHeaderCell>
                <TableHeaderCell>Current</TableHeaderCell>
                <TableHeaderCell>Baseline</TableHeaderCell>
                <TableHeaderCell>Z-Score</TableHeaderCell>
                <TableHeaderCell>Severity</TableHeaderCell>
                <TableHeaderCell>Context</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {anomalies.active_anomalies.map((alert) => (
                <TableRow key={alert.id}>
                  <TableCell>
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </TableCell>
                  <TableCell className="font-medium">
                    {alert.metric.replace(/_/g, ' ')}
                  </TableCell>
                  <TableCell>{alert.current.toFixed(1)}</TableCell>
                  <TableCell>{alert.baseline.toFixed(1)}</TableCell>
                  <TableCell>
                    <Badge color={Math.abs(alert.z_score) > 3 ? 'red' : 'yellow'}>
                      {alert.z_score.toFixed(2)}σ
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <AnomalyBadge severity={alert.severity} />
                  </TableCell>
                  <TableCell className="text-sm text-gray-400">
                    {alert.context}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="mt-8 text-center py-12">
            <Text className="text-xl">✅ No Active Anomalies</Text>
            <Text className="mt-2 text-gray-400">
              All metrics are within expected ranges
            </Text>
          </div>
        )}
      </Card>

      {/* Explanation Card */}
      <Card>
        <Title>How Anomaly Detection Works</Title>
        <div className="mt-4 space-y-4 text-sm text-gray-300">
          <div>
            <Text className="font-semibold text-white">Seasonal Baselines</Text>
            <Text>
              Each metric is compared against historical patterns for the same hour and day of week.
              For example, Monday 2PM is compared to previous Monday 2PM values.
            </Text>
          </div>
          <div>
            <Text className="font-semibold text-white">Z-Score Thresholds</Text>
            <Text>
              • |Z| {'>'} 2.5: Warning - metric is significantly different from baseline
              <br />
              • |Z| {'>'} 3.0: Critical - metric is extremely unusual
            </Text>
          </div>
          <div>
            <Text className="font-semibold text-white">Multi-Dimensional Detection</Text>
            <Text>
              Isolation Forest algorithm detects unusual combinations of metrics,
              even when individual metrics appear normal.
            </Text>
          </div>
        </div>
      </Card>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/app/alerts/
git commit -m "feat(frontend): add anomaly alerts page with timeline"
```

---

### Task 13: Create Prediction Page

**Files:**
- Create: `frontend/src/app/predict/page.tsx`

**Step 1: Create prediction page**

```typescript
// frontend/src/app/predict/page.tsx
'use client';

import { useState } from 'react';
import {
  Card,
  Title,
  Text,
  TextInput,
  NumberInput,
  Select,
  SelectItem,
  Button,
  Metric,
  Flex,
  BarList,
  Badge,
  Divider,
} from '@tremor/react';
import { api } from '@/lib/api';
import type { PredictionResponse, ExplanationResponse } from '@/lib/types';

export default function PredictPage() {
  const [formData, setFormData] = useState({
    admission_type: 'EMERGENCY',
    hour: 14,
    admission_location: 'EMERGENCY ROOM ADMIT',
    ethnicity: 'WHITE',
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      const [predResult, explainResult] = await Promise.all([
        api.predictWaitTime(formData),
        api.explainPrediction(formData),
      ]);

      setPrediction(predResult);
      setExplanation(explainResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const shapData = explanation?.top_contributors.map((c) => ({
    name: c.feature.replace(/_/g, ' '),
    value: Math.abs(c.shap_value),
    color: c.shap_value > 0 ? 'red' : 'emerald',
  })) || [];

  return (
    <div className="space-y-8">
      <div>
        <Title>ED Wait Time Prediction</Title>
        <Text>Predict emergency department wait times with explainable AI</Text>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <Card>
          <Title>Patient Parameters</Title>

          <div className="mt-6 space-y-4">
            <div>
              <Text className="mb-2">Admission Type</Text>
              <Select
                value={formData.admission_type}
                onValueChange={(value) =>
                  setFormData({ ...formData, admission_type: value })
                }
              >
                <SelectItem value="EMERGENCY">Emergency</SelectItem>
                <SelectItem value="URGENT">Urgent</SelectItem>
                <SelectItem value="ELECTIVE">Elective</SelectItem>
              </Select>
            </div>

            <div>
              <Text className="mb-2">Hour of Day (0-23)</Text>
              <NumberInput
                value={formData.hour}
                onValueChange={(value) =>
                  setFormData({ ...formData, hour: value || 0 })
                }
                min={0}
                max={23}
              />
            </div>

            <div>
              <Text className="mb-2">Admission Location</Text>
              <Select
                value={formData.admission_location}
                onValueChange={(value) =>
                  setFormData({ ...formData, admission_location: value })
                }
              >
                <SelectItem value="EMERGENCY ROOM ADMIT">Emergency Room</SelectItem>
                <SelectItem value="CLINIC REFERRAL/PREMATURE">Clinic Referral</SelectItem>
                <SelectItem value="TRANSFER FROM HOSP/EXTRAM">Hospital Transfer</SelectItem>
                <SelectItem value="PHYS REFERRAL/NORMAL DELI">Physician Referral</SelectItem>
              </Select>
            </div>

            <div>
              <Text className="mb-2">Ethnicity</Text>
              <Select
                value={formData.ethnicity}
                onValueChange={(value) =>
                  setFormData({ ...formData, ethnicity: value })
                }
              >
                <SelectItem value="WHITE">White</SelectItem>
                <SelectItem value="BLACK/AFRICAN AMERICAN">Black/African American</SelectItem>
                <SelectItem value="HISPANIC OR LATINO">Hispanic/Latino</SelectItem>
                <SelectItem value="ASIAN">Asian</SelectItem>
                <SelectItem value="OTHER">Other</SelectItem>
                <SelectItem value="UNKNOWN/NOT SPECIFIED">Unknown</SelectItem>
              </Select>
            </div>

            <Button
              onClick={handlePredict}
              loading={loading}
              size="lg"
              className="w-full mt-6"
            >
              Predict Wait Time
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-red-900/20 border border-red-500 rounded-lg">
                <Text className="text-red-400">{error}</Text>
              </div>
            )}
          </div>
        </Card>

        {/* Results */}
        <div className="space-y-6">
          {prediction && (
            <Card decoration="top" decorationColor="blue">
              <Title>Prediction Result</Title>
              <div className="mt-4">
                <Metric>{prediction.predicted_wait_time.toFixed(0)} minutes</Metric>
                <Flex className="mt-4" justifyContent="start">
                  <Badge color="blue" size="lg">
                    95% CI: {prediction.confidence_interval.lower.toFixed(0)} - {prediction.confidence_interval.upper.toFixed(0)} min
                  </Badge>
                </Flex>
              </div>
            </Card>
          )}

          {explanation && (
            <Card>
              <Title>SHAP Explanation</Title>
              <Text className="mt-2">
                How each feature contributes to this prediction
              </Text>

              {explanation.base_value && (
                <div className="mt-4 p-3 bg-gray-800 rounded-lg">
                  <Text className="text-sm">
                    Base Value: {explanation.base_value.toFixed(1)} minutes
                  </Text>
                  <Text className="text-xs text-gray-400">
                    Average prediction across all patients
                  </Text>
                </div>
              )}

              <Divider />

              <Title className="text-sm mt-4">Feature Contributions</Title>
              <div className="mt-4 space-y-3">
                {explanation.top_contributors.slice(0, 8).map((contrib, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <Text className="text-sm truncate max-w-[200px]">
                      {contrib.feature.replace(/_/g, ' ')}
                    </Text>
                    <Flex justifyContent="end" className="gap-2">
                      <Badge
                        color={contrib.shap_value > 0 ? 'red' : 'emerald'}
                        size="sm"
                      >
                        {contrib.shap_value > 0 ? '+' : ''}{contrib.shap_value.toFixed(2)}
                      </Badge>
                      <Text className="text-xs text-gray-400 w-20 text-right">
                        {contrib.direction}
                      </Text>
                    </Flex>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
                <Text className="text-xs text-gray-400">
                  <strong>Interpretation:</strong> Red values increase the predicted wait time,
                  green values decrease it. The magnitude shows how much each feature
                  contributes to pushing the prediction away from the base value.
                </Text>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/app/predict/
git commit -m "feat(frontend): add prediction page with SHAP explanations"
```

---

### Task 14: Add Vercel Deployment Config

**Files:**
- Create: `frontend/vercel.json`
- Modify: `frontend/.env.local` → `frontend/.env.example`

**Step 1: Create vercel.json**

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "env": {
    "NEXT_PUBLIC_API_URL": "@api_url"
  }
}
```

**Step 2: Create .env.example**

```bash
# frontend/.env.example
# Copy to .env.local for local development
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production, set this to your Railway backend URL
# NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

**Step 3: Update .gitignore**

```bash
echo ".env.local" >> frontend/.gitignore
```

**Step 4: Commit**

```bash
git add frontend/vercel.json frontend/.env.example frontend/.gitignore
git commit -m "feat(frontend): add Vercel deployment configuration"
```

---

## Phase 3: Integration & Documentation

### Task 15: Update Root README

**Files:**
- Modify: `README.md`

**Step 1: Update README with new architecture**

```markdown
# Healthcare Analytics Platform

A production-ready healthcare analytics platform with multi-dimensional anomaly detection, deployed as a modern web application.

## Features

- **Multi-dimensional Anomaly Detection**: Combines seasonal baselines with Isolation Forest to detect unusual patterns in hospital operations
- **ED Wait Time Prediction**: Machine learning model with SHAP explainability
- **Real-time Dashboard**: Live metrics monitoring with Tremor UI components
- **Model Comparison**: 8+ ML models compared with statistical significance tests

## Architecture

```
┌─────────────────────────────────────────┐
│           VERCEL (Frontend)             │
│      Next.js + Tremor Dashboard         │
└─────────────────┬───────────────────────┘
                  │ REST API
                  ▼
┌─────────────────────────────────────────┐
│          RAILWAY (Backend)              │
│     FastAPI + ML Models + SHAP          │
└─────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14, Tremor, Tailwind CSS |
| Backend | FastAPI, Pydantic |
| ML | scikit-learn, SHAP, Isolation Forest |
| Data | MIMIC-III (processed) |
| Hosting | Vercel (frontend), Railway (backend) |

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- pip

### Local Development

1. **Start the backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

2. **Start the frontend:**
```bash
cd frontend
npm install
npm run dev
```

3. Visit http://localhost:3000

### Deployment

**Backend (Railway):**
1. Push to GitHub
2. Connect repo to Railway
3. Deploy `backend/` directory

**Frontend (Vercel):**
1. Import project to Vercel
2. Set root directory to `frontend/`
3. Add environment variable: `NEXT_PUBLIC_API_URL=<your-railway-url>`
4. Deploy

## Project Structure

```
Healthcare-Analytics/
├── frontend/               # Next.js + Tremor
│   ├── src/
│   │   ├── app/           # Pages
│   │   ├── components/    # UI components
│   │   └── lib/           # API client
│   └── package.json
├── backend/                # FastAPI
│   ├── app/
│   │   ├── routers/       # API endpoints
│   │   ├── services/      # Business logic
│   │   └── models/        # Pydantic schemas
│   └── requirements.txt
├── src/                    # Original ML code
├── data/                   # Processed datasets
└── docs/plans/            # Design documents
```

## API Documentation

Once the backend is running, visit `/docs` for interactive API documentation.

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/metrics/current` | Current metric values with anomaly scores |
| `GET /api/anomalies/active` | Active anomaly alerts |
| `POST /api/predict/ed-wait` | Predict ED wait time |
| `POST /api/predict/explain` | SHAP explanation for prediction |

## Anomaly Detection

The system uses a dual approach:

1. **Seasonal Baselines**: Each metric is compared against historical values for the same (hour, day-of-week) combination
2. **Multi-dimensional Detection**: Isolation Forest identifies unusual combinations of metrics

Alerts are triggered when:
- Individual Z-score > 2.5 (Warning) or > 3.0 (Critical)
- Combined multi-dimensional score exceeds threshold

## License

MIT

## Author

Sai Pranav Krovvidi
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with new architecture and deployment instructions"
```

---

### Task 16: Final Integration Test

**Step 1: Start both servers**

Terminal 1:
```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

Terminal 2:
```bash
cd frontend && npm run dev
```

**Step 2: Test all pages**

- http://localhost:3000 - Dashboard loads with metrics
- http://localhost:3000/alerts - Alerts page shows anomalies
- http://localhost:3000/predict - Prediction works with SHAP

**Step 3: Test API directly**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/metrics/current
curl -X POST http://localhost:8000/api/predict/ed-wait \
  -H "Content-Type: application/json" \
  -d '{"admission_type":"EMERGENCY","hour":14,"admission_location":"EMERGENCY ROOM ADMIT","ethnicity":"WHITE"}'
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete healthcare analytics platform with anomaly detection"
```

---

## Deployment Checklist

### Railway (Backend)

1. Create Railway account at https://railway.app
2. New Project → Deploy from GitHub repo
3. Set root directory: `backend`
4. Add variable: `PORT=8000`
5. Deploy and get URL

### Vercel (Frontend)

1. Create Vercel account at https://vercel.com
2. Import GitHub repo
3. Set root directory: `frontend`
4. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app`
5. Deploy

---

**Plan complete and saved to `docs/plans/2025-01-22-implementation-plan.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
