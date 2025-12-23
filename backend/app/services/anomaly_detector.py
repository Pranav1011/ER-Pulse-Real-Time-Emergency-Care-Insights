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

        # Individual z-scores (convert to float for JSON serialization)
        z_scores = {m: float(abs(metrics[m]['z_score'])) for m in metrics}

        # Combined multi-dimensional score
        combined = sum(z_scores.values())
        is_anomalous = combined > self.MULTI_DIM_THRESHOLD or any(z > self.Z_THRESHOLD for z in z_scores.values())

        return {
            'scores': z_scores,
            'combined_score': round(float(combined), 2),
            'is_anomalous': bool(is_anomalous)
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
