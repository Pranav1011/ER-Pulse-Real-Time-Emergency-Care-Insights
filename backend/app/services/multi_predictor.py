"""Multi-target prediction service with SHAP explanations."""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class MultiPredictor:
    """Unified predictor for all patient outcome models."""

    MODELS = {
        'ed_wait_time': {'type': 'regression', 'unit': 'minutes'},
        'length_of_stay': {'type': 'regression', 'unit': 'days'},
        'mortality_risk': {'type': 'classification', 'unit': 'probability'},
    }

    def __init__(self, models_dir: str = "src/models"):
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            self.models_dir = Path("../src/models")

        self.models = {}
        self.encoder = None
        self.metadata = {}
        self.explainers = {}
        self._load_models()

    def _load_models(self):
        """Load all models and metadata."""
        # Load encoder
        encoder_path = self.models_dir / "encoder.pkl"
        if encoder_path.exists():
            self.encoder = joblib.load(encoder_path)

        # Load metadata
        meta_path = self.models_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

        # Load each model
        for name in self.MODELS:
            model_path = self.models_dir / f"{name}_model.pkl"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                if SHAP_AVAILABLE:
                    try:
                        self.explainers[name] = shap.TreeExplainer(self.models[name])
                    except:
                        pass

    def _prepare_input(self, patient_data: Dict) -> np.ndarray:
        """Prepare input features from patient data."""
        cat_features = self.metadata.get('cat_features',
            ['admission_type', 'admission_location', 'ethnicity'])
        num_features = self.metadata.get('num_features',
            ['hour', 'day_of_week', 'age', 'prev_admissions', 'num_diagnoses'])

        # Create dataframe for encoding
        df = pd.DataFrame([{f: patient_data.get(f, '') for f in cat_features}])
        X_cat = self.encoder.transform(df)

        # Numeric features
        X_num = np.array([[patient_data.get(f, 0) for f in num_features]])

        return np.hstack([X_cat, X_num])

    def predict_all(self, patient_data: Dict) -> Dict:
        """Predict all outcomes for a patient."""
        X = self._prepare_input(patient_data)
        results = {}

        for name, config in self.MODELS.items():
            if name not in self.models:
                continue

            model = self.models[name]

            if config['type'] == 'regression':
                pred = float(model.predict(X)[0])
                # Confidence interval from tree variance
                if hasattr(model, 'estimators_'):
                    preds = [t.predict(X)[0] for t in model.estimators_]
                    ci_low, ci_high = np.percentile(preds, [10, 90])
                else:
                    ci_low, ci_high = pred * 0.85, pred * 1.15

                results[name] = {
                    'value': round(pred, 2),
                    'unit': config['unit'],
                    'confidence_interval': [round(ci_low, 2), round(ci_high, 2)]
                }
            else:
                proba = float(model.predict_proba(X)[0][1])
                pred_class = int(proba > 0.5)
                risk_level = 'High' if proba > 0.6 else 'Medium' if proba > 0.3 else 'Low'

                results[name] = {
                    'value': round(proba, 4),
                    'class': pred_class,
                    'risk_level': risk_level,
                    'unit': config['unit']
                }

        return results

    def explain(self, patient_data: Dict, target: str, top_k: int = 8) -> Dict:
        """Get SHAP explanation for a specific prediction."""
        if target not in self.models:
            return {'error': f'Model {target} not found'}

        if not SHAP_AVAILABLE or target not in self.explainers:
            return {'error': 'SHAP not available'}

        X = self._prepare_input(patient_data)
        model = self.models[target]
        explainer = self.explainers[target]

        # Get prediction
        if self.MODELS[target]['type'] == 'regression':
            prediction = float(model.predict(X)[0])
        else:
            prediction = float(model.predict_proba(X)[0][1])

        # Get SHAP values
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        shap_values = shap_values[0]

        feature_names = self.metadata.get('feature_names', [])

        # Sort by absolute value
        indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

        contributors = []
        for idx in indices:
            if idx < len(feature_names):
                contributors.append({
                    'feature': feature_names[idx],
                    'impact': round(float(shap_values[idx]), 4),
                    'direction': 'increases' if shap_values[idx] > 0 else 'decreases'
                })

        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        return {
            'target': target,
            'prediction': round(prediction, 4),
            'base_value': round(float(base_value), 4),
            'contributors': contributors
        }

    def get_model_metrics(self) -> Dict:
        """Get performance metrics for all models."""
        return self.metadata.get('metrics', {})


class HospitalLoadForecaster:
    """Time-series forecaster for hospital load."""

    def __init__(self, models_dir: str = "src/models"):
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            self.models_dir = Path("../src/models")

        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = self.models_dir / "hospital_load_model.pkl"
        if model_path.exists():
            self.model = joblib.load(model_path)

    def forecast(self, current_load: float, history: List[float], hours_ahead: int = 24) -> List[Dict]:
        """Forecast hospital load for next N hours."""
        if self.model is None:
            return []

        # Build initial lag features from history
        all_loads = history[-24:] + [current_load] if len(history) >= 24 else [current_load] * 24

        forecasts = []
        current_hour = pd.Timestamp.now().hour
        current_dow = pd.Timestamp.now().dayofweek
        current_dom = pd.Timestamp.now().day

        for h in range(hours_ahead):
            hour = (current_hour + h + 1) % 24
            dow = (current_dow + (current_hour + h + 1) // 24) % 7
            dom = current_dom + (current_hour + h + 1) // 24
            is_weekend = 1 if dow >= 5 else 0

            # Get lag features
            lag_1 = all_loads[-1]
            lag_2 = all_loads[-2] if len(all_loads) > 1 else lag_1
            lag_3 = all_loads[-3] if len(all_loads) > 2 else lag_1
            lag_6 = all_loads[-6] if len(all_loads) > 5 else lag_1
            lag_12 = all_loads[-12] if len(all_loads) > 11 else lag_1
            lag_24 = all_loads[-24] if len(all_loads) > 23 else lag_1

            X = np.array([[hour, dow, dom, is_weekend, lag_1, lag_2, lag_3, lag_6, lag_12, lag_24]])
            pred = float(self.model.predict(X)[0])
            pred = np.clip(pred, 40, 100)

            forecasts.append({
                'hours_ahead': h + 1,
                'predicted_load': round(pred, 1),
                'hour': hour,
                'is_weekend': bool(is_weekend)
            })

            all_loads.append(pred)

        return forecasts
