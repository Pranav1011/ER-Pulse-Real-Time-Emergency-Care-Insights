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
