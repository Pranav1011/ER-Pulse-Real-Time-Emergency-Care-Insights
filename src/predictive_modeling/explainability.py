"""
SHAP-based Model Explainability for Healthcare Analytics.
Provides global and local feature importance explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import joblib
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """
    SHAP-based explainability for healthcare prediction models.
    Provides feature importance analysis and individual prediction explanations.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        max_background_samples: int = 100
    ):
        """
        Initialize the explainer.

        Args:
            model: Trained sklearn-compatible model
            feature_names: List of feature names
            background_data: Background dataset for SHAP (sample if large)
            max_background_samples: Max samples for background data
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = model
        self.feature_names = feature_names
        self.max_background_samples = max_background_samples

        # Sample background data if needed
        if background_data is not None:
            if len(background_data) > max_background_samples:
                idx = np.random.choice(
                    len(background_data),
                    max_background_samples,
                    replace=False
                )
                self.background_data = background_data[idx]
            else:
                self.background_data = background_data
        else:
            self.background_data = None

        # Initialize SHAP explainer
        self._explainer = None
        self._shap_values = None

    @property
    def explainer(self):
        """Lazy initialization of SHAP explainer."""
        if self._explainer is None:
            self._create_explainer()
        return self._explainer

    def _create_explainer(self) -> None:
        """Create appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__

        # Tree-based models
        tree_models = [
            'RandomForestRegressor', 'RandomForestClassifier',
            'GradientBoostingRegressor', 'GradientBoostingClassifier',
            'XGBRegressor', 'XGBClassifier',
            'LGBMRegressor', 'LGBMClassifier',
            'DecisionTreeRegressor', 'DecisionTreeClassifier'
        ]

        if model_type in tree_models:
            self._explainer = shap.TreeExplainer(self.model)
        elif self.background_data is not None:
            # Use KernelExplainer for other models
            self._explainer = shap.KernelExplainer(
                self.model.predict,
                self.background_data
            )
        else:
            raise ValueError(
                f"Background data required for {model_type}. "
                "Provide background_data during initialization."
            )

    def compute_shap_values(
        self,
        X: np.ndarray,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.

        Args:
            X: Feature matrix
            check_additivity: Whether to check SHAP additivity

        Returns:
            SHAP values array
        """
        self._shap_values = self.explainer.shap_values(
            X,
            check_additivity=check_additivity
        )
        return self._shap_values

    def get_feature_importance(
        self,
        X: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.

        Args:
            X: Feature matrix (uses cached values if None)

        Returns:
            DataFrame with feature importance scores
        """
        if X is not None:
            self.compute_shap_values(X)
        elif self._shap_values is None:
            raise ValueError("No SHAP values computed. Provide X or call compute_shap_values first.")

        # Mean absolute SHAP values
        importance = np.abs(self._shap_values).mean(axis=0)

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)

        return df

    def explain_prediction(
        self,
        X_instance: np.ndarray,
        top_k: int = 10
    ) -> Dict:
        """
        Explain a single prediction.

        Args:
            X_instance: Single instance (1D or 2D array)
            top_k: Number of top features to return

        Returns:
            Dict with explanation details
        """
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        # Get prediction
        prediction = self.model.predict(X_instance)[0]

        # Get SHAP values for this instance
        shap_vals = self.explainer.shap_values(X_instance)[0]

        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_instance[0],
            'shap_value': shap_vals,
            'abs_shap': np.abs(shap_vals)
        })
        explanation_df = explanation_df.sort_values('abs_shap', ascending=False)

        # Top contributing features
        top_features = explanation_df.head(top_k)

        # Positive vs negative contributions
        positive_contrib = explanation_df[explanation_df['shap_value'] > 0].head(5)
        negative_contrib = explanation_df[explanation_df['shap_value'] < 0].head(5)

        # Base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = None

        return {
            'prediction': float(prediction),
            'base_value': float(base_value) if base_value else None,
            'top_features': top_features.to_dict('records'),
            'positive_contributors': positive_contrib[['feature', 'shap_value']].to_dict('records'),
            'negative_contributors': negative_contrib[['feature', 'shap_value']].to_dict('records'),
            'all_contributions': explanation_df.to_dict('records')
        }

    def get_summary_stats(self, X: np.ndarray) -> Dict:
        """
        Get summary statistics of feature impacts.

        Args:
            X: Feature matrix

        Returns:
            Dict with summary statistics
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        importance_df = self.get_feature_importance()

        # Calculate additional statistics
        stats = {
            'total_features': len(self.feature_names),
            'top_5_features': importance_df.head(5)['feature'].tolist(),
            'top_5_importance': importance_df.head(5)['importance'].tolist(),
            'mean_shap_magnitude': float(np.abs(self._shap_values).mean()),
            'max_shap_value': float(np.max(self._shap_values)),
            'min_shap_value': float(np.min(self._shap_values)),
            'feature_importance_dict': dict(zip(
                importance_df['feature'],
                importance_df['importance']
            ))
        }

        return stats

    def generate_explanation_report(
        self,
        X: np.ndarray,
        sample_indices: List[int] = None
    ) -> str:
        """
        Generate a markdown report explaining model behavior.

        Args:
            X: Feature matrix
            sample_indices: Optional indices for sample explanations

        Returns:
            Markdown formatted report
        """
        # Compute SHAP values
        self.compute_shap_values(X)
        importance_df = self.get_feature_importance()
        stats = self.get_summary_stats(X)

        report = """# Model Explainability Report

## Overview

This report provides SHAP-based explanations for the healthcare prediction model.
SHAP (SHapley Additive exPlanations) values show how each feature contributes
to individual predictions, with positive values increasing and negative values
decreasing the predicted wait time.

## Global Feature Importance

The following features have the highest average impact on predictions:

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
"""
        for _, row in importance_df.head(15).iterrows():
            report += f"| {int(row['rank'])} | {row['feature']} | {row['importance']:.4f} |\n"

        report += f"""
## Summary Statistics

- **Total Features:** {stats['total_features']}
- **Mean SHAP Magnitude:** {stats['mean_shap_magnitude']:.4f}
- **Max SHAP Value:** {stats['max_shap_value']:.4f}
- **Min SHAP Value:** {stats['min_shap_value']:.4f}

## Top 5 Most Influential Features

"""
        for i, (feat, imp) in enumerate(zip(stats['top_5_features'], stats['top_5_importance']), 1):
            report += f"{i}. **{feat}**: {imp:.4f}\n"

        # Sample explanations
        if sample_indices:
            report += "\n## Sample Predictions Explained\n\n"
            for idx in sample_indices[:3]:
                if idx < len(X):
                    explanation = self.explain_prediction(X[idx])
                    report += f"### Sample {idx}\n\n"
                    report += f"- **Predicted Wait Time:** {explanation['prediction']:.2f} minutes\n"
                    if explanation['base_value']:
                        report += f"- **Base Value:** {explanation['base_value']:.2f}\n"
                    report += "\nTop contributing features:\n\n"
                    for feat in explanation['top_features'][:5]:
                        direction = "+" if feat['shap_value'] > 0 else ""
                        report += f"- {feat['feature']}: {direction}{feat['shap_value']:.2f}\n"
                    report += "\n"

        report += """
## Interpretation Guide

- **Positive SHAP values** indicate the feature pushes the prediction higher (longer wait time)
- **Negative SHAP values** indicate the feature pushes the prediction lower (shorter wait time)
- **Feature importance** is the mean absolute SHAP value across all predictions
- Features with high importance have consistent impact; features with high variance have context-dependent impact
"""

        return report

    def save_explanations(
        self,
        X: np.ndarray,
        output_dir: str = "src/models/explanations"
    ) -> None:
        """
        Save SHAP explanations to files.

        Args:
            X: Feature matrix
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Compute SHAP values
        self.compute_shap_values(X)

        # Save feature importance
        importance_df = self.get_feature_importance()
        importance_df.to_csv(output_path / "feature_importance.csv", index=False)

        # Save summary stats
        stats = self.get_summary_stats(X)
        with open(output_path / "summary_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Save report
        report = self.generate_explanation_report(X, sample_indices=[0, 1, 2])
        with open(output_path / "explainability_report.md", "w") as f:
            f.write(report)

        # Save SHAP values
        np.save(output_path / "shap_values.npy", self._shap_values)

        print(f"Explanations saved to {output_path}")


def explain_model(
    model_path: str = "src/models/best_model.pkl",
    encoder_path: str = "src/models/encoder.pkl",
    data_path: str = "data/processed/admissions.csv",
    output_dir: str = "src/models/explanations"
) -> Dict:
    """
    Run full explainability pipeline.

    Args:
        model_path: Path to saved model
        encoder_path: Path to saved encoder
        data_path: Path to data CSV
        output_dir: Directory to save explanations

    Returns:
        Dict with explainability results
    """
    print("=" * 60)
    print("HEALTHCARE ANALYTICS - MODEL EXPLAINABILITY")
    print("=" * 60)

    # Load model and encoder
    print("\nLoading model and encoder...")
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Load and prepare data
    print("Loading data...")
    features = ['admission_type', 'hour', 'admission_location', 'ethnicity']
    data = pd.read_csv(data_path)
    data = data.dropna(subset=features + ['ed_wait_time'])

    X_encoded = encoder.transform(data[features])
    feature_names = encoder.get_feature_names_out(features)

    print(f"Data shape: {X_encoded.shape}")

    # Create explainer
    print("\nInitializing SHAP explainer...")
    explainer = ModelExplainer(
        model=model,
        feature_names=list(feature_names),
        background_data=X_encoded
    )

    # Compute and save explanations
    print("Computing SHAP values...")
    explainer.save_explanations(X_encoded, output_dir)

    # Get results
    importance_df = explainer.get_feature_importance()
    stats = explainer.get_summary_stats(X_encoded)

    print("\nTop 10 Features by Importance:")
    print(importance_df.head(10).to_string(index=False))

    return {
        'importance_df': importance_df,
        'stats': stats,
        'explainer': explainer
    }


if __name__ == "__main__":
    explain_model()
