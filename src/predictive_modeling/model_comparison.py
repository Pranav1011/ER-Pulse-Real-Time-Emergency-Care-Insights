"""
Model Comparison Framework for Healthcare Analytics.
Compares multiple ML models with cross-validation and statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy import stats

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass
class ModelResult:
    """Container for model evaluation results."""
    name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    training_time: float
    model: object


class ModelComparison:
    """
    Framework for comparing multiple ML models on healthcare data.
    Includes cross-validation, statistical significance tests, and visualization.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the model comparison framework.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.results: Dict[str, ModelResult] = {}
        self.best_model: Optional[ModelResult] = None

    def get_models(self) -> Dict[str, object]:
        """
        Get dictionary of models to compare.

        Returns:
            Dict mapping model names to model instances
        """
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=self.random_state),
            "Lasso": Lasso(random_state=self.random_state),
            "ElasticNet": ElasticNet(random_state=self.random_state),
            "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=self.n_jobs),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )

        return models

    def prepare_data(
        self,
        filepath: str,
        features: List[str] = None,
        target: str = "ed_wait_time"
    ) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder]:
        """
        Load and prepare data for modeling.

        Args:
            filepath: Path to CSV file
            features: List of feature column names
            target: Target column name

        Returns:
            Tuple of (X, y, encoder)
        """
        if features is None:
            features = ['admission_type', 'hour', 'admission_location', 'ethnicity']

        data = pd.read_csv(filepath)
        data = data.dropna(subset=features + [target])

        # Encode categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(data[features])

        # Create feature DataFrame
        feature_names = encoder.get_feature_names_out(features)
        X = pd.DataFrame(X_encoded, columns=feature_names)
        y = data[target].values

        return X.values, y, encoder

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: Dict[str, object] = None
    ) -> Dict[str, ModelResult]:
        """
        Train and evaluate all models using cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            models: Optional dict of models (uses default if None)

        Returns:
            Dict mapping model names to ModelResult objects
        """
        import time

        if models is None:
            models = self.get_models()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Cross-validation setup
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        results = {}

        for name, model in models.items():
            print(f"Training {name}...")
            start_time = time.time()

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=kfold,
                scoring='neg_mean_absolute_error',
                n_jobs=self.n_jobs
            )
            cv_scores = -cv_scores  # Convert to positive MAE

            # Train final model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate on test set
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Handle potential division issues with MAPE
            mask = y_test != 0
            if mask.sum() > 0:
                mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
            else:
                mape = np.nan

            result = ModelResult(
                name=name,
                mae=mae,
                rmse=rmse,
                r2=r2,
                mape=mape,
                cv_scores=cv_scores,
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std(),
                training_time=training_time,
                model=model
            )

            results[name] = result
            print(f"  MAE: {mae:.2f}, R²: {r2:.4f}, CV: {cv_scores.mean():.2f}±{cv_scores.std():.2f}")

        self.results = results
        self._find_best_model()

        return results

    def _find_best_model(self) -> None:
        """Identify the best model based on cross-validation MAE."""
        if not self.results:
            return

        best = min(self.results.values(), key=lambda x: x.cv_mean)
        self.best_model = best
        print(f"\nBest model: {best.name} (CV MAE: {best.cv_mean:.2f}±{best.cv_std:.2f})")

    def statistical_comparison(
        self,
        model1_name: str,
        model2_name: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform statistical test comparing two models.

        Uses paired t-test on cross-validation scores.

        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            alpha: Significance level

        Returns:
            Dict with test results
        """
        if model1_name not in self.results or model2_name not in self.results:
            raise ValueError("Both models must exist in results")

        scores1 = self.results[model1_name].cv_scores
        scores2 = self.results[model2_name].cv_scores

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Effect size (Cohen's d)
        diff = scores1 - scores2
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_pvalue = stats.wilcoxon(scores1, scores2)
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan

        significant = p_value < alpha

        return {
            "model1": model1_name,
            "model2": model2_name,
            "model1_cv_mean": scores1.mean(),
            "model2_cv_mean": scores2.mean(),
            "difference": scores1.mean() - scores2.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": significant,
            "cohens_d": cohens_d,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_pvalue,
            "alpha": alpha
        }

    def compare_all_pairs(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare all pairs of models statistically.

        Args:
            alpha: Significance level

        Returns:
            DataFrame with pairwise comparison results
        """
        model_names = list(self.results.keys())
        comparisons = []

        for i, m1 in enumerate(model_names):
            for m2 in model_names[i+1:]:
                result = self.statistical_comparison(m1, m2, alpha)
                comparisons.append(result)

        return pd.DataFrame(comparisons)

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a formatted DataFrame.

        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            return pd.DataFrame()

        data = []
        for name, result in self.results.items():
            data.append({
                "Model": name,
                "MAE": result.mae,
                "RMSE": result.rmse,
                "R²": result.r2,
                "MAPE (%)": result.mape,
                "CV MAE (mean)": result.cv_mean,
                "CV MAE (std)": result.cv_std,
                "Training Time (s)": result.training_time
            })

        df = pd.DataFrame(data)
        df = df.sort_values("CV MAE (mean)")
        return df

    def save_best_model(
        self,
        model_path: str,
        encoder: OneHotEncoder,
        encoder_path: str
    ) -> None:
        """
        Save the best model and encoder.

        Args:
            model_path: Path to save model
            encoder: Fitted encoder to save
            encoder_path: Path to save encoder
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run train_and_evaluate first.")

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(encoder_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.best_model.model, model_path)
        joblib.dump(encoder, encoder_path)

        # Save comparison results
        results_path = Path(model_path).parent / "model_comparison_results.csv"
        self.get_results_dataframe().to_csv(results_path, index=False)

        print(f"Best model ({self.best_model.name}) saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
        print(f"Comparison results saved to {results_path}")

    def generate_report(self) -> str:
        """
        Generate a markdown report of model comparison.

        Returns:
            Markdown formatted report string
        """
        if not self.results:
            return "No results available. Run train_and_evaluate first."

        df = self.get_results_dataframe()

        report = """# Model Comparison Report

## Summary

"""
        report += f"- **Number of models compared:** {len(self.results)}\n"
        report += f"- **Cross-validation folds:** {self.cv_folds}\n"
        report += f"- **Best model:** {self.best_model.name}\n"
        report += f"- **Best CV MAE:** {self.best_model.cv_mean:.2f} (±{self.best_model.cv_std:.2f})\n\n"

        report += "## Model Performance\n\n"
        report += "| Model | MAE | RMSE | R² | MAPE (%) | CV MAE | Training Time (s) |\n"
        report += "|-------|-----|------|-----|----------|--------|-------------------|\n"

        for _, row in df.iterrows():
            report += f"| {row['Model']} | {row['MAE']:.2f} | {row['RMSE']:.2f} | "
            report += f"{row['R²']:.4f} | {row['MAPE (%)']:.1f} | "
            report += f"{row['CV MAE (mean)']:.2f}±{row['CV MAE (std)']:.2f} | "
            report += f"{row['Training Time (s)']:.2f} |\n"

        report += "\n## Metric Definitions\n\n"
        report += "- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values\n"
        report += "- **RMSE (Root Mean Squared Error):** Square root of average squared differences\n"
        report += "- **R² (Coefficient of Determination):** Proportion of variance explained by the model\n"
        report += "- **MAPE (Mean Absolute Percentage Error):** Average percentage error\n"
        report += "- **CV MAE:** Cross-validated MAE with standard deviation\n"

        return report


def run_comparison(
    data_path: str = "data/processed/admissions.csv",
    model_output_path: str = "src/models/best_model.pkl",
    encoder_output_path: str = "src/models/encoder.pkl"
) -> Dict:
    """
    Run full model comparison pipeline.

    Args:
        data_path: Path to data CSV
        model_output_path: Path to save best model
        encoder_output_path: Path to save encoder

    Returns:
        Dict with comparison results
    """
    print("=" * 60)
    print("HEALTHCARE ANALYTICS - MODEL COMPARISON")
    print("=" * 60)

    # Initialize comparison
    comparison = ModelComparison(cv_folds=5, random_state=42)

    # Prepare data
    print("\nPreparing data...")
    X, y, encoder = comparison.prepare_data(data_path)
    print(f"Data shape: {X.shape}")

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = comparison.train_and_evaluate(X, y)

    # Statistical comparisons
    print("\nStatistical comparisons with best model:")
    if comparison.best_model:
        for name in results.keys():
            if name != comparison.best_model.name:
                test = comparison.statistical_comparison(comparison.best_model.name, name)
                sig = "significant" if test["significant"] else "not significant"
                print(f"  {comparison.best_model.name} vs {name}: p={test['p_value']:.4f} ({sig})")

    # Save best model
    print("\nSaving best model...")
    comparison.save_best_model(model_output_path, encoder, encoder_output_path)

    # Generate report
    report = comparison.generate_report()
    report_path = Path(model_output_path).parent / "model_comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    return {
        "results_df": comparison.get_results_dataframe(),
        "best_model": comparison.best_model.name,
        "best_mae": comparison.best_model.cv_mean,
        "comparison": comparison
    }


if __name__ == "__main__":
    run_comparison()
