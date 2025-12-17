"""
Statistical Rigor Module for Healthcare Analytics.
Provides confidence intervals, hypothesis tests, and bootstrap analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import json
from pathlib import Path


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float]
    interpretation: str


class StatisticalAnalyzer:
    """
    Provides rigorous statistical analysis for model evaluation.
    Includes bootstrap confidence intervals, hypothesis tests, and effect sizes.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the statistical analyzer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval for a statistic.

        Args:
            data: 1D array of data
            statistic: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)

        Returns:
            ConfidenceInterval object
        """
        n = len(data)
        bootstrap_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic(sample)

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        estimate = statistic(data)

        return ConfidenceInterval(
            estimate=estimate,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            method="bootstrap"
        )

    def bootstrap_model_evaluation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 100,
        test_size: float = 0.2,
        confidence_level: float = 0.95
    ) -> Dict[str, ConfidenceInterval]:
        """
        Bootstrap confidence intervals for model metrics.

        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target vector
            n_bootstrap: Number of bootstrap iterations
            test_size: Proportion of data for testing
            confidence_level: Confidence level

        Returns:
            Dict mapping metric names to ConfidenceInterval objects
        """
        n = len(y)
        n_test = int(n * test_size)

        mae_scores = []
        r2_scores = []

        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            train_idx = np.random.choice(n, size=n-n_test, replace=True)
            test_idx = np.random.choice(n, size=n_test, replace=True)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        mae_scores = np.array(mae_scores)
        r2_scores = np.array(r2_scores)

        alpha = 1 - confidence_level

        return {
            'mae': ConfidenceInterval(
                estimate=np.mean(mae_scores),
                lower=np.percentile(mae_scores, alpha/2 * 100),
                upper=np.percentile(mae_scores, (1 - alpha/2) * 100),
                confidence_level=confidence_level,
                method="bootstrap"
            ),
            'r2': ConfidenceInterval(
                estimate=np.mean(r2_scores),
                lower=np.percentile(r2_scores, alpha/2 * 100),
                upper=np.percentile(r2_scores, (1 - alpha/2) * 100),
                confidence_level=confidence_level,
                method="bootstrap"
            )
        }

    def paired_t_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> HypothesisTestResult:
        """
        Perform paired t-test comparing two models' CV scores.

        Args:
            scores1: CV scores from model 1
            scores2: CV scores from model 2
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            HypothesisTestResult object
        """
        t_stat, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)

        # Cohen's d effect size
        diff = scores1 - scores2
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

        significant = p_value < alpha

        # Interpretation
        if not significant:
            interpretation = "No significant difference between models"
        else:
            better = "first" if diff.mean() > 0 else "second"
            effect = "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
            interpretation = f"The {better} model is significantly better with {effect} effect size"

        return HypothesisTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            alpha=alpha,
            effect_size=cohens_d,
            interpretation=interpretation
        )

    def wilcoxon_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            scores1: CV scores from model 1
            scores2: CV scores from model 2
            alpha: Significance level

        Returns:
            HypothesisTestResult object
        """
        try:
            w_stat, p_value = stats.wilcoxon(scores1, scores2)
        except ValueError as e:
            return HypothesisTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=np.nan,
                p_value=1.0,
                significant=False,
                alpha=alpha,
                effect_size=None,
                interpretation=f"Test could not be performed: {e}"
            )

        significant = p_value < alpha

        # Effect size (r = Z / sqrt(N))
        n = len(scores1)
        z_stat = stats.norm.ppf(p_value / 2)
        effect_size = abs(z_stat) / np.sqrt(n)

        if not significant:
            interpretation = "No significant difference between models (non-parametric test)"
        else:
            diff_median = np.median(scores1 - scores2)
            better = "first" if diff_median > 0 else "second"
            interpretation = f"The {better} model is significantly better (non-parametric test)"

        return HypothesisTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=w_stat,
            p_value=p_value,
            significant=significant,
            alpha=alpha,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def friedman_test(
        self,
        *score_arrays,
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Perform Friedman test for comparing multiple models.

        Args:
            score_arrays: Variable number of CV score arrays (one per model)
            alpha: Significance level

        Returns:
            HypothesisTestResult object
        """
        stat, p_value = stats.friedmanchisquare(*score_arrays)
        significant = p_value < alpha

        n_models = len(score_arrays)
        n_folds = len(score_arrays[0])

        # Kendall's W effect size
        kendall_w = stat / (n_folds * (n_models - 1))

        if significant:
            interpretation = f"Significant difference among {n_models} models. Post-hoc tests recommended."
        else:
            interpretation = f"No significant difference among {n_models} models."

        return HypothesisTestResult(
            test_name="Friedman test",
            statistic=stat,
            p_value=p_value,
            significant=significant,
            alpha=alpha,
            effect_size=kendall_w,
            interpretation=interpretation
        )

    def normality_test(
        self,
        data: np.ndarray,
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Test for normality using Shapiro-Wilk test.

        Args:
            data: 1D array of data
            alpha: Significance level

        Returns:
            HypothesisTestResult object
        """
        stat, p_value = stats.shapiro(data)
        significant = p_value < alpha

        if significant:
            interpretation = "Data significantly deviates from normal distribution"
        else:
            interpretation = "Data is consistent with normal distribution"

        return HypothesisTestResult(
            test_name="Shapiro-Wilk test",
            statistic=stat,
            p_value=p_value,
            significant=significant,
            alpha=alpha,
            effect_size=None,
            interpretation=interpretation
        )

    def compute_effect_size(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        method: str = "cohens_d"
    ) -> float:
        """
        Compute effect size between two sets of scores.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            method: Effect size method ('cohens_d', 'hedges_g', 'glass_delta')

        Returns:
            Effect size value
        """
        diff = scores1 - scores2

        if method == "cohens_d":
            return diff.mean() / diff.std() if diff.std() > 0 else 0

        elif method == "hedges_g":
            # Corrected for small sample sizes
            n = len(scores1)
            cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
            correction = 1 - (3 / (4 * (2 * n - 2) - 1))
            return cohens_d * correction

        elif method == "glass_delta":
            # Uses control group (scores2) SD
            return diff.mean() / scores2.std() if scores2.std() > 0 else 0

        else:
            raise ValueError(f"Unknown method: {method}")

    def power_analysis(
        self,
        effect_size: float,
        n_samples: int,
        alpha: float = 0.05
    ) -> float:
        """
        Compute statistical power for given effect size and sample size.

        Args:
            effect_size: Expected effect size (Cohen's d)
            n_samples: Sample size (number of CV folds)
            alpha: Significance level

        Returns:
            Statistical power (0 to 1)
        """
        # For paired t-test
        df = n_samples - 1
        ncp = effect_size * np.sqrt(n_samples)  # Non-centrality parameter
        t_crit = stats.t.ppf(1 - alpha/2, df)

        # Power = P(reject H0 | H1 is true)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

        return power

    def sample_size_for_power(
        self,
        effect_size: float,
        desired_power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """
        Calculate required sample size for desired power.

        Args:
            effect_size: Expected effect size (Cohen's d)
            desired_power: Desired statistical power
            alpha: Significance level

        Returns:
            Required sample size
        """
        for n in range(2, 1000):
            if self.power_analysis(effect_size, n, alpha) >= desired_power:
                return n
        return 1000


class StatisticalReport:
    """
    Generate comprehensive statistical analysis reports.
    """

    def __init__(self):
        self.analyzer = StatisticalAnalyzer()

    def full_model_comparison(
        self,
        cv_scores_dict: Dict[str, np.ndarray],
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform full statistical comparison of models.

        Args:
            cv_scores_dict: Dict mapping model names to CV scores
            alpha: Significance level

        Returns:
            Dict with all statistical analysis results
        """
        results = {
            'normality_tests': {},
            'pairwise_tests': [],
            'effect_sizes': {},
            'confidence_intervals': {},
            'summary': {}
        }

        model_names = list(cv_scores_dict.keys())
        all_scores = list(cv_scores_dict.values())

        # Normality tests
        for name, scores in cv_scores_dict.items():
            results['normality_tests'][name] = self.analyzer.normality_test(scores, alpha)

        # Check if parametric tests are appropriate
        all_normal = all(
            not r.significant for r in results['normality_tests'].values()
        )

        # Friedman test for multiple models
        if len(model_names) > 2:
            friedman = self.analyzer.friedman_test(*all_scores, alpha=alpha)
            results['friedman_test'] = friedman

        # Pairwise comparisons
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                scores1 = cv_scores_dict[name1]
                scores2 = cv_scores_dict[name2]

                # Choose test based on normality
                if all_normal:
                    test_result = self.analyzer.paired_t_test(scores1, scores2, alpha)
                else:
                    test_result = self.analyzer.wilcoxon_test(scores1, scores2, alpha)

                # Effect size
                effect_size = self.analyzer.compute_effect_size(scores1, scores2)

                results['pairwise_tests'].append({
                    'model1': name1,
                    'model2': name2,
                    'test': test_result,
                    'effect_size': effect_size
                })

        # Confidence intervals for each model's mean CV score
        for name, scores in cv_scores_dict.items():
            ci = self.analyzer.bootstrap_ci(scores, np.mean, confidence_level=1-alpha)
            results['confidence_intervals'][name] = ci

        # Summary
        best_model = min(cv_scores_dict.items(), key=lambda x: x[1].mean())
        results['summary'] = {
            'best_model': best_model[0],
            'best_mean_score': float(best_model[1].mean()),
            'parametric_appropriate': all_normal,
            'significant_differences': sum(
                1 for p in results['pairwise_tests'] if p['test'].significant
            )
        }

        return results

    def generate_report(
        self,
        analysis_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate markdown report from analysis results.

        Args:
            analysis_results: Results from full_model_comparison
            output_path: Optional path to save report

        Returns:
            Markdown formatted report
        """
        report = """# Statistical Analysis Report

## Executive Summary

"""
        summary = analysis_results['summary']
        report += f"- **Best Model:** {summary['best_model']}\n"
        report += f"- **Best Mean CV Score:** {summary['best_mean_score']:.4f}\n"
        report += f"- **Parametric Tests Appropriate:** {'Yes' if summary['parametric_appropriate'] else 'No'}\n"
        report += f"- **Significant Pairwise Differences:** {summary['significant_differences']}\n\n"

        # Normality Tests
        report += "## Normality Tests\n\n"
        report += "| Model | Test Statistic | p-value | Normal? |\n"
        report += "|-------|----------------|---------|--------|\n"

        for name, test in analysis_results['normality_tests'].items():
            normal = "Yes" if not test.significant else "No"
            report += f"| {name} | {test.statistic:.4f} | {test.p_value:.4f} | {normal} |\n"

        # Confidence Intervals
        report += "\n## Confidence Intervals (95%)\n\n"
        report += "| Model | Mean | Lower | Upper |\n"
        report += "|-------|------|-------|-------|\n"

        for name, ci in analysis_results['confidence_intervals'].items():
            report += f"| {name} | {ci.estimate:.4f} | {ci.lower:.4f} | {ci.upper:.4f} |\n"

        # Friedman test
        if 'friedman_test' in analysis_results:
            friedman = analysis_results['friedman_test']
            report += f"\n## Friedman Test (Multiple Models)\n\n"
            report += f"- **Test Statistic:** {friedman.statistic:.4f}\n"
            report += f"- **p-value:** {friedman.p_value:.4f}\n"
            report += f"- **Significant:** {'Yes' if friedman.significant else 'No'}\n"
            report += f"- **Interpretation:** {friedman.interpretation}\n"

        # Pairwise Comparisons
        report += "\n## Pairwise Model Comparisons\n\n"
        report += "| Model 1 | Model 2 | Test | p-value | Significant | Effect Size | Interpretation |\n"
        report += "|---------|---------|------|---------|-------------|-------------|----------------|\n"

        for comparison in analysis_results['pairwise_tests']:
            test = comparison['test']
            effect = comparison['effect_size']
            sig = "Yes" if test.significant else "No"
            report += f"| {comparison['model1']} | {comparison['model2']} | {test.test_name} | "
            report += f"{test.p_value:.4f} | {sig} | {effect:.4f} | {test.interpretation} |\n"

        # Effect Size Guide
        report += """
## Effect Size Interpretation Guide

- **Small effect:** |d| < 0.2
- **Medium effect:** 0.2 <= |d| < 0.8
- **Large effect:** |d| >= 0.8

## Methodology Notes

- Normality was assessed using the Shapiro-Wilk test
- Parametric tests (paired t-test) used when all distributions were normal
- Non-parametric tests (Wilcoxon signed-rank) used otherwise
- Effect sizes reported as Cohen's d
- Confidence intervals computed using bootstrap (10,000 iterations)
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_path}")

        return report


def run_statistical_analysis(
    cv_scores_path: str = "src/models/cv_scores.json",
    output_path: str = "src/models/statistical_report.md"
) -> Dict:
    """
    Run full statistical analysis on saved CV scores.

    Args:
        cv_scores_path: Path to JSON file with CV scores
        output_path: Path to save report

    Returns:
        Analysis results dict
    """
    print("=" * 60)
    print("HEALTHCARE ANALYTICS - STATISTICAL ANALYSIS")
    print("=" * 60)

    # Load CV scores
    with open(cv_scores_path, 'r') as f:
        cv_scores_dict = {k: np.array(v) for k, v in json.load(f).items()}

    # Run analysis
    reporter = StatisticalReport()
    results = reporter.full_model_comparison(cv_scores_dict)

    # Generate report
    report = reporter.generate_report(results, output_path)

    print("\nAnalysis complete!")
    print(f"Best model: {results['summary']['best_model']}")
    print(f"Report saved to: {output_path}")

    return results


if __name__ == "__main__":
    # Example usage with dummy data
    cv_scores = {
        "Random Forest": np.array([15.2, 14.8, 15.5, 14.9, 15.1]),
        "XGBoost": np.array([14.5, 14.2, 14.8, 14.3, 14.6]),
        "Linear Regression": np.array([18.2, 17.9, 18.5, 18.1, 18.3])
    }

    reporter = StatisticalReport()
    results = reporter.full_model_comparison(cv_scores)
    report = reporter.generate_report(results, "src/models/statistical_report.md")
    print(report)
