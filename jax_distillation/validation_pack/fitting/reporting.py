"""Fit report generation for parameter estimation results.

This module generates comprehensive reports documenting the
results of parameter fitting, including fit quality metrics,
diagnostic plots, and recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import os

import numpy as np

from jax_distillation.validation_pack.fitting.parameter_estimation import (
    ParameterEstimationResult,
)
from jax_distillation.validation_pack.fitting.identifiability import (
    IdentifiabilityResult,
)


@dataclass
class FitReport:
    """Comprehensive fit report.

    Attributes:
        timestamp: Report generation time
        dataset_info: Information about the dataset used
        parameter_results: Parameter estimation results
        identifiability: Identifiability analysis results
        validation_metrics: Validation set metrics
        residual_analysis: Residual analysis results
        recommendations: Fitting recommendations
        warnings: Any warnings or issues
    """

    timestamp: str
    dataset_info: Dict[str, Any]
    parameter_results: ParameterEstimationResult
    identifiability: Optional[IdentifiabilityResult] = None
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    residual_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _analyze_residuals(
    residuals: np.ndarray,
    times: np.ndarray,
) -> Dict[str, Any]:
    """Analyze residuals for fit quality assessment.

    Args:
        residuals: Residual array [n_times, n_outputs]
        times: Time array [n_times]

    Returns:
        Dict with residual analysis metrics.
    """
    analysis = {}

    # Basic statistics
    analysis["mean"] = float(np.mean(residuals))
    analysis["std"] = float(np.std(residuals))
    analysis["max_abs"] = float(np.max(np.abs(residuals)))

    # Check for bias
    analysis["bias_significant"] = abs(analysis["mean"]) > 2 * analysis["std"] / np.sqrt(len(residuals))

    # Check for autocorrelation (Durbin-Watson statistic approximation)
    if len(residuals) > 1:
        diff = np.diff(residuals.flatten())
        dw = float(np.sum(diff ** 2) / np.sum(residuals.flatten() ** 2))
        analysis["durbin_watson"] = dw
        analysis["autocorrelation_concern"] = dw < 1.5 or dw > 2.5
    else:
        analysis["durbin_watson"] = np.nan
        analysis["autocorrelation_concern"] = False

    # Check for heteroscedasticity (simple test: compare variance in halves)
    n = len(residuals)
    if n > 10:
        var_first = np.var(residuals[:n // 2])
        var_second = np.var(residuals[n // 2:])
        var_ratio = max(var_first, var_second) / max(min(var_first, var_second), 1e-10)
        analysis["variance_ratio"] = float(var_ratio)
        analysis["heteroscedasticity_concern"] = var_ratio > 2.0
    else:
        analysis["variance_ratio"] = np.nan
        analysis["heteroscedasticity_concern"] = False

    return analysis


def generate_fit_report(
    parameter_results: ParameterEstimationResult,
    identifiability: Optional[IdentifiabilityResult] = None,
    measurements: Optional[np.ndarray] = None,
    simulated: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    dataset_name: str = "Unknown",
    n_samples: Optional[int] = None,
) -> FitReport:
    """Generate comprehensive fit report.

    Args:
        parameter_results: Parameter estimation results
        identifiability: Identifiability analysis results
        measurements: Measured data (for residual analysis)
        simulated: Simulated data (for residual analysis)
        times: Time array
        dataset_name: Name of dataset used
        n_samples: Number of samples in dataset

    Returns:
        FitReport with all analysis results.
    """
    timestamp = datetime.now().isoformat()

    # Dataset info
    dataset_info = {
        "name": dataset_name,
        "n_samples": n_samples or (len(times) if times is not None else 0),
        "n_parameters": len(parameter_results.fitted_params),
    }

    # Residual analysis
    residual_analysis = {}
    if measurements is not None and simulated is not None:
        residuals = measurements - simulated
        residual_analysis = _analyze_residuals(
            residuals, times if times is not None else np.arange(len(residuals))
        )

    # Validation metrics
    validation_metrics = {}
    if parameter_results.cross_validation:
        cv = parameter_results.cross_validation
        validation_metrics = {
            "cv_mean_val_loss": cv["mean_val_loss"],
            "cv_std_val_loss": cv["std_val_loss"],
            "cv_train_val_ratio": cv["mean_train_loss"] / cv["mean_val_loss"]
            if cv["mean_val_loss"] > 0 else np.nan,
        }

    # Generate recommendations
    recommendations = []
    warnings = []

    # Check fit quality
    if parameter_results.loss_reduction < 0.1:
        warnings.append("Limited loss reduction - parameters may be near optimal or data may not support fitting")

    if parameter_results.fit_metrics.get("r_squared", 0) < 0.7:
        recommendations.append("Low R² - consider adding parameters or checking model structure")

    # Check identifiability
    if identifiability:
        if len(identifiability.problematic_params) > 0:
            warnings.append(
                f"Potentially unidentifiable parameters: {identifiability.problematic_params}"
            )
            recommendations.append("Consider fixing unidentifiable parameters or adding measurements")

        if identifiability.condition_number > 1000:
            warnings.append(f"High condition number ({identifiability.condition_number:.0f})")
            recommendations.append("Parameter estimates may be sensitive to noise")

    # Check residuals
    if residual_analysis.get("bias_significant"):
        warnings.append("Significant residual bias detected")
        recommendations.append("Check for systematic model error")

    if residual_analysis.get("autocorrelation_concern"):
        warnings.append("Residual autocorrelation detected")
        recommendations.append("Model may be missing dynamics")

    if residual_analysis.get("heteroscedasticity_concern"):
        warnings.append("Non-constant residual variance detected")
        recommendations.append("Consider weighted fitting or model refinement")

    return FitReport(
        timestamp=timestamp,
        dataset_info=dataset_info,
        parameter_results=parameter_results,
        identifiability=identifiability,
        validation_metrics=validation_metrics,
        residual_analysis=residual_analysis,
        recommendations=recommendations,
        warnings=warnings,
    )


def print_fit_report(report: FitReport) -> None:
    """Print a formatted fit report.

    Args:
        report: FitReport from generate_fit_report.
    """
    print("=" * 70)
    print("PARAMETER FIT REPORT")
    print("=" * 70)
    print(f"Generated: {report.timestamp}")

    print("\n--- Dataset Information ---")
    for key, value in report.dataset_info.items():
        print(f"  {key}: {value}")

    print("\n--- Fit Results ---")
    pr = report.parameter_results
    print(f"  Converged: {pr.converged}")
    print(f"  Loss reduction: {pr.loss_reduction * 100:.1f}%")
    print(f"  R²: {pr.fit_metrics.get('r_squared', np.nan):.4f}")
    print(f"  RMSE: {pr.fit_metrics.get('rmse', np.nan):.6f}")

    print("\n  Fitted Parameters:")
    for name, value in pr.fitted_params.items():
        initial = pr.initial_params[name]
        print(f"    {name}: {initial:.4f} → {value:.4f}")

    if report.validation_metrics:
        print("\n--- Cross-Validation ---")
        for key, value in report.validation_metrics.items():
            print(f"  {key}: {value:.6f}")

    if report.residual_analysis:
        print("\n--- Residual Analysis ---")
        ra = report.residual_analysis
        print(f"  Mean: {ra.get('mean', np.nan):.6f}")
        print(f"  Std:  {ra.get('std', np.nan):.6f}")
        print(f"  Durbin-Watson: {ra.get('durbin_watson', np.nan):.3f}")

    if report.warnings:
        print("\n--- Warnings ---")
        for w in report.warnings:
            print(f"  ⚠ {w}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for r in report.recommendations:
            print(f"  → {r}")

    print("=" * 70)


def save_fit_report_markdown(report: FitReport, output_path: str) -> None:
    """Save fit report as Markdown file.

    Args:
        report: FitReport to save.
        output_path: Path to output Markdown file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = []
    lines.append("# Parameter Fit Report")
    lines.append(f"\nGenerated: {report.timestamp}")

    lines.append("\n## Dataset Information")
    for key, value in report.dataset_info.items():
        lines.append(f"- **{key}**: {value}")

    lines.append("\n## Fit Results")
    pr = report.parameter_results
    lines.append(f"- **Converged**: {pr.converged}")
    lines.append(f"- **Loss reduction**: {pr.loss_reduction * 100:.1f}%")
    lines.append(f"- **R²**: {pr.fit_metrics.get('r_squared', np.nan):.4f}")
    lines.append(f"- **RMSE**: {pr.fit_metrics.get('rmse', np.nan):.6f}")

    lines.append("\n### Fitted Parameters")
    lines.append("| Parameter | Initial | Fitted | Change |")
    lines.append("|-----------|---------|--------|--------|")
    for name, value in pr.fitted_params.items():
        initial = pr.initial_params[name]
        change = (value - initial) / initial * 100 if initial != 0 else 0
        lines.append(f"| {name} | {initial:.4f} | {value:.4f} | {change:+.1f}% |")

    if report.warnings:
        lines.append("\n## Warnings")
        for w in report.warnings:
            lines.append(f"- ⚠️ {w}")

    if report.recommendations:
        lines.append("\n## Recommendations")
        for r in report.recommendations:
            lines.append(f"- {r}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
