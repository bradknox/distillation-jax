"""Identifiability analysis for parameter estimation.

This module provides tools for analyzing whether parameters
can be uniquely estimated from available measurements.

Methods:
- Sensitivity analysis
- Fisher information matrix
- Collinearity analysis
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IdentifiabilityResult:
    """Result of identifiability analysis.

    Attributes:
        parameter_names: List of parameter names analyzed
        sensitivity_matrix: Sensitivity of outputs to parameters [n_out, n_params]
        sensitivity_norms: L2 norm of sensitivity for each parameter
        fisher_information: Fisher information matrix [n_params, n_params]
        condition_number: Condition number of Fisher matrix
        collinearity_indices: Collinearity index for each parameter
        identifiable_params: List of parameters that appear identifiable
        problematic_params: List of parameters that may be unidentifiable
        recommendations: Analysis recommendations
    """

    parameter_names: List[str]
    sensitivity_matrix: np.ndarray
    sensitivity_norms: Dict[str, float]
    fisher_information: np.ndarray
    condition_number: float
    collinearity_indices: Dict[str, float]
    identifiable_params: List[str]
    problematic_params: List[str]
    recommendations: List[str]


def compute_sensitivity(
    simulator: Callable,
    parameters: Dict[str, float],
    inputs: np.ndarray,
    times: np.ndarray,
    perturbation: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute sensitivity of outputs to parameters.

    Uses central finite differences:
    dY/dp ≈ (Y(p+δ) - Y(p-δ)) / (2δ)

    Args:
        simulator: Function (params, inputs, times) -> outputs
        parameters: Nominal parameter values
        inputs: Input trajectory
        times: Time points
        perturbation: Relative perturbation size

    Returns:
        Tuple of (sensitivity_matrix, sensitivity_dict)
        sensitivity_matrix: [n_times*n_outputs, n_params]
        sensitivity_dict: {param_name: sensitivity_array}
    """
    # Nominal output
    y_nom = simulator(parameters, inputs, times)
    n_times = len(times)
    n_outputs = y_nom.shape[1] if y_nom.ndim > 1 else 1
    n_params = len(parameters)

    param_names = list(parameters.keys())
    sensitivity_dict = {}
    sensitivity_matrix = np.zeros((n_times * n_outputs, n_params))

    for i, name in enumerate(param_names):
        p_val = parameters[name]
        delta = max(abs(p_val * perturbation), 1e-8)

        # Forward perturbation
        params_plus = parameters.copy()
        params_plus[name] = p_val + delta
        y_plus = simulator(params_plus, inputs, times)

        # Backward perturbation
        params_minus = parameters.copy()
        params_minus[name] = p_val - delta
        y_minus = simulator(params_minus, inputs, times)

        # Central difference
        sensitivity = (y_plus - y_minus) / (2 * delta)
        sensitivity_dict[name] = sensitivity

        # Flatten for matrix
        sensitivity_matrix[:, i] = sensitivity.flatten()

    return sensitivity_matrix, sensitivity_dict


def compute_fisher_information(
    sensitivity_matrix: np.ndarray,
    measurement_covariance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Fisher Information Matrix.

    FIM = S' * Σ^{-1} * S

    where S is the sensitivity matrix and Σ is measurement covariance.

    Args:
        sensitivity_matrix: [n_measurements, n_params]
        measurement_covariance: [n_measurements, n_measurements]
            If None, assumes identity (equal weights)

    Returns:
        Fisher Information Matrix [n_params, n_params]
    """
    S = sensitivity_matrix

    if measurement_covariance is None:
        # Identity covariance
        FIM = S.T @ S
    else:
        try:
            Sigma_inv = np.linalg.inv(measurement_covariance)
            FIM = S.T @ Sigma_inv @ S
        except np.linalg.LinAlgError:
            # Use pseudoinverse
            Sigma_inv = np.linalg.pinv(measurement_covariance)
            FIM = S.T @ Sigma_inv @ S

    return FIM


def compute_collinearity_indices(
    sensitivity_matrix: np.ndarray,
) -> Dict[int, float]:
    """Compute collinearity indices for parameter subsets.

    High collinearity index indicates parameters that are hard
    to distinguish from each other.

    Args:
        sensitivity_matrix: [n_measurements, n_params]

    Returns:
        Dict mapping parameter index to collinearity index
    """
    n_params = sensitivity_matrix.shape[1]
    indices = {}

    # Normalize columns
    S_norm = sensitivity_matrix / np.linalg.norm(sensitivity_matrix, axis=0, keepdims=True)

    # SVD
    try:
        U, sigma, Vt = np.linalg.svd(S_norm, full_matrices=False)

        # Collinearity index based on smallest singular value contribution
        for i in range(n_params):
            # How much does this parameter contribute to the smallest SV?
            indices[i] = abs(Vt[-1, i])

    except np.linalg.LinAlgError:
        for i in range(n_params):
            indices[i] = 0.0

    return indices


def analyze_identifiability(
    simulator: Callable,
    parameters: Dict[str, float],
    inputs: np.ndarray,
    times: np.ndarray,
    perturbation: float = 0.01,
    sensitivity_threshold: float = 0.01,
    condition_threshold: float = 1000,
    collinearity_threshold: float = 0.9,
) -> IdentifiabilityResult:
    """Perform complete identifiability analysis.

    Args:
        simulator: Simulation function
        parameters: Nominal parameter values
        inputs: Input trajectory
        times: Time points
        perturbation: Perturbation for sensitivity calculation
        sensitivity_threshold: Minimum sensitivity norm for identifiability
        condition_threshold: Maximum condition number for well-posed problem
        collinearity_threshold: Maximum collinearity for distinguishable params

    Returns:
        IdentifiabilityResult with analysis results.
    """
    param_names = list(parameters.keys())
    n_params = len(param_names)

    # Compute sensitivities
    sensitivity_matrix, sensitivity_dict = compute_sensitivity(
        simulator, parameters, inputs, times, perturbation
    )

    # Sensitivity norms
    sensitivity_norms = {}
    for i, name in enumerate(param_names):
        sensitivity_norms[name] = float(np.linalg.norm(sensitivity_matrix[:, i]))

    # Normalize sensitivity norms
    max_norm = max(sensitivity_norms.values()) if sensitivity_norms else 1.0
    sensitivity_norms_normalized = {
        k: v / max_norm for k, v in sensitivity_norms.items()
    }

    # Fisher Information Matrix
    FIM = compute_fisher_information(sensitivity_matrix)

    # Condition number
    try:
        eigenvalues = np.linalg.eigvalsh(FIM)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            condition_number = float(eigenvalues.max() / eigenvalues.min())
        else:
            condition_number = float("inf")
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    # Collinearity indices
    collinearity_idx = compute_collinearity_indices(sensitivity_matrix)
    collinearity_indices = {
        param_names[i]: float(collinearity_idx[i]) for i in range(n_params)
    }

    # Classify parameters
    identifiable = []
    problematic = []
    recommendations = []

    for name in param_names:
        issues = []

        # Check sensitivity
        if sensitivity_norms_normalized[name] < sensitivity_threshold:
            issues.append(f"low sensitivity ({sensitivity_norms_normalized[name]:.4f})")

        # Check collinearity
        if collinearity_indices[name] > collinearity_threshold:
            issues.append(f"high collinearity ({collinearity_indices[name]:.4f})")

        if issues:
            problematic.append(name)
            recommendations.append(f"{name}: " + ", ".join(issues))
        else:
            identifiable.append(name)

    # Overall condition number check
    if condition_number > condition_threshold:
        recommendations.append(
            f"High condition number ({condition_number:.1f}) suggests ill-conditioning"
        )

    return IdentifiabilityResult(
        parameter_names=param_names,
        sensitivity_matrix=sensitivity_matrix,
        sensitivity_norms=sensitivity_norms,
        fisher_information=FIM,
        condition_number=condition_number,
        collinearity_indices=collinearity_indices,
        identifiable_params=identifiable,
        problematic_params=problematic,
        recommendations=recommendations,
    )


def print_identifiability_report(result: IdentifiabilityResult) -> None:
    """Print a formatted identifiability analysis report.

    Args:
        result: IdentifiabilityResult from analyze_identifiability.
    """
    print("=" * 60)
    print("PARAMETER IDENTIFIABILITY ANALYSIS")
    print("=" * 60)

    print(f"\nCondition number: {result.condition_number:.2f}")
    status = "OK" if result.condition_number < 1000 else "HIGH (potential issues)"
    print(f"  Status: {status}")

    print("\nSensitivity Analysis:")
    print(f"{'Parameter':<20} {'Sensitivity':>12} {'Collinearity':>12}")
    print("-" * 44)
    for name in result.parameter_names:
        sens = result.sensitivity_norms[name]
        coll = result.collinearity_indices[name]
        print(f"{name:<20} {sens:>12.4f} {coll:>12.4f}")

    print(f"\nIdentifiable parameters: {len(result.identifiable_params)}")
    for name in result.identifiable_params:
        print(f"  ✓ {name}")

    print(f"\nProblematic parameters: {len(result.problematic_params)}")
    for name in result.problematic_params:
        print(f"  ✗ {name}")

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

    print("=" * 60)
