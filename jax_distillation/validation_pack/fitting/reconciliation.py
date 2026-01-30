"""Data reconciliation for mass and energy balance closure.

This module provides tools for reconciling plant measurements
to satisfy mass and energy balance constraints, accounting for
measurement uncertainty.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class ReconciliationResult:
    """Result of measurement reconciliation.

    Attributes:
        original_values: Original measured values
        reconciled_values: Adjusted values satisfying constraints
        adjustments: Difference between reconciled and original
        constraint_residuals: Residuals of constraint equations
        success: True if reconciliation converged
        mass_balance_error: Residual mass balance error
        message: Status message
    """

    original_values: Dict[str, float]
    reconciled_values: Dict[str, float]
    adjustments: Dict[str, float]
    constraint_residuals: List[float]
    success: bool
    mass_balance_error: float
    message: str


def reconcile_mass_balance(
    F: float,
    D: float,
    B: float,
    z_F: float,
    x_D: float,
    x_B: float,
    uncertainties: Optional[Dict[str, float]] = None,
) -> ReconciliationResult:
    """Reconcile measurements to satisfy mass balance constraints.

    Constraints:
    1. Total mass: F = D + B
    2. Component: F * z_F = D * x_D + B * x_B

    Uses weighted least squares to adjust measurements within
    their uncertainty bounds to satisfy constraints.

    Args:
        F: Feed flow rate [mol/s]
        D: Distillate flow rate [mol/s]
        B: Bottoms flow rate [mol/s]
        z_F: Feed composition (light component)
        x_D: Distillate composition
        x_B: Bottoms composition
        uncertainties: Dict of measurement uncertainties (std dev).
            Keys: "F", "D", "B", "z_F", "x_D", "x_B"

    Returns:
        ReconciliationResult with reconciled values.
    """
    # Default uncertainties (as fraction of measured value or absolute)
    if uncertainties is None:
        uncertainties = {
            "F": 0.02 * F,
            "D": 0.02 * D,
            "B": 0.02 * B,
            "z_F": 0.01,
            "x_D": 0.01,
            "x_B": 0.01,
        }

    # Original values
    x0 = np.array([F, D, B, z_F, x_D, x_B])
    original = {"F": F, "D": D, "B": B, "z_F": z_F, "x_D": x_D, "x_B": x_B}

    # Weights (inverse variance)
    sigma = np.array([
        uncertainties.get("F", 0.02 * F),
        uncertainties.get("D", 0.02 * D),
        uncertainties.get("B", 0.02 * B),
        uncertainties.get("z_F", 0.01),
        uncertainties.get("x_D", 0.01),
        uncertainties.get("x_B", 0.01),
    ])
    sigma = np.maximum(sigma, 1e-6)  # Avoid division by zero
    weights = 1.0 / sigma**2

    def objective(x):
        """Weighted sum of squared adjustments."""
        return np.sum(weights * (x - x0) ** 2)

    def mass_constraint(x):
        """Total mass balance: F - D - B = 0"""
        return x[0] - x[1] - x[2]

    def component_constraint(x):
        """Component balance: F*z_F - D*x_D - B*x_B = 0"""
        return x[0] * x[3] - x[1] * x[4] - x[2] * x[5]

    # Bounds (compositions must be in [0, 1], flows must be positive)
    bounds = [
        (0, None),  # F
        (0, None),  # D
        (0, None),  # B
        (0, 1),     # z_F
        (0, 1),     # x_D
        (0, 1),     # x_B
    ]

    # Constraints
    constraints = [
        {"type": "eq", "fun": mass_constraint},
        {"type": "eq", "fun": component_constraint},
    ]

    # Solve
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 100},
    )

    x_opt = result.x

    reconciled = {
        "F": x_opt[0],
        "D": x_opt[1],
        "B": x_opt[2],
        "z_F": x_opt[3],
        "x_D": x_opt[4],
        "x_B": x_opt[5],
    }

    adjustments = {k: reconciled[k] - original[k] for k in original}

    # Residuals
    mass_residual = mass_constraint(x_opt)
    comp_residual = component_constraint(x_opt)
    mass_balance_error = abs(mass_residual) / max(x_opt[0], 1e-6)

    return ReconciliationResult(
        original_values=original,
        reconciled_values=reconciled,
        adjustments=adjustments,
        constraint_residuals=[mass_residual, comp_residual],
        success=result.success,
        mass_balance_error=mass_balance_error,
        message=result.message,
    )


def reconcile_measurements(
    measurements: Dict[str, float],
    uncertainties: Dict[str, float],
    constraint_matrix: np.ndarray,
    constraint_rhs: np.ndarray,
) -> ReconciliationResult:
    """General measurement reconciliation with linear constraints.

    Solves: min (x - x_meas)' W (x - x_meas)
            s.t. A x = b

    Args:
        measurements: Dict of measured values.
        uncertainties: Dict of measurement uncertainties (std dev).
        constraint_matrix: A matrix (n_constraints x n_measurements).
        constraint_rhs: b vector (n_constraints,).

    Returns:
        ReconciliationResult with reconciled values.
    """
    keys = list(measurements.keys())
    x0 = np.array([measurements[k] for k in keys])
    sigma = np.array([uncertainties.get(k, 0.01 * abs(x0[i]) + 1e-6)
                      for i, k in enumerate(keys)])
    sigma = np.maximum(sigma, 1e-6)
    W = np.diag(1.0 / sigma**2)

    # Analytical solution for linear constraints:
    # x_rec = x_meas - W^{-1} A' (A W^{-1} A')^{-1} (A x_meas - b)

    A = constraint_matrix
    b = constraint_rhs

    W_inv = np.diag(sigma**2)
    AW_inv = A @ W_inv
    AW_invAt = AW_inv @ A.T

    try:
        AW_invAt_inv = np.linalg.inv(AW_invAt)
        residual = A @ x0 - b
        correction = W_inv @ A.T @ AW_invAt_inv @ residual
        x_rec = x0 - correction
        success = True
        message = "Converged"
    except np.linalg.LinAlgError:
        # Fall back to optimization
        x_rec = x0
        success = False
        message = "Failed to solve"

    reconciled = {k: x_rec[i] for i, k in enumerate(keys)}
    original = {k: x0[i] for i, k in enumerate(keys)}
    adjustments = {k: reconciled[k] - original[k] for k in keys}

    residuals = list(A @ x_rec - b)
    mass_error = np.max(np.abs(residuals)) if residuals else 0.0

    return ReconciliationResult(
        original_values=original,
        reconciled_values=reconciled,
        adjustments=adjustments,
        constraint_residuals=residuals,
        success=success,
        mass_balance_error=mass_error,
        message=message,
    )


def print_reconciliation_report(result: ReconciliationResult) -> None:
    """Print a formatted reconciliation report.

    Args:
        result: ReconciliationResult from reconciliation function.
    """
    print("=" * 60)
    print("MEASUREMENT RECONCILIATION REPORT")
    print("=" * 60)

    print(f"\nStatus: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Message: {result.message}")

    print("\nOriginal vs Reconciled Values:")
    print(f"{'Variable':<10} {'Original':>12} {'Reconciled':>12} {'Adjustment':>12}")
    print("-" * 48)
    for key in result.original_values:
        orig = result.original_values[key]
        rec = result.reconciled_values[key]
        adj = result.adjustments[key]
        print(f"{key:<10} {orig:>12.6f} {rec:>12.6f} {adj:>+12.6f}")

    print(f"\nMass balance error: {result.mass_balance_error:.2e}")
    print(f"Constraint residuals: {result.constraint_residuals}")
    print("=" * 60)
