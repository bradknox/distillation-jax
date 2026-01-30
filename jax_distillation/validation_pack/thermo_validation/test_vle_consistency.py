"""Validate VLE calculation consistency.

This module tests that the simulator's VLE calculations are internally
consistent and produce physically reasonable results.

Validation includes:
- Composition bounds: x, y ∈ [0, 1]
- Monotonicity: T_bubble decreases with increasing light component
- Activity coefficient limits: gamma → 1 at pure component limits
- K-value physical bounds
- Relative volatility > 1 for light component
"""

from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from jax_distillation.core.thermodynamics import (
    bubble_point_temperature,
    equilibrium_vapor_composition,
    relative_volatility,
    nrtl_activity_coefficients,
    compute_k_values,
    create_methanol_water_thermo,
    create_benzene_toluene_thermo,
    ThermoParams,
)


@dataclass
class VLEConsistencyResult:
    """Result of VLE consistency validation.

    Attributes:
        mixture: Binary mixture name
        composition_bounds_ok: True if all x, y ∈ [0, 1]
        monotonicity_ok: True if T_bubble decreases with x
        activity_limits_ok: True if gamma → 1 at pure limits
        k_values_ok: True if K-values are positive and finite
        volatility_ok: True if alpha > 1 for light component
        all_passed: True if all checks passed
        details: List of (check_name, passed, message) tuples
    """

    mixture: str
    composition_bounds_ok: bool
    monotonicity_ok: bool
    activity_limits_ok: bool
    k_values_ok: bool
    volatility_ok: bool
    all_passed: bool
    details: List[Tuple[str, bool, str]]


def _check_composition_bounds(
    thermo: ThermoParams,
    P: float = 1.0,
    n_points: int = 21,
) -> Tuple[bool, str]:
    """Check that y* is always in [0, 1] for valid x."""
    x_values = np.linspace(0.0, 1.0, n_points)
    all_valid = True
    violations = []

    for x in x_values:
        x_jax = jnp.array(x)
        P_jax = jnp.array(P)

        # Get bubble point temperature
        T = bubble_point_temperature(x_jax, P_jax, thermo)

        # Get equilibrium vapor composition
        y = float(equilibrium_vapor_composition(x_jax, T, P_jax, thermo))

        if y < 0.0 or y > 1.0:
            all_valid = False
            violations.append(f"x={x:.3f}: y={y:.6f}")

    if all_valid:
        return True, f"All {n_points} points have y ∈ [0, 1]"
    else:
        return False, f"Violations: {', '.join(violations[:5])}"


def _check_monotonicity(
    thermo: ThermoParams,
    P: float = 1.0,
    n_points: int = 21,
) -> Tuple[bool, str]:
    """Check that T_bubble decreases monotonically with increasing x (light)."""
    x_values = np.linspace(0.0, 1.0, n_points)
    T_values = []

    for x in x_values:
        x_jax = jnp.array(x)
        P_jax = jnp.array(P)
        T = float(bubble_point_temperature(x_jax, P_jax, thermo))
        T_values.append(T)

    # Check monotonicity: each T should be >= next T (decreasing)
    # Allow small numerical tolerance
    tol = 0.1  # K
    monotonic = True
    violations = []

    for i in range(len(T_values) - 1):
        if T_values[i + 1] > T_values[i] + tol:
            monotonic = False
            violations.append(
                f"x={x_values[i]:.2f}→{x_values[i+1]:.2f}: "
                f"T={T_values[i]:.1f}→{T_values[i+1]:.1f} K"
            )

    if monotonic:
        return True, f"T_bubble monotonically decreasing: {T_values[0]:.1f}→{T_values[-1]:.1f} K"
    else:
        return False, f"Non-monotonic: {', '.join(violations[:3])}"


def _check_activity_limits(
    thermo: ThermoParams,
    T: float = 340.0,
    tol: float = 0.01,
) -> Tuple[bool, str]:
    """Check that activity coefficients approach 1 at pure component limits."""
    if thermo.nrtl is None:
        return True, "Ideal mixture (gamma = 1 everywhere)"

    T_jax = jnp.array(T)

    # At x = 0 (pure heavy component)
    gamma_1_at_0, gamma_2_at_0 = nrtl_activity_coefficients(jnp.array(0.0), T_jax, thermo.nrtl)

    # At x = 1 (pure light component)
    gamma_1_at_1, gamma_2_at_1 = nrtl_activity_coefficients(jnp.array(1.0), T_jax, thermo.nrtl)

    # Check limits: gamma_2(x=0) → 1, gamma_1(x=1) → 1
    g2_at_0 = float(gamma_2_at_0)
    g1_at_1 = float(gamma_1_at_1)

    ok = abs(g2_at_0 - 1.0) < tol and abs(g1_at_1 - 1.0) < tol

    msg = f"gamma_2(x=0)={g2_at_0:.4f}, gamma_1(x=1)={g1_at_1:.4f}"
    if ok:
        msg += " (within tolerance)"
    else:
        msg += f" (tol={tol})"

    return ok, msg


def _check_k_values(
    thermo: ThermoParams,
    P: float = 1.0,
    n_points: int = 11,
) -> Tuple[bool, str]:
    """Check that K-values are positive and finite."""
    x_values = np.linspace(0.0, 1.0, n_points)
    all_valid = True
    issues = []

    for x in x_values:
        x_jax = jnp.array(x)
        P_jax = jnp.array(P)
        T = bubble_point_temperature(x_jax, P_jax, thermo)

        K1, K2 = compute_k_values(x_jax, T, P_jax, thermo)
        K1, K2 = float(K1), float(K2)

        if not (np.isfinite(K1) and K1 > 0):
            all_valid = False
            issues.append(f"K1={K1:.4f} at x={x:.2f}")

        if not (np.isfinite(K2) and K2 > 0):
            all_valid = False
            issues.append(f"K2={K2:.4f} at x={x:.2f}")

    if all_valid:
        return True, f"All K-values positive and finite across {n_points} points"
    else:
        return False, f"Issues: {', '.join(issues[:5])}"


def _check_relative_volatility(
    thermo: ThermoParams,
    P: float = 1.0,
    n_points: int = 11,
) -> Tuple[bool, str]:
    """Check that relative volatility alpha > 1 (light component is more volatile)."""
    x_values = np.linspace(0.01, 0.99, n_points)  # Avoid pure limits
    all_valid = True
    min_alpha = float("inf")
    max_alpha = 0.0
    issues = []

    for x in x_values:
        x_jax = jnp.array(x)
        P_jax = jnp.array(P)
        T = bubble_point_temperature(x_jax, P_jax, thermo)

        alpha = float(relative_volatility(x_jax, T, P_jax, thermo))

        min_alpha = min(min_alpha, alpha)
        max_alpha = max(max_alpha, alpha)

        if alpha <= 1.0:
            all_valid = False
            issues.append(f"alpha={alpha:.3f} at x={x:.2f}")

    if all_valid:
        return True, f"alpha ∈ [{min_alpha:.2f}, {max_alpha:.2f}] > 1"
    else:
        return False, f"alpha <= 1: {', '.join(issues[:5])}"


def validate_vle_consistency_single_mixture(
    mixture: str,
    P: float = 1.0,
) -> VLEConsistencyResult:
    """Validate VLE consistency for a binary mixture.

    Args:
        mixture: Mixture name (e.g., "methanol-water")
        P: System pressure [bar]

    Returns:
        VLEConsistencyResult with all check results.
    """
    params_map = {
        "methanol-water": create_methanol_water_thermo,
        "benzene-toluene": create_benzene_toluene_thermo,
    }

    key = mixture.lower().replace(" ", "-")
    if key not in params_map:
        raise ValueError(f"Unknown mixture: {mixture}")

    thermo = params_map[key]()

    details = []

    # Run all checks
    bounds_ok, bounds_msg = _check_composition_bounds(thermo, P)
    details.append(("Composition bounds", bounds_ok, bounds_msg))

    mono_ok, mono_msg = _check_monotonicity(thermo, P)
    details.append(("Monotonicity", mono_ok, mono_msg))

    activity_ok, activity_msg = _check_activity_limits(thermo)
    details.append(("Activity limits", activity_ok, activity_msg))

    k_ok, k_msg = _check_k_values(thermo, P)
    details.append(("K-values", k_ok, k_msg))

    vol_ok, vol_msg = _check_relative_volatility(thermo, P)
    details.append(("Relative volatility", vol_ok, vol_msg))

    all_passed = bounds_ok and mono_ok and activity_ok and k_ok and vol_ok

    return VLEConsistencyResult(
        mixture=mixture,
        composition_bounds_ok=bounds_ok,
        monotonicity_ok=mono_ok,
        activity_limits_ok=activity_ok,
        k_values_ok=k_ok,
        volatility_ok=vol_ok,
        all_passed=all_passed,
        details=details,
    )


def validate_vle_consistency(
    mixtures: List[str] = None,
    P: float = 1.0,
) -> dict:
    """Validate VLE consistency for multiple mixtures.

    Args:
        mixtures: List of mixture names. If None, validates all available.
        P: System pressure [bar]

    Returns:
        Dict mapping mixture names to VLEConsistencyResult.
    """
    if mixtures is None:
        mixtures = ["methanol-water", "benzene-toluene"]

    results = {}
    for mixture in mixtures:
        try:
            results[mixture] = validate_vle_consistency_single_mixture(mixture, P)
        except ValueError as e:
            print(f"Warning: Skipping {mixture}: {e}")

    return results


def print_vle_consistency_report(results: dict) -> None:
    """Print a formatted VLE consistency report.

    Args:
        results: Dict of validation results from validate_vle_consistency.
    """
    print("=" * 70)
    print("VLE CONSISTENCY VALIDATION")
    print("=" * 70)

    all_passed = True

    for mixture, result in results.items():
        status = "PASS" if result.all_passed else "FAIL"
        all_passed = all_passed and result.all_passed

        print(f"\n{mixture.upper()}")
        print("-" * 50)

        for check_name, passed, message in result.details:
            check_status = "✓" if passed else "✗"
            print(f"  [{check_status}] {check_name}: {message}")

    print("\n" + "=" * 70)
    overall = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"OVERALL: {overall}")
    print("=" * 70)


def run_vle_consistency_validation() -> bool:
    """Run full VLE consistency validation and print report.

    Returns:
        True if all checks passed.
    """
    results = validate_vle_consistency()
    print_vle_consistency_report(results)
    return all(r.all_passed for r in results.values())


if __name__ == "__main__":
    success = run_vle_consistency_validation()
    exit(0 if success else 1)
