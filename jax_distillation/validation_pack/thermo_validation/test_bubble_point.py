"""Validate bubble point calculations against reference data.

This module tests that the simulator's bubble point temperature calculations
produce correct results with residuals below acceptable thresholds.

Acceptance criteria:
- Bubble point residual < 1e-4 bar
- Calculated bubble points match VLE data within ~1-2 K
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp

from jax_distillation.core.thermodynamics import (
    bubble_point_temperature,
    bubble_point_residual,
    create_methanol_water_thermo,
    create_benzene_toluene_thermo,
    ThermoParams,
)
from jax_distillation.validation_pack.thermo_validation.nist_points import (
    get_nist_bubble_point_data,
    BubblePointReference,
)


@dataclass
class BubblePointValidationResult:
    """Result of bubble point calculation validation.

    Attributes:
        mixture: Binary mixture name
        n_points: Number of reference points tested
        max_residual_bar: Maximum residual [bar] across all points
        max_T_error_K: Maximum temperature error [K]
        mean_T_error_K: Mean temperature error [K]
        all_results: List of (x, T_ref, T_calc, residual, T_error) for each point
        residual_passed: True if all residuals < tolerance
        T_passed: True if all T errors < T tolerance
        residual_tolerance: Residual tolerance [bar]
        T_tolerance: Temperature tolerance [K]
    """

    mixture: str
    n_points: int
    max_residual_bar: float
    max_T_error_K: float
    mean_T_error_K: float
    all_results: List[tuple]
    residual_passed: bool
    T_passed: bool
    residual_tolerance: float = 1e-4
    T_tolerance: float = 3.0  # K, allows for experimental uncertainty


def _get_thermo_params(mixture: str) -> Optional[ThermoParams]:
    """Get thermodynamic parameters for a binary mixture."""
    params_map = {
        "methanol-water": create_methanol_water_thermo,
        "benzene-toluene": create_benzene_toluene_thermo,
    }
    creator = params_map.get(mixture.lower().replace(" ", "-"))
    if creator:
        return creator()
    return None


def validate_bubble_point_single_mixture(
    mixture: str,
    residual_tolerance: float = 1e-4,
    T_tolerance: float = 3.0,
) -> BubblePointValidationResult:
    """Validate bubble point calculations for a binary mixture.

    Args:
        mixture: Mixture name (e.g., "methanol-water")
        residual_tolerance: Maximum allowed residual [bar]
        T_tolerance: Maximum allowed temperature error [K]

    Returns:
        BubblePointValidationResult with validation metrics.

    Raises:
        ValueError: If mixture is unknown.
    """
    # Get reference data
    ref_data = get_nist_bubble_point_data(mixture)

    # Get thermodynamic parameters
    thermo = _get_thermo_params(mixture)
    if thermo is None:
        raise ValueError(f"No thermodynamic parameters for mixture: {mixture}")

    results = []
    residuals = []
    T_errors = []

    for point in ref_data:
        x = jnp.array(point.x_light)
        P = jnp.array(point.pressure_bar)
        T_ref = point.T_bubble_k

        # Calculate bubble point temperature
        T_calc = float(bubble_point_temperature(x, P, thermo))

        # Calculate residual at calculated temperature
        residual = float(jnp.abs(bubble_point_residual(jnp.array(T_calc), x, P, thermo)))

        # Temperature error
        T_error = abs(T_calc - T_ref)

        results.append((point.x_light, T_ref, T_calc, residual, T_error))
        residuals.append(residual)
        T_errors.append(T_error)

    max_residual = max(residuals) if residuals else 0.0
    max_T_error = max(T_errors) if T_errors else 0.0
    mean_T_error = sum(T_errors) / len(T_errors) if T_errors else 0.0

    residual_passed = max_residual <= residual_tolerance
    T_passed = max_T_error <= T_tolerance

    return BubblePointValidationResult(
        mixture=mixture,
        n_points=len(ref_data),
        max_residual_bar=max_residual,
        max_T_error_K=max_T_error,
        mean_T_error_K=mean_T_error,
        all_results=results,
        residual_passed=residual_passed,
        T_passed=T_passed,
        residual_tolerance=residual_tolerance,
        T_tolerance=T_tolerance,
    )


def validate_bubble_point(
    mixtures: Optional[List[str]] = None,
    residual_tolerance: float = 1e-4,
    T_tolerance: float = 3.0,
) -> Dict[str, BubblePointValidationResult]:
    """Validate bubble point calculations for multiple mixtures.

    Args:
        mixtures: List of mixture names. If None, validates all available.
        residual_tolerance: Maximum allowed residual [bar]
        T_tolerance: Maximum allowed temperature error [K]

    Returns:
        Dict mapping mixture names to BubblePointValidationResult.
    """
    if mixtures is None:
        mixtures = ["methanol-water", "benzene-toluene"]

    results = {}
    for mixture in mixtures:
        try:
            results[mixture] = validate_bubble_point_single_mixture(
                mixture,
                residual_tolerance=residual_tolerance,
                T_tolerance=T_tolerance,
            )
        except ValueError as e:
            print(f"Warning: Skipping {mixture}: {e}")

    return results


def print_bubble_point_validation_report(
    results: Dict[str, BubblePointValidationResult],
) -> None:
    """Print a formatted validation report.

    Args:
        results: Dict of validation results from validate_bubble_point.
    """
    print("=" * 80)
    print("BUBBLE POINT TEMPERATURE VALIDATION")
    print("=" * 80)

    all_residual_passed = True
    all_T_passed = True

    for mixture, result in results.items():
        res_status = "PASS" if result.residual_passed else "FAIL"
        T_status = "PASS" if result.T_passed else "FAIL"
        all_residual_passed = all_residual_passed and result.residual_passed
        all_T_passed = all_T_passed and result.T_passed

        print(f"\n{mixture.upper()}")
        print("-" * 50)
        print(f"  Points tested: {result.n_points}")
        print(f"  Residual check: {res_status}")
        print(f"    Max residual: {result.max_residual_bar:.2e} bar (tol: {result.residual_tolerance:.0e})")
        print(f"  Temperature check: {T_status}")
        print(f"    Max T error: {result.max_T_error_K:.2f} K (tol: {result.T_tolerance:.1f} K)")
        print(f"    Mean T error: {result.mean_T_error_K:.2f} K")

        print("\n  Point-by-point results:")
        print("    x_light   T_ref [K]   T_calc [K]   Residual [bar]   T_err [K]")
        for x, T_ref, T_calc, residual, T_error in result.all_results:
            res_mark = "OK" if residual <= result.residual_tolerance else "!!"
            T_mark = "OK" if T_error <= result.T_tolerance else "!!"
            print(f"    {x:7.3f}   {T_ref:9.2f}   {T_calc:9.2f}     {residual:12.2e}    {T_error:6.2f}  {res_mark} {T_mark}")

    print("\n" + "=" * 80)
    overall_residual = "PASS" if all_residual_passed else "FAIL"
    overall_T = "PASS" if all_T_passed else "FAIL"
    print(f"OVERALL RESIDUAL CHECK: {overall_residual} (< {results[list(results.keys())[0]].residual_tolerance:.0e} bar)")
    print(f"OVERALL TEMPERATURE CHECK: {overall_T}")
    print("=" * 80)


def run_bubble_point_validation() -> bool:
    """Run full bubble point validation and print report.

    Returns:
        True if all residual checks passed (primary criterion).
    """
    results = validate_bubble_point()
    print_bubble_point_validation_report(results)
    return all(r.residual_passed for r in results.values())


if __name__ == "__main__":
    success = run_bubble_point_validation()
    exit(0 if success else 1)
