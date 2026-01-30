"""Validate Antoine vapor pressure calculations against NIST reference data.

This module tests that the simulator's Antoine equation implementation
produces vapor pressures within 2% of NIST WebBook tabulated values.

Acceptance criteria:
- Relative error < 2% for all NIST reference points
- At normal boiling points: pressure should equal 1.01325 bar within tolerance
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp

from jax_distillation.core.thermodynamics import (
    antoine_vapor_pressure,
    ANTOINE_METHANOL,
    ANTOINE_WATER,
    ANTOINE_ETHANOL,
    ANTOINE_BENZENE,
    ANTOINE_TOLUENE,
    AntoineParams,
)
from jax_distillation.validation_pack.thermo_validation.nist_points import (
    get_nist_vapor_pressure_data,
    get_all_nist_vapor_pressure_data,
    NISTReferencePoint,
)


@dataclass
class AntoineValidationResult:
    """Result of Antoine equation validation against NIST data.

    Attributes:
        compound: Chemical compound name
        n_points: Number of reference points tested
        max_relative_error: Maximum relative error across all points
        mean_relative_error: Mean relative error across all points
        all_errors: List of (T, P_ref, P_calc, rel_error) for each point
        passed: True if all points within tolerance
        tolerance: Tolerance used for pass/fail (default 2%)
    """

    compound: str
    n_points: int
    max_relative_error: float
    mean_relative_error: float
    all_errors: List[tuple]
    passed: bool
    tolerance: float = 0.02  # 2%


def _get_antoine_params(compound: str) -> Optional[AntoineParams]:
    """Get Antoine parameters for a compound."""
    params_map = {
        "methanol": ANTOINE_METHANOL,
        "water": ANTOINE_WATER,
        "ethanol": ANTOINE_ETHANOL,
        "benzene": ANTOINE_BENZENE,
        "toluene": ANTOINE_TOLUENE,
    }
    return params_map.get(compound.lower())


def validate_antoine_single_compound(
    compound: str,
    tolerance: float = 0.02,
) -> AntoineValidationResult:
    """Validate Antoine equation for a single compound against NIST data.

    Args:
        compound: Compound name (methanol, water, ethanol, benzene, toluene)
        tolerance: Maximum allowed relative error (default 2%)

    Returns:
        AntoineValidationResult with validation metrics.

    Raises:
        ValueError: If compound is unknown or has no NIST data.
    """
    # Get NIST reference data
    nist_data = get_nist_vapor_pressure_data(compound)

    # Get Antoine parameters
    antoine_params = _get_antoine_params(compound)
    if antoine_params is None:
        raise ValueError(f"No Antoine parameters for compound: {compound}")

    errors = []
    rel_errors = []

    for point in nist_data:
        T = jnp.array(point.temperature_k)
        P_ref = point.pressure_bar

        # Calculate vapor pressure using Antoine equation
        P_calc = float(antoine_vapor_pressure(T, antoine_params))

        # Relative error
        if P_ref > 0:
            rel_error = abs(P_calc - P_ref) / P_ref
        else:
            rel_error = float("inf")

        errors.append((point.temperature_k, P_ref, P_calc, rel_error))
        rel_errors.append(rel_error)

    max_error = max(rel_errors) if rel_errors else 0.0
    mean_error = sum(rel_errors) / len(rel_errors) if rel_errors else 0.0
    passed = max_error <= tolerance

    return AntoineValidationResult(
        compound=compound,
        n_points=len(nist_data),
        max_relative_error=max_error,
        mean_relative_error=mean_error,
        all_errors=errors,
        passed=passed,
        tolerance=tolerance,
    )


def validate_antoine_against_nist(
    compounds: Optional[List[str]] = None,
    tolerance: float = 0.02,
) -> Dict[str, AntoineValidationResult]:
    """Validate Antoine equation for multiple compounds against NIST data.

    Args:
        compounds: List of compound names. If None, validates all available.
        tolerance: Maximum allowed relative error (default 2%)

    Returns:
        Dict mapping compound names to AntoineValidationResult.
    """
    if compounds is None:
        compounds = ["methanol", "water", "ethanol", "benzene", "toluene"]

    results = {}
    for compound in compounds:
        try:
            results[compound] = validate_antoine_single_compound(
                compound, tolerance=tolerance
            )
        except ValueError as e:
            # Skip compounds with missing data
            print(f"Warning: Skipping {compound}: {e}")

    return results


def print_antoine_validation_report(
    results: Dict[str, AntoineValidationResult],
) -> None:
    """Print a formatted validation report.

    Args:
        results: Dict of validation results from validate_antoine_against_nist.
    """
    print("=" * 70)
    print("ANTOINE EQUATION VALIDATION AGAINST NIST DATA")
    print("=" * 70)

    all_passed = True

    for compound, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        all_passed = all_passed and result.passed

        print(f"\n{compound.upper()}")
        print("-" * 40)
        print(f"  Status: {status}")
        print(f"  Points tested: {result.n_points}")
        print(f"  Max relative error: {result.max_relative_error:.4f} ({result.max_relative_error*100:.2f}%)")
        print(f"  Mean relative error: {result.mean_relative_error:.4f} ({result.mean_relative_error*100:.2f}%)")
        print(f"  Tolerance: {result.tolerance*100:.1f}%")

        if not result.passed or result.max_relative_error > 0.01:
            print("\n  Point-by-point results:")
            print("  T [K]     P_ref [bar]  P_calc [bar]  Error [%]")
            for T, P_ref, P_calc, err in result.all_errors:
                status_mark = "OK" if err <= result.tolerance else "!!"
                print(f"  {T:7.2f}   {P_ref:10.4f}   {P_calc:10.4f}    {err*100:6.2f}  {status_mark}")

    print("\n" + "=" * 70)
    overall = "ALL VALIDATIONS PASSED" if all_passed else "SOME VALIDATIONS FAILED"
    print(f"OVERALL: {overall}")
    print("=" * 70)


def run_antoine_validation() -> bool:
    """Run full Antoine validation and print report.

    Returns:
        True if all validations passed.
    """
    results = validate_antoine_against_nist()
    print_antoine_validation_report(results)
    return all(r.passed for r in results.values())


if __name__ == "__main__":
    success = run_antoine_validation()
    exit(0 if success else 1)
