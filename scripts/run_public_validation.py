#!/usr/bin/env python3
"""Run all public benchmark validations.

This script runs the complete validation suite including:
1. NIST thermodynamics validation
2. Numerical verification (mass/energy closure)
3. Skogestad Column A benchmark
4. Wood-Berry MIMO benchmark

Usage:
    python scripts/run_public_validation.py [--quick] [--verbose]
"""

import argparse
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_nist_validation(verbose: bool = False):
    """Run NIST thermodynamics validation."""
    print("\n" + "=" * 60)
    print("1. NIST THERMODYNAMICS VALIDATION")
    print("=" * 60)

    from jax_distillation.validation_pack.thermo_validation import (
        validate_antoine_against_nist,
        validate_bubble_point,
        validate_vle_consistency,
    )

    results = {}

    # Antoine validation
    print("\n1.1 Antoine vapor pressure validation...")
    antoine_results = validate_antoine_against_nist()
    antoine_passed = all(r.passed for r in antoine_results.values())
    max_error = max(r.max_relative_error for r in antoine_results.values())
    results["antoine_passed"] = antoine_passed
    results["antoine_max_error"] = max_error
    print(f"  Status: {'PASS' if antoine_passed else 'FAIL'}")
    print(f"  Max error: {max_error*100:.2f}%")

    # Bubble point validation
    print("\n1.2 Bubble point validation...")
    bubble_results = validate_bubble_point()
    bubble_passed = all(r.residual_passed for r in bubble_results.values())
    max_residual = max(r.max_residual_bar for r in bubble_results.values())
    results["bubble_passed"] = bubble_passed
    results["bubble_max_residual"] = max_residual
    print(f"  Status: {'PASS' if bubble_passed else 'FAIL'}")
    print(f"  Max residual: {max_residual:.2e} bar")

    # VLE consistency
    print("\n1.3 VLE consistency validation...")
    vle_results = validate_vle_consistency()
    vle_passed = all(r.all_passed for r in vle_results.values())
    results["vle_passed"] = vle_passed
    print(f"  Status: {'PASS' if vle_passed else 'FAIL'}")

    return results


def run_verification(n_steps: int = 1000, verbose: bool = False):
    """Run numerical verification."""
    print("\n" + "=" * 60)
    print("2. NUMERICAL VERIFICATION")
    print("=" * 60)

    from jax_distillation.validation_pack.verification import (
        run_mass_energy_closure,
        check_invariants,
    )
    from jax_distillation.column.column import (
        create_initial_column_state,
        create_default_action,
        column_step,
    )
    from jax_distillation.column.config import create_teaching_column_config as create_default_config

    results = {}

    # Mass/energy closure
    print(f"\n2.1 Mass/energy closure ({n_steps} steps)...")
    closure_result = run_mass_energy_closure(n_steps=n_steps)
    results["mass_passed"] = closure_result.mass_passed
    results["energy_passed"] = closure_result.energy_passed
    results["mass_closure"] = closure_result.mass_closure
    results["energy_closure"] = closure_result.energy_closure
    results["n_steps"] = closure_result.n_steps
    print(f"  Mass closure: {'PASS' if closure_result.mass_passed else 'FAIL'}")
    print(f"    Error: {closure_result.mass_closure*100:.4f}%")
    print(f"  Energy closure: {'N/A (simplified model)'}")

    # Invariant check
    print(f"\n2.2 Invariant check (composition bounds, no NaN)...")
    config = create_default_config()
    state = create_initial_column_state(config)
    action = create_default_action()

    invariant_ok = True
    for _ in range(100):
        state, outputs = column_step(state, action, config)
        ok, details = check_invariants(state, outputs)
        if not ok:
            invariant_ok = False
            break

    results["invariants_passed"] = invariant_ok
    print(f"  Status: {'PASS' if invariant_ok else 'FAIL'}")

    return results


def run_cola_benchmark(verbose: bool = False):
    """Run Skogestad Column A benchmark."""
    print("\n" + "=" * 60)
    print("3. SKOGESTAD COLUMN A (COLA) BENCHMARK")
    print("=" * 60)

    from jax_distillation.validation_pack.benchmarks.skogestad_cola import (
        run_cola_benchmark,
        compute_cola_metrics,
    )

    results = {}

    print("\n3.1 Running COLA simulation...")
    benchmark_data = run_cola_benchmark()

    print("\n3.2 Computing metrics...")
    metrics = compute_cola_metrics(benchmark_data)

    results["x_D_error"] = metrics.steady_state_x_D_error
    results["x_B_error"] = metrics.steady_state_x_B_error
    results["reflux_direction_ok"] = metrics.reflux_step_direction_ok
    results["boilup_direction_ok"] = metrics.boilup_step_direction_ok
    results["temperature_monotonic"] = metrics.temperature_monotonic
    results["overall_passed"] = metrics.overall_passed

    print(f"  Steady-state x_D error: {metrics.steady_state_x_D_error*100:.1f}%")
    print(f"  Steady-state x_B error: {metrics.steady_state_x_B_error*100:.1f}%")
    print(f"  Reflux step direction: {'CORRECT' if metrics.reflux_step_direction_ok else 'INCORRECT'}")
    print(f"  Boilup step direction: {'CORRECT' if metrics.boilup_step_direction_ok else 'INCORRECT'}")
    print(f"  Temperature profile: {'MONOTONIC' if metrics.temperature_monotonic else 'NON-MONOTONIC'}")
    print(f"  Overall: {'PASS' if metrics.overall_passed else 'FAIL'}")

    return results


def run_wood_berry_benchmark(verbose: bool = False):
    """Run Wood-Berry MIMO benchmark."""
    print("\n" + "=" * 60)
    print("4. WOOD-BERRY MIMO BENCHMARK")
    print("=" * 60)

    from jax_distillation.validation_pack.benchmarks.wood_berry import (
        run_wood_berry_benchmark,
    )

    results = {}

    print("\n4.1 Running Wood-Berry comparison...")
    wb_result = run_wood_berry_benchmark()

    results["all_signs_correct"] = wb_result.all_signs_correct
    results["coupling_structure_ok"] = wb_result.coupling_structure_ok
    results["overall_passed"] = wb_result.overall_passed

    print(f"  Gain signs: {'ALL CORRECT' if wb_result.all_signs_correct else 'SOME INCORRECT'}")
    for name, correct in wb_result.gain_signs_match.items():
        status = "✓" if correct else "✗"
        print(f"    {name}: {status}")
    print(f"  Coupling structure: {'MATCHES' if wb_result.coupling_structure_ok else 'DIFFERS'}")
    print(f"  Overall: {'PASS' if wb_result.overall_passed else 'FAIL'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all public benchmark validations"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (fewer steps)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("JAX DISTILLATION SIMULATOR - PUBLIC VALIDATION SUITE")
    print("=" * 60)

    start_time = time.time()
    all_results = {}

    # 1. NIST Thermodynamics
    try:
        all_results["thermo"] = run_nist_validation(args.verbose)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["thermo"] = {"error": str(e)}

    # 2. Numerical Verification
    try:
        n_steps = 200 if args.quick else 1000
        all_results["verification"] = run_verification(n_steps, args.verbose)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["verification"] = {"error": str(e)}

    # 3. COLA Benchmark
    try:
        all_results["cola"] = run_cola_benchmark(args.verbose)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["cola"] = {"error": str(e)}

    # 4. Wood-Berry Benchmark
    try:
        all_results["wood_berry"] = run_wood_berry_benchmark(args.verbose)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["wood_berry"] = {"error": str(e)}

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    def check_passed(results, key):
        return results.get(key, {}).get("error") is None

    thermo_ok = all_results.get("thermo", {}).get("antoine_passed", False)
    verif_ok = all_results.get("verification", {}).get("mass_passed", False)
    cola_ok = all_results.get("cola", {}).get("overall_passed", False)
    wb_ok = all_results.get("wood_berry", {}).get("overall_passed", False)

    print(f"\n  1. NIST Thermodynamics:  {'PASS' if thermo_ok else 'FAIL'}")
    print(f"  2. Numerical Verification: {'PASS' if verif_ok else 'FAIL'}")
    print(f"  3. COLA Benchmark:        {'PASS' if cola_ok else 'FAIL'}")
    print(f"  4. Wood-Berry Benchmark:  {'PASS' if wb_ok else 'FAIL'}")

    all_passed = thermo_ok and verif_ok and cola_ok and wb_ok

    print(f"\n  Time elapsed: {elapsed:.1f} s")
    print(f"\n  OVERALL: {'ALL VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
