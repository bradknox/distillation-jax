"""Invariant checking for simulation robustness.

This module verifies that the simulator maintains critical invariants
over extended runs:
- No NaN or Inf values
- Compositions always in [0, 1]
- Holdups always non-negative
- Temperatures in physically reasonable range

Acceptance criteria:
- No NaN/Inf for 50,000+ steps
- All compositions remain in [0, 1]
- All holdups remain >= 0
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    ColumnOutputs,
    column_step,
    create_initial_column_state,
    create_default_action,
)
from jax_distillation.column.config import ColumnConfig, create_teaching_column_config


@dataclass
class InvariantCheckResult:
    """Result of invariant checking.

    Attributes:
        n_steps: Number of simulation steps checked
        nan_inf_ok: True if no NaN/Inf values encountered
        composition_ok: True if all compositions in [0, 1]
        holdup_ok: True if all holdups >= 0
        temperature_ok: True if all temperatures in valid range
        all_passed: True if all invariants satisfied
        first_violation_step: Step number of first violation (-1 if none)
        violation_details: Description of first violation
        final_state_valid: True if final state satisfies all invariants
    """

    n_steps: int
    nan_inf_ok: bool
    composition_ok: bool
    holdup_ok: bool
    temperature_ok: bool
    all_passed: bool
    first_violation_step: int
    violation_details: str
    final_state_valid: bool


def _check_state_invariants(
    state: FullColumnState,
    outputs: Optional[ColumnOutputs] = None,
    T_min: float = 250.0,
    T_max: float = 500.0,
) -> Tuple[bool, str]:
    """Check invariants for a single state.

    Args:
        state: Column state to check.
        outputs: Optional outputs to also check.
        T_min: Minimum allowed temperature [K].
        T_max: Maximum allowed temperature [K].

    Returns:
        Tuple of (all_ok, violation_description).
    """
    violations = []

    # Check tray compositions
    tray_x = np.array(state.tray_x)
    if not np.all(np.isfinite(tray_x)):
        violations.append("tray_x contains NaN/Inf")
    if np.any(tray_x < 0) or np.any(tray_x > 1):
        violations.append(f"tray_x out of bounds: [{tray_x.min():.6f}, {tray_x.max():.6f}]")

    # Check tray holdups
    tray_M = np.array(state.tray_M)
    if not np.all(np.isfinite(tray_M)):
        violations.append("tray_M contains NaN/Inf")
    if np.any(tray_M < 0):
        violations.append(f"tray_M negative: min={tray_M.min():.6f}")

    # Check tray temperatures
    tray_T = np.array(state.tray_T)
    if not np.all(np.isfinite(tray_T)):
        violations.append("tray_T contains NaN/Inf")
    if np.any(tray_T < T_min) or np.any(tray_T > T_max):
        violations.append(f"tray_T out of range: [{tray_T.min():.1f}, {tray_T.max():.1f}] K")

    # Check reboiler
    reb_x = float(state.reboiler.x)
    reb_M = float(state.reboiler.M)
    reb_T = float(state.reboiler.T)

    if not np.isfinite(reb_x):
        violations.append("reboiler.x is NaN/Inf")
    if reb_x < 0 or reb_x > 1:
        violations.append(f"reboiler.x out of bounds: {reb_x:.6f}")
    if not np.isfinite(reb_M) or reb_M < 0:
        violations.append(f"reboiler.M invalid: {reb_M:.6f}")
    if not np.isfinite(reb_T) or reb_T < T_min or reb_T > T_max:
        violations.append(f"reboiler.T out of range: {reb_T:.1f} K")

    # Check condenser
    cond_x = float(state.condenser.x)
    cond_M = float(state.condenser.M)
    cond_T = float(state.condenser.T)

    if not np.isfinite(cond_x):
        violations.append("condenser.x is NaN/Inf")
    if cond_x < 0 or cond_x > 1:
        violations.append(f"condenser.x out of bounds: {cond_x:.6f}")
    if not np.isfinite(cond_M) or cond_M < 0:
        violations.append(f"condenser.M invalid: {cond_M:.6f}")
    if not np.isfinite(cond_T) or cond_T < T_min or cond_T > T_max:
        violations.append(f"condenser.T out of range: {cond_T:.1f} K")

    # Check outputs if provided
    if outputs is not None:
        x_D = float(outputs.x_D)
        x_B = float(outputs.x_B)
        D = float(outputs.D)
        B = float(outputs.B)

        if not np.isfinite(x_D) or x_D < 0 or x_D > 1:
            violations.append(f"x_D invalid: {x_D:.6f}")
        if not np.isfinite(x_B) or x_B < 0 or x_B > 1:
            violations.append(f"x_B invalid: {x_B:.6f}")
        if not np.isfinite(D) or D < 0:
            violations.append(f"D invalid: {D:.6f}")
        if not np.isfinite(B) or B < 0:
            violations.append(f"B invalid: {B:.6f}")

    all_ok = len(violations) == 0
    violation_str = "; ".join(violations) if violations else "None"

    return all_ok, violation_str


def check_invariants(
    state: FullColumnState,
    outputs: Optional[ColumnOutputs] = None,
) -> Tuple[bool, str]:
    """Public wrapper for invariant checking on a single state.

    Args:
        state: Column state to check.
        outputs: Optional outputs to check.

    Returns:
        Tuple of (all_ok, violation_description).
    """
    return _check_state_invariants(state, outputs)


def run_long_simulation_invariant_check(
    config: Optional[ColumnConfig] = None,
    action: Optional[ColumnAction] = None,
    n_steps: int = 50000,
    check_interval: int = 100,
    stop_on_violation: bool = True,
) -> InvariantCheckResult:
    """Run simulation and check invariants throughout.

    Args:
        config: Column configuration (uses default if None).
        action: Control action (uses default if None).
        n_steps: Number of simulation steps.
        check_interval: Steps between invariant checks.
        stop_on_violation: If True, stop at first violation.

    Returns:
        InvariantCheckResult with check results.
    """
    if config is None:
        config = create_teaching_column_config()
    if action is None:
        action = create_default_action()

    state = create_initial_column_state(config)

    # Track violations
    nan_inf_ok = True
    composition_ok = True
    holdup_ok = True
    temperature_ok = True
    first_violation_step = -1
    violation_details = ""

    steps_completed = 0

    for step in range(n_steps):
        try:
            state, outputs = column_step(state, action, config)
            steps_completed = step + 1

            # Check invariants at intervals
            if (step + 1) % check_interval == 0:
                ok, details = _check_state_invariants(state, outputs)

                if not ok:
                    if first_violation_step < 0:
                        first_violation_step = step + 1
                        violation_details = details

                    # Categorize violations
                    if "NaN" in details or "Inf" in details:
                        nan_inf_ok = False
                    if "out of bounds" in details and ("_x" in details):
                        composition_ok = False
                    if "negative" in details or "invalid" in details:
                        holdup_ok = False
                    if "out of range" in details and "_T" in details:
                        temperature_ok = False

                    if stop_on_violation:
                        break

        except Exception as e:
            first_violation_step = step + 1
            violation_details = f"Exception: {str(e)}"
            nan_inf_ok = False
            break

    # Final state check
    final_ok, final_details = _check_state_invariants(state)

    all_passed = nan_inf_ok and composition_ok and holdup_ok and temperature_ok

    return InvariantCheckResult(
        n_steps=steps_completed,
        nan_inf_ok=nan_inf_ok,
        composition_ok=composition_ok,
        holdup_ok=holdup_ok,
        temperature_ok=temperature_ok,
        all_passed=all_passed,
        first_violation_step=first_violation_step,
        violation_details=violation_details,
        final_state_valid=final_ok,
    )


def print_invariant_check_report(result: InvariantCheckResult) -> None:
    """Print a formatted invariant check report.

    Args:
        result: InvariantCheckResult from run_long_simulation_invariant_check.
    """
    print("=" * 70)
    print("INVARIANT CHECK VERIFICATION")
    print("=" * 70)

    print(f"\nSteps completed: {result.n_steps}")

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"\nInvariant Checks:")
    print(f"  No NaN/Inf:        {status(result.nan_inf_ok)}")
    print(f"  Compositions [0,1]: {status(result.composition_ok)}")
    print(f"  Holdups >= 0:       {status(result.holdup_ok)}")
    print(f"  Temperatures valid: {status(result.temperature_ok)}")

    if result.first_violation_step >= 0:
        print(f"\nFirst violation at step {result.first_violation_step}:")
        print(f"  {result.violation_details}")

    print(f"\nFinal state valid: {status(result.final_state_valid)}")

    print("\n" + "=" * 70)
    overall = "PASS" if result.all_passed else "FAIL"
    print(f"OVERALL: {overall}")
    print("=" * 70)


if __name__ == "__main__":
    print("Running invariant check for 10,000 steps...")
    result = run_long_simulation_invariant_check(n_steps=10000, check_interval=100)
    print_invariant_check_report(result)
