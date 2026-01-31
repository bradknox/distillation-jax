"""Mass and energy conservation closure verification.

This module provides comprehensive verification that the simulator
properly conserves mass and energy over extended simulation runs.

Acceptance criteria:
- Mass closure error < 0.1% over long runs
- Energy closure error < 1% (when implemented)
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
from jax_distillation.validation.conservation import (
    compute_total_mass,
    compute_total_component,
    ConservationMetrics,
)
from jax_distillation.core.thermodynamics import liquid_enthalpy
from jax_distillation.core.types import ThermoParams


@dataclass
class MassEnergyClosureResult:
    """Result of mass/energy closure verification.

    Attributes:
        n_steps: Number of simulation steps run
        total_time: Total simulation time [s]
        mass_closure: Final mass closure metrics
        component_closure: Final component closure metrics
        energy_closure: Final energy closure metrics (if computed)
        mass_passed: True if mass closure < tolerance
        component_passed: True if component closure < tolerance
        energy_passed: True if energy closure < tolerance
        trajectory: List of (time, mass_error, component_error) tuples
        mass_tolerance: Tolerance used for mass check (default 0.001 = 0.1%)
        component_tolerance: Tolerance for component check
        energy_tolerance: Tolerance for energy check (default 0.01 = 1%)
    """

    n_steps: int
    total_time: float
    mass_closure: float
    component_closure: float
    energy_closure: float
    mass_passed: bool
    component_passed: bool
    energy_passed: bool
    trajectory: List[Tuple[float, float, float]]
    mass_tolerance: float = 0.001
    component_tolerance: float = 0.001
    energy_tolerance: float = 0.01


def check_mass_closure(
    state_initial: FullColumnState,
    state_final: FullColumnState,
    cumulative_feed: float,
    cumulative_distillate: float,
    cumulative_bottoms: float,
) -> Tuple[float, float]:
    """Check mass closure over a simulation period.

    Conservation: M_final - M_initial = Feed - (D + B)

    Args:
        state_initial: State at start of period.
        state_final: State at end of period.
        cumulative_feed: Total feed added [mol].
        cumulative_distillate: Total distillate removed [mol].
        cumulative_bottoms: Total bottoms removed [mol].

    Returns:
        Tuple of (absolute_error [mol], relative_error [fraction]).
    """
    M_initial = compute_total_mass(state_initial)
    M_final = compute_total_mass(state_final)

    expected_change = cumulative_feed - cumulative_distillate - cumulative_bottoms
    actual_change = M_final - M_initial

    absolute_error = abs(actual_change - expected_change)
    relative_error = absolute_error / max(cumulative_feed, 1.0)

    return absolute_error, relative_error


def check_component_closure(
    state_initial: FullColumnState,
    state_final: FullColumnState,
    cumulative_feed_component: float,
    cumulative_distillate_component: float,
    cumulative_bottoms_component: float,
) -> Tuple[float, float]:
    """Check component (light species) closure over a simulation period.

    Conservation: C_final - C_initial = F*z_F - (D*x_D + B*x_B)

    Args:
        state_initial: State at start.
        state_final: State at end.
        cumulative_feed_component: Total light component in feed [mol].
        cumulative_distillate_component: Total light component in distillate [mol].
        cumulative_bottoms_component: Total light component in bottoms [mol].

    Returns:
        Tuple of (absolute_error [mol], relative_error [fraction]).
    """
    C_initial = compute_total_component(state_initial)
    C_final = compute_total_component(state_final)

    expected_change = (
        cumulative_feed_component
        - cumulative_distillate_component
        - cumulative_bottoms_component
    )
    actual_change = C_final - C_initial

    absolute_error = abs(actual_change - expected_change)
    relative_error = absolute_error / max(cumulative_feed_component, 0.1)

    return absolute_error, relative_error


def compute_column_energy(state: FullColumnState, thermo: ThermoParams) -> float:
    """Compute total internal energy of the column.

    E = sum(M_i * h_L(x_i, T_i)) for all trays + reboiler + condenser

    Args:
        state: Column state.
        thermo: Thermodynamic parameters.

    Returns:
        Total column energy [J].
    """
    # Tray energies
    tray_h = liquid_enthalpy(state.tray_x, state.tray_T, thermo)
    tray_energy = float(jnp.sum(state.tray_M * tray_h))

    # Reboiler energy
    reb_h = liquid_enthalpy(state.reboiler.x, state.reboiler.T, thermo)
    reb_energy = float(state.reboiler.M * reb_h)

    # Condenser energy
    cond_h = liquid_enthalpy(state.condenser.x, state.condenser.T, thermo)
    cond_energy = float(state.condenser.M * cond_h)

    return tray_energy + reb_energy + cond_energy


def check_energy_closure(
    state_initial: FullColumnState,
    state_final: FullColumnState,
    cumulative_Q_R: float,
    cumulative_Q_C: float,
    cumulative_feed_enthalpy: float,
    cumulative_product_enthalpy: float,
    thermo: Optional[ThermoParams] = None,
) -> Tuple[float, float]:
    """Check energy closure over a simulation period.

    Conservation: E_final - E_initial = Q_R - Q_C + H_feed - H_products

    Args:
        state_initial: State at start.
        state_final: State at end.
        cumulative_Q_R: Total reboiler duty [J].
        cumulative_Q_C: Total condenser duty [J].
        cumulative_feed_enthalpy: Total feed enthalpy [J].
        cumulative_product_enthalpy: Total product enthalpy [J].
        thermo: Thermodynamic parameters (required for energy calculation).

    Returns:
        Tuple of (absolute_error [J], relative_error [fraction]).
    """
    if thermo is None:
        # Cannot compute without thermo params - return placeholder
        return 0.0, 0.0

    # Compute actual energy change in the column
    E_initial = compute_column_energy(state_initial, thermo)
    E_final = compute_column_energy(state_final, thermo)
    actual_change = E_final - E_initial

    # Expected change based on energy balance
    # Energy in: Q_R (reboiler heat), H_feed (feed enthalpy)
    # Energy out: Q_C (condenser heat), H_products (product enthalpies)
    expected_change = cumulative_Q_R - cumulative_Q_C + cumulative_feed_enthalpy - cumulative_product_enthalpy

    # Compute errors
    absolute_error = abs(actual_change - expected_change)

    # Normalize by total energy input (Q_R is typically dominant)
    reference = max(abs(cumulative_Q_R), 1.0)
    relative_error = absolute_error / reference

    return absolute_error, relative_error


def run_mass_energy_closure(
    config: Optional[ColumnConfig] = None,
    action: Optional[ColumnAction] = None,
    n_steps: int = 1000,
    warmup_steps: int = 500,
    mass_tolerance: float = 0.001,
    component_tolerance: float = 0.001,
    energy_tolerance: float = 0.01,
    record_interval: int = 100,
) -> MassEnergyClosureResult:
    """Run simulation and verify mass/energy closure.

    The simulation runs a warmup period first to reach near-steady-state,
    then measures cumulative mass/energy closure over the measurement period.
    This avoids startup transient effects dominating the closure error.

    Args:
        config: Column configuration (uses default if None).
        action: Control action (uses default if None).
        n_steps: Number of simulation steps for measurement period.
        warmup_steps: Number of warmup steps before measurement (default 500).
        mass_tolerance: Maximum allowed mass closure error.
        component_tolerance: Maximum allowed component closure error.
        energy_tolerance: Maximum allowed energy closure error.
        record_interval: Steps between recording trajectory points.

    Returns:
        MassEnergyClosureResult with verification results.
    """
    if config is None:
        config = create_teaching_column_config()
    if action is None:
        action = create_default_action()

    dt = float(config.simulation.dt)
    F = float(config.feed.F)
    z_F = float(config.feed.z_F)

    # Initialize
    state = create_initial_column_state(config)

    # Warmup period - run to near steady state
    for _ in range(warmup_steps):
        state, _ = column_step(state, action, config)

    # Now start measurement from warmed-up state
    state_initial = state

    # Get thermodynamic parameters for enthalpy calculations
    thermo = config.thermo
    T_feed = float(config.feed.T_feed) if hasattr(config.feed, 'T_feed') else 350.0  # K

    # Accumulators
    cumulative_feed = 0.0
    cumulative_distillate = 0.0
    cumulative_bottoms = 0.0
    cumulative_feed_component = 0.0
    cumulative_distillate_component = 0.0
    cumulative_bottoms_component = 0.0
    cumulative_Q_R = 0.0
    cumulative_Q_C = 0.0
    cumulative_feed_enthalpy = 0.0
    cumulative_product_enthalpy = 0.0

    trajectory = []

    for step in range(n_steps):
        # Run one step
        state, outputs = column_step(state, action, config)

        # Accumulate flows
        D = float(outputs.D)
        B = float(outputs.B)
        x_D = float(outputs.x_D)
        x_B = float(outputs.x_B)
        Q_R = float(outputs.Q_R)
        Q_C = float(outputs.Q_C)

        cumulative_feed += F * dt
        cumulative_distillate += D * dt
        cumulative_bottoms += B * dt
        cumulative_feed_component += F * z_F * dt
        cumulative_distillate_component += D * x_D * dt
        cumulative_bottoms_component += B * x_B * dt
        cumulative_Q_R += Q_R * dt
        cumulative_Q_C += Q_C * dt

        # Accumulate enthalpies for energy balance
        # Feed enthalpy (liquid feed at feed temperature)
        h_feed = float(liquid_enthalpy(jnp.array(z_F), jnp.array(T_feed), thermo))
        cumulative_feed_enthalpy += F * h_feed * dt

        # Product enthalpies (liquid products at their respective temperatures)
        T_D = float(state.condenser.T)
        T_B = float(state.reboiler.T)
        h_D = float(liquid_enthalpy(jnp.array(x_D), jnp.array(T_D), thermo))
        h_B = float(liquid_enthalpy(jnp.array(x_B), jnp.array(T_B), thermo))
        cumulative_product_enthalpy += (D * h_D + B * h_B) * dt

        # Record trajectory at intervals
        if (step + 1) % record_interval == 0:
            _, mass_err = check_mass_closure(
                state_initial,
                state,
                cumulative_feed,
                cumulative_distillate,
                cumulative_bottoms,
            )
            _, comp_err = check_component_closure(
                state_initial,
                state,
                cumulative_feed_component,
                cumulative_distillate_component,
                cumulative_bottoms_component,
            )
            t = float(state.t)
            trajectory.append((t, mass_err, comp_err))

    # Final closure checks
    _, mass_closure = check_mass_closure(
        state_initial,
        state,
        cumulative_feed,
        cumulative_distillate,
        cumulative_bottoms,
    )
    _, component_closure = check_component_closure(
        state_initial,
        state,
        cumulative_feed_component,
        cumulative_distillate_component,
        cumulative_bottoms_component,
    )
    _, energy_closure = check_energy_closure(
        state_initial,
        state,
        cumulative_Q_R,
        cumulative_Q_C,
        cumulative_feed_enthalpy,
        cumulative_product_enthalpy,
        thermo=thermo,
    )

    return MassEnergyClosureResult(
        n_steps=n_steps,
        total_time=float(state.t),
        mass_closure=mass_closure,
        component_closure=component_closure,
        energy_closure=energy_closure,
        mass_passed=mass_closure < mass_tolerance,
        component_passed=component_closure < component_tolerance,
        energy_passed=energy_closure < energy_tolerance,
        trajectory=trajectory,
        mass_tolerance=mass_tolerance,
        component_tolerance=component_tolerance,
        energy_tolerance=energy_tolerance,
    )


def print_mass_energy_closure_report(result: MassEnergyClosureResult) -> None:
    """Print a formatted closure verification report.

    Args:
        result: MassEnergyClosureResult from run_mass_energy_closure.
    """
    print("=" * 70)
    print("MASS/ENERGY CLOSURE VERIFICATION")
    print("=" * 70)

    print(f"\nSimulation: {result.n_steps} steps, {result.total_time:.1f} s")

    mass_status = "PASS" if result.mass_passed else "FAIL"
    comp_status = "PASS" if result.component_passed else "FAIL"

    print(f"\nMass Closure: {mass_status}")
    print(f"  Error: {result.mass_closure:.6f} ({result.mass_closure*100:.4f}%)")
    print(f"  Tolerance: {result.mass_tolerance:.6f} ({result.mass_tolerance*100:.4f}%)")

    print(f"\nComponent Closure: {comp_status}")
    print(f"  Error: {result.component_closure:.6f} ({result.component_closure*100:.4f}%)")
    print(f"  Tolerance: {result.component_tolerance:.6f} ({result.component_tolerance*100:.4f}%)")

    energy_status = "PASS" if result.energy_passed else "FAIL"
    print(f"\nEnergy Closure: {energy_status}")
    print(f"  Error: {result.energy_closure:.6f} ({result.energy_closure*100:.4f}%)")
    print(f"  Tolerance: {result.energy_tolerance:.6f} ({result.energy_tolerance*100:.4f}%)")

    if result.trajectory:
        print(f"\nTrajectory ({len(result.trajectory)} points):")
        print("  Time [s]    Mass Err    Comp Err")
        for t, m_err, c_err in result.trajectory[-5:]:  # Last 5 points
            print(f"  {t:8.1f}    {m_err:.6f}    {c_err:.6f}")

    print("\n" + "=" * 70)
    overall = "PASS" if (result.mass_passed and result.component_passed and result.energy_passed) else "FAIL"
    print(f"OVERALL: {overall}")
    print("=" * 70)


if __name__ == "__main__":
    result = run_mass_energy_closure(n_steps=1000)
    print_mass_energy_closure_report(result)
