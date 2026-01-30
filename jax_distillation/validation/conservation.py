"""Mass and energy balance conservation checks.

This module provides validation tools to verify that the simulation
maintains physical conservation laws:
- Total mass conservation
- Component mass conservation
- Energy conservation

These checks are essential for validating simulation fidelity.
"""

import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from jax_distillation.column.column import (
    FullColumnState,
    ColumnOutputs,
)
from jax_distillation.column.config import ColumnConfig


class ConservationMetrics(NamedTuple):
    """Metrics for conservation law validation.

    Attributes:
        mass_error: Relative mass balance error.
        component_error: Relative component balance error.
        energy_error: Relative energy balance error (if computed).
        is_conserved: True if all errors within tolerance.
    """

    mass_error: float
    component_error: float
    energy_error: float
    is_conserved: bool


def compute_total_mass(state: FullColumnState) -> float:
    """Compute total moles in the column.

    Args:
        state: Current column state.

    Returns:
        Total moles [mol].
    """
    tray_mass = float(jnp.sum(state.tray_M))
    reboiler_mass = float(state.reboiler.M)
    condenser_mass = float(state.condenser.M)

    return tray_mass + reboiler_mass + condenser_mass


def compute_total_component(state: FullColumnState) -> float:
    """Compute total moles of light component in the column.

    Args:
        state: Current column state.

    Returns:
        Total moles of light component [mol].
    """
    tray_component = float(jnp.sum(state.tray_M * state.tray_x))
    reboiler_component = float(state.reboiler.M * state.reboiler.x)
    condenser_component = float(state.condenser.M * state.condenser.x)

    return tray_component + reboiler_component + condenser_component


def check_mass_balance(
    state_initial: FullColumnState,
    state_final: FullColumnState,
    feed_added: float,
    products_removed: float,
    tolerance: float = 0.01,
) -> tuple[float, bool]:
    """Check overall mass balance conservation.

    Conservation: M_final - M_initial = Feed_added - Products_removed

    Args:
        state_initial: State at start of period.
        state_final: State at end of period.
        feed_added: Total feed added during period [mol].
        products_removed: Total products (D + B) removed [mol].
        tolerance: Relative tolerance for conservation.

    Returns:
        Tuple of (relative_error, is_conserved).
    """
    M_initial = compute_total_mass(state_initial)
    M_final = compute_total_mass(state_final)

    # Expected change
    expected_change = feed_added - products_removed
    actual_change = M_final - M_initial

    # Relative error (normalized by total mass)
    reference = max(M_initial, 1.0)
    error = abs(actual_change - expected_change) / reference

    return error, error < tolerance


def check_component_balance(
    state_initial: FullColumnState,
    state_final: FullColumnState,
    feed_component_added: float,
    products_component_removed: float,
    tolerance: float = 0.01,
) -> tuple[float, bool]:
    """Check component (light species) mass balance.

    Args:
        state_initial: State at start of period.
        state_final: State at end of period.
        feed_component_added: Moles of light component in feed [mol].
        products_component_removed: Moles of light component in products [mol].
        tolerance: Relative tolerance.

    Returns:
        Tuple of (relative_error, is_conserved).
    """
    C_initial = compute_total_component(state_initial)
    C_final = compute_total_component(state_final)

    expected_change = feed_component_added - products_component_removed
    actual_change = C_final - C_initial

    reference = max(C_initial, 0.1)
    error = abs(actual_change - expected_change) / reference

    return error, error < tolerance


def validate_simulation_step(
    state_before: FullColumnState,
    state_after: FullColumnState,
    outputs: ColumnOutputs,
    config: ColumnConfig,
    tolerance: float = 0.05,
) -> ConservationMetrics:
    """Validate conservation laws for a single simulation step.

    Args:
        state_before: State before step.
        state_after: State after step.
        outputs: Outputs from the step.
        config: Column configuration.
        tolerance: Acceptable relative error.

    Returns:
        ConservationMetrics with validation results.
    """
    dt = float(config.simulation.dt)

    # Mass added and removed during step
    feed_added = float(config.feed.F) * dt
    products_removed = (float(outputs.D) + float(outputs.B)) * dt

    mass_error, mass_ok = check_mass_balance(
        state_before, state_after, feed_added, products_removed, tolerance
    )

    # Component balance
    z_F = float(config.feed.z_F)
    feed_component = feed_added * z_F
    products_component = (
        float(outputs.D) * float(outputs.x_D) + float(outputs.B) * float(outputs.x_B)
    ) * dt

    component_error, component_ok = check_component_balance(
        state_before, state_after, feed_component, products_component, tolerance
    )

    # Energy balance (simplified - full implementation would track enthalpies)
    energy_error = 0.0  # Placeholder

    is_conserved = mass_ok and component_ok

    return ConservationMetrics(
        mass_error=mass_error,
        component_error=component_error,
        energy_error=energy_error,
        is_conserved=is_conserved,
    )


def run_conservation_validation(
    states: list[FullColumnState],
    outputs_list: list[ColumnOutputs],
    config: ColumnConfig,
    tolerance: float = 0.05,
) -> dict:
    """Run conservation validation over a trajectory.

    Args:
        states: List of states from simulation.
        outputs_list: List of outputs from each step.
        config: Column configuration.
        tolerance: Acceptable relative error.

    Returns:
        Dictionary with validation summary.
    """
    if len(states) < 2:
        return {"error": "Need at least 2 states for validation"}

    mass_errors = []
    component_errors = []
    all_conserved = True

    for i in range(len(outputs_list)):
        if i + 1 >= len(states):
            break

        metrics = validate_simulation_step(
            states[i], states[i + 1], outputs_list[i], config, tolerance
        )

        mass_errors.append(metrics.mass_error)
        component_errors.append(metrics.component_error)
        all_conserved = all_conserved and metrics.is_conserved

    return {
        "n_steps": len(mass_errors),
        "mean_mass_error": np.mean(mass_errors) if mass_errors else 0.0,
        "max_mass_error": np.max(mass_errors) if mass_errors else 0.0,
        "mean_component_error": np.mean(component_errors) if component_errors else 0.0,
        "max_component_error": np.max(component_errors) if component_errors else 0.0,
        "all_conserved": all_conserved,
        "tolerance": tolerance,
    }


def check_steady_state_mass_balance(
    outputs: ColumnOutputs,
    config: ColumnConfig,
    tolerance: float = 0.01,
) -> tuple[float, bool]:
    """Check mass balance at steady state.

    At steady state: F = D + B

    Args:
        outputs: Outputs at steady state.
        config: Column configuration.
        tolerance: Acceptable relative error.

    Returns:
        Tuple of (relative_error, is_balanced).
    """
    F = float(config.feed.F)
    D = float(outputs.D)
    B = float(outputs.B)

    error = abs(D + B - F) / max(F, 1e-6)

    return error, error < tolerance


def check_steady_state_component_balance(
    outputs: ColumnOutputs,
    config: ColumnConfig,
    tolerance: float = 0.02,
) -> tuple[float, bool]:
    """Check component balance at steady state.

    At steady state: F * z_F = D * x_D + B * x_B

    Args:
        outputs: Outputs at steady state.
        config: Column configuration.
        tolerance: Acceptable relative error.

    Returns:
        Tuple of (relative_error, is_balanced).
    """
    F = float(config.feed.F)
    z_F = float(config.feed.z_F)
    D = float(outputs.D)
    x_D = float(outputs.x_D)
    B = float(outputs.B)
    x_B = float(outputs.x_B)

    feed_component = F * z_F
    product_component = D * x_D + B * x_B

    error = abs(product_component - feed_component) / max(feed_component, 1e-6)

    return error, error < tolerance
