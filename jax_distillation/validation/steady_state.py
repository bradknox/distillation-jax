"""Steady state validation for distillation column simulation.

This module provides tools to validate that the column reaches
expected steady state conditions and matches analytical predictions.
"""

import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    ColumnOutputs,
    column_step,
    create_initial_column_state,
    create_default_action,
)
from jax_distillation.column.config import (
    ColumnConfig,
    create_teaching_column_config,
)


class SteadyStateMetrics(NamedTuple):
    """Metrics for steady state validation.

    Attributes:
        is_steady: True if state changes are below threshold.
        max_dx: Maximum composition change per step.
        max_dT: Maximum temperature change per step.
        max_dM: Maximum holdup change per step.
        x_D: Distillate composition at steady state.
        x_B: Bottoms composition at steady state.
    """

    is_steady: bool
    max_dx: float
    max_dT: float
    max_dM: float
    x_D: float
    x_B: float


def check_steady_state(
    state_old: FullColumnState,
    state_new: FullColumnState,
    dx_threshold: float = 1e-4,
    dT_threshold: float = 0.1,
    dM_threshold: float = 0.01,
) -> SteadyStateMetrics:
    """Check if column is at steady state.

    Args:
        state_old: State from previous step.
        state_new: State from current step.
        dx_threshold: Max composition change for steady state.
        dT_threshold: Max temperature change for steady state [K].
        dM_threshold: Max holdup change for steady state [mol].

    Returns:
        SteadyStateMetrics with validation results.
    """
    # Composition changes
    dx_tray = jnp.abs(state_new.tray_x - state_old.tray_x)
    dx_reboiler = jnp.abs(state_new.reboiler.x - state_old.reboiler.x)
    dx_condenser = jnp.abs(state_new.condenser.x - state_old.condenser.x)
    max_dx = float(jnp.max(jnp.array([jnp.max(dx_tray), dx_reboiler, dx_condenser])))

    # Temperature changes
    dT_tray = jnp.abs(state_new.tray_T - state_old.tray_T)
    dT_reboiler = jnp.abs(state_new.reboiler.T - state_old.reboiler.T)
    dT_condenser = jnp.abs(state_new.condenser.T - state_old.condenser.T)
    max_dT = float(jnp.max(jnp.array([jnp.max(dT_tray), dT_reboiler, dT_condenser])))

    # Holdup changes
    dM_tray = jnp.abs(state_new.tray_M - state_old.tray_M)
    dM_reboiler = jnp.abs(state_new.reboiler.M - state_old.reboiler.M)
    dM_condenser = jnp.abs(state_new.condenser.M - state_old.condenser.M)
    max_dM = float(jnp.max(jnp.array([jnp.max(dM_tray), dM_reboiler, dM_condenser])))

    is_steady = (max_dx < dx_threshold) and (max_dT < dT_threshold) and (max_dM < dM_threshold)

    return SteadyStateMetrics(
        is_steady=is_steady,
        max_dx=max_dx,
        max_dT=max_dT,
        max_dM=max_dM,
        x_D=float(state_new.condenser.x),
        x_B=float(state_new.reboiler.x),
    )


def run_to_steady_state(
    config: ColumnConfig,
    action: ColumnAction,
    max_steps: int = 1000,
    check_interval: int = 10,
    dx_threshold: float = 1e-4,
) -> tuple[FullColumnState, ColumnOutputs, int, bool]:
    """Run simulation until steady state is reached.

    Args:
        config: Column configuration.
        action: Control action to hold constant.
        max_steps: Maximum simulation steps.
        check_interval: Steps between steady state checks.
        dx_threshold: Composition change threshold for steady state.

    Returns:
        Tuple of (final_state, final_outputs, steps_taken, reached_steady).
    """
    state = create_initial_column_state(config)
    reached_steady = False

    for step in range(max_steps):
        state_old = state
        state, outputs = column_step(state, action, config)

        if step > 0 and step % check_interval == 0:
            metrics = check_steady_state(
                state_old, state, dx_threshold=dx_threshold
            )
            if metrics.is_steady:
                reached_steady = True
                break

    return state, outputs, step + 1, reached_steady


def fenske_minimum_stages(
    x_D: float,
    x_B: float,
    alpha: float,
) -> float:
    """Calculate minimum stages using Fenske equation.

    N_min = ln[(x_D/(1-x_D)) * ((1-x_B)/x_B)] / ln(alpha)

    Args:
        x_D: Distillate composition (light component).
        x_B: Bottoms composition (light component).
        alpha: Relative volatility.

    Returns:
        Minimum number of theoretical stages.
    """
    if x_D <= 0 or x_D >= 1 or x_B <= 0 or x_B >= 1:
        return float("inf")

    numerator = np.log((x_D / (1 - x_D)) * ((1 - x_B) / x_B))
    denominator = np.log(alpha)

    return numerator / denominator


def underwood_minimum_reflux(
    z_F: float,
    x_D: float,
    q: float,
    alpha: float,
) -> float:
    """Estimate minimum reflux ratio using Underwood equation.

    For binary mixture with constant relative volatility.

    Args:
        z_F: Feed composition.
        x_D: Distillate composition.
        q: Feed quality.
        alpha: Relative volatility.

    Returns:
        Minimum reflux ratio.
    """
    # Simplified for binary mixture
    # R_min = (1/(alpha-1)) * (x_D/z_F - alpha*(1-x_D)/(1-z_F))

    if alpha <= 1:
        return float("inf")

    term1 = x_D - alpha * (1 - x_D) * z_F / (1 - z_F)
    R_min = term1 / (alpha - 1) / z_F

    return max(0, R_min)


def validate_separation_quality(
    outputs: ColumnOutputs,
    config: ColumnConfig,
    expected_x_D_min: float = 0.9,
    expected_x_B_max: float = 0.1,
) -> dict:
    """Validate that separation meets expected quality.

    Args:
        outputs: Outputs at steady state.
        config: Column configuration.
        expected_x_D_min: Minimum expected distillate purity.
        expected_x_B_max: Maximum expected bottoms impurity.

    Returns:
        Dictionary with validation results.
    """
    x_D = float(outputs.x_D)
    x_B = float(outputs.x_B)
    z_F = float(config.feed.z_F)

    # Check that we have enrichment (x_D > z_F) and stripping (x_B < z_F)
    has_enrichment = x_D > z_F
    has_stripping = x_B < z_F
    meets_distillate_spec = x_D >= expected_x_D_min
    meets_bottoms_spec = x_B <= expected_x_B_max

    # Recovery calculations
    F = float(config.feed.F)
    D = float(outputs.D)
    B = float(outputs.B)

    if F > 0 and z_F > 0:
        light_recovery = (D * x_D) / (F * z_F) if D > 0 else 0.0
    else:
        light_recovery = 0.0

    return {
        "x_D": x_D,
        "x_B": x_B,
        "z_F": z_F,
        "has_enrichment": has_enrichment,
        "has_stripping": has_stripping,
        "meets_distillate_spec": meets_distillate_spec,
        "meets_bottoms_spec": meets_bottoms_spec,
        "light_recovery": light_recovery,
        "separation_factor": (x_D * (1 - x_B)) / ((1 - x_D) * x_B) if x_B > 0 and x_D < 1 else float("inf"),
    }


def validate_temperature_profile(
    state: FullColumnState,
) -> dict:
    """Validate that temperature profile is physically reasonable.

    Temperature should generally increase from top to bottom.

    Args:
        state: Column state at steady state.

    Returns:
        Dictionary with validation results.
    """
    tray_T = np.array(state.tray_T)
    reboiler_T = float(state.reboiler.T)
    condenser_T = float(state.condenser.T)

    # Check monotonicity (temperature increases downward)
    # Some non-monotonicity is acceptable near feed tray
    temp_diffs = np.diff(tray_T)
    n_inversions = np.sum(temp_diffs < -0.5)  # Allow small inversions

    is_monotonic = n_inversions <= len(tray_T) // 4  # Allow up to 25% inversions

    # Check that reboiler is hottest and condenser is coldest
    reboiler_hottest = reboiler_T >= np.max(tray_T) - 5.0
    condenser_coldest = condenser_T <= np.min(tray_T) + 5.0

    return {
        "tray_T_range": (float(np.min(tray_T)), float(np.max(tray_T))),
        "reboiler_T": reboiler_T,
        "condenser_T": condenser_T,
        "is_monotonic": is_monotonic,
        "n_inversions": int(n_inversions),
        "reboiler_hottest": reboiler_hottest,
        "condenser_coldest": condenser_coldest,
    }


def run_steady_state_validation(
    n_trays: int = 10,
    feed_composition: float = 0.5,
    reflux_ratio: float = 3.0,
    reboiler_duty: float = 5000.0,
) -> dict:
    """Run comprehensive steady state validation.

    Args:
        n_trays: Number of trays.
        feed_composition: Feed composition.
        reflux_ratio: Reflux ratio.
        reboiler_duty: Reboiler duty [W].

    Returns:
        Dictionary with all validation results.
    """
    config = create_teaching_column_config(
        n_trays=n_trays,
        feed_tray=n_trays // 2,
        feed_composition=feed_composition,
    )

    action = create_default_action(
        Q_R=reboiler_duty,
        reflux_ratio=reflux_ratio,
    )

    # Run to steady state
    state, outputs, steps, reached_steady = run_to_steady_state(
        config, action, max_steps=500
    )

    # Validate separation
    separation = validate_separation_quality(outputs, config)

    # Validate temperature profile
    temperature = validate_temperature_profile(state)

    return {
        "reached_steady_state": reached_steady,
        "steps_to_steady_state": steps,
        "separation": separation,
        "temperature": temperature,
        "final_state": {
            "x_D": float(outputs.x_D),
            "x_B": float(outputs.x_B),
            "D": float(outputs.D),
            "B": float(outputs.B),
            "Q_R": float(outputs.Q_R),
            "Q_C": float(outputs.Q_C),
        },
    }
