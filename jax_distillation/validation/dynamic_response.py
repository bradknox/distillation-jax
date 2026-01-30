"""Dynamic response validation for distillation column simulation.

This module provides tools to validate the dynamic behavior of the column:
- Step response characteristics
- Time constant validation
- Response direction verification
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
from jax_distillation.validation.steady_state import run_to_steady_state


class StepResponseMetrics(NamedTuple):
    """Metrics from step response analysis.

    Attributes:
        initial_value: Value before step.
        final_value: Value after reaching new steady state.
        change: Total change (final - initial).
        time_constant: Approximate time constant [s].
        rise_time: Time to reach 90% of change [s].
        overshoot: Overshoot as fraction of change.
        settling_time: Time to stay within 5% of final [s].
    """

    initial_value: float
    final_value: float
    change: float
    time_constant: float
    rise_time: float
    overshoot: float
    settling_time: float


def run_step_response(
    config: ColumnConfig,
    action_before: ColumnAction,
    action_after: ColumnAction,
    warmup_steps: int = 200,
    response_steps: int = 300,
) -> tuple[list[FullColumnState], list[ColumnOutputs], list[ColumnOutputs]]:
    """Run a step response experiment.

    Args:
        config: Column configuration.
        action_before: Action before step change.
        action_after: Action after step change.
        warmup_steps: Steps to run before step change.
        response_steps: Steps to run after step change.

    Returns:
        Tuple of (states, outputs_before, outputs_after).
    """
    # Warmup to initial steady state
    state = create_initial_column_state(config)
    outputs_before = []

    for _ in range(warmup_steps):
        state, outputs = column_step(state, action_before, config)
        outputs_before.append(outputs)

    # Apply step and record response
    states = [state]
    outputs_after = []

    for _ in range(response_steps):
        state, outputs = column_step(state, action_after, config)
        states.append(state)
        outputs_after.append(outputs)

    return states, outputs_before, outputs_after


def analyze_step_response(
    values: np.ndarray,
    dt: float,
) -> StepResponseMetrics:
    """Analyze step response from time series data.

    Args:
        values: Array of response values over time.
        dt: Time step [s].

    Returns:
        StepResponseMetrics with analysis results.
    """
    n = len(values)
    if n < 10:
        return StepResponseMetrics(
            initial_value=values[0] if n > 0 else 0.0,
            final_value=values[-1] if n > 0 else 0.0,
            change=0.0,
            time_constant=0.0,
            rise_time=0.0,
            overshoot=0.0,
            settling_time=0.0,
        )

    initial_value = values[0]
    final_value = values[-1]
    change = final_value - initial_value

    if abs(change) < 1e-10:
        return StepResponseMetrics(
            initial_value=initial_value,
            final_value=final_value,
            change=change,
            time_constant=0.0,
            rise_time=0.0,
            overshoot=0.0,
            settling_time=0.0,
        )

    # Normalize response
    normalized = (values - initial_value) / change

    # Time constant (time to reach 63.2%)
    idx_63 = np.argmax(normalized >= 0.632)
    time_constant = idx_63 * dt if idx_63 > 0 else n * dt

    # Rise time (10% to 90%)
    idx_10 = np.argmax(normalized >= 0.1)
    idx_90 = np.argmax(normalized >= 0.9)
    rise_time = (idx_90 - idx_10) * dt if idx_90 > idx_10 else n * dt

    # Overshoot
    if change > 0:
        max_value = np.max(values)
        overshoot = (max_value - final_value) / change if change != 0 else 0
    else:
        min_value = np.min(values)
        overshoot = (final_value - min_value) / abs(change) if change != 0 else 0

    overshoot = max(0, overshoot)

    # Settling time (time to stay within 5% of final)
    within_band = np.abs(values - final_value) <= 0.05 * abs(change)
    # Find last time outside band
    outside_indices = np.where(~within_band)[0]
    settling_time = (outside_indices[-1] + 1) * dt if len(outside_indices) > 0 else 0.0

    return StepResponseMetrics(
        initial_value=initial_value,
        final_value=final_value,
        change=change,
        time_constant=time_constant,
        rise_time=rise_time,
        overshoot=overshoot,
        settling_time=settling_time,
    )


def validate_response_direction(
    outputs_before: list[ColumnOutputs],
    outputs_after: list[ColumnOutputs],
    variable: str,
    expected_direction: str,  # "increase" or "decrease"
) -> dict:
    """Validate that response goes in expected direction.

    Args:
        outputs_before: Outputs before step.
        outputs_after: Outputs after step.
        variable: Variable to check ("x_D", "x_B", "D", "B", "Q_C").
        expected_direction: Expected direction of change.

    Returns:
        Dictionary with validation results.
    """
    # Get initial and final values
    initial = float(getattr(outputs_before[-1], variable))
    final = float(getattr(outputs_after[-1], variable))

    change = final - initial
    actual_direction = "increase" if change > 0 else "decrease" if change < 0 else "none"

    correct = (expected_direction == actual_direction) or (expected_direction == "none")

    return {
        "variable": variable,
        "initial": initial,
        "final": final,
        "change": change,
        "expected_direction": expected_direction,
        "actual_direction": actual_direction,
        "correct": correct,
    }


def run_reboiler_duty_step_test(
    config: ColumnConfig | None = None,
    Q_R_before: float = 3000.0,
    Q_R_after: float = 6000.0,
) -> dict:
    """Run step test on reboiler duty and validate response.

    Expected behavior:
    - Increasing Q_R should increase vapor flow
    - This should increase x_D (more light component in distillate)
    - And decrease x_B (less light component in bottoms)

    Args:
        config: Column configuration.
        Q_R_before: Reboiler duty before step [W].
        Q_R_after: Reboiler duty after step [W].

    Returns:
        Dictionary with test results.
    """
    if config is None:
        config = create_teaching_column_config(n_trays=5)

    action_before = create_default_action(Q_R=Q_R_before)
    action_after = create_default_action(Q_R=Q_R_after)

    states, outputs_before, outputs_after = run_step_response(
        config, action_before, action_after, warmup_steps=100, response_steps=200
    )

    # Extract time series
    dt = float(config.simulation.dt)
    x_D_values = np.array([float(o.x_D) for o in outputs_after])
    x_B_values = np.array([float(o.x_B) for o in outputs_after])

    # Analyze responses
    x_D_metrics = analyze_step_response(x_D_values, dt)
    x_B_metrics = analyze_step_response(x_B_values, dt)

    # Validate directions
    increasing_Q_R = Q_R_after > Q_R_before
    expected_x_D_dir = "increase" if increasing_Q_R else "decrease"
    expected_x_B_dir = "decrease" if increasing_Q_R else "increase"

    x_D_direction = validate_response_direction(
        outputs_before, outputs_after, "x_D", expected_x_D_dir
    )
    x_B_direction = validate_response_direction(
        outputs_before, outputs_after, "x_B", expected_x_B_dir
    )

    return {
        "step_input": {"Q_R_before": Q_R_before, "Q_R_after": Q_R_after},
        "x_D_response": {
            "metrics": x_D_metrics._asdict(),
            "direction_check": x_D_direction,
        },
        "x_B_response": {
            "metrics": x_B_metrics._asdict(),
            "direction_check": x_B_direction,
        },
        "all_directions_correct": x_D_direction["correct"] and x_B_direction["correct"],
    }


def run_reflux_ratio_step_test(
    config: ColumnConfig | None = None,
    RR_before: float = 2.0,
    RR_after: float = 4.0,
) -> dict:
    """Run step test on reflux ratio and validate response.

    Expected behavior:
    - Increasing reflux ratio should increase separation
    - This should increase x_D
    - And decrease x_B

    Args:
        config: Column configuration.
        RR_before: Reflux ratio before step.
        RR_after: Reflux ratio after step.

    Returns:
        Dictionary with test results.
    """
    if config is None:
        config = create_teaching_column_config(n_trays=5)

    action_before = create_default_action(reflux_ratio=RR_before)
    action_after = create_default_action(reflux_ratio=RR_after)

    states, outputs_before, outputs_after = run_step_response(
        config, action_before, action_after, warmup_steps=100, response_steps=200
    )

    dt = float(config.simulation.dt)
    x_D_values = np.array([float(o.x_D) for o in outputs_after])
    x_B_values = np.array([float(o.x_B) for o in outputs_after])

    x_D_metrics = analyze_step_response(x_D_values, dt)
    x_B_metrics = analyze_step_response(x_B_values, dt)

    increasing_RR = RR_after > RR_before
    expected_x_D_dir = "increase" if increasing_RR else "decrease"
    expected_x_B_dir = "decrease" if increasing_RR else "increase"

    x_D_direction = validate_response_direction(
        outputs_before, outputs_after, "x_D", expected_x_D_dir
    )
    x_B_direction = validate_response_direction(
        outputs_before, outputs_after, "x_B", expected_x_B_dir
    )

    return {
        "step_input": {"RR_before": RR_before, "RR_after": RR_after},
        "x_D_response": {
            "metrics": x_D_metrics._asdict(),
            "direction_check": x_D_direction,
        },
        "x_B_response": {
            "metrics": x_B_metrics._asdict(),
            "direction_check": x_B_direction,
        },
        "all_directions_correct": x_D_direction["correct"] and x_B_direction["correct"],
    }


def validate_time_constants(
    config: ColumnConfig | None = None,
    expected_tau_min: float = 5.0,
    expected_tau_max: float = 500.0,
) -> dict:
    """Validate that response time constants are physically reasonable.

    Distillation columns typically have time constants on the order of
    minutes to hours, depending on holdup and flow rates.

    Args:
        config: Column configuration.
        expected_tau_min: Minimum expected time constant [s].
        expected_tau_max: Maximum expected time constant [s].

    Returns:
        Dictionary with validation results.
    """
    if config is None:
        config = create_teaching_column_config(n_trays=5)

    # Run Q_R step test
    results = run_reboiler_duty_step_test(config)

    x_D_tau = results["x_D_response"]["metrics"]["time_constant"]
    x_B_tau = results["x_B_response"]["metrics"]["time_constant"]

    tau_in_range_x_D = expected_tau_min <= x_D_tau <= expected_tau_max
    tau_in_range_x_B = expected_tau_min <= x_B_tau <= expected_tau_max

    return {
        "x_D_time_constant": x_D_tau,
        "x_B_time_constant": x_B_tau,
        "expected_range": (expected_tau_min, expected_tau_max),
        "x_D_in_range": tau_in_range_x_D,
        "x_B_in_range": tau_in_range_x_B,
        "all_in_range": tau_in_range_x_D and tau_in_range_x_B,
    }
