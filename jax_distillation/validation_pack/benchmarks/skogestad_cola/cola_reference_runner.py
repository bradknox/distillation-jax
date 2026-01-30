"""Reference runner for Skogestad Column A benchmark.

This module runs the JAX simulator in COLA-like configurations
and collects trajectory data for comparison with published results.

Performance: Uses JIT-compiled jax.lax.scan for 10-100x speedup
over Python loops.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    ColumnOutputs,
    make_column_step_fn,
    create_initial_column_state,
)
from jax_distillation.column.config import ColumnConfig
from jax_distillation.validation_pack.benchmarks.skogestad_cola.cola_config_builder import (
    build_cola_config,
    get_cola_nominal_action,
    get_cola_parameters,
)


@dataclass
class ColaTrajectory:
    """Trajectory data from a COLA benchmark run.

    Attributes:
        times: Time points [s]
        x_D: Distillate composition trajectory
        x_B: Bottoms composition trajectory
        D: Distillate flow trajectory [mol/s]
        B: Bottoms flow trajectory [mol/s]
        T_top: Top tray temperature trajectory [K]
        T_bottom: Bottom tray temperature trajectory [K]
        T_profile: Temperature profile at final time (n_trays,)
        x_profile: Composition profile at final time (n_trays,)
    """

    times: np.ndarray
    x_D: np.ndarray
    x_B: np.ndarray
    D: np.ndarray
    B: np.ndarray
    T_top: np.ndarray
    T_bottom: np.ndarray
    T_profile: np.ndarray
    x_profile: np.ndarray


def run_cola_steady_state(
    config: Optional[ColumnConfig] = None,
    action: Optional[ColumnAction] = None,
    n_steps: int = 2000,
    steady_state_tol: float = 1e-5,
) -> Tuple[FullColumnState, ColumnOutputs, ColaTrajectory]:
    """Run simulation to reach COLA nominal steady state.

    Uses JIT-compiled jax.lax.scan for performance (10-100x faster than
    Python loops).

    Args:
        config: Column configuration (uses COLA default if None).
        action: Control action (uses COLA nominal if None).
        n_steps: Maximum number of steps.
        steady_state_tol: Tolerance for steady-state detection (checked post-hoc).

    Returns:
        Tuple of (final_state, final_outputs, trajectory).
    """
    if config is None:
        config = build_cola_config()
    if action is None:
        action = get_cola_nominal_action()

    state = create_initial_column_state(config, action=action)
    dt = float(config.simulation.dt)

    # Create JIT-compiled step function
    step_fn = make_column_step_fn(config)

    # Define scan body that collects trajectory data
    def scan_body(state, _):
        new_state, outputs = step_fn(state, action)
        # Collect trajectory data in a pytree
        traj_point = {
            "t": new_state.t,
            "x_D": outputs.x_D,
            "x_B": outputs.x_B,
            "D": outputs.D,
            "B": outputs.B,
            "T_top": new_state.tray_T[0],
            "T_bottom": new_state.tray_T[-1],
        }
        return new_state, traj_point

    # JIT-compile and run the scan
    @jax.jit
    def run_simulation(initial_state):
        return jax.lax.scan(scan_body, initial_state, None, length=n_steps)

    final_state, traj_data = run_simulation(state)

    # Get final outputs
    _, final_outputs = step_fn(final_state, action)

    # Convert JAX arrays to numpy for trajectory
    trajectory = ColaTrajectory(
        times=np.asarray(traj_data["t"]),
        x_D=np.asarray(traj_data["x_D"]),
        x_B=np.asarray(traj_data["x_B"]),
        D=np.asarray(traj_data["D"]),
        B=np.asarray(traj_data["B"]),
        T_top=np.asarray(traj_data["T_top"]),
        T_bottom=np.asarray(traj_data["T_bottom"]),
        T_profile=np.asarray(final_state.tray_T),
        x_profile=np.asarray(final_state.tray_x),
    )

    # Check for steady state (post-hoc)
    if len(trajectory.x_D) > 1:
        dx_D = np.abs(np.diff(trajectory.x_D))
        steady_idx = np.where(dx_D < steady_state_tol)[0]
        if len(steady_idx) > 0:
            print(f"Steady state reached at step {steady_idx[0] + 1}")

    return final_state, final_outputs, trajectory


def run_cola_step_response(
    variable: str = "reflux",
    step_size: float = 0.01,
    config: Optional[ColumnConfig] = None,
    warmup_steps: int = 1000,
    response_steps: int = 500,
) -> Tuple[ColaTrajectory, ColaTrajectory]:
    """Run step response experiment for COLA validation.

    Uses JIT-compiled jax.lax.scan for performance (10-100x faster than
    Python loops).

    Args:
        variable: Which variable to step ("reflux" or "boilup").
        step_size: Relative step size (e.g., 0.01 = 1%).
        config: Column configuration (uses COLA default if None).
        warmup_steps: Steps to reach steady state before step.
        response_steps: Steps to record after step change.

    Returns:
        Tuple of (baseline_trajectory, step_trajectory).
    """
    if config is None:
        config = build_cola_config()

    action_base = get_cola_nominal_action()

    # Create JIT-compiled step function
    step_fn = make_column_step_fn(config)

    # Define scan bodies
    def warmup_body(state, _):
        new_state, outputs = step_fn(state, action_base)
        return new_state, None

    def response_body_base(state, _):
        new_state, outputs = step_fn(state, action_base)
        traj_point = {
            "t": new_state.t,
            "x_D": outputs.x_D,
            "x_B": outputs.x_B,
        }
        return new_state, traj_point

    # JIT-compile simulation functions
    @jax.jit
    def run_warmup(initial_state):
        final_state, _ = jax.lax.scan(warmup_body, initial_state, None, length=warmup_steps)
        return final_state

    @jax.jit
    def run_baseline(initial_state):
        return jax.lax.scan(response_body_base, initial_state, None, length=response_steps)

    # Run warmup
    state = create_initial_column_state(config, action=action_base)
    state = run_warmup(state)
    baseline_state = state

    # Run baseline response
    _, baseline_data = run_baseline(baseline_state)

    baseline_times = np.asarray(baseline_data["t"])
    baseline_trajectory = ColaTrajectory(
        times=baseline_times - float(baseline_times[0]),
        x_D=np.asarray(baseline_data["x_D"]),
        x_B=np.asarray(baseline_data["x_B"]),
        D=np.zeros(response_steps),
        B=np.zeros(response_steps),
        T_top=np.zeros(response_steps),
        T_bottom=np.zeros(response_steps),
        T_profile=np.asarray(baseline_state.tray_T),
        x_profile=np.asarray(baseline_state.tray_x),
    )

    # Create stepped action
    if variable == "reflux":
        action_step = ColumnAction(
            Q_R=action_base.Q_R,
            reflux_ratio=action_base.reflux_ratio * (1 + step_size),
            B_setpoint=action_base.B_setpoint,
            D_setpoint=action_base.D_setpoint,
        )
    elif variable == "boilup":
        action_step = ColumnAction(
            Q_R=action_base.Q_R * (1 + step_size),
            reflux_ratio=action_base.reflux_ratio,
            B_setpoint=action_base.B_setpoint,
            D_setpoint=action_base.D_setpoint,
        )
    else:
        raise ValueError(f"Unknown variable: {variable}")

    # Define step response body with stepped action
    def response_body_step(state, _):
        new_state, outputs = step_fn(state, action_step)
        traj_point = {
            "t": new_state.t,
            "x_D": outputs.x_D,
            "x_B": outputs.x_B,
            "D": outputs.D,
            "B": outputs.B,
            "T_top": new_state.tray_T[0],
            "T_bottom": new_state.tray_T[-1],
        }
        return new_state, traj_point

    @jax.jit
    def run_step_response(initial_state):
        return jax.lax.scan(response_body_step, initial_state, None, length=response_steps)

    # Run step response from baseline state
    final_state, step_data = run_step_response(baseline_state)

    step_times = np.asarray(step_data["t"])
    step_trajectory = ColaTrajectory(
        times=step_times - float(step_times[0]),
        x_D=np.asarray(step_data["x_D"]),
        x_B=np.asarray(step_data["x_B"]),
        D=np.asarray(step_data["D"]),
        B=np.asarray(step_data["B"]),
        T_top=np.asarray(step_data["T_top"]),
        T_bottom=np.asarray(step_data["T_bottom"]),
        T_profile=np.asarray(final_state.tray_T),
        x_profile=np.asarray(final_state.tray_x),
    )

    return baseline_trajectory, step_trajectory


def run_cola_benchmark(
    config: Optional[ColumnConfig] = None,
) -> dict:
    """Run complete COLA benchmark validation.

    Runs steady-state and step response tests, returning all
    trajectory data for analysis.

    Args:
        config: Column configuration (uses COLA default if None).

    Returns:
        Dict containing all benchmark results.
    """
    if config is None:
        config = build_cola_config()

    results = {}

    # Steady-state test
    print("Running steady-state test...")
    final_state, final_outputs, ss_trajectory = run_cola_steady_state(config)
    results["steady_state"] = {
        "trajectory": ss_trajectory,
        "final_x_D": float(final_outputs.x_D),
        "final_x_B": float(final_outputs.x_B),
        "final_D": float(final_outputs.D),
        "final_B": float(final_outputs.B),
    }

    # Reflux step test
    print("Running reflux step response...")
    baseline_R, step_R = run_cola_step_response(
        variable="reflux", step_size=0.01, config=config
    )
    results["reflux_step"] = {
        "baseline": baseline_R,
        "step": step_R,
        "step_size": 0.01,
    }

    # Boilup step test
    print("Running boilup step response...")
    baseline_Q, step_Q = run_cola_step_response(
        variable="boilup", step_size=0.01, config=config
    )
    results["boilup_step"] = {
        "baseline": baseline_Q,
        "step": step_Q,
        "step_size": 0.01,
    }

    # Get COLA reference values
    cola = get_cola_parameters()
    results["reference"] = {
        "x_D_target": cola.x_D,
        "x_B_target": cola.x_B,
        "D_target": cola.D,
        "B_target": cola.B,
    }

    return results


if __name__ == "__main__":
    results = run_cola_benchmark()

    print("\n" + "=" * 60)
    print("COLA Benchmark Results")
    print("=" * 60)

    ss = results["steady_state"]
    ref = results["reference"]

    print("\nSteady-State Comparison:")
    print(f"  x_D: {ss['final_x_D']:.4f} (target: {ref['x_D_target']:.4f})")
    print(f"  x_B: {ss['final_x_B']:.4f} (target: {ref['x_B_target']:.4f})")

    r_step = results["reflux_step"]
    print("\nReflux Step Response:")
    print(f"  Initial x_D: {r_step['baseline'].x_D[0]:.4f}")
    print(f"  Final x_D:   {r_step['step'].x_D[-1]:.4f}")
    print(f"  Change:      {r_step['step'].x_D[-1] - r_step['baseline'].x_D[0]:.6f}")

    q_step = results["boilup_step"]
    print("\nBoilup Step Response:")
    print(f"  Initial x_B: {q_step['baseline'].x_B[0]:.4f}")
    print(f"  Final x_B:   {q_step['step'].x_B[-1]:.4f}")
    print(f"  Change:      {q_step['step'].x_B[-1] - q_step['baseline'].x_B[0]:.6f}")
