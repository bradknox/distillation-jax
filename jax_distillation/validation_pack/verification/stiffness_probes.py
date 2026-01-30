"""Time scale analysis and stiffness probing.

This module analyzes the time scales present in the simulation
to verify that the integration scheme is appropriate and to
identify potential stiffness issues.

Time scales modeled:
- Hydraulic: τ_L ~ 0.5-15 s (liquid level response)
- Thermal: ~ 10-100 s (temperature averaging)
- Composition: ~ 100-1000 s (mass transfer + holdup effects)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np

from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    column_step,
    create_initial_column_state,
    create_default_action,
)
from jax_distillation.column.config import ColumnConfig, create_teaching_column_config


@dataclass
class StiffnessAnalysisResult:
    """Result of stiffness/time scale analysis.

    Attributes:
        shortest_time_scale: Estimated shortest time scale [s]
        longest_time_scale: Estimated longest time scale [s]
        stiffness_ratio: Ratio of longest to shortest time scale
        recommended_dt: Recommended timestep based on fastest dynamics [s]
        current_dt: Current simulation timestep [s]
        dt_stability_ok: True if current dt is appropriate
        hydraulic_time_scale: Estimated hydraulic time constant [s]
        thermal_time_scale: Estimated thermal time constant [s]
        composition_time_scale: Estimated composition time constant [s]
        eigenvalue_analysis_available: True if eigenvalues were computed
    """

    shortest_time_scale: float
    longest_time_scale: float
    stiffness_ratio: float
    recommended_dt: float
    current_dt: float
    dt_stability_ok: bool
    hydraulic_time_scale: float
    thermal_time_scale: float
    composition_time_scale: float
    eigenvalue_analysis_available: bool


def _estimate_time_scale_from_step_response(
    values: np.ndarray,
    times: np.ndarray,
    method: str = "63pct",
) -> float:
    """Estimate time scale from step response data.

    Args:
        values: Response values over time.
        times: Time points [s].
        method: Method to use ("63pct" for 63% of final value).

    Returns:
        Estimated time constant [s].
    """
    if len(values) < 2:
        return float("inf")

    # Normalize to [0, 1] range
    v_min, v_max = values.min(), values.max()
    if abs(v_max - v_min) < 1e-10:
        return float("inf")

    normalized = (values - v_min) / (v_max - v_min)

    # Find time to reach 63% of final value
    target = 0.632  # 1 - e^(-1)

    # Check if response is increasing or decreasing
    if normalized[-1] < normalized[0]:
        normalized = 1 - normalized

    # Find crossing point
    for i, v in enumerate(normalized):
        if v >= target:
            if i == 0:
                return times[0]
            # Linear interpolation
            t_prev, t_curr = times[i - 1], times[i]
            v_prev, v_curr = normalized[i - 1], normalized[i]
            tau = t_prev + (target - v_prev) / (v_curr - v_prev) * (t_curr - t_prev)
            return tau

    return times[-1]  # Didn't reach 63%


def _run_step_response_probe(
    config: ColumnConfig,
    action_base: ColumnAction,
    action_step: ColumnAction,
    n_steps: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a step response experiment to probe time scales.

    Args:
        config: Column configuration.
        action_base: Baseline control action.
        action_step: Control action after step change.
        n_steps: Number of simulation steps.

    Returns:
        Tuple of (times, x_D_values, x_B_values, T_avg_values).
    """
    state = create_initial_column_state(config)
    dt = float(config.simulation.dt)

    times = []
    x_D_values = []
    x_B_values = []
    T_avg_values = []

    # Run with baseline action first (warmup)
    for _ in range(100):
        state, _ = column_step(state, action_base, config)

    # Apply step change
    for step in range(n_steps):
        state, outputs = column_step(state, action_step, config)

        times.append(float(state.t))
        x_D_values.append(float(outputs.x_D))
        x_B_values.append(float(outputs.x_B))
        T_avg_values.append(float(jnp.mean(state.tray_T)))

    return (
        np.array(times),
        np.array(x_D_values),
        np.array(x_B_values),
        np.array(T_avg_values),
    )


def analyze_time_scales(
    config: Optional[ColumnConfig] = None,
    n_steps: int = 500,
) -> StiffnessAnalysisResult:
    """Analyze time scales in the simulation.

    Applies step changes to reboiler duty and reflux ratio
    to estimate the dominant time scales for each response.

    Args:
        config: Column configuration (uses default if None).
        n_steps: Number of steps for step response.

    Returns:
        StiffnessAnalysisResult with time scale analysis.
    """
    if config is None:
        config = create_teaching_column_config()

    current_dt = float(config.simulation.dt)

    # Create baseline and stepped actions
    action_base = create_default_action()

    # Step 1: Reboiler duty step (probes thermal/composition dynamics)
    action_Q_step = ColumnAction(
        Q_R=action_base.Q_R * 1.1,  # 10% increase
        reflux_ratio=action_base.reflux_ratio,
        B_setpoint=action_base.B_setpoint,
        D_setpoint=action_base.D_setpoint,
    )

    times, x_D, x_B, T_avg = _run_step_response_probe(
        config, action_base, action_Q_step, n_steps
    )

    # Estimate time scales from responses
    tau_composition_D = _estimate_time_scale_from_step_response(x_D, times)
    tau_composition_B = _estimate_time_scale_from_step_response(x_B, times)
    tau_thermal = _estimate_time_scale_from_step_response(T_avg, times)

    # Step 2: Reflux step (different dynamics)
    action_R_step = ColumnAction(
        Q_R=action_base.Q_R,
        reflux_ratio=action_base.reflux_ratio * 1.1,
        B_setpoint=action_base.B_setpoint,
        D_setpoint=action_base.D_setpoint,
    )

    times2, x_D2, x_B2, _ = _run_step_response_probe(
        config, action_base, action_R_step, n_steps
    )

    tau_reflux_D = _estimate_time_scale_from_step_response(x_D2, times2)
    tau_reflux_B = _estimate_time_scale_from_step_response(x_B2, times2)

    # Combine estimates
    composition_taus = [tau_composition_D, tau_composition_B, tau_reflux_D, tau_reflux_B]
    composition_taus = [t for t in composition_taus if np.isfinite(t) and t > 0]

    if composition_taus:
        composition_time_scale = np.median(composition_taus)
    else:
        composition_time_scale = 100.0  # Default estimate

    thermal_time_scale = tau_thermal if np.isfinite(tau_thermal) and tau_thermal > 0 else 30.0

    # Hydraulic time scale from typical values
    # This would require analyzing holdup dynamics, use typical range
    hydraulic_time_scale = 5.0  # Typical for teaching columns

    # Calculate stiffness metrics
    time_scales = [hydraulic_time_scale, thermal_time_scale, composition_time_scale]
    shortest = min(time_scales)
    longest = max(time_scales)
    stiffness_ratio = longest / shortest if shortest > 0 else float("inf")

    # Recommended dt (should be less than shortest time scale / 10)
    recommended_dt = shortest / 10.0

    # Check if current dt is appropriate
    dt_stability_ok = current_dt <= shortest / 5.0

    return StiffnessAnalysisResult(
        shortest_time_scale=shortest,
        longest_time_scale=longest,
        stiffness_ratio=stiffness_ratio,
        recommended_dt=recommended_dt,
        current_dt=current_dt,
        dt_stability_ok=dt_stability_ok,
        hydraulic_time_scale=hydraulic_time_scale,
        thermal_time_scale=thermal_time_scale,
        composition_time_scale=composition_time_scale,
        eigenvalue_analysis_available=False,
    )


def print_stiffness_analysis_report(result: StiffnessAnalysisResult) -> None:
    """Print a formatted stiffness analysis report.

    Args:
        result: StiffnessAnalysisResult from analyze_time_scales.
    """
    print("=" * 70)
    print("TIME SCALE / STIFFNESS ANALYSIS")
    print("=" * 70)

    print("\nEstimated Time Scales:")
    print(f"  Hydraulic:   {result.hydraulic_time_scale:8.2f} s")
    print(f"  Thermal:     {result.thermal_time_scale:8.2f} s")
    print(f"  Composition: {result.composition_time_scale:8.2f} s")

    print(f"\nTime Scale Range:")
    print(f"  Shortest: {result.shortest_time_scale:.2f} s")
    print(f"  Longest:  {result.longest_time_scale:.2f} s")
    print(f"  Stiffness ratio: {result.stiffness_ratio:.1f}")

    print(f"\nTimestep Analysis:")
    print(f"  Current dt:     {result.current_dt:.4f} s")
    print(f"  Recommended dt: {result.recommended_dt:.4f} s")

    status = "OK" if result.dt_stability_ok else "TOO LARGE"
    print(f"  Stability: {status}")

    print("\n" + "=" * 70)
    if result.dt_stability_ok:
        print("CONCLUSION: Timestep is appropriate for dynamics")
    else:
        print(f"WARNING: Consider reducing dt to {result.recommended_dt:.4f} s or less")
    print("=" * 70)


def plot_step_response_analysis(
    config: Optional[ColumnConfig] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Generate step response plots for time scale visualization.

    Args:
        config: Column configuration (uses default if None).
        output_dir: Directory to save plots.

    Returns:
        Path to the saved plot file.
    """
    import matplotlib.pyplot as plt

    if config is None:
        config = create_teaching_column_config()

    action_base = create_default_action()
    action_step = ColumnAction(
        Q_R=action_base.Q_R * 1.1,
        reflux_ratio=action_base.reflux_ratio,
        B_setpoint=action_base.B_setpoint,
        D_setpoint=action_base.D_setpoint,
    )

    times, x_D, x_B, T_avg = _run_step_response_probe(config, action_base, action_step, 500)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(times, x_D, "b-", linewidth=1.5)
    axes[0].set_ylabel("x_D (distillate)")
    axes[0].set_title("Step Response to 10% Reboiler Duty Increase")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, x_B, "r-", linewidth=1.5)
    axes[1].set_ylabel("x_B (bottoms)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, T_avg, "g-", linewidth=1.5)
    axes[2].set_ylabel("Avg Temperature [K]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is None:
        output_dir = "artifacts/plots"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "step_response_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


if __name__ == "__main__":
    result = analyze_time_scales()
    print_stiffness_analysis_report(result)
