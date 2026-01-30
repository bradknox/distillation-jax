"""Timestep convergence study for numerical verification.

This module verifies that the simulator converges to a consistent
solution as the timestep is refined (dt, dt/2, dt/4, ...).

Acceptance criteria:
- Solutions converge as dt decreases
- Convergence rate matches expected order (e.g., first-order for Euler)
- Convergence plots demonstrate stable behavior
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
from jax_distillation.column.config import (
    ColumnConfig,
    create_teaching_column_config,
    SimulationParams,
)


@dataclass
class TimestepConvergenceResult:
    """Result of timestep convergence study.

    Attributes:
        timesteps: List of timestep sizes tested [s]
        final_x_D: Final distillate composition for each timestep
        final_x_B: Final bottoms composition for each timestep
        final_T_avg: Final average temperature for each timestep
        x_D_convergence_rate: Estimated convergence rate for x_D
        x_B_convergence_rate: Estimated convergence rate for x_B
        converged: True if solutions show convergence trend
        simulation_time: Total simulation time [s]
        refinement_factor: Factor between successive timesteps
    """

    timesteps: List[float]
    final_x_D: List[float]
    final_x_B: List[float]
    final_T_avg: List[float]
    x_D_convergence_rate: float
    x_B_convergence_rate: float
    converged: bool
    simulation_time: float
    refinement_factor: float = 2.0


def _run_simulation_with_dt(
    config: ColumnConfig,
    action: ColumnAction,
    dt: float,
    total_time: float,
) -> Tuple[FullColumnState, float, float, float]:
    """Run simulation with a specific timestep.

    Args:
        config: Base column configuration.
        action: Control action.
        dt: Timestep size [s].
        total_time: Total simulation time [s].

    Returns:
        Tuple of (final_state, x_D, x_B, T_avg).
    """
    # Create config with modified timestep
    sim_params = SimulationParams(
        dt=dt,
        n_substeps=config.simulation.n_substeps,
        murphree_efficiency=config.simulation.murphree_efficiency,
    )

    modified_config = ColumnConfig(
        geometry=config.geometry,
        thermo=config.thermo,
        feed=config.feed,
        controllers=config.controllers,
        simulation=sim_params,
        hydraulics=config.hydraulics,
        P=config.P,
    )

    # Calculate number of steps
    n_steps = int(total_time / dt)

    # Initialize and run
    state = create_initial_column_state(modified_config)

    for _ in range(n_steps):
        state, outputs = column_step(state, action, modified_config)

    # Extract final values
    x_D = float(outputs.x_D)
    x_B = float(outputs.x_B)
    T_avg = float(jnp.mean(state.tray_T))

    return state, x_D, x_B, T_avg


def _estimate_convergence_rate(
    values: List[float],
    timesteps: List[float],
) -> float:
    """Estimate convergence rate from Richardson extrapolation.

    For a method of order p: error ~ C * dt^p
    Rate p = log(|v2 - v1| / |v3 - v2|) / log(r)

    where r is the refinement ratio.

    Args:
        values: Solution values at different timesteps.
        timesteps: Corresponding timestep sizes.

    Returns:
        Estimated convergence rate (order).
    """
    if len(values) < 3:
        return 0.0

    # Use last three values
    v1, v2, v3 = values[-3:]
    dt1, dt2, dt3 = timesteps[-3:]

    # Refinement ratio
    r = dt1 / dt2

    # Differences
    diff1 = abs(v2 - v1)
    diff2 = abs(v3 - v2)

    if diff2 < 1e-12:
        return float("inf")  # Already converged

    if diff1 < 1e-12:
        return 0.0

    rate = np.log(diff1 / diff2) / np.log(r)
    return max(0.0, rate)  # Rate should be positive


def run_timestep_convergence(
    config: Optional[ColumnConfig] = None,
    action: Optional[ColumnAction] = None,
    base_dt: float = 1.0,
    n_refinements: int = 4,
    refinement_factor: float = 2.0,
    total_time: float = 100.0,
) -> TimestepConvergenceResult:
    """Run timestep convergence study.

    Args:
        config: Column configuration (uses default if None).
        action: Control action (uses default if None).
        base_dt: Starting (coarsest) timestep [s].
        n_refinements: Number of timestep refinements.
        refinement_factor: Factor to reduce dt by each refinement.
        total_time: Total simulation time [s].

    Returns:
        TimestepConvergenceResult with convergence data.
    """
    if config is None:
        config = create_teaching_column_config()
    if action is None:
        action = create_default_action()

    timesteps = []
    final_x_D = []
    final_x_B = []
    final_T_avg = []

    dt = base_dt
    for i in range(n_refinements):
        print(f"Running with dt = {dt:.4f} s...")

        _, x_D, x_B, T_avg = _run_simulation_with_dt(config, action, dt, total_time)

        timesteps.append(dt)
        final_x_D.append(x_D)
        final_x_B.append(x_B)
        final_T_avg.append(T_avg)

        dt /= refinement_factor

    # Estimate convergence rates
    x_D_rate = _estimate_convergence_rate(final_x_D, timesteps)
    x_B_rate = _estimate_convergence_rate(final_x_B, timesteps)

    # Check if converged (differences decreasing)
    if len(final_x_D) >= 3:
        diff1 = abs(final_x_D[-2] - final_x_D[-3])
        diff2 = abs(final_x_D[-1] - final_x_D[-2])
        converged = diff2 < diff1 * 0.9  # Differences should decrease
    else:
        converged = True

    return TimestepConvergenceResult(
        timesteps=timesteps,
        final_x_D=final_x_D,
        final_x_B=final_x_B,
        final_T_avg=final_T_avg,
        x_D_convergence_rate=x_D_rate,
        x_B_convergence_rate=x_B_rate,
        converged=converged,
        simulation_time=total_time,
        refinement_factor=refinement_factor,
    )


def plot_convergence(
    result: TimestepConvergenceResult,
    output_dir: Optional[str] = None,
) -> str:
    """Generate convergence plots.

    Args:
        result: TimestepConvergenceResult from run_timestep_convergence.
        output_dir: Directory to save plots (creates if needed).

    Returns:
        Path to the saved plot file.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: x_D vs dt
    ax1 = axes[0]
    ax1.semilogx(result.timesteps, result.final_x_D, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Timestep dt [s]")
    ax1.set_ylabel("Final x_D")
    ax1.set_title(f"Distillate Composition (rate ≈ {result.x_D_convergence_rate:.2f})")
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Plot 2: x_B vs dt
    ax2 = axes[1]
    ax2.semilogx(result.timesteps, result.final_x_B, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Timestep dt [s]")
    ax2.set_ylabel("Final x_B")
    ax2.set_title(f"Bottoms Composition (rate ≈ {result.x_B_convergence_rate:.2f})")
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Plot 3: Average temperature vs dt
    ax3 = axes[2]
    ax3.semilogx(result.timesteps, result.final_T_avg, "go-", linewidth=2, markersize=8)
    ax3.set_xlabel("Timestep dt [s]")
    ax3.set_ylabel("Average Temperature [K]")
    ax3.set_title("Average Tray Temperature")
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    plt.suptitle(
        f"Timestep Convergence Study (t = {result.simulation_time:.1f} s)",
        fontsize=14,
    )
    plt.tight_layout()

    if output_dir is None:
        output_dir = "artifacts/plots"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "timestep_convergence.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def print_timestep_convergence_report(result: TimestepConvergenceResult) -> None:
    """Print a formatted convergence report.

    Args:
        result: TimestepConvergenceResult from run_timestep_convergence.
    """
    print("=" * 70)
    print("TIMESTEP CONVERGENCE STUDY")
    print("=" * 70)

    print(f"\nSimulation time: {result.simulation_time:.1f} s")
    print(f"Refinement factor: {result.refinement_factor:.1f}")
    print(f"Timesteps tested: {len(result.timesteps)}")

    print("\nResults:")
    print("  dt [s]      x_D         x_B         T_avg [K]")
    print("  " + "-" * 50)
    for i, dt in enumerate(result.timesteps):
        print(
            f"  {dt:8.4f}    {result.final_x_D[i]:.6f}    "
            f"{result.final_x_B[i]:.6f}    {result.final_T_avg[i]:.2f}"
        )

    print(f"\nConvergence rates:")
    print(f"  x_D: {result.x_D_convergence_rate:.2f}")
    print(f"  x_B: {result.x_B_convergence_rate:.2f}")

    print("\n" + "=" * 70)
    status = "CONVERGED" if result.converged else "NOT CONVERGED"
    print(f"OVERALL: {status}")
    print("=" * 70)


if __name__ == "__main__":
    result = run_timestep_convergence(n_refinements=4, total_time=100.0)
    print_timestep_convergence_report(result)

    try:
        plot_path = plot_convergence(result)
        print(f"\nPlot saved to: {plot_path}")
    except ImportError:
        print("\nNote: matplotlib not available for plotting")
