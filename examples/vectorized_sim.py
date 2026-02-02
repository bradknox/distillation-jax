#!/usr/bin/env python3
"""Vectorized (parallel) simulation example using JAX vmap.

This script demonstrates how to:
1. Run multiple simulations in parallel using vmap
2. Sweep over parameter ranges efficiently
3. Leverage JAX's automatic vectorization

Run with: python examples/vectorized_sim.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax_distillation.column.config import create_teaching_column_config
from jax_distillation.column.column import (
    FullColumnState,
    ColumnAction,
    column_step,
    create_initial_column_state,
    create_default_action,
)


def create_batched_actions(
    Q_R_values: jnp.ndarray,
    reflux_ratio: float = 3.0,
    B_setpoint: float = 0.03,
    D_setpoint: float = 0.02,
) -> ColumnAction:
    """Create a batch of actions with different Q_R values.

    Args:
        Q_R_values: Array of reboiler duty values [W].
        reflux_ratio: Fixed reflux ratio.
        B_setpoint: Fixed bottoms setpoint.
        D_setpoint: Fixed distillate setpoint.

    Returns:
        ColumnAction with batched Q_R dimension.
    """
    batch_size = len(Q_R_values)
    return ColumnAction(
        Q_R=Q_R_values,
        reflux_ratio=jnp.full(batch_size, reflux_ratio),
        B_setpoint=jnp.full(batch_size, B_setpoint),
        D_setpoint=jnp.full(batch_size, D_setpoint),
    )


def create_batched_states(
    base_state: FullColumnState,
    batch_size: int,
) -> FullColumnState:
    """Replicate a state across a batch dimension.

    Args:
        base_state: Single column state.
        batch_size: Number of copies to create.

    Returns:
        FullColumnState with batch dimension added.
    """
    return FullColumnState(
        tray_M=jnp.tile(base_state.tray_M[None, :], (batch_size, 1)),
        tray_x=jnp.tile(base_state.tray_x[None, :], (batch_size, 1)),
        tray_T=jnp.tile(base_state.tray_T[None, :], (batch_size, 1)),
        tray_L_out=jnp.tile(base_state.tray_L_out[None, :], (batch_size, 1)),
        reboiler=base_state.reboiler,  # Shared (will be broadcast)
        condenser=base_state.condenser,  # Shared (will be broadcast)
        t=jnp.tile(base_state.t[None], (batch_size,)),
        V_prev=jnp.tile(base_state.V_prev[None, :], (batch_size, 1)),
    )


def main():
    print("=" * 60)
    print("Vectorized Simulation with JAX vmap")
    print("=" * 60)

    # Create base configuration
    config = create_teaching_column_config(n_trays=8, feed_tray=4)
    base_state = create_initial_column_state(config)

    # Define parameter sweep: different reboiler duties
    n_sims = 16
    Q_R_values = jnp.linspace(2000.0, 10000.0, n_sims)

    print(f"\nRunning {n_sims} parallel simulations")
    print(f"Q_R range: {float(Q_R_values[0]):.0f} - {float(Q_R_values[-1]):.0f} W")

    # Create batched inputs
    batched_states = create_batched_states(base_state, n_sims)
    batched_actions = create_batched_actions(Q_R_values)

    # Create vmapped step function
    # We vmap over the state and action, keeping config fixed
    def single_step(state, action):
        return column_step(state, action, config)

    # For full vmap, we need to handle the nested structure carefully
    # Here we demonstrate a simpler approach: run steps sequentially but
    # process multiple parameter values

    print("\n--- Method 1: Sequential with JIT ---")

    @jax.jit
    def run_simulation(Q_R, n_steps=50):
        """Run a single simulation to near steady state."""
        state = create_initial_column_state(config)
        action = create_default_action(Q_R=Q_R)

        def step_fn(state, _):
            new_state, outputs = column_step(state, action, config)
            return new_state, outputs

        final_state, outputs = jax.lax.scan(step_fn, state, None, length=n_steps)
        return final_state, outputs

    # Warmup JIT
    _ = run_simulation(5000.0)

    # Time sequential execution
    start = time.perf_counter()
    results_sequential = []
    for Q_R in Q_R_values:
        final_state, outputs = run_simulation(float(Q_R))
        results_sequential.append({
            "Q_R": float(Q_R),
            "x_D": float(outputs.x_D[-1]),
            "x_B": float(outputs.x_B[-1]),
        })
    jax.block_until_ready(results_sequential[-1]["x_D"])
    time_sequential = time.perf_counter() - start

    print(f"Sequential time: {time_sequential:.3f} s")
    print(f"Time per simulation: {time_sequential/n_sims*1000:.1f} ms")

    print("\n--- Method 2: Batched with vmap ---")

    # Create a vmapped version that processes all Q_R values at once
    @jax.jit
    def run_batched_simulation(Q_R_array, n_steps=50):
        """Run multiple simulations in parallel."""

        def run_one(Q_R):
            state = create_initial_column_state(config)
            action = create_default_action(Q_R=Q_R)

            def step_fn(state, _):
                new_state, outputs = column_step(state, action, config)
                return new_state, outputs

            final_state, outputs = jax.lax.scan(step_fn, state, None, length=n_steps)
            return outputs.x_D[-1], outputs.x_B[-1]

        return jax.vmap(run_one)(Q_R_array)

    # Warmup
    _ = run_batched_simulation(Q_R_values[:2])

    # Time batched execution
    start = time.perf_counter()
    x_D_batched, x_B_batched = run_batched_simulation(Q_R_values)
    jax.block_until_ready(x_D_batched)
    time_batched = time.perf_counter() - start

    print(f"Batched time: {time_batched:.3f} s")
    print(f"Speedup: {time_sequential/time_batched:.1f}x")

    # Verify results match
    x_D_seq = np.array([r["x_D"] for r in results_sequential])
    x_D_bat = np.array(x_D_batched)
    print(f"\nResults match: {np.allclose(x_D_seq, x_D_bat, rtol=1e-4)}")

    # Print results table
    print("\n" + "-" * 50)
    print(f"{'Q_R [W]':>10} {'x_D':>10} {'x_B':>10} {'Separation':>12}")
    print("-" * 50)
    for i in range(0, n_sims, 2):  # Print every other result
        Q_R = float(Q_R_values[i])
        x_D = float(x_D_batched[i])
        x_B = float(x_B_batched[i])
        sep = x_D - x_B
        print(f"{Q_R:>10.0f} {x_D:>10.4f} {x_B:>10.4f} {sep:>12.4f}")

    # Plot results
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Product compositions vs Q_R
        axes[0].plot(np.array(Q_R_values), np.array(x_D_batched), "b-o", label="x_D (distillate)")
        axes[0].plot(np.array(Q_R_values), np.array(x_B_batched), "r-s", label="x_B (bottoms)")
        axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Feed composition")
        axes[0].set_xlabel("Reboiler Duty Q_R [W]")
        axes[0].set_ylabel("Composition [mol frac]")
        axes[0].set_title("Product Quality vs Reboiler Duty")
        axes[0].legend()
        axes[0].grid(True)

        # Separation factor
        separation = np.array(x_D_batched) - np.array(x_B_batched)
        axes[1].plot(np.array(Q_R_values), separation, "g-^", linewidth=2)
        axes[1].set_xlabel("Reboiler Duty Q_R [W]")
        axes[1].set_ylabel("Separation (x_D - x_B)")
        axes[1].set_title("Separation Quality vs Reboiler Duty")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("artifacts/plots/vectorized_results.png", dpi=150)
        print("\nPlot saved to: artifacts/plots/vectorized_results.png")
        plt.show()

    except Exception as e:
        print(f"\nNote: Could not create plots ({e})")

    # Demonstrate larger batch
    print("\n--- Scaling Test ---")
    for batch_size in [16, 64, 256]:
        Q_R_large = jnp.linspace(2000.0, 10000.0, batch_size)

        # Warmup
        _ = run_batched_simulation(Q_R_large[:4])

        start = time.perf_counter()
        x_D, x_B = run_batched_simulation(Q_R_large)
        jax.block_until_ready(x_D)
        elapsed = time.perf_counter() - start

        print(f"Batch size {batch_size:>4}: {elapsed:.3f} s ({elapsed/batch_size*1000:.2f} ms/sim)")

    print("\n" + "=" * 60)
    print("Vectorization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
