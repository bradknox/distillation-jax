#!/usr/bin/env python3
"""Basic distillation column simulation example.

This script demonstrates how to:
1. Create a column configuration
2. Initialize the column state
3. Run a simulation
4. Analyze the results

Run with: python examples/basic_simulation.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_distillation.column.config import create_teaching_column_config
from jax_distillation.column.column import (
    column_step,
    create_initial_column_state,
    create_default_action,
    simulate_column,
)


def main():
    print("=" * 60)
    print("JAX Distillation Column Simulation")
    print("=" * 60)

    # Create a teaching column configuration
    # This simulates a 10-tray methanol-water column
    config = create_teaching_column_config(
        n_trays=10,
        feed_tray=5,
        feed_rate=0.1,  # mol/s
        feed_composition=0.5,  # 50% methanol
        pressure=1.0,  # bar
    )

    print(f"\nColumn Configuration:")
    print(f"  Number of trays: {config.geometry.n_trays}")
    print(f"  Feed tray: {config.geometry.feed_tray}")
    print(f"  Feed rate: {float(config.feed.F):.3f} mol/s")
    print(f"  Feed composition: {float(config.feed.z_F):.1%} methanol")
    print(f"  Operating pressure: {float(config.P):.1f} bar")

    # Create initial state
    state = create_initial_column_state(config)
    print(f"\nInitial State:")
    print(f"  Tray holdups: {float(jnp.mean(state.tray_M)):.2f} mol (average)")
    print(f"  Top tray composition: {float(state.tray_x[0]):.3f}")
    print(f"  Bottom tray composition: {float(state.tray_x[-1]):.3f}")

    # Create control action
    # Reboiler duty and reflux ratio are the main control variables
    action = create_default_action(
        Q_R=5000.0,  # 5 kW reboiler duty
        reflux_ratio=3.0,  # Reflux ratio
        B_setpoint=0.03,  # Bottoms flow setpoint
        D_setpoint=0.02,  # Distillate flow setpoint
    )

    print(f"\nControl Action:")
    print(f"  Reboiler duty: {float(action.Q_R):.0f} W")
    print(f"  Reflux ratio: {float(action.reflux_ratio):.1f}")

    # Run simulation for 100 steps
    n_steps = 100
    print(f"\nRunning simulation for {n_steps} steps...")

    final_state, outputs_list = simulate_column(config, action, n_steps=n_steps)

    # Extract results
    x_D_history = [float(o.x_D) for o in outputs_list]
    x_B_history = [float(o.x_B) for o in outputs_list]
    D_history = [float(o.D) for o in outputs_list]
    B_history = [float(o.B) for o in outputs_list]
    Q_C_history = [float(o.Q_C) for o in outputs_list]

    # Print final results
    print(f"\nFinal State (after {n_steps} steps):")
    print(f"  Distillate composition (x_D): {x_D_history[-1]:.3f}")
    print(f"  Bottoms composition (x_B): {x_B_history[-1]:.3f}")
    print(f"  Distillate flow (D): {D_history[-1]:.4f} mol/s")
    print(f"  Bottoms flow (B): {B_history[-1]:.4f} mol/s")
    print(f"  Condenser duty (Q_C): {Q_C_history[-1]:.0f} W")

    # Check mass balance
    F = float(config.feed.F)
    D_final = D_history[-1]
    B_final = B_history[-1]
    mass_balance_error = abs(F - D_final - B_final) / F * 100
    print(f"\nMass Balance Check:")
    print(f"  Feed: {F:.4f} mol/s")
    print(f"  D + B: {D_final + B_final:.4f} mol/s")
    print(f"  Error: {mass_balance_error:.2f}%")

    # Temperature profile
    print(f"\nTemperature Profile:")
    print(f"  Condenser: {float(final_state.condenser.T):.1f} K")
    print(f"  Top tray: {float(final_state.tray_T[0]):.1f} K")
    print(f"  Bottom tray: {float(final_state.tray_T[-1]):.1f} K")
    print(f"  Reboiler: {float(final_state.reboiler.T):.1f} K")

    # Plot results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Composition dynamics
        time = np.arange(n_steps) * float(config.simulation.dt)

        axes[0, 0].plot(time, x_D_history, label="Distillate (x_D)", color="blue")
        axes[0, 0].plot(time, x_B_history, label="Bottoms (x_B)", color="red")
        axes[0, 0].axhline(float(config.feed.z_F), color="gray", linestyle="--", label="Feed")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("Composition [mol frac]")
        axes[0, 0].set_title("Product Compositions")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Flow rates
        axes[0, 1].plot(time, D_history, label="Distillate (D)", color="blue")
        axes[0, 1].plot(time, B_history, label="Bottoms (B)", color="red")
        axes[0, 1].axhline(F, color="gray", linestyle="--", label="Feed")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Flow [mol/s]")
        axes[0, 1].set_title("Product Flows")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Temperature profile
        tray_numbers = np.arange(1, config.geometry.n_trays + 1)
        tray_T = np.array(final_state.tray_T)
        axes[1, 0].plot(tray_numbers, tray_T, "o-", color="orange")
        axes[1, 0].axhline(float(final_state.condenser.T), color="blue", linestyle="--", label="Condenser")
        axes[1, 0].axhline(float(final_state.reboiler.T), color="red", linestyle="--", label="Reboiler")
        axes[1, 0].set_xlabel("Tray Number (1=top)")
        axes[1, 0].set_ylabel("Temperature [K]")
        axes[1, 0].set_title("Temperature Profile")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Composition profile
        tray_x = np.array(final_state.tray_x)
        axes[1, 1].plot(tray_numbers, tray_x, "o-", color="green")
        axes[1, 1].axhline(float(final_state.condenser.x), color="blue", linestyle="--", label="Condenser")
        axes[1, 1].axhline(float(final_state.reboiler.x), color="red", linestyle="--", label="Reboiler")
        axes[1, 1].set_xlabel("Tray Number (1=top)")
        axes[1, 1].set_ylabel("Composition [mol frac]")
        axes[1, 1].set_title("Composition Profile")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("artifacts/plots/simulation_results.png", dpi=150)
        print("\nPlot saved to: artifacts/plots/simulation_results.png")
        plt.show()

    except Exception as e:
        print(f"\nNote: Could not create plots ({e})")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
