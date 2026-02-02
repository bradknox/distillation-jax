#!/usr/bin/env python3
"""Setpoint tracking demonstration for distillation column.

This script demonstrates the three key phases of column operation:
1. STARTUP: Initial transient from cold start conditions
2. STEADY STATE 1: Column reaches first operating point
3. SETPOINT CHANGE: Step to new operating conditions
4. STEADY STATE 2: Column reaches new operating point

The plot shows a single continuous timeline with all phases clearly marked.

Run with: python examples/setpoint_tracking.py
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from jax_distillation.column.config import create_teaching_column_config
from jax_distillation.column.column import (
    create_initial_column_state,
    create_default_action,
    simulate_column_jit,  # Uses lax.scan - efficient!
)


def main():
    print("=" * 70)
    print("Setpoint Tracking Demonstration")
    print("=" * 70)
    print("\nThis example shows the three phases of column operation:")
    print("  Phase 1: Startup transient")
    print("  Phase 2: Steady state at first operating point")
    print("  Phase 3: Transition to new setpoint")
    print()

    # Create column configuration
    config = create_teaching_column_config(
        n_trays=10,
        feed_tray=5,
        feed_rate=0.1,
        feed_composition=0.5,
        pressure=1.0,
    )
    dt = float(config.simulation.dt)

    print(f"Column: {config.geometry.n_trays} trays, feed at tray {config.geometry.feed_tray}")
    print(f"Timestep: {dt:.1f} s")

    # =========================================================================
    # Define Operating Points
    # =========================================================================
    action_1 = create_default_action(
        Q_R=4000.0,      # 4 kW reboiler duty
        reflux_ratio=2.5,
        B_setpoint=0.03,
        D_setpoint=0.02,
    )

    action_2 = create_default_action(
        Q_R=5500.0,      # 5.5 kW reboiler duty (moderate step increase)
        reflux_ratio=2.5,
        B_setpoint=0.03,
        D_setpoint=0.02,
    )

    print(f"\nOperating Point 1: Q_R = {float(action_1.Q_R):.0f} W, RR = {float(action_1.reflux_ratio):.1f}")
    print(f"Operating Point 2: Q_R = {float(action_2.Q_R):.0f} W, RR = {float(action_2.reflux_ratio):.1f}")

    # =========================================================================
    # JIT compile the simulation function
    # =========================================================================
    print("\nJIT compiling simulation...")
    import time
    t0 = time.time()

    # Create a JIT-compiled function with config baked in and n_steps as static
    from functools import partial

    @partial(jax.jit, static_argnums=(1,))  # n_steps must be static for lax.scan
    def sim_fn(action, n_steps, initial_state):
        return simulate_column_jit(config, action, n_steps, initial_state)

    # Warmup JIT with the step counts we'll use
    initial_state = create_initial_column_state(config)
    _ = sim_fn(action_1, 1000, initial_state)  # Compile for 1000 steps
    _ = sim_fn(action_1, 500, initial_state)   # Compile for 500 steps
    jit_time = time.time() - t0
    print(f"JIT compilation done in {jit_time:.1f}s")

    # =========================================================================
    # Phase 1: Startup (1000 steps = 1000s to reach steady state)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Phase 1: Startup from initial conditions (1000 steps)")
    print("-" * 50)

    n_startup = 1000
    t0 = time.time()
    state_after_startup, outputs_startup = sim_fn(action_1, n_startup, initial_state)
    phase1_time = time.time() - t0

    x_D_startup = np.array(outputs_startup.x_D)
    x_B_startup = np.array(outputs_startup.x_B)

    print(f"  Completed in {phase1_time:.2f}s ({n_startup/phase1_time:.0f} steps/sec)")
    print(f"  Final: x_D = {float(x_D_startup[-1]):.4f}, x_B = {float(x_B_startup[-1]):.4f}")

    # =========================================================================
    # Phase 2: Hold at Steady State 1 (500 steps to show flat steady state)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Phase 2: Hold at first steady state (500 steps)")
    print("-" * 50)

    n_hold = 500
    t0 = time.time()
    state_after_hold, outputs_hold = sim_fn(action_1, n_hold, state_after_startup)
    phase2_time = time.time() - t0

    x_D_hold = np.array(outputs_hold.x_D)
    x_B_hold = np.array(outputs_hold.x_B)

    x_D_before_step = float(x_D_hold[-1])
    x_B_before_step = float(x_B_hold[-1])

    print(f"  Completed in {phase2_time:.2f}s ({n_hold/phase2_time:.0f} steps/sec)")
    print(f"  Steady state 1: x_D = {x_D_before_step:.4f}, x_B = {x_B_before_step:.4f}")

    # =========================================================================
    # Phase 3: Setpoint Change and Transition (1000 steps to reach new steady state)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Phase 3: Step change to new setpoint (1000 steps)")
    print("-" * 50)
    print(f"Applying step: Q_R {float(action_1.Q_R):.0f} W -> {float(action_2.Q_R):.0f} W")

    n_transition = 1000
    t0 = time.time()
    state_final, outputs_transition = sim_fn(action_2, n_transition, state_after_hold)
    phase3_time = time.time() - t0

    x_D_transition = np.array(outputs_transition.x_D)
    x_B_transition = np.array(outputs_transition.x_B)

    x_D_final = float(x_D_transition[-1])
    x_B_final = float(x_B_transition[-1])

    print(f"  Completed in {phase3_time:.2f}s ({n_transition/phase3_time:.0f} steps/sec)")
    print(f"  Final state: x_D = {x_D_final:.4f}, x_B = {x_B_final:.4f}")

    # =========================================================================
    # Combine histories for plotting
    # =========================================================================
    x_D_history = np.concatenate([x_D_startup, x_D_hold, x_D_transition])
    x_B_history = np.concatenate([x_B_startup, x_B_hold, x_B_transition])

    Q_R_history = np.concatenate([
        np.full(n_startup + n_hold, float(action_1.Q_R)),
        np.full(n_transition, float(action_2.Q_R))
    ])

    total_steps = n_startup + n_hold + n_transition
    time_array = np.arange(total_steps) * dt

    startup_end_time = n_startup * dt
    setpoint_change_time = (n_startup + n_hold) * dt

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Setpoint Tracking Summary")
    print("=" * 70)

    total_sim_time = phase1_time + phase2_time + phase3_time
    print(f"\nTotal simulation: {total_steps} steps in {total_sim_time:.2f}s ({total_steps/total_sim_time:.0f} steps/sec)")

    x_D_change = x_D_final - x_D_before_step
    x_B_change = x_B_final - x_B_before_step

    print(f"\nPhase 1 (Startup): {n_startup} steps, x_D={float(x_D_startup[-1]):.4f}, x_B={float(x_B_startup[-1]):.4f}")
    print(f"Phase 2 (Hold): {n_hold} steps, x_D={x_D_before_step:.4f}, x_B={x_B_before_step:.4f}")
    print(f"Phase 3 (Transition): {n_transition} steps")
    print(f"  Q_R: {float(action_1.Q_R):.0f} -> {float(action_2.Q_R):.0f} W")
    print(f"  x_D: {x_D_before_step:.4f} -> {x_D_final:.4f} ({x_D_change:+.4f})")
    print(f"  x_B: {x_B_before_step:.4f} -> {x_B_final:.4f} ({x_B_change:+.4f})")

    print("\nExpected behavior:")
    if x_D_change > 0:
        print("  [OK] Increasing Q_R increased distillate purity (x_D)")
    else:
        print("  [!] Unexpected: x_D decreased or unchanged")

    if x_B_change < 0:
        print("  [OK] Increasing Q_R decreased bottoms impurity (x_B)")
    else:
        print("  [!] Unexpected: x_B increased or unchanged")

    # =========================================================================
    # Plot Results
    # =========================================================================
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        plt.subplots_adjust(hspace=0.25)  # Reduce vertical gap between subplots

        # Plot 1: Distillate composition
        axes[0].plot(time_array, x_D_history, "b-", linewidth=1.5, label="x_D (distillate)")
        axes[0].axvline(startup_end_time, color="green", linestyle="--", alpha=0.7, label="Startup complete")
        axes[0].axvline(setpoint_change_time, color="red", linestyle="--", alpha=0.7, label="Setpoint change")
        axes[0].set_ylabel("Distillate Composition\nx_D [mol frac]", fontsize=10)
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Setpoint Tracking: Startup -> Steady State -> New Setpoint", fontsize=12, fontweight="bold")

        # Add phase labels inside the bottom subplot (Q_R plot has room)
        axes[2].text(startup_end_time / 2, 8, "Phase 1:\nStartup",
                     ha="center", va="center", fontsize=8, color="gray")
        axes[2].text((startup_end_time + setpoint_change_time) / 2, 8, "Phase 2:\nSteady State 1",
                     ha="center", va="center", fontsize=8, color="gray")
        axes[2].text((setpoint_change_time + time_array[-1]) / 2, 8, "Phase 3:\nTransition",
                     ha="center", va="center", fontsize=8, color="gray")

        # Plot 2: Bottoms composition
        axes[1].plot(time_array, x_B_history, "r-", linewidth=1.5, label="x_B (bottoms)")
        axes[1].axvline(startup_end_time, color="green", linestyle="--", alpha=0.7)
        axes[1].axvline(setpoint_change_time, color="red", linestyle="--", alpha=0.7)
        axes[1].set_ylabel("Bottoms Composition\nx_B [mol frac]", fontsize=10)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Reboiler duty (control input)
        axes[2].plot(time_array, Q_R_history / 1000, "k-", linewidth=2, label="Q_R (reboiler duty)")
        axes[2].axvline(startup_end_time, color="green", linestyle="--", alpha=0.7)
        axes[2].axvline(setpoint_change_time, color="red", linestyle="--", alpha=0.7)
        axes[2].set_ylabel("Reboiler Duty\nQ_R [kW]", fontsize=10)
        axes[2].set_xlabel("Time [s]", fontsize=10)
        axes[2].legend(loc="right")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 10])

        plt.tight_layout()
        plt.savefig("artifacts/plots/setpoint_tracking.png", dpi=150)
        print("\nPlot saved to: artifacts/plots/setpoint_tracking.png")
        plt.show()

    except Exception as e:
        print(f"\nNote: Could not create plots ({e})")

    print("\n" + "=" * 70)
    print("Setpoint tracking demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
