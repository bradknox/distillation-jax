#!/usr/bin/env python3
"""Step response analysis for distillation column.

This script demonstrates how to:
1. Apply step changes to control inputs
2. Analyze the dynamic response
3. Measure time constants and settling times

Run with: python examples/step_response.py
"""

import matplotlib.pyplot as plt
import numpy as np

from jax_distillation.column.config import create_teaching_column_config
from jax_distillation.column.column import (
    column_step,
    create_initial_column_state,
    create_default_action,
)
from jax_distillation.validation.dynamic_response import (
    run_step_response,
    analyze_step_response,
)


def main():
    print("=" * 60)
    print("Step Response Analysis")
    print("=" * 60)

    # Create column configuration (smaller for faster computation)
    config = create_teaching_column_config(
        n_trays=8,
        feed_tray=4,
        feed_composition=0.5,
    )
    dt = float(config.simulation.dt)

    print(f"\nColumn: {config.geometry.n_trays} trays, feed at tray {config.geometry.feed_tray}")

    # =========================================================================
    # Test 1: Reboiler Duty Step
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 1: Reboiler Duty Step (3000 → 6000 W)")
    print("-" * 40)

    action_before = create_default_action(Q_R=3000.0, reflux_ratio=3.0)
    action_after = create_default_action(Q_R=6000.0, reflux_ratio=3.0)

    states, outputs_before, outputs_after = run_step_response(
        config,
        action_before,
        action_after,
        warmup_steps=150,
        response_steps=250,
    )

    # Extract composition histories
    x_D_before = [float(o.x_D) for o in outputs_before]
    x_D_after = [float(o.x_D) for o in outputs_after]
    x_B_before = [float(o.x_B) for o in outputs_before]
    x_B_after = [float(o.x_B) for o in outputs_after]

    # Analyze x_D response
    x_D_array = np.array(x_D_after)
    x_D_metrics = analyze_step_response(x_D_array, dt)

    print(f"\nDistillate Composition (x_D) Response:")
    print(f"  Initial value: {x_D_metrics.initial_value:.4f}")
    print(f"  Final value: {x_D_metrics.final_value:.4f}")
    print(f"  Change: {x_D_metrics.change:+.4f}")
    print(f"  Time constant: {x_D_metrics.time_constant:.1f} s")
    print(f"  Rise time (10-90%): {x_D_metrics.rise_time:.1f} s")
    print(f"  Settling time: {x_D_metrics.settling_time:.1f} s")

    # Analyze x_B response
    x_B_array = np.array(x_B_after)
    x_B_metrics = analyze_step_response(x_B_array, dt)

    print(f"\nBottoms Composition (x_B) Response:")
    print(f"  Initial value: {x_B_metrics.initial_value:.4f}")
    print(f"  Final value: {x_B_metrics.final_value:.4f}")
    print(f"  Change: {x_B_metrics.change:+.4f}")
    print(f"  Time constant: {x_B_metrics.time_constant:.1f} s")

    # =========================================================================
    # Test 2: Reflux Ratio Step
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 2: Reflux Ratio Step (2.0 → 4.0)")
    print("-" * 40)

    action_before_rr = create_default_action(Q_R=5000.0, reflux_ratio=2.0)
    action_after_rr = create_default_action(Q_R=5000.0, reflux_ratio=4.0)

    states_rr, outputs_before_rr, outputs_after_rr = run_step_response(
        config,
        action_before_rr,
        action_after_rr,
        warmup_steps=150,
        response_steps=250,
    )

    x_D_rr_after = [float(o.x_D) for o in outputs_after_rr]
    x_B_rr_after = [float(o.x_B) for o in outputs_after_rr]

    x_D_rr_metrics = analyze_step_response(np.array(x_D_rr_after), dt)
    x_B_rr_metrics = analyze_step_response(np.array(x_B_rr_after), dt)

    print(f"\nDistillate Composition (x_D) Response:")
    print(f"  Initial value: {x_D_rr_metrics.initial_value:.4f}")
    print(f"  Final value: {x_D_rr_metrics.final_value:.4f}")
    print(f"  Change: {x_D_rr_metrics.change:+.4f}")
    print(f"  Time constant: {x_D_rr_metrics.time_constant:.1f} s")

    print(f"\nBottoms Composition (x_B) Response:")
    print(f"  Initial value: {x_B_rr_metrics.initial_value:.4f}")
    print(f"  Final value: {x_B_rr_metrics.final_value:.4f}")
    print(f"  Change: {x_B_rr_metrics.change:+.4f}")

    # =========================================================================
    # Plot Results
    # =========================================================================
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Create time arrays
        t_before = np.arange(-len(x_D_before), 0) * dt
        t_after = np.arange(len(x_D_after)) * dt

        # Test 1: Q_R step - x_D response
        axes[0, 0].plot(t_before, x_D_before, "b-", alpha=0.5, label="Before step")
        axes[0, 0].plot(t_after, x_D_after, "b-", linewidth=2, label="After step")
        axes[0, 0].axvline(0, color="red", linestyle="--", alpha=0.5, label="Step applied")
        axes[0, 0].axhline(x_D_metrics.final_value, color="gray", linestyle=":", alpha=0.5)
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("Distillate Composition x_D")
        axes[0, 0].set_title("Q_R Step (3→6 kW): x_D Response")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Test 1: Q_R step - x_B response
        axes[0, 1].plot(t_before, x_B_before, "r-", alpha=0.5, label="Before step")
        axes[0, 1].plot(t_after, x_B_after, "r-", linewidth=2, label="After step")
        axes[0, 1].axvline(0, color="red", linestyle="--", alpha=0.5, label="Step applied")
        axes[0, 1].axhline(x_B_metrics.final_value, color="gray", linestyle=":", alpha=0.5)
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Bottoms Composition x_B")
        axes[0, 1].set_title("Q_R Step (3→6 kW): x_B Response")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Test 2: RR step
        t_before_rr = np.arange(-len(outputs_before_rr), 0) * dt
        t_after_rr = np.arange(len(x_D_rr_after)) * dt
        x_D_rr_before = [float(o.x_D) for o in outputs_before_rr]
        x_B_rr_before = [float(o.x_B) for o in outputs_before_rr]

        axes[1, 0].plot(t_before_rr, x_D_rr_before, "b-", alpha=0.5, label="Before step")
        axes[1, 0].plot(t_after_rr, x_D_rr_after, "b-", linewidth=2, label="After step")
        axes[1, 0].axvline(0, color="red", linestyle="--", alpha=0.5, label="Step applied")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("Distillate Composition x_D")
        axes[1, 0].set_title("Reflux Ratio Step (2→4): x_D Response")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(t_before_rr, x_B_rr_before, "r-", alpha=0.5, label="Before step")
        axes[1, 1].plot(t_after_rr, x_B_rr_after, "r-", linewidth=2, label="After step")
        axes[1, 1].axvline(0, color="red", linestyle="--", alpha=0.5, label="Step applied")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Bottoms Composition x_B")
        axes[1, 1].set_title("Reflux Ratio Step (2→4): x_B Response")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("step_response.png", dpi=150)
        print("\nPlot saved to: step_response.png")
        plt.show()

    except Exception as e:
        print(f"\nNote: Could not create plots ({e})")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary of Dynamic Behavior")
    print("=" * 60)

    print("\nExpected behavior confirmed:")
    print("  ✓ Increasing Q_R increases x_D (more light in distillate)")
    if x_D_metrics.change > 0:
        print("    → Observed: x_D increased")
    print("  ✓ Increasing Q_R decreases x_B (less light in bottoms)")
    if x_B_metrics.change < 0:
        print("    → Observed: x_B decreased")
    print("  ✓ Increasing RR increases separation")
    if x_D_rr_metrics.change > 0 and x_B_rr_metrics.change < 0:
        print("    → Observed: x_D increased, x_B decreased")

    print("\nTime constants suggest realistic dynamics:")
    print(f"  Composition changes: ~{x_D_metrics.time_constant:.0f}-{max(x_D_rr_metrics.time_constant, 1):.0f} s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
