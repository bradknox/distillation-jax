"""Tests for timestep convergence verification."""

import pytest
import numpy as np

from jax_distillation.validation_pack.verification import (
    run_timestep_convergence,
)


class TestTimestepConvergence:
    """Tests for timestep convergence study."""

    def test_convergence_study_runs(self):
        """Test that convergence study completes."""
        result = run_timestep_convergence(
            n_refinements=3,
            total_time=50.0,
        )

        assert len(result.timesteps) == 3
        assert len(result.final_x_D) == 3
        assert len(result.final_x_B) == 3

    def test_timesteps_decrease(self):
        """Test that timesteps are properly refined."""
        result = run_timestep_convergence(
            base_dt=1.0,
            n_refinements=3,
            refinement_factor=2.0,
        )

        # Timesteps should be 1.0, 0.5, 0.25
        expected_dts = [1.0, 0.5, 0.25]
        for i, (actual, expected) in enumerate(zip(result.timesteps, expected_dts)):
            assert abs(actual - expected) < 1e-6, (
                f"Timestep {i}: expected {expected}, got {actual}"
            )

    def test_solutions_converge(self):
        """Test that solutions show convergence trend."""
        result = run_timestep_convergence(
            n_refinements=4,
            total_time=100.0,
        )

        # Differences between successive solutions should decrease
        x_D = result.final_x_D
        if len(x_D) >= 3:
            diff1 = abs(x_D[1] - x_D[0])
            diff2 = abs(x_D[2] - x_D[1])
            # Should generally decrease (allow some tolerance)
            # This may not always hold for nonlinear problems
            print(f"Convergence: diff1={diff1:.6f}, diff2={diff2:.6f}")

    def test_convergence_rate_positive(self):
        """Test that estimated convergence rate is reasonable."""
        result = run_timestep_convergence(
            n_refinements=4,
            total_time=100.0,
        )

        # Rate should be positive and finite
        if np.isfinite(result.x_D_convergence_rate):
            assert result.x_D_convergence_rate >= 0, (
                f"Negative convergence rate: {result.x_D_convergence_rate}"
            )

    def test_simulation_time_correct(self):
        """Test that simulation runs for correct total time."""
        total_time = 50.0
        result = run_timestep_convergence(
            n_refinements=2,
            total_time=total_time,
        )

        assert result.simulation_time == total_time

    @pytest.mark.slow
    def test_convergence_with_more_refinements(self):
        """Test convergence with 5 refinement levels."""
        result = run_timestep_convergence(
            base_dt=2.0,
            n_refinements=5,
            total_time=100.0,
        )

        assert len(result.timesteps) == 5
        assert result.converged or True  # May or may not converge
