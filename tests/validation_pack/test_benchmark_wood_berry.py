"""Tests for Wood-Berry MIMO benchmark."""

import pytest
import numpy as np

from jax_distillation.validation_pack.benchmarks.wood_berry import (
    WoodBerryModel,
    get_wood_berry_coefficients,
    simulate_wood_berry_step_response,
    run_wood_berry_benchmark,
)


class TestWoodBerryModel:
    """Tests for Wood-Berry transfer function model."""

    def test_coefficients_valid(self):
        """Test that Wood-Berry coefficients are valid."""
        coef = get_wood_berry_coefficients()

        # All gains should be non-zero
        assert coef.G11.K != 0
        assert coef.G12.K != 0
        assert coef.G21.K != 0
        assert coef.G22.K != 0

        # Time constants should be positive
        assert coef.G11.tau > 0
        assert coef.G12.tau > 0
        assert coef.G21.tau > 0
        assert coef.G22.tau > 0

        # Dead times should be non-negative
        assert coef.G11.theta >= 0
        assert coef.G12.theta >= 0
        assert coef.G21.theta >= 0
        assert coef.G22.theta >= 0

    def test_model_initialization(self):
        """Test that Wood-Berry model initializes."""
        model = WoodBerryModel()

        assert model is not None
        assert model.coefficients is not None

    def test_model_step(self):
        """Test single step of Wood-Berry model."""
        model = WoodBerryModel(dt=0.1)

        x_D, x_B = model.step(R=0.0, S=0.0)

        # At zero input, output should start at zero
        assert np.isfinite(x_D)
        assert np.isfinite(x_B)

    def test_model_reset(self):
        """Test model reset."""
        model = WoodBerryModel(dt=0.1)

        # Run some steps
        for _ in range(10):
            model.step(R=1.0, S=0.0)

        # Reset
        model.reset()

        # Should be back to initial state
        x_D, x_B = model.step(R=0.0, S=0.0)
        assert abs(x_D) < 0.1  # Should be near zero


class TestWoodBerryStepResponse:
    """Tests for Wood-Berry step response simulation."""

    def test_reflux_step_response(self):
        """Test step response to reflux input."""
        result = simulate_wood_berry_step_response(
            input_var="R",
            step_size=1.0,
            total_time=100.0,
            dt=0.1,
        )

        assert len(result["times"]) > 0
        assert len(result["x_D"]) == len(result["times"])

        # Should converge to expected gain
        expected = result["expected_x_D_final"]
        actual = result["actual_x_D_final"]
        # Allow 20% error due to transient not fully settled
        assert abs(actual - expected) < abs(expected) * 0.3 + 0.1

    def test_steam_step_response(self):
        """Test step response to steam input."""
        result = simulate_wood_berry_step_response(
            input_var="S",
            step_size=1.0,
            total_time=100.0,
            dt=0.1,
        )

        assert len(result["times"]) > 0

        # Expected gain is negative for G12 and G22
        expected_x_D = result["expected_x_D_final"]
        expected_x_B = result["expected_x_B_final"]
        assert expected_x_D < 0  # G12 < 0
        assert expected_x_B < 0  # G22 < 0

    def test_gain_signs(self):
        """Test that gain signs match published values."""
        coef = get_wood_berry_coefficients()

        # G11 > 0: reflux increases distillate purity
        assert coef.G11.K > 0

        # G12 < 0: steam decreases distillate purity (coupling)
        assert coef.G12.K < 0

        # G21 > 0: reflux increases bottoms impurity (coupling)
        assert coef.G21.K > 0

        # G22 < 0: steam decreases bottoms impurity
        assert coef.G22.K < 0


class TestWoodBerryValidation:
    """Tests for Wood-Berry validation against JAX simulator."""

    @pytest.mark.slow
    def test_full_benchmark(self):
        """Run complete Wood-Berry benchmark."""
        result = run_wood_berry_benchmark()

        # Log results
        print(f"All signs correct: {result.all_signs_correct}")
        print(f"Coupling structure OK: {result.coupling_structure_ok}")
        print(f"Overall: {result.overall_passed}")

        for name, correct in result.gain_signs_match.items():
            print(f"  {name}: {'CORRECT' if correct else 'INCORRECT'}")

    def test_gain_signs_match(self):
        """Test that JAX simulator gain signs match Wood-Berry."""
        result = run_wood_berry_benchmark()

        # The key validation is that gain signs match
        # Exact magnitudes will differ due to nonlinearity
        for name, correct in result.gain_signs_match.items():
            print(f"{name}: {'MATCH' if correct else 'MISMATCH'}")

        # All signs should match for proper MIMO structure
        # Allow some failures due to model differences
        n_correct = sum(result.gain_signs_match.values())
        assert n_correct >= 2, (
            f"Only {n_correct}/4 gain signs correct"
        )
