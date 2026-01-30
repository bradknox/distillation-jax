"""Smoke tests for fitting pipeline."""

import pytest
import numpy as np


class TestUnitConversion:
    """Tests for unit conversion utilities."""

    def test_temperature_conversion(self):
        """Test temperature unit conversion."""
        from jax_distillation.validation_pack.fitting.units import UnitConverter

        # Celsius to Kelvin
        T_C = 25.0
        T_K = UnitConverter.temperature_to_si(T_C, "C")
        assert abs(T_K - 298.15) < 0.01

        # Fahrenheit to Kelvin
        T_F = 77.0
        T_K = UnitConverter.temperature_to_si(T_F, "F")
        assert abs(T_K - 298.15) < 0.1

    def test_pressure_conversion(self):
        """Test pressure unit conversion."""
        from jax_distillation.validation_pack.fitting.units import UnitConverter

        # atm to bar
        P_atm = 1.0
        P_bar = UnitConverter.pressure_to_si(P_atm, "atm")
        assert abs(P_bar - 1.01325) < 0.001

    def test_time_conversion(self):
        """Test time unit conversion."""
        from jax_distillation.validation_pack.fitting.units import UnitConverter

        # minutes to seconds
        t_min = 5.0
        t_s = UnitConverter.time_to_si(t_min, "min")
        assert t_s == 300.0


class TestReconciliation:
    """Tests for data reconciliation."""

    def test_mass_balance_reconciliation(self):
        """Test mass balance reconciliation."""
        from jax_distillation.validation_pack.fitting.reconciliation import (
            reconcile_mass_balance,
        )

        # Measurements with small errors
        F = 1.0
        D = 0.48  # Slightly off
        B = 0.54  # Slightly off
        z_F = 0.5
        x_D = 0.95
        x_B = 0.05

        result = reconcile_mass_balance(F, D, B, z_F, x_D, x_B)

        assert result.success
        # Reconciled D + B should equal F
        D_rec = result.reconciled_values["D"]
        B_rec = result.reconciled_values["B"]
        F_rec = result.reconciled_values["F"]
        assert abs(D_rec + B_rec - F_rec) < 1e-6

    def test_component_balance(self):
        """Test that component balance is satisfied."""
        from jax_distillation.validation_pack.fitting.reconciliation import (
            reconcile_mass_balance,
        )

        result = reconcile_mass_balance(
            F=1.0, D=0.5, B=0.5,
            z_F=0.5, x_D=0.9, x_B=0.1,
        )

        # Check component balance: F*z_F = D*x_D + B*x_B
        rec = result.reconciled_values
        component_in = rec["F"] * rec["z_F"]
        component_out = rec["D"] * rec["x_D"] + rec["B"] * rec["x_B"]
        assert abs(component_in - component_out) < 1e-4


class TestStateEstimation:
    """Tests for state estimation."""

    def test_ekf_initialization(self):
        """Test EKF initialization."""
        from jax_distillation.validation_pack.fitting.state_estimation import (
            ExtendedKalmanFilter,
        )

        def f(x, u):
            return x  # Identity dynamics

        def h(x):
            return x[:2]  # Measure first 2 states

        Q = np.eye(4) * 0.01
        R = np.eye(2) * 0.01

        ekf = ExtendedKalmanFilter(f, h, Q, R, n_states=4, n_measurements=2)
        state = ekf.initialize(np.zeros(4))

        assert state.x.shape == (4,)
        assert state.P.shape == (4, 4)


class TestParameterEstimation:
    """Tests for parameter estimation."""

    def test_fittable_parameter(self):
        """Test FittableParameter dataclass."""
        from jax_distillation.validation_pack.fitting.parameter_estimation import (
            FittableParameter,
        )

        param = FittableParameter(
            name="efficiency",
            initial_value=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
        )

        assert param.name == "efficiency"
        assert param.initial_value == 1.0
        assert param.lower_bound == 0.5
        assert param.upper_bound == 1.5

    def test_simple_fit(self):
        """Test simple parameter fitting."""
        from jax_distillation.validation_pack.fitting.parameter_estimation import (
            fit_parameters,
            FittableParameter,
        )

        # Simple linear model y = a * x
        def simulator(params, inputs, times):
            a = params["a"]
            return inputs * a

        # Generate data with a = 2
        np.random.seed(42)
        times = np.arange(20)
        inputs = np.random.randn(20, 1)
        measurements = inputs * 2.0 + np.random.randn(20, 1) * 0.1

        parameters = [
            FittableParameter(name="a", initial_value=1.0, lower_bound=0, upper_bound=5),
        ]

        result = fit_parameters(
            parameters=parameters,
            simulator=simulator,
            measurements=measurements,
            inputs=inputs,
            times=times,
            max_iterations=50,
        )

        # Fitted 'a' should be close to 2
        a_fitted = result.fitted_params["a"]
        assert abs(a_fitted - 2.0) < 0.5, f"Fitted a={a_fitted}, expected ~2.0"


class TestIdentifiability:
    """Tests for identifiability analysis."""

    def test_sensitivity_computation(self):
        """Test sensitivity computation."""
        from jax_distillation.validation_pack.fitting.identifiability import (
            compute_sensitivity,
        )

        # Simple model: y = a * x + b
        def simulator(params, inputs, times):
            a = params["a"]
            b = params["b"]
            return inputs * a + b

        times = np.arange(10)
        inputs = np.random.randn(10, 1)
        params = {"a": 1.0, "b": 0.5}

        S, S_dict = compute_sensitivity(simulator, params, inputs, times)

        assert S.shape[0] == 10  # n_times * n_outputs
        assert S.shape[1] == 2   # n_params
        assert "a" in S_dict
        assert "b" in S_dict


class TestReporting:
    """Tests for fit reporting."""

    def test_report_generation(self):
        """Test fit report generation."""
        from jax_distillation.validation_pack.fitting.reporting import (
            generate_fit_report,
            FitReport,
        )
        from jax_distillation.validation_pack.fitting.parameter_estimation import (
            ParameterEstimationResult,
        )

        # Create mock parameter estimation result
        result = ParameterEstimationResult(
            initial_params={"a": 1.0},
            fitted_params={"a": 2.0},
            initial_loss=1.0,
            final_loss=0.1,
            loss_reduction=0.9,
            n_iterations=10,
            converged=True,
            fit_metrics={"rmse": 0.1, "r_squared": 0.95},
        )

        report = generate_fit_report(result, dataset_name="Test")

        assert isinstance(report, FitReport)
        assert report.dataset_info["name"] == "Test"
        assert report.parameter_results.converged
