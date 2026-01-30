"""Tests for reboiler and condenser modules."""

import jax
import jax.numpy as jnp
import pytest

from jax_distillation.column.reboiler import (
    ReboilerState,
    ReboilerInputs,
    reboiler_vapor_generation,
    reboiler_level_control,
    reboiler_material_balance,
    reboiler_component_balance,
    reboiler_step,
    create_initial_reboiler_state,
    create_reboiler_inputs,
)
from jax_distillation.column.condenser import (
    CondenserState,
    CondenserInputs,
    condenser_duty,
    reflux_drum_level_control,
    compute_reflux_and_distillate,
    condenser_material_balance,
    condenser_component_balance,
    condenser_step,
    create_initial_condenser_state,
    create_condenser_inputs,
)
from jax_distillation.core.thermodynamics import create_methanol_water_thermo


class TestReboilerVaporGeneration:
    """Tests for reboiler vapor generation."""

    def test_zero_heat_zero_vapor(self):
        """Zero heat input should produce zero vapor."""
        thermo = create_methanol_water_thermo()
        Q_R = jnp.array(0.0)
        x = jnp.array(0.3)
        T = jnp.array(365.0)

        V = reboiler_vapor_generation(Q_R, x, T, thermo)
        assert jnp.allclose(V, 0.0)

    def test_positive_heat_positive_vapor(self):
        """Positive heat input should produce positive vapor."""
        thermo = create_methanol_water_thermo()
        Q_R = jnp.array(10000.0)  # 10 kW
        x = jnp.array(0.3)
        T = jnp.array(365.0)

        V = reboiler_vapor_generation(Q_R, x, T, thermo)
        assert V > 0

    def test_vapor_increases_with_heat(self):
        """More heat should produce more vapor."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.3)
        T = jnp.array(365.0)

        Q_values = jnp.array([5000.0, 10000.0, 15000.0])
        V_values = jax.vmap(lambda Q: reboiler_vapor_generation(Q, x, T, thermo))(Q_values)

        assert jnp.all(jnp.diff(V_values) > 0)


class TestReboilerLevelControl:
    """Tests for reboiler level control."""

    def test_at_setpoint(self):
        """At setpoint, flow equals nominal setpoint."""
        M = jnp.array(10.0)
        M_setpoint = jnp.array(10.0)
        B_setpoint = jnp.array(0.05)

        B = reboiler_level_control(M, M_setpoint, B_setpoint)
        assert jnp.allclose(B, B_setpoint)

    def test_above_setpoint(self):
        """Above setpoint, flow increases."""
        M = jnp.array(12.0)
        M_setpoint = jnp.array(10.0)
        B_setpoint = jnp.array(0.05)

        B = reboiler_level_control(M, M_setpoint, B_setpoint)
        assert B > B_setpoint

    def test_below_setpoint(self):
        """Below setpoint, flow decreases (but stays non-negative)."""
        M = jnp.array(8.0)
        M_setpoint = jnp.array(10.0)
        B_setpoint = jnp.array(0.05)

        B = reboiler_level_control(M, M_setpoint, B_setpoint)
        assert B >= 0
        assert B < B_setpoint


class TestReboilerMaterialBalance:
    """Tests for reboiler material balance."""

    def test_steady_state(self):
        """At steady state, dM/dt = 0."""
        L_in = jnp.array(0.1)
        V_out = jnp.array(0.05)
        B = jnp.array(0.05)

        dM_dt = reboiler_material_balance(L_in, V_out, B)
        assert jnp.allclose(dM_dt, 0.0)

    def test_accumulation(self):
        """More in than out causes accumulation."""
        L_in = jnp.array(0.15)
        V_out = jnp.array(0.05)
        B = jnp.array(0.05)

        dM_dt = reboiler_material_balance(L_in, V_out, B)
        assert dM_dt > 0


class TestReboilerStep:
    """Tests for complete reboiler step."""

    def test_reboiler_step_runs(self):
        """Reboiler step should execute without error."""
        thermo = create_methanol_water_thermo()

        state = create_initial_reboiler_state(M=10.0, x=0.1, T=370.0)
        inputs = create_reboiler_inputs(
            L_in=0.1, x_in=0.15, T_in=365.0, Q_R=10000.0, B_setpoint=0.03
        )
        P = jnp.array(1.0)
        M_setpoint = jnp.array(10.0)
        dt = jnp.array(1.0)

        new_state, outputs = reboiler_step(state, inputs, P, M_setpoint, thermo, dt)

        assert new_state.M > 0
        assert 0.0 <= new_state.x <= 1.0
        assert new_state.T > 0
        assert outputs.V_out >= 0
        assert outputs.B >= 0

    def test_reboiler_stable_over_time(self):
        """Reboiler should remain stable over multiple steps."""
        thermo = create_methanol_water_thermo()

        state = create_initial_reboiler_state(M=10.0, x=0.1, T=370.0)
        inputs = create_reboiler_inputs(
            L_in=0.1, x_in=0.15, T_in=365.0, Q_R=5000.0, B_setpoint=0.03
        )
        P = jnp.array(1.0)
        M_setpoint = jnp.array(10.0)
        dt = jnp.array(1.0)

        for _ in range(50):
            state, outputs = reboiler_step(state, inputs, P, M_setpoint, thermo, dt)
            assert jnp.isfinite(state.M)
            assert jnp.isfinite(state.x)
            assert jnp.isfinite(state.T)


class TestCondenserDuty:
    """Tests for condenser duty calculation."""

    def test_duty_positive_for_condensation(self):
        """Condenser duty should be positive (heat removed)."""
        thermo = create_methanol_water_thermo()
        V_in = jnp.array(0.1)
        y_in = jnp.array(0.8)
        T_V_in = jnp.array(345.0)
        x_out = jnp.array(0.8)
        T_out = jnp.array(340.0)

        Q_C = condenser_duty(V_in, y_in, T_V_in, x_out, T_out, thermo)
        assert Q_C > 0

    def test_duty_increases_with_flow(self):
        """Condenser duty should increase with vapor flow."""
        thermo = create_methanol_water_thermo()
        y_in = jnp.array(0.8)
        T_V_in = jnp.array(345.0)
        x_out = jnp.array(0.8)
        T_out = jnp.array(340.0)

        V_values = jnp.array([0.05, 0.1, 0.15])
        Q_values = jax.vmap(
            lambda V: condenser_duty(V, y_in, T_V_in, x_out, T_out, thermo)
        )(V_values)

        assert jnp.all(jnp.diff(Q_values) > 0)


class TestRefluxDrumLevelControl:
    """Tests for reflux drum level control."""

    def test_at_setpoint(self):
        """At setpoint, flow equals nominal setpoint."""
        M = jnp.array(5.0)
        M_setpoint = jnp.array(5.0)
        D_setpoint = jnp.array(0.03)

        D = reflux_drum_level_control(M, M_setpoint, D_setpoint)
        assert jnp.allclose(D, D_setpoint)

    def test_above_setpoint(self):
        """Above setpoint, distillate flow increases."""
        M = jnp.array(6.0)
        M_setpoint = jnp.array(5.0)
        D_setpoint = jnp.array(0.03)

        D = reflux_drum_level_control(M, M_setpoint, D_setpoint)
        assert D > D_setpoint


class TestRefluxAndDistillate:
    """Tests for reflux and distillate calculation."""

    def test_total_flow_conservation(self):
        """R + D should approximately equal V_in at steady state."""
        V_in = jnp.array(0.1)
        reflux_ratio = jnp.array(3.0)
        M = jnp.array(5.0)
        M_setpoint = jnp.array(5.0)
        D_setpoint = jnp.array(0.025)

        R, D = compute_reflux_and_distillate(V_in, reflux_ratio, M, M_setpoint, D_setpoint)

        assert jnp.allclose(R + D, V_in, rtol=0.01)

    def test_nonnegative_flows(self):
        """Both reflux and distillate should be non-negative."""
        V_in = jnp.array(0.1)
        reflux_ratio = jnp.array(3.0)
        M = jnp.array(5.0)
        M_setpoint = jnp.array(5.0)
        D_setpoint = jnp.array(0.025)

        R, D = compute_reflux_and_distillate(V_in, reflux_ratio, M, M_setpoint, D_setpoint)

        assert R >= 0
        assert D >= 0


class TestCondenserMaterialBalance:
    """Tests for condenser material balance."""

    def test_steady_state(self):
        """At steady state, dM/dt = 0."""
        V_in = jnp.array(0.1)
        R = jnp.array(0.07)
        D = jnp.array(0.03)

        dM_dt = condenser_material_balance(V_in, R, D)
        assert jnp.allclose(dM_dt, 0.0)


class TestCondenserStep:
    """Tests for complete condenser step."""

    def test_condenser_step_runs(self):
        """Condenser step should execute without error."""
        thermo = create_methanol_water_thermo()

        state = create_initial_condenser_state(M=5.0, x=0.9, T=340.0)
        inputs = create_condenser_inputs(
            V_in=0.1, y_in=0.85, T_V_in=345.0, reflux_ratio=3.0, D_setpoint=0.025
        )
        P = jnp.array(1.0)
        M_setpoint = jnp.array(5.0)
        dt = jnp.array(1.0)

        new_state, outputs = condenser_step(state, inputs, P, M_setpoint, thermo, dt)

        assert new_state.M > 0
        assert 0.0 <= new_state.x <= 1.0
        assert new_state.T > 0
        assert outputs.R >= 0
        assert outputs.D >= 0
        assert outputs.Q_C > 0

    def test_condenser_stable_over_time(self):
        """Condenser should remain stable over multiple steps."""
        thermo = create_methanol_water_thermo()

        state = create_initial_condenser_state(M=5.0, x=0.9, T=340.0)
        inputs = create_condenser_inputs(
            V_in=0.1, y_in=0.85, T_V_in=345.0, reflux_ratio=3.0, D_setpoint=0.025
        )
        P = jnp.array(1.0)
        M_setpoint = jnp.array(5.0)
        dt = jnp.array(1.0)

        for _ in range(50):
            state, outputs = condenser_step(state, inputs, P, M_setpoint, thermo, dt)
            assert jnp.isfinite(state.M)
            assert jnp.isfinite(state.x)
            assert jnp.isfinite(state.T)


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    def test_reboiler_step_jit(self):
        """Reboiler step should be JIT-compilable."""
        thermo = create_methanol_water_thermo()
        reboiler_step_jit = jax.jit(reboiler_step)

        state = create_initial_reboiler_state(M=10.0, x=0.1, T=370.0)
        inputs = create_reboiler_inputs(
            L_in=0.1, x_in=0.15, T_in=365.0, Q_R=10000.0, B_setpoint=0.03
        )
        P = jnp.array(1.0)
        M_setpoint = jnp.array(10.0)
        dt = jnp.array(1.0)

        new_state_jit, _ = reboiler_step_jit(state, inputs, P, M_setpoint, thermo, dt)
        new_state, _ = reboiler_step(state, inputs, P, M_setpoint, thermo, dt)

        assert jnp.allclose(new_state_jit.M, new_state.M)
        assert jnp.allclose(new_state_jit.x, new_state.x)

    def test_condenser_step_jit(self):
        """Condenser step should be JIT-compilable."""
        thermo = create_methanol_water_thermo()
        condenser_step_jit = jax.jit(condenser_step)

        state = create_initial_condenser_state(M=5.0, x=0.9, T=340.0)
        inputs = create_condenser_inputs(
            V_in=0.1, y_in=0.85, T_V_in=345.0, reflux_ratio=3.0, D_setpoint=0.025
        )
        P = jnp.array(1.0)
        M_setpoint = jnp.array(5.0)
        dt = jnp.array(1.0)

        new_state_jit, _ = condenser_step_jit(state, inputs, P, M_setpoint, thermo, dt)
        new_state, _ = condenser_step(state, inputs, P, M_setpoint, thermo, dt)

        assert jnp.allclose(new_state_jit.M, new_state.M)
        assert jnp.allclose(new_state_jit.x, new_state.x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
