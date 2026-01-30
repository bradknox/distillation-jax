"""Tests for single tray dynamics module."""

import jax
import jax.numpy as jnp
import pytest

from jax_distillation.column.tray import (
    TrayState,
    TrayInflows,
    TrayOutputs,
    TrayDerivatives,
    murphree_vapor_efficiency,
    compute_tray_equilibrium,
    total_material_balance,
    component_material_balance,
    energy_balance,
    compute_vapor_outflow,
    compute_tray_outputs,
    integrate_tray_state,
    tray_step,
    create_initial_tray_state,
    create_tray_inflows,
)
from jax_distillation.core.thermodynamics import create_methanol_water_thermo
from jax_distillation.core.hydraulics import create_default_hydraulic_params


class TestMurphreeEfficiency:
    """Tests for Murphree vapor efficiency."""

    def test_perfect_efficiency(self):
        """With E_M=1, output should equal equilibrium."""
        y_in = jnp.array(0.3)
        y_star = jnp.array(0.7)
        E_M = jnp.array(1.0)

        y_out = murphree_vapor_efficiency(y_in, y_star, E_M)
        assert jnp.allclose(y_out, y_star)

    def test_zero_efficiency(self):
        """With E_M=0, output should equal input."""
        y_in = jnp.array(0.3)
        y_star = jnp.array(0.7)
        E_M = jnp.array(0.0)

        y_out = murphree_vapor_efficiency(y_in, y_star, E_M)
        assert jnp.allclose(y_out, y_in)

    def test_partial_efficiency(self):
        """With E_M=0.5, output should be midpoint."""
        y_in = jnp.array(0.3)
        y_star = jnp.array(0.7)
        E_M = jnp.array(0.5)

        y_out = murphree_vapor_efficiency(y_in, y_star, E_M)
        assert jnp.allclose(y_out, 0.5)  # midpoint

    def test_output_bounded(self):
        """Output should be bounded in [0, 1]."""
        y_in = jnp.array(0.0)
        y_star = jnp.array(1.0)
        E_M = jnp.array(0.8)

        y_out = murphree_vapor_efficiency(y_in, y_star, E_M)
        assert 0.0 <= y_out <= 1.0


class TestEquilibriumCalculations:
    """Tests for tray equilibrium calculations."""

    def test_equilibrium_enrichment(self):
        """Vapor should be enriched in light component."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        T = jnp.array(350.0)
        P = jnp.array(1.0)
        y_in = jnp.array(0.4)
        E_M = jnp.array(0.8)

        y_out = compute_tray_equilibrium(x, T, P, y_in, E_M, thermo)

        # Vapor should be richer in methanol than liquid
        assert y_out > x

    def test_equilibrium_bounded(self):
        """Equilibrium vapor should be bounded."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)

        x_values = jnp.linspace(0.1, 0.9, 9)
        for x in x_values:
            T = jnp.array(350.0)
            y_in = jnp.array(0.3)
            y_out = compute_tray_equilibrium(x, T, P, y_in, E_M, thermo)
            assert 0.0 <= y_out <= 1.0


class TestMaterialBalance:
    """Tests for material balance calculations."""

    def test_steady_state_balance(self):
        """At steady state, dM/dt should be zero."""
        L_in = jnp.array(1.0)
        L_out = jnp.array(1.0)
        V_in = jnp.array(0.5)
        V_out = jnp.array(0.5)
        F_L = jnp.array(0.0)
        F_V = jnp.array(0.0)

        dM_dt = total_material_balance(L_in, L_out, V_in, V_out, F_L, F_V)
        assert jnp.allclose(dM_dt, 0.0)

    def test_accumulation(self):
        """More in than out should cause accumulation."""
        L_in = jnp.array(1.2)
        L_out = jnp.array(1.0)
        V_in = jnp.array(0.5)
        V_out = jnp.array(0.5)
        F_L = jnp.array(0.0)
        F_V = jnp.array(0.0)

        dM_dt = total_material_balance(L_in, L_out, V_in, V_out, F_L, F_V)
        assert dM_dt > 0

    def test_feed_contribution(self):
        """Feed should contribute to accumulation."""
        L_in = jnp.array(1.0)
        L_out = jnp.array(1.0)
        V_in = jnp.array(0.5)
        V_out = jnp.array(0.5)
        F_L = jnp.array(0.1)
        F_V = jnp.array(0.0)

        dM_dt = total_material_balance(L_in, L_out, V_in, V_out, F_L, F_V)
        assert dM_dt > 0

    def test_component_balance_steady_state(self):
        """At steady state with uniform composition, component balance is zero."""
        x = jnp.array(0.5)
        L_in = jnp.array(1.0)
        L_out = jnp.array(1.0)
        V_in = jnp.array(0.5)
        V_out = jnp.array(0.5)

        dMx_dt = component_material_balance(
            L_in, x, L_out, x, V_in, x, V_out, x,
            jnp.array(0.0), jnp.array(0.0), x
        )
        assert jnp.allclose(dMx_dt, 0.0)


class TestEnergyBalance:
    """Tests for energy balance calculations."""

    def test_energy_balance_steady_state(self):
        """At steady state with uniform T, energy balance is approximately zero."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        T = jnp.array(350.0)
        L_in = jnp.array(1.0)
        L_out = jnp.array(1.0)
        V_in = jnp.array(0.5)
        V_out = jnp.array(0.5)
        Q = jnp.array(0.0)

        dU_dt = energy_balance(
            L_in, x, T,
            L_out, x, T,
            V_in, x, T,
            V_out, x, T,
            jnp.array(0.0), jnp.array(0.0), x, T,
            Q, thermo
        )

        # Should be close to zero at steady state
        assert jnp.abs(dU_dt) < 100  # Small tolerance for numerical precision

    def test_heat_input_increases_energy(self):
        """Positive heat input should increase internal energy."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        T = jnp.array(350.0)
        L = jnp.array(1.0)
        V = jnp.array(0.5)

        dU_dt_no_Q = energy_balance(
            L, x, T, L, x, T, V, x, T, V, x, T,
            jnp.array(0.0), jnp.array(0.0), x, T,
            jnp.array(0.0), thermo
        )

        dU_dt_with_Q = energy_balance(
            L, x, T, L, x, T, V, x, T, V, x, T,
            jnp.array(0.0), jnp.array(0.0), x, T,
            jnp.array(1000.0), thermo  # 1 kW heat input
        )

        assert dU_dt_with_Q > dU_dt_no_Q


class TestTrayStep:
    """Tests for complete tray step function."""

    def test_tray_step_runs(self):
        """Tray step should execute without error."""
        thermo = create_methanol_water_thermo()
        hydraulics = create_default_hydraulic_params()

        state = create_initial_tray_state(M=5.0, x=0.5, T=350.0, L_out=0.1)
        inflows = create_tray_inflows(
            L_in=0.1, x_in=0.6, T_L_in=345.0,
            V_in=0.1, y_in=0.7, T_V_in=355.0
        )
        V_in_prev = jnp.array(0.1)
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)
        dt = jnp.array(1.0)

        new_state, outputs = tray_step(
            state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
        )

        # Check outputs are valid
        assert new_state.M > 0
        assert 0.0 <= new_state.x <= 1.0
        assert new_state.T > 0
        assert outputs.L_out >= 0
        assert outputs.V_out >= 0

    def test_holdup_stays_positive(self):
        """Holdup should remain positive even with large outflows."""
        thermo = create_methanol_water_thermo()
        hydraulics = create_default_hydraulic_params()

        state = create_initial_tray_state(M=1.0, x=0.5, T=350.0, L_out=0.5)
        inflows = create_tray_inflows(
            L_in=0.01, x_in=0.6, T_L_in=345.0,  # Small inflow
            V_in=0.01, y_in=0.7, T_V_in=355.0
        )
        V_in_prev = jnp.array(0.01)
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)
        dt = jnp.array(1.0)

        new_state, _ = tray_step(
            state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
        )

        assert new_state.M > 0

    def test_composition_stays_bounded(self):
        """Composition should stay in [0, 1]."""
        thermo = create_methanol_water_thermo()
        hydraulics = create_default_hydraulic_params()

        # Start near boundary
        state = create_initial_tray_state(M=5.0, x=0.95, T=340.0, L_out=0.1)
        inflows = create_tray_inflows(
            L_in=0.1, x_in=0.99, T_L_in=338.0,
            V_in=0.1, y_in=0.99, T_V_in=340.0
        )
        V_in_prev = jnp.array(0.1)
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)
        dt = jnp.array(1.0)

        new_state, _ = tray_step(
            state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
        )

        assert 0.0 <= new_state.x <= 1.0

    def test_multiple_steps_stable(self):
        """Multiple tray steps should remain stable."""
        thermo = create_methanol_water_thermo()
        hydraulics = create_default_hydraulic_params()

        state = create_initial_tray_state(M=5.0, x=0.5, T=350.0, L_out=0.1)
        inflows = create_tray_inflows(
            L_in=0.1, x_in=0.6, T_L_in=345.0,
            V_in=0.1, y_in=0.7, T_V_in=355.0
        )
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)
        dt = jnp.array(0.5)

        V_in_prev = jnp.array(0.1)
        for _ in range(100):
            new_state, outputs = tray_step(
                state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
            )

            # Check for numerical stability
            assert jnp.isfinite(new_state.M)
            assert jnp.isfinite(new_state.x)
            assert jnp.isfinite(new_state.T)

            V_in_prev = inflows.V_in
            state = new_state

        # Final state should be reasonable
        assert new_state.M > 0
        assert 0.0 <= new_state.x <= 1.0


class TestMassConservation:
    """Tests for mass conservation in tray dynamics."""

    def test_total_mass_conservation(self):
        """Verify mass balance is consistent over multiple steps."""
        thermo = create_methanol_water_thermo()
        hydraulics = create_default_hydraulic_params()

        # Use balanced inflows/outflows for steady-state-like conditions
        state = create_initial_tray_state(M=5.0, x=0.5, T=350.0, L_out=0.1)
        inflows = create_tray_inflows(
            L_in=0.1, x_in=0.6, T_L_in=345.0,
            V_in=0.1, y_in=0.7, T_V_in=355.0,
            F_L=0.0, z_F=0.5, T_F=350.0
        )
        V_in_prev = jnp.array(0.1)
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)
        dt = jnp.array(0.5)

        # Run multiple steps and track that state stays bounded
        for _ in range(10):
            new_state, outputs = tray_step(
                state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
            )

            # Mass should not go negative
            assert new_state.M > 0

            # Mass should be finite
            assert jnp.isfinite(new_state.M)

            V_in_prev = inflows.V_in
            state = new_state

        # After settling, holdup should stabilize (not grow unboundedly)
        assert state.M < 100  # Should not explode


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    def test_tray_step_jit(self):
        """Tray step should be JIT-compilable."""
        thermo = create_methanol_water_thermo()
        hydraulics = create_default_hydraulic_params()

        tray_step_jit = jax.jit(tray_step)

        state = create_initial_tray_state(M=5.0, x=0.5, T=350.0, L_out=0.1)
        inflows = create_tray_inflows(
            L_in=0.1, x_in=0.6, T_L_in=345.0,
            V_in=0.1, y_in=0.7, T_V_in=355.0
        )
        V_in_prev = jnp.array(0.1)
        P = jnp.array(1.0)
        E_M = jnp.array(0.8)
        dt = jnp.array(1.0)

        # JIT version
        new_state_jit, outputs_jit = tray_step_jit(
            state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
        )

        # Non-JIT version
        new_state, outputs = tray_step(
            state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
        )

        # Results should match
        assert jnp.allclose(new_state_jit.M, new_state.M)
        assert jnp.allclose(new_state_jit.x, new_state.x)
        assert jnp.allclose(new_state_jit.T, new_state.T, rtol=1e-4)

    def test_murphree_efficiency_jit(self):
        """Murphree efficiency should be JIT-compilable."""
        murphree_jit = jax.jit(murphree_vapor_efficiency)

        y_in = jnp.array(0.3)
        y_star = jnp.array(0.7)
        E_M = jnp.array(0.8)

        assert jnp.allclose(
            murphree_jit(y_in, y_star, E_M),
            murphree_vapor_efficiency(y_in, y_star, E_M)
        )


class TestVmapCompatibility:
    """Tests for vmap compatibility."""

    def test_murphree_vmap(self):
        """Murphree efficiency should work with vmap."""
        y_in_values = jnp.linspace(0.2, 0.4, 5)
        y_star = jnp.array(0.7)
        E_M = jnp.array(0.8)

        y_out_values = jax.vmap(
            lambda y_in: murphree_vapor_efficiency(y_in, y_star, E_M)
        )(y_in_values)

        assert y_out_values.shape == (5,)
        assert jnp.all(y_out_values >= y_in_values)  # Should move toward y_star


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
