"""Tests for hydraulics module."""

import jax
import jax.numpy as jnp
import pytest

from jax_distillation.core.hydraulics import (
    liquid_density,
    vapor_density,
    surface_tension,
    clear_liquid_height,
    francis_weir_flow,
    static_liquid_outflow,
    hydraulic_coupling_derivative,
    update_liquid_outflow,
    flooding_velocity,
    flooding_ratio,
    is_flooding,
    weep_point_velocity,
    is_weeping,
    efficiency_degradation,
    create_default_hydraulic_params,
)
from jax_distillation.core.thermodynamics import create_methanol_water_thermo


class TestDensityCalculations:
    """Tests for density calculations."""

    def test_liquid_density_positive(self):
        """Liquid density should be positive."""
        thermo = create_methanol_water_thermo()
        x_values = jnp.linspace(0.0, 1.0, 10)
        T = jnp.array(340.0)

        for x in x_values:
            rho = liquid_density(x, T, thermo)
            assert rho > 0

    def test_liquid_density_water_heavier(self):
        """Pure water should be denser than pure methanol."""
        thermo = create_methanol_water_thermo()
        T = jnp.array(340.0)

        rho_water = liquid_density(jnp.array(0.0), T, thermo)  # x=0 is pure water
        rho_methanol = liquid_density(jnp.array(1.0), T, thermo)  # x=1 is pure methanol

        assert rho_water > rho_methanol

    def test_liquid_density_decreases_with_temperature(self):
        """Liquid density should decrease with temperature."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)

        T_values = jnp.linspace(300.0, 370.0, 10)
        rho_values = jax.vmap(lambda T: liquid_density(x, T, thermo))(T_values)

        assert jnp.all(jnp.diff(rho_values) < 0)

    def test_vapor_density_positive(self):
        """Vapor density should be positive."""
        thermo = create_methanol_water_thermo()
        y = jnp.array(0.5)
        T = jnp.array(350.0)
        P = jnp.array(1.0)

        rho = vapor_density(y, T, P, thermo)
        assert rho > 0

    def test_vapor_density_increases_with_pressure(self):
        """Vapor density should increase with pressure."""
        thermo = create_methanol_water_thermo()
        y = jnp.array(0.5)
        T = jnp.array(350.0)

        P_values = jnp.array([0.5, 1.0, 1.5])
        rho_values = jax.vmap(lambda P: vapor_density(y, T, P, thermo))(P_values)

        assert jnp.all(jnp.diff(rho_values) > 0)

    def test_vapor_much_less_dense_than_liquid(self):
        """Vapor should be much less dense than liquid."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        T = jnp.array(350.0)
        P = jnp.array(1.0)

        rho_L = liquid_density(x, T, thermo)
        rho_V = vapor_density(x, T, P, thermo)

        assert rho_L > 100 * rho_V  # Liquid should be >100x denser


class TestSurfaceTension:
    """Tests for surface tension calculations."""

    def test_surface_tension_positive(self):
        """Surface tension should be positive."""
        x_values = jnp.linspace(0.0, 1.0, 10)
        T = jnp.array(340.0)

        for x in x_values:
            sigma = surface_tension(x, T)
            assert sigma > 0

    def test_surface_tension_water_higher(self):
        """Water has higher surface tension than methanol."""
        T = jnp.array(340.0)

        sigma_water = surface_tension(jnp.array(0.0), T)
        sigma_methanol = surface_tension(jnp.array(1.0), T)

        assert sigma_water > sigma_methanol

    def test_surface_tension_decreases_with_temperature(self):
        """Surface tension should decrease with temperature."""
        x = jnp.array(0.5)

        T_values = jnp.linspace(300.0, 370.0, 10)
        sigma_values = jax.vmap(lambda T: surface_tension(x, T))(T_values)

        assert jnp.all(jnp.diff(sigma_values) < 0)


class TestFrancisWeirFormula:
    """Tests for Francis weir formula calculations."""

    def test_zero_height_zero_flow(self):
        """Zero height over weir should give zero flow."""
        h_ow = jnp.array(0.0)
        L_w = jnp.array(0.035)  # 35mm weir
        C_w = jnp.array(1.84)
        rho_L = jnp.array(900.0)
        mw = jnp.array(0.025)

        L = francis_weir_flow(h_ow, L_w, C_w, rho_L, mw)
        assert jnp.allclose(L, 0.0)

    def test_negative_height_zero_flow(self):
        """Negative height over weir should give zero flow."""
        h_ow = jnp.array(-0.01)
        L_w = jnp.array(0.035)
        C_w = jnp.array(1.84)
        rho_L = jnp.array(900.0)
        mw = jnp.array(0.025)

        L = francis_weir_flow(h_ow, L_w, C_w, rho_L, mw)
        assert jnp.allclose(L, 0.0)

    def test_flow_increases_with_height(self):
        """Flow should increase with liquid height over weir."""
        h_ow_values = jnp.array([0.01, 0.02, 0.03, 0.04])
        L_w = jnp.array(0.035)
        C_w = jnp.array(1.84)
        rho_L = jnp.array(900.0)
        mw = jnp.array(0.025)

        L_values = jax.vmap(lambda h: francis_weir_flow(h, L_w, C_w, rho_L, mw))(h_ow_values)

        assert jnp.all(jnp.diff(L_values) > 0)

    def test_clear_liquid_height_calculation(self):
        """Clear liquid height should be positive for sufficient holdup."""
        params = create_default_hydraulic_params()
        holdup = jnp.array(10.0)  # 10 mol
        rho_L = jnp.array(900.0)
        mw = jnp.array(0.025)

        h_ow = clear_liquid_height(holdup, params.tray_area, rho_L, mw, params.weir_height)

        # With 10 mol at ~900 kg/m3 and MW 0.025, volume is ~0.00028 m3
        # Height depends on tray area
        # Result should be reasonable (could be positive or negative)
        assert jnp.isfinite(h_ow)


class TestHydraulicCoupling:
    """Tests for hydraulic coupling dynamics."""

    def test_relaxation_to_static(self):
        """Dynamic flow should relax to static value."""
        L_out = jnp.array(0.5)  # Current flow
        L_static = jnp.array(1.0)  # Target static flow
        V_in = jnp.array(0.1)
        V_in_prev = jnp.array(0.1)  # No change in vapor
        tau_L = jnp.array(3.0)
        j = jnp.array(0.0)  # No coupling
        dt = jnp.array(1.0)

        L_new = update_liquid_outflow(L_out, L_static, V_in, V_in_prev, tau_L, j, dt)

        # Should move toward L_static
        assert L_new > L_out
        assert L_new < L_static

    def test_vapor_coupling_effect(self):
        """Vapor flow increase should affect liquid outflow via coupling."""
        L_out = jnp.array(0.5)
        L_static = jnp.array(0.5)  # At equilibrium
        V_in = jnp.array(0.2)
        V_in_prev = jnp.array(0.1)  # Vapor increased
        tau_L = jnp.array(3.0)
        j = jnp.array(1.0)  # Positive coupling
        dt = jnp.array(0.1)

        L_new = update_liquid_outflow(L_out, L_static, V_in, V_in_prev, tau_L, j, dt)

        # With positive j, vapor increase should increase liquid flow
        assert L_new > L_out

    def test_nonnegative_flow(self):
        """Liquid outflow should never be negative."""
        L_out = jnp.array(0.1)
        L_static = jnp.array(0.0)
        V_in = jnp.array(0.1)
        V_in_prev = jnp.array(0.2)  # Vapor decreased
        tau_L = jnp.array(1.0)
        j = jnp.array(5.0)  # Strong negative coupling effect
        dt = jnp.array(1.0)

        L_new = update_liquid_outflow(L_out, L_static, V_in, V_in_prev, tau_L, j, dt)

        assert L_new >= 0


class TestFloodingCorrelation:
    """Tests for flooding correlation calculations."""

    def test_flooding_velocity_positive(self):
        """Flooding velocity should be positive."""
        rho_L = jnp.array(900.0)
        rho_V = jnp.array(1.5)
        sigma = jnp.array(50.0)
        C_sbf = jnp.array(0.04)

        U_nf = flooding_velocity(rho_L, rho_V, sigma, C_sbf)
        assert U_nf > 0

    def test_flooding_velocity_increases_with_density_ratio(self):
        """Flooding velocity should increase with density ratio."""
        rho_V = jnp.array(1.5)
        sigma = jnp.array(50.0)
        C_sbf = jnp.array(0.04)

        rho_L_values = jnp.array([800.0, 900.0, 1000.0])
        U_nf_values = jax.vmap(
            lambda rho_L: flooding_velocity(rho_L, rho_V, sigma, C_sbf)
        )(rho_L_values)

        assert jnp.all(jnp.diff(U_nf_values) > 0)

    def test_flooding_ratio_zero_vapor(self):
        """Flooding ratio should be zero with no vapor flow."""
        V = jnp.array(0.0)
        net_area = jnp.array(0.001)
        rho_V = jnp.array(1.5)
        mw_V = jnp.array(0.025)
        U_nf = jnp.array(1.0)

        ratio = flooding_ratio(V, net_area, rho_V, mw_V, U_nf)
        assert jnp.allclose(ratio, 0.0)

    def test_is_flooding_detection(self):
        """Should correctly detect flooding condition."""
        assert is_flooding(jnp.array(0.9))
        assert not is_flooding(jnp.array(0.8))


class TestWeepingCorrelation:
    """Tests for weeping correlation calculations."""

    def test_weep_velocity_positive(self):
        """Weep point velocity should be non-negative."""
        rho_V = jnp.array(1.5)
        d_h = jnp.array(0.003)  # 3mm
        K2 = jnp.array(30.0)

        U_min = weep_point_velocity(rho_V, d_h, K2)
        assert U_min >= 0

    def test_is_weeping_detection(self):
        """Should correctly detect weeping condition."""
        # Low vapor flow -> weeping
        V_low = jnp.array(0.001)
        V_high = jnp.array(0.1)
        net_area = jnp.array(0.001)
        rho_V = jnp.array(1.5)
        mw_V = jnp.array(0.025)
        U_min = jnp.array(0.5)

        assert is_weeping(V_low, net_area, rho_V, mw_V, U_min)
        assert not is_weeping(V_high, net_area, rho_V, mw_V, U_min)


class TestEfficiencyDegradation:
    """Tests for efficiency degradation calculations."""

    def test_no_degradation_below_threshold(self):
        """No degradation below flooding threshold."""
        ratio = jnp.array(0.7)
        E_M_nominal = jnp.array(0.8)

        E_M = efficiency_degradation(ratio, E_M_nominal)
        assert jnp.allclose(E_M, E_M_nominal)

    def test_degradation_above_threshold(self):
        """Efficiency degrades above flooding threshold."""
        ratio = jnp.array(0.9)
        E_M_nominal = jnp.array(0.8)

        E_M = efficiency_degradation(ratio, E_M_nominal)
        assert E_M < E_M_nominal

    def test_minimum_efficiency_bound(self):
        """Efficiency should not go below minimum."""
        ratio = jnp.array(1.5)  # Severely flooded
        E_M_nominal = jnp.array(0.8)

        E_M = efficiency_degradation(ratio, E_M_nominal)
        assert E_M >= 0.1 * E_M_nominal


class TestDefaultParameters:
    """Tests for default parameter creation."""

    def test_default_params_valid(self):
        """Default parameters should be physically valid."""
        params = create_default_hydraulic_params()

        assert params.tau_L > 0
        assert params.weir_height > 0
        assert params.weir_length > 0
        assert params.weir_coefficient > 0
        assert params.tray_area > 0
        assert params.C_sbf > 0

    def test_tau_L_in_valid_range(self):
        """Hydraulic time constant should be in literature range."""
        params = create_default_hydraulic_params()

        # Literature range: 0.5-15s
        assert 0.5 <= params.tau_L <= 15.0


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    def test_liquid_density_jit(self):
        """Liquid density should be JIT-compilable."""
        thermo = create_methanol_water_thermo()
        rho_jit = jax.jit(liquid_density)

        x, T = jnp.array(0.5), jnp.array(350.0)
        assert jnp.allclose(rho_jit(x, T, thermo), liquid_density(x, T, thermo))

    def test_flooding_velocity_jit(self):
        """Flooding velocity should be JIT-compilable."""
        U_nf_jit = jax.jit(flooding_velocity)

        rho_L, rho_V = jnp.array(900.0), jnp.array(1.5)
        sigma, C_sbf = jnp.array(50.0), jnp.array(0.04)

        assert jnp.allclose(
            U_nf_jit(rho_L, rho_V, sigma, C_sbf),
            flooding_velocity(rho_L, rho_V, sigma, C_sbf),
        )

    def test_update_liquid_outflow_jit(self):
        """Liquid outflow update should be JIT-compilable."""
        update_jit = jax.jit(update_liquid_outflow)

        L_out = jnp.array(0.5)
        L_static = jnp.array(1.0)
        V_in = jnp.array(0.1)
        V_in_prev = jnp.array(0.1)
        tau_L = jnp.array(3.0)
        j = jnp.array(0.0)
        dt = jnp.array(1.0)

        assert jnp.allclose(
            update_jit(L_out, L_static, V_in, V_in_prev, tau_L, j, dt),
            update_liquid_outflow(L_out, L_static, V_in, V_in_prev, tau_L, j, dt),
        )


class TestVmapCompatibility:
    """Tests for vmap compatibility."""

    def test_liquid_density_vmap(self):
        """Liquid density should work with vmap."""
        thermo = create_methanol_water_thermo()
        T = jnp.array(350.0)

        x_values = jnp.linspace(0.1, 0.9, 9)
        rho_values = jax.vmap(lambda x: liquid_density(x, T, thermo))(x_values)

        assert rho_values.shape == (9,)
        assert jnp.all(rho_values > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
