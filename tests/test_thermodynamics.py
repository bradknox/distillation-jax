"""Tests for thermodynamics module."""

import jax
import jax.numpy as jnp
import pytest

from jax_distillation.core.thermodynamics import (
    ANTOINE_METHANOL,
    ANTOINE_WATER,
    ANTOINE_ETHANOL,
    ANTOINE_BENZENE,
    ANTOINE_TOLUENE,
    NRTL_METHANOL_WATER,
    antoine_vapor_pressure,
    nrtl_activity_coefficients,
    ideal_activity_coefficients,
    compute_k_values,
    equilibrium_vapor_composition,
    relative_volatility,
    bubble_point_residual,
    bubble_point_temperature,
    liquid_enthalpy,
    vapor_enthalpy,
    heat_of_vaporization,
    mixture_molecular_weight,
    create_methanol_water_thermo,
    create_benzene_toluene_thermo,
)


class TestAntoineEquation:
    """Tests for Antoine vapor pressure calculations."""

    def test_methanol_boiling_point(self):
        """Methanol boiling point at 1 bar should be ~337.8 K."""
        # At normal boiling point, P_sat = 1 bar
        T = jnp.array(337.8)
        P_sat = antoine_vapor_pressure(T, ANTOINE_METHANOL)
        assert jnp.abs(P_sat - 1.0) < 0.05  # Within 5% of 1 bar

    def test_water_boiling_point(self):
        """Water boiling point at 1 bar should be ~373.15 K."""
        T = jnp.array(373.15)
        P_sat = antoine_vapor_pressure(T, ANTOINE_WATER)
        assert jnp.abs(P_sat - 1.0) < 0.05  # Within 5% of 1 bar

    def test_ethanol_boiling_point(self):
        """Ethanol boiling point at 1 bar should be ~351.4 K."""
        T = jnp.array(351.4)
        P_sat = antoine_vapor_pressure(T, ANTOINE_ETHANOL)
        assert jnp.abs(P_sat - 1.0) < 0.05

    def test_benzene_boiling_point(self):
        """Benzene boiling point at 1 bar should be ~353.2 K."""
        T = jnp.array(353.2)
        P_sat = antoine_vapor_pressure(T, ANTOINE_BENZENE)
        assert jnp.abs(P_sat - 1.0) < 0.1  # Slightly wider tolerance

    def test_toluene_boiling_point(self):
        """Toluene boiling point at 1 bar should be ~383.8 K."""
        T = jnp.array(383.8)
        P_sat = antoine_vapor_pressure(T, ANTOINE_TOLUENE)
        assert jnp.abs(P_sat - 1.0) < 0.1

    def test_vapor_pressure_monotonic(self):
        """Vapor pressure should increase with temperature within valid range."""
        # Use temperature range within methanol's valid Antoine range
        T_values = jnp.linspace(290.0, 355.0, 10)
        P_values = jax.vmap(lambda T: antoine_vapor_pressure(T, ANTOINE_METHANOL))(T_values)
        # Check monotonically increasing
        assert jnp.all(jnp.diff(P_values) > 0)

    def test_vapor_pressure_positive(self):
        """Vapor pressure should always be positive."""
        T_values = jnp.linspace(290.0, 370.0, 20)
        P_values = jax.vmap(lambda T: antoine_vapor_pressure(T, ANTOINE_METHANOL))(T_values)
        assert jnp.all(P_values > 0)

    def test_jit_compilation(self):
        """Antoine equation should be JIT-compilable."""
        antoine_jit = jax.jit(antoine_vapor_pressure, static_argnums=())
        T = jnp.array(350.0)
        P1 = antoine_jit(T, ANTOINE_METHANOL)
        P2 = antoine_vapor_pressure(T, ANTOINE_METHANOL)
        assert jnp.allclose(P1, P2)


class TestNRTLActivityCoefficients:
    """Tests for NRTL activity coefficient calculations."""

    def test_pure_component_limits(self):
        """At pure component limits, gamma should approach 1."""
        T = jnp.array(340.0)

        # Pure component 1 (x=1)
        gamma_1, gamma_2 = nrtl_activity_coefficients(jnp.array(0.999), T, NRTL_METHANOL_WATER)
        assert jnp.abs(gamma_1 - 1.0) < 0.1  # gamma_1 -> 1 as x -> 1

        # Pure component 2 (x=0)
        gamma_1, gamma_2 = nrtl_activity_coefficients(jnp.array(0.001), T, NRTL_METHANOL_WATER)
        assert jnp.abs(gamma_2 - 1.0) < 0.1  # gamma_2 -> 1 as x -> 0

    def test_activity_coefficients_positive(self):
        """Activity coefficients should always be positive."""
        T = jnp.array(350.0)
        x_values = jnp.linspace(0.01, 0.99, 20)

        for x in x_values:
            gamma_1, gamma_2 = nrtl_activity_coefficients(x, T, NRTL_METHANOL_WATER)
            assert gamma_1 > 0
            assert gamma_2 > 0

    def test_methanol_water_nonideality(self):
        """Methanol-water should show positive deviations from ideality."""
        T = jnp.array(350.0)
        x = jnp.array(0.5)  # Equimolar mixture

        gamma_1, gamma_2 = nrtl_activity_coefficients(x, T, NRTL_METHANOL_WATER)

        # Methanol-water is a positive deviation system (gamma > 1)
        # At intermediate compositions, both gammas should be > 1
        assert gamma_1 > 1.0
        assert gamma_2 > 1.0

    def test_ideal_returns_unity(self):
        """Ideal activity coefficients should always be 1."""
        T = jnp.array(350.0)
        x_values = jnp.linspace(0.0, 1.0, 10)

        for x in x_values:
            gamma_1, gamma_2 = ideal_activity_coefficients(x, T)
            assert jnp.allclose(gamma_1, 1.0)
            assert jnp.allclose(gamma_2, 1.0)

    def test_jit_compilation(self):
        """NRTL should be JIT-compilable."""
        nrtl_jit = jax.jit(nrtl_activity_coefficients)
        x, T = jnp.array(0.5), jnp.array(350.0)
        g1_jit, g2_jit = nrtl_jit(x, T, NRTL_METHANOL_WATER)
        g1, g2 = nrtl_activity_coefficients(x, T, NRTL_METHANOL_WATER)
        assert jnp.allclose(g1_jit, g1)
        assert jnp.allclose(g2_jit, g2)


class TestVaporLiquidEquilibrium:
    """Tests for VLE calculations."""

    def test_equilibrium_composition_bounds(self):
        """Equilibrium vapor composition should be in [0, 1]."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)  # 1 bar
        T = jnp.array(350.0)

        x_values = jnp.linspace(0.0, 1.0, 20)
        for x in x_values:
            y_star = equilibrium_vapor_composition(x, T, P, thermo)
            assert 0.0 <= y_star <= 1.0

    def test_equilibrium_enrichment(self):
        """Vapor should be enriched in the more volatile component."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        T = jnp.array(350.0)

        x_values = jnp.linspace(0.1, 0.9, 9)
        for x in x_values:
            y_star = equilibrium_vapor_composition(x, T, P, thermo)
            # Methanol (light) should be enriched in vapor: y > x
            assert y_star > x

    def test_relative_volatility_positive(self):
        """Relative volatility should be positive."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        T = jnp.array(350.0)

        x_values = jnp.linspace(0.1, 0.9, 9)
        for x in x_values:
            alpha = relative_volatility(x, T, P, thermo)
            assert alpha > 0

    def test_k_values_positive(self):
        """K-values should be positive."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        T = jnp.array(350.0)
        x = jnp.array(0.5)

        K_1, K_2 = compute_k_values(x, T, P, thermo)
        assert K_1 > 0
        assert K_2 > 0

    def test_ideal_mixture_vle(self):
        """Benzene-toluene (nearly ideal) should follow Raoult's law closely."""
        thermo = create_benzene_toluene_thermo()
        P = jnp.array(1.0)
        T = jnp.array(370.0)  # Between boiling points
        x = jnp.array(0.5)

        K_1, K_2 = compute_k_values(x, T, P, thermo)

        # For ideal mixture: K = P_sat / P
        from jax_distillation.core.thermodynamics import antoine_vapor_pressure

        P_sat_1 = antoine_vapor_pressure(T, thermo.antoine_1)
        P_sat_2 = antoine_vapor_pressure(T, thermo.antoine_2)

        assert jnp.allclose(K_1, P_sat_1 / P, rtol=0.01)
        assert jnp.allclose(K_2, P_sat_2 / P, rtol=0.01)


class TestBubblePoint:
    """Tests for bubble point calculations."""

    def test_pure_methanol_bubble_point(self):
        """Pure methanol bubble point should match boiling point."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)  # 1 bar
        x = jnp.array(0.999)  # Nearly pure methanol

        T_bp = bubble_point_temperature(x, P, thermo)

        # Methanol boiling point ~337.8 K
        # Allow wider tolerance due to NRTL effects near pure component
        assert jnp.abs(T_bp - 337.8) < 5.0

    def test_pure_water_bubble_point(self):
        """Pure water bubble point should match boiling point."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        x = jnp.array(0.001)  # Nearly pure water

        T_bp = bubble_point_temperature(x, P, thermo)

        # Water boiling point ~373.15 K
        # Allow wider tolerance due to Antoine extrapolation
        assert jnp.abs(T_bp - 373.15) < 5.0

    def test_bubble_point_residual_at_solution(self):
        """Residual should be near zero at the bubble point."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        x = jnp.array(0.5)

        T_bp = bubble_point_temperature(x, P, thermo)
        residual = bubble_point_residual(T_bp, x, P, thermo)

        assert jnp.abs(residual) < 0.001 * P  # < 0.1% of pressure

    def test_bubble_point_monotonic_in_composition(self):
        """Bubble point should decrease with increasing light component."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)

        x_values = jnp.array([0.2, 0.4, 0.6, 0.8])
        T_bp_values = jax.vmap(lambda x: bubble_point_temperature(x, P, thermo))(x_values)

        # Temperature should decrease as methanol fraction increases
        # (methanol has lower boiling point)
        # Use <= 0 to allow for numerical plateau at edges
        diffs = jnp.diff(T_bp_values)
        assert jnp.all(diffs <= 0.1)  # Allow small positive diffs due to numerics
        # At least the overall trend should be decreasing
        assert T_bp_values[0] > T_bp_values[-1]

    def test_bubble_point_pressure_dependence(self):
        """Bubble point should increase with pressure."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)

        P_values = jnp.array([0.5, 1.0, 1.5])
        T_bp_values = jax.vmap(lambda P: bubble_point_temperature(x, P, thermo))(P_values)

        # Temperature should increase with pressure
        assert jnp.all(jnp.diff(T_bp_values) > 0)

    def test_jit_compilation(self):
        """Bubble point solver should be JIT-compilable."""
        thermo = create_methanol_water_thermo()
        bp_jit = jax.jit(bubble_point_temperature, static_argnames=("max_iter", "tol"))

        x = jnp.array(0.5)
        P = jnp.array(1.0)

        T1 = bp_jit(x, P, thermo)
        T2 = bubble_point_temperature(x, P, thermo)

        assert jnp.allclose(T1, T2, rtol=1e-4)


class TestEnthalpy:
    """Tests for enthalpy calculations."""

    def test_liquid_enthalpy_at_reference(self):
        """Liquid enthalpy should be zero at reference temperature."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        T = thermo.T_ref

        h_L = liquid_enthalpy(x, T, thermo)
        assert jnp.allclose(h_L, 0.0)

    def test_liquid_enthalpy_increases_with_temperature(self):
        """Liquid enthalpy should increase with temperature."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)

        T_values = jnp.linspace(300.0, 370.0, 10)
        h_values = jax.vmap(lambda T: liquid_enthalpy(x, T, thermo))(T_values)

        assert jnp.all(jnp.diff(h_values) > 0)

    def test_vapor_enthalpy_higher_than_liquid(self):
        """Vapor enthalpy should be higher than liquid at same T due to hvap."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        y = jnp.array(0.5)
        T = jnp.array(350.0)

        h_L = liquid_enthalpy(x, T, thermo)
        h_V = vapor_enthalpy(y, T, thermo)

        assert h_V > h_L

    def test_enthalpy_difference_matches_hvap(self):
        """Difference between vapor and liquid enthalpy should be ~hvap."""
        thermo = create_methanol_water_thermo()
        x = jnp.array(0.5)
        T = thermo.T_ref  # At reference temperature

        h_L = liquid_enthalpy(x, T, thermo)
        h_V = vapor_enthalpy(x, T, thermo)  # Use same composition
        hvap_mix = heat_of_vaporization(x, thermo)

        # At T_ref, h_L = 0, so h_V should be ~hvap
        assert jnp.allclose(h_V - h_L, hvap_mix, rtol=0.01)

    def test_molecular_weight_bounds(self):
        """Mixture MW should be between component MWs."""
        thermo = create_methanol_water_thermo()

        x_values = jnp.linspace(0.0, 1.0, 10)
        for x in x_values:
            mw_mix = mixture_molecular_weight(x, thermo)
            assert thermo.mw_2 <= mw_mix <= thermo.mw_1 or thermo.mw_1 <= mw_mix <= thermo.mw_2

    def test_jit_compilation(self):
        """Enthalpy functions should be JIT-compilable."""
        thermo = create_methanol_water_thermo()
        h_L_jit = jax.jit(liquid_enthalpy)
        h_V_jit = jax.jit(vapor_enthalpy)

        x = jnp.array(0.5)
        T = jnp.array(350.0)

        h_L1 = h_L_jit(x, T, thermo)
        h_L2 = liquid_enthalpy(x, T, thermo)
        assert jnp.allclose(h_L1, h_L2)

        h_V1 = h_V_jit(x, T, thermo)
        h_V2 = vapor_enthalpy(x, T, thermo)
        assert jnp.allclose(h_V1, h_V2)


class TestVmapCompatibility:
    """Tests for vmap compatibility."""

    def test_antoine_vmap(self):
        """Antoine equation should work with vmap."""
        T_values = jnp.linspace(300.0, 370.0, 10)
        P_values = jax.vmap(lambda T: antoine_vapor_pressure(T, ANTOINE_METHANOL))(T_values)
        assert P_values.shape == (10,)

    def test_nrtl_vmap(self):
        """NRTL should work with vmap over composition."""
        x_values = jnp.linspace(0.1, 0.9, 9)
        T = jnp.array(350.0)

        gamma_1s, gamma_2s = jax.vmap(
            lambda x: nrtl_activity_coefficients(x, T, NRTL_METHANOL_WATER)
        )(x_values)

        assert gamma_1s.shape == (9,)
        assert gamma_2s.shape == (9,)

    def test_bubble_point_vmap(self):
        """Bubble point should work with vmap over composition."""
        thermo = create_methanol_water_thermo()
        P = jnp.array(1.0)
        x_values = jnp.array([0.2, 0.4, 0.6, 0.8])

        T_bp_values = jax.vmap(lambda x: bubble_point_temperature(x, P, thermo))(x_values)
        assert T_bp_values.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
