"""Tray hydraulics calculations for distillation simulation.

This module provides JIT-compilable functions for:
- Francis weir formula for liquid outflow
- Hydraulic coupling dynamics
- Flooding and weeping constraints
- Density calculations

All functions are pure and compatible with JAX transformations.
"""

import jax.numpy as jnp

from jax_distillation.core.types import HydraulicParams, ThermoParams


# =============================================================================
# Density Calculations
# =============================================================================


def liquid_density(
    x: jnp.ndarray,
    T: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Estimate liquid mixture density using ideal mixing.

    This is a simplified model; for higher accuracy, use DIPPR correlations.

    Args:
        x: Liquid mole fraction of light component.
        T: Temperature [K].
        thermo: Thermodynamic parameters.

    Returns:
        Liquid density [kg/m^3].
    """
    # Pure component densities (approximate, temperature-corrected)
    # Using typical values for methanol and water
    # rho = rho_ref * (1 - beta * (T - T_ref))
    # where beta is thermal expansion coefficient

    # Reference densities at 298.15 K [kg/m^3]
    rho_1_ref = 791.0  # Methanol
    rho_2_ref = 997.0  # Water

    # Thermal expansion coefficients [1/K]
    beta_1 = 1.2e-3  # Methanol
    beta_2 = 2.1e-4  # Water

    T_ref = 298.15

    # Temperature-corrected densities
    rho_1 = rho_1_ref * (1.0 - beta_1 * (T - T_ref))
    rho_2 = rho_2_ref * (1.0 - beta_2 * (T - T_ref))

    # Ideal mixing (volume additivity)
    # 1/rho_mix = x1/rho_1 + x2/rho_2 (mass basis requires conversion)
    # For simplicity, use mole-fraction weighted average
    mw_1 = thermo.mw_1
    mw_2 = thermo.mw_2
    mw_mix = x * mw_1 + (1.0 - x) * mw_2

    # Molar volumes
    V_1 = mw_1 / rho_1  # m^3/mol
    V_2 = mw_2 / rho_2

    # Mixture molar volume (ideal)
    V_mix = x * V_1 + (1.0 - x) * V_2

    # Mixture density
    rho_mix = mw_mix / V_mix

    return rho_mix


def vapor_density(
    y: jnp.ndarray,
    T: jnp.ndarray,
    P: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate vapor mixture density using ideal gas law.

    Args:
        y: Vapor mole fraction of light component.
        T: Temperature [K].
        P: Pressure [bar].
        thermo: Thermodynamic parameters.

    Returns:
        Vapor density [kg/m^3].
    """
    # Mixture molecular weight
    mw_mix = y * thermo.mw_1 + (1.0 - y) * thermo.mw_2

    # Ideal gas law: PV = nRT => rho = P * MW / (R * T)
    R = 8.314e-5  # Gas constant [bar * m^3 / (mol * K)]

    rho_v = P * mw_mix / (R * T)

    return rho_v


# =============================================================================
# Surface Tension (simplified)
# =============================================================================


def surface_tension(
    x: jnp.ndarray,
    T: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate surface tension for mixture.

    Uses simple linear interpolation between pure component values.

    Args:
        x: Liquid mole fraction of light component.
        T: Temperature [K].

    Returns:
        Surface tension [mN/m].
    """
    # Surface tension at 298 K [mN/m]
    sigma_1_ref = 22.5  # Methanol
    sigma_2_ref = 72.0  # Water

    # Temperature coefficient [mN/m/K]
    dsigma_dT_1 = -0.08
    dsigma_dT_2 = -0.15

    T_ref = 298.15

    # Temperature-corrected surface tensions
    sigma_1 = sigma_1_ref + dsigma_dT_1 * (T - T_ref)
    sigma_2 = sigma_2_ref + dsigma_dT_2 * (T - T_ref)

    # Linear mixing (simplified)
    sigma = x * sigma_1 + (1.0 - x) * sigma_2

    # Ensure positive
    return jnp.maximum(sigma, 5.0)


# =============================================================================
# Francis Weir Formula
# =============================================================================


def clear_liquid_height(
    holdup: jnp.ndarray,
    tray_area: jnp.ndarray,
    rho_L: jnp.ndarray,
    mw: jnp.ndarray,
    weir_height: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate clear liquid height over weir from holdup.

    Args:
        holdup: Liquid holdup on tray [mol].
        tray_area: Active tray area [m^2].
        rho_L: Liquid density [kg/m^3].
        mw: Mixture molecular weight [kg/mol].
        weir_height: Weir height [m].

    Returns:
        Clear liquid height over weir [m]. Can be negative if holdup is low.
    """
    # Total liquid volume on tray
    volume = holdup * mw / rho_L  # [m^3]

    # Liquid height on tray (assuming uniform distribution)
    h_total = volume / tray_area  # [m]

    # Height over weir
    h_ow = h_total - weir_height

    return h_ow


def francis_weir_flow(
    h_ow: jnp.ndarray,
    weir_length: jnp.ndarray,
    weir_coefficient: jnp.ndarray,
    rho_L: jnp.ndarray,
    mw: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate liquid flow rate over weir using Francis formula.

    Q_L = C_w * L_w * h_ow^(3/2)  [m^3/s]
    L = Q_L * rho_L / mw  [mol/s]

    Args:
        h_ow: Clear liquid height over weir [m].
        weir_length: Weir length [m].
        weir_coefficient: Weir discharge coefficient (~1.84 for SI units).
        rho_L: Liquid density [kg/m^3].
        mw: Mixture molecular weight [kg/mol].

    Returns:
        Liquid molar flow rate [mol/s].
    """
    # Only positive height contributes to flow
    h_ow_pos = jnp.maximum(h_ow, 0.0)

    # Volumetric flow rate [m^3/s]
    Q_L = weir_coefficient * weir_length * jnp.power(h_ow_pos, 1.5)

    # Convert to molar flow rate [mol/s]
    L = Q_L * rho_L / mw

    return L


def static_liquid_outflow(
    holdup: jnp.ndarray,
    params: HydraulicParams,
    rho_L: jnp.ndarray,
    mw: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate static (steady-state) liquid outflow from weir relation.

    Args:
        holdup: Liquid holdup on tray [mol].
        params: Hydraulic parameters.
        rho_L: Liquid density [kg/m^3].
        mw: Mixture molecular weight [kg/mol].

    Returns:
        Static liquid outflow rate [mol/s].
    """
    # Clear liquid height over weir
    h_ow = clear_liquid_height(
        holdup, params.tray_area, rho_L, mw, params.weir_height
    )

    # Francis weir flow
    L_static = francis_weir_flow(
        h_ow, params.weir_length, params.weir_coefficient, rho_L, mw
    )

    return L_static


# =============================================================================
# Hydraulic Coupling Dynamics
# =============================================================================


def hydraulic_coupling_derivative(
    L_out: jnp.ndarray,
    L_static: jnp.ndarray,
    V_in: jnp.ndarray,
    V_in_prev: jnp.ndarray,
    tau_L: jnp.ndarray,
    j: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate derivative of dynamic liquid outflow.

    The hydraulic coupling model captures the initial response of liquid
    flow to changes in holdup and vapor flow:

    dL_out/dt = (1/tau_L) * (L_static - L_out) + j * dV_in/dt

    This is critical for realistic transient behavior in control applications.

    Args:
        L_out: Current dynamic liquid outflow [mol/s].
        L_static: Static (weir-based) liquid outflow [mol/s].
        V_in: Current vapor inflow [mol/s].
        V_in_prev: Previous vapor inflow [mol/s].
        tau_L: Hydraulic time constant [s].
        j: Vapor-liquid coupling parameter (dimensionless).
        dt: Time step [s].

    Returns:
        Time derivative of liquid outflow [mol/s^2].
    """
    # First-order lag toward static value
    dL_dt_lag = (L_static - L_out) / tau_L

    # Vapor coupling term
    dV_dt = (V_in - V_in_prev) / dt
    dL_dt_vapor = j * dV_dt

    return dL_dt_lag + dL_dt_vapor


def update_liquid_outflow(
    L_out: jnp.ndarray,
    L_static: jnp.ndarray,
    V_in: jnp.ndarray,
    V_in_prev: jnp.ndarray,
    tau_L: jnp.ndarray,
    j: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Update dynamic liquid outflow with hydraulic coupling.

    Uses a semi-implicit update for stability.

    Args:
        L_out: Current dynamic liquid outflow [mol/s].
        L_static: Static (weir-based) liquid outflow [mol/s].
        V_in: Current vapor inflow [mol/s].
        V_in_prev: Previous vapor inflow [mol/s].
        tau_L: Hydraulic time constant [s].
        j: Vapor-liquid coupling parameter.
        dt: Time step [s].

    Returns:
        Updated liquid outflow [mol/s].
    """
    # Semi-implicit treatment of first-order lag
    # L_new = L_old + dt * [(L_static - L_new) / tau_L + j * dV/dt]
    # L_new * (1 + dt/tau_L) = L_old + dt * L_static / tau_L + dt * j * dV/dt
    # L_new = [L_old + dt * L_static / tau_L + dt * j * dV/dt] / (1 + dt/tau_L)

    dV_dt = (V_in - V_in_prev) / jnp.maximum(dt, 1e-6)

    L_new = (L_out + dt * L_static / tau_L + dt * j * dV_dt) / (1.0 + dt / tau_L)

    # Ensure non-negative
    return jnp.maximum(L_new, 0.0)


# =============================================================================
# Flooding Correlation
# =============================================================================


def flooding_velocity(
    rho_L: jnp.ndarray,
    rho_V: jnp.ndarray,
    sigma: jnp.ndarray,
    C_sbf: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate flooding superficial gas velocity using Fair correlation.

    U_n,f = C_sbf * (sigma/20)^0.2 * ((rho_L - rho_V) / rho_V)^0.5

    Args:
        rho_L: Liquid density [kg/m^3].
        rho_V: Vapor density [kg/m^3].
        sigma: Surface tension [mN/m].
        C_sbf: Flooding capacity parameter (depends on tray spacing).

    Returns:
        Flooding superficial velocity [m/s].
    """
    # Ensure positive density difference
    rho_diff = jnp.maximum(rho_L - rho_V, 1.0)

    U_nf = C_sbf * jnp.power(sigma / 20.0, 0.2) * jnp.sqrt(rho_diff / rho_V)

    return U_nf


def flooding_ratio(
    V: jnp.ndarray,
    net_area: jnp.ndarray,
    rho_V: jnp.ndarray,
    mw_V: jnp.ndarray,
    U_nf: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the ratio of actual to flooding vapor velocity.

    Args:
        V: Vapor molar flow rate [mol/s].
        net_area: Net column area for vapor flow [m^2].
        rho_V: Vapor density [kg/m^3].
        mw_V: Vapor molecular weight [kg/mol].
        U_nf: Flooding superficial velocity [m/s].

    Returns:
        Flooding ratio (dimensionless). Values > 0.85 indicate near-flooding.
    """
    # Volumetric vapor flow [m^3/s]
    Q_V = V * mw_V / rho_V

    # Superficial velocity [m/s]
    U_n = Q_V / net_area

    # Flooding ratio
    return U_n / jnp.maximum(U_nf, 1e-6)


def is_flooding(flooding_ratio: jnp.ndarray, threshold: float = 0.85) -> jnp.ndarray:
    """Check if column is flooding.

    Args:
        flooding_ratio: Ratio of actual to flooding velocity.
        threshold: Flooding threshold (default 0.85).

    Returns:
        Boolean indicating flooding condition.
    """
    return flooding_ratio > threshold


# =============================================================================
# Weeping Correlation
# =============================================================================


def weep_point_velocity(
    rho_V: jnp.ndarray,
    hole_diameter: jnp.ndarray,
    K2: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate minimum vapor velocity to prevent weeping.

    U_min = (K2 - 0.9 * (25.4 - d_h)) / sqrt(rho_V)

    Args:
        rho_V: Vapor density [kg/m^3].
        hole_diameter: Hole diameter [mm].
        K2: Weeping correlation coefficient.

    Returns:
        Minimum vapor velocity [m/s].
    """
    # Convert hole diameter to mm if in meters
    d_h_mm = hole_diameter * 1000.0  # Assume input in meters

    # Weep point velocity
    U_min = (K2 - 0.9 * (25.4 - d_h_mm)) / jnp.sqrt(rho_V)

    return jnp.maximum(U_min, 0.0)


def is_weeping(
    V: jnp.ndarray,
    net_area: jnp.ndarray,
    rho_V: jnp.ndarray,
    mw_V: jnp.ndarray,
    U_min: jnp.ndarray,
) -> jnp.ndarray:
    """Check if tray is weeping (vapor velocity too low).

    Args:
        V: Vapor molar flow rate [mol/s].
        net_area: Net column area for vapor flow [m^2].
        rho_V: Vapor density [kg/m^3].
        mw_V: Vapor molecular weight [kg/mol].
        U_min: Minimum velocity to prevent weeping [m/s].

    Returns:
        Boolean indicating weeping condition.
    """
    # Volumetric vapor flow [m^3/s]
    Q_V = V * mw_V / rho_V

    # Superficial velocity [m/s]
    U_n = Q_V / net_area

    return U_n < U_min


# =============================================================================
# Tray Efficiency Degradation
# =============================================================================


def efficiency_degradation(
    flooding_ratio: jnp.ndarray,
    E_M_nominal: jnp.ndarray,
    flooding_threshold: float = 0.85,
) -> jnp.ndarray:
    """Calculate degraded tray efficiency near flooding.

    Efficiency decreases as flooding is approached.

    Args:
        flooding_ratio: Current flooding ratio.
        E_M_nominal: Nominal Murphree efficiency.
        flooding_threshold: Threshold above which efficiency degrades.

    Returns:
        Degraded Murphree efficiency.
    """
    # Linear degradation above threshold
    degradation_factor = jnp.where(
        flooding_ratio > flooding_threshold,
        1.0 - 2.0 * (flooding_ratio - flooding_threshold),
        1.0,
    )

    # Clamp to [0.1, 1.0]
    degradation_factor = jnp.clip(degradation_factor, 0.1, 1.0)

    return E_M_nominal * degradation_factor


# =============================================================================
# Default Hydraulic Parameters
# =============================================================================


def create_default_hydraulic_params(
    column_diameter: float = 0.05,  # 50mm Armfield column
    weir_height: float = 0.02,  # 20mm
    tray_spacing: float = 0.25,  # 250mm
    tau_L: float = 3.0,  # 3s (middle of 0.5-15s range)
    j: float = 0.0,  # No vapor coupling by default
) -> HydraulicParams:
    """Create default hydraulic parameters for a teaching column.

    Args:
        column_diameter: Internal column diameter [m].
        weir_height: Weir height [m].
        tray_spacing: Tray spacing [m].
        tau_L: Hydraulic time constant [s].
        j: Vapor-liquid coupling parameter.

    Returns:
        HydraulicParams dataclass.
    """
    # Derived parameters
    column_area = jnp.pi * (column_diameter / 2) ** 2
    downcomer_fraction = 0.1  # 10% for downcomer
    active_area = column_area * (1.0 - downcomer_fraction)
    downcomer_area = column_area * downcomer_fraction
    weir_length = 0.7 * column_diameter  # Typical weir length

    # Weir coefficient for SI units (m, s)
    weir_coefficient = 1.84

    # Flooding capacity parameter (depends on tray spacing)
    # Approximate from correlations
    C_sbf = 0.03 + 0.015 * tray_spacing  # Simplified correlation

    # Weeping K2 parameter (typical for sieve trays)
    K2 = 30.0

    return HydraulicParams(
        tau_L=jnp.array(tau_L),
        j=jnp.array(j),
        weir_height=jnp.array(weir_height),
        weir_length=jnp.array(weir_length),
        weir_coefficient=jnp.array(weir_coefficient),
        tray_area=jnp.array(active_area),
        downcomer_area=jnp.array(downcomer_area),
        hole_diameter=jnp.array(0.003),  # 3mm holes
        C_sbf=jnp.array(C_sbf),
        K2_weep=jnp.array(K2),
    )
