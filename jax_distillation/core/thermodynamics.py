"""Thermodynamic calculations for vapor-liquid equilibrium.

This module provides JIT-compilable functions for:
- Antoine equation (vapor pressure)
- NRTL activity coefficients
- K-values and equilibrium vapor composition
- Bubble point calculations
- Enthalpy calculations

All functions are pure and compatible with JAX transformations (jit, vmap, grad).
"""

import jax
import jax.numpy as jnp
from jax import lax

from jax_distillation.core.types import AntoineParams, NRTLParams, ThermoParams


# =============================================================================
# Antoine Equation
# =============================================================================


def antoine_vapor_pressure(T: jnp.ndarray, params: AntoineParams) -> jnp.ndarray:
    """Calculate saturation vapor pressure using the Antoine equation.

    Uses the NIST form: log10(P_sat[bar]) = A - B / (T[K] + C)

    Args:
        T: Temperature [K].
        params: Antoine parameters for the component.

    Returns:
        Saturation vapor pressure [bar].
    """
    # Clamp temperature to valid range to avoid numerical issues
    T_clamped = jnp.clip(T, params.T_min, params.T_max)
    log_p = params.A - params.B / (T_clamped + params.C)
    return jnp.power(10.0, log_p)


def create_antoine_params(
    A: float, B: float, C: float, T_min: float, T_max: float
) -> AntoineParams:
    """Create Antoine parameters from scalar values.

    Args:
        A: Antoine A coefficient.
        B: Antoine B coefficient.
        C: Antoine C coefficient.
        T_min: Minimum valid temperature [K].
        T_max: Maximum valid temperature [K].

    Returns:
        AntoineParams dataclass.
    """
    return AntoineParams(
        A=jnp.array(A),
        B=jnp.array(B),
        C=jnp.array(C),
        T_min=jnp.array(T_min),
        T_max=jnp.array(T_max),
    )


# Default Antoine parameters from NIST WebBook
ANTOINE_METHANOL = create_antoine_params(
    A=5.20409, B=1581.341, C=-33.50, T_min=288.10, T_max=356.83
)

ANTOINE_WATER = create_antoine_params(
    A=5.08354, B=1663.125, C=-45.622, T_min=344.00, T_max=373.00
)

ANTOINE_ETHANOL = create_antoine_params(
    A=5.24677, B=1598.673, C=-46.424, T_min=292.77, T_max=366.63
)

ANTOINE_BENZENE = create_antoine_params(
    A=4.01814, B=1203.835, C=-53.226, T_min=287.70, T_max=354.07
)

ANTOINE_TOLUENE = create_antoine_params(
    A=4.07827, B=1343.943, C=-53.773, T_min=308.52, T_max=384.66
)


# =============================================================================
# Activity Coefficients
# =============================================================================


def nrtl_activity_coefficients(
    x: jnp.ndarray, T: jnp.ndarray, params: NRTLParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate activity coefficients using the NRTL model for a binary mixture.

    NRTL equations (binary form):
        tau_12 = A_12 + B_12/T, tau_21 = A_21 + B_21/T
        G_12 = exp(-alpha * tau_12), G_21 = exp(-alpha * tau_21)
        ln(gamma_1) = x_2^2 * [tau_21 * (G_21 / (x_1 + x_2*G_21))^2
                              + tau_12 * G_12 / (x_2 + x_1*G_12)^2]
        ln(gamma_2) = x_1^2 * [tau_12 * (G_12 / (x_2 + x_1*G_12))^2
                              + tau_21 * G_21 / (x_1 + x_2*G_21)^2]

    Args:
        x: Mole fraction of light component (component 1).
        T: Temperature [K].
        params: NRTL parameters.

    Returns:
        Tuple of (gamma_1, gamma_2) activity coefficients.
    """
    x1 = x
    x2 = 1.0 - x

    # Temperature-dependent interaction parameters
    tau_12 = params.A_12 + params.B_12 / T
    tau_21 = params.A_21 + params.B_21 / T

    # G parameters
    G_12 = jnp.exp(-params.alpha * tau_12)
    G_21 = jnp.exp(-params.alpha * tau_21)

    # Denominators (avoid division by zero)
    denom_1 = x1 + x2 * G_21
    denom_2 = x2 + x1 * G_12

    # Add small epsilon for numerical stability at pure component limits
    eps = 1e-10
    denom_1 = jnp.maximum(denom_1, eps)
    denom_2 = jnp.maximum(denom_2, eps)

    # Activity coefficients
    ln_gamma_1 = x2 * x2 * (
        tau_21 * jnp.square(G_21 / denom_1) + tau_12 * G_12 / jnp.square(denom_2)
    )
    ln_gamma_2 = x1 * x1 * (
        tau_12 * jnp.square(G_12 / denom_2) + tau_21 * G_21 / jnp.square(denom_1)
    )

    gamma_1 = jnp.exp(ln_gamma_1)
    gamma_2 = jnp.exp(ln_gamma_2)

    return gamma_1, gamma_2


def ideal_activity_coefficients(
    x: jnp.ndarray, T: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ideal activity coefficients (gamma = 1).

    Args:
        x: Mole fraction of light component (unused, for API consistency).
        T: Temperature [K] (unused, for API consistency).

    Returns:
        Tuple of (1.0, 1.0) activity coefficients.
    """
    return jnp.ones_like(x), jnp.ones_like(x)


def create_nrtl_params(
    alpha: float, A_12: float, B_12: float, A_21: float, B_21: float
) -> NRTLParams:
    """Create NRTL parameters from scalar values.

    Args:
        alpha: Non-randomness parameter.
        A_12: tau_12 intercept.
        B_12: tau_12 temperature coefficient [K].
        A_21: tau_21 intercept.
        B_21: tau_21 temperature coefficient [K].

    Returns:
        NRTLParams dataclass.
    """
    return NRTLParams(
        alpha=jnp.array(alpha),
        A_12=jnp.array(A_12),
        B_12=jnp.array(B_12),
        A_21=jnp.array(A_21),
        B_21=jnp.array(B_21),
    )


# Default NRTL parameters for methanol-water from IUPAC
NRTL_METHANOL_WATER = create_nrtl_params(
    alpha=0.1,
    A_12=9.23811,
    B_12=-2432.61,
    A_21=-5.70743,
    B_21=1538.74,
)


# =============================================================================
# Vapor-Liquid Equilibrium
# =============================================================================


def compute_k_values(
    x: jnp.ndarray,
    T: jnp.ndarray,
    P: jnp.ndarray,
    thermo: ThermoParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute K-values (equilibrium ratios) for both components.

    K_k = gamma_k * P_k^sat / P

    Args:
        x: Liquid mole fraction of light component.
        T: Temperature [K].
        P: Pressure [bar].
        thermo: Thermodynamic parameters.

    Returns:
        Tuple of (K_1, K_2) equilibrium ratios.
    """
    # Vapor pressures
    P_sat_1 = antoine_vapor_pressure(T, thermo.antoine_1)
    P_sat_2 = antoine_vapor_pressure(T, thermo.antoine_2)

    # Activity coefficients
    if thermo.nrtl is not None:
        gamma_1, gamma_2 = nrtl_activity_coefficients(x, T, thermo.nrtl)
    else:
        gamma_1, gamma_2 = ideal_activity_coefficients(x, T)

    # K-values
    K_1 = gamma_1 * P_sat_1 / P
    K_2 = gamma_2 * P_sat_2 / P

    return K_1, K_2


def equilibrium_vapor_composition(
    x: jnp.ndarray,
    T: jnp.ndarray,
    P: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Compute equilibrium vapor composition for binary mixture.

    y_1* = K_1 * x / (K_1 * x + K_2 * (1 - x))

    Args:
        x: Liquid mole fraction of light component.
        T: Temperature [K].
        P: Pressure [bar].
        thermo: Thermodynamic parameters.

    Returns:
        Equilibrium vapor mole fraction of light component.
    """
    K_1, K_2 = compute_k_values(x, T, P, thermo)

    # Avoid division by zero
    denom = K_1 * x + K_2 * (1.0 - x)
    denom = jnp.maximum(denom, 1e-10)

    y_star = K_1 * x / denom

    # Clamp to valid range
    return jnp.clip(y_star, 0.0, 1.0)


def relative_volatility(
    x: jnp.ndarray,
    T: jnp.ndarray,
    P: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Compute relative volatility alpha = K_1 / K_2.

    Args:
        x: Liquid mole fraction of light component.
        T: Temperature [K].
        P: Pressure [bar].
        thermo: Thermodynamic parameters.

    Returns:
        Relative volatility.
    """
    K_1, K_2 = compute_k_values(x, T, P, thermo)
    return K_1 / jnp.maximum(K_2, 1e-10)


# =============================================================================
# Bubble Point Calculation
# =============================================================================


def bubble_point_residual(
    T: jnp.ndarray,
    x: jnp.ndarray,
    P: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Compute bubble point residual: sum(x_i * gamma_i * P_i^sat) - P.

    At the bubble point, this residual equals zero.

    Args:
        T: Temperature [K].
        x: Liquid mole fraction of light component.
        P: Pressure [bar].
        thermo: Thermodynamic parameters.

    Returns:
        Residual (should be zero at bubble point).
    """
    # Vapor pressures
    P_sat_1 = antoine_vapor_pressure(T, thermo.antoine_1)
    P_sat_2 = antoine_vapor_pressure(T, thermo.antoine_2)

    # Activity coefficients
    if thermo.nrtl is not None:
        gamma_1, gamma_2 = nrtl_activity_coefficients(x, T, thermo.nrtl)
    else:
        gamma_1, gamma_2 = ideal_activity_coefficients(x, T)

    # Bubble point condition: P = sum(x_i * gamma_i * P_i^sat)
    P_calc = x * gamma_1 * P_sat_1 + (1.0 - x) * gamma_2 * P_sat_2

    return P_calc - P


def bubble_point_temperature(
    x: jnp.ndarray,
    P: jnp.ndarray,
    thermo: ThermoParams,
    T_init: jnp.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Calculate bubble point temperature using Newton's method with bracketing.

    Algorithm:
    1. Bracket the solution between T_low and T_high
    2. Use a few bisection steps to narrow the bracket
    3. Switch to Newton's method with finite-difference derivative

    Args:
        x: Liquid mole fraction of light component.
        P: Pressure [bar].
        thermo: Thermodynamic parameters.
        T_init: Initial temperature guess [K]. If None, uses midpoint of search range.
        max_iter: Maximum iterations.
        tol: Convergence tolerance [K].

    Returns:
        Bubble point temperature [K].
    """
    # Use wider temperature bounds for search (Antoine will clamp internally)
    # This allows finding bubble points outside strict Antoine validity ranges
    T_low = jnp.minimum(thermo.antoine_1.T_min, thermo.antoine_2.T_min) - 20.0
    T_high = jnp.maximum(thermo.antoine_1.T_max, thermo.antoine_2.T_max) + 20.0

    # Ensure reasonable physical bounds
    T_low = jnp.maximum(T_low, 250.0)  # Below 250K is unrealistic for these mixtures
    T_high = jnp.minimum(T_high, 450.0)  # Above 450K is unrealistic

    # Initial guess
    if T_init is None:
        T = 0.5 * (T_low + T_high)
    else:
        T = T_init

    def cond_fn(state):
        T, iteration, converged = state
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))

    def body_fn(state):
        T, iteration, _ = state

        # Compute residual
        f = bubble_point_residual(T, x, P, thermo)

        # Finite difference derivative
        dT = 0.01  # Small temperature perturbation
        f_plus = bubble_point_residual(T + dT, x, P, thermo)
        df_dT = (f_plus - f) / dT

        # Newton step with damping and bounds
        df_dT_safe = jnp.where(jnp.abs(df_dT) < 1e-10, 1e-10, df_dT)
        delta_T = -f / df_dT_safe

        # Limit step size
        delta_T = jnp.clip(delta_T, -10.0, 10.0)

        # Update temperature
        T_new = T + delta_T

        # Keep within bounds
        T_new = jnp.clip(T_new, T_low, T_high)

        # Check convergence
        converged = jnp.abs(f) < tol * P

        return T_new, iteration + 1, converged

    # Initial state
    init_state = (T, 0, False)

    # Run iteration
    T_final, _, _ = lax.while_loop(cond_fn, body_fn, init_state)

    return T_final


# =============================================================================
# Enthalpy Calculations
# =============================================================================


def liquid_enthalpy(
    x: jnp.ndarray,
    T: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate molar liquid enthalpy.

    h_L(x, T) = x * cp_L1 * (T - T_ref) + (1 - x) * cp_L2 * (T - T_ref)

    Args:
        x: Liquid mole fraction of light component.
        T: Temperature [K].
        thermo: Thermodynamic parameters.

    Returns:
        Molar liquid enthalpy [J/mol].
    """
    dT = T - thermo.T_ref
    h_L = x * thermo.cp_liquid_1 * dT + (1.0 - x) * thermo.cp_liquid_2 * dT
    return h_L


def vapor_enthalpy(
    y: jnp.ndarray,
    T: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate molar vapor enthalpy.

    h_V(y, T) = y * (hvap_1 + cp_V1 * (T - T_ref))
              + (1 - y) * (hvap_2 + cp_V2 * (T - T_ref))

    Args:
        y: Vapor mole fraction of light component.
        T: Temperature [K].
        thermo: Thermodynamic parameters.

    Returns:
        Molar vapor enthalpy [J/mol].
    """
    dT = T - thermo.T_ref
    h_V = y * (thermo.hvap_1 + thermo.cp_vapor_1 * dT) + (1.0 - y) * (
        thermo.hvap_2 + thermo.cp_vapor_2 * dT
    )
    return h_V


def heat_of_vaporization(
    x: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate mixture heat of vaporization (approximate, at reference).

    Args:
        x: Liquid mole fraction of light component.
        thermo: Thermodynamic parameters.

    Returns:
        Mixture heat of vaporization [J/mol].
    """
    return x * thermo.hvap_1 + (1.0 - x) * thermo.hvap_2


def mixture_molecular_weight(
    x: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate mixture molecular weight.

    Args:
        x: Mole fraction of light component.
        thermo: Thermodynamic parameters.

    Returns:
        Mixture molecular weight [kg/mol].
    """
    return x * thermo.mw_1 + (1.0 - x) * thermo.mw_2


# =============================================================================
# Default Mixture Parameters
# =============================================================================


def create_methanol_water_thermo() -> ThermoParams:
    """Create thermodynamic parameters for methanol-water mixture.

    Returns:
        ThermoParams for methanol (light) - water (heavy) system.
    """
    return ThermoParams(
        antoine_1=ANTOINE_METHANOL,
        antoine_2=ANTOINE_WATER,
        nrtl=NRTL_METHANOL_WATER,
        cp_liquid_1=jnp.array(81.6),  # Methanol liquid Cp [J/mol/K]
        cp_liquid_2=jnp.array(75.3),  # Water liquid Cp [J/mol/K]
        cp_vapor_1=jnp.array(44.1),  # Methanol vapor Cp [J/mol/K]
        cp_vapor_2=jnp.array(33.6),  # Water vapor Cp [J/mol/K]
        hvap_1=jnp.array(35300.0),  # Methanol heat of vaporization [J/mol]
        hvap_2=jnp.array(40660.0),  # Water heat of vaporization [J/mol]
        mw_1=jnp.array(0.03204),  # Methanol MW [kg/mol]
        mw_2=jnp.array(0.01802),  # Water MW [kg/mol]
        T_ref=jnp.array(298.15),  # Reference temperature [K]
    )


def create_ethanol_water_thermo() -> ThermoParams:
    """Create thermodynamic parameters for ethanol-water mixture.

    Note: Uses ideal activity coefficients as placeholder.
    For accurate results, add NRTL parameters for ethanol-water.

    Returns:
        ThermoParams for ethanol (light) - water (heavy) system.
    """
    return ThermoParams(
        antoine_1=ANTOINE_ETHANOL,
        antoine_2=ANTOINE_WATER,
        nrtl=None,  # Ideal for now; add NRTL params for accuracy
        cp_liquid_1=jnp.array(112.3),  # Ethanol liquid Cp [J/mol/K]
        cp_liquid_2=jnp.array(75.3),  # Water liquid Cp [J/mol/K]
        cp_vapor_1=jnp.array(65.6),  # Ethanol vapor Cp [J/mol/K]
        cp_vapor_2=jnp.array(33.6),  # Water vapor Cp [J/mol/K]
        hvap_1=jnp.array(38600.0),  # Ethanol heat of vaporization [J/mol]
        hvap_2=jnp.array(40660.0),  # Water heat of vaporization [J/mol]
        mw_1=jnp.array(0.04607),  # Ethanol MW [kg/mol]
        mw_2=jnp.array(0.01802),  # Water MW [kg/mol]
        T_ref=jnp.array(298.15),  # Reference temperature [K]
    )


def create_benzene_toluene_thermo() -> ThermoParams:
    """Create thermodynamic parameters for benzene-toluene mixture.

    This is a nearly ideal mixture, so uses ideal activity coefficients.

    Returns:
        ThermoParams for benzene (light) - toluene (heavy) system.
    """
    return ThermoParams(
        antoine_1=ANTOINE_BENZENE,
        antoine_2=ANTOINE_TOLUENE,
        nrtl=None,  # Nearly ideal mixture
        cp_liquid_1=jnp.array(136.0),  # Benzene liquid Cp [J/mol/K]
        cp_liquid_2=jnp.array(157.0),  # Toluene liquid Cp [J/mol/K]
        cp_vapor_1=jnp.array(82.4),  # Benzene vapor Cp [J/mol/K]
        cp_vapor_2=jnp.array(104.0),  # Toluene vapor Cp [J/mol/K]
        hvap_1=jnp.array(30720.0),  # Benzene heat of vaporization [J/mol]
        hvap_2=jnp.array(33180.0),  # Toluene heat of vaporization [J/mol]
        mw_1=jnp.array(0.07811),  # Benzene MW [kg/mol]
        mw_2=jnp.array(0.09214),  # Toluene MW [kg/mol]
        T_ref=jnp.array(298.15),  # Reference temperature [K]
    )
