"""Single tray dynamics for distillation simulation.

This module implements the MESH equations for a single distillation tray:
- Material balance (total and component)
- Equilibrium (VLE with Murphree efficiency)
- Summation (implicit for binary)
- Heat (energy balance)

All functions are pure and compatible with JAX transformations.
"""

import jax.numpy as jnp
from typing import NamedTuple

from jax_distillation.core.types import ThermoParams, HydraulicParams
from jax_distillation.core.thermodynamics import (
    equilibrium_vapor_composition,
    bubble_point_temperature,
    liquid_enthalpy,
    vapor_enthalpy,
    mixture_molecular_weight,
)
from jax_distillation.core.hydraulics import (
    liquid_density,
    vapor_density,
    static_liquid_outflow,
    update_liquid_outflow,
)


class TrayInflows(NamedTuple):
    """Inflows to a single tray.

    Attributes:
        L_in: Liquid inflow from tray above [mol/s].
        x_in: Liquid composition of inflow [mol fraction].
        T_L_in: Temperature of liquid inflow [K].
        V_in: Vapor inflow from tray below [mol/s].
        y_in: Vapor composition of inflow [mol fraction].
        T_V_in: Temperature of vapor inflow [K].
        F_L: Liquid feed to this tray [mol/s] (0 if not feed tray).
        F_V: Vapor feed to this tray [mol/s] (0 if not feed tray).
        z_F: Feed composition [mol fraction].
        T_F: Feed temperature [K].
    """

    L_in: jnp.ndarray
    x_in: jnp.ndarray
    T_L_in: jnp.ndarray
    V_in: jnp.ndarray
    y_in: jnp.ndarray
    T_V_in: jnp.ndarray
    F_L: jnp.ndarray
    F_V: jnp.ndarray
    z_F: jnp.ndarray
    T_F: jnp.ndarray


class TrayState(NamedTuple):
    """State of a single tray.

    Attributes:
        M: Liquid holdup [mol].
        x: Liquid composition [mol fraction of light component].
        T: Temperature [K].
        L_out: Dynamic liquid outflow [mol/s].
    """

    M: jnp.ndarray
    x: jnp.ndarray
    T: jnp.ndarray
    L_out: jnp.ndarray


class TrayOutputs(NamedTuple):
    """Outputs from a single tray.

    Attributes:
        L_out: Liquid outflow [mol/s].
        x_out: Liquid composition of outflow [mol fraction].
        T_L_out: Temperature of liquid outflow [K].
        V_out: Vapor outflow [mol/s].
        y_out: Vapor composition of outflow [mol fraction].
        T_V_out: Temperature of vapor outflow [K].
    """

    L_out: jnp.ndarray
    x_out: jnp.ndarray
    T_L_out: jnp.ndarray
    V_out: jnp.ndarray
    y_out: jnp.ndarray
    T_V_out: jnp.ndarray


class TrayDerivatives(NamedTuple):
    """Time derivatives for tray state.

    Attributes:
        dM_dt: Rate of change of holdup [mol/s].
        dMx_dt: Rate of change of component holdup [mol/s].
        dU_dt: Rate of change of internal energy [J/s].
    """

    dM_dt: jnp.ndarray
    dMx_dt: jnp.ndarray
    dU_dt: jnp.ndarray


# =============================================================================
# Equilibrium Calculations
# =============================================================================


def murphree_vapor_efficiency(
    y_in: jnp.ndarray,
    y_star: jnp.ndarray,
    E_M: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Murphree vapor efficiency correction.

    y_out = y_in + E_M * (y_star - y_in)

    Args:
        y_in: Vapor composition entering tray from below [mol fraction].
        y_star: Equilibrium vapor composition [mol fraction].
        E_M: Murphree vapor efficiency (0 to 1).

    Returns:
        Actual vapor composition leaving tray [mol fraction].
    """
    y_out = y_in + E_M * (y_star - y_in)
    return jnp.clip(y_out, 0.0, 1.0)


def compute_tray_equilibrium(
    x: jnp.ndarray,
    T: jnp.ndarray,
    P: jnp.ndarray,
    y_in: jnp.ndarray,
    E_M: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Compute vapor composition leaving tray with Murphree efficiency.

    Args:
        x: Liquid composition on tray [mol fraction].
        T: Tray temperature [K].
        P: Pressure [bar].
        y_in: Vapor composition entering from below [mol fraction].
        E_M: Murphree vapor efficiency.
        thermo: Thermodynamic parameters.

    Returns:
        Vapor composition leaving tray [mol fraction].
    """
    # Equilibrium vapor composition
    y_star = equilibrium_vapor_composition(x, T, P, thermo)

    # Apply Murphree efficiency
    y_out = murphree_vapor_efficiency(y_in, y_star, E_M)

    return y_out


# =============================================================================
# Material Balances
# =============================================================================


def total_material_balance(
    L_in: jnp.ndarray,
    L_out: jnp.ndarray,
    V_in: jnp.ndarray,
    V_out: jnp.ndarray,
    F_L: jnp.ndarray,
    F_V: jnp.ndarray,
) -> jnp.ndarray:
    """Compute rate of change of total holdup.

    dM/dt = L_in + V_in + F_L + F_V - L_out - V_out

    Args:
        L_in: Liquid inflow [mol/s].
        L_out: Liquid outflow [mol/s].
        V_in: Vapor inflow [mol/s].
        V_out: Vapor outflow [mol/s].
        F_L: Liquid feed [mol/s].
        F_V: Vapor feed [mol/s].

    Returns:
        Rate of change of holdup [mol/s].
    """
    return L_in + V_in + F_L + F_V - L_out - V_out


def component_material_balance(
    L_in: jnp.ndarray,
    x_in: jnp.ndarray,
    L_out: jnp.ndarray,
    x_out: jnp.ndarray,
    V_in: jnp.ndarray,
    y_in: jnp.ndarray,
    V_out: jnp.ndarray,
    y_out: jnp.ndarray,
    F_L: jnp.ndarray,
    F_V: jnp.ndarray,
    z_F: jnp.ndarray,
) -> jnp.ndarray:
    """Compute rate of change of component holdup.

    d(M*x)/dt = L_in*x_in + V_in*y_in + F_L*z_F + F_V*z_F - L_out*x_out - V_out*y_out

    Args:
        L_in: Liquid inflow [mol/s].
        x_in: Liquid composition of inflow [mol fraction].
        L_out: Liquid outflow [mol/s].
        x_out: Liquid composition of outflow [mol fraction].
        V_in: Vapor inflow [mol/s].
        y_in: Vapor composition of inflow [mol fraction].
        V_out: Vapor outflow [mol/s].
        y_out: Vapor composition of outflow [mol fraction].
        F_L: Liquid feed [mol/s].
        F_V: Vapor feed [mol/s].
        z_F: Feed composition [mol fraction].

    Returns:
        Rate of change of component holdup [mol/s].
    """
    return (
        L_in * x_in
        + V_in * y_in
        + F_L * z_F
        + F_V * z_F
        - L_out * x_out
        - V_out * y_out
    )


# =============================================================================
# Energy Balance
# =============================================================================


def energy_balance(
    L_in: jnp.ndarray,
    x_in: jnp.ndarray,
    T_L_in: jnp.ndarray,
    L_out: jnp.ndarray,
    x_out: jnp.ndarray,
    T_L_out: jnp.ndarray,
    V_in: jnp.ndarray,
    y_in: jnp.ndarray,
    T_V_in: jnp.ndarray,
    V_out: jnp.ndarray,
    y_out: jnp.ndarray,
    T_V_out: jnp.ndarray,
    F_L: jnp.ndarray,
    F_V: jnp.ndarray,
    z_F: jnp.ndarray,
    T_F: jnp.ndarray,
    Q: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Compute rate of change of internal energy.

    dU/dt = sum(inlet enthalpy flows) - sum(outlet enthalpy flows) + Q

    Args:
        L_in, x_in, T_L_in: Liquid inflow properties.
        L_out, x_out, T_L_out: Liquid outflow properties.
        V_in, y_in, T_V_in: Vapor inflow properties.
        V_out, y_out, T_V_out: Vapor outflow properties.
        F_L, F_V, z_F, T_F: Feed properties.
        Q: External heat input [W].
        thermo: Thermodynamic parameters.

    Returns:
        Rate of change of internal energy [J/s = W].
    """
    # Inlet enthalpy flows
    H_L_in = L_in * liquid_enthalpy(x_in, T_L_in, thermo)
    H_V_in = V_in * vapor_enthalpy(y_in, T_V_in, thermo)
    H_F_L = F_L * liquid_enthalpy(z_F, T_F, thermo)
    H_F_V = F_V * vapor_enthalpy(z_F, T_F, thermo)

    # Outlet enthalpy flows
    H_L_out = L_out * liquid_enthalpy(x_out, T_L_out, thermo)
    H_V_out = V_out * vapor_enthalpy(y_out, T_V_out, thermo)

    # Energy balance
    dU_dt = H_L_in + H_V_in + H_F_L + H_F_V - H_L_out - H_V_out + Q

    return dU_dt


# =============================================================================
# Vapor Flow Calculation
# =============================================================================


def compute_vapor_outflow(
    V_in: jnp.ndarray,
    F_V: jnp.ndarray,
    dM_dt: jnp.ndarray,
    L_in: jnp.ndarray,
    L_out: jnp.ndarray,
    F_L: jnp.ndarray,
) -> jnp.ndarray:
    """Compute vapor outflow from mass balance (assuming negligible vapor holdup).

    From total balance: dM/dt = L_in + V_in + F_L + F_V - L_out - V_out
    Rearranging: V_out = L_in + V_in + F_L + F_V - L_out - dM/dt

    For quasi-steady vapor flow (common assumption):
    V_out ≈ V_in + F_V (if dM/dt accounts for liquid only)

    Args:
        V_in: Vapor inflow [mol/s].
        F_V: Vapor feed [mol/s].
        dM_dt: Rate of liquid holdup change [mol/s].
        L_in: Liquid inflow [mol/s].
        L_out: Liquid outflow [mol/s].
        F_L: Liquid feed [mol/s].

    Returns:
        Vapor outflow [mol/s].
    """
    # From mass balance (liquid holdup only, vapor holdup negligible)
    V_out = L_in + V_in + F_L + F_V - L_out - dM_dt

    # Ensure non-negative
    return jnp.maximum(V_out, 0.0)


# =============================================================================
# Complete Tray Step
# =============================================================================


def compute_tray_outputs(
    state: TrayState,
    inflows: TrayInflows,
    V_in_prev: jnp.ndarray,
    P: jnp.ndarray,
    E_M: jnp.ndarray,
    thermo: ThermoParams,
    hydraulics: HydraulicParams,
    dt: jnp.ndarray,
) -> tuple[TrayOutputs, TrayDerivatives, TrayState]:
    """Compute tray outputs and derivatives for one timestep.

    Args:
        state: Current tray state.
        inflows: Inflows to the tray.
        V_in_prev: Previous vapor inflow (for hydraulic coupling).
        P: Operating pressure [bar].
        E_M: Murphree vapor efficiency.
        thermo: Thermodynamic parameters.
        hydraulics: Hydraulic parameters.
        dt: Time step [s].

    Returns:
        Tuple of (outputs, derivatives, new_state).
    """
    M, x, T, L_out_dyn = state.M, state.x, state.T, state.L_out

    # Compute liquid density and molecular weight
    rho_L = liquid_density(x, T, thermo)
    mw = mixture_molecular_weight(x, thermo)

    # Compute static liquid outflow from weir relation
    L_static = static_liquid_outflow(M, hydraulics, rho_L, mw)

    # Update dynamic liquid outflow with hydraulic coupling
    L_out_new = update_liquid_outflow(
        L_out_dyn,
        L_static,
        inflows.V_in,
        V_in_prev,
        hydraulics.tau_L,
        hydraulics.j,
        dt,
    )

    # Liquid outflow properties (same as tray liquid)
    x_out = x
    T_L_out = T

    # Compute equilibrium vapor composition with Murphree efficiency
    y_out = compute_tray_equilibrium(x, T, P, inflows.y_in, E_M, thermo)

    # Vapor outflow temperature (assume thermal equilibrium with liquid)
    T_V_out = T

    # Total material balance derivative
    # For now, assume V_out ≈ V_in + F_V (quasi-steady vapor)
    V_out = inflows.V_in + inflows.F_V

    dM_dt = total_material_balance(
        inflows.L_in,
        L_out_new,
        inflows.V_in,
        V_out,
        inflows.F_L,
        inflows.F_V,
    )

    # Component material balance derivative
    dMx_dt = component_material_balance(
        inflows.L_in,
        inflows.x_in,
        L_out_new,
        x_out,
        inflows.V_in,
        inflows.y_in,
        V_out,
        y_out,
        inflows.F_L,
        inflows.F_V,
        inflows.z_F,
    )

    # Energy balance derivative (no external heat for intermediate trays)
    Q = jnp.array(0.0)
    dU_dt = energy_balance(
        inflows.L_in,
        inflows.x_in,
        inflows.T_L_in,
        L_out_new,
        x_out,
        T_L_out,
        inflows.V_in,
        inflows.y_in,
        inflows.T_V_in,
        V_out,
        y_out,
        T_V_out,
        inflows.F_L,
        inflows.F_V,
        inflows.z_F,
        inflows.T_F,
        Q,
        thermo,
    )

    outputs = TrayOutputs(
        L_out=L_out_new,
        x_out=x_out,
        T_L_out=T_L_out,
        V_out=V_out,
        y_out=y_out,
        T_V_out=T_V_out,
    )

    derivatives = TrayDerivatives(dM_dt=dM_dt, dMx_dt=dMx_dt, dU_dt=dU_dt)

    new_state = TrayState(M=M, x=x, T=T, L_out=L_out_new)

    return outputs, derivatives, new_state


def integrate_tray_state(
    state: TrayState,
    derivatives: TrayDerivatives,
    P: jnp.ndarray,
    thermo: ThermoParams,
    dt: jnp.ndarray,
) -> TrayState:
    """Integrate tray state forward by one timestep using Euler method.

    Args:
        state: Current tray state.
        derivatives: Time derivatives.
        P: Operating pressure [bar].
        thermo: Thermodynamic parameters.
        dt: Time step [s].

    Returns:
        Updated tray state.
    """
    M, x, T, L_out = state.M, state.x, state.T, state.L_out

    # Update holdup
    M_new = M + dt * derivatives.dM_dt
    M_new = jnp.maximum(M_new, 1e-6)  # Ensure positive holdup

    # Update composition from component balance
    # d(Mx)/dt = dMx_dt => x_new = (M*x + dt*dMx_dt) / M_new
    Mx_new = M * x + dt * derivatives.dMx_dt
    x_new = Mx_new / M_new
    x_new = jnp.clip(x_new, 0.0, 1.0)

    # Update temperature from bubble point (for liquid phase)
    # This ensures thermodynamic consistency
    T_new = bubble_point_temperature(x_new, P, thermo, T_init=T)

    return TrayState(M=M_new, x=x_new, T=T_new, L_out=L_out)


def tray_step(
    state: TrayState,
    inflows: TrayInflows,
    V_in_prev: jnp.ndarray,
    P: jnp.ndarray,
    E_M: jnp.ndarray,
    thermo: ThermoParams,
    hydraulics: HydraulicParams,
    dt: jnp.ndarray,
) -> tuple[TrayState, TrayOutputs]:
    """Complete single tray timestep.

    This is the main function for simulating a single tray.

    Args:
        state: Current tray state.
        inflows: Inflows to the tray.
        V_in_prev: Previous vapor inflow (for hydraulic coupling).
        P: Operating pressure [bar].
        E_M: Murphree vapor efficiency.
        thermo: Thermodynamic parameters.
        hydraulics: Hydraulic parameters.
        dt: Time step [s].

    Returns:
        Tuple of (new_state, outputs).
    """
    # Compute outputs and derivatives
    outputs, derivatives, state_with_L = compute_tray_outputs(
        state, inflows, V_in_prev, P, E_M, thermo, hydraulics, dt
    )

    # Integrate state forward
    new_state = integrate_tray_state(state_with_L, derivatives, P, thermo, dt)

    # Keep the updated L_out from compute_tray_outputs
    new_state = TrayState(
        M=new_state.M, x=new_state.x, T=new_state.T, L_out=state_with_L.L_out
    )

    return new_state, outputs


# =============================================================================
# Utility Functions
# =============================================================================


def create_initial_tray_state(
    M: float,
    x: float,
    T: float,
    L_out: float,
) -> TrayState:
    """Create an initial tray state.

    Args:
        M: Initial liquid holdup [mol].
        x: Initial liquid composition [mol fraction].
        T: Initial temperature [K].
        L_out: Initial liquid outflow [mol/s].

    Returns:
        TrayState namedtuple.
    """
    return TrayState(
        M=jnp.array(M),
        x=jnp.array(x),
        T=jnp.array(T),
        L_out=jnp.array(L_out),
    )


def create_tray_inflows(
    L_in: float,
    x_in: float,
    T_L_in: float,
    V_in: float,
    y_in: float,
    T_V_in: float,
    F_L: float = 0.0,
    F_V: float = 0.0,
    z_F: float = 0.5,
    T_F: float = 350.0,
) -> TrayInflows:
    """Create tray inflows namedtuple.

    Args:
        L_in: Liquid inflow [mol/s].
        x_in: Liquid composition of inflow [mol fraction].
        T_L_in: Temperature of liquid inflow [K].
        V_in: Vapor inflow [mol/s].
        y_in: Vapor composition of inflow [mol fraction].
        T_V_in: Temperature of vapor inflow [K].
        F_L: Liquid feed [mol/s].
        F_V: Vapor feed [mol/s].
        z_F: Feed composition [mol fraction].
        T_F: Feed temperature [K].

    Returns:
        TrayInflows namedtuple.
    """
    return TrayInflows(
        L_in=jnp.array(L_in),
        x_in=jnp.array(x_in),
        T_L_in=jnp.array(T_L_in),
        V_in=jnp.array(V_in),
        y_in=jnp.array(y_in),
        T_V_in=jnp.array(T_V_in),
        F_L=jnp.array(F_L),
        F_V=jnp.array(F_V),
        z_F=jnp.array(z_F),
        T_F=jnp.array(T_F),
    )
