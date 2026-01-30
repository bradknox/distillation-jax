"""Reboiler model for distillation simulation.

This module implements a kettle-type reboiler with:
- Material balances (total and component)
- Energy balance with heat input
- VLE for vapor generation

All functions are pure and compatible with JAX transformations.
"""

import jax.numpy as jnp
from typing import NamedTuple

from jax_distillation.core.types import ThermoParams
from jax_distillation.core.thermodynamics import (
    equilibrium_vapor_composition,
    bubble_point_temperature,
    liquid_enthalpy,
    vapor_enthalpy,
)


class ReboilerState(NamedTuple):
    """State of the reboiler.

    Attributes:
        M: Liquid holdup [mol].
        x: Liquid composition [mol fraction of light component].
        T: Temperature [K].
    """

    M: jnp.ndarray
    x: jnp.ndarray
    T: jnp.ndarray


class ReboilerInputs(NamedTuple):
    """Inputs to the reboiler.

    Attributes:
        L_in: Liquid inflow from bottom tray [mol/s].
        x_in: Liquid composition of inflow [mol fraction].
        T_in: Temperature of liquid inflow [K].
        Q_R: Reboiler duty (heat input) [W].
        B_setpoint: Bottoms flow setpoint [mol/s] (for level control).
    """

    L_in: jnp.ndarray
    x_in: jnp.ndarray
    T_in: jnp.ndarray
    Q_R: jnp.ndarray
    B_setpoint: jnp.ndarray


class ReboilerOutputs(NamedTuple):
    """Outputs from the reboiler.

    Attributes:
        V_out: Vapor flow to bottom tray [mol/s].
        y_out: Vapor composition [mol fraction].
        T_V_out: Vapor temperature [K].
        B: Bottoms flow [mol/s].
        x_B: Bottoms composition [mol fraction].
        T_B: Bottoms temperature [K].
    """

    V_out: jnp.ndarray
    y_out: jnp.ndarray
    T_V_out: jnp.ndarray
    B: jnp.ndarray
    x_B: jnp.ndarray
    T_B: jnp.ndarray


# =============================================================================
# Reboiler Dynamics
# =============================================================================


def reboiler_vapor_generation(
    Q_R: jnp.ndarray,
    x: jnp.ndarray,
    T: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate vapor generation rate from heat input.

    V = Q_R / delta_H_vap

    Args:
        Q_R: Reboiler duty [W].
        x: Liquid composition [mol fraction].
        T: Temperature [K].
        thermo: Thermodynamic parameters.

    Returns:
        Vapor generation rate [mol/s].
    """
    # Compute enthalpy difference for vaporization
    # Approximate: use heat of vaporization at current composition
    y_eq = equilibrium_vapor_composition(x, T, jnp.array(1.0), thermo)

    h_L = liquid_enthalpy(x, T, thermo)
    h_V = vapor_enthalpy(y_eq, T, thermo)

    delta_H = h_V - h_L

    # Ensure positive enthalpy difference
    delta_H = jnp.maximum(delta_H, 1000.0)  # Minimum 1 kJ/mol

    # Vapor generation rate
    V = Q_R / delta_H

    return jnp.maximum(V, 0.0)


def reboiler_level_control(
    M: jnp.ndarray,
    M_setpoint: jnp.ndarray,
    B_setpoint: jnp.ndarray,
    Kp: jnp.ndarray = jnp.array(0.1),
) -> jnp.ndarray:
    """Simple proportional level control for bottoms flow.

    B = B_setpoint + Kp * (M - M_setpoint)

    Args:
        M: Current liquid holdup [mol].
        M_setpoint: Setpoint holdup [mol].
        B_setpoint: Nominal bottoms flow [mol/s].
        Kp: Proportional gain [1/s].

    Returns:
        Bottoms flow rate [mol/s].
    """
    error = M - M_setpoint
    B = B_setpoint + Kp * error
    return jnp.maximum(B, 0.0)


def reboiler_material_balance(
    L_in: jnp.ndarray,
    V_out: jnp.ndarray,
    B: jnp.ndarray,
) -> jnp.ndarray:
    """Compute rate of change of reboiler holdup.

    dM/dt = L_in - V_out - B

    Args:
        L_in: Liquid inflow [mol/s].
        V_out: Vapor outflow [mol/s].
        B: Bottoms flow [mol/s].

    Returns:
        Rate of change of holdup [mol/s].
    """
    return L_in - V_out - B


def reboiler_component_balance(
    L_in: jnp.ndarray,
    x_in: jnp.ndarray,
    V_out: jnp.ndarray,
    y_out: jnp.ndarray,
    B: jnp.ndarray,
    x_B: jnp.ndarray,
) -> jnp.ndarray:
    """Compute rate of change of component holdup in reboiler.

    d(M*x)/dt = L_in*x_in - V_out*y_out - B*x_B

    Args:
        L_in: Liquid inflow [mol/s].
        x_in: Liquid composition of inflow [mol fraction].
        V_out: Vapor outflow [mol/s].
        y_out: Vapor composition [mol fraction].
        B: Bottoms flow [mol/s].
        x_B: Bottoms composition [mol fraction].

    Returns:
        Rate of change of component holdup [mol/s].
    """
    return L_in * x_in - V_out * y_out - B * x_B


def reboiler_energy_balance(
    L_in: jnp.ndarray,
    x_in: jnp.ndarray,
    T_in: jnp.ndarray,
    V_out: jnp.ndarray,
    y_out: jnp.ndarray,
    T_V: jnp.ndarray,
    B: jnp.ndarray,
    x_B: jnp.ndarray,
    T_B: jnp.ndarray,
    Q_R: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Compute rate of change of internal energy in reboiler.

    dU/dt = L_in*h_L(x_in,T_in) - V_out*h_V(y_out,T_V) - B*h_L(x_B,T_B) + Q_R

    Args:
        L_in, x_in, T_in: Liquid inflow properties.
        V_out, y_out, T_V: Vapor outflow properties.
        B, x_B, T_B: Bottoms properties.
        Q_R: Reboiler duty [W].
        thermo: Thermodynamic parameters.

    Returns:
        Rate of change of internal energy [W].
    """
    H_in = L_in * liquid_enthalpy(x_in, T_in, thermo)
    H_V_out = V_out * vapor_enthalpy(y_out, T_V, thermo)
    H_B = B * liquid_enthalpy(x_B, T_B, thermo)

    return H_in - H_V_out - H_B + Q_R


# =============================================================================
# Complete Reboiler Step
# =============================================================================


def reboiler_step(
    state: ReboilerState,
    inputs: ReboilerInputs,
    P: jnp.ndarray,
    M_setpoint: jnp.ndarray,
    thermo: ThermoParams,
    dt: jnp.ndarray,
    Kp_level: jnp.ndarray = jnp.array(0.1),
) -> tuple[ReboilerState, ReboilerOutputs]:
    """Complete reboiler timestep.

    Args:
        state: Current reboiler state.
        inputs: Inputs to the reboiler.
        P: Operating pressure [bar].
        M_setpoint: Holdup setpoint for level control [mol].
        thermo: Thermodynamic parameters.
        dt: Time step [s].
        Kp_level: Proportional gain for level control [1/s].

    Returns:
        Tuple of (new_state, outputs).
    """
    M, x, T = state.M, state.x, state.T

    # Compute vapor generation from heat input
    V_out = reboiler_vapor_generation(inputs.Q_R, x, T, thermo)

    # Compute vapor composition (equilibrium)
    y_out = equilibrium_vapor_composition(x, T, P, thermo)

    # Vapor temperature (same as reboiler liquid)
    T_V_out = T

    # Compute bottoms flow from level control
    B = reboiler_level_control(M, M_setpoint, inputs.B_setpoint, Kp_level)

    # Bottoms properties (same as reboiler liquid)
    x_B = x
    T_B = T

    # --- Physical vapor limiting ---
    # Cannot vaporize more liquid than exists in the reboiler.
    # Constraint: M + dt*(L_in - V_out - B) >= M_min
    # => V_out <= (M - M_min)/dt + L_in - B
    M_min = jnp.array(0.01)  # Minimum physical reboiler holdup [mol]
    V_max = jnp.where(
        dt > 0.0,
        jnp.maximum(
            (M - M_min) / jnp.maximum(dt, 1e-10) + inputs.L_in - B,
            0.0,
        ),
        jnp.array(jnp.inf),  # No limit for dt=0 output-only queries
    )
    V_out = jnp.minimum(V_out, V_max)

    # Material balance
    dM_dt = reboiler_material_balance(inputs.L_in, V_out, B)

    # Component balance
    dMx_dt = reboiler_component_balance(inputs.L_in, inputs.x_in, V_out, y_out, B, x_B)

    # Integrate state
    M_new = M + dt * dM_dt
    M_new = jnp.maximum(M_new, 1e-6)

    Mx_new = M * x + dt * dMx_dt
    x_new = Mx_new / M_new
    x_new = jnp.clip(x_new, 0.0, 1.0)

    # Update temperature from bubble point
    T_new = bubble_point_temperature(x_new, P, thermo, T_init=T)

    new_state = ReboilerState(M=M_new, x=x_new, T=T_new)

    outputs = ReboilerOutputs(
        V_out=V_out,
        y_out=y_out,
        T_V_out=T_V_out,
        B=B,
        x_B=x_B,
        T_B=T_B,
    )

    return new_state, outputs


# =============================================================================
# Utility Functions
# =============================================================================


def create_initial_reboiler_state(
    M: float,
    x: float,
    T: float,
) -> ReboilerState:
    """Create an initial reboiler state.

    Args:
        M: Initial liquid holdup [mol].
        x: Initial liquid composition [mol fraction].
        T: Initial temperature [K].

    Returns:
        ReboilerState namedtuple.
    """
    return ReboilerState(
        M=jnp.array(M),
        x=jnp.array(x),
        T=jnp.array(T),
    )


def create_reboiler_inputs(
    L_in: float,
    x_in: float,
    T_in: float,
    Q_R: float,
    B_setpoint: float,
) -> ReboilerInputs:
    """Create reboiler inputs namedtuple.

    Args:
        L_in: Liquid inflow [mol/s].
        x_in: Liquid composition of inflow [mol fraction].
        T_in: Temperature of liquid inflow [K].
        Q_R: Reboiler duty [W].
        B_setpoint: Bottoms flow setpoint [mol/s].

    Returns:
        ReboilerInputs namedtuple.
    """
    return ReboilerInputs(
        L_in=jnp.array(L_in),
        x_in=jnp.array(x_in),
        T_in=jnp.array(T_in),
        Q_R=jnp.array(Q_R),
        B_setpoint=jnp.array(B_setpoint),
    )
