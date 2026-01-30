"""Condenser and reflux drum model for distillation simulation.

This module implements a total condenser with reflux drum:
- Total condensation of overhead vapor
- Reflux drum material balance
- Reflux and distillate flows

All functions are pure and compatible with JAX transformations.
"""

import jax.numpy as jnp
from typing import NamedTuple

from jax_distillation.core.types import ThermoParams
from jax_distillation.core.thermodynamics import (
    bubble_point_temperature,
    liquid_enthalpy,
    vapor_enthalpy,
)


class CondenserState(NamedTuple):
    """State of the condenser/reflux drum.

    Attributes:
        M: Liquid holdup in reflux drum [mol].
        x: Liquid composition [mol fraction of light component].
        T: Temperature [K].
    """

    M: jnp.ndarray
    x: jnp.ndarray
    T: jnp.ndarray


class CondenserInputs(NamedTuple):
    """Inputs to the condenser.

    Attributes:
        V_in: Vapor inflow from top tray [mol/s].
        y_in: Vapor composition of inflow [mol fraction].
        T_V_in: Temperature of vapor inflow [K].
        reflux_ratio: Reflux ratio setpoint (R/D).
        D_setpoint: Distillate flow setpoint [mol/s] (for level control).
    """

    V_in: jnp.ndarray
    y_in: jnp.ndarray
    T_V_in: jnp.ndarray
    reflux_ratio: jnp.ndarray
    D_setpoint: jnp.ndarray


class CondenserOutputs(NamedTuple):
    """Outputs from the condenser.

    Attributes:
        R: Reflux flow [mol/s].
        x_R: Reflux composition [mol fraction].
        T_R: Reflux temperature [K].
        D: Distillate flow [mol/s].
        x_D: Distillate composition [mol fraction].
        T_D: Distillate temperature [K].
        Q_C: Condenser duty (heat removed) [W].
    """

    R: jnp.ndarray
    x_R: jnp.ndarray
    T_R: jnp.ndarray
    D: jnp.ndarray
    x_D: jnp.ndarray
    T_D: jnp.ndarray
    Q_C: jnp.ndarray


# =============================================================================
# Condenser Dynamics
# =============================================================================


def condenser_duty(
    V_in: jnp.ndarray,
    y_in: jnp.ndarray,
    T_V_in: jnp.ndarray,
    x_out: jnp.ndarray,
    T_out: jnp.ndarray,
    thermo: ThermoParams,
) -> jnp.ndarray:
    """Calculate condenser duty (heat removal) for total condensation.

    Q_C = V_in * (h_V(y_in, T_V_in) - h_L(x_out, T_out))

    Args:
        V_in: Vapor inflow [mol/s].
        y_in: Vapor composition [mol fraction].
        T_V_in: Vapor temperature [K].
        x_out: Liquid composition (same as vapor for total condenser).
        T_out: Condensate temperature [K].
        thermo: Thermodynamic parameters.

    Returns:
        Condenser duty [W] (positive means heat removed).
    """
    h_V = vapor_enthalpy(y_in, T_V_in, thermo)
    h_L = liquid_enthalpy(x_out, T_out, thermo)

    Q_C = V_in * (h_V - h_L)
    return Q_C


def reflux_drum_level_control(
    M: jnp.ndarray,
    M_setpoint: jnp.ndarray,
    D_setpoint: jnp.ndarray,
    Kp: jnp.ndarray = jnp.array(0.1),
) -> jnp.ndarray:
    """Simple proportional level control for distillate flow.

    D = D_setpoint + Kp * (M - M_setpoint)

    Args:
        M: Current liquid holdup [mol].
        M_setpoint: Setpoint holdup [mol].
        D_setpoint: Nominal distillate flow [mol/s].
        Kp: Proportional gain [1/s].

    Returns:
        Distillate flow rate [mol/s].
    """
    error = M - M_setpoint
    D = D_setpoint + Kp * error
    return jnp.maximum(D, 0.0)


def compute_reflux_and_distillate(
    V_in: jnp.ndarray,
    reflux_ratio: jnp.ndarray,
    M: jnp.ndarray,
    M_setpoint: jnp.ndarray,
    D_setpoint: jnp.ndarray,
    Kp: jnp.ndarray = jnp.array(0.5),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute reflux and distillate flows.

    Uses reflux ratio control: the level controller sets D, then
    R = RR * D. Any V - D - R imbalance is absorbed by the condenser
    drum holdup, which the level controller corrects. At steady state,
    D = V/(1+RR) and R = RR*V/(1+RR).

    Args:
        V_in: Vapor inflow [mol/s].
        reflux_ratio: Reflux ratio (R/D).
        M: Current drum holdup [mol].
        M_setpoint: Setpoint holdup [mol].
        D_setpoint: Nominal distillate flow [mol/s].
        Kp: Level controller proportional gain [1/s].

    Returns:
        Tuple of (R, D) flow rates [mol/s].
    """
    # Level control determines distillate
    D = reflux_drum_level_control(M, M_setpoint, D_setpoint, Kp=Kp)

    # Reflux is set by reflux ratio
    R = reflux_ratio * D
    R = jnp.maximum(R, 0.0)

    return R, D


def condenser_material_balance(
    V_in: jnp.ndarray,
    R: jnp.ndarray,
    D: jnp.ndarray,
) -> jnp.ndarray:
    """Compute rate of change of reflux drum holdup.

    dM/dt = V_in - R - D

    Args:
        V_in: Vapor inflow [mol/s].
        R: Reflux flow [mol/s].
        D: Distillate flow [mol/s].

    Returns:
        Rate of change of holdup [mol/s].
    """
    return V_in - R - D


def condenser_component_balance(
    V_in: jnp.ndarray,
    y_in: jnp.ndarray,
    R: jnp.ndarray,
    x_R: jnp.ndarray,
    D: jnp.ndarray,
    x_D: jnp.ndarray,
) -> jnp.ndarray:
    """Compute rate of change of component holdup in reflux drum.

    d(M*x)/dt = V_in*y_in - R*x_R - D*x_D

    For total condenser: x_R = x_D = y_in (at equilibrium)

    Args:
        V_in: Vapor inflow [mol/s].
        y_in: Vapor composition [mol fraction].
        R: Reflux flow [mol/s].
        x_R: Reflux composition [mol fraction].
        D: Distillate flow [mol/s].
        x_D: Distillate composition [mol fraction].

    Returns:
        Rate of change of component holdup [mol/s].
    """
    return V_in * y_in - R * x_R - D * x_D


# =============================================================================
# Complete Condenser Step
# =============================================================================


def condenser_step(
    state: CondenserState,
    inputs: CondenserInputs,
    P: jnp.ndarray,
    M_setpoint: jnp.ndarray,
    thermo: ThermoParams,
    dt: jnp.ndarray,
    Kp_level: jnp.ndarray = jnp.array(0.5),
) -> tuple[CondenserState, CondenserOutputs]:
    """Complete condenser/reflux drum timestep.

    Args:
        state: Current condenser state.
        inputs: Inputs to the condenser.
        P: Operating pressure [bar].
        M_setpoint: Holdup setpoint for level control [mol].
        thermo: Thermodynamic parameters.
        dt: Time step [s].
        Kp_level: Level controller proportional gain [1/s].

    Returns:
        Tuple of (new_state, outputs).
    """
    M, x, T = state.M, state.x, state.T

    # For total condenser, condensate composition equals vapor composition
    # (assuming well-mixed drum reaches equilibrium)
    # In practice, x evolves based on mixing with drum contents
    x_condensate = inputs.y_in

    # Compute condensate temperature (bubble point at drum composition)
    T_condensate = bubble_point_temperature(x_condensate, P, thermo, T_init=T)

    # Compute reflux and distillate flows
    R, D = compute_reflux_and_distillate(
        inputs.V_in,
        inputs.reflux_ratio,
        M,
        M_setpoint,
        inputs.D_setpoint,
        Kp=Kp_level,
    )

    # Reflux and distillate compositions (from drum)
    x_R = x
    x_D = x
    T_R = T
    T_D = T

    # Compute condenser duty
    Q_C = condenser_duty(inputs.V_in, inputs.y_in, inputs.T_V_in, x, T, thermo)

    # Material balance
    dM_dt = condenser_material_balance(inputs.V_in, R, D)

    # Component balance
    dMx_dt = condenser_component_balance(inputs.V_in, inputs.y_in, R, x_R, D, x_D)

    # Integrate state
    M_new = M + dt * dM_dt
    M_new = jnp.maximum(M_new, 1e-6)

    Mx_new = M * x + dt * dMx_dt
    x_new = Mx_new / M_new
    x_new = jnp.clip(x_new, 0.0, 1.0)

    # Update temperature from bubble point
    T_new = bubble_point_temperature(x_new, P, thermo, T_init=T)

    new_state = CondenserState(M=M_new, x=x_new, T=T_new)

    outputs = CondenserOutputs(
        R=R,
        x_R=x_R,
        T_R=T_R,
        D=D,
        x_D=x_D,
        T_D=T_D,
        Q_C=Q_C,
    )

    return new_state, outputs


# =============================================================================
# Utility Functions
# =============================================================================


def create_initial_condenser_state(
    M: float,
    x: float,
    T: float,
) -> CondenserState:
    """Create an initial condenser/reflux drum state.

    Args:
        M: Initial liquid holdup [mol].
        x: Initial liquid composition [mol fraction].
        T: Initial temperature [K].

    Returns:
        CondenserState namedtuple.
    """
    return CondenserState(
        M=jnp.array(M),
        x=jnp.array(x),
        T=jnp.array(T),
    )


def create_condenser_inputs(
    V_in: float,
    y_in: float,
    T_V_in: float,
    reflux_ratio: float,
    D_setpoint: float,
) -> CondenserInputs:
    """Create condenser inputs namedtuple.

    Args:
        V_in: Vapor inflow [mol/s].
        y_in: Vapor composition [mol fraction].
        T_V_in: Vapor temperature [K].
        reflux_ratio: Reflux ratio (R/D).
        D_setpoint: Distillate flow setpoint [mol/s].

    Returns:
        CondenserInputs namedtuple.
    """
    return CondenserInputs(
        V_in=jnp.array(V_in),
        y_in=jnp.array(y_in),
        T_V_in=jnp.array(T_V_in),
        reflux_ratio=jnp.array(reflux_ratio),
        D_setpoint=jnp.array(D_setpoint),
    )
