"""JAX-compatible dataclasses for distillation column state and parameters."""

from typing import NamedTuple

import chex
import jax.numpy as jnp


@chex.dataclass
class TrayState:
    """State of a single tray or vessel (reboiler/condenser).

    Attributes:
        liquid_holdup: Liquid holdup on the tray [mol].
        liquid_composition: Mole fraction of light component in liquid [mol/mol].
        temperature: Tray temperature [K].
    """

    liquid_holdup: chex.Array  # [mol]
    liquid_composition: chex.Array  # [mol fraction]
    temperature: chex.Array  # [K]


@chex.dataclass
class ColumnState:
    """Full column state including all trays and vessels.

    Attributes:
        tray_holdups: Liquid holdups for all trays, shape (n_trays,) [mol].
        tray_compositions: Liquid compositions for all trays, shape (n_trays,) [mol fraction].
        tray_temperatures: Temperatures for all trays, shape (n_trays,) [K].
        reboiler_holdup: Reboiler liquid holdup [mol].
        reboiler_composition: Reboiler liquid composition [mol fraction].
        reboiler_temperature: Reboiler temperature [K].
        condenser_holdup: Reflux drum liquid holdup [mol].
        condenser_composition: Reflux drum liquid composition [mol fraction].
        condenser_temperature: Reflux drum temperature [K].
        liquid_outflows: Dynamic liquid outflow states for hydraulic coupling, shape (n_trays,) [mol/s].
    """

    # Tray states (vectorized)
    tray_holdups: chex.Array  # shape (n_trays,)
    tray_compositions: chex.Array  # shape (n_trays,)
    tray_temperatures: chex.Array  # shape (n_trays,)

    # Reboiler state
    reboiler_holdup: chex.Array  # scalar
    reboiler_composition: chex.Array  # scalar
    reboiler_temperature: chex.Array  # scalar

    # Condenser/reflux drum state
    condenser_holdup: chex.Array  # scalar
    condenser_composition: chex.Array  # scalar
    condenser_temperature: chex.Array  # scalar

    # Hydraulic coupling state (dynamic liquid outflows)
    liquid_outflows: chex.Array  # shape (n_trays,)


@chex.dataclass
class AntoineParams:
    """Antoine equation parameters for a component.

    log10(P_sat[bar]) = A - B / (T[K] + C)

    Attributes:
        A: Antoine A coefficient.
        B: Antoine B coefficient.
        C: Antoine C coefficient.
        T_min: Minimum valid temperature [K].
        T_max: Maximum valid temperature [K].
    """

    A: chex.Array
    B: chex.Array
    C: chex.Array
    T_min: chex.Array  # Valid temperature range
    T_max: chex.Array


@chex.dataclass
class NRTLParams:
    """NRTL activity coefficient model parameters for a binary mixture.

    tau_12 = A_12 + B_12/T, tau_21 = A_21 + B_21/T
    G_12 = exp(-alpha * tau_12), G_21 = exp(-alpha * tau_21)

    Attributes:
        alpha: Non-randomness parameter (typically 0.1-0.3).
        A_12: tau_12 intercept.
        B_12: tau_12 temperature coefficient [K].
        A_21: tau_21 intercept.
        B_21: tau_21 temperature coefficient [K].
    """

    alpha: chex.Array
    A_12: chex.Array
    B_12: chex.Array
    A_21: chex.Array
    B_21: chex.Array


@chex.dataclass
class ThermoParams:
    """Thermodynamic parameters for a binary mixture.

    Attributes:
        antoine_1: Antoine parameters for light component.
        antoine_2: Antoine parameters for heavy component.
        nrtl: NRTL parameters (None for ideal mixture).
        cp_liquid_1: Liquid heat capacity of light component [J/mol/K].
        cp_liquid_2: Liquid heat capacity of heavy component [J/mol/K].
        cp_vapor_1: Vapor heat capacity of light component [J/mol/K].
        cp_vapor_2: Vapor heat capacity of heavy component [J/mol/K].
        hvap_1: Heat of vaporization of light component [J/mol].
        hvap_2: Heat of vaporization of heavy component [J/mol].
        mw_1: Molecular weight of light component [kg/mol].
        mw_2: Molecular weight of heavy component [kg/mol].
        T_ref: Reference temperature for enthalpy [K].
    """

    antoine_1: AntoineParams
    antoine_2: AntoineParams
    nrtl: NRTLParams | None
    cp_liquid_1: chex.Array  # [J/mol/K]
    cp_liquid_2: chex.Array  # [J/mol/K]
    cp_vapor_1: chex.Array  # [J/mol/K]
    cp_vapor_2: chex.Array  # [J/mol/K]
    hvap_1: chex.Array  # [J/mol]
    hvap_2: chex.Array  # [J/mol]
    mw_1: chex.Array  # [kg/mol]
    mw_2: chex.Array  # [kg/mol]
    T_ref: chex.Array  # [K]


@chex.dataclass
class HydraulicParams:
    """Hydraulic parameters for tray dynamics.

    Attributes:
        tau_L: Hydraulic time constant [s].
        j: Vapor-liquid coupling parameter (dimensionless).
        weir_height: Weir height [m].
        weir_length: Weir length [m].
        weir_coefficient: Weir discharge coefficient.
        tray_area: Active tray area [m^2].
        downcomer_area: Downcomer area [m^2].
        hole_diameter: Sieve tray hole diameter [m].
        C_sbf: Flooding capacity parameter.
        K2_weep: Weeping correlation coefficient.
    """

    tau_L: chex.Array  # [s]
    j: chex.Array  # dimensionless
    weir_height: chex.Array  # [m]
    weir_length: chex.Array  # [m]
    weir_coefficient: chex.Array  # dimensionless
    tray_area: chex.Array  # [m^2]
    downcomer_area: chex.Array  # [m^2]
    hole_diameter: chex.Array  # [m]
    C_sbf: chex.Array  # flooding parameter
    K2_weep: chex.Array  # weeping parameter


@chex.dataclass
class ColumnGeometry:
    """Column geometry parameters.

    Attributes:
        n_trays: Number of trays.
        feed_tray: Feed tray index (1-indexed, 1=top).
        column_diameter: Internal column diameter [m].
        tray_spacing: Spacing between trays [m].
    """

    n_trays: int
    feed_tray: int
    column_diameter: chex.Array  # [m]
    tray_spacing: chex.Array  # [m]


@chex.dataclass
class OperatingConditions:
    """Operating condition parameters.

    Attributes:
        pressure: Operating pressure [bar].
        feed_rate: Feed molar flow rate [mol/s].
        feed_composition: Feed light component mole fraction.
        feed_quality: Feed quality (fraction liquid, 0=all vapor, 1=all liquid).
        feed_temperature: Feed temperature [K].
    """

    pressure: chex.Array  # [bar]
    feed_rate: chex.Array  # [mol/s]
    feed_composition: chex.Array  # [mol fraction]
    feed_quality: chex.Array  # [0-1]
    feed_temperature: chex.Array  # [K]


@chex.dataclass
class ColumnParams:
    """Complete set of parameters for column simulation.

    Attributes:
        geometry: Column geometry parameters.
        thermo: Thermodynamic parameters.
        hydraulics: Hydraulic parameters.
        operating: Operating condition parameters.
        murphree_efficiency: Murphree vapor efficiency for each tray, shape (n_trays,).
        reboiler_holdup_nominal: Nominal reboiler holdup [mol].
        condenser_holdup_nominal: Nominal condenser holdup [mol].
        tray_holdup_nominal: Nominal tray holdup [mol].
    """

    geometry: ColumnGeometry
    thermo: ThermoParams
    hydraulics: HydraulicParams
    operating: OperatingConditions
    murphree_efficiency: chex.Array  # shape (n_trays,)
    reboiler_holdup_nominal: chex.Array  # [mol]
    condenser_holdup_nominal: chex.Array  # [mol]
    tray_holdup_nominal: chex.Array  # [mol]


@chex.dataclass
class Action:
    """Control actions for the distillation column.

    Attributes:
        reflux_ratio: Reflux ratio setpoint (R/D).
        reboiler_duty: Reboiler duty setpoint [W].
    """

    reflux_ratio: chex.Array  # R/D ratio
    reboiler_duty: chex.Array  # [W]


@chex.dataclass
class Flows:
    """Computed flow rates in the column.

    Attributes:
        liquid_flows: Liquid flows leaving each tray downward, shape (n_trays+1,) [mol/s].
                      Index 0 is reflux, index i is liquid leaving tray i.
        vapor_flows: Vapor flows leaving each tray upward, shape (n_trays+1,) [mol/s].
                     Index 0 is vapor from reboiler, index i is vapor leaving tray i.
        distillate: Distillate flow rate [mol/s].
        bottoms: Bottoms flow rate [mol/s].
        reflux: Reflux flow rate [mol/s].
        boilup: Vapor boilup from reboiler [mol/s].
    """

    liquid_flows: chex.Array  # shape (n_trays+1,)
    vapor_flows: chex.Array  # shape (n_trays+1,)
    distillate: chex.Array
    bottoms: chex.Array
    reflux: chex.Array
    boilup: chex.Array


class StepInfo(NamedTuple):
    """Additional information from a simulation step.

    Attributes:
        mass_balance_error: Relative mass balance error.
        energy_balance_error: Relative energy balance error.
        flooding_ratio: Maximum flooding ratio across trays.
        weeping: Whether any tray is weeping.
        constraint_violation: Whether any constraint is violated.
    """

    mass_balance_error: chex.Array
    energy_balance_error: chex.Array
    flooding_ratio: chex.Array
    weeping: chex.Array
    constraint_violation: chex.Array


def create_initial_state(params: ColumnParams) -> ColumnState:
    """Create an initial column state with nominal values.

    Args:
        params: Column parameters.

    Returns:
        Initial column state at approximate steady-state conditions.
    """
    n_trays = params.geometry.n_trays

    # Initialize with linear composition profile
    x_top = 0.95  # High purity at top
    x_bot = 0.05  # Low purity at bottom
    tray_compositions = jnp.linspace(x_top, x_bot, n_trays)

    # Initialize with nominal holdups
    tray_holdups = jnp.full(n_trays, params.tray_holdup_nominal)

    # Initialize temperatures (will be computed from bubble point in practice)
    # For now, use a linear approximation between typical boiling points
    T_top = 338.0  # Approximate for methanol-rich
    T_bot = 373.0  # Approximate for water-rich
    tray_temperatures = jnp.linspace(T_top, T_bot, n_trays)

    # Initialize liquid outflows (will reach steady state)
    # Start with approximate values based on feed rate
    liquid_outflows = jnp.full(n_trays, params.operating.feed_rate * 0.5)

    return ColumnState(
        tray_holdups=tray_holdups,
        tray_compositions=tray_compositions,
        tray_temperatures=tray_temperatures,
        reboiler_holdup=params.reboiler_holdup_nominal,
        reboiler_composition=jnp.array(x_bot),
        reboiler_temperature=jnp.array(T_bot),
        condenser_holdup=params.condenser_holdup_nominal,
        condenser_composition=jnp.array(x_top),
        condenser_temperature=jnp.array(T_top),
        liquid_outflows=liquid_outflows,
    )
