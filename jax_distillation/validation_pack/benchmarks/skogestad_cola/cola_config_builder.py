"""Configuration builder for Skogestad Column A benchmark.

This module builds a ColumnConfig that approximates the Skogestad
Column A (COLA) benchmark parameters as closely as possible given
the JAX simulator's model structure.

Note: COLA uses a simplified constant relative volatility model,
while our simulator uses NRTL activity coefficients. This will
cause some quantitative differences that are expected and documented.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from jax_distillation.column.config import (
    ColumnConfig,
    ColumnGeometry,
    FeedConditions,
    ControllerParams,
    SimulationParams,
)
from jax_distillation.core.thermodynamics import (
    create_methanol_water_thermo,
    ThermoParams,
)
from jax_distillation.core.hydraulics import create_default_hydraulic_params


@dataclass
class ColaParameters:
    """Skogestad Column A published parameters.

    These are the canonical values from the published benchmark.
    """

    # Column geometry
    n_trays: int = 40
    feed_tray: int = 21  # From bottom (1-indexed)

    # Operating conditions
    F: float = 1.0  # Feed rate [mol/s]
    z_F: float = 0.5  # Feed composition (light component)
    q: float = 1.0  # Feed quality (1.0 = saturated liquid)

    # Nominal products
    D: float = 0.5  # Distillate rate [mol/s]
    B: float = 0.5  # Bottoms rate [mol/s]
    x_D: float = 0.99  # Distillate purity
    x_B: float = 0.01  # Bottoms impurity

    # Flows
    L: float = 2.7059  # Reflux flow [mol/s]
    V: float = 3.2059  # Vapor boilup [mol/s]

    # Physical properties
    alpha: float = 1.5  # Relative volatility (constant)

    # Control parameters
    reflux_ratio: float = 5.4118  # L/D = 2.7059/0.5


def get_cola_parameters() -> ColaParameters:
    """Get the canonical Skogestad Column A parameters.

    Returns:
        ColaParameters with published values.
    """
    return ColaParameters()


def build_cola_config(
    thermo: Optional[ThermoParams] = None,
    dt: float = 1.0,
    n_substeps: int = 5,
    murphree_efficiency: float = 1.0,
    scale_factor: float = 0.1,
) -> ColumnConfig:
    """Build ColumnConfig matching COLA benchmark specifications.

    Since COLA uses constant relative volatility (α=1.5) and our
    simulator uses NRTL thermodynamics, we use methanol-water as
    the model mixture. The relative volatility of methanol-water
    varies (~2-5 depending on composition and temperature) but
    provides reasonable qualitative behavior.

    Args:
        thermo: Thermodynamic parameters. If None, uses methanol-water.
        dt: Simulation timestep [s].
        n_substeps: Number of integration substeps.
        murphree_efficiency: Tray efficiency (1.0 = ideal stages).
        scale_factor: Factor to scale flows (COLA uses 1 mol/s, teaching
                     columns typically use 0.1 mol/s).

    Returns:
        ColumnConfig configured for COLA-like behavior.
    """
    cola = get_cola_parameters()

    if thermo is None:
        thermo = create_methanol_water_thermo()

    # Geometry
    # Note: Our teaching column model has fewer trays, so we use
    # a scaled-down version that preserves the key characteristics
    geometry = ColumnGeometry(
        n_trays=min(cola.n_trays, 20),  # Scale down for faster simulation
        feed_tray=min(cola.feed_tray, 11),  # Proportionally scaled feed tray
        column_diameter=jnp.array(0.05),  # Small teaching column
        weir_height=jnp.array(0.02),  # 20mm weir height for proper Francis weir operation
        weir_length=jnp.array(0.035),  # 70% of column diameter
        tray_spacing=jnp.array(0.15),
        downcomer_area=jnp.array(0.001),
        active_area=jnp.array(0.0015),  # Active bubbling area
        hole_diameter=jnp.array(0.005),  # 5mm holes
        hole_area=jnp.array(0.0002),  # Total hole area
    )

    # Feed conditions (scaled)
    feed = FeedConditions(
        F=cola.F * scale_factor,
        z_F=cola.z_F,
        T_F=jnp.array(340.0),  # Approximate bubble point
        q=cola.q,
    )

    # Controller parameters
    controllers = ControllerParams(
        Kp_level_reboiler=jnp.array(0.5),
        Kp_level_condenser=jnp.array(0.5),
        M_setpoint_reboiler=jnp.array(2.0),
        M_setpoint_condenser=jnp.array(1.0),
    )

    # Simulation parameters
    simulation = SimulationParams(
        dt=dt,
        n_substeps=n_substeps,
        murphree_efficiency=murphree_efficiency,
    )

    # Hydraulic parameters - must match geometry for physical consistency.
    # COLA is a CMO benchmark: liquid flows are constant in each section
    # (no hydraulic dynamics). Our simulator uses Francis weir + dynamic
    # lag, which introduces composition-dependent flow sensitivity that
    # destabilizes the small-column geometry (dL/dM ≈ 3.5 mol/s/mol).
    # Using tau_L=1e8 effectively implements CMO by freezing L at the
    # initialized CMO values. Combined with R = RR * D in the condenser
    # (which keeps R ≈ L_above_init) and action-consistent initialization
    # (L_above = RR*D, L_below = L_above + q*F), this prevents persistent
    # holdup drift while still allowing step responses through the reflux
    # ratio control.
    hydraulics = create_default_hydraulic_params(
        column_diameter=float(geometry.column_diameter),
        weir_height=float(geometry.weir_height),
        tray_spacing=float(geometry.tray_spacing),
        tau_L=1e8,
    )

    config = ColumnConfig(
        geometry=geometry,
        thermo=thermo,
        feed=feed,
        controllers=controllers,
        simulation=simulation,
        hydraulics=hydraulics,
        P=jnp.array(1.0),  # 1 bar
    )

    return config


def get_cola_nominal_action(scale_factor: float = 0.1):
    """Get control action corresponding to COLA nominal operating point.

    Args:
        scale_factor: Flow scaling factor.

    Returns:
        ColumnAction for nominal operation.
    """
    from jax_distillation.column.column import ColumnAction
    from jax_distillation.core.thermodynamics import (
        equilibrium_vapor_composition,
        bubble_point_temperature,
        liquid_enthalpy,
        vapor_enthalpy,
    )

    cola = get_cola_parameters()
    thermo = create_methanol_water_thermo()

    V_scaled = cola.V * scale_factor

    # Compute Q_R from actual NRTL thermodynamics at the reboiler's
    # converged steady-state composition. Our NRTL methanol-water column
    # achieves x_B ≈ 0 (vs COLA's target x_B=0.01 with constant α=1.5)
    # because NRTL gives higher relative volatility. Using x_B=0 for
    # delta_H ensures V matches V_target at steady state, so D+B=F
    # (flow closure) holds when L is frozen at CMO values.
    x_B = jnp.array(0.0)
    P = jnp.array(1.0)
    T_B = bubble_point_temperature(x_B, P, thermo, T_init=jnp.array(370.0))
    y_eq = equilibrium_vapor_composition(x_B, T_B, P, thermo)
    h_L = liquid_enthalpy(x_B, T_B, thermo)
    h_V = vapor_enthalpy(y_eq, T_B, thermo)
    delta_H_vap = float(h_V - h_L)
    delta_H_vap = max(delta_H_vap, 1000.0)  # Safety floor

    Q_R = V_scaled * delta_H_vap

    return ColumnAction(
        Q_R=jnp.array(Q_R),
        reflux_ratio=jnp.array(cola.reflux_ratio),
        B_setpoint=jnp.array(cola.B * scale_factor),
        D_setpoint=jnp.array(cola.D * scale_factor),
    )


def print_cola_config_comparison(config: ColumnConfig) -> None:
    """Print comparison between built config and COLA parameters.

    Args:
        config: ColumnConfig to compare.
    """
    cola = get_cola_parameters()

    print("=" * 60)
    print("COLA Configuration Comparison")
    print("=" * 60)

    print("\nGeometry:")
    print(f"  COLA n_trays:   {cola.n_trays}")
    print(f"  Model n_trays:  {config.geometry.n_trays}")
    print(f"  COLA feed_tray: {cola.feed_tray}")
    print(f"  Model feed_tray: {config.geometry.feed_tray}")

    print("\nFeed Conditions:")
    print(f"  COLA F:   {cola.F} mol/s")
    print(f"  Model F:  {config.feed.F} mol/s")
    print(f"  COLA z_F: {cola.z_F}")
    print(f"  Model z_F: {float(config.feed.z_F)}")
    print(f"  COLA q:   {cola.q}")
    print(f"  Model q:  {float(config.feed.q)}")

    print("\nNominal Products (COLA targets):")
    print(f"  x_D: {cola.x_D}")
    print(f"  x_B: {cola.x_B}")
    print(f"  Reflux ratio: {cola.reflux_ratio:.2f}")

    print("\nModel Differences:")
    print("  - COLA uses constant α = 1.5")
    print("  - Our model uses NRTL activity coefficients")
    print("  - Quantitative differences expected")
    print("=" * 60)
