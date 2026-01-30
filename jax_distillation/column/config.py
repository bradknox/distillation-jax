"""Configuration dataclasses for distillation column simulation.

This module provides structured configuration for:
- Column geometry (trays, diameters, weir heights)
- Operating conditions (feed, pressure)
- Simulation parameters (time step, substeps)
- Controller tuning parameters

All configurations are JAX-compatible using chex.dataclass.
"""

import chex
import jax.numpy as jnp
from typing import Optional

from jax_distillation.core.types import ThermoParams, HydraulicParams
from jax_distillation.core.thermodynamics import create_methanol_water_thermo
from jax_distillation.core.hydraulics import create_default_hydraulic_params


@chex.dataclass
class ColumnGeometry:
    """Physical geometry of the distillation column.

    Attributes:
        n_trays: Number of theoretical trays (excluding reboiler/condenser).
        feed_tray: Feed tray location (1=top, n_trays=bottom).
        column_diameter: Column diameter [m].
        weir_height: Weir height [m].
        weir_length: Weir length [m].
        tray_spacing: Spacing between trays [m].
        downcomer_area: Downcomer area [m²].
        active_area: Active (bubbling) area [m²].
        hole_diameter: Sieve tray hole diameter [m].
        hole_area: Total hole area [m²].
    """

    n_trays: int
    feed_tray: int
    column_diameter: jnp.ndarray
    weir_height: jnp.ndarray
    weir_length: jnp.ndarray
    tray_spacing: jnp.ndarray
    downcomer_area: jnp.ndarray
    active_area: jnp.ndarray
    hole_diameter: jnp.ndarray
    hole_area: jnp.ndarray


@chex.dataclass
class FeedConditions:
    """Feed stream conditions.

    Attributes:
        F: Feed flow rate [mol/s].
        z_F: Feed composition [mol fraction of light component].
        T_F: Feed temperature [K].
        q: Feed quality (1=saturated liquid, 0=saturated vapor).
    """

    F: jnp.ndarray
    z_F: jnp.ndarray
    T_F: jnp.ndarray
    q: jnp.ndarray


@chex.dataclass
class ControllerParams:
    """Controller tuning parameters.

    Attributes:
        Kp_level_reboiler: Proportional gain for reboiler level [1/s].
        Kp_level_condenser: Proportional gain for condenser level [1/s].
        M_setpoint_reboiler: Reboiler holdup setpoint [mol].
        M_setpoint_condenser: Condenser holdup setpoint [mol].
    """

    Kp_level_reboiler: jnp.ndarray
    Kp_level_condenser: jnp.ndarray
    M_setpoint_reboiler: jnp.ndarray
    M_setpoint_condenser: jnp.ndarray


@chex.dataclass
class SimulationParams:
    """Simulation parameters.

    Attributes:
        dt: Outer time step [s].
        n_substeps: Number of integration substeps per outer step.
        murphree_efficiency: Murphree vapor efficiency (0-1).
    """

    dt: jnp.ndarray
    n_substeps: int
    murphree_efficiency: jnp.ndarray


@chex.dataclass
class ColumnConfig:
    """Complete configuration for distillation column simulation.

    This dataclass bundles all configuration needed to run a simulation:
    - Physical geometry
    - Thermodynamic model parameters
    - Operating conditions
    - Controller tuning
    - Simulation parameters
    - Hydraulic parameters

    Attributes:
        geometry: Column geometry specification.
        thermo: Thermodynamic parameters.
        feed: Feed stream conditions.
        controllers: Controller tuning parameters.
        simulation: Simulation parameters.
        hydraulics: Hydraulic parameters for tray dynamics.
        P: Operating pressure [bar].
    """

    geometry: ColumnGeometry
    thermo: ThermoParams
    feed: FeedConditions
    controllers: ControllerParams
    simulation: SimulationParams
    hydraulics: HydraulicParams
    P: jnp.ndarray


def create_default_geometry(
    n_trays: int = 10,
    feed_tray: int = 5,
    column_diameter: float = 0.05,
) -> ColumnGeometry:
    """Create default column geometry for teaching column.

    Based on Armfield UOP3CC specifications:
    - Small-scale teaching column
    - Sieve tray configuration

    Args:
        n_trays: Number of theoretical trays.
        feed_tray: Feed tray location (1=top, n_trays=bottom).
        column_diameter: Column diameter [m].

    Returns:
        ColumnGeometry with default parameters.
    """
    D = column_diameter
    A_column = jnp.pi * (D / 2) ** 2

    # Typical proportions for sieve trays
    weir_length = 0.7 * D  # 70% of diameter
    downcomer_area = 0.1 * A_column  # 10% of column area
    active_area = 0.8 * A_column  # 80% active
    hole_area = 0.1 * active_area  # 10% open area

    return ColumnGeometry(
        n_trays=n_trays,
        feed_tray=feed_tray,
        column_diameter=jnp.array(D),
        weir_height=jnp.array(0.025),  # 25 mm
        weir_length=jnp.array(weir_length),
        tray_spacing=jnp.array(0.15),  # 150 mm
        downcomer_area=jnp.array(downcomer_area),
        active_area=jnp.array(active_area),
        hole_diameter=jnp.array(0.005),  # 5 mm holes
        hole_area=jnp.array(hole_area),
    )


def create_default_feed(
    F: float = 0.1,
    z_F: float = 0.5,
    T_F: float = 350.0,
    q: float = 1.0,
) -> FeedConditions:
    """Create default feed conditions.

    Args:
        F: Feed flow rate [mol/s].
        z_F: Feed composition [mol fraction].
        T_F: Feed temperature [K].
        q: Feed quality (1=saturated liquid).

    Returns:
        FeedConditions with specified parameters.
    """
    return FeedConditions(
        F=jnp.array(F),
        z_F=jnp.array(z_F),
        T_F=jnp.array(T_F),
        q=jnp.array(q),
    )


def create_default_controllers(
    M_setpoint_reboiler: float = 10.0,
    M_setpoint_condenser: float = 5.0,
) -> ControllerParams:
    """Create default controller parameters.

    Args:
        M_setpoint_reboiler: Reboiler holdup setpoint [mol].
        M_setpoint_condenser: Condenser holdup setpoint [mol].

    Returns:
        ControllerParams with default tuning.
    """
    return ControllerParams(
        Kp_level_reboiler=jnp.array(0.1),
        Kp_level_condenser=jnp.array(0.1),
        M_setpoint_reboiler=jnp.array(M_setpoint_reboiler),
        M_setpoint_condenser=jnp.array(M_setpoint_condenser),
    )


def create_default_simulation(
    dt: float = 1.0,
    n_substeps: int = 10,
    murphree_efficiency: float = 0.7,
) -> SimulationParams:
    """Create default simulation parameters.

    Args:
        dt: Outer time step [s].
        n_substeps: Number of integration substeps.
        murphree_efficiency: Murphree vapor efficiency.

    Returns:
        SimulationParams with default values.
    """
    return SimulationParams(
        dt=jnp.array(dt),
        n_substeps=n_substeps,
        murphree_efficiency=jnp.array(murphree_efficiency),
    )


def create_teaching_column_config(
    n_trays: int = 10,
    feed_tray: int = 5,
    feed_rate: float = 0.1,
    feed_composition: float = 0.5,
    pressure: float = 1.0,
    tau_L: float = 3.0,
    j: float = 0.0,
) -> ColumnConfig:
    """Create configuration for a teaching-scale distillation column.

    Default configuration based on Armfield UOP3CC:
    - 10 theoretical trays
    - Methanol-water mixture
    - Atmospheric pressure

    Args:
        n_trays: Number of theoretical trays.
        feed_tray: Feed tray location.
        feed_rate: Feed flow rate [mol/s].
        feed_composition: Feed composition [mol fraction methanol].
        pressure: Operating pressure [bar].
        tau_L: Hydraulic time constant [s] (default 3.0, range 0.5-15s).
        j: Vapor-liquid coupling parameter (default 0.0, range -5 to +5).

    Returns:
        ColumnConfig ready for simulation.
    """
    geometry = create_default_geometry(n_trays=n_trays, feed_tray=feed_tray)
    hydraulics = create_default_hydraulic_params(
        column_diameter=float(geometry.column_diameter),
        weir_height=float(geometry.weir_height),
        tray_spacing=float(geometry.tray_spacing),
        tau_L=tau_L,
        j=j,
    )
    return ColumnConfig(
        geometry=geometry,
        thermo=create_methanol_water_thermo(),
        feed=create_default_feed(F=feed_rate, z_F=feed_composition),
        controllers=create_default_controllers(),
        simulation=create_default_simulation(),
        hydraulics=hydraulics,
        P=jnp.array(pressure),
    )


def validate_config(config: ColumnConfig) -> list[str]:
    """Validate column configuration for physical consistency.

    Args:
        config: Column configuration to validate.

    Returns:
        List of warning/error messages (empty if valid).
    """
    warnings = []

    # Check tray numbering
    if config.geometry.feed_tray < 1:
        warnings.append("Feed tray must be >= 1")
    if config.geometry.feed_tray > config.geometry.n_trays:
        warnings.append(
            f"Feed tray ({config.geometry.feed_tray}) > n_trays ({config.geometry.n_trays})"
        )

    # Check physical dimensions
    if config.geometry.column_diameter <= 0:
        warnings.append("Column diameter must be positive")
    if config.geometry.weir_height <= 0:
        warnings.append("Weir height must be positive")
    if config.geometry.tray_spacing <= 0:
        warnings.append("Tray spacing must be positive")

    # Check feed conditions
    if config.feed.F < 0:
        warnings.append("Feed rate must be non-negative")
    if config.feed.z_F < 0 or config.feed.z_F > 1:
        warnings.append("Feed composition must be in [0, 1]")
    if config.feed.q < 0 or config.feed.q > 1:
        warnings.append("Feed quality must be in [0, 1]")

    # Check simulation parameters
    if config.simulation.dt <= 0:
        warnings.append("Time step must be positive")
    if config.simulation.n_substeps < 1:
        warnings.append("Number of substeps must be >= 1")
    if config.simulation.murphree_efficiency < 0 or config.simulation.murphree_efficiency > 1:
        warnings.append("Murphree efficiency must be in [0, 1]")

    # Check pressure
    if config.P <= 0:
        warnings.append("Pressure must be positive")

    return warnings
