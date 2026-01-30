"""Unit conversion helpers for data import/export.

This module provides utilities for converting between common
engineering units and the SI units used internally by the simulator.

Internal units:
- Temperature: K (Kelvin)
- Pressure: bar
- Flow: mol/s
- Heat: W (J/s)
- Time: s (seconds)
- Composition: mole fraction (0-1)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np


class TemperatureUnit(Enum):
    """Temperature units."""
    KELVIN = "K"
    CELSIUS = "C"
    FAHRENHEIT = "F"


class PressureUnit(Enum):
    """Pressure units."""
    BAR = "bar"
    ATM = "atm"
    PSI = "psi"
    KPA = "kPa"
    MPA = "MPa"
    MMHG = "mmHg"


class FlowUnit(Enum):
    """Flow rate units."""
    MOL_PER_S = "mol/s"
    KMOL_PER_H = "kmol/h"
    KG_PER_H = "kg/h"
    LB_PER_H = "lb/h"
    GPM = "gpm"  # Gallons per minute (volumetric)


class HeatUnit(Enum):
    """Heat/power units."""
    WATT = "W"
    KILOWATT = "kW"
    BTU_PER_H = "BTU/h"
    KCAL_PER_H = "kcal/h"


class TimeUnit(Enum):
    """Time units."""
    SECOND = "s"
    MINUTE = "min"
    HOUR = "h"


@dataclass
class StandardUnits:
    """Standard (SI) units used internally.

    This dataclass documents the internal unit conventions.
    """
    temperature: str = "K"
    pressure: str = "bar"
    flow: str = "mol/s"
    heat: str = "W"
    time: str = "s"
    composition: str = "mole_fraction"


# Conversion factors to SI units
TEMPERATURE_TO_K = {
    "K": lambda T: T,
    "C": lambda T: T + 273.15,
    "F": lambda T: (T - 32) * 5 / 9 + 273.15,
}

TEMPERATURE_FROM_K = {
    "K": lambda T: T,
    "C": lambda T: T - 273.15,
    "F": lambda T: (T - 273.15) * 9 / 5 + 32,
}

PRESSURE_TO_BAR = {
    "bar": 1.0,
    "atm": 1.01325,
    "psi": 0.0689476,
    "kPa": 0.01,
    "MPa": 10.0,
    "mmHg": 0.00133322,
}

FLOW_TO_MOL_PER_S = {
    "mol/s": 1.0,
    "kmol/h": 1000 / 3600,  # kmol/h to mol/s
}

HEAT_TO_W = {
    "W": 1.0,
    "kW": 1000.0,
    "BTU/h": 0.293071,
    "kcal/h": 1.163,
}

TIME_TO_S = {
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}


class UnitConverter:
    """Unit conversion utility class.

    Provides methods for converting between engineering units
    and the SI units used internally by the simulator.
    """

    @staticmethod
    def temperature_to_si(
        value: Union[float, np.ndarray],
        from_unit: str,
    ) -> Union[float, np.ndarray]:
        """Convert temperature to Kelvin.

        Args:
            value: Temperature value(s).
            from_unit: Source unit ("K", "C", "F").

        Returns:
            Temperature in Kelvin.
        """
        if from_unit not in TEMPERATURE_TO_K:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        return TEMPERATURE_TO_K[from_unit](value)

    @staticmethod
    def temperature_from_si(
        value: Union[float, np.ndarray],
        to_unit: str,
    ) -> Union[float, np.ndarray]:
        """Convert temperature from Kelvin.

        Args:
            value: Temperature in Kelvin.
            to_unit: Target unit ("K", "C", "F").

        Returns:
            Temperature in target unit.
        """
        if to_unit not in TEMPERATURE_FROM_K:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
        return TEMPERATURE_FROM_K[to_unit](value)

    @staticmethod
    def pressure_to_si(
        value: Union[float, np.ndarray],
        from_unit: str,
    ) -> Union[float, np.ndarray]:
        """Convert pressure to bar.

        Args:
            value: Pressure value(s).
            from_unit: Source unit.

        Returns:
            Pressure in bar.
        """
        if from_unit not in PRESSURE_TO_BAR:
            raise ValueError(f"Unknown pressure unit: {from_unit}")
        return value * PRESSURE_TO_BAR[from_unit]

    @staticmethod
    def pressure_from_si(
        value: Union[float, np.ndarray],
        to_unit: str,
    ) -> Union[float, np.ndarray]:
        """Convert pressure from bar.

        Args:
            value: Pressure in bar.
            to_unit: Target unit.

        Returns:
            Pressure in target unit.
        """
        if to_unit not in PRESSURE_TO_BAR:
            raise ValueError(f"Unknown pressure unit: {to_unit}")
        return value / PRESSURE_TO_BAR[to_unit]

    @staticmethod
    def flow_to_si(
        value: Union[float, np.ndarray],
        from_unit: str,
        mw: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Convert flow rate to mol/s.

        Args:
            value: Flow rate value(s).
            from_unit: Source unit.
            mw: Molecular weight [kg/mol] (required for mass flow conversion).

        Returns:
            Flow rate in mol/s.
        """
        if from_unit in FLOW_TO_MOL_PER_S:
            return value * FLOW_TO_MOL_PER_S[from_unit]
        elif from_unit == "kg/h" and mw is not None:
            return value / 3600 / mw  # kg/h to mol/s
        elif from_unit == "lb/h" and mw is not None:
            return value * 0.453592 / 3600 / mw  # lb/h to mol/s
        else:
            raise ValueError(f"Cannot convert {from_unit} to mol/s")

    @staticmethod
    def heat_to_si(
        value: Union[float, np.ndarray],
        from_unit: str,
    ) -> Union[float, np.ndarray]:
        """Convert heat/power to Watts.

        Args:
            value: Heat/power value(s).
            from_unit: Source unit.

        Returns:
            Power in Watts.
        """
        if from_unit not in HEAT_TO_W:
            raise ValueError(f"Unknown heat unit: {from_unit}")
        return value * HEAT_TO_W[from_unit]

    @staticmethod
    def time_to_si(
        value: Union[float, np.ndarray],
        from_unit: str,
    ) -> Union[float, np.ndarray]:
        """Convert time to seconds.

        Args:
            value: Time value(s).
            from_unit: Source unit.

        Returns:
            Time in seconds.
        """
        if from_unit not in TIME_TO_S:
            raise ValueError(f"Unknown time unit: {from_unit}")
        return value * TIME_TO_S[from_unit]


def convert_to_si(
    value: Union[float, np.ndarray],
    from_unit: str,
    quantity: str,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Generic conversion to SI units.

    Args:
        value: Value(s) to convert.
        from_unit: Source unit string.
        quantity: Quantity type ("temperature", "pressure", "flow", "heat", "time").
        **kwargs: Additional arguments (e.g., mw for mass flow).

    Returns:
        Value in SI units.
    """
    converter = UnitConverter()

    if quantity == "temperature":
        return converter.temperature_to_si(value, from_unit)
    elif quantity == "pressure":
        return converter.pressure_to_si(value, from_unit)
    elif quantity == "flow":
        return converter.flow_to_si(value, from_unit, **kwargs)
    elif quantity == "heat":
        return converter.heat_to_si(value, from_unit)
    elif quantity == "time":
        return converter.time_to_si(value, from_unit)
    else:
        raise ValueError(f"Unknown quantity type: {quantity}")


def convert_from_si(
    value: Union[float, np.ndarray],
    to_unit: str,
    quantity: str,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Generic conversion from SI units.

    Args:
        value: Value(s) in SI units.
        to_unit: Target unit string.
        quantity: Quantity type.
        **kwargs: Additional arguments.

    Returns:
        Value in target units.
    """
    converter = UnitConverter()

    if quantity == "temperature":
        return converter.temperature_from_si(value, to_unit)
    elif quantity == "pressure":
        return converter.pressure_from_si(value, to_unit)
    else:
        raise ValueError(f"Conversion from SI not implemented for: {quantity}")
