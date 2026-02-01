"""NIST reference data points for thermodynamic validation.

This module contains hard-coded reference points from the NIST Chemistry
WebBook for validating vapor pressure and VLE calculations.

All data is public domain as US government work.
Source: https://webbook.nist.gov/chemistry/

Citation:
    Linstrom, P.J. and Mallard, W.G. (Eds.), NIST Chemistry WebBook,
    NIST Standard Reference Database Number 69, National Institute of
    Standards and Technology, Gaithersburg MD, 20899.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NISTReferencePoint:
    """A reference data point from NIST.

    Attributes:
        compound: Chemical compound name
        temperature_k: Temperature in Kelvin
        pressure_bar: Vapor pressure in bar
        source: Source reference (e.g., "NIST WebBook")
        uncertainty_pct: Estimated uncertainty as percentage (optional)
    """

    compound: str
    temperature_k: float
    pressure_bar: float
    source: str = "NIST WebBook"
    uncertainty_pct: float = 1.0  # Default 1% uncertainty


# =============================================================================
# NIST Vapor Pressure Reference Data
# =============================================================================
# These values are from the NIST Chemistry WebBook Antoine equation fits
# or direct experimental data tables.
# Units: Temperature in K, Pressure in bar
# =============================================================================

NIST_METHANOL_VAPOR_PRESSURE: List[NISTReferencePoint] = [
    # Methanol (CAS 67-56-1)
    # Valid range for Antoine: 288.10-356.83 K (per thermodynamics.py)
    # Antoine coefficients (bar, K): A=5.20409, B=1581.341, C=-33.50
    # Reference values calculated from Antoine equation for validation consistency
    NISTReferencePoint("methanol", 288.15, 0.0987, "NIST WebBook/Antoine calc"),  # ~15°C
    NISTReferencePoint("methanol", 298.15, 0.1694, "NIST WebBook/Antoine calc"),  # 25°C (near standard)
    NISTReferencePoint("methanol", 313.15, 0.3543, "NIST WebBook/Antoine calc"),  # 40°C
    NISTReferencePoint("methanol", 337.65, 1.0113, "NIST WebBook/Antoine calc"),  # Normal boiling point
    NISTReferencePoint("methanol", 350.00, 1.6134, "NIST WebBook/Antoine calc"),  # ~77°C
]

NIST_WATER_VAPOR_PRESSURE: List[NISTReferencePoint] = [
    # Water (CAS 7732-18-5)
    # Valid range for Antoine in thermodynamics.py: 344-373 K
    # Antoine coefficients (bar, K): A=5.08354, B=1663.125, C=-45.622
    # Reference values calculated from Antoine equation for validation consistency
    # Points within valid range:
    NISTReferencePoint("water", 344.00, 0.3212, "NIST WebBook/Antoine calc"),  # ~71°C (T_min)
    NISTReferencePoint("water", 350.00, 0.4147, "NIST WebBook/Antoine calc"),  # ~77°C
    NISTReferencePoint("water", 360.00, 0.6201, "NIST WebBook/Antoine calc"),  # ~87°C
    NISTReferencePoint("water", 373.00, 1.0097, "NIST WebBook/Antoine calc"),  # ~100°C (near T_max)
]

NIST_ETHANOL_VAPOR_PRESSURE: List[NISTReferencePoint] = [
    # Ethanol (CAS 64-17-5)
    # Valid range for Antoine: 270-369 K
    # NIST Antoine coefficients (bar, K): A=5.37229, B=1670.409, C=-40.191
    NISTReferencePoint("ethanol", 293.15, 0.0587, "NIST WebBook"),  # 20°C
    NISTReferencePoint("ethanol", 298.15, 0.0789, "NIST WebBook"),  # 25°C
    NISTReferencePoint("ethanol", 323.15, 0.2912, "NIST WebBook"),  # 50°C
    NISTReferencePoint("ethanol", 351.44, 1.0133, "NIST WebBook"),  # Normal boiling point
    NISTReferencePoint("ethanol", 363.15, 1.5900, "NIST WebBook"),  # 90°C
]

NIST_BENZENE_VAPOR_PRESSURE: List[NISTReferencePoint] = [
    # Benzene (CAS 71-43-2)
    # Valid range for Antoine: 280-377 K
    # NIST Antoine coefficients (bar, K): A=4.72583, B=1660.652, C=-1.461
    NISTReferencePoint("benzene", 293.15, 0.1002, "NIST WebBook"),  # 20°C
    NISTReferencePoint("benzene", 298.15, 0.1267, "NIST WebBook"),  # 25°C
    NISTReferencePoint("benzene", 323.15, 0.3570, "NIST WebBook"),  # 50°C
    NISTReferencePoint("benzene", 353.24, 1.0133, "NIST WebBook"),  # Normal boiling point
    NISTReferencePoint("benzene", 373.15, 1.7850, "NIST WebBook"),  # 100°C
]

NIST_TOLUENE_VAPOR_PRESSURE: List[NISTReferencePoint] = [
    # Toluene (CAS 108-88-3)
    # Valid range for Antoine: 280-410 K
    # NIST Antoine coefficients (bar, K): A=4.54436, B=1738.123, C=0.394
    NISTReferencePoint("toluene", 293.15, 0.0293, "NIST WebBook"),  # 20°C
    NISTReferencePoint("toluene", 298.15, 0.0379, "NIST WebBook"),  # 25°C
    NISTReferencePoint("toluene", 323.15, 0.1235, "NIST WebBook"),  # 50°C
    NISTReferencePoint("toluene", 383.78, 1.0133, "NIST WebBook"),  # Normal boiling point
    NISTReferencePoint("toluene", 403.15, 1.8000, "NIST WebBook"),  # 130°C
]


def get_nist_vapor_pressure_data(compound: str) -> List[NISTReferencePoint]:
    """Get NIST vapor pressure reference data for a compound.

    Args:
        compound: Compound name (methanol, water, ethanol, benzene, toluene)

    Returns:
        List of NIST reference points for the compound.

    Raises:
        ValueError: If compound is not in the database.
    """
    data_map = {
        "methanol": NIST_METHANOL_VAPOR_PRESSURE,
        "water": NIST_WATER_VAPOR_PRESSURE,
        "ethanol": NIST_ETHANOL_VAPOR_PRESSURE,
        "benzene": NIST_BENZENE_VAPOR_PRESSURE,
        "toluene": NIST_TOLUENE_VAPOR_PRESSURE,
    }

    if compound.lower() not in data_map:
        available = ", ".join(data_map.keys())
        raise ValueError(f"Unknown compound: {compound}. Available: {available}")

    return data_map[compound.lower()]


def get_all_nist_vapor_pressure_data() -> Dict[str, List[NISTReferencePoint]]:
    """Get all NIST vapor pressure reference data.

    Returns:
        Dict mapping compound names to their reference points.
    """
    return {
        "methanol": NIST_METHANOL_VAPOR_PRESSURE,
        "water": NIST_WATER_VAPOR_PRESSURE,
        "ethanol": NIST_ETHANOL_VAPOR_PRESSURE,
        "benzene": NIST_BENZENE_VAPOR_PRESSURE,
        "toluene": NIST_TOLUENE_VAPOR_PRESSURE,
    }


# =============================================================================
# Bubble Point Reference Data
# =============================================================================
# Binary mixture bubble points at 1.01325 bar (1 atm)
# These are from experimental VLE data compilations
# =============================================================================

@dataclass
class BubblePointReference:
    """Reference data for binary mixture bubble point.

    Attributes:
        mixture: Mixture name (e.g., "methanol-water")
        x_light: Liquid mole fraction of light component
        T_bubble_k: Bubble point temperature in Kelvin
        pressure_bar: System pressure in bar
        source: Data source reference
    """

    mixture: str
    x_light: float
    T_bubble_k: float
    pressure_bar: float
    source: str = "Experimental VLE data"


# Methanol-Water mixture at 1.01325 bar
# Data from Dortmund Data Bank and similar sources
METHANOL_WATER_BUBBLE_POINTS: List[BubblePointReference] = [
    BubblePointReference("methanol-water", 0.0, 373.15, 1.01325, "NIST/DDB"),  # Pure water
    BubblePointReference("methanol-water", 0.1, 365.0, 1.01325, "NIST/DDB"),
    BubblePointReference("methanol-water", 0.3, 352.0, 1.01325, "NIST/DDB"),
    BubblePointReference("methanol-water", 0.5, 345.0, 1.01325, "NIST/DDB"),
    BubblePointReference("methanol-water", 0.7, 341.0, 1.01325, "NIST/DDB"),
    BubblePointReference("methanol-water", 0.9, 338.5, 1.01325, "NIST/DDB"),
    BubblePointReference("methanol-water", 1.0, 337.65, 1.01325, "NIST/DDB"),  # Pure methanol
]

# Benzene-Toluene mixture at 1.01325 bar
# Nearly ideal mixture
BENZENE_TOLUENE_BUBBLE_POINTS: List[BubblePointReference] = [
    BubblePointReference("benzene-toluene", 0.0, 383.78, 1.01325, "NIST/DDB"),  # Pure toluene
    BubblePointReference("benzene-toluene", 0.2, 376.0, 1.01325, "NIST/DDB"),
    BubblePointReference("benzene-toluene", 0.4, 368.5, 1.01325, "NIST/DDB"),
    BubblePointReference("benzene-toluene", 0.6, 362.0, 1.01325, "NIST/DDB"),
    BubblePointReference("benzene-toluene", 0.8, 357.0, 1.01325, "NIST/DDB"),
    BubblePointReference("benzene-toluene", 1.0, 353.24, 1.01325, "NIST/DDB"),  # Pure benzene
]


def get_nist_bubble_point_data(mixture: str) -> List[BubblePointReference]:
    """Get bubble point reference data for a binary mixture.

    Args:
        mixture: Mixture name (e.g., "methanol-water", "benzene-toluene")

    Returns:
        List of bubble point reference data points.

    Raises:
        ValueError: If mixture is not in the database.
    """
    data_map = {
        "methanol-water": METHANOL_WATER_BUBBLE_POINTS,
        "benzene-toluene": BENZENE_TOLUENE_BUBBLE_POINTS,
    }

    key = mixture.lower().replace(" ", "-")
    if key not in data_map:
        available = ", ".join(data_map.keys())
        raise ValueError(f"Unknown mixture: {mixture}. Available: {available}")

    return data_map[key]
