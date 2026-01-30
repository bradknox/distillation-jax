"""Thermodynamics validation against NIST reference data.

This module validates the simulator's thermodynamic calculations
against public NIST WebBook reference data.

Validation includes:
- Antoine vapor pressure calculations vs NIST tabulated values
- Bubble point calculations
- VLE consistency checks (bounds, monotonicity)
"""

from jax_distillation.validation_pack.thermo_validation.nist_points import (
    NISTReferencePoint,
    get_nist_vapor_pressure_data,
    get_nist_bubble_point_data,
)
from jax_distillation.validation_pack.thermo_validation.test_antoine_against_nist import (
    validate_antoine_against_nist,
    AntoineValidationResult,
)
from jax_distillation.validation_pack.thermo_validation.test_bubble_point import (
    validate_bubble_point,
    BubblePointValidationResult,
)
from jax_distillation.validation_pack.thermo_validation.test_vle_consistency import (
    validate_vle_consistency,
    VLEConsistencyResult,
)

__all__ = [
    "NISTReferencePoint",
    "get_nist_vapor_pressure_data",
    "get_nist_bubble_point_data",
    "validate_antoine_against_nist",
    "AntoineValidationResult",
    "validate_bubble_point",
    "BubblePointValidationResult",
    "validate_vle_consistency",
    "VLEConsistencyResult",
]
