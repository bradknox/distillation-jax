"""Fitting pipeline for parameter estimation and data reconciliation.

This package provides tools for fitting the simulator to real plant data
when it becomes available. The pipeline includes:

- Unit conversion helpers
- Data reconciliation (mass/energy balance)
- State estimation (MHE/EKF with delay handling)
- Parameter estimation using JAX autodiff
- Identifiability analysis
- Fit quality reporting
"""

from jax_distillation.validation_pack.fitting.units import (
    UnitConverter,
    StandardUnits,
    convert_to_si,
    convert_from_si,
)
from jax_distillation.validation_pack.fitting.reconciliation import (
    reconcile_mass_balance,
    reconcile_measurements,
    ReconciliationResult,
)
from jax_distillation.validation_pack.fitting.state_estimation import (
    ExtendedKalmanFilter,
    MovingHorizonEstimator,
    run_state_estimation,
)
from jax_distillation.validation_pack.fitting.parameter_estimation import (
    fit_parameters,
    ParameterEstimationResult,
    FittableParameter,
)
from jax_distillation.validation_pack.fitting.identifiability import (
    analyze_identifiability,
    compute_sensitivity,
    IdentifiabilityResult,
)
from jax_distillation.validation_pack.fitting.reporting import (
    generate_fit_report,
    FitReport,
)

__all__ = [
    "UnitConverter",
    "StandardUnits",
    "convert_to_si",
    "convert_from_si",
    "reconcile_mass_balance",
    "reconcile_measurements",
    "ReconciliationResult",
    "ExtendedKalmanFilter",
    "MovingHorizonEstimator",
    "run_state_estimation",
    "fit_parameters",
    "ParameterEstimationResult",
    "FittableParameter",
    "analyze_identifiability",
    "compute_sensitivity",
    "IdentifiabilityResult",
    "generate_fit_report",
    "FitReport",
]
