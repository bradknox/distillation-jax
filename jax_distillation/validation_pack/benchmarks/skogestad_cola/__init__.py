"""Skogestad Column A (COLA) benchmark implementation.

Column A is a 40-tray binary distillation column separating a
light/heavy mixture. It is a widely used academic benchmark for
control-oriented distillation studies.

Reference:
    Skogestad, S. (2007). "The dos and don'ts of distillation column control."
    Chemical Engineering Research and Design, 85(1), 13-23.
"""

from jax_distillation.validation_pack.benchmarks.skogestad_cola.cola_config_builder import (
    build_cola_config,
    get_cola_parameters,
    ColaParameters,
)
from jax_distillation.validation_pack.benchmarks.skogestad_cola.cola_reference_runner import (
    run_cola_benchmark,
    run_cola_steady_state,
    run_cola_step_response,
)
from jax_distillation.validation_pack.benchmarks.skogestad_cola.cola_metrics import (
    compute_cola_metrics,
    ColaValidationResult,
    compute_nrmse,
)

__all__ = [
    "build_cola_config",
    "get_cola_parameters",
    "ColaParameters",
    "run_cola_benchmark",
    "run_cola_steady_state",
    "run_cola_step_response",
    "compute_cola_metrics",
    "ColaValidationResult",
    "compute_nrmse",
]
