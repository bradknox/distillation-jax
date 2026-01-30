"""Public benchmark implementations for validation.

This package provides implementations of standard academic benchmarks
for validating the distillation simulator against published results.

Benchmarks:
    skogestad_cola: Skogestad Column A (40-tray binary column)
    wood_berry: Wood-Berry 2x2 MIMO transfer function model
    debutanizer_delay: Delayed measurement wrapper for RL
"""

from jax_distillation.validation_pack.benchmarks.skogestad_cola import (
    build_cola_config,
    run_cola_benchmark,
    ColaValidationResult,
)
from jax_distillation.validation_pack.benchmarks.wood_berry import (
    WoodBerryModel,
    run_wood_berry_benchmark,
    WoodBerryValidationResult,
)
from jax_distillation.validation_pack.benchmarks.debutanizer_delay import (
    DelayWrapper,
    run_delay_validation,
    DelayValidationResult,
)

__all__ = [
    "build_cola_config",
    "run_cola_benchmark",
    "ColaValidationResult",
    "WoodBerryModel",
    "run_wood_berry_benchmark",
    "WoodBerryValidationResult",
    "DelayWrapper",
    "run_delay_validation",
    "DelayValidationResult",
]
