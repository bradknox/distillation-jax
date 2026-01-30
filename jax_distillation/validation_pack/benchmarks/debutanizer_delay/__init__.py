"""Debutanizer-inspired delay wrapper for RL environments.

This module implements a Gymnasium wrapper that adds realistic
measurement delays to composition observations, mimicking the
behavior of real distillation columns with gas chromatograph
analyzers.

Reference:
    Fortuna, L., Graziani, S., Rizzo, A., and Xibilia, M.G. (2007).
    "Soft Sensors for Monitoring and Control of Industrial Processes."
    Springer-Verlag London.
"""

from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_wrapper import (
    DelayWrapper,
    DelayConfig,
)
from jax_distillation.validation_pack.benchmarks.debutanizer_delay.delay_metrics import (
    run_delay_validation,
    DelayValidationResult,
)

__all__ = [
    "DelayWrapper",
    "DelayConfig",
    "run_delay_validation",
    "DelayValidationResult",
]
