"""Wood-Berry distillation column MIMO benchmark.

The Wood-Berry model is a classic 2x2 MIMO transfer function model
for distillation column control benchmarking. It represents the
linearized dynamics around an operating point.

Reference:
    Wood, R.K. and Berry, M.W. (1973). "Terminal composition control of a
    binary distillation column." Chemical Engineering Science, 28(9), 1707-1717.
"""

from jax_distillation.validation_pack.benchmarks.wood_berry.wood_berry_model import (
    WoodBerryModel,
    get_wood_berry_coefficients,
    simulate_wood_berry_step_response,
)
from jax_distillation.validation_pack.benchmarks.wood_berry.wood_berry_metrics import (
    run_wood_berry_benchmark,
    WoodBerryValidationResult,
    compare_with_jax_simulator,
)

__all__ = [
    "WoodBerryModel",
    "get_wood_berry_coefficients",
    "simulate_wood_berry_step_response",
    "run_wood_berry_benchmark",
    "WoodBerryValidationResult",
    "compare_with_jax_simulator",
]
