"""Validation suite for physical consistency and benchmark comparisons."""

from jax_distillation.validation.conservation import (
    ConservationMetrics,
    compute_total_mass,
    check_mass_balance,
    check_component_balance,
    validate_simulation_step,
    run_conservation_validation,
)
from jax_distillation.validation.steady_state import (
    SteadyStateMetrics,
    check_steady_state,
    run_to_steady_state,
    validate_separation_quality,
    validate_temperature_profile,
    run_steady_state_validation,
)
from jax_distillation.validation.dynamic_response import (
    StepResponseMetrics,
    run_step_response,
    analyze_step_response,
    run_reboiler_duty_step_test,
    run_reflux_ratio_step_test,
    validate_time_constants,
)
from jax_distillation.validation.benchmarks import (
    BenchmarkResults,
    benchmark_single_step,
    benchmark_trajectory,
    benchmark_jit_compilation,
    run_all_benchmarks,
    validate_performance_requirements,
)

__all__ = [
    # Conservation
    "ConservationMetrics",
    "compute_total_mass",
    "check_mass_balance",
    "run_conservation_validation",
    # Steady state
    "SteadyStateMetrics",
    "check_steady_state",
    "run_to_steady_state",
    "run_steady_state_validation",
    # Dynamic response
    "StepResponseMetrics",
    "run_step_response",
    "run_reboiler_duty_step_test",
    "run_reflux_ratio_step_test",
    # Benchmarks
    "BenchmarkResults",
    "benchmark_single_step",
    "run_all_benchmarks",
    "validate_performance_requirements",
]
