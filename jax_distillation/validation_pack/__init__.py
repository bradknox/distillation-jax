"""Phase 3: Public Benchmark Validation and Fitting Readiness Pack.

This package provides tools for validating the JAX distillation simulator
against publicly available benchmarks and preparing it for sim-to-real transfer.

Subpackages:
    data_sources: Download and manage benchmark data
    benchmarks: Skogestad Column A, Wood-Berry, debutanizer delay
    thermo_validation: NIST thermodynamics validation
    verification: Numerical verification (conservation, convergence)
    fitting: Parameter estimation and state reconciliation
    reports: Credibility report generation
"""

from jax_distillation.validation_pack.data_sources import (
    download_all_benchmarks,
    get_data_registry,
)
from jax_distillation.validation_pack.verification import (
    run_mass_energy_closure,
    run_timestep_convergence,
    check_invariants,
)
from jax_distillation.validation_pack.thermo_validation import (
    validate_antoine_against_nist,
    validate_bubble_point,
    validate_vle_consistency,
)

__all__ = [
    "download_all_benchmarks",
    "get_data_registry",
    "run_mass_energy_closure",
    "run_timestep_convergence",
    "check_invariants",
    "validate_antoine_against_nist",
    "validate_bubble_point",
    "validate_vle_consistency",
]
