"""Verification suite for numerical correctness.

This module provides tools for verifying that the simulator correctly
solves its own equations, independent of whether those equations
accurately represent physical reality.

Verification includes:
- Mass/energy conservation closure
- Timestep convergence studies
- Invariant checking (no NaN/Inf, bounds enforcement)
- Stiffness analysis and time scale probing
"""

from jax_distillation.validation_pack.verification.mass_energy_closure import (
    run_mass_energy_closure,
    MassEnergyClosureResult,
    check_mass_closure,
    check_energy_closure,
)
from jax_distillation.validation_pack.verification.timestep_convergence import (
    run_timestep_convergence,
    TimestepConvergenceResult,
    plot_convergence,
)
from jax_distillation.validation_pack.verification.invariants import (
    check_invariants,
    InvariantCheckResult,
    run_long_simulation_invariant_check,
)
from jax_distillation.validation_pack.verification.stiffness_probes import (
    analyze_time_scales,
    StiffnessAnalysisResult,
)

__all__ = [
    "run_mass_energy_closure",
    "MassEnergyClosureResult",
    "check_mass_closure",
    "check_energy_closure",
    "run_timestep_convergence",
    "TimestepConvergenceResult",
    "plot_convergence",
    "check_invariants",
    "InvariantCheckResult",
    "run_long_simulation_invariant_check",
    "analyze_time_scales",
    "StiffnessAnalysisResult",
]
