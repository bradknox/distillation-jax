"""Credibility report generation for validation results.

This module generates comprehensive reports documenting the
validation status of the simulator against public benchmarks.
"""

from jax_distillation.validation_pack.reports.credibility_report import (
    generate_credibility_report,
    CredibilityReport,
    ValidationSection,
)

__all__ = [
    "generate_credibility_report",
    "CredibilityReport",
    "ValidationSection",
]
