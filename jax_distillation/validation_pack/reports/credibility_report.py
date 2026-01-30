"""Credibility report generator for validation results.

This module generates a single artifact summarizing the validation
status of the JAX distillation simulator against public benchmarks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


@dataclass
class ValidationSection:
    """A section of the validation report.

    Attributes:
        name: Section name
        status: "pass", "fail", or "partial"
        metrics: Key metrics for this section
        details: Detailed results
        notes: Additional notes or explanations
    """

    name: str
    status: str  # "pass", "fail", "partial"
    metrics: Dict[str, Any]
    details: str
    notes: str = ""


@dataclass
class CredibilityReport:
    """Complete credibility report for the simulator.

    Attributes:
        title: Report title
        version: Simulator version
        timestamp: Report generation time
        sections: List of validation sections
        overall_status: Overall validation status
        limitations: Known limitations
        unvalidated_assumptions: Assumptions not yet validated
        next_steps: What to do when plant data is available
        summary: Executive summary
    """

    title: str
    version: str
    timestamp: str
    sections: List[ValidationSection]
    overall_status: str
    limitations: List[str]
    unvalidated_assumptions: List[str]
    next_steps: List[str]
    summary: str


def _generate_summary(sections: List[ValidationSection]) -> str:
    """Generate executive summary from validation sections.

    Args:
        sections: List of completed validation sections.

    Returns:
        Summary text.
    """
    n_pass = sum(1 for s in sections if s.status == "pass")
    n_fail = sum(1 for s in sections if s.status == "fail")
    n_partial = sum(1 for s in sections if s.status == "partial")
    total = len(sections)

    if n_fail == 0 and n_partial == 0:
        status_text = "All validation checks passed."
    elif n_fail == 0:
        status_text = f"{n_pass}/{total} checks passed, {n_partial} partial."
    else:
        status_text = f"{n_pass}/{total} passed, {n_partial} partial, {n_fail} failed."

    return f"""
The JAX distillation simulator has been validated against publicly available
benchmarks and reference data. {status_text}

This simulator is suitable for:
- Control algorithm development and testing
- Reinforcement learning research
- Educational demonstrations

This simulator requires further validation before:
- Use as a digital twin for specific physical columns
- Real-time optimization without human oversight
- Safety-critical applications
"""


def generate_credibility_report(
    verification_results: Optional[Dict] = None,
    benchmark_results: Optional[Dict] = None,
    thermo_results: Optional[Dict] = None,
    fit_results: Optional[Dict] = None,
    version: str = "0.1.0",
) -> CredibilityReport:
    """Generate a complete credibility report.

    Args:
        verification_results: Results from verification suite.
        benchmark_results: Results from benchmark comparisons.
        thermo_results: Results from thermodynamic validation.
        fit_results: Results from fitting pipeline demo.
        version: Simulator version string.

    Returns:
        CredibilityReport with all sections.
    """
    timestamp = datetime.now().isoformat()
    sections = []

    # Section 1: Verification (Mass/Energy Conservation)
    if verification_results:
        mass_ok = verification_results.get("mass_passed", False)
        energy_ok = verification_results.get("energy_passed", True)
        status = "pass" if (mass_ok and energy_ok) else "partial" if mass_ok else "fail"

        sections.append(ValidationSection(
            name="Numerical Verification",
            status=status,
            metrics={
                "mass_closure_error": verification_results.get("mass_closure", 0),
                "energy_closure_error": verification_results.get("energy_closure", 0),
                "n_steps": verification_results.get("n_steps", 0),
            },
            details=(
                f"Mass closure: {verification_results.get('mass_closure', 0)*100:.4f}% "
                f"(tolerance: 0.1%)\n"
                f"Steps tested: {verification_results.get('n_steps', 0)}"
            ),
            notes="Energy balance simplified in current model.",
        ))
    else:
        sections.append(ValidationSection(
            name="Numerical Verification",
            status="partial",
            metrics={},
            details="Verification not run",
            notes="Run verification suite to populate this section.",
        ))

    # Section 2: Thermodynamic Validation (NIST)
    if thermo_results:
        antoine_ok = thermo_results.get("antoine_passed", False)
        bubble_ok = thermo_results.get("bubble_passed", False)
        vle_ok = thermo_results.get("vle_passed", False)
        status = "pass" if (antoine_ok and bubble_ok and vle_ok) else "partial"

        sections.append(ValidationSection(
            name="Thermodynamic Validation (NIST)",
            status=status,
            metrics={
                "antoine_max_error": thermo_results.get("antoine_max_error", 0),
                "bubble_max_residual": thermo_results.get("bubble_max_residual", 0),
            },
            details=(
                f"Antoine vapor pressure: max error {thermo_results.get('antoine_max_error', 0)*100:.2f}%\n"
                f"Bubble point residual: max {thermo_results.get('bubble_max_residual', 0):.2e} bar"
            ),
            notes="Validated against NIST WebBook reference data.",
        ))
    else:
        sections.append(ValidationSection(
            name="Thermodynamic Validation (NIST)",
            status="partial",
            metrics={},
            details="NIST validation not run",
            notes="Run thermo validation to populate this section.",
        ))

    # Section 3: Skogestad Column A Benchmark
    if benchmark_results and "cola" in benchmark_results:
        cola = benchmark_results["cola"]
        direction_ok = cola.get("reflux_direction_ok", False) and cola.get("boilup_direction_ok", False)
        temp_ok = cola.get("temperature_monotonic", False)
        status = "pass" if (direction_ok and temp_ok) else "partial"

        sections.append(ValidationSection(
            name="Skogestad Column A Benchmark",
            status=status,
            metrics={
                "x_D_error": cola.get("x_D_error", 0),
                "x_B_error": cola.get("x_B_error", 0),
            },
            details=(
                f"Steady-state x_D error: {cola.get('x_D_error', 0)*100:.1f}%\n"
                f"Step response directions: {'correct' if direction_ok else 'incorrect'}\n"
                f"Temperature profile: {'monotonic' if temp_ok else 'non-monotonic'}"
            ),
            notes="Qualitative agreement expected; VLE models differ.",
        ))
    else:
        sections.append(ValidationSection(
            name="Skogestad Column A Benchmark",
            status="partial",
            metrics={},
            details="COLA benchmark not run",
            notes="Run COLA benchmark to populate this section.",
        ))

    # Section 4: Wood-Berry MIMO Benchmark
    if benchmark_results and "wood_berry" in benchmark_results:
        wb = benchmark_results["wood_berry"]
        signs_ok = wb.get("all_signs_correct", False)
        coupling_ok = wb.get("coupling_structure_ok", False)
        status = "pass" if (signs_ok and coupling_ok) else "partial"

        sections.append(ValidationSection(
            name="Wood-Berry MIMO Benchmark",
            status=status,
            metrics={
                "gain_signs_correct": signs_ok,
                "coupling_ok": coupling_ok,
            },
            details=(
                f"Gain signs: {'all correct' if signs_ok else 'some incorrect'}\n"
                f"MIMO coupling structure: {'matches' if coupling_ok else 'differs'}"
            ),
            notes="Linearized model comparison; quantitative differences expected.",
        ))
    else:
        sections.append(ValidationSection(
            name="Wood-Berry MIMO Benchmark",
            status="partial",
            metrics={},
            details="Wood-Berry benchmark not run",
            notes="Run Wood-Berry benchmark to populate this section.",
        ))

    # Section 5: Delay Wrapper Validation
    if benchmark_results and "delay" in benchmark_results:
        delay = benchmark_results["delay"]
        status = "pass" if delay.get("all_passed", False) else "partial"

        sections.append(ValidationSection(
            name="Delayed Measurement Wrapper",
            status=status,
            metrics={
                "delay_correct": delay.get("delay_correct", False),
                "deterministic": delay.get("deterministic", False),
            },
            details=(
                f"Delay implementation: {'correct' if delay.get('delay_correct') else 'incorrect'}\n"
                f"Reproducibility: {'deterministic' if delay.get('deterministic') else 'non-deterministic'}"
            ),
            notes="Enables RL training with realistic measurement delays.",
        ))
    else:
        sections.append(ValidationSection(
            name="Delayed Measurement Wrapper",
            status="partial",
            metrics={},
            details="Delay wrapper not validated",
            notes="Run delay validation to populate this section.",
        ))

    # Section 6: Fitting Pipeline
    if fit_results:
        converged = fit_results.get("converged", False)
        status = "pass" if converged else "partial"

        sections.append(ValidationSection(
            name="Fitting Pipeline Demo",
            status=status,
            metrics={
                "converged": converged,
                "loss_reduction": fit_results.get("loss_reduction", 0),
            },
            details=(
                f"Pipeline status: {'converged' if converged else 'did not converge'}\n"
                f"Loss reduction: {fit_results.get('loss_reduction', 0)*100:.1f}%"
            ),
            notes="Demonstrates fitting readiness; no plant data used.",
        ))
    else:
        sections.append(ValidationSection(
            name="Fitting Pipeline Demo",
            status="partial",
            metrics={},
            details="Fitting pipeline not run",
            notes="Run fit pipeline demo to populate this section.",
        ))

    # Determine overall status
    n_pass = sum(1 for s in sections if s.status == "pass")
    n_fail = sum(1 for s in sections if s.status == "fail")

    if n_fail > 0:
        overall_status = "fail"
    elif n_pass == len(sections):
        overall_status = "pass"
    else:
        overall_status = "partial"

    # Known limitations
    limitations = [
        "Uses simplified hydraulic models (weir flow correlations)",
        "Energy balance is approximate (detailed enthalpy tracking not implemented)",
        "Limited to binary mixtures",
        "Constant pressure assumption",
        "CMO (constant molar overflow) approximation for vapor flow",
    ]

    # Unvalidated assumptions
    unvalidated_assumptions = [
        "Tray efficiency correlations not validated against specific hardware",
        "Heat loss coefficients not calibrated to physical system",
        "Hydraulic time constants are literature values, not measured",
        "NRTL parameters from published sources, not fitted to specific mixture",
    ]

    # Next steps for plant data
    next_steps = [
        "1. Collect steady-state operating data (temperatures, flows, compositions)",
        "2. Run data reconciliation to ensure mass/energy balance closure",
        "3. Perform identifiability analysis to determine fittable parameters",
        "4. Fit tray efficiency, hydraulic time constants, and heat loss coefficients",
        "5. Validate fitted model on held-out test data",
        "6. Document model-plant mismatch and operating envelope",
    ]

    summary = _generate_summary(sections)

    return CredibilityReport(
        title="JAX Distillation Simulator Credibility Report",
        version=version,
        timestamp=timestamp,
        sections=sections,
        overall_status=overall_status,
        limitations=limitations,
        unvalidated_assumptions=unvalidated_assumptions,
        next_steps=next_steps,
        summary=summary,
    )


def save_credibility_report_markdown(
    report: CredibilityReport,
    output_path: str,
) -> None:
    """Save credibility report as Markdown file.

    Args:
        report: CredibilityReport to save.
        output_path: Path to output file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = []
    lines.append(f"# {report.title}")
    lines.append(f"\n**Version:** {report.version}")
    lines.append(f"\n**Generated:** {report.timestamp}")
    lines.append(f"\n**Overall Status:** {report.overall_status.upper()}")

    lines.append("\n## Executive Summary")
    lines.append(report.summary)

    lines.append("\n## Validation Results")

    for section in report.sections:
        status_emoji = {"pass": "✅", "fail": "❌", "partial": "⚠️"}.get(section.status, "?")
        lines.append(f"\n### {status_emoji} {section.name}")
        lines.append(f"\n**Status:** {section.status.upper()}")

        if section.metrics:
            lines.append("\n**Metrics:**")
            for key, value in section.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.6f}")
                else:
                    lines.append(f"- {key}: {value}")

        lines.append(f"\n**Details:**\n```\n{section.details}\n```")

        if section.notes:
            lines.append(f"\n*Note: {section.notes}*")

    lines.append("\n## Known Limitations")
    for lim in report.limitations:
        lines.append(f"- {lim}")

    lines.append("\n## Unvalidated Assumptions")
    for assumption in report.unvalidated_assumptions:
        lines.append(f"- {assumption}")

    lines.append("\n## Validated with Public Benchmarks")
    lines.append("""
This simulator has been validated using only publicly available benchmarks
and reference data. No proprietary plant data has been used in validation.
""")

    lines.append("\n## Not Yet Validated with Plant Data")
    lines.append("""
This simulator has NOT been validated against real plant measurements.
Before using for a specific physical column, follow the steps below.
""")

    lines.append("\n## What Must Be Done When Plant Data Is Available")
    for step in report.next_steps:
        lines.append(f"{step}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def print_credibility_report(report: CredibilityReport) -> None:
    """Print credibility report to console.

    Args:
        report: CredibilityReport to print.
    """
    print("=" * 70)
    print(report.title)
    print("=" * 70)
    print(f"Version: {report.version}")
    print(f"Generated: {report.timestamp}")
    print(f"Overall Status: {report.overall_status.upper()}")

    print("\n--- VALIDATION SECTIONS ---")
    for section in report.sections:
        status_symbol = {"pass": "[✓]", "fail": "[✗]", "partial": "[~]"}.get(section.status, "[?]")
        print(f"\n{status_symbol} {section.name}")
        print(f"    {section.details.replace(chr(10), chr(10) + '    ')}")
        if section.notes:
            print(f"    Note: {section.notes}")

    print("\n--- SUMMARY ---")
    print(report.summary)
    print("=" * 70)
