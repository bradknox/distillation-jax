"""Registry of public data sources for benchmark validation.

This module maintains a registry of all public data sources used for
validation, including their URLs, checksums, and licensing information.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class LicenseType(Enum):
    """License types for benchmark data."""

    PUBLIC_DOMAIN = "public_domain"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC0 = "cc0"
    MIT = "mit"
    ACADEMIC = "academic"
    FAIR_USE = "fair_use"  # Published academic constants


@dataclass
class DataSource:
    """Metadata for a benchmark data source."""

    name: str
    description: str
    url: Optional[str]  # None if data is hardcoded from publications
    license: LicenseType
    citation: str
    checksum: Optional[str] = None  # SHA256, None if hardcoded
    local_path: Optional[str] = None  # Relative to cache directory
    version: str = "1.0"
    notes: str = ""


@dataclass
class BenchmarkRegistry:
    """Registry of all benchmark data sources."""

    sources: Dict[str, DataSource] = field(default_factory=dict)

    def register(self, source: DataSource) -> None:
        """Register a new data source."""
        self.sources[source.name] = source

    def get(self, name: str) -> Optional[DataSource]:
        """Get a data source by name."""
        return self.sources.get(name)

    def list_sources(self) -> List[str]:
        """List all registered source names."""
        return list(self.sources.keys())

    def get_by_license(self, license_type: LicenseType) -> List[DataSource]:
        """Get all sources with a specific license type."""
        return [s for s in self.sources.values() if s.license == license_type]


# Global registry instance
_REGISTRY = BenchmarkRegistry()


def _initialize_registry() -> None:
    """Initialize the registry with all known data sources."""

    # Skogestad Column A benchmark
    _REGISTRY.register(DataSource(
        name="skogestad_cola",
        description="Skogestad Column A (COLA) binary distillation benchmark",
        url="https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html",
        license=LicenseType.ACADEMIC,
        citation=(
            "Skogestad, S. (2007). 'The dos and don'ts of distillation column control.' "
            "Chemical Engineering Research and Design, 85(1), 13-23."
        ),
        notes=(
            "Column A is a 40-tray binary distillation column separating a "
            "methanol-water mixture. Parameters are published in academic literature "
            "and freely available for research purposes."
        ),
    ))

    # Wood-Berry MIMO model
    _REGISTRY.register(DataSource(
        name="wood_berry",
        description="Wood-Berry distillation column MIMO transfer function model",
        url=None,  # Classic published model, parameters are hardcoded
        license=LicenseType.FAIR_USE,
        citation=(
            "Wood, R.K. and Berry, M.W. (1973). 'Terminal composition control of a "
            "binary distillation column.' Chemical Engineering Science, 28(9), 1707-1717."
        ),
        notes=(
            "Classic 2x2 MIMO transfer function model. Parameters are published "
            "constants from the original 1973 paper, widely used in control literature."
        ),
    ))

    # NIST thermodynamic data
    _REGISTRY.register(DataSource(
        name="nist_webbook",
        description="NIST Chemistry WebBook thermodynamic reference data",
        url="https://webbook.nist.gov/chemistry/",
        license=LicenseType.PUBLIC_DOMAIN,
        citation=(
            "Linstrom, P.J. and Mallard, W.G. (Eds.), NIST Chemistry WebBook, "
            "NIST Standard Reference Database Number 69, National Institute of "
            "Standards and Technology, Gaithersburg MD, 20899."
        ),
        notes=(
            "Vapor pressure data from Antoine equation fits. Data is public domain "
            "as a US government publication."
        ),
    ))

    # Debutanizer soft sensor dataset
    _REGISTRY.register(DataSource(
        name="debutanizer_dataset",
        description="Debutanizer column soft sensor benchmark dataset",
        url=None,  # Will use synthetic data matching published characteristics
        license=LicenseType.ACADEMIC,
        citation=(
            "Fortuna, L., Graziani, S., Rizzo, A., and Xibilia, M.G. (2007). "
            "'Soft Sensors for Monitoring and Control of Industrial Processes.' "
            "Springer-Verlag London. ISBN 978-1-84628-479-3."
        ),
        notes=(
            "The debutanizer benchmark is characterized by significant measurement "
            "delay (gas chromatograph analysis time ~15-30 minutes). We use synthetic "
            "data matching published characteristics for delay validation."
        ),
    ))

    # Antoine coefficients from NIST
    _REGISTRY.register(DataSource(
        name="antoine_coefficients",
        description="Antoine equation coefficients for vapor pressure calculations",
        url="https://webbook.nist.gov/chemistry/",
        license=LicenseType.PUBLIC_DOMAIN,
        citation=(
            "NIST Chemistry WebBook, NIST Standard Reference Database Number 69. "
            "Antoine Equation Parameters."
        ),
        notes=(
            "Coefficients for methanol, water, ethanol, benzene, toluene. "
            "Units: pressure in bar, temperature in Kelvin."
        ),
    ))


# Initialize on module import
_initialize_registry()


def get_data_registry() -> BenchmarkRegistry:
    """Get the global benchmark data registry.

    Returns:
        BenchmarkRegistry: The registry containing all known data sources.
    """
    return _REGISTRY
