"""Data sources management for public benchmarks.

This module handles downloading, caching, and verifying benchmark data
from public sources. All data is properly licensed and documented.
"""

from jax_distillation.validation_pack.data_sources.download import (
    download_all_benchmarks,
    download_benchmark,
    get_cache_dir,
)
from jax_distillation.validation_pack.data_sources.registry import (
    get_data_registry,
    DataSource,
    BenchmarkRegistry,
)
from jax_distillation.validation_pack.data_sources.checksums import (
    verify_checksum,
    compute_checksum,
)

__all__ = [
    "download_all_benchmarks",
    "download_benchmark",
    "get_cache_dir",
    "get_data_registry",
    "DataSource",
    "BenchmarkRegistry",
    "verify_checksum",
    "compute_checksum",
]
