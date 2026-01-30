#!/usr/bin/env python3
"""Download public benchmark resources.

This script downloads and caches any required benchmark data
from public sources. Most benchmark data is hard-coded from
published papers and doesn't require downloading.

Usage:
    python scripts/download_public_benchmarks.py [--force]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_distillation.validation_pack.data_sources import (
    download_all_benchmarks,
    get_data_registry,
    get_cache_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download public benchmark resources"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks without downloading",
    )
    args = parser.parse_args()

    registry = get_data_registry()
    cache_dir = get_cache_dir()

    print("=" * 60)
    print("PUBLIC BENCHMARK DOWNLOAD")
    print("=" * 60)
    print(f"\nCache directory: {cache_dir}")

    if args.list:
        print("\nAvailable benchmarks:")
        for name in registry.list_sources():
            source = registry.get(name)
            if source:
                status = "hardcoded" if source.url is None else "downloadable"
                print(f"  - {name}: {source.description} [{status}]")
        return 0

    print("\nBenchmark data sources:")
    for name in registry.list_sources():
        source = registry.get(name)
        if source:
            if source.url is None:
                print(f"  ✓ {name}: hardcoded from publications")
            else:
                print(f"  ↓ {name}: {source.url}")

    print("\n" + "-" * 60)
    print("Downloading benchmarks...")

    results = download_all_benchmarks(force=args.force)

    print("\nResults:")
    for name, path in results.items():
        if path is None:
            print(f"  {name}: using hardcoded data (no download needed)")
        else:
            print(f"  {name}: {path}")

    print("\n" + "=" * 60)
    print("Download complete.")
    print("=" * 60)

    # Note about data usage
    print("\nData Usage Notes:")
    print("  - NIST data is public domain (US government work)")
    print("  - Skogestad/Wood-Berry coefficients are from published papers")
    print("  - All usage complies with fair use for research")
    print("\nSee jax_distillation/validation_pack/data_sources/licenses.md for details.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
