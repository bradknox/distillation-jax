"""Download and cache management for benchmark data.

This module provides utilities for downloading benchmark data from
public sources and managing a local cache for reproducibility.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from jax_distillation.validation_pack.data_sources.registry import (
    get_data_registry,
    DataSource,
)
from jax_distillation.validation_pack.data_sources.checksums import verify_checksum

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory for benchmark data.

    The cache directory is determined in the following order:
    1. JAX_DISTILLATION_CACHE environment variable
    2. ~/.cache/jax_distillation/benchmarks

    Returns:
        Path: The cache directory path.
    """
    env_cache = os.environ.get("JAX_DISTILLATION_CACHE")
    if env_cache:
        cache_dir = Path(env_cache)
    else:
        cache_dir = Path.home() / ".cache" / "jax_distillation" / "benchmarks"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a file from a URL.

    Args:
        url: The URL to download from.
        dest: The destination file path.
        timeout: Request timeout in seconds.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    try:
        request = Request(url, headers={"User-Agent": "jax-distillation/0.1.0"})
        with urlopen(request, timeout=timeout) as response:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(response.read())
        logger.info(f"Downloaded {url} to {dest}")
        return True
    except HTTPError as e:
        logger.error(f"HTTP error downloading {url}: {e.code} {e.reason}")
        return False
    except URLError as e:
        logger.error(f"URL error downloading {url}: {e.reason}")
        return False
    except TimeoutError:
        logger.error(f"Timeout downloading {url}")
        return False


def download_benchmark(
    name: str,
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Download a specific benchmark dataset.

    Args:
        name: The benchmark name (from registry).
        force: If True, re-download even if cached.
        cache_dir: Override the default cache directory.

    Returns:
        Path to the downloaded file, or None if download failed or
        the benchmark doesn't require downloading (hardcoded data).
    """
    registry = get_data_registry()
    source = registry.get(name)

    if source is None:
        logger.error(f"Unknown benchmark: {name}")
        return None

    # Some sources are hardcoded (no URL)
    if source.url is None:
        logger.info(f"Benchmark {name} uses hardcoded data, no download needed")
        return None

    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Determine local path
    if source.local_path:
        local_path = cache_dir / source.local_path
    else:
        # Generate a filename from the URL
        url_hash = hashlib.md5(source.url.encode()).hexdigest()[:8]
        local_path = cache_dir / name / f"{name}_{url_hash}.dat"

    # Check if already cached
    if local_path.exists() and not force:
        if source.checksum:
            if verify_checksum(local_path, source.checksum):
                logger.info(f"Using cached {name} at {local_path}")
                return local_path
            else:
                logger.warning(f"Checksum mismatch for {name}, re-downloading")
        else:
            logger.info(f"Using cached {name} at {local_path} (no checksum)")
            return local_path

    # Download
    success = download_file(source.url, local_path)
    if not success:
        return None

    # Verify checksum if available
    if source.checksum and not verify_checksum(local_path, source.checksum):
        logger.error(f"Checksum verification failed for {name}")
        local_path.unlink()  # Remove corrupted file
        return None

    return local_path


def download_all_benchmarks(
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    """Download all registered benchmarks.

    Args:
        force: If True, re-download even if cached.
        cache_dir: Override the default cache directory.

    Returns:
        Dict mapping benchmark names to their local paths (or None if
        the benchmark uses hardcoded data or download failed).
    """
    registry = get_data_registry()
    results = {}

    for name in registry.list_sources():
        source = registry.get(name)
        if source and source.url:
            results[name] = download_benchmark(name, force=force, cache_dir=cache_dir)
        else:
            # Hardcoded data, no download needed
            results[name] = None

    return results


def list_cached_benchmarks(cache_dir: Optional[Path] = None) -> List[str]:
    """List all benchmarks that are currently cached.

    Args:
        cache_dir: Override the default cache directory.

    Returns:
        List of benchmark names that have cached data.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    cached = []
    registry = get_data_registry()

    for name in registry.list_sources():
        source = registry.get(name)
        if source and source.local_path:
            local_path = cache_dir / source.local_path
            if local_path.exists():
                cached.append(name)

    return cached


def clear_cache(cache_dir: Optional[Path] = None) -> None:
    """Clear all cached benchmark data.

    Args:
        cache_dir: Override the default cache directory.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    import shutil

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"Cleared cache at {cache_dir}")
