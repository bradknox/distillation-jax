"""Checksum verification utilities for benchmark data.

This module provides utilities for computing and verifying SHA256
checksums to ensure data integrity of downloaded benchmark files.
"""

import hashlib
from pathlib import Path
from typing import Optional


def compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute the checksum of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm to use (default: sha256).

    Returns:
        Hexadecimal string of the file's hash.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the algorithm is not supported.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_checksum(
    file_path: Path,
    expected_checksum: str,
    algorithm: str = "sha256",
) -> bool:
    """Verify a file's checksum against an expected value.

    Args:
        file_path: Path to the file.
        expected_checksum: The expected hexadecimal checksum.
        algorithm: Hash algorithm to use (default: sha256).

    Returns:
        True if the checksum matches, False otherwise.
    """
    try:
        actual = compute_checksum(file_path, algorithm)
        return actual.lower() == expected_checksum.lower()
    except FileNotFoundError:
        return False


# Known checksums for benchmark data files
# These are computed when files are first obtained and verified
KNOWN_CHECKSUMS = {
    # Placeholder - will be populated when actual files are downloaded
    # Format: "filename": "sha256_hexdigest"
}


def get_known_checksum(filename: str) -> Optional[str]:
    """Get the known checksum for a benchmark file.

    Args:
        filename: The filename to look up.

    Returns:
        The known SHA256 checksum, or None if not known.
    """
    return KNOWN_CHECKSUMS.get(filename)
