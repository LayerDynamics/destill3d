"""
Security and license compliance for Destill3D.

Provides input validation for files and URLs, and license
filtering for acquisition results.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urlparse

from destill3d.core.config import SecurityConfig

logger = logging.getLogger(__name__)

# Default allowed 3D file extensions
DEFAULT_ALLOWED_EXTENSIONS = {
    ".stl", ".obj", ".ply", ".off",
    ".gltf", ".glb",
    ".step", ".stp", ".iges", ".igs", ".brep",
    ".fbx", ".3mf", ".dae",
    ".pcd", ".xyz", ".las", ".laz",
}

# Default allowed domains for downloads
DEFAULT_ALLOWED_DOMAINS = {
    "thingiverse.com",
    "api.thingiverse.com",
    "sketchfab.com",
    "api.sketchfab.com",
    "grabcad.com",
    "cults3d.com",
    "myminifactory.com",
    "thangs.com",
    "github.com",
    "raw.githubusercontent.com",
}


@dataclass
class ValidationResult:
    """Result of input validation."""

    valid: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class InputValidator:
    """Validates file and URL inputs for security."""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a file for safe processing.

        Checks:
        - File exists and is readable
        - Extension is allowed
        - File size is within limits
        - No path traversal

        Args:
            file_path: Path to validate.

        Returns:
            ValidationResult with status and any warnings.
        """
        file_path = Path(file_path)
        warnings = []

        if not self.config.validate_files:
            return ValidationResult(valid=True)

        # Check existence
        if not file_path.exists():
            return ValidationResult(valid=False, error=f"File not found: {file_path}")

        if not file_path.is_file():
            return ValidationResult(valid=False, error=f"Not a file: {file_path}")

        # Check for path traversal
        try:
            resolved = file_path.resolve()
            if ".." in str(file_path):
                warnings.append("Path contains '..', resolved to absolute path")
        except (OSError, ValueError) as e:
            return ValidationResult(valid=False, error=f"Invalid path: {e}")

        # Check extension
        ext = file_path.suffix.lower()
        allowed_extensions = self.config.allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS
        if ext not in allowed_extensions:
            return ValidationResult(
                valid=False,
                error=f"File extension '{ext}' is not allowed. "
                      f"Allowed: {', '.join(sorted(allowed_extensions))}",
            )

        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > self.config.max_file_size:
                size_mb = file_size / (1024 * 1024)
                limit_mb = self.config.max_file_size / (1024 * 1024)
                return ValidationResult(
                    valid=False,
                    error=f"File too large: {size_mb:.1f} MB (limit: {limit_mb:.1f} MB)",
                )
            if file_size == 0:
                return ValidationResult(valid=False, error="File is empty")
        except OSError as e:
            return ValidationResult(valid=False, error=f"Cannot read file: {e}")

        return ValidationResult(valid=True, warnings=warnings)

    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate a URL for safe downloading.

        Checks:
        - URL is well-formed
        - Scheme is HTTPS (or HTTP with warning)
        - Domain is in allowed list
        - No suspicious patterns

        Args:
            url: URL to validate.

        Returns:
            ValidationResult with status and any warnings.
        """
        warnings = []

        if not self.config.validate_urls:
            return ValidationResult(valid=True)

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidationResult(valid=False, error=f"Invalid URL: {e}")

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return ValidationResult(
                valid=False,
                error=f"Unsupported URL scheme: {parsed.scheme}",
            )

        if parsed.scheme == "http":
            warnings.append("URL uses HTTP instead of HTTPS")

        # Check hostname
        if not parsed.hostname:
            return ValidationResult(valid=False, error="URL has no hostname")

        # Check domain allowlist
        allowed_domains = self.config.allowed_domains or DEFAULT_ALLOWED_DOMAINS
        hostname = parsed.hostname.lower()
        domain_allowed = any(
            hostname == d or hostname.endswith(f".{d}")
            for d in allowed_domains
        )
        if not domain_allowed:
            return ValidationResult(
                valid=False,
                error=f"Domain '{hostname}' is not in the allowed list",
            )

        # Check for suspicious patterns
        if parsed.username or parsed.password:
            return ValidationResult(
                valid=False,
                error="URL contains embedded credentials",
            )

        return ValidationResult(valid=True, warnings=warnings)

    def compute_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Compute hash of a file for integrity verification."""
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# Creative Commons license identifiers commonly used by 3D platforms
OPEN_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "public-domain",
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "gpl-3.0",
    "lgpl-3.0",
}

# Non-commercial licenses
NC_LICENSES = {
    "cc-by-nc",
    "cc-by-nc-sa",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-nd",
    "cc-by-nc-nd-4.0",
}

# No-derivatives licenses
ND_LICENSES = {
    "cc-by-nd",
    "cc-by-nd-4.0",
    "cc-by-nc-nd",
    "cc-by-nc-nd-4.0",
}


class LicenseFilter:
    """Filter acquisition results by license compatibility."""

    def __init__(
        self,
        allow_commercial: bool = True,
        allow_derivatives: bool = True,
        allow_share_alike: bool = True,
        allowed_licenses: Optional[Set[str]] = None,
        blocked_licenses: Optional[Set[str]] = None,
    ):
        self.allow_commercial = allow_commercial
        self.allow_derivatives = allow_derivatives
        self.allow_share_alike = allow_share_alike
        self.allowed_licenses = allowed_licenses
        self.blocked_licenses = blocked_licenses or set()

    def is_allowed(self, license_str: str) -> bool:
        """
        Check if a license is allowed under current filter settings.

        Args:
            license_str: License identifier string.

        Returns:
            True if the license is allowed.
        """
        if not license_str:
            return False

        normalized = license_str.lower().strip()

        # Check explicit block list
        if normalized in self.blocked_licenses:
            return False

        # Check explicit allow list
        if self.allowed_licenses is not None:
            return normalized in self.allowed_licenses

        # Check non-commercial restriction
        if not self.allow_commercial and normalized in NC_LICENSES:
            return False

        # Check no-derivatives restriction
        if not self.allow_derivatives and normalized in ND_LICENSES:
            return False

        # Allow open licenses by default
        if normalized in OPEN_LICENSES:
            return True

        # Allow NC licenses if commercial use is not required
        if normalized in NC_LICENSES and not self.allow_commercial:
            return True

        # Unknown license - allow by default but log warning
        logger.warning(f"Unknown license: {license_str}")
        return True

    def filter_results(self, results: list) -> list:
        """
        Filter a list of search results by license.

        Args:
            results: List of SearchResult objects with license attribute.

        Returns:
            Filtered list of allowed results.
        """
        filtered = []
        for result in results:
            license_str = getattr(result, "license", None) or ""
            if self.is_allowed(license_str):
                filtered.append(result)
            else:
                logger.debug(
                    f"Filtered out {getattr(result, 'model_id', 'unknown')} "
                    f"due to license: {license_str}"
                )
        return filtered
