"""
Exception hierarchy for Destill3D.

All Destill3D exceptions inherit from Destill3DError for easy catching.
"""

from typing import Optional


class Destill3DError(Exception):
    """Base exception for all Destill3D errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Errors
# ─────────────────────────────────────────────────────────────────────────────


class ConfigurationError(Destill3DError):
    """Configuration-related errors."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Extraction Errors
# ─────────────────────────────────────────────────────────────────────────────


class ExtractionError(Destill3DError):
    """Error during feature extraction."""

    pass


class FormatError(ExtractionError):
    """Unsupported or malformed file format."""

    def __init__(self, format: str, details: str):
        self.format = format
        super().__init__(f"Format error ({format})", details)


class GeometryError(ExtractionError):
    """Invalid or degenerate geometry."""

    def __init__(self, issue: str):
        self.issue = issue
        super().__init__("Geometry error", issue)


class TessellationError(ExtractionError):
    """B-rep tessellation failed."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(f"Tessellation failed: {message}", details)


class SamplingError(ExtractionError):
    """Point cloud sampling failed."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(f"Sampling failed: {message}", details)


class NormalizationError(ExtractionError):
    """Point cloud normalization failed."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(f"Normalization failed: {message}", details)


# ─────────────────────────────────────────────────────────────────────────────
# Classification Errors
# ─────────────────────────────────────────────────────────────────────────────


class ClassificationError(Destill3DError):
    """Error during classification."""

    pass


class ModelNotFoundError(ClassificationError):
    """Requested model not available."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class ModelDownloadError(ClassificationError):
    """Failed to download model weights."""

    def __init__(self, model_id: str, url: str, details: Optional[str] = None):
        self.model_id = model_id
        self.url = url
        super().__init__(f"Failed to download model '{model_id}' from {url}", details)


class InferenceError(ClassificationError):
    """Model inference failed."""

    def __init__(self, model_id: str, details: Optional[str] = None):
        self.model_id = model_id
        super().__init__(f"Inference failed for model '{model_id}'", details)


class TaxonomyError(ClassificationError):
    """Unknown or invalid taxonomy."""

    def __init__(self, taxonomy: str):
        self.taxonomy = taxonomy
        super().__init__(f"Unknown taxonomy: {taxonomy}")


# ─────────────────────────────────────────────────────────────────────────────
# Database Errors
# ─────────────────────────────────────────────────────────────────────────────


class DatabaseError(Destill3DError):
    """Database operation failed."""

    pass


class SnapshotNotFoundError(DatabaseError):
    """Snapshot not found in database."""

    def __init__(self, snapshot_id: str):
        self.snapshot_id = snapshot_id
        super().__init__(f"Snapshot not found: {snapshot_id}")


class DuplicateSnapshotError(DatabaseError):
    """Snapshot already exists in database."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Snapshot already exists for model: {model_id}")


class ExportError(DatabaseError):
    """Export operation failed."""

    def __init__(self, format: str, details: Optional[str] = None):
        self.format = format
        super().__init__(f"Export to {format} failed", details)


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition Errors (for future platform adapters)
# ─────────────────────────────────────────────────────────────────────────────


class AcquisitionError(Destill3DError):
    """Error during model acquisition from platforms."""

    pass


class DownloadError(AcquisitionError):
    """Failed to download a file."""

    def __init__(self, url: str, reason: str, details: Optional[str] = None):
        self.url = url
        self.reason = reason
        super().__init__(f"Download failed for {url}: {reason}", details)


class RateLimitError(AcquisitionError):
    """Rate limit exceeded on platform."""

    def __init__(self, platform: str, retry_after: Optional[int] = None):
        self.platform = platform
        self.retry_after = retry_after
        details = f"Retry after {retry_after}s" if retry_after else None
        super().__init__(f"Rate limit exceeded on {platform}", details)


class AuthenticationError(AcquisitionError):
    """Authentication failed for platform."""

    def __init__(self, platform: str, details: Optional[str] = None):
        self.platform = platform
        super().__init__(f"Authentication failed for {platform}", details)


class PlatformError(AcquisitionError):
    """Platform-specific error."""

    def __init__(self, platform: str, message: str, details: Optional[str] = None):
        self.platform = platform
        super().__init__(f"{platform}: {message}", details)
