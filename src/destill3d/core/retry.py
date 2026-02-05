"""
Retry infrastructure for Destill3D.

Provides configurable retry logic for network operations,
extraction, and classification tasks.
"""

from dataclasses import dataclass
import logging
from typing import Tuple, Type

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry configuration for various operations."""

    # Download retry settings
    download_attempts: int = 3
    download_wait_min: int = 1
    download_wait_max: int = 60

    # Extraction retry settings
    extraction_attempts: int = 2
    extraction_wait_min: int = 1
    extraction_wait_max: int = 10

    # Classification retry settings
    classification_attempts: int = 2
    classification_wait_min: int = 1
    classification_wait_max: int = 10


def with_retry(
    attempts: int = 3,
    wait_min: int = 1,
    wait_max: int = 60,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator factory for retry logic using tenacity.

    Args:
        attempts: Maximum number of retry attempts.
        wait_min: Minimum wait time between retries (seconds).
        wait_max: Maximum wait time between retries (seconds).
        exceptions: Tuple of exception types to retry on.

    Returns:
        A tenacity retry decorator.

    Usage:
        @with_retry(attempts=3, exceptions=(ConnectionError, TimeoutError))
        async def download_file(url):
            ...
    """
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def download_retry(config: RetryConfig = None):
    """Get a retry decorator configured for download operations."""
    config = config or RetryConfig()
    from destill3d.core.exceptions import AcquisitionError, RateLimitError
    return with_retry(
        attempts=config.download_attempts,
        wait_min=config.download_wait_min,
        wait_max=config.download_wait_max,
        exceptions=(AcquisitionError, RateLimitError, ConnectionError, TimeoutError),
    )


def extraction_retry(config: RetryConfig = None):
    """Get a retry decorator configured for extraction operations."""
    config = config or RetryConfig()
    from destill3d.core.exceptions import ExtractionError
    return with_retry(
        attempts=config.extraction_attempts,
        wait_min=config.extraction_wait_min,
        wait_max=config.extraction_wait_max,
        exceptions=(ExtractionError,),
    )


def classification_retry(config: RetryConfig = None):
    """Get a retry decorator configured for classification operations."""
    config = config or RetryConfig()
    from destill3d.core.exceptions import ClassificationError
    return with_retry(
        attempts=config.classification_attempts,
        wait_min=config.classification_wait_min,
        wait_max=config.classification_wait_max,
        exceptions=(ClassificationError,),
    )
