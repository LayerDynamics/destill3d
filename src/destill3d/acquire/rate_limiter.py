"""
Rate limiting infrastructure for platform API calls.

Implements a token bucket algorithm with per-platform limits.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict

from destill3d.acquire.base import RateLimit


class TokenBucket:
    """Token bucket rate limiting implementation."""

    def __init__(self, capacity: int, refill_period: float):
        self.capacity = capacity
        self.refill_period = refill_period
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> float:
        """
        Consume tokens, return wait time if insufficient.

        Returns:
            0.0 if tokens were available, or seconds to wait.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0

        # Calculate wait time for enough tokens
        needed = tokens - self.tokens
        wait_time = (needed / self.capacity) * self.refill_period
        return wait_time

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = (elapsed / self.refill_period) * self.capacity
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    @property
    def available(self) -> int:
        """Get current available tokens."""
        self._refill()
        return int(self.tokens)


class RateLimiter:
    """Token bucket rate limiter with per-platform limits."""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def configure(self, platform: str, rate_limit: RateLimit) -> None:
        """Configure rate limit for a platform."""
        self._buckets[platform] = TokenBucket(
            capacity=rate_limit.requests,
            refill_period=rate_limit.period_seconds,
        )

    async def acquire(self, platform: str, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns:
            Wait time in seconds (0.0 if no wait needed).
        """
        async with self._locks[platform]:
            bucket = self._buckets.get(platform)
            if not bucket:
                return 0.0

            wait_time = bucket.consume(tokens)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            return wait_time

    def available(self, platform: str) -> int:
        """Get available tokens for platform."""
        bucket = self._buckets.get(platform)
        return bucket.available if bucket else float("inf")
