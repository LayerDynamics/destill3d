"""Unit tests for TokenBucket and RateLimiter."""

import asyncio
import time

import pytest

from destill3d.acquire.base import RateLimit
from destill3d.acquire.rate_limiter import RateLimiter, TokenBucket


class TestTokenBucket:
    def test_creation(self):
        bucket = TokenBucket(capacity=10, refill_period=60.0)
        assert bucket.capacity == 10
        assert bucket.refill_period == 60.0

    def test_consume_success(self):
        bucket = TokenBucket(capacity=5, refill_period=60.0)
        assert bucket.consume() == 0.0  # 0.0 means tokens available

    def test_consume_depletes(self):
        bucket = TokenBucket(capacity=2, refill_period=60.0)
        assert bucket.consume() == 0.0
        assert bucket.consume() == 0.0
        assert bucket.consume() > 0  # Must wait

    def test_available(self):
        bucket = TokenBucket(capacity=3, refill_period=60.0)
        assert bucket.available == 3
        bucket.consume()
        assert bucket.available == 2

    def test_refill(self):
        bucket = TokenBucket(capacity=10, refill_period=0.1)
        for _ in range(10):
            bucket.consume()
        assert bucket.available < 1.0
        time.sleep(0.15)
        bucket._refill()
        assert bucket.available > 0


class TestRateLimiter:
    def test_configure(self):
        rl = RateLimiter()
        rl.configure("test", RateLimit(requests=100, period_seconds=60))
        assert rl.available("test") > 0

    def test_available_unconfigured(self):
        rl = RateLimiter()
        assert rl.available("unknown") == float("inf")

    @pytest.mark.asyncio
    async def test_acquire(self):
        rl = RateLimiter()
        rl.configure("test", RateLimit(requests=5, period_seconds=60))
        await rl.acquire("test")
        assert rl.available("test") < 5
