from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from loguru import logger


class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: float) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = elapsed * self.rate
            if refill > 0:
                self._tokens = min(self.capacity, self._tokens + refill)
                self._last_refill = now
            if tokens <= self._tokens:
                self._tokens -= tokens
                return True
            return False


class RateLimiter:
    """Simple token bucket rate limiter keyed by endpoint name."""

    def __init__(self, rate_per_minute: float, capacity: Optional[float] = None) -> None:
        capacity = capacity if capacity is not None else rate_per_minute
        self.bucket = TokenBucket(rate_per_minute / 60.0, capacity)

    async def acquire(self, weight: float = 1.0, max_wait_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            allowed = await self.bucket.consume(weight)
            if allowed:
                return True
            await asyncio.sleep(0.05)
        logger.warning("RateLimiter timeout after waiting for {wait}s", wait=max_wait_s)
        return False


class EndpointRateLimiter:
    def __init__(self, default_rate: float, capacities: Optional[Dict[str, float]] = None) -> None:
        self.default_rate = default_rate
        self.capacities = capacities or {}
        self.buckets: Dict[str, RateLimiter] = {}

    def get_limiter(self, endpoint: str) -> RateLimiter:
        if endpoint not in self.buckets:
            capacity = self.capacities.get(endpoint, self.default_rate)
            self.buckets[endpoint] = RateLimiter(rate_per_minute=self.default_rate, capacity=capacity)
        return self.buckets[endpoint]

    async def acquire(self, endpoint: str, weight: float = 1.0, max_wait_s: float = 5.0) -> bool:
        limiter = self.get_limiter(endpoint)
        return await limiter.acquire(weight=weight, max_wait_s=max_wait_s)
