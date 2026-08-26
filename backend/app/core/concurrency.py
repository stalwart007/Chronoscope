"""Concurrency primitives: bounded fan-out, keyed limiting, caching, breakers.

The pipeline mixes network I/O, CPU-bound numeric work and memory-hungry model
inference. Each needs different backpressure, and none should block the event
loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

_CPU_POOL: ThreadPoolExecutor | None = None


def cpu_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """Shared pool for blocking work. Lazily created; threads (not processes)
    because the hot paths release the GIL inside numpy/OpenCV/torch."""
    global _CPU_POOL
    if _CPU_POOL is None:
        _CPU_POOL = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cs-cpu")
    return _CPU_POOL


async def run_blocking(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(cpu_pool(), lambda: fn(*args, **kwargs))


class KeyedLimiter:
    """One semaphore per key, e.g. serialise work per video without blocking
    other videos."""

    def __init__(self, limit: int = 1) -> None:
        self._limit = limit
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._refs: dict[str, int] = {}
        self._guard = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(self, key: str):  # type: ignore[no-untyped-def]
        async with self._guard:
            sem = self._sems.setdefault(key, asyncio.Semaphore(self._limit))
            self._refs[key] = self._refs.get(key, 0) + 1
        try:
            async with sem:
                yield
        finally:
            async with self._guard:
                self._refs[key] -= 1
                if self._refs[key] <= 0:
                    self._refs.pop(key, None)
                    self._sems.pop(key, None)


@dataclass(slots=True)
class TokenBucket:
    """Classic token bucket, smooths bursts against free-tier LLM rate limits."""

    rate: float
    capacity: float
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity

    async def take(self, n: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                await asyncio.sleep(max(0.01, (n - self._tokens) / max(self.rate, 1e-6)))


class TTLCache(Generic[T]):
    """Thread-safe LRU + TTL cache with single-flight coalescing.

    Concurrent misses on the same key await one shared task instead of firing N
    identical model calls, so repeated queries cost one embedding pass.
    """

    def __init__(self, maxsize: int = 512, ttl: float = 900.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[T]] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    def _get_fresh(self, key: str) -> T | None:
        item = self._data.get(key)
        if item is None:
            return None
        ts, val = item
        if time.monotonic() - ts > self.ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return val

    async def get_or_set(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            hit = self._get_fresh(key)
            if hit is not None:
                self.hits += 1
                return hit
            fut = self._inflight.get(key)
            if fut is None:
                self.misses += 1
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = fut
                owner = True
            else:
                owner = False
        if not owner:
            return await asyncio.shield(fut)
        try:
            value = await factory()
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(exc)
            raise
        async with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            self._inflight.pop(key, None)
        if not fut.done():
            fut.set_result(value)
        return value

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def clear(self) -> None:
        self._data.clear()


class CircuitBreaker:
    """Three-state breaker (closed -> open -> half-open).

    Keeps a dead LLM provider from burning the per-request latency budget on
    every single call: after ``threshold`` consecutive failures the circuit
    opens for ``cooldown`` seconds, then admits one probe request.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at = 0.0
        self._half_open = False

    @property
    def state(self) -> str:
        if self._failures < self.threshold:
            return "closed"
        if time.monotonic() - self._opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open" and not self._half_open:
            self._half_open = True
            return True
        return st == "half_open"

    def record_success(self) -> None:
        self._failures = 0
        self._half_open = False

    def record_failure(self) -> None:
        self._failures += 1
        self._half_open = False
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Exponential backoff with decorrelated jitter."""
    import random

    last: BaseException | None = None
    for i in range(max(1, attempts)):
        try:
            return await fn()
        except retry_on as exc:
            last = exc
            if i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2**i))
            await asyncio.sleep(delay * (1.0 + random.uniform(-jitter, jitter)))
    assert last is not None
    raise last
