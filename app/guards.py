"""Abuse protection: per-client rate limit, answer-slot limit, answer cache."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class RateLimiter:
    """Sliding window: at most `limit` requests per `window_s` seconds per client."""

    def __init__(self, limit: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, max_clients: int = 10_000):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float | None:
        """Record one request. Returns None if allowed, else seconds until retry."""
        with self._lock:
            now = self._clock()
            hits = self._hits.get(client)
            if hits is None:
                if len(self._hits) >= self._max_clients:
                    self._forget_idle(now)
                hits = self._hits[client] = deque()
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window_s - (now - hits[0])
            hits.append(now)
            return None

    def _forget_idle(self, now: float) -> None:
        """Drop clients with no requests inside the window (keeps memory bounded)."""
        idle = [c for c, h in self._hits.items() if not h or now - h[-1] >= self.window_s]
        for c in idle:
            del self._hits[c]


class SlotLimiter:
    """At most `limit` questions answered at once. Extra requests are turned away, not queued."""

    def __init__(self, limit: int):
        self.limit = limit
        self._sem = threading.BoundedSemaphore(limit)

    def try_acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


def normalize(question: str) -> str:
    """Cache key: case, extra spaces and trailing punctuation don't matter."""
    return " ".join(question.casefold().split()).rstrip("?!. ")


class AnswerCache(Generic[T]):
    """Small LRU cache of answers, keyed by the normalized question."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, question: str) -> T | None:
        key = normalize(question)
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, question: str, value: T) -> None:
        if self.size <= 0:
            return
        key = normalize(question)
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)
