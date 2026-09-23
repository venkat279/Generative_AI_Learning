"""Exact-match caching: key by a hash of the prompt, TTL expiry, 
in-memory dict instead of Redis.
"""

import hashlib
import time


class ExactCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[str, float]] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    def get(self, prompt: str) -> str | None:
        entry = self._store.get(self._key(prompt))
        if entry is None:
            self.misses += 1
            return None
        value, stored_at = entry
        if time.time() - stored_at > self.ttl_seconds:
            del self._store[self._key(prompt)]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, prompt: str, value: str) -> None:
        self._store[self._key(prompt)] = (value, time.time())

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0
