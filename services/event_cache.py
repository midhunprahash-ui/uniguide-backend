"""
Simple in-memory cache for event discovery results.
"""
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class CachedEvents:
    events: list[dict]
    citations: list[dict]
    cached_at: datetime
    cache_key: str
    hit_count: int = 0
    last_hit_at: datetime = field(default_factory=datetime.now)


class EventCache:
    def __init__(self, ttl_hours: int = 6, max_entries: int = 1000):
        self.ttl = timedelta(hours=ttl_hours)
        self.max_entries = max_entries
        self._cache: dict[str, CachedEvents] = {}
        self._lock = threading.RLock()

    def _compute_key(self, query: str, org_id: str, nearby: bool, nearby_location: str | None) -> str:
        normalized = query.lower().strip()
        loc = (nearby_location or "").lower().strip()
        key_parts = f"{org_id}:{normalized}:{nearby}:{loc}"
        return hashlib.sha256(key_parts.encode()).hexdigest()[:24]

    def _is_expired(self, cached: CachedEvents) -> bool:
        return datetime.now() - cached.cached_at >= self.ttl

    def _evict_lru(self) -> None:
        if len(self._cache) < self.max_entries:
            return
        entries_to_remove = len(self._cache) - self.max_entries + 100
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_hit_at)
        for cache_key, _ in sorted_entries[:entries_to_remove]:
            del self._cache[cache_key]

    def get(self, query: str, org_id: str, nearby: bool, nearby_location: str | None) -> Optional[CachedEvents]:
        cache_key = self._compute_key(query, org_id, nearby, nearby_location)
        with self._lock:
            cached = self._cache.get(cache_key)
            if not cached:
                return None
            if self._is_expired(cached):
                del self._cache[cache_key]
                return None
            cached.hit_count += 1
            cached.last_hit_at = datetime.now()
            return cached

    def set(self, query: str, org_id: str, nearby: bool, nearby_location: str | None,
            events: list[dict], citations: list[dict]) -> None:
        cache_key = self._compute_key(query, org_id, nearby, nearby_location)
        with self._lock:
            self._evict_lru()
            self._cache[cache_key] = CachedEvents(
                events=events,
                citations=citations,
                cached_at=datetime.now(),
                cache_key=cache_key,
            )


event_cache = EventCache()
