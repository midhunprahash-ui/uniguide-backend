"""
Event discovery cache with Supabase persistence and in-memory fallback.
"""
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


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
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_supabase_admin_client()
        return self._client

    def _compute_key(
        self,
        query: str,
        org_id: str,
        nearby: bool,
        nearby_location: str | None,
        nearby_lat: float | None,
        nearby_lng: float | None,
    ) -> str:
        normalized = query.lower().strip()
        loc = (nearby_location or "").lower().strip()
        lat = "" if nearby_lat is None else f"{nearby_lat:.6f}"
        lng = "" if nearby_lng is None else f"{nearby_lng:.6f}"
        key_parts = f"{org_id}:{normalized}:{nearby}:{loc}:{lat}:{lng}"
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

    def _parse_timestamp(self, value: str | None) -> datetime:
        if not value:
            return datetime.now()
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None)
        except Exception:
            return datetime.now()

    def _get_memory(self, cache_key: str) -> Optional[CachedEvents]:
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

    def _set_memory(self, cache_key: str, cached: CachedEvents) -> None:
        with self._lock:
            self._evict_lru()
            self._cache[cache_key] = cached

    def _get_supabase(self, cache_key: str) -> Optional[CachedEvents]:
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = (
                self.client.table("event_discover_cache")
                .select("*")
                .eq("cache_key", cache_key)
                .gt("expires_at", now)
                .limit(1)
                .execute()
            )
            if not result.data:
                return None
            row = result.data[0]

            self.client.table("event_discover_cache").update({
                "hit_count": int(row.get("hit_count", 0)) + 1,
                "last_hit_at": now,
            }).eq("id", row.get("id")).execute()

            cached = CachedEvents(
                events=row.get("events", []) or [],
                citations=row.get("citations", []) or [],
                cached_at=self._parse_timestamp(row.get("created_at")),
                cache_key=cache_key,
                hit_count=int(row.get("hit_count", 0)) + 1,
                last_hit_at=datetime.now(),
            )
            return cached
        except Exception as e:
            logger.warning("Supabase event cache read failed: %s", e)
            return None

    def get(
        self,
        query: str,
        org_id: str,
        nearby: bool,
        nearby_location: str | None,
        nearby_lat: float | None,
        nearby_lng: float | None,
    ) -> Optional[CachedEvents]:
        cache_key = self._compute_key(query, org_id, nearby, nearby_location, nearby_lat, nearby_lng)

        cached = self._get_supabase(cache_key)
        if cached:
            self._set_memory(cache_key, cached)
            return cached

        return self._get_memory(cache_key)

    def set(
        self,
        query: str,
        org_id: str,
        nearby: bool,
        nearby_location: str | None,
        nearby_lat: float | None,
        nearby_lng: float | None,
        events: list[dict],
        citations: list[dict],
    ) -> None:
        cache_key = self._compute_key(query, org_id, nearby, nearby_location, nearby_lat, nearby_lng)
        now = datetime.now(timezone.utc)
        expires_at = now + self.ttl

        payload = {
            "org_id": org_id,
            "question": query,
            "nearby": nearby,
            "nearby_location": nearby_location,
            "nearby_lat": nearby_lat,
            "nearby_lng": nearby_lng,
            "cache_key": cache_key,
            "citations": citations,
            "events": events,
            "expires_at": expires_at.isoformat(),
            "last_hit_at": now.isoformat(),
        }

        try:
            self.client.table("event_discover_cache").upsert(
                payload,
                on_conflict="cache_key",
            ).execute()
        except Exception as e:
            logger.warning("Supabase event cache write failed: %s", e)

        self._set_memory(
            cache_key,
            CachedEvents(
                events=events,
                citations=citations,
                cached_at=datetime.now(),
                cache_key=cache_key,
            ),
        )


event_cache = EventCache()
