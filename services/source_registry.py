"""
Trusted source registry for Event Discover.

Primary source: Supabase table `event_source_registry`.
Fallback source: built-in defaults for resiliency.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

STRICT_TRUST_TIER_MAX = 2


@dataclass(frozen=True)
class TrustedSource:
    domain: str
    trust_tier: int = 2
    weight: int = 0
    categories: tuple[str, ...] = field(default_factory=tuple)
    country: str | None = None
    is_active: bool = True
    requires_js: bool = False
    metadata: dict = field(default_factory=dict)

    def matches_category(self, category: str) -> bool:
        if not category or category in {"all", "general_student_opportunity", "unknown"}:
            return True
        if not self.categories:
            return True
        return category in self.categories


DEFAULT_SOURCES: tuple[TrustedSource, ...] = (
    TrustedSource("devpost.com", trust_tier=1, weight=8, categories=("hackathons",)),
    TrustedSource("mlh.io", trust_tier=1, weight=8, categories=("hackathons",)),
    TrustedSource("unstop.com", trust_tier=1, weight=8, categories=("hackathons", "internships", "conferences", "jobs")),
    TrustedSource("ieee.org", trust_tier=1, weight=7, categories=("conferences", "scholarships")),
    TrustedSource("acm.org", trust_tier=1, weight=7, categories=("conferences",)),
    TrustedSource("gdg.community.dev", trust_tier=1, weight=6, categories=("hackathons", "conferences")),
    TrustedSource("developers.google.com", trust_tier=1, weight=6, categories=("hackathons", "conferences")),
    TrustedSource("internshala.com", trust_tier=2, weight=6, categories=("internships", "jobs")),
    TrustedSource("linkedin.com", trust_tier=2, weight=5, categories=("internships", "jobs")),
    TrustedSource("wellfound.com", trust_tier=2, weight=5, categories=("internships", "jobs")),
    TrustedSource("angel.co", trust_tier=2, weight=4, categories=("internships", "jobs")),
    TrustedSource("hackerearth.com", trust_tier=2, weight=6, categories=("hackathons",)),
    TrustedSource("hackerrank.com", trust_tier=2, weight=5, categories=("hackathons",)),
    TrustedSource("kaggle.com", trust_tier=2, weight=5, categories=("hackathons",)),
    TrustedSource("codechef.com", trust_tier=2, weight=4, categories=("hackathons",)),
    TrustedSource("topcoder.com", trust_tier=2, weight=4, categories=("hackathons",)),
    TrustedSource("aicte-india.org", trust_tier=1, weight=6, categories=("internships", "jobs", "scholarships")),
    TrustedSource("scholarships.gov.in", trust_tier=1, weight=7, categories=("scholarships",)),
    TrustedSource("mygov.in", trust_tier=1, weight=5, categories=("scholarships", "jobs")),
    TrustedSource("startupindia.gov.in", trust_tier=1, weight=5, categories=("internships", "jobs")),
    TrustedSource("daad.de", trust_tier=1, weight=6, categories=("scholarships",)),
    TrustedSource("mitacs.ca", trust_tier=1, weight=6, categories=("scholarships",)),
    TrustedSource("fulbrightonline.org", trust_tier=1, weight=6, categories=("scholarships",)),
    TrustedSource("fulbright.org", trust_tier=1, weight=6, categories=("scholarships",)),
    TrustedSource("unesco.org", trust_tier=1, weight=5, categories=("scholarships", "conferences")),
    TrustedSource("un.org", trust_tier=1, weight=5, categories=("scholarships", "conferences")),
    TrustedSource("careers.un.org", trust_tier=1, weight=5, categories=("internships", "jobs")),
    TrustedSource("youth.un.org", trust_tier=1, weight=5, categories=("conferences", "scholarships")),
    TrustedSource("ycombinator.com", trust_tier=2, weight=3, categories=("jobs", "internships")),
    TrustedSource("techstars.com", trust_tier=2, weight=3, categories=("jobs", "internships")),
    TrustedSource("t-hub.co", trust_tier=2, weight=3, categories=("internships", "jobs")),
    TrustedSource("thub.co.in", trust_tier=2, weight=3, categories=("internships", "jobs")),
)


def _normalize_domain(value: str) -> str:
    return (value or "").strip().lower().replace("www.", "")


def _extract_domain(url_or_domain: str) -> str:
    if "://" in url_or_domain:
        try:
            return _normalize_domain(urlparse(url_or_domain).netloc)
        except Exception:
            return ""
    return _normalize_domain(url_or_domain)


class SourceRegistry:
    def __init__(self, ttl_minutes: int = 5):
        self.ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.RLock()
        self._sources: tuple[TrustedSource, ...] = DEFAULT_SOURCES
        self._loaded_at = datetime.min.replace(tzinfo=UTC)
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            # Lazy import keeps fallback mode available even if Supabase deps are unavailable.
            from services.supabase_client import get_supabase_admin_client

            self._client = get_supabase_admin_client()
        return self._client

    def _is_fresh(self) -> bool:
        now = datetime.now(UTC)
        return now - self._loaded_at < self.ttl

    def _row_to_source(self, row: dict) -> TrustedSource | None:
        domain = _normalize_domain(row.get("domain") or "")
        if not domain:
            return None
        categories = tuple(
            c.strip().lower()
            for c in (row.get("categories") or [])
            if isinstance(c, str) and c.strip()
        )
        return TrustedSource(
            domain=domain,
            trust_tier=int(row.get("trust_tier") or 2),
            weight=int(row.get("weight") or 0),
            categories=categories,
            country=row.get("country"),
            is_active=bool(row.get("is_active", True)),
            requires_js=bool(row.get("requires_js", False)),
            metadata=row.get("metadata") or {},
        )

    def _load_from_db(self) -> tuple[TrustedSource, ...]:
        try:
            result = (
                self.client.table("event_source_registry")
                .select(
                    "domain,trust_tier,weight,categories,country,is_active,requires_js,metadata"
                )
                .eq("is_active", True)
                .execute()
            )
            rows = result.data or []
            sources = [self._row_to_source(row) for row in rows]
            active_sources = tuple(
                sorted(
                    (s for s in sources if s and s.is_active),
                    key=lambda s: (s.trust_tier, -s.weight, s.domain),
                )
            )
            if active_sources:
                return active_sources
        except Exception as e:
            logger.warning("Source registry DB load failed, using fallback list: %s", e)
        return DEFAULT_SOURCES

    def list_active_sources(self, force_refresh: bool = False) -> tuple[TrustedSource, ...]:
        with self._lock:
            if force_refresh or not self._is_fresh():
                self._sources = self._load_from_db()
                self._loaded_at = datetime.now(UTC)
            return self._sources

    def get_include_domains(
        self,
        category_hint: str,
        strict_trust: bool,
        max_domains: int = 40,
    ) -> list[str]:
        sources = self.list_active_sources()

        filtered: Iterable[TrustedSource] = sources
        if strict_trust:
            filtered = [s for s in filtered if s.trust_tier <= STRICT_TRUST_TIER_MAX]

        filtered = [s for s in filtered if s.matches_category(category_hint)]
        if not filtered:
            filtered = list(sources)

        return [s.domain for s in list(filtered)[:max_domains]]

    def score_url(
        self,
        url: str,
        category_hint: str,
        strict_trust: bool,
    ) -> tuple[bool, int, int, list[str]]:
        domain = _extract_domain(url)
        if not domain:
            return False, -100, 99, ["invalid_domain"]

        sources = self.list_active_sources()
        matched = next(
            (s for s in sources if domain == s.domain or domain.endswith(f".{s.domain}")),
            None,
        )

        if not matched:
            if strict_trust:
                return False, -100, 99, ["domain_not_trusted"]
            return True, 1, 3, ["domain_unlisted"]

        if strict_trust and matched.trust_tier > STRICT_TRUST_TIER_MAX:
            return False, -50, matched.trust_tier, [f"trust_tier_{matched.trust_tier}"]

        score = max(0, matched.weight) + max(0, 4 - matched.trust_tier) * 2
        flags = [f"trust_tier_{matched.trust_tier}"]
        if matched.matches_category(category_hint):
            score += 2
            flags.append("category_aligned")
        elif category_hint not in {"all", "general_student_opportunity", "unknown"}:
            score -= 1
            flags.append("category_mismatch")

        return True, score, matched.trust_tier, flags


source_registry = SourceRegistry()
