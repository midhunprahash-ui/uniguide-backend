"""
Event discovery service using Tavily + Gemini.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Tuple
from urllib.parse import urlparse

import google.generativeai as genai

from config import get_settings
from services.query_processor import query_processor

logger = logging.getLogger(__name__)
settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)

try:
    from tavily import TavilyClient
except Exception as e:  # pragma: no cover - dependency validation handled at runtime
    TavilyClient = None
    logger.warning("Tavily client import failed: %s", e)


@dataclass
class DiscoverCitation:
    url: str
    title: str | None
    domain: str | None


@dataclass
class DiscoverEvent:
    name: str
    date: str | None
    start_date: str | None
    end_date: str | None
    location: str | None
    cash_prize: str | None
    short_description: str | None
    url: str
    source_url: str | None
    status: str | None


SEARCH_PROMPT = """
You craft precise web search queries for student events.

User request: "{question}"
Nearby: {nearby}
Nearby location: {nearby_location}

Return JSON only with keys:
- query: a concise search query (8-12 words)
- must_include: list of short phrases to include
- exclude: list of short phrases to exclude

Rules:
- Focus on college/university events: hackathons, symposiums, workshops, competitions, fests.
- If nearby is true and location is provided, include it (e.g., "in {nearby_location}").
- If nearby is true and location is empty, include "near me" or "local".
- Avoid quotes. No markdown.
"""

EXTRACT_PROMPT = """
Extract up to 3 student-relevant events from the page content below.
Return ONLY a valid JSON array. If no events, return [].

For each event, include:
- name
- date (string as written on the page)
- start_date (YYYY-MM-DD, required if date is known)
- end_date (YYYY-MM-DD, optional; include if multi-day)
- location
- cash_prize (string, include currency if present)
- short_description (max 200 chars)
- url (event page if explicit, otherwise use source_url)

Prefer events that match the user's intent: {question}

source_url: {source_url}

content:
"""

EVENT_KEYWORDS = [
    "hackathon", "symposium", "workshop", "competition", "conference",
    "summit", "seminar", "fest", "festival", "challenge", "meetup",
    "conclave", "bootcamp", "code", "coding", "innovation", "research",
    "poster", "paper", "call for papers", "techfest", "startup",
]

ALLOWED_DOMAINS = [
    # Global / International
    "devpost.com",
    "mlh.io",
    "linkedin.com",
    "angel.co",
    "wellfound.com",
    "unesco.org",
    "un.org",
    "careers.un.org",
    "youth.un.org",
    "developers.google.com",
    "gdg.community.dev",
    # India-focused
    "unstop.com",
    "aicte-india.org",
    "scholarships.gov.in",
    "internshala.com",
    "mygov.in",
    "startupindia.gov.in",
    # Tech-centric
    "hackerearth.com",
    "hackerrank.com",
    "kaggle.com",
    "codechef.com",
    "topcoder.com",
    # Research / Fellowships
    "daad.de",
    "mitacs.ca",
    "fulbrightonline.org",
    "fulbright.org",
    "ieee.org",
    "acm.org",
    # Startup / Innovation
    "ycombinator.com",
    "techstars.com",
    "t-hub.co",
    "thub.co.in",
]

DOMAIN_WEIGHTS = {
    "devpost.com": 4,
    "mlh.io": 4,
    "unstop.com": 4,
    "hackerearth.com": 3,
    "hackerrank.com": 3,
    "kaggle.com": 3,
    "codechef.com": 3,
    "topcoder.com": 3,
    "internshala.com": 3,
    "linkedin.com": 2,
    "wellfound.com": 2,
    "angel.co": 2,
    "aicte-india.org": 3,
    "scholarships.gov.in": 3,
    "mygov.in": 3,
    "startupindia.gov.in": 3,
    "unesco.org": 3,
    "un.org": 3,
    "careers.un.org": 3,
    "youth.un.org": 3,
    "developers.google.com": 3,
    "gdg.community.dev": 3,
    "daad.de": 3,
    "mitacs.ca": 3,
    "fulbrightonline.org": 3,
    "fulbright.org": 3,
    "ieee.org": 3,
    "acm.org": 3,
    "ycombinator.com": 2,
    "techstars.com": 2,
    "t-hub.co": 2,
    "thub.co.in": 2,
}

INTENT_DOMAIN_BOOSTS = [
    (["hackathon", "competition", "contest", "challenge"], ["devpost.com", "mlh.io", "unstop.com", "hackerearth.com", "hackerrank.com", "codechef.com", "topcoder.com"], 3),
    (["internship", "intern", "role", "job", "opportunity"], ["internshala.com", "linkedin.com", "wellfound.com", "angel.co", "aicte-india.org", "startupindia.gov.in"], 3),
    (["fellowship", "scholarship", "research"], ["daad.de", "mitacs.ca", "fulbrightonline.org", "fulbright.org", "scholarships.gov.in", "unesco.org", "un.org"], 3),
    (["conference", "symposium", "seminar"], ["ieee.org", "acm.org", "gdg.community.dev", "developers.google.com"], 2),
]

MONTHS_REGEX = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)

DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS_REGEX}\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{MONTHS_REGEX}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b", re.IGNORECASE),
]

LOCATION_PATTERNS = [
    re.compile(r"(?:venue|location|place)\s*[:\-]\s*([A-Z][^,\n]{3,80})"),
    re.compile(r"\b(?:at|in)\s+([A-Z][A-Za-z0-9&'().\- ]{3,80})"),
]

PRIZE_PATTERNS = [
    re.compile(r"(?:cash\s+prize|prize\s+pool|prize)\s*[:\-]?\s*(₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\s?(?:k|K|lakhs|lakh|L|crore|cr))?", re.IGNORECASE),
    re.compile(r"(₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\s?(?:k|K|lakhs|lakh|L|crore|cr))?", re.IGNORECASE),
]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = text.rstrip("`").rstrip()
    return text.strip()


def _safe_json_loads(text: str, fallback):
    try:
        return json.loads(_strip_code_fences(text))
    except Exception:
        return fallback


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_first_match(patterns: list[re.Pattern], text: str) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(0).strip()
            return value
    return None


def _extract_location(text: str) -> Optional[str]:
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = (match.group(1) or "").strip()
        if "http" in value.lower():
            continue
        return value
    return None


def _extract_prize(text: str) -> Optional[str]:
    for pattern in PRIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def _clean_title(title: str) -> str:
    for sep in [" | ", " - ", " — ", " – "]:
        if sep in title:
            return title.split(sep)[0].strip()
    return title.strip()


def _is_event_candidate(title: str, snippet: str, question: str) -> bool:
    text = f"{title} {snippet}".lower()
    if any(keyword in text for keyword in EVENT_KEYWORDS):
        return True
    tokens = re.findall(r"[a-zA-Z]{3,}", question.lower())
    return any(token in text for token in tokens[:6])

def _normalize_domain(domain: str) -> str:
    return domain.lower().replace("www.", "").strip()


def _is_allowed_domain(url: str) -> bool:
    try:
        domain = _normalize_domain(urlparse(url).netloc)
    except Exception:
        return False
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in ALLOWED_DOMAINS)


def _domain_weight(url: str) -> int:
    domain = _normalize_domain(urlparse(url).netloc) if url else ""
    for key, weight in DOMAIN_WEIGHTS.items():
        if domain == key or domain.endswith(f".{key}"):
            return weight
    return 0


def _intent_match_score(question: str, title: str, snippet: str) -> int:
    processed = query_processor.process(question)
    keywords = processed.keywords
    text = f"{title} {snippet}".lower()
    score = 0
    for kw in keywords[:8]:
        if kw in text:
            score += 2
    return score


def _intent_domain_boost(question: str, url: str) -> int:
    domain = _normalize_domain(urlparse(url).netloc) if url else ""
    lower_q = question.lower()
    for keywords, domains, boost in INTENT_DOMAIN_BOOSTS:
        if any(kw in lower_q for kw in keywords):
            if any(domain == d or domain.endswith(f".{d}") for d in domains):
                return boost
    return 0

def _month_to_number(month: str) -> Optional[int]:
    mapping = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return mapping.get(month.lower())


def _parse_date_range(text: str) -> tuple[Optional[date], Optional[date], Optional[str]]:
    today = date.today()
    cleaned = _normalize_whitespace(text)

    # ISO date
    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", cleaned)
    if iso_match:
        try:
            start = date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
            return start, None, iso_match.group(0)
        except ValueError:
            pass

    # Numeric date: dd/mm/yyyy or mm/dd/yyyy
    numeric_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", cleaned)
    if numeric_match:
        day = int(numeric_match.group(1))
        month = int(numeric_match.group(2))
        year = int(numeric_match.group(3))
        if year < 100:
            year += 2000
        try:
            start = date(year, month, day)
            return start, None, numeric_match.group(0)
        except ValueError:
            pass

    # Range like 3-5 March 2026
    range_match = re.search(
        rf"\b(\d{{1,2}})\s*(?:-|–|to)\s*(\d{{1,2}})\s+{MONTHS_REGEX}\s+(\d{{4}})\b",
        cleaned,
        re.IGNORECASE,
    )
    if range_match:
        start_day = int(range_match.group(1))
        end_day = int(range_match.group(2))
        month_str = range_match.group(3)
        year = int(range_match.group(4))
        month_num = _month_to_number(month_str)
        if month_num:
            try:
                start = date(year, month_num, start_day)
                end = date(year, month_num, end_day)
                return start, end, range_match.group(0)
            except ValueError:
                pass

    # Month name with day + year
    match = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONTHS_REGEX}\s+(\d{{4}})\b", cleaned, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        month_num = _month_to_number(month_str)
        if month_num:
            try:
                start = date(year, month_num, day)
                return start, None, match.group(0)
            except ValueError:
                pass

    # Month name with day, optional year
    match = re.search(rf"\b{MONTHS_REGEX}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?\b", cleaned, re.IGNORECASE)
    if match:
        month_str = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year
        month_num = _month_to_number(month_str)
        if month_num:
            try:
                start = date(year, month_num, day)
                if not match.group(3) and start < today:
                    start = date(year + 1, month_num, day)
                return start, None, match.group(0)
            except ValueError:
                pass

    return None, None, None


def _compute_status(start: date, end: Optional[date]) -> str:
    today = date.today()
    if end and start <= today <= end:
        return "live"
    if start == today:
        return "live"
    return "upcoming"


def _heuristic_event_from_result(
    result: dict,
    question: str,
) -> Tuple[Optional[DiscoverEvent], int]:
    title = _clean_title(_normalize_whitespace(result.get("title") or ""))
    snippet = _normalize_whitespace(result.get("content") or "")
    url = (result.get("url") or "").strip()

    if not title or not url:
        return None, 0

    text_for_fields = _normalize_whitespace(f"{title} {snippet}")
    parsed_start, parsed_end, parsed_date_str = _parse_date_range(text_for_fields)
    date_str = parsed_date_str or _extract_first_match(DATE_PATTERNS, text_for_fields)
    location = _extract_location(text_for_fields)
    cash_prize = _extract_prize(text_for_fields)

    short_description = snippet[:200] if snippet else None

    score = 0
    for field in [date_str, location, cash_prize, short_description]:
        if field:
            score += 1

    if not parsed_start:
        return None, 0
    if parsed_end and parsed_end < date.today():
        return None, 0
    if parsed_start < date.today() and (not parsed_end or parsed_end < date.today()):
        return None, 0

    status = _compute_status(parsed_start, parsed_end)
    event = DiscoverEvent(
        name=title[:120],
        date=date_str,
        start_date=parsed_start.isoformat() if parsed_start else None,
        end_date=parsed_end.isoformat() if parsed_end else None,
        location=location,
        cash_prize=cash_prize,
        short_description=short_description,
        url=url,
        source_url=url,
        status=status,
    )

    if not _is_event_candidate(title, snippet, question):
        return None, 0

    return event, score


class EventDiscovery:
    def __init__(self) -> None:
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not configured")
        if TavilyClient is None:
            raise ImportError("tavily-python is not installed")

        self.client = TavilyClient(api_key=settings.tavily_api_key)
        self.query_model = genai.GenerativeModel("gemini-2.5-flash")
        self.extract_model = genai.GenerativeModel("gemini-2.5-flash")

    def build_search_query(self, question: str, nearby: bool, nearby_location: str | None) -> str:
        prompt = SEARCH_PROMPT.format(
            question=question.strip(),
            nearby=str(nearby).lower(),
            nearby_location=(nearby_location or "").strip(),
        )
        try:
            response = self.query_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 200,
                },
            )
            data = _safe_json_loads(response.text or "", fallback={})
            query = data.get("query") or question
            must_include = data.get("must_include") or []
            exclude = data.get("exclude") or []
        except Exception as e:
            logger.warning("Query augmentation failed: %s", e)
            query = question
            must_include = []
            exclude = []

        # Fallback enrichment to keep searches student-event focused
        extras = ["hackathon", "symposium", "workshop", "competition", "college event"]
        if nearby:
            if nearby_location:
                extras.append(f"in {nearby_location.strip()}")
            else:
                extras.append("near me")

        tokens = [query] + must_include + extras
        if exclude:
            tokens.append(" " + " ".join([f"-{w}" for w in exclude if isinstance(w, str)]))

        return " ".join([t for t in tokens if t]).strip()

    def _extract_events_from_content(self, question: str, content: str, source_url: str) -> list[DiscoverEvent]:
        if not content or len(content.strip()) < 200:
            return []

        prompt = EXTRACT_PROMPT.format(
            question=question.strip(),
            source_url=source_url,
        ) + _truncate(content)

        try:
            response = self.extract_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 500,
                },
            )
            data = _safe_json_loads(response.text or "", fallback=[])
        except Exception as e:
            logger.warning("Event extraction failed for %s: %s", source_url, e)
            return []

        events: list[DiscoverEvent] = []
        if isinstance(data, dict):
            data = [data]

        for item in data:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            url = (item.get("url") or source_url or "").strip()
            if not name or not url:
                continue

            start_date = (item.get("start_date") or "").strip()
            end_date = (item.get("end_date") or "").strip()
            date_str = (item.get("date") or "").strip()

            parsed_start = None
            parsed_end = None
            if start_date:
                try:
                    parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
                except ValueError:
                    parsed_start = None
            if end_date:
                try:
                    parsed_end = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    parsed_end = None

            if not parsed_start:
                parsed_start, parsed_end, parsed_text = _parse_date_range(date_str)
                if not date_str:
                    date_str = parsed_text or None

            if not parsed_start:
                continue

            if parsed_end and parsed_end < date.today():
                continue
            if parsed_start < date.today() and (not parsed_end or parsed_end < date.today()):
                continue

            status = _compute_status(parsed_start, parsed_end)

            event = DiscoverEvent(
                name=name[:120],
                date=date_str or None,
                start_date=parsed_start.isoformat() if parsed_start else None,
                end_date=parsed_end.isoformat() if parsed_end else None,
                location=(item.get("location") or "").strip() or None,
                cash_prize=(item.get("cash_prize") or "").strip() or None,
                short_description=(item.get("short_description") or "").strip() or None,
                url=url,
                source_url=source_url,
                status=status,
            )
            events.append(event)

        return events

    def discover_stream(
        self,
        question: str,
        max_results: int = 40,
        nearby: bool = False,
        nearby_location: str | None = None,
        nearby_lat: float | None = None,
        nearby_lng: float | None = None,
    ) -> Iterable[tuple[str, dict]]:
        search_query = self.build_search_query(question, nearby, nearby_location)

        search_results = self.client.search(
            query=search_query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True,
            include_images=False,
            include_answer=False,
            include_domains=ALLOWED_DOMAINS,
            auto_parameters=False,
        )

        results = search_results.get("results", []) if isinstance(search_results, dict) else []

        def _result_score(res: dict) -> int:
            url = (res.get("url") or "")
            title = (res.get("title") or "")
            snippet = (res.get("content") or "")
            score = 0
            score += _domain_weight(url) * 3
            score += _intent_match_score(question, title, snippet)
            score += _intent_domain_boost(question, url)
            if any(keyword in title.lower() for keyword in EVENT_KEYWORDS):
                score += 2
            return score

        results = [r for r in results if _is_allowed_domain(r.get("url") or "")]
        results = sorted(results, key=_result_score, reverse=True)

        seen_events: set[str] = set()
        emitted = 0
        gemini_budget = max(0, settings.event_discovery_gemini_budget)

        for result in results:
            if emitted >= max_results:
                break

            url = (result.get("url") or "").strip()
            if not url:
                continue

            parsed = urlparse(url)
            domain = parsed.netloc if parsed else None
            citation = DiscoverCitation(
                url=url,
                title=(result.get("title") or "").strip() or None,
                domain=domain,
            )
            yield ("citation", citation.__dict__)

            raw_content = result.get("raw_content") or result.get("content") or ""
            heuristic_event, heuristic_score = _heuristic_event_from_result(result, question)
            if heuristic_event and heuristic_score >= 2:
                key = f"{heuristic_event.name.lower()}::{heuristic_event.date or ''}".strip()
                if key not in seen_events:
                    seen_events.add(key)
                    emitted += 1
                    yield ("event", heuristic_event.__dict__)
                if emitted >= max_results:
                    break
                continue

            if len(raw_content.strip()) < 200:
                try:
                    if hasattr(self.client, "extract"):
                        extract_data = self.client.extract(urls=[url], include_raw_content=True)
                        extract_results = extract_data.get("results", []) if isinstance(extract_data, dict) else []
                        if extract_results:
                            raw_content = extract_results[0].get("raw_content") or extract_results[0].get("content") or raw_content
                except Exception as e:
                    logger.warning("Tavily extract failed for %s: %s", url, e)

            events: list[DiscoverEvent] = []
            if gemini_budget > 0 and _is_event_candidate(result.get("title") or "", result.get("content") or "", question):
                events = self._extract_events_from_content(question, raw_content, url)
                gemini_budget -= 1

            if not events and heuristic_event:
                events = [heuristic_event]

            for event in events:
                key = f"{event.name.lower()}::{event.date or ''}".strip()
                if key in seen_events:
                    continue
                seen_events.add(key)

                emitted += 1
                yield ("event", event.__dict__)
                if emitted >= max_results:
                    break


event_discovery = EventDiscovery()
