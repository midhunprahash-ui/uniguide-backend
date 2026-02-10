"""
Event discovery service using Tavily + Crawl4AI + Gemini.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Iterable, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import google.generativeai as genai

from config import get_settings
from services.event_extractor_crawl4ai import Crawl4AIExtractor
from services.event_discovery_policy import ScopeDecision, classify_discovery_scope
from services.query_processor import query_processor
from services.source_registry import source_registry

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
    stage_id: str | None = None


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
    prize_type: str | None = None
    cash_prize_amount: float | None = None
    cash_prize_currency: str | None = None
    prize_display_text: str | None = None
    confidence: float | None = None
    trust_tier: int | None = None
    validation_flags: list[str] | None = None
    match_category: str | None = None
    registration_status: str | None = None
    registration_deadline: str | None = None


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
- registration_status ("open" | "closed" | "unknown")
- registration_deadline (string as written on page if available)

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
    re.compile(
        r"(?:cash\s+prize|prize\s+pool|prize)\s*[:\-]?\s*"
        r"(₹|Rs\.?|INR|USD|EUR|GBP|CAD|AUD|SGD|US\$|C\$|\$)\s?\d[\d,]*(?:\.\d+)?"
        r"(?:\s?(?:k|K|lakhs?|lakh|crores?|crore|cr|m|million))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(₹|Rs\.?|INR|USD|EUR|GBP|CAD|AUD|SGD|US\$|C\$|\$)\s?\d[\d,]*(?:\.\d+)?"
        r"(?:\s?(?:k|K|lakhs?|lakh|crores?|crore|cr|m|million))?",
        re.IGNORECASE,
    ),
]

REGISTRATION_OPEN_TERMS = (
    "registration open",
    "registrations open",
    "applications open",
    "application open",
    "register now",
    "apply now",
    "accepting applications",
    "submissions open",
    "open for registration",
    "open until",
)

REGISTRATION_CLOSED_TERMS = (
    "registration closed",
    "registrations closed",
    "application closed",
    "applications closed",
    "registrations are closed",
    "entries closed",
    "submission closed",
    "registration over",
    "deadline passed",
    "applications are closed",
)

PRIZE_CONTEXT_TERMS = (
    "prize",
    "prize pool",
    "cash",
    "reward",
    "winner",
    "winning",
    "worth",
)

NON_CASH_PRIZE_TERMS = (
    "non-cash",
    "non cash",
    "swag",
    "goodies",
    "voucher",
    "certificate",
    "mentorship",
    "credits",
    "merch",
    "internship opportunity",
    "gift card",
)

LOCATION_NOISE_TERMS = (
    "join",
    "apply",
    "register",
    "registration",
    "submission",
    "deadline",
    "participants",
    "happening",
    "mark",
    "celebrates",
    "innovation",
    "collaboration",
    "truth and service",
)

MARKDOWN_NOISE_PATTERN = re.compile(r"(?:^|[\s])#{1,6}(?=\S)|[*`_~]+")
PRIZE_CURRENCY_TOKEN_PATTERN = (
    r"₹|Rs\.?|INR|USD|EUR|GBP|CAD|AUD|SGD|US\$|C\$|A\$|\$|€|£"
)
PRIZE_SCALE_TOKEN_PATTERN = r"k|K|lakhs?|lakh|crores?|crore|cr|m|million"
PRIZE_CURRENCY_PATTERN = re.compile(
    rf"(?P<currency>{PRIZE_CURRENCY_TOKEN_PATTERN})\s*"
    r"(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    rf"(?:\s*(?P<scale>{PRIZE_SCALE_TOKEN_PATTERN})\b)?",
    re.IGNORECASE,
)
PRIZE_SUFFIX_CURRENCY_PATTERN = re.compile(
    r"(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    rf"(?:\s*(?P<scale>{PRIZE_SCALE_TOKEN_PATTERN})\b)?\s*"
    rf"(?P<currency>INR|USD|EUR|GBP|CAD|AUD|SGD)\b",
    re.IGNORECASE,
)
PRIZE_UNIT_ONLY_PATTERN = re.compile(
    r"(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*"
    rf"(?P<scale>{PRIZE_SCALE_TOKEN_PATTERN})\b",
    re.IGNORECASE,
)


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


TRACKING_QUERY_PREFIXES = (
    "utm_",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
)


def _canonicalize_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return url.strip()

    if not parsed.scheme or not parsed.netloc:
        return url.strip()

    clean_path = parsed.path.rstrip("/") or "/"
    query_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k and not any(k.lower().startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    query_params.sort(key=lambda item: (item[0], item[1]))
    clean_query = urlencode(query_params, doseq=True)

    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            clean_path,
            "",
            clean_query,
            "",
        )
    )


def _extract_first_match(patterns: list[re.Pattern], text: str) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(0).strip()
            return value
    return None


def _sanitize_event_name(name: str) -> str:
    cleaned = MARKDOWN_NOISE_PATTERN.sub(" ", name or "")
    cleaned = _normalize_whitespace(cleaned)
    if not cleaned:
        return ""
    # Remove duplicated immediate title fragments.
    parts = [part.strip() for part in re.split(r"\s+[-|]\s+", cleaned) if part.strip()]
    if not parts:
        return cleaned[:120]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return " - ".join(deduped)[:120]


def _sanitize_short_description(text: str, event_name: str | None = None) -> Optional[str]:
    cleaned = MARKDOWN_NOISE_PATTERN.sub(" ", text or "")
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    cleaned = re.sub(r"\s*\|\s*", " | ", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    if not cleaned:
        return None

    if event_name:
        event_tokens = set(re.findall(r"[a-z0-9]{3,}", event_name.lower()))
    else:
        event_tokens = set()

    raw_segments = [
        segment.strip(" -|,;:")
        for segment in re.split(r"[|.;\n]", cleaned)
        if segment.strip()
    ]
    scored_segments: list[tuple[int, str]] = []
    for segment in raw_segments:
        lower_segment = segment.lower()
        if "http" in lower_segment:
            continue
        if len(segment) < 12:
            continue
        words = re.findall(r"[a-z0-9]+", lower_segment)
        if len(words) < 4:
            continue
        is_heading_like = bool(
            re.fullmatch(r"(?:[A-Za-z][A-Za-z&'/()-]*\s*){1,7}", segment)
        )
        if is_heading_like and not any(
            term in lower_segment
            for term in ("register", "apply", "open", "deadline", "online", "prize", "cash")
        ):
            continue
        overlap = len([token for token in words if token in event_tokens])
        score = 0
        if overlap < max(1, len(words) // 2):
            score += 2
        if any(term in lower_segment for term in PRIZE_CONTEXT_TERMS):
            score += 1
        if any(term in lower_segment for term in REGISTRATION_OPEN_TERMS):
            score += 1
        if len(words) <= 28:
            score += 1
        scored_segments.append((score, segment))

    if not scored_segments:
        fallback = cleaned[:220].strip(" -|,;")
        return fallback or None

    scored_segments.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen_segment_keys: set[str] = set()
    for _, segment in scored_segments:
        key = segment.lower()
        if key in seen_segment_keys:
            continue
        seen_segment_keys.add(key)
        selected.append(segment)
        if len(selected) >= 2:
            break

    summary = " | ".join(selected)
    summary = _normalize_whitespace(summary)
    return summary[:220] if summary else None


def _sanitize_location(value: str | None) -> Optional[str]:
    if not value:
        return None
    cleaned = _normalize_whitespace(value)
    cleaned = re.split(r"\b(?:happening|starting|ending|register|apply|deadline|join)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = cleaned.strip(" -|,.;")
    if not cleaned:
        return None
    lower = cleaned.lower()
    if "http" in lower:
        return None
    if any(term in lower for term in LOCATION_NOISE_TERMS):
        return None
    if len(cleaned.split()) > 8:
        return None
    return cleaned[:80]


def _normalize_prize_text(value: str | None) -> Optional[str]:
    if not value:
        return None
    cleaned = _normalize_whitespace(value)
    cleaned = re.sub(r"\bprize(?:\s*pool)?\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", cleaned)
    cleaned = cleaned.strip(" -|,;")
    if re.fullmatch(r"\d{4}\s*cr", cleaned, flags=re.IGNORECASE):
        return None
    return cleaned[:80] if cleaned else None


def _normalize_currency_token(token: str | None) -> Optional[str]:
    if not token:
        return None
    cleaned = token.strip().upper().replace(".", "")
    mapping = {
        "₹": "INR",
        "RS": "INR",
        "INR": "INR",
        "US$": "USD",
        "$": "USD",
        "USD": "USD",
        "€": "EUR",
        "EUR": "EUR",
        "£": "GBP",
        "GBP": "GBP",
        "C$": "CAD",
        "CAD": "CAD",
        "A$": "AUD",
        "AUD": "AUD",
        "SGD": "SGD",
    }
    return mapping.get(cleaned)


def _parse_scaled_amount(amount_text: str, scale_text: str | None) -> Optional[float]:
    try:
        base_amount = float((amount_text or "").replace(",", ""))
    except Exception:
        return None

    if base_amount <= 0:
        return None

    scale_key = (scale_text or "").strip().lower()
    multipliers = {
        "k": 1_000,
        "lakhs": 100_000,
        "lakh": 100_000,
        "crores": 10_000_000,
        "crore": 10_000_000,
        "cr": 10_000_000,
        "m": 1_000_000,
        "million": 1_000_000,
    }
    multiplier = multipliers.get(scale_key, 1.0)
    total = base_amount * multiplier

    if total <= 0 or total > 10_000_000_000:
        return None
    return round(total, 2)


def _format_cash_prize(currency: str, amount: float) -> str:
    if float(amount).is_integer():
        amount_text = f"{int(amount):,}"
    else:
        amount_text = f"{amount:,.2f}".rstrip("0").rstrip(".")
    return f"{currency} {amount_text}"


def _extract_prize_details(text: str | None) -> tuple[Optional[str], Optional[float], Optional[str], str]:
    normalized = _normalize_whitespace(text or "")
    if not normalized:
        return None, None, None, "unknown"

    segments = [segment.strip() for segment in re.split(r"[|.;\n]", normalized) if segment.strip()]
    candidate_segments = [
        segment
        for segment in segments
        if any(term in segment.lower() for term in PRIZE_CONTEXT_TERMS + NON_CASH_PRIZE_TERMS)
    ] or segments

    non_cash_candidate: str | None = None

    for segment in candidate_segments:
        lowered = segment.lower()

        currency_match = PRIZE_CURRENCY_PATTERN.search(segment)
        if currency_match:
            currency = _normalize_currency_token(currency_match.group("currency"))
            amount = _parse_scaled_amount(
                currency_match.group("amount"),
                currency_match.group("scale"),
            )
            if currency and amount is not None:
                return _format_cash_prize(currency, amount), amount, currency, "cash"

        suffix_match = PRIZE_SUFFIX_CURRENCY_PATTERN.search(segment)
        if suffix_match:
            currency = _normalize_currency_token(suffix_match.group("currency"))
            amount = _parse_scaled_amount(
                suffix_match.group("amount"),
                suffix_match.group("scale"),
            )
            if currency and amount is not None:
                return _format_cash_prize(currency, amount), amount, currency, "cash"

        if any(term in lowered for term in NON_CASH_PRIZE_TERMS):
            normalized_non_cash = _normalize_prize_text(segment)
            if normalized_non_cash and not non_cash_candidate:
                non_cash_candidate = normalized_non_cash

    # Conservative fallback: require explicit prize context and currency token.
    for pattern in PRIZE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        matched = _normalize_prize_text(match.group(0))
        if not matched:
            continue
        parsed_match = PRIZE_CURRENCY_PATTERN.search(matched) or PRIZE_SUFFIX_CURRENCY_PATTERN.search(matched)
        if not parsed_match:
            continue
        currency = _normalize_currency_token(parsed_match.group("currency"))
        amount = _parse_scaled_amount(
            parsed_match.group("amount"),
            parsed_match.group("scale"),
        )
        if currency and amount is not None:
            return _format_cash_prize(currency, amount), amount, currency, "cash"

    if non_cash_candidate:
        return non_cash_candidate[:80], None, None, "non_cash"
    return None, None, None, "unknown"


def _derive_prize_fields(
    primary_text: str | None,
    context_text: str | None = None,
) -> tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
    display_text, amount, currency, prize_type = _extract_prize_details(primary_text)
    if prize_type == "unknown" and context_text:
        display_text, amount, currency, prize_type = _extract_prize_details(context_text)

    normalized_type = prize_type if prize_type != "unknown" else None
    cash_prize = display_text if prize_type == "cash" else None
    return cash_prize, display_text, amount, currency, normalized_type


def _extract_location(text: str) -> Optional[str]:
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = _sanitize_location((match.group(1) or "").strip())
        if value:
            return value
    return None


def _extract_prize(text: str) -> Optional[str]:
    cash_prize, _, _, _, _ = _derive_prize_fields(primary_text=text, context_text=None)
    return cash_prize


def _clean_title(title: str) -> str:
    cleaned = title
    for sep in [" | ", " - ", " — ", " – "]:
        if sep in cleaned:
            cleaned = cleaned.split(sep)[0].strip()
            break
    return _sanitize_event_name(cleaned)


def _is_event_candidate(title: str, snippet: str, question: str) -> bool:
    text = f"{title} {snippet}".lower()
    if any(keyword in text for keyword in EVENT_KEYWORDS):
        return True
    tokens = re.findall(r"[a-zA-Z]{3,}", question.lower())
    return any(token in text for token in tokens[:6])


def _intent_match_score(question: str, title: str, snippet: str) -> int:
    processed = query_processor.process(question)
    keywords = processed.keywords
    text = f"{title} {snippet}".lower()
    score = 0
    for kw in keywords[:8]:
        if kw in text:
            score += 2
    return score

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


def _extract_registration_deadline(text: str) -> tuple[Optional[date], Optional[str]]:
    cleaned = _normalize_whitespace(text)
    deadline_patterns = [
        re.compile(
            r"(?:registration|registrations|application|applications|submission|submissions)"
            r"\s*(?:deadline|closes?|close|ends?)\s*[:\-]?\s*([^\n.;|]{4,60})",
            re.IGNORECASE,
        ),
        re.compile(r"(?:last date to apply|last date to register)\s*[:\-]?\s*([^\n.;|]{4,60})", re.IGNORECASE),
    ]

    for pattern in deadline_patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        candidate = _normalize_whitespace(match.group(1))
        parsed_start, _, parsed_text = _parse_date_range(candidate)
        if parsed_start:
            return parsed_start, parsed_text or candidate
    return None, None


def _detect_registration_status(text: str) -> tuple[str, Optional[date], Optional[str]]:
    normalized = _normalize_whitespace(text).lower()
    today = date.today()
    deadline_date, deadline_text = _extract_registration_deadline(text)

    if any(term in normalized for term in REGISTRATION_CLOSED_TERMS):
        return "closed", deadline_date, deadline_text
    if deadline_date and deadline_date < today:
        return "closed", deadline_date, deadline_text
    if any(term in normalized for term in REGISTRATION_OPEN_TERMS):
        return "open", deadline_date, deadline_text
    if deadline_date and deadline_date >= today:
        return "open", deadline_date, deadline_text
    return "unknown", deadline_date, deadline_text


def _is_future_event_date(start_date: Optional[date]) -> bool:
    return bool(start_date and start_date > date.today())


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
    location = _sanitize_location(_extract_location(text_for_fields))
    cash_prize, prize_display_text, cash_prize_amount, cash_prize_currency, prize_type = _derive_prize_fields(
        primary_text=text_for_fields,
        context_text=None,
    )
    short_description = _sanitize_short_description(snippet, title)

    score = 0
    for field in [date_str, location, prize_display_text, short_description]:
        if field:
            score += 1

    if not parsed_start or not _is_future_event_date(parsed_start):
        return None, 0
    if parsed_end and parsed_end < date.today():
        return None, 0

    registration_status, registration_deadline_date, registration_deadline_text = (
        _detect_registration_status(text_for_fields)
    )
    if registration_status != "open":
        return None, 0

    status = _compute_status(parsed_start, parsed_end)
    event = DiscoverEvent(
        name=_sanitize_event_name(title)[:120],
        date=date_str,
        start_date=parsed_start.isoformat() if parsed_start else None,
        end_date=parsed_end.isoformat() if parsed_end else None,
        location=location,
        cash_prize=cash_prize,
        prize_type=prize_type,
        cash_prize_amount=cash_prize_amount,
        cash_prize_currency=cash_prize_currency,
        prize_display_text=prize_display_text,
        short_description=short_description,
        url=url,
        source_url=url,
        status=status,
        registration_status=registration_status,
        registration_deadline=(
            registration_deadline_date.isoformat()
            if registration_deadline_date
            else registration_deadline_text
        ),
    )

    if not _is_event_candidate(title, snippet, question):
        return None, 0

    return event, score


def _confidence_from_signals(
    domain_score: int,
    intent_score: int,
    heuristic_score: int,
    has_date: bool,
    has_location: bool,
    has_prize: bool,
) -> float:
    raw = 25 + (domain_score * 4) + (intent_score * 3) + (heuristic_score * 5)
    if has_date:
        raw += 10
    if has_location:
        raw += 5
    if has_prize:
        raw += 3

    bounded = max(0.0, min(100.0, float(raw)))
    return round(bounded / 100.0, 2)


def _build_event_validation_flags(event: DiscoverEvent, domain_flags: list[str]) -> list[str]:
    flags = list(domain_flags)
    if event.start_date:
        flags.append("has_start_date")
    if event.end_date:
        flags.append("has_end_date")
    if event.location:
        flags.append("has_location")
    if event.prize_display_text:
        flags.append("has_prize")
    if event.short_description:
        flags.append("has_description")
    if event.registration_status:
        flags.append(f"registration_{event.registration_status}")
    if event.registration_deadline:
        flags.append("has_registration_deadline")
    return flags


class EventDiscovery:
    def __init__(self) -> None:
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not configured")
        if TavilyClient is None:
            raise ImportError("tavily-python is not installed")

        self.client = TavilyClient(api_key=settings.tavily_api_key)
        self.query_model = genai.GenerativeModel("gemini-2.5-flash")
        self.extract_model = genai.GenerativeModel("gemini-2.5-flash")
        mode = (settings.event_discovery_extractor_mode or "hybrid").strip().lower()
        self.extractor_mode = mode if mode in {"legacy", "crawl4ai", "hybrid"} else "hybrid"
        self.crawl4ai_max_urls = max(0, int(settings.event_discovery_crawl4ai_max_urls))
        self.crawl4ai_extractor = Crawl4AIExtractor(
            timeout_ms=max(4000, int(settings.event_discovery_crawl4ai_timeout_ms)),
            max_chars=18000,
            check_robots_txt=bool(settings.event_discovery_crawl4ai_check_robots_txt),
        )

    def evaluate_policy(self, question: str, category_hint: str) -> ScopeDecision:
        return classify_discovery_scope(question, category_hint=category_hint or "all")

    def build_search_plan(
        self,
        question: str,
        normalized_intent: str,
        nearby: bool,
        nearby_location: str | None,
        max_results: int,
    ) -> dict:
        intent_token_map = {
            "hackathons": "hackathons",
            "internships": "student internships",
            "conferences": "student conferences",
            "scholarships": "student scholarships",
            "jobs": "entry level roles for students",
            "general_student_opportunity": "student opportunities",
        }
        intent_phrase = intent_token_map.get(normalized_intent, "student opportunities")

        base = question.strip()
        if normalized_intent != "general_student_opportunity" and normalized_intent not in base.lower():
            base = f"{base} {intent_phrase}".strip()

        primary_query = self.build_search_query(
            question=base,
            nearby=nearby,
            nearby_location=nearby_location,
            intent_hint=normalized_intent,
        )
        expanded_query = self.build_search_query(
            question=f"{base} official portals and trusted listings",
            nearby=nearby,
            nearby_location=nearby_location,
            intent_hint=normalized_intent,
        )

        first_stage_limit = max(8, min(max_results, 20))
        second_stage_limit = max(8, min(max_results, 20))

        return {
            "intent": normalized_intent,
            "stages": [
                {
                    "id": "initial",
                    "label": f"Searching sites for {question.strip()}.",
                    "queries": [
                        primary_query,
                        f"{question.strip()} {datetime.now().year}",
                    ],
                    "search_depth": "basic",
                    "max_results": first_stage_limit,
                },
                {
                    "id": "expanded",
                    "label": (
                        f"Expanding search across specialized platforms for {question.strip()}."
                    ),
                    "queries": [
                        expanded_query,
                        f"{intent_phrase} trusted sources {datetime.now().year}",
                    ],
                    "search_depth": "advanced",
                    "max_results": second_stage_limit,
                },
            ],
        }

    def build_search_query(
        self,
        question: str,
        nearby: bool,
        nearby_location: str | None,
        intent_hint: str | None = None,
    ) -> str:
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
        if intent_hint and intent_hint not in {"unknown", "general_student_opportunity", "all"}:
            extras.append(intent_hint)
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
            name = _sanitize_event_name((item.get("name") or "").strip())
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
            if not _is_future_event_date(parsed_start):
                continue

            registration_context = _normalize_whitespace(
                " ".join(
                    [
                        name,
                        date_str or "",
                        (item.get("short_description") or "").strip(),
                        (item.get("location") or "").strip(),
                        (item.get("registration_status") or "").strip(),
                        (item.get("registration_deadline") or "").strip(),
                        content[:2500],
                    ]
                )
            )
            registration_status, registration_deadline_date, registration_deadline_text = (
                _detect_registration_status(registration_context)
            )
            if registration_status != "open":
                continue

            normalized_location = _sanitize_location((item.get("location") or "").strip() or None)
            extracted_location = _extract_location(registration_context)
            if not normalized_location:
                normalized_location = _sanitize_location(extracted_location)

            raw_prize = (item.get("cash_prize") or "").strip()
            (
                normalized_prize,
                prize_display_text,
                cash_prize_amount,
                cash_prize_currency,
                prize_type,
            ) = _derive_prize_fields(
                primary_text=raw_prize,
                context_text=registration_context,
            )

            raw_description = (item.get("short_description") or "").strip()
            short_description = _sanitize_short_description(
                raw_description or registration_context[:400],
                event_name=name,
            )

            status = _compute_status(parsed_start, parsed_end)

            event = DiscoverEvent(
                name=name[:120],
                date=date_str or None,
                start_date=parsed_start.isoformat() if parsed_start else None,
                end_date=parsed_end.isoformat() if parsed_end else None,
                location=normalized_location,
                cash_prize=normalized_prize,
                prize_type=prize_type,
                cash_prize_amount=cash_prize_amount,
                cash_prize_currency=cash_prize_currency,
                prize_display_text=prize_display_text,
                short_description=short_description,
                url=url,
                source_url=source_url,
                status=status,
                registration_status=registration_status,
                registration_deadline=(
                    registration_deadline_date.isoformat()
                    if registration_deadline_date
                    else registration_deadline_text
                ),
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
        category_hint: str = "all",
        strict_trust: bool = True,
    ) -> Iterable[tuple[str, dict]]:
        start_time = time.perf_counter()
        policy = self.evaluate_policy(question, category_hint)
        yield ("policy", policy.to_payload())

        if not policy.allowed:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            yield (
                "metrics",
                {
                    "cached": False,
                    "policy_allowed": False,
                    "citations_emitted": 0,
                    "events_emitted": 0,
                    "first_citation_ms": None,
                    "first_event_ms": None,
                    "total_ms": total_ms,
                },
            )
            return

        search_plan = self.build_search_plan(
            question=question,
            normalized_intent=policy.normalized_intent,
            nearby=nearby,
            nearby_location=nearby_location,
            max_results=max_results,
        )
        yield ("search_plan", search_plan)

        include_domains = source_registry.get_include_domains(
            category_hint=policy.normalized_intent,
            strict_trust=strict_trust,
            max_domains=50,
        )
        processed_query = query_processor.process(question)

        seen_citation_urls: set[str] = set()
        processed_result_urls: set[str] = set()
        seen_events: set[str] = set()
        citations_emitted = 0
        events_emitted = 0
        gemini_budget = max(0, settings.event_discovery_gemini_budget)
        first_citation_ms: int | None = None
        first_event_ms: int | None = None
        crawl4ai_attempts = 0
        crawl4ai_hits = 0

        def _result_score(result: dict) -> int:
            title = (result.get("title") or "").strip()
            snippet = (result.get("content") or "").strip()
            url = (result.get("url") or "").strip()

            allowed, domain_score, _, _ = source_registry.score_url(
                url=url,
                category_hint=policy.normalized_intent,
                strict_trust=strict_trust,
            )
            if not allowed:
                return -1000

            score = domain_score * 3
            score += _intent_match_score(question, title, snippet)
            for keyword in processed_query.keywords[:8]:
                if keyword in f"{title} {snippet}".lower():
                    score += 1
            if any(keyword in title.lower() for keyword in EVENT_KEYWORDS):
                score += 3

            provider_score = result.get("score")
            if isinstance(provider_score, (int, float)):
                score += int(float(provider_score) * 10)
            return score

        for stage in search_plan.get("stages", []):
            if events_emitted >= max_results:
                break

            stage_id = str(stage.get("id") or "stage")
            stage_depth = str(stage.get("search_depth") or "basic")
            stage_limit = int(stage.get("max_results") or max_results)
            stage_queries = stage.get("queries") or []

            for stage_query in stage_queries:
                if events_emitted >= max_results:
                    break
                try:
                    search_results = self.client.search(
                        query=stage_query,
                        search_depth=stage_depth,
                        max_results=stage_limit,
                        include_raw_content=True,
                        include_images=False,
                        include_answer=False,
                        include_domains=include_domains,
                        auto_parameters=True,
                    )
                except Exception as e:
                    logger.warning("Tavily search failed for stage '%s': %s", stage_id, e)
                    continue

                results = search_results.get("results", []) if isinstance(search_results, dict) else []

                for result in results:
                    url = (result.get("url") or "").strip()
                    if not url:
                        continue

                    canonical_url = _canonicalize_url(url)
                    if canonical_url in seen_citation_urls:
                        continue

                    seen_citation_urls.add(canonical_url)
                    citation = DiscoverCitation(
                        url=canonical_url,
                        title=(result.get("title") or "").strip() or None,
                        domain=(urlparse(url).netloc or None),
                        stage_id=stage_id,
                    )
                    citations_emitted += 1
                    if first_citation_ms is None:
                        first_citation_ms = int((time.perf_counter() - start_time) * 1000)
                    yield ("citation", citation.__dict__)

                ranked_results = sorted(results, key=_result_score, reverse=True)

                for ranked_index, result in enumerate(ranked_results):
                    if events_emitted >= max_results:
                        break

                    url = (result.get("url") or "").strip()
                    if not url:
                        continue

                    canonical_url = _canonicalize_url(url)
                    if canonical_url in processed_result_urls:
                        continue
                    processed_result_urls.add(canonical_url)

                    allowed, domain_score, trust_tier, domain_flags = source_registry.score_url(
                        url=url,
                        category_hint=policy.normalized_intent,
                        strict_trust=strict_trust,
                    )
                    if not allowed:
                        continue

                    raw_content = result.get("raw_content") or result.get("content") or ""
                    used_crawl4ai = False
                    if (
                        self.extractor_mode in {"hybrid", "crawl4ai"}
                        and self.crawl4ai_max_urls > 0
                        and ranked_index < self.crawl4ai_max_urls
                    ):
                        crawl4ai_attempts += 1
                        crawl_payload = self.crawl4ai_extractor.extract_page(canonical_url)
                        if crawl_payload and crawl_payload.text:
                            used_crawl4ai = True
                            crawl4ai_hits += 1
                            raw_content = _normalize_whitespace(
                                f"{raw_content}\n{crawl_payload.text}"
                            )

                            for crawled_url in crawl_payload.links[:8]:
                                crawled_canonical = _canonicalize_url(crawled_url)
                                if not crawled_canonical:
                                    continue
                                if crawled_canonical in seen_citation_urls:
                                    continue
                                seen_citation_urls.add(crawled_canonical)
                                citations_emitted += 1
                                if first_citation_ms is None:
                                    first_citation_ms = int((time.perf_counter() - start_time) * 1000)
                                crawled_citation = DiscoverCitation(
                                    url=crawled_canonical,
                                    title=None,
                                    domain=(urlparse(crawled_canonical).netloc or None),
                                    stage_id=stage_id,
                                )
                                yield ("citation", crawled_citation.__dict__)

                    heuristic_event, heuristic_score = _heuristic_event_from_result(result, question)
                    candidate_events: list[DiscoverEvent] = []

                    if heuristic_event and heuristic_score >= 2 and self.extractor_mode == "legacy":
                        candidate_events = [heuristic_event]
                    else:
                        if len(raw_content.strip()) < 200:
                            try:
                                if hasattr(self.client, "extract"):
                                    extract_data = self.client.extract(
                                        urls=[url],
                                        include_raw_content=True,
                                    )
                                    extract_results = (
                                        extract_data.get("results", [])
                                        if isinstance(extract_data, dict)
                                        else []
                                    )
                                    if extract_results:
                                        raw_content = (
                                            extract_results[0].get("raw_content")
                                            or extract_results[0].get("content")
                                            or raw_content
                                        )
                            except Exception as e:
                                logger.warning("Tavily extract failed for %s: %s", url, e)

                        should_call_gemini = gemini_budget > 0 and (
                            used_crawl4ai
                            or _is_event_candidate(
                                result.get("title") or "",
                                result.get("content") or "",
                                question,
                            )
                        )
                        if should_call_gemini:
                            candidate_events = self._extract_events_from_content(
                                question=question,
                                content=raw_content,
                                source_url=canonical_url,
                            )
                            gemini_budget -= 1

                        if not candidate_events and heuristic_event:
                            candidate_events = [heuristic_event]

                    for event in candidate_events:
                        event.url = _canonicalize_url(event.url or canonical_url)
                        event.source_url = event.source_url or canonical_url
                        event.match_category = policy.normalized_intent
                        event.trust_tier = trust_tier if trust_tier < 99 else None
                        event.name = _sanitize_event_name(event.name)
                        if not event.name:
                            continue

                        normalization_context = _normalize_whitespace(
                            " ".join(
                                [
                                    result.get("title") or "",
                                    result.get("content") or "",
                                    raw_content[:3000],
                                    event.short_description or "",
                                    event.location or "",
                                ]
                            )
                        )
                        event.short_description = _sanitize_short_description(
                            event.short_description or normalization_context[:450],
                            event_name=event.name,
                        )
                        event.location = _sanitize_location(event.location) or _sanitize_location(
                            _extract_location(normalization_context)
                        )
                        (
                            event.cash_prize,
                            event.prize_display_text,
                            event.cash_prize_amount,
                            event.cash_prize_currency,
                            event.prize_type,
                        ) = _derive_prize_fields(
                            primary_text=event.cash_prize or event.prize_display_text,
                            context_text=normalization_context,
                        )

                        parsed_start_date: Optional[date] = None
                        if event.start_date:
                            try:
                                parsed_start_date = datetime.strptime(event.start_date, "%Y-%m-%d").date()
                            except ValueError:
                                parsed_start_date = None
                        if not _is_future_event_date(parsed_start_date):
                            continue

                        registration_status = event.registration_status
                        if registration_status != "open":
                            registration_context = _normalize_whitespace(
                                " ".join(
                                    [
                                        result.get("title") or "",
                                        result.get("content") or "",
                                        raw_content[:2500],
                                        event.name or "",
                                        event.short_description or "",
                                    ]
                                )
                            )
                            status_guess, deadline_date, deadline_text = _detect_registration_status(
                                registration_context
                            )
                            if status_guess != "open":
                                continue
                            event.registration_status = status_guess
                            if not event.registration_deadline:
                                event.registration_deadline = (
                                    deadline_date.isoformat() if deadline_date else deadline_text
                                )

                        intent_score = _intent_match_score(
                            question,
                            event.name or "",
                            event.short_description or "",
                        )
                        confidence = _confidence_from_signals(
                            domain_score=domain_score,
                            intent_score=intent_score,
                            heuristic_score=heuristic_score,
                            has_date=bool(event.start_date),
                            has_location=bool(event.location),
                            has_prize=bool(event.prize_display_text),
                        )
                        event.confidence = confidence
                        event.validation_flags = _build_event_validation_flags(event, domain_flags)

                        # Keep only valid high-signal results.
                        if confidence < settings.event_discovery_confidence_threshold:
                            continue

                        dedupe_key = (
                            f"{(event.name or '').strip().lower()}::"
                            f"{event.start_date or event.date or ''}::"
                            f"{_canonicalize_url(event.url or canonical_url)}"
                        )
                        if dedupe_key in seen_events:
                            continue
                        seen_events.add(dedupe_key)

                        events_emitted += 1
                        if first_event_ms is None:
                            first_event_ms = int((time.perf_counter() - start_time) * 1000)
                        yield ("event", event.__dict__)

                        if events_emitted >= max_results:
                            break

        total_ms = int((time.perf_counter() - start_time) * 1000)
        yield (
            "metrics",
            {
                "cached": False,
                "policy_allowed": True,
                "citations_emitted": citations_emitted,
                "events_emitted": events_emitted,
                "first_citation_ms": first_citation_ms,
                "first_event_ms": first_event_ms,
                "crawl4ai_attempts": crawl4ai_attempts,
                "crawl4ai_hits": crawl4ai_hits,
                "total_ms": total_ms,
            },
        )


@lru_cache
def get_event_discovery() -> "EventDiscovery":
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    if TavilyClient is None:
        raise RuntimeError("tavily-python is not installed")
    return EventDiscovery()
