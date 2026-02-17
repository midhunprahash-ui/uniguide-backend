"""
Event discovery service using Tavily + Crawl4AI + Gemini.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import google.generativeai as genai

from config import get_settings
from services.event_discovery_policy import ScopeDecision, classify_discovery_scope
from services.event_extractor_crawl4ai import Crawl4AIExtractor
from services.query_processor import QueryIntent, query_processor
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
    registration_fee_text: str | None = None
    registration_fee_amount: float | None = None
    registration_fee_currency: str | None = None
    prize_confidence: float | None = None
    location_confidence: float | None = None
    resolved_region: str | None = None
    is_remote: bool | None = None
    geo_match: str | None = None
    summary_structured: dict[str, str | None] | None = None
    evidence: list[dict[str, Any]] | None = None


SEARCH_PROMPT = """
You craft precise web search queries for student events.

User request: "{question}"
Nearby: {nearby}
Nearby location: {nearby_location}
Intent: {intent}
Timeframe focus: {timeframe}
Location scope: {location_scope}
Focus terms: {focus_terms}
Exclude terms: {exclude_terms}
Query mode: {query_mode}

Return JSON only with keys:
- query: a concise search query (8-12 words)
- must_include: list of short phrases to include
- exclude: list of short phrases to exclude

Rules:
- Think like a web search user. Be specific and contextual.
- Focus on college/university opportunities and events.
- If nearby is true and location is provided, include it (e.g., "in {nearby_location}").
- If nearby is true and location is empty, include "near me" or "local".
- Avoid quotes. No markdown.
"""

INTENT_CLASSIFIER_PROMPT = """
Classify the user's intent for a student opportunity discovery system.

User request: "{question}"
Category hint: {category_hint}
Nearby enabled: {nearby}
Nearby location: {nearby_location}
Current year: {current_year}

Return JSON only with keys:
- intent: one of ["hackathons","internships","conferences","scholarships","jobs",
  "general_student_opportunity","unknown"]
- search_required: boolean (false for greetings/chitchat/too vague requests)
- confidence: number from 0 to 1
- timeframe: one of ["upcoming","ongoing","past","any"]
- location_scope: one of ["nearby","city","country","global","any"]
- focus_terms: list of concise keyword phrases
- exclude_terms: list of concise keyword phrases
- rewritten_query: a specific web-search-friendly rewrite of the request
- reason: one short sentence

Rules:
- Prioritize student/college opportunity intent.
- If category hint is specific (not "all"), align intent with it unless clearly impossible.
- Keep focus_terms <= 8 and exclude_terms <= 6.
- rewritten_query should add specific context words that improve web retrieval.
- Return only valid JSON.
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
- registration_fee (string, include currency if present)
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

SUPPORTED_INTENTS = {
    "hackathons",
    "internships",
    "conferences",
    "scholarships",
    "jobs",
    "general_student_opportunity",
    "unknown",
}

TIMEFRAME_TERMS: dict[str, tuple[str, ...]] = {
    "upcoming": ("upcoming", "next", "this month", "this week", "soon", "open now"),
    "ongoing": ("ongoing", "happening now", "currently running", "live now"),
    "past": ("past", "previous", "last year", "closed", "ended"),
}

LOCATION_SCOPE_TERMS: dict[str, tuple[str, ...]] = {
    "global": ("global", "worldwide", "international"),
    "city": ("in ", "at "),
    "country": ("india", "usa", "united states", "uk", "canada"),
}

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

FEE_CONTEXT_TERMS = (
    "registration fee",
    "entry fee",
    "application fee",
    "participation fee",
    "ticket fee",
    "fee:",
    "fees:",
    "pay",
    "payment",
    "paid entry",
)

REMOTE_TERMS = (
    "online",
    "virtual",
    "remote",
    "work from home",
    "wfh",
    "anywhere",
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

INDIA_STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "andhra pradesh": ("andhra", "andhra pradesh", "ap"),
    "arunachal pradesh": ("arunachal", "arunachal pradesh"),
    "assam": ("assam",),
    "bihar": ("bihar",),
    "chhattisgarh": ("chhattisgarh",),
    "goa": ("goa",),
    "gujarat": ("gujarat",),
    "haryana": ("haryana",),
    "himachal pradesh": ("himachal", "himachal pradesh"),
    "jharkhand": ("jharkhand",),
    "karnataka": ("karnataka",),
    "kerala": ("kerala",),
    "madhya pradesh": ("madhya pradesh", "mp"),
    "maharashtra": ("maharashtra",),
    "manipur": ("manipur",),
    "meghalaya": ("meghalaya",),
    "mizoram": ("mizoram",),
    "nagaland": ("nagaland",),
    "odisha": ("odisha", "orissa"),
    "punjab": ("punjab",),
    "rajasthan": ("rajasthan",),
    "sikkim": ("sikkim",),
    "tamil nadu": ("tamil nadu", "tn"),
    "telangana": ("telangana",),
    "tripura": ("tripura",),
    "uttar pradesh": ("uttar pradesh", "up"),
    "uttarakhand": ("uttarakhand", "uttaranchal"),
    "west bengal": ("west bengal", "wb"),
}

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


def _sanitize_phrase_list(values: Any, *, max_items: int = 8, max_len: int = 48) -> list[str]:
    if not isinstance(values, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        phrase = _normalize_whitespace(re.sub(r"[^a-zA-Z0-9\s/-]", " ", value))
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(phrase[:max_len])
        if len(cleaned) >= max_items:
            break
    return cleaned


def _normalize_intent_name(value: str | None) -> str:
    normalized = _normalize_whitespace((value or "").lower()).replace(" ", "_")
    aliases = {
        "hackathon": "hackathons",
        "internship": "internships",
        "conference": "conferences",
        "scholarship": "scholarships",
        "job": "jobs",
        "general": "general_student_opportunity",
        "student_opportunities": "general_student_opportunity",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in SUPPORTED_INTENTS:
        return normalized
    return "unknown"


def _normalize_choice(value: str | None, allowed: set[str], fallback: str) -> str:
    normalized = _normalize_whitespace(value or "").lower()
    if normalized in allowed:
        return normalized
    return fallback


def _detect_timeframe_from_text(text: str) -> str:
    lowered = text.lower()
    for timeframe, terms in TIMEFRAME_TERMS.items():
        if any(term in lowered for term in terms):
            return timeframe
    return "upcoming"


def _detect_location_scope_from_text(text: str, nearby: bool, nearby_location: str | None) -> str:
    if nearby:
        return "nearby"

    lowered = text.lower()
    if any(term in lowered for term in LOCATION_SCOPE_TERMS["global"]):
        return "global"
    if any(term in lowered for term in LOCATION_SCOPE_TERMS["country"]):
        return "country"
    if nearby_location:
        return "city"
    if re.search(r"\b(in|at)\s+[a-z]{3,}", lowered):
        return "city"
    return "any"


def _build_negative_terms(exclude_terms: list[str]) -> list[str]:
    negatives: list[str] = []
    for term in exclude_terms:
        tokens = [token for token in term.split() if token]
        if not tokens:
            continue
        for token in tokens[:2]:
            lowered = token.lower()
            if len(lowered) < 3:
                continue
            negatives.append(f"-{lowered}")
            if len(negatives) >= 10:
                return negatives
    return negatives


def _normalize_accuracy_mode(value: str | None) -> str:
    normalized = _normalize_whitespace(value or "").lower()
    if normalized in {"fast", "balanced", "max"}:
        return normalized
    return "max"


def _normalize_geo_scope(value: str | None) -> str:
    normalized = _normalize_whitespace(value or "").lower()
    if normalized in {"state_remote", "strict_state", "soft"}:
        return normalized
    return "state_remote"


def _extract_state_from_text(text: str) -> str | None:
    lowered = text.lower()
    for canonical, aliases in INDIA_STATE_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
            return canonical
    return None


def _detect_remote_hint(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in REMOTE_TERMS)


def _extract_query_constraints(
    question: str,
    *,
    geo_scope: str,
    nearby: bool,
    nearby_location: str | None,
) -> dict[str, Any]:
    normalized_question = _normalize_whitespace(question)
    lowered = normalized_question.lower()

    require_cash_prize = bool(
        re.search(r"\b(cash\s+prize|prize\s+money|prize\s+pool|with\s+prize)\b", lowered)
    )
    requested_state = _extract_state_from_text(lowered)
    if not requested_state and nearby_location:
        requested_state = _extract_state_from_text(nearby_location)

    allow_remote = False
    normalized_scope = _normalize_geo_scope(geo_scope)
    if requested_state:
        allow_remote = normalized_scope == "state_remote" or _detect_remote_hint(lowered)
    elif nearby:
        allow_remote = True

    return {
        "require_cash_prize": require_cash_prize,
        "requested_state": requested_state,
        "allow_remote": allow_remote,
    }


def _extract_amount_from_segment(segment: str) -> tuple[str | None, float | None, str | None]:
    currency_match = PRIZE_CURRENCY_PATTERN.search(segment)
    if currency_match:
        currency = _normalize_currency_token(currency_match.group("currency"))
        amount = _parse_scaled_amount(
            currency_match.group("amount"),
            currency_match.group("scale"),
        )
        if currency and amount is not None:
            return _format_cash_prize(currency, amount), amount, currency

    suffix_match = PRIZE_SUFFIX_CURRENCY_PATTERN.search(segment)
    if suffix_match:
        currency = _normalize_currency_token(suffix_match.group("currency"))
        amount = _parse_scaled_amount(
            suffix_match.group("amount"),
            suffix_match.group("scale"),
        )
        if currency and amount is not None:
            return _format_cash_prize(currency, amount), amount, currency

    return None, None, None


def _extract_snippet_with_terms(text: str, terms: tuple[str, ...], max_len: int = 180) -> str | None:
    for segment in [seg.strip() for seg in re.split(r"[\n|.;]", text) if seg.strip()]:
        lowered = segment.lower()
        if any(term in lowered for term in terms):
            return _normalize_whitespace(segment)[:max_len]
    return None


def _resolve_prize_and_fee(
    primary_text: str | None,
    context_text: str | None = None,
) -> dict[str, Any]:
    merged = _normalize_whitespace(" ".join([primary_text or "", context_text or ""]))
    if not merged:
        return {
            "cash_prize": None,
            "prize_display_text": None,
            "cash_prize_amount": None,
            "cash_prize_currency": None,
            "prize_type": None,
            "registration_fee_text": None,
            "registration_fee_amount": None,
            "registration_fee_currency": None,
            "prize_confidence": None,
            "evidence": [],
        }

    segments = [segment.strip() for segment in re.split(r"[|.;\n]", merged) if segment.strip()]

    prize_candidate: tuple[str | None, float | None, str | None, str | None] = (None, None, None, None)
    fee_candidate: tuple[str | None, float | None, str | None, str | None] = (None, None, None, None)

    for segment in segments:
        lowered = segment.lower()
        has_fee_context = any(term in lowered for term in FEE_CONTEXT_TERMS)
        has_prize_context = any(term in lowered for term in PRIZE_CONTEXT_TERMS)
        has_non_cash_prize = any(term in lowered for term in NON_CASH_PRIZE_TERMS)
        parsed_text, parsed_amount, parsed_currency = _extract_amount_from_segment(segment)

        if has_prize_context and parsed_text and not has_fee_context and prize_candidate[0] is None:
            prize_candidate = (parsed_text, parsed_amount, parsed_currency, segment)
        elif has_prize_context and has_non_cash_prize and prize_candidate[0] is None:
            normalized_non_cash = _normalize_prize_text(segment)
            if normalized_non_cash:
                prize_candidate = (normalized_non_cash, None, None, segment)

        if has_fee_context and parsed_text and fee_candidate[0] is None:
            fee_candidate = (parsed_text, parsed_amount, parsed_currency, segment)

    prize_display_text = _normalize_prize_text(prize_candidate[0])
    prize_type: str | None = None
    cash_prize = None
    cash_prize_amount: float | None = None
    cash_prize_currency: str | None = None
    prize_confidence: float | None = None

    if prize_display_text:
        if prize_candidate[1] is not None and prize_candidate[2]:
            prize_type = "cash"
            cash_prize = prize_display_text
            cash_prize_amount = prize_candidate[1]
            cash_prize_currency = prize_candidate[2]
            prize_confidence = 0.95
        else:
            prize_type = "non_cash"
            prize_confidence = 0.7

    registration_fee_text = _normalize_prize_text(fee_candidate[0])
    registration_fee_amount = fee_candidate[1]
    registration_fee_currency = fee_candidate[2]

    evidence: list[dict[str, Any]] = []
    if prize_candidate[3]:
        evidence.append(
            {
                "field": "prize",
                "snippet": _normalize_whitespace(prize_candidate[3])[:180],
                "confidence": prize_confidence or 0.7,
            }
        )
    if fee_candidate[3]:
        evidence.append(
            {
                "field": "fee",
                "snippet": _normalize_whitespace(fee_candidate[3])[:180],
                "confidence": 0.9,
            }
        )

    return {
        "cash_prize": cash_prize,
        "prize_display_text": prize_display_text,
        "cash_prize_amount": cash_prize_amount,
        "cash_prize_currency": cash_prize_currency,
        "prize_type": prize_type,
        "registration_fee_text": registration_fee_text,
        "registration_fee_amount": registration_fee_amount,
        "registration_fee_currency": registration_fee_currency,
        "prize_confidence": prize_confidence,
        "evidence": evidence,
    }


def _evaluate_geo_alignment(
    *,
    location: str | None,
    text_context: str,
    constraints: dict[str, Any],
    geo_scope: str,
) -> tuple[str, str | None, bool | None, float | None, str | None]:
    requested_state = str(constraints.get("requested_state") or "").strip().lower() or None
    normalized_scope = _normalize_geo_scope(geo_scope)

    location_text = _normalize_whitespace(location or "")
    merged = _normalize_whitespace(" ".join([location_text, text_context]))
    resolved_region = _extract_state_from_text(merged)
    is_remote = _detect_remote_hint(merged)
    location_evidence = _extract_snippet_with_terms(
        merged,
        tuple([resolved_region] if resolved_region else REMOTE_TERMS),
    )

    if not requested_state:
        if resolved_region:
            return "exact", resolved_region, is_remote, 0.75, location_evidence
        if is_remote:
            return "remote", None, True, 0.7, location_evidence
        return "unknown", None, None, 0.4, location_evidence

    if is_remote:
        if normalized_scope == "strict_state":
            return "mismatch", resolved_region, True, 0.8, location_evidence
        return "remote", resolved_region, True, 0.85, location_evidence

    if resolved_region == requested_state:
        return "exact", resolved_region, False, 0.95, location_evidence
    if resolved_region and resolved_region != requested_state:
        return "mismatch", resolved_region, False, 0.95, location_evidence
    return "unknown", None, None, 0.35, location_evidence


def _passes_geo_filter(geo_match: str, geo_scope: str) -> bool:
    normalized_scope = _normalize_geo_scope(geo_scope)
    if normalized_scope == "strict_state":
        return geo_match == "exact"
    if normalized_scope == "state_remote":
        return geo_match in {"exact", "remote"}
    return geo_match in {"exact", "remote", "unknown"}


def _build_summary_structured(event: DiscoverEvent) -> dict[str, str | None]:
    who = None
    if event.short_description:
        who_segment = _extract_snippet_with_terms(
            event.short_description,
            ("students", "student", "undergraduate", "graduates", "freshers"),
            max_len=120,
        )
        who = who_segment
    return {
        "what": (event.name or "")[:120] or None,
        "who": who,
        "prize": event.prize_display_text,
        "location": event.location,
        "deadline": event.registration_deadline or event.date,
    }


def _attach_source_to_evidence(
    evidence: list[dict[str, Any]],
    *,
    source_url: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        snippet = _normalize_whitespace(str(item.get("snippet") or ""))
        if not snippet:
            continue
        output.append(
            {
                "field": str(item.get("field") or "unknown"),
                "snippet": snippet[:200],
                "source_url": source_url,
                "confidence": float(item.get("confidence") or 0.5),
            }
        )
    return output


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
    resolved = _resolve_prize_and_fee(primary_text=text, context_text=None)
    prize_type = resolved.get("prize_type") or "unknown"
    return (
        resolved.get("prize_display_text"),
        resolved.get("cash_prize_amount"),
        resolved.get("cash_prize_currency"),
        prize_type,
    )


def _derive_prize_fields(
    primary_text: str | None,
    context_text: str | None = None,
) -> tuple[Optional[str], Optional[str], Optional[float], Optional[str], Optional[str]]:
    resolved = _resolve_prize_and_fee(primary_text=primary_text, context_text=context_text)
    normalized_type = resolved["prize_type"] if resolved["prize_type"] else None
    return (
        resolved["cash_prize"],
        resolved["prize_display_text"],
        resolved["cash_prize_amount"],
        resolved["cash_prize_currency"],
        normalized_type,
    )


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
) -> tuple[DiscoverEvent | None, int]:
    title = _clean_title(_normalize_whitespace(result.get("title") or ""))
    snippet = _normalize_whitespace(result.get("content") or "")
    url = (result.get("url") or "").strip()

    if not title or not url:
        return None, 0

    text_for_fields = _normalize_whitespace(f"{title} {snippet}")
    parsed_start, parsed_end, parsed_date_str = _parse_date_range(text_for_fields)
    date_str = parsed_date_str or _extract_first_match(DATE_PATTERNS, text_for_fields)
    location = _sanitize_location(_extract_location(text_for_fields))
    prize_resolution = _resolve_prize_and_fee(
        primary_text=text_for_fields,
        context_text=None,
    )
    short_description = _sanitize_short_description(snippet, title)

    score = 0
    for field in [date_str, location, prize_resolution.get("prize_display_text"), short_description]:
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
        cash_prize=prize_resolution.get("cash_prize"),
        prize_type=prize_resolution.get("prize_type"),
        cash_prize_amount=prize_resolution.get("cash_prize_amount"),
        cash_prize_currency=prize_resolution.get("cash_prize_currency"),
        prize_display_text=prize_resolution.get("prize_display_text"),
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
        registration_fee_text=prize_resolution.get("registration_fee_text"),
        registration_fee_amount=prize_resolution.get("registration_fee_amount"),
        registration_fee_currency=prize_resolution.get("registration_fee_currency"),
        prize_confidence=prize_resolution.get("prize_confidence"),
        summary_structured={
            "what": _sanitize_event_name(title)[:120],
            "who": None,
            "prize": prize_resolution.get("prize_display_text"),
            "location": location,
            "deadline": (
                registration_deadline_date.isoformat()
                if registration_deadline_date
                else registration_deadline_text
            ),
        },
        evidence=_attach_source_to_evidence(
            prize_resolution.get("evidence") or [],
            source_url=url,
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
    if event.registration_fee_text:
        flags.append("has_registration_fee")
    if event.short_description:
        flags.append("has_description")
    if event.registration_status:
        flags.append(f"registration_{event.registration_status}")
    if event.registration_deadline:
        flags.append("has_registration_deadline")
    if event.geo_match:
        flags.append(f"geo_{event.geo_match}")
    if event.is_remote:
        flags.append("is_remote")
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
        self.verify_model = genai.GenerativeModel("gemini-2.5-flash")
        mode = (settings.event_discovery_extractor_mode or "hybrid").strip().lower()
        self.extractor_mode = mode if mode in {"legacy", "crawl4ai", "hybrid"} else "hybrid"
        self.crawl4ai_max_urls = max(0, int(settings.event_discovery_crawl4ai_max_urls))
        self.crawl4ai_extractor = Crawl4AIExtractor(
            timeout_ms=max(4000, int(settings.event_discovery_crawl4ai_timeout_ms)),
            max_chars=18000,
            check_robots_txt=bool(settings.event_discovery_crawl4ai_check_robots_txt),
        )

    def _fallback_intent_context(
        self,
        question: str,
        category_hint: str,
        nearby: bool,
        nearby_location: str | None,
        geo_scope: str = "state_remote",
    ) -> dict[str, Any]:
        normalized_question = _normalize_whitespace(question)
        processed = query_processor.process(normalized_question)
        lowered = normalized_question.lower()

        intent = "unknown"
        if category_hint and category_hint != "all":
            intent = _normalize_intent_name(category_hint)
        else:
            intent_signals: list[tuple[str, tuple[str, ...]]] = [
                (
                    "hackathons",
                    ("hackathon", "ideathon", "coding challenge", "contest", "competition"),
                ),
                ("internships", ("internship", "intern", "training")),
                (
                    "conferences",
                    ("conference", "symposium", "seminar", "workshop", "summit", "call for papers"),
                ),
                ("scholarships", ("scholarship", "fellowship", "grant")),
                ("jobs", ("job", "jobs", "fresher role", "entry level", "placement", "hiring")),
            ]
            for signal_intent, signals in intent_signals:
                if any(signal in lowered for signal in signals):
                    intent = signal_intent
                    break
            if intent == "unknown" and (
                "student" in lowered or "college" in lowered or "university" in lowered
            ):
                intent = "general_student_opportunity"

        search_required = processed.intent == QueryIntent.QUESTION and (
            len(processed.keywords) >= 2 or intent != "unknown"
        )
        timeframe = _detect_timeframe_from_text(normalized_question)
        location_scope = _detect_location_scope_from_text(
            normalized_question,
            nearby,
            nearby_location,
        )
        focus_terms = _sanitize_phrase_list(processed.keywords[:8], max_items=8, max_len=28)

        rewrite_parts = [normalized_question]
        intent_phrase_map = {
            "hackathons": "student hackathons",
            "internships": "student internships",
            "conferences": "student conferences",
            "scholarships": "student scholarships",
            "jobs": "entry level student roles",
        }
        intent_phrase = intent_phrase_map.get(intent)
        if intent_phrase and intent_phrase not in lowered:
            rewrite_parts.append(intent_phrase)
        if timeframe == "upcoming" and not re.search(r"\b20\d{2}\b", lowered):
            rewrite_parts.append(str(datetime.now().year))
        if nearby and nearby_location:
            rewrite_parts.append(f"in {nearby_location.strip()}")
        elif nearby:
            rewrite_parts.append("near me")

        rewritten_query = _normalize_whitespace(" ".join(rewrite_parts))[:220]
        confidence = (
            0.75
            if category_hint and category_hint != "all"
            else (0.65 if intent != "unknown" else 0.35)
        )

        return {
            "intent": intent,
            "search_required": search_required,
            "confidence": confidence,
            "timeframe": timeframe,
            "location_scope": location_scope,
            "focus_terms": focus_terms,
            "exclude_terms": [],
            "rewritten_query": rewritten_query,
            "reason": "heuristic fallback",
            "geo_scope": _normalize_geo_scope(geo_scope),
        }

    def _classify_intent_context(
        self,
        question: str,
        category_hint: str,
        nearby: bool,
        nearby_location: str | None,
        geo_scope: str = "state_remote",
    ) -> dict[str, Any]:
        fallback = self._fallback_intent_context(
            question,
            category_hint,
            nearby,
            nearby_location,
            geo_scope=geo_scope,
        )
        prompt = INTENT_CLASSIFIER_PROMPT.format(
            question=_normalize_whitespace(question),
            category_hint=(category_hint or "all").strip() or "all",
            nearby=str(bool(nearby)).lower(),
            nearby_location=(nearby_location or "").strip(),
            current_year=datetime.now().year,
        )

        try:
            response = self.query_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 260,
                },
            )
            data = _safe_json_loads(response.text or "", fallback={})
        except Exception as e:
            logger.warning("Intent classification failed, using fallback: %s", e)
            return fallback

        if not isinstance(data, dict):
            return fallback

        context = dict(fallback)
        intent = _normalize_intent_name(data.get("intent"))
        if intent != "unknown":
            context["intent"] = intent

        if category_hint and category_hint != "all":
            context["intent"] = _normalize_intent_name(category_hint)

        search_required = data.get("search_required")
        if isinstance(search_required, bool):
            context["search_required"] = search_required

        confidence = data.get("confidence")
        if isinstance(confidence, (int, float)):
            context["confidence"] = round(max(0.0, min(1.0, float(confidence))), 2)

        context["timeframe"] = _normalize_choice(
            data.get("timeframe"),
            {"upcoming", "ongoing", "past", "any"},
            fallback["timeframe"],
        )
        context["location_scope"] = _normalize_choice(
            data.get("location_scope"),
            {"nearby", "city", "country", "global", "any"},
            fallback["location_scope"],
        )

        focus_terms = _sanitize_phrase_list(data.get("focus_terms"), max_items=8, max_len=48)
        if focus_terms:
            context["focus_terms"] = focus_terms

        exclude_terms = _sanitize_phrase_list(data.get("exclude_terms"), max_items=6, max_len=48)
        if exclude_terms:
            context["exclude_terms"] = exclude_terms

        rewritten_query = _normalize_whitespace(str(data.get("rewritten_query") or ""))
        if rewritten_query:
            context["rewritten_query"] = rewritten_query[:220]

        reason = _normalize_whitespace(str(data.get("reason") or ""))
        if reason:
            context["reason"] = reason[:180]

        # Preserve practical behavior: if intent is unknown and the classifier has low confidence,
        # do not trigger web search.
        if context["intent"] == "unknown" and float(context.get("confidence") or 0.0) < 0.35:
            context["search_required"] = False

        context["geo_scope"] = _normalize_geo_scope(geo_scope)
        return context

    def evaluate_policy(
        self,
        question: str,
        category_hint: str,
        nearby: bool = False,
        nearby_location: str | None = None,
        geo_scope: str = "state_remote",
    ) -> ScopeDecision:
        intent_context = self._classify_intent_context(
            question=question,
            category_hint=category_hint or "all",
            nearby=nearby,
            nearby_location=nearby_location,
            geo_scope=geo_scope,
        )
        decision = classify_discovery_scope(
            question,
            category_hint=category_hint or "all",
            intent_context=intent_context,
        )
        decision.intent_context = self._augment_intent_context_with_constraints(
            question=question,
            nearby=nearby,
            nearby_location=nearby_location,
            geo_scope=geo_scope,
            intent_context=decision.intent_context,
        )
        return decision

    def _compute_budgets(self, accuracy_mode: str) -> tuple[int, int]:
        base_budget = max(0, settings.event_discovery_gemini_budget)
        normalized_mode = _normalize_accuracy_mode(accuracy_mode)
        if normalized_mode == "fast":
            return base_budget, max(0, base_budget // 2)
        if normalized_mode == "balanced":
            return base_budget * 2, max(1, base_budget)
        return base_budget * 3, max(2, base_budget * 2)

    def _augment_intent_context_with_constraints(
        self,
        question: str,
        *,
        nearby: bool,
        nearby_location: str | None,
        geo_scope: str,
        intent_context: dict[str, Any],
    ) -> dict[str, Any]:
        context = dict(intent_context or {})
        constraints = _extract_query_constraints(
            question,
            geo_scope=geo_scope,
            nearby=nearby,
            nearby_location=nearby_location,
        )
        focus_terms = _sanitize_phrase_list(context.get("focus_terms"), max_items=8, max_len=48)
        exclude_terms = _sanitize_phrase_list(context.get("exclude_terms"), max_items=8, max_len=48)

        if constraints.get("require_cash_prize"):
            focus_terms.extend(["cash prize", "prize pool", "winning amount"])
            exclude_terms.extend(["registration fee", "entry fee", "payment required"])

        requested_state = constraints.get("requested_state")
        if requested_state:
            focus_terms.append(requested_state.title())
            context["location_scope"] = "city"

        context["focus_terms"] = _sanitize_phrase_list(focus_terms, max_items=8, max_len=48)
        context["exclude_terms"] = _sanitize_phrase_list(exclude_terms, max_items=8, max_len=48)
        context["constraints"] = constraints
        context["geo_scope"] = _normalize_geo_scope(geo_scope)
        return context

    def _verify_event_fields(
        self,
        *,
        question: str,
        event: DiscoverEvent,
        context_text: str,
        source_url: str,
    ) -> dict[str, Any]:
        prompt = f"""
Validate extracted event fields for high-accuracy event discovery.

Question: {question}
Event name: {event.name or ""}
Extracted location: {event.location or ""}
Extracted prize text: {event.prize_display_text or event.cash_prize or ""}
Extracted registration fee text: {event.registration_fee_text or ""}
Registration status: {event.registration_status or ""}
Registration deadline: {event.registration_deadline or ""}
Start date: {event.start_date or ""}

Context:
{_truncate(context_text, 2400)}

Return JSON only:
{{
  "verified_prize_type": "cash|non_cash|none|fee_only|unknown",
  "verified_cash_prize_text": "string|null",
  "verified_fee_text": "string|null",
  "verified_location": "string|null",
  "resolved_region": "string|null",
  "is_remote": true,
  "prize_confidence": 0.0,
  "location_confidence": 0.0,
  "summary_structured": {{
    "what": "string|null",
    "who": "string|null",
    "prize": "string|null",
    "location": "string|null",
    "deadline": "string|null"
  }},
  "evidence": [
    {{"field":"prize|fee|location|deadline","snippet":"string","confidence":0.0}}
  ]
}}
"""
        try:
            response = self.verify_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 320,
                },
            )
            data = _safe_json_loads(response.text or "", fallback={})
        except Exception as e:
            logger.warning("Event verification failed for %s: %s", source_url, e)
            return {}

        if not isinstance(data, dict):
            return {}
        verified = {
            "verified_prize_type": str(data.get("verified_prize_type") or "unknown").strip().lower(),
            "verified_cash_prize_text": _normalize_prize_text(str(data.get("verified_cash_prize_text") or "")),
            "verified_fee_text": _normalize_prize_text(str(data.get("verified_fee_text") or "")),
            "verified_location": _sanitize_location(str(data.get("verified_location") or "")),
            "resolved_region": _extract_state_from_text(str(data.get("resolved_region") or "")),
            "is_remote": bool(data.get("is_remote")) if isinstance(data.get("is_remote"), bool) else None,
            "prize_confidence": None,
            "location_confidence": None,
            "summary_structured": data.get("summary_structured")
            if isinstance(data.get("summary_structured"), dict)
            else None,
            "evidence": _attach_source_to_evidence(
                data.get("evidence") if isinstance(data.get("evidence"), list) else [],
                source_url=source_url,
            ),
        }
        prize_confidence = data.get("prize_confidence")
        if isinstance(prize_confidence, (int, float)):
            verified["prize_confidence"] = round(max(0.0, min(1.0, float(prize_confidence))), 2)
        location_confidence = data.get("location_confidence")
        if isinstance(location_confidence, (int, float)):
            verified["location_confidence"] = round(max(0.0, min(1.0, float(location_confidence))), 2)

        return verified

    def build_search_plan(
        self,
        question: str,
        normalized_intent: str,
        nearby: bool,
        nearby_location: str | None,
        max_results: int,
        intent_context: dict[str, Any] | None = None,
    ) -> dict:
        context = intent_context if isinstance(intent_context, dict) else {}
        intent_token_map = {
            "hackathons": "hackathons",
            "internships": "student internships",
            "conferences": "student conferences",
            "scholarships": "student scholarships",
            "jobs": "entry level roles for students",
            "general_student_opportunity": "student opportunities",
        }
        intent_phrase = intent_token_map.get(normalized_intent, "student opportunities")
        timeframe = _normalize_choice(
            str(context.get("timeframe") or ""),
            {"upcoming", "ongoing", "past", "any"},
            "upcoming",
        )
        location_scope = _normalize_choice(
            str(context.get("location_scope") or ""),
            {"nearby", "city", "country", "global", "any"},
            "any",
        )
        focus_terms = _sanitize_phrase_list(context.get("focus_terms"), max_items=6, max_len=40)
        exclude_terms = _sanitize_phrase_list(context.get("exclude_terms"), max_items=6, max_len=32)
        rewritten_query = _normalize_whitespace(str(context.get("rewritten_query") or question))
        base = rewritten_query or question.strip()

        if (
            normalized_intent != "general_student_opportunity"
            and normalized_intent not in base.lower()
        ):
            base = f"{base} {intent_phrase}".strip()
        if timeframe == "upcoming" and not re.search(r"\b20\d{2}\b", base):
            base = f"{base} {datetime.now().year}".strip()
        if timeframe == "past":
            base = f"{base} previous cycle".strip()
        if location_scope == "global" and "global" not in base.lower():
            base = f"global {base}".strip()

        primary_query = self.build_search_query(
            question=base,
            nearby=nearby,
            nearby_location=nearby_location,
            intent_hint=normalized_intent,
            intent_context=context,
            query_mode="initial",
        )
        expanded_query = self.build_search_query(
            question=f"{base} official portals and trusted listings",
            nearby=nearby,
            nearby_location=nearby_location,
            intent_hint=normalized_intent,
            intent_context=context,
            query_mode="expanded",
        )

        first_stage_limit = max(8, min(max_results, 20))
        second_stage_limit = max(8, min(max_results, 20))
        focus_suffix = " ".join(focus_terms[:3])
        exclusion_suffix = " ".join(_build_negative_terms(exclude_terms))
        initial_fallback_query = _normalize_whitespace(
            f"{question.strip()} {focus_suffix} {datetime.now().year} {exclusion_suffix}"
        )
        expanded_fallback_query = _normalize_whitespace(
            f"{intent_phrase} trusted sources registration deadline {focus_suffix} "
            f"{datetime.now().year} {exclusion_suffix}"
        )

        return {
            "intent": normalized_intent,
            "classifier": {
                "timeframe": timeframe,
                "location_scope": location_scope,
                "confidence": float(context.get("confidence") or 0.0),
                "search_required": bool(context.get("search_required", True)),
                "geo_scope": _normalize_geo_scope(str(context.get("geo_scope") or "state_remote")),
            },
            "stages": [
                {
                    "id": "initial",
                    "label": f"Searching sites for {question.strip()}.",
                    "queries": [
                        primary_query,
                        initial_fallback_query or f"{question.strip()} {datetime.now().year}",
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
                        (
                            expanded_fallback_query
                            or f"{intent_phrase} trusted sources {datetime.now().year}"
                        ),
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
        intent_context: dict[str, Any] | None = None,
        query_mode: str = "initial",
    ) -> str:
        context = intent_context if isinstance(intent_context, dict) else {}
        timeframe = _normalize_choice(
            str(context.get("timeframe") or ""),
            {"upcoming", "ongoing", "past", "any"},
            "upcoming",
        )
        location_scope = _normalize_choice(
            str(context.get("location_scope") or ""),
            {"nearby", "city", "country", "global", "any"},
            "any",
        )
        focus_terms = _sanitize_phrase_list(context.get("focus_terms"), max_items=6, max_len=40)
        context_exclude_terms = _sanitize_phrase_list(
            context.get("exclude_terms"),
            max_items=6,
            max_len=32,
        )

        prompt = SEARCH_PROMPT.format(
            question=question.strip(),
            nearby=str(nearby).lower(),
            nearby_location=(nearby_location or "").strip(),
            intent=(intent_hint or "general_student_opportunity"),
            timeframe=timeframe,
            location_scope=location_scope,
            focus_terms=", ".join(focus_terms) if focus_terms else "none",
            exclude_terms=", ".join(context_exclude_terms) if context_exclude_terms else "none",
            query_mode=query_mode,
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
            query = _normalize_whitespace(str(data.get("query") or question))
            must_include = _sanitize_phrase_list(data.get("must_include"), max_items=6, max_len=36)
            exclude = _sanitize_phrase_list(data.get("exclude"), max_items=6, max_len=24)
        except Exception as e:
            logger.warning("Query augmentation failed: %s", e)
            query = question
            must_include = []
            exclude = []

        intent_extras_map = {
            "hackathons": ["hackathon", "coding challenge", "registration deadline"],
            "internships": ["student internship", "apply now", "eligibility"],
            "conferences": ["conference", "call for papers", "submission deadline"],
            "scholarships": ["scholarship", "fellowship", "eligibility"],
            "jobs": ["entry level", "fresher", "application deadline"],
            "general_student_opportunity": ["student opportunity", "registration", "deadline"],
        }
        extras = intent_extras_map.get(intent_hint or "", ["student opportunity", "college event"])
        extras.extend(focus_terms[:3])
        if query_mode == "expanded":
            extras.extend(["official", "trusted sources"])
        if timeframe == "upcoming":
            extras.append(str(datetime.now().year))
        if timeframe == "past":
            extras.append("previous cycle")
        if intent_hint and intent_hint not in {"unknown", "general_student_opportunity", "all"}:
            extras.append(intent_hint)
        if nearby:
            if nearby_location:
                extras.append(f"in {nearby_location.strip()}")
            else:
                extras.append("near me")
        elif location_scope == "global":
            extras.append("global")

        tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in [query] + must_include + extras:
            cleaned = _normalize_whitespace(token)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_tokens:
                continue
            seen_tokens.add(key)
            tokens.append(cleaned)

        negative_terms = _build_negative_terms(exclude + context_exclude_terms)
        tokens.extend(negative_terms)

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

            prize_resolution = _resolve_prize_and_fee(
                primary_text=_normalize_whitespace(
                    " ".join(
                        [
                            str(item.get("cash_prize") or "").strip(),
                            str(item.get("registration_fee") or "").strip(),
                        ]
                    )
                ),
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
                cash_prize=prize_resolution.get("cash_prize"),
                prize_type=prize_resolution.get("prize_type"),
                cash_prize_amount=prize_resolution.get("cash_prize_amount"),
                cash_prize_currency=prize_resolution.get("cash_prize_currency"),
                prize_display_text=prize_resolution.get("prize_display_text"),
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
                registration_fee_text=prize_resolution.get("registration_fee_text"),
                registration_fee_amount=prize_resolution.get("registration_fee_amount"),
                registration_fee_currency=prize_resolution.get("registration_fee_currency"),
                prize_confidence=prize_resolution.get("prize_confidence"),
                summary_structured={
                    "what": name[:120],
                    "who": None,
                    "prize": prize_resolution.get("prize_display_text"),
                    "location": normalized_location,
                    "deadline": (
                        registration_deadline_date.isoformat()
                        if registration_deadline_date
                        else registration_deadline_text
                    ),
                },
                evidence=_attach_source_to_evidence(
                    prize_resolution.get("evidence") or [],
                    source_url=source_url,
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
        accuracy_mode: str = "max",
        geo_scope: str = "state_remote",
    ) -> Iterable[tuple[str, dict]]:
        start_time = time.perf_counter()
        normalized_accuracy_mode = _normalize_accuracy_mode(accuracy_mode)
        normalized_geo_scope = _normalize_geo_scope(geo_scope)
        policy = self.evaluate_policy(
            question=question,
            category_hint=category_hint,
            nearby=nearby,
            nearby_location=nearby_location,
            geo_scope=normalized_geo_scope,
        )
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
            intent_context=policy.intent_context,
        )
        yield ("search_plan", search_plan)

        include_domains = source_registry.get_include_domains(
            category_hint=policy.normalized_intent,
            strict_trust=strict_trust,
            max_domains=50,
        )
        scoring_query = str(policy.intent_context.get("rewritten_query") or question)
        processed_query = query_processor.process(scoring_query)

        seen_citation_urls: set[str] = set()
        processed_result_urls: set[str] = set()
        seen_events: set[str] = set()
        citations_emitted = 0
        events_emitted = 0
        gemini_budget, verification_budget = self._compute_budgets(normalized_accuracy_mode)
        first_citation_ms: int | None = None
        first_event_ms: int | None = None
        crawl4ai_attempts = 0
        crawl4ai_hits = 0
        constraints = policy.intent_context.get("constraints") or {}

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
            requested_state = str(constraints.get("requested_state") or "").strip().lower()
            if requested_state:
                combined = f"{title} {snippet}".lower()
                if requested_state in combined:
                    score += 5
                if any(term in combined for term in REMOTE_TERMS):
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
                        prize_resolution = _resolve_prize_and_fee(
                            primary_text=event.cash_prize or event.prize_display_text,
                            context_text=normalization_context,
                        )
                        event.cash_prize = prize_resolution.get("cash_prize")
                        event.prize_display_text = prize_resolution.get("prize_display_text")
                        event.cash_prize_amount = prize_resolution.get("cash_prize_amount")
                        event.cash_prize_currency = prize_resolution.get("cash_prize_currency")
                        event.prize_type = prize_resolution.get("prize_type")
                        event.registration_fee_text = prize_resolution.get("registration_fee_text")
                        event.registration_fee_amount = prize_resolution.get("registration_fee_amount")
                        event.registration_fee_currency = prize_resolution.get("registration_fee_currency")
                        event.prize_confidence = prize_resolution.get("prize_confidence")

                        evidence_items: list[dict[str, Any]] = []
                        if event.evidence:
                            evidence_items.extend(
                                _attach_source_to_evidence(event.evidence, source_url=canonical_url)
                            )
                        evidence_items.extend(
                            _attach_source_to_evidence(
                                prize_resolution.get("evidence") or [],
                                source_url=canonical_url,
                            )
                        )

                        (
                            geo_match,
                            resolved_region,
                            is_remote,
                            location_confidence,
                            location_evidence,
                        ) = _evaluate_geo_alignment(
                            location=event.location,
                            text_context=normalization_context,
                            constraints=constraints,
                            geo_scope=normalized_geo_scope,
                        )
                        event.geo_match = geo_match
                        event.resolved_region = resolved_region
                        event.is_remote = is_remote
                        event.location_confidence = location_confidence
                        if location_evidence:
                            evidence_items.append(
                                {
                                    "field": "location",
                                    "snippet": location_evidence[:200],
                                    "source_url": canonical_url,
                                    "confidence": location_confidence or 0.6,
                                }
                            )

                        if constraints.get("requested_state") and not _passes_geo_filter(
                            event.geo_match or "unknown",
                            normalized_geo_scope,
                        ):
                            continue

                        if constraints.get("require_cash_prize"):
                            if event.prize_type != "cash":
                                continue
                            if (event.prize_confidence or 0.0) < 0.6:
                                continue

                        parsed_start_date: date | None = None
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

                        if verification_budget > 0 and normalized_accuracy_mode != "fast":
                            verified = self._verify_event_fields(
                                question=question,
                                event=event,
                                context_text=normalization_context,
                                source_url=canonical_url,
                            )
                            verification_budget -= 1
                            if verified:
                                verified_type = str(
                                    verified.get("verified_prize_type") or ""
                                ).strip().lower()
                                if verified_type == "fee_only":
                                    event.cash_prize = None
                                    event.prize_display_text = None
                                    event.cash_prize_amount = None
                                    event.cash_prize_currency = None
                                    event.prize_type = None
                                elif verified_type == "cash" and verified.get("verified_cash_prize_text"):
                                    verified_cash_text = str(
                                        verified.get("verified_cash_prize_text") or ""
                                    ).strip()
                                    parsed_text, parsed_amount, parsed_currency = (
                                        _extract_amount_from_segment(verified_cash_text)
                                    )
                                    if parsed_text and parsed_amount is not None and parsed_currency:
                                        event.cash_prize = parsed_text
                                        event.prize_display_text = parsed_text
                                        event.cash_prize_amount = parsed_amount
                                        event.cash_prize_currency = parsed_currency
                                        event.prize_type = "cash"
                                    else:
                                        verified_prize = _resolve_prize_and_fee(
                                            primary_text=verified_cash_text,
                                            context_text=normalization_context,
                                        )
                                        event.cash_prize = verified_prize.get("cash_prize")
                                        event.prize_display_text = verified_prize.get(
                                            "prize_display_text"
                                        )
                                        event.cash_prize_amount = verified_prize.get(
                                            "cash_prize_amount"
                                        )
                                        event.cash_prize_currency = verified_prize.get(
                                            "cash_prize_currency"
                                        )
                                        event.prize_type = verified_prize.get("prize_type")
                                verified_fee = verified.get("verified_fee_text")
                                if verified_fee:
                                    fee_resolution = _resolve_prize_and_fee(
                                        primary_text=str(verified_fee),
                                        context_text=None,
                                    )
                                    event.registration_fee_text = fee_resolution.get("registration_fee_text")
                                    event.registration_fee_amount = fee_resolution.get(
                                        "registration_fee_amount"
                                    )
                                    event.registration_fee_currency = fee_resolution.get(
                                        "registration_fee_currency"
                                    )

                                verified_location = verified.get("verified_location")
                                if verified_location:
                                    event.location = _sanitize_location(str(verified_location)) or event.location
                                if verified.get("resolved_region"):
                                    event.resolved_region = str(verified.get("resolved_region"))
                                if isinstance(verified.get("is_remote"), bool):
                                    event.is_remote = bool(verified.get("is_remote"))
                                if isinstance(verified.get("prize_confidence"), (int, float)):
                                    event.prize_confidence = float(verified.get("prize_confidence"))
                                if isinstance(verified.get("location_confidence"), (int, float)):
                                    event.location_confidence = float(verified.get("location_confidence"))

                                if isinstance(verified.get("summary_structured"), dict):
                                    event.summary_structured = {
                                        "what": str(
                                            (verified["summary_structured"].get("what") or event.name)
                                        )[:120],
                                        "who": str(
                                            verified["summary_structured"].get("who") or ""
                                        )[:120]
                                        or None,
                                        "prize": str(
                                            verified["summary_structured"].get("prize")
                                            or event.prize_display_text
                                            or ""
                                        )[:80]
                                        or None,
                                        "location": str(
                                            verified["summary_structured"].get("location")
                                            or event.location
                                            or ""
                                        )[:80]
                                        or None,
                                        "deadline": str(
                                            verified["summary_structured"].get("deadline")
                                            or event.registration_deadline
                                            or event.date
                                            or ""
                                        )[:80]
                                        or None,
                                    }
                                evidence_items.extend(verified.get("evidence") or [])

                        if not event.summary_structured:
                            event.summary_structured = _build_summary_structured(event)

                        if constraints.get("requested_state"):
                            (
                                event.geo_match,
                                event.resolved_region,
                                event.is_remote,
                                event.location_confidence,
                                _,
                            ) = _evaluate_geo_alignment(
                                location=event.location,
                                text_context=normalization_context,
                                constraints=constraints,
                                geo_scope=normalized_geo_scope,
                            )
                            if not _passes_geo_filter(
                                event.geo_match or "unknown",
                                normalized_geo_scope,
                            ):
                                continue

                        if constraints.get("require_cash_prize"):
                            if event.prize_type != "cash":
                                continue
                            if (event.prize_confidence or 0.0) < 0.6:
                                continue

                        # De-duplicate evidence snippets by field+text.
                        deduped_evidence: list[dict[str, Any]] = []
                        seen_evidence: set[str] = set()
                        for item in evidence_items:
                            if not isinstance(item, dict):
                                continue
                            field = str(item.get("field") or "")
                            snippet = _normalize_whitespace(str(item.get("snippet") or ""))
                            if not snippet:
                                continue
                            evidence_key = f"{field}:{snippet.lower()}"
                            if evidence_key in seen_evidence:
                                continue
                            seen_evidence.add(evidence_key)
                            deduped_evidence.append(
                                {
                                    "field": field or "unknown",
                                    "snippet": snippet[:200],
                                    "source_url": str(item.get("source_url") or canonical_url),
                                    "confidence": float(item.get("confidence") or 0.5),
                                }
                            )
                        event.evidence = deduped_evidence[:8]

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
                            has_prize=bool(event.prize_type == "cash" and event.prize_display_text),
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
def get_event_discovery() -> EventDiscovery:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    if TavilyClient is None:
        raise RuntimeError("tavily-python is not installed")
    return EventDiscovery()
