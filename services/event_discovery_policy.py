"""
Policy guardrails for Event Discover queries.

This module enforces student-opportunity scope and provides
recommendations when a query is out of scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

DISCOVER_CATEGORIES = {
    "all",
    "hackathons",
    "internships",
    "conferences",
    "scholarships",
    "jobs",
}
DISCOVER_INTENTS = {
    "hackathons",
    "internships",
    "conferences",
    "scholarships",
    "jobs",
    "general_student_opportunity",
    "unknown",
}

INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hackathons": (
        "hackathon",
        "coding challenge",
        "code challenge",
        "competition",
        "contest",
        "ideathon",
    ),
    "internships": (
        "internship",
        "intern",
        "industrial training",
        "summer training",
        "winter training",
    ),
    "conferences": (
        "conference",
        "symposium",
        "seminar",
        "workshop",
        "summit",
        "research event",
        "call for papers",
    ),
    "scholarships": (
        "scholarship",
        "fellowship",
        "grant",
        "research funding",
    ),
    "jobs": (
        "job",
        "jobs",
        "role",
        "roles",
        "placement",
        "career opportunity",
        "recruitment",
        "hiring",
    ),
}

GENERAL_STUDENT_TERMS = (
    "student",
    "college",
    "campus",
    "university",
    "academic",
    "fresher",
    "entry level",
    "graduate",
)

DISALLOWED_TERMS = (
    "phone",
    "laptop",
    "netflix",
    "movie",
    "series",
    "recipe",
    "restaurant",
    "hotel",
    "flight",
    "tour",
    "travel package",
    "crypto signal",
    "casino",
    "betting",
    "fantasy",
    "share tips",
    "stock tips",
    "sports prediction",
    "adult",
    "dating",
)

SENIOR_EXPERIENCE_TERMS = (
    "senior",
    "lead",
    "principal",
    "architect",
    "manager",
    "director",
    "10 years",
    "5 years",
    "experienced only",
)

CATEGORY_RECOMMENDATIONS: dict[str, list[str]] = {
    "hackathons": [
        "AI hackathons in India this month",
        "College hackathons with cash prizes for students",
        "Upcoming ML hackathons with online participation",
    ],
    "internships": [
        "Data science internships for students in 2026",
        "Remote software internships for freshers",
        "Summer internships for CSE students",
    ],
    "conferences": [
        "Student conferences on AI and robotics in 2026",
        "IEEE student symposium deadlines this semester",
        "Workshops and seminars for engineering students",
    ],
    "scholarships": [
        "Scholarships for engineering students in India",
        "International fellowships for undergraduate research",
        "Government scholarship deadlines for college students",
    ],
    "jobs": [
        "Entry-level software roles for freshers in India",
        "Campus placement drives for final year students",
        "Graduate trainee roles in AI and data science",
    ],
}


@dataclass
class ScopeDecision:
    allowed: bool
    reason_code: Literal["ok", "non_academic", "unsafe", "too_broad"]
    normalized_intent: str
    message: str
    recommendations: list[str]
    search_required: bool = True
    intent_context: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "normalized_intent": self.normalized_intent,
            "message": self.message,
            "recommendations": self.recommendations,
            "search_required": self.search_required,
            "intent_context": self.intent_context,
        }


def recommendations_for_intent(intent: str) -> list[str]:
    return CATEGORY_RECOMMENDATIONS.get(intent, CATEGORY_RECOMMENDATIONS["hackathons"])


def _sanitize_intent_context(
    question: str,
    intent: str,
    intent_context: dict[str, Any] | None,
) -> dict[str, Any]:
    context = intent_context if isinstance(intent_context, dict) else {}
    raw_focus = context.get("focus_terms")
    focus_terms = (
        [str(item).strip() for item in raw_focus if str(item).strip()]
        if isinstance(raw_focus, list)
        else []
    )
    raw_exclude = context.get("exclude_terms")
    exclude_terms = (
        [str(item).strip() for item in raw_exclude if str(item).strip()]
        if isinstance(raw_exclude, list)
        else []
    )
    timeframe = str(context.get("timeframe") or "any").strip().lower()
    if timeframe not in {"upcoming", "ongoing", "past", "any"}:
        timeframe = "any"
    location_scope = str(context.get("location_scope") or "any").strip().lower()
    if location_scope not in {"nearby", "city", "country", "global", "any"}:
        location_scope = "any"
    geo_scope = str(context.get("geo_scope") or "state_remote").strip().lower()
    if geo_scope not in {"state_remote", "strict_state", "soft"}:
        geo_scope = "state_remote"
    rewritten_query = str(context.get("rewritten_query") or question).strip()
    if not rewritten_query:
        rewritten_query = question.strip()

    confidence = context.get("confidence")
    confidence_value = 0.0
    if isinstance(confidence, (int, float)):
        confidence_value = max(0.0, min(1.0, float(confidence)))

    search_required_raw = context.get("search_required")
    search_required = bool(search_required_raw) if isinstance(search_required_raw, bool) else True

    reason = str(context.get("reason") or "").strip()
    constraints = context.get("constraints") if isinstance(context.get("constraints"), dict) else {}

    return {
        "intent": intent,
        "search_required": search_required,
        "confidence": round(confidence_value, 2),
        "timeframe": timeframe,
        "location_scope": location_scope,
        "geo_scope": geo_scope,
        "focus_terms": focus_terms[:8],
        "exclude_terms": exclude_terms[:6],
        "rewritten_query": rewritten_query[:220],
        "reason": reason[:180] if reason else "",
        "constraints": constraints,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(p in text for p in patterns)


def _detect_intent(text: str, category_hint: str) -> str:
    if category_hint in DISCOVER_CATEGORIES and category_hint != "all":
        return category_hint

    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[intent] = score

    if not scores:
        if _contains_any(text, GENERAL_STUDENT_TERMS):
            return "general_student_opportunity"
        return "unknown"

    return max(scores.items(), key=lambda item: item[1])[0]


def classify_discovery_scope(
    question: str,
    category_hint: str = "all",
    intent_context: dict[str, Any] | None = None,
) -> ScopeDecision:
    text = _normalize(question)
    if not text:
        fallback_intent = (
            category_hint
            if category_hint in DISCOVER_CATEGORIES and category_hint != "all"
            else "unknown"
        )
        context_payload = _sanitize_intent_context(question, fallback_intent, intent_context)
        return ScopeDecision(
            allowed=False,
            reason_code="too_broad",
            normalized_intent="unknown",
            message="Please describe a student opportunity you want to discover.",
            recommendations=CATEGORY_RECOMMENDATIONS["hackathons"][:2]
            + CATEGORY_RECOMMENDATIONS["internships"][:1],
            search_required=False,
            intent_context=context_payload,
        )

    context_intent = None
    if isinstance(intent_context, dict):
        raw_context_intent = str(intent_context.get("intent") or "").strip().lower()
        if raw_context_intent in DISCOVER_INTENTS:
            context_intent = raw_context_intent

    intent = context_intent or _detect_intent(text, category_hint)
    if category_hint in DISCOVER_CATEGORIES and category_hint != "all":
        intent = category_hint
    context_payload = _sanitize_intent_context(question, intent, intent_context)
    has_academic_signal = (
        _contains_any(text, GENERAL_STUDENT_TERMS) or intent in DISCOVER_CATEGORIES
    )

    if not context_payload["search_required"]:
        recommendations = recommendations_for_intent(intent)[:3]
        return ScopeDecision(
            allowed=False,
            reason_code="too_broad",
            normalized_intent=intent if intent in DISCOVER_INTENTS else "unknown",
            message="Please ask for a specific student opportunity so I can run web search.",
            recommendations=recommendations,
            search_required=False,
            intent_context=context_payload,
        )

    if _contains_any(text, DISALLOWED_TERMS) and not has_academic_signal:
        return ScopeDecision(
            allowed=False,
            reason_code="non_academic",
            normalized_intent="unknown",
            message="This search is outside student-academic opportunity scope.",
            recommendations=CATEGORY_RECOMMENDATIONS["hackathons"][:1]
            + CATEGORY_RECOMMENDATIONS["conferences"][:1]
            + CATEGORY_RECOMMENDATIONS["internships"][:1],
            search_required=False,
            intent_context=context_payload,
        )

    if intent == "jobs":
        has_student_job_signal = _contains_any(
            text,
            ("student", "fresher", "entry level", "graduate", "campus", "intern"),
        )
        if _contains_any(text, SENIOR_EXPERIENCE_TERMS) and not has_student_job_signal:
            return ScopeDecision(
                allowed=False,
                reason_code="non_academic",
                normalized_intent="jobs",
                message="Only student and entry-level opportunities are supported.",
                recommendations=CATEGORY_RECOMMENDATIONS["jobs"],
                search_required=False,
                intent_context=context_payload,
            )

    if intent == "unknown" and not has_academic_signal:
        return ScopeDecision(
            allowed=False,
            reason_code="non_academic",
            normalized_intent="unknown",
            message=(
                "This query cannot be searched here. "
                "Try academic events or student opportunities."
            ),
            recommendations=CATEGORY_RECOMMENDATIONS["hackathons"][:1]
            + CATEGORY_RECOMMENDATIONS["internships"][:1]
            + CATEGORY_RECOMMENDATIONS["scholarships"][:1],
            search_required=False,
            intent_context=context_payload,
        )

    if intent == "unknown":
        intent = "general_student_opportunity"
        context_payload["intent"] = intent

    recommendations = recommendations_for_intent(intent)

    return ScopeDecision(
        allowed=True,
        reason_code="ok",
        normalized_intent=intent,
        message="Searching trusted student-opportunity sources.",
        recommendations=recommendations,
        search_required=True,
        intent_context=context_payload,
    )
