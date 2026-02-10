"""
Policy guardrails for Event Discover queries.

This module enforces student-opportunity scope and provides
recommendations when a query is out of scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DISCOVER_CATEGORIES = {
    "all",
    "hackathons",
    "internships",
    "conferences",
    "scholarships",
    "jobs",
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

    def to_payload(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "normalized_intent": self.normalized_intent,
            "message": self.message,
            "recommendations": self.recommendations,
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


def classify_discovery_scope(question: str, category_hint: str = "all") -> ScopeDecision:
    text = _normalize(question)
    if not text:
        return ScopeDecision(
            allowed=False,
            reason_code="too_broad",
            normalized_intent="unknown",
            message="Please describe a student opportunity you want to discover.",
            recommendations=CATEGORY_RECOMMENDATIONS["hackathons"][:2]
            + CATEGORY_RECOMMENDATIONS["internships"][:1],
        )

    intent = _detect_intent(text, category_hint)
    has_academic_signal = _contains_any(text, GENERAL_STUDENT_TERMS) or intent in DISCOVER_CATEGORIES

    if _contains_any(text, DISALLOWED_TERMS) and not has_academic_signal:
        return ScopeDecision(
            allowed=False,
            reason_code="non_academic",
            normalized_intent="unknown",
            message="This search is outside student-academic opportunity scope.",
            recommendations=CATEGORY_RECOMMENDATIONS["hackathons"][:1]
            + CATEGORY_RECOMMENDATIONS["conferences"][:1]
            + CATEGORY_RECOMMENDATIONS["internships"][:1],
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
            )

    if intent == "unknown" and not has_academic_signal:
        return ScopeDecision(
            allowed=False,
            reason_code="non_academic",
            normalized_intent="unknown",
            message="This query cannot be searched here. Try academic events or student opportunities.",
            recommendations=CATEGORY_RECOMMENDATIONS["hackathons"][:1]
            + CATEGORY_RECOMMENDATIONS["internships"][:1]
            + CATEGORY_RECOMMENDATIONS["scholarships"][:1],
        )

    if intent == "unknown":
        intent = "general_student_opportunity"

    recommendations = CATEGORY_RECOMMENDATIONS.get(intent, CATEGORY_RECOMMENDATIONS["hackathons"])

    return ScopeDecision(
        allowed=True,
        reason_code="ok",
        normalized_intent=intent,
        message="Searching trusted student-opportunity sources.",
        recommendations=recommendations,
    )
