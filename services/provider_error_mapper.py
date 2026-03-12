"""
Provider error mapping utilities for user-safe API and SSE responses.

Industry-aligned goals:
- Do not leak raw upstream provider/internal error details to end users.
- Map known provider failures to stable, actionable user-facing messages.
- Keep detailed exception logging on the server side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from google.api_core import exceptions as google_exceptions
except Exception:  # pragma: no cover - optional dependency at runtime
    google_exceptions = None


@dataclass(frozen=True)
class UserFacingProviderError:
    message: str
    code: str
    status_code: int
    retryable: bool


GENERIC_PROVIDER_MESSAGE = (
    "We could not process your request right now. Please try again shortly."
)
QUOTA_EXCEEDED_MESSAGE = (
    "The AI service has reached its current usage limit. Please try again in a few minutes."
)
RATE_LIMITED_MESSAGE = (
    "The AI service is receiving too many requests right now. Please wait a moment and retry."
)
TEMP_UNAVAILABLE_MESSAGE = (
    "The AI service is temporarily unavailable. Please try again shortly."
)
INVALID_REQUEST_MESSAGE = (
    "We could not process this request. Please rephrase it and try again."
)
CONTENT_BLOCKED_MESSAGE = (
    "Your request could not be completed due to safety checks. Please rephrase and try again."
)
SERVICE_CONFIG_MESSAGE = (
    "This AI feature is temporarily unavailable. Please try again later."
)


def _is_google_exception(error: Exception, name: str) -> bool:
    if not google_exceptions:
        return False
    exception_cls = getattr(google_exceptions, name, None)
    return bool(exception_cls and isinstance(error, exception_cls))


def _error_text(error: Exception) -> str:
    raw = str(error or "").strip()
    if raw:
        return raw.lower()
    message = getattr(error, "message", "")
    if isinstance(message, str) and message.strip():
        return message.strip().lower()
    return ""


def map_provider_error(
    error: Exception,
    *,
    fallback_message: str | None = None,
) -> UserFacingProviderError:
    text = _error_text(error)
    fallback = fallback_message or GENERIC_PROVIDER_MESSAGE

    has_quota_signal = any(
        token in text
        for token in (
            "quota",
            "resource_exhausted",
            "insufficient_quota",
            "usage limit",
            "billing",
        )
    )
    has_rate_signal = any(
        token in text
        for token in (
            "429",
            "rate limit",
            "too many requests",
            "request limit",
        )
    )
    has_service_signal = any(
        token in text
        for token in (
            "service unavailable",
            "temporarily unavailable",
            "deadline exceeded",
            "timed out",
            "timeout",
            "internal server error",
            "connection reset",
            "unavailable",
            "503",
            "504",
        )
    )
    has_auth_signal = any(
        token in text
        for token in (
            "permission denied",
            "unauthorized",
            "invalid api key",
            "api key not valid",
            "api_key_invalid",
            "forbidden",
            "403",
            "401",
        )
    )
    has_invalid_signal = any(
        token in text
        for token in (
            "invalid argument",
            "invalid request",
            "bad request",
            "malformed",
            "400",
        )
    ) and not has_auth_signal
    has_safety_signal = any(
        token in text
        for token in (
            "safety",
            "blocked",
            "content policy",
            "prohibited",
            "disallowed",
        )
    )

    if _is_google_exception(error, "ResourceExhausted") or has_quota_signal:
        return UserFacingProviderError(
            message=QUOTA_EXCEEDED_MESSAGE,
            code="provider_quota_exceeded",
            status_code=503,
            retryable=True,
        )

    if _is_google_exception(error, "TooManyRequests") or has_rate_signal:
        return UserFacingProviderError(
            message=RATE_LIMITED_MESSAGE,
            code="provider_rate_limited",
            status_code=429,
            retryable=True,
        )

    if any(
        _is_google_exception(error, name)
        for name in ("ServiceUnavailable", "DeadlineExceeded", "InternalServerError")
    ) or has_service_signal:
        return UserFacingProviderError(
            message=TEMP_UNAVAILABLE_MESSAGE,
            code="provider_temporarily_unavailable",
            status_code=503,
            retryable=True,
        )

    if any(_is_google_exception(error, name) for name in ("PermissionDenied", "Unauthorized")):
        return UserFacingProviderError(
            message=SERVICE_CONFIG_MESSAGE,
            code="provider_auth_error",
            status_code=503,
            retryable=False,
        )

    if has_auth_signal:
        return UserFacingProviderError(
            message=SERVICE_CONFIG_MESSAGE,
            code="provider_auth_error",
            status_code=503,
            retryable=False,
        )

    if _is_google_exception(error, "InvalidArgument") or has_invalid_signal:
        return UserFacingProviderError(
            message=INVALID_REQUEST_MESSAGE,
            code="provider_invalid_request",
            status_code=400,
            retryable=False,
        )

    if has_safety_signal:
        return UserFacingProviderError(
            message=CONTENT_BLOCKED_MESSAGE,
            code="provider_content_blocked",
            status_code=400,
            retryable=False,
        )

    return UserFacingProviderError(
        message=fallback,
        code="provider_unknown_error",
        status_code=500,
        retryable=False,
    )


def provider_error_sse_payload(
    error: Exception,
    *,
    fallback_message: str | None = None,
) -> dict[str, Any]:
    mapped = map_provider_error(error, fallback_message=fallback_message)
    return {
        "type": "error",
        "data": mapped.message,
        "code": mapped.code,
        "retryable": mapped.retryable,
    }
