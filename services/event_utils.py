"""
Helpers for normalizing and keying discover events.
"""
import hashlib
from urllib.parse import urlparse


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def compute_event_key(
    name: str,
    start_date: str | None,
    end_date: str | None,
    date_text: str | None,
    url: str | None,
) -> str:
    anchor = start_date or date_text or ""
    payload = "|".join([
        _norm(name),
        _norm(anchor),
        _norm(end_date),
        _norm(url),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def get_source_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        return parsed.netloc or None
    except Exception:
        return None


def enrich_event_payload(event: dict) -> dict:
    name = event.get("name") or ""
    start_date = event.get("start_date")
    end_date = event.get("end_date")
    date_text = event.get("date")
    url = event.get("url") or event.get("source_url")

    event_key = event.get("event_key") or compute_event_key(
        name=name,
        start_date=start_date,
        end_date=end_date,
        date_text=date_text,
        url=url,
    )
    source_url = event.get("source_url") or event.get("url")

    enriched = {
        **event,
        "event_key": event_key,
        "source_domain": event.get("source_domain") or get_source_domain(source_url),
    }
    return enriched
