"""
Deadline & Event Routes for Smart Calendar
Provides API endpoints for managing and retrieving deadlines.

Design Principles:
- Efficiency: Batch processing with configurable concurrency
- Reliability: Duplicate detection, progress tracking
- Clean Code: Type hints, dataclasses, clear separation of concerns
"""
import logging
import asyncio
import hashlib
import re
from datetime import date, datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from services.auth import require_auth
from services.supabase_client import get_supabase_admin_client
from services.deadline_extractor import (
    extract_deadlines_from_text,
    calculate_smart_score,
    EventType,
    Priority
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DeadlineResponse(BaseModel):
    """Response model for a single deadline/event."""
    id: str
    title: str
    description: Optional[str] = None
    event_type: str
    deadline_date: str  # Start date
    end_date: Optional[str] = None  # For multi-day events (exams, holidays)
    deadline_time: Optional[str] = None
    priority: str
    days_remaining: int
    is_urgent: bool
    is_multi_day: bool = False  # True if spans multiple days
    duration_days: int = 1  # Number of days the event spans
    smart_score: int = Field(default=0, description="Ranking score 0-100")
    target_streams: list[str] = []
    target_departments: list[str] = []
    target_years: list[int] = []
    document_id: Optional[str] = None
    circular_id: Optional[str] = None
    series_key: Optional[str] = None
    series_occurrence: Optional[int] = None
    series_total: Optional[int] = None
    is_series_start: Optional[bool] = None
    is_series_end: Optional[bool] = None


class DeadlineInteraction(BaseModel):
    """Payload for user interaction with a deadline."""
    interaction_type: str  # 'dismissed', 'completed', 'snoozed'
    snooze_until: Optional[str] = None  # ISO datetime string


class DeadlineStats(BaseModel):
    """Statistics about deadlines for dashboard widgets."""
    total_active: int
    urgent_count: int
    by_type: dict[str, int]
    by_priority: dict[str, int] = {}


class ReprocessingResult(BaseModel):
    """Result of bulk reprocessing operation."""
    message: str
    documents_processed: int
    documents_skipped: int
    events_created: int
    errors: list[str] = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_org_id_from_slug(slug: str = "sjit") -> Optional[str]:
    """Get organization ID from slug."""
    client = get_supabase_admin_client()
    try:
        result = client.table("organizations").select("id").eq("slug", slug).single().execute()
        return result.data.get("id") if result.data else None
    except Exception as e:
        logger.error(f"Error getting org ID: {e}")
        return None


def get_existing_deadline_document_ids(org_id: str) -> set[str]:
    """Get set of document IDs that already have deadlines extracted."""
    client = get_supabase_admin_client()
    try:
        result = client.table("deadlines").select("document_id").eq("org_id", org_id).execute()
        return {d["document_id"] for d in (result.data or []) if d.get("document_id")}
    except Exception as e:
        logger.error(f"Error getting existing deadline docs: {e}")
        return set()


DATE_HINT_PATTERN = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r"|today|tomorrow"
    r")\b",
    re.IGNORECASE,
)
MAX_EXTRACTION_SEGMENTS = 24
SEGMENT_SIZE_CHARS = 11_000
SEGMENT_OVERLAP_CHARS = 1_200


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _parse_csv_tokens(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    tokens = [token.strip() for token in str(raw).split(",")]
    return sorted({token for token in tokens if token and token.lower() != "all"})


def _parse_year_tokens(raw: Optional[str]) -> list[int]:
    if not raw:
        return []
    values: set[int] = set()
    for token in str(raw).split(","):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            values.add(int(digits))
    return sorted(values)


def _normalize_string_list(values: Optional[list]) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = {_normalize_text(str(v)) for v in values if str(v).strip()}
    return sorted(v for v in cleaned if v and v != "all")


def _normalize_int_list(values: Optional[list]) -> list[int]:
    if not isinstance(values, list):
        return []
    cleaned: set[int] = set()
    for value in values:
        if isinstance(value, int):
            cleaned.add(value)
            continue
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits:
            cleaned.add(int(digits))
    return sorted(cleaned)


def _build_deadline_signature(deadline_data: dict) -> str:
    signature_source = "|".join(
        [
            _normalize_text(deadline_data.get("title", "")),
            _normalize_text(deadline_data.get("event_type", "other")),
            str(deadline_data.get("deadline_date") or ""),
            str(deadline_data.get("end_date") or ""),
            str(deadline_data.get("deadline_time") or ""),
            ",".join(_normalize_string_list(deadline_data.get("target_streams"))),
            ",".join(_normalize_string_list(deadline_data.get("target_departments"))),
            ",".join(str(v) for v in _normalize_int_list(deadline_data.get("target_years"))),
        ]
    )
    return hashlib.sha256(signature_source.encode("utf-8")).hexdigest()


def _event_signature(event: dict) -> str:
    return _build_deadline_signature(
        {
            "title": event.get("title"),
            "event_type": event.get("event_type"),
            "deadline_date": event.get("deadline_date"),
            "end_date": event.get("end_date"),
            "deadline_time": event.get("deadline_time"),
            "target_streams": event.get("target_streams"),
            "target_departments": event.get("target_departments"),
            "target_years": event.get("target_years"),
        }
    )


def _build_extraction_segments(document_text: str) -> list[str]:
    text = (document_text or "").strip()
    if not text:
        return []

    segments: list[tuple[int, str, int]] = []
    step = max(SEGMENT_SIZE_CHARS - SEGMENT_OVERLAP_CHARS, 1_000)
    start = 0
    index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + SEGMENT_SIZE_CHARS, text_len)
        segment = text[start:end]
        score = len(DATE_HINT_PATTERN.findall(segment))
        segments.append((index, segment, score))
        if end >= text_len:
            break
        start += step
        index += 1

    if len(segments) <= MAX_EXTRACTION_SEGMENTS:
        return [segment for _, segment, _ in segments]

    selected: set[int] = {0, len(segments) - 1}
    ranked = sorted(
        segments[1:-1],
        key=lambda entry: (entry[2], -entry[0]),
        reverse=True,
    )

    for idx, _, _ in ranked:
        if len(selected) >= MAX_EXTRACTION_SEGMENTS:
            break
        selected.add(idx)

    return [segments[idx][1] for idx in sorted(selected)]


def _get_document_semester(value: object) -> str:
    if value is None:
        return "all"
    return str(value).strip().lower() or "all"


def _matches_semester_filter(document_semester: str, semester: Optional[int]) -> bool:
    if semester is None:
        return True
    if not document_semester or document_semester == "all":
        return True
    tokens = [token.strip().lower() for token in document_semester.split(",") if token.strip()]
    if not tokens:
        return True
    return str(semester) in tokens


def _matches_stream_filter(target_streams: list[str], stream: Optional[str], fallback_stream: Optional[str]) -> bool:
    if not stream:
        return True
    selected = stream.strip().lower()
    if selected == "all":
        return True

    normalized_targets = [value.lower() for value in target_streams if value]
    if normalized_targets:
        return selected in normalized_targets

    fallback_values = [value.lower() for value in _parse_csv_tokens(fallback_stream)]
    if not fallback_values:
        return True
    return selected in fallback_values


def _matches_department_filter(
    target_departments: list[str],
    department: Optional[str],
    fallback_department: Optional[str],
) -> bool:
    if not department:
        return True
    selected = department.strip().lower()
    if selected == "all":
        return True

    normalized_targets = [value.lower() for value in target_departments if value]
    if normalized_targets:
        return selected in normalized_targets

    fallback_values = [value.lower() for value in _parse_csv_tokens(fallback_department)]
    if not fallback_values:
        return True
    return selected in fallback_values


def _matches_year_filter(target_years: list[int], year: Optional[int], fallback_year: Optional[str]) -> bool:
    if year is None:
        return True
    if target_years:
        return year in target_years
    fallback_values = _parse_year_tokens(fallback_year)
    if not fallback_values:
        return True
    return year in fallback_values


def _build_series_key(deadline_data: dict) -> str:
    source = "|".join(
        [
            _normalize_text(deadline_data.get("title", "")),
            _normalize_text(deadline_data.get("event_type", "other")),
            ",".join(_normalize_string_list(deadline_data.get("target_streams"))),
            ",".join(_normalize_string_list(deadline_data.get("target_departments"))),
            ",".join(str(v) for v in _normalize_int_list(deadline_data.get("target_years"))),
            str(deadline_data.get("document_id") or ""),
            str(deadline_data.get("circular_id") or ""),
        ]
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/upcoming", response_model=list[DeadlineResponse])
async def get_upcoming_deadlines(
    user_id: str = Query(..., description="User identifier (session ID or user ID)"),
    stream: Optional[str] = Query(None, description="Stream code filter"),
    department: Optional[str] = Query(None, description="Department code filter"),
    year: Optional[int] = Query(None, description="Year number filter"),
    semester: Optional[int] = Query(None, ge=1, le=8, description="Semester number filter"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(20, ge=1, le=100, description="Max number of events to return")
):
    """
    Get upcoming deadlines/events for a user, sorted by smart score.
    
    Features:
    - Excludes dismissed deadlines for this user
    - Filters by stream/department/year/semester if specified
    - Orders by smart_score (urgency × importance)
    - Includes ongoing multi-day events
    """
    client = get_supabase_admin_client()

    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            return []

        today = date.today()

        columns = (
            "id,title,description,event_type,deadline_date,end_date,deadline_time,"
            "priority,target_streams,target_departments,target_years,document_id,circular_id"
        )

        # Build query for active, relevant deadlines.
        # Include:
        # - future single-day events (deadline_date >= today)
        # - ongoing multi-day events (end_date >= today)
        query = (
            client.table("deadlines")
            .select(columns)
            .eq("org_id", org_id)
            .eq("status", "active")
            .or_(
                f"deadline_date.gte.{today.isoformat()},"
                f"and(end_date.not.is.null,end_date.gte.{today.isoformat()})"
            )
        )

        if event_type:
            query = query.eq("event_type", event_type)

        # Fetch extra rows to allow selector filtering while still keeping payload bounded.
        fetch_limit = min(max(limit * 12, 120), 600)
        result = query.order("deadline_date").limit(fetch_limit).execute()
        rows = result.data or []
        if not rows:
            return []

        # Get user's dismissed deadlines.
        dismissed_result = client.table("user_deadline_interactions").select(
            "deadline_id"
        ).eq(
            "user_identifier", user_id
        ).eq(
            "interaction_type", "dismissed"
        ).execute()

        dismissed_ids = {d["deadline_id"] for d in (dismissed_result.data or [])}

        # Fetch document metadata once for fallback selector matching + semester filtering.
        document_ids = sorted({row.get("document_id") for row in rows if row.get("document_id")})
        doc_meta_by_id: dict[str, dict] = {}
        if document_ids:
            documents_result = (
                client.table("documents")
                .select("id,stream,department,year,semester")
                .in_("id", document_ids)
                .execute()
            )
            for doc in documents_result.data or []:
                doc_meta_by_id[str(doc["id"])] = doc

        deadlines: list[dict] = []
        for dl in rows:
            if dl["id"] in dismissed_ids:
                continue

            doc_meta = doc_meta_by_id.get(str(dl.get("document_id")), {})
            target_streams = _normalize_string_list(dl.get("target_streams"))
            target_departments = _normalize_string_list(dl.get("target_departments"))
            target_years = _normalize_int_list(dl.get("target_years"))

            if not _matches_stream_filter(
                target_streams,
                stream,
                doc_meta.get("stream"),
            ):
                continue
            if not _matches_department_filter(
                target_departments,
                department,
                doc_meta.get("department"),
            ):
                continue
            if not _matches_year_filter(target_years, year, doc_meta.get("year")):
                continue
            if not _matches_semester_filter(
                _get_document_semester(doc_meta.get("semester")),
                semester,
            ):
                continue

            deadline_date = datetime.strptime(dl["deadline_date"], "%Y-%m-%d").date()
            end_date_str = dl.get("end_date")
            end_date = None
            if end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                except ValueError:
                    end_date = None

            # Ongoing multi-day events should not look overdue.
            if end_date and deadline_date < today <= end_date:
                score_date = today
                days_remaining = 0
            else:
                score_date = deadline_date
                days_remaining = (deadline_date - today).days

            smart_score = calculate_smart_score(
                deadline_date=score_date,
                event_type=dl.get("event_type", "other"),
                priority=dl.get("priority", "normal"),
                reference_date=today,
            )

            is_urgent = days_remaining <= 3 or dl.get("priority") in ["critical", "high"]

            is_multi_day = end_date is not None
            duration_days = 1
            if is_multi_day and end_date:
                duration_days = max((end_date - deadline_date).days + 1, 1)

            deadline_payload: dict = {
                "id": dl["id"],
                "title": dl["title"],
                "description": dl.get("description"),
                "event_type": dl["event_type"],
                "deadline_date": dl["deadline_date"],
                "end_date": end_date_str,
                "deadline_time": dl.get("deadline_time"),
                "priority": dl.get("priority", "normal"),
                "days_remaining": days_remaining,
                "is_urgent": is_urgent,
                "is_multi_day": is_multi_day,
                "duration_days": duration_days,
                "smart_score": smart_score,
                "target_streams": target_streams,
                "target_departments": target_departments,
                "target_years": target_years,
                "document_id": dl.get("document_id"),
                "circular_id": dl.get("circular_id"),
            }
            deadline_payload["series_key"] = _build_series_key(deadline_payload)
            deadlines.append(deadline_payload)

        # Annotate series positions for repeated events.
        series_groups: dict[str, list[dict]] = {}
        for item in deadlines:
            series_key = item.get("series_key")
            if not series_key:
                continue
            series_groups.setdefault(series_key, []).append(item)

        for series_key, items in series_groups.items():
            items.sort(
                key=lambda item: (
                    item.get("deadline_date", ""),
                    item.get("end_date") or "",
                    item.get("title") or "",
                )
            )
            total = len(items)
            for idx, item in enumerate(items):
                item["series_key"] = series_key
                item["series_total"] = total
                item["series_occurrence"] = idx + 1
                item["is_series_start"] = idx == 0
                item["is_series_end"] = idx == total - 1

        deadlines.sort(
            key=lambda item: (
                -item["smart_score"],
                item["deadline_date"],
                item.get("title", ""),
            )
        )

        return [DeadlineResponse(**item) for item in deadlines[:limit]]

    except Exception as e:
        logger.error(f"Error getting upcoming deadlines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{deadline_id}/interact")
async def interact_with_deadline(
    deadline_id: str,
    interaction: DeadlineInteraction,
    current_user: dict = Depends(require_auth),
):
    """
    Record a user interaction with a deadline (dismiss, complete, snooze).
    """
    client = get_supabase_admin_client()
    
    try:
        valid_types = ["dismissed", "completed", "snoozed", "reminded"]
        if interaction.interaction_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid interaction type. Must be one of: {valid_types}"
            )

        user_identifier = current_user.get("id")
        if not user_identifier:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Verify deadline exists
        deadline = client.table("deadlines").select("id").eq("id", deadline_id).single().execute()
        if not deadline.data:
            raise HTTPException(status_code=404, detail="Deadline not found")
        
        # Upsert interaction (delete old, insert new)
        interaction_data = {
            "deadline_id": deadline_id,
            "user_identifier": user_identifier,
            "interaction_type": interaction.interaction_type,
        }
        
        if interaction.snooze_until and interaction.interaction_type == "snoozed":
            interaction_data["snooze_until"] = interaction.snooze_until
        
        # Delete existing interaction of same type
        client.table("user_deadline_interactions").delete().eq(
            "deadline_id", deadline_id
        ).eq(
            "user_identifier", user_identifier
        ).eq(
            "interaction_type", interaction.interaction_type
        ).execute()
        
        # Insert new interaction
        client.table("user_deadline_interactions").insert(interaction_data).execute()
        
        return {"status": "success", "message": f"Deadline marked as {interaction.interaction_type}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording deadline interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DeadlineStats)
async def get_deadline_stats(
    user_id: str = Query(..., description="User identifier")
):
    """
    Get deadline statistics for the dashboard widget.
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            return DeadlineStats(total_active=0, urgent_count=0, by_type={}, by_priority={})
        
        today = date.today()
        
        # Get all active deadlines
        result = client.table("deadlines").select("*").eq(
            "org_id", org_id
        ).eq(
            "status", "active"
        ).gte(
            "deadline_date", today.isoformat()
        ).execute()
        
        if not result.data:
            return DeadlineStats(total_active=0, urgent_count=0, by_type={}, by_priority={})
        
        # Get dismissed
        dismissed_result = client.table("user_deadline_interactions").select(
            "deadline_id"
        ).eq(
            "user_identifier", user_id
        ).eq(
            "interaction_type", "dismissed"
        ).execute()
        
        dismissed_ids = {d["deadline_id"] for d in (dismissed_result.data or [])}
        
        # Calculate stats
        total_active = 0
        urgent_count = 0
        by_type: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        
        for dl in result.data:
            if dl["id"] in dismissed_ids:
                continue
            
            total_active += 1
            
            # Check urgency
            deadline_date = datetime.strptime(dl["deadline_date"], "%Y-%m-%d").date()
            days_remaining = (deadline_date - today).days
            if days_remaining <= 3 or dl.get("priority") in ["critical", "high"]:
                urgent_count += 1
            
            # Count by type
            event_type = dl.get("event_type", "other")
            by_type[event_type] = by_type.get(event_type, 0) + 1
            
            # Count by priority
            priority = dl.get("priority", "normal")
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        return DeadlineStats(
            total_active=total_active,
            urgent_count=urgent_count,
            by_type=by_type,
            by_priority=by_priority
        )
        
    except Exception as e:
        logger.error(f"Error getting deadline stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/event-types")
async def get_available_event_types():
    """Return all available event types for filtering."""
    return {
        "event_types": [e.value for e in EventType],
        "priorities": [p.value for p in Priority]
    }


# ============================================================================
# DEADLINE REGISTRATION (Called during document upload)
# ============================================================================

def register_deadlines_from_document(
    document_id: str,
    document_text: str,
    org_id: str,
    circular_id: Optional[str] = None
) -> int:
    """
    Extract and register deadlines from any document.
    
    Called automatically during document upload.
    
    Args:
        document_id: The document's UUID
        document_text: Full text content of the document
        org_id: Organization ID
        circular_id: Optional circular ID if this is a circular
        
    Returns:
        Number of deadlines registered
    """
    client = get_supabase_admin_client()

    def _clean_date_value(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        value = str(raw).strip()
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            return None

    def _clean_time_value(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        value = str(raw).strip()
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime("%H:%M")
            except ValueError:
                continue
        return None

    try:
        segments = _build_extraction_segments(document_text)
        if not segments:
            logger.info(f"No extractable content for document {document_id}")
            return 0

        # Extract events from selected segments and de-duplicate at extraction level.
        extracted_events: list[dict] = []
        seen_extracted_signatures: set[str] = set()
        for segment in segments:
            segment_events = extract_deadlines_from_text(segment)
            for event in segment_events:
                signature = _event_signature(event)
                if signature in seen_extracted_signatures:
                    continue
                seen_extracted_signatures.add(signature)
                extracted_events.append(event)

        if not extracted_events:
            logger.info(f"No events extracted from document {document_id}")
            return 0

        # Existing signatures for this document, so re-runs don't create duplicates.
        existing_result = (
            client.table("deadlines")
            .select(
                "title,event_type,deadline_date,end_date,deadline_time,"
                "target_streams,target_departments,target_years"
            )
            .eq("org_id", org_id)
            .eq("document_id", document_id)
            .execute()
        )
        existing_signatures = {
            _build_deadline_signature(row)
            for row in (existing_result.data or [])
        }

        doc_meta_result = (
            client.table("documents")
            .select("stream,department,year")
            .eq("id", document_id)
            .maybe_single()
            .execute()
        )
        doc_meta = doc_meta_result.data or {}
        fallback_streams = _parse_csv_tokens(doc_meta.get("stream"))
        fallback_departments = _parse_csv_tokens(doc_meta.get("department"))
        fallback_years = _parse_year_tokens(doc_meta.get("year"))

        payloads: list[dict] = []
        for event in extracted_events:
            deadline_date = _clean_date_value(event.get("deadline_date"))
            if not deadline_date:
                continue

            end_date = _clean_date_value(event.get("end_date"))
            if end_date and end_date < deadline_date:
                end_date = None

            deadline_time = _clean_time_value(event.get("deadline_time"))
            target_streams = _normalize_string_list(event.get("target_streams")) or fallback_streams
            target_departments = _normalize_string_list(event.get("target_departments")) or fallback_departments
            target_years = _normalize_int_list(event.get("target_years")) or fallback_years

            deadline_data = {
                "org_id": org_id,
                "circular_id": circular_id,
                "document_id": document_id,
                "title": (event.get("title") or "Untitled Event")[:100],
                "description": str(event.get("description") or "")[:500],
                "event_type": event.get("event_type", "other"),
                "deadline_date": deadline_date,
                "end_date": end_date,
                "deadline_time": deadline_time,
                "is_all_day": deadline_time is None,
                "target_streams": target_streams,
                "target_departments": target_departments,
                "target_years": target_years,
                "priority": event.get("priority", "normal"),
                "status": "active",
                "confidence_score": event.get("confidence", 0.5),
                "extracted_text": str(event.get("extracted_text") or "")[:500],
            }

            signature = _build_deadline_signature(deadline_data)
            if signature in existing_signatures:
                continue
            existing_signatures.add(signature)
            payloads.append(deadline_data)

        if not payloads:
            logger.info(f"No new deadlines to register for document {document_id}")
            return 0

        inserted_count = 0
        try:
            client.table("deadlines").insert(payloads).execute()
            inserted_count = len(payloads)
        except Exception as batch_error:
            logger.warning(f"Batch insert failed for document {document_id}: {batch_error}")
            for payload in payloads:
                try:
                    client.table("deadlines").insert(payload).execute()
                    inserted_count += 1
                except Exception as row_error:
                    logger.error(f"Failed to insert event '{payload.get('title', 'unknown')}': {row_error}")

        logger.info(f"Registered {inserted_count} events from document {document_id}")
        return inserted_count

    except Exception as e:
        logger.error(f"Error registering events from document: {e}")
        return 0


# ============================================================================
# BULK REPROCESSING (Optimized for many documents)
# ============================================================================

@router.post("/reprocess-all", response_model=ReprocessingResult)
async def reprocess_all_documents(
    skip_processed: bool = Query(True, description="Skip documents that already have deadlines"),
    category: Optional[str] = Query(None, description="Only process specific category"),
    batch_size: int = Query(5, ge=1, le=20, description="Documents to process in parallel"),
    background_tasks: BackgroundTasks = None
):
    """
    Re-extract events from all existing documents.
    
    Optimized for bulk processing:
    - Skips already-processed documents
    - Processes in batches to avoid timeout
    - Reports progress and errors
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Get documents that already have deadlines
        processed_doc_ids = get_existing_deadline_document_ids(org_id) if skip_processed else set()
        logger.info(f"Found {len(processed_doc_ids)} documents already processed")
        
        # Build query
        query = client.table("documents").select("id, filename, category").eq("org_id", org_id)
        if category:
            query = query.eq("category", category)
        
        docs_result = query.execute()
        
        if not docs_result.data:
            return ReprocessingResult(
                message="No documents found",
                documents_processed=0,
                documents_skipped=0,
                events_created=0
            )
        
        total_processed = 0
        total_skipped = 0
        total_events = 0
        errors: list[str] = []
        
        # Process in batches
        for doc in docs_result.data:
            doc_id = doc["id"]
            
            # Skip already processed
            if doc_id in processed_doc_ids:
                total_skipped += 1
                continue
            
            try:
                # Get document text from chunks
                chunks_result = client.table("document_chunks").select(
                    "content"
                ).eq("document_id", doc_id).order("chunk_number").execute()
                
                if not chunks_result.data:
                    continue
                
                # Combine chunks
                full_text = " ".join([c["content"] for c in chunks_result.data])
                
                if len(full_text.strip()) < 50:
                    continue
                
                # Get circular_id if applicable
                circular_id = None
                if doc.get("category") == "circulars":
                    circ_result = client.table("circulars").select("id").eq(
                        "document_id", doc_id
                    ).maybe_single().execute()
                    if circ_result.data:
                        circular_id = circ_result.data["id"]
                
                # Extract and register
                count = register_deadlines_from_document(
                    document_id=doc_id,
                    document_text=full_text,
                    org_id=org_id,
                    circular_id=circular_id
                )
                
                total_processed += 1
                total_events += count
                logger.info(f"Processed {doc['filename']}: {count} events")
                
            except Exception as e:
                error_msg = f"{doc['filename']}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to process document: {error_msg}")
        
        return ReprocessingResult(
            message="Reprocessing complete",
            documents_processed=total_processed,
            documents_skipped=total_skipped,
            events_created=total_events,
            errors=errors[:10]  # Limit error list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk reprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-deadline")
async def create_test_deadline():
    """Create a test deadline for calendar verification."""
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Create test deadline 7 days from now
        future_date = (date.today() + timedelta(days=7)).isoformat()
        
        deadline_data = {
            "org_id": org_id,
            "title": "Test Event - Calendar Verification",
            "description": "This is a test event to verify calendar functionality. You can dismiss this.",
            "event_type": "event",
            "deadline_date": future_date,
            "deadline_time": "10:00",
            "is_all_day": False,
            "target_streams": [],
            "target_departments": [],
            "target_years": [],
            "priority": "normal",
            "status": "active",
            "confidence_score": 1.0,
            "extracted_text": "Test event created for calendar verification"
        }
        
        result = client.table("deadlines").insert(deadline_data).execute()
        
        if result.data:
            return {
                "message": "Test deadline created successfully",
                "deadline": {
                    "id": result.data[0]["id"],
                    "title": deadline_data["title"],
                    "date": future_date,
                    "event_type": "event"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create test deadline")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating test deadline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-all")
async def clear_all_deadlines():
    """
    Clear all deadlines for reprocessing (admin use only).
    Use with caution - deletes all extracted deadlines.
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Delete all deadlines for this org
        result = client.table("deadlines").delete().eq("org_id", org_id).execute()
        
        # Also clear interactions
        client.table("user_deadline_interactions").delete().neq("id", "").execute()
        
        return {
            "message": "All deadlines cleared",
            "deleted_count": len(result.data) if result.data else 0
        }
        
    except Exception as e:
        logger.error(f"Error clearing deadlines: {e}")
        raise HTTPException(status_code=500, detail=str(e))
