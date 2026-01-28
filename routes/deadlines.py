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
from datetime import date, datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

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


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/upcoming", response_model=list[DeadlineResponse])
async def get_upcoming_deadlines(
    user_id: str = Query(..., description="User identifier (session ID or user ID)"),
    stream: Optional[str] = Query(None, description="Stream code filter"),
    department: Optional[str] = Query(None, description="Department code filter"),
    year: Optional[int] = Query(None, description="Year number filter"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(20, ge=1, le=100, description="Max number of events to return")
):
    """
    Get upcoming deadlines/events for a user, sorted by smart score.
    
    Features:
    - Excludes dismissed deadlines for this user
    - Filters by stream/department/year if specified
    - Orders by smart_score (urgency × importance)
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            return []
        
        today = date.today()
        
        # Build query for active, future deadlines
        query = client.table("deadlines").select("*").eq(
            "org_id", org_id
        ).eq(
            "status", "active"
        ).gte(
            "deadline_date", today.isoformat()
        )
        
        # Add event type filter if specified
        if event_type:
            query = query.eq("event_type", event_type)
        
        # Execute - get extra to allow for filtering
        result = query.order("deadline_date").limit(limit * 2).execute()
        
        if not result.data:
            return []
        
        # Get user's dismissed deadlines (single query)
        dismissed_result = client.table("user_deadline_interactions").select(
            "deadline_id"
        ).eq(
            "user_identifier", user_id
        ).eq(
            "interaction_type", "dismissed"
        ).execute()
        
        dismissed_ids = {d["deadline_id"] for d in (dismissed_result.data or [])}
        
        # Filter, enhance, and collect
        deadlines = []
        for dl in result.data:
            # Skip dismissed
            if dl["id"] in dismissed_ids:
                continue
            
            # Apply targeting filters
            if stream and dl.get("target_streams") and stream not in dl["target_streams"]:
                continue
            if department and dl.get("target_departments") and department not in dl["target_departments"]:
                continue
            if year and dl.get("target_years") and year not in dl["target_years"]:
                continue
            
            # Calculate days remaining and smart score
            deadline_date = datetime.strptime(dl["deadline_date"], "%Y-%m-%d").date()
            days_remaining = (deadline_date - today).days
            
            smart_score = calculate_smart_score(
                deadline_date=deadline_date,
                event_type=dl.get("event_type", "other"),
                priority=dl.get("priority", "normal")
            )
            
            is_urgent = days_remaining <= 3 or dl.get("priority") in ["critical", "high"]
            
            # Calculate multi-day event properties
            end_date_str = dl.get("end_date")
            is_multi_day = end_date_str is not None
            duration_days = 1
            if is_multi_day and end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                    duration_days = (end_date - deadline_date).days + 1
                except ValueError:
                    pass
            
            deadlines.append(DeadlineResponse(
                id=dl["id"],
                title=dl["title"],
                description=dl.get("description"),
                event_type=dl["event_type"],
                deadline_date=dl["deadline_date"],
                end_date=end_date_str,
                deadline_time=dl.get("deadline_time"),
                priority=dl.get("priority", "normal"),
                days_remaining=days_remaining,
                is_urgent=is_urgent,
                is_multi_day=is_multi_day,
                duration_days=duration_days,
                smart_score=smart_score,
                target_streams=dl.get("target_streams", []),
                target_departments=dl.get("target_departments", []),
                target_years=dl.get("target_years", []),
                document_id=dl.get("document_id"),
                circular_id=dl.get("circular_id")
            ))
            
            if len(deadlines) >= limit:
                break
        
        # Sort by smart score (highest first)
        deadlines.sort(key=lambda d: d.smart_score, reverse=True)
        
        return deadlines
        
    except Exception as e:
        logger.error(f"Error getting upcoming deadlines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{deadline_id}/interact")
async def interact_with_deadline(
    deadline_id: str,
    interaction: DeadlineInteraction,
    user_id: str = Query(..., description="User identifier")
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
        
        # Verify deadline exists
        deadline = client.table("deadlines").select("id").eq("id", deadline_id).single().execute()
        if not deadline.data:
            raise HTTPException(status_code=404, detail="Deadline not found")
        
        # Upsert interaction (delete old, insert new)
        interaction_data = {
            "deadline_id": deadline_id,
            "user_identifier": user_id,
            "interaction_type": interaction.interaction_type,
        }
        
        if interaction.snooze_until and interaction.interaction_type == "snoozed":
            interaction_data["snooze_until"] = interaction.snooze_until
        
        # Delete existing interaction of same type
        client.table("user_deadline_interactions").delete().eq(
            "deadline_id", deadline_id
        ).eq(
            "user_identifier", user_id
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
    
    try:
        # Extract events using AI
        extracted = extract_deadlines_from_text(document_text)
        
        if not extracted:
            logger.info(f"No events extracted from document {document_id}")
            return 0
        
        # Insert each event
        inserted_count = 0
        for event in extracted:
            try:
                deadline_data = {
                    "org_id": org_id,
                    "circular_id": circular_id,
                    "document_id": document_id,
                    "title": event["title"][:100],
                    "description": event.get("description", "")[:500],
                    "event_type": event.get("event_type", "other"),
                    "deadline_date": event["deadline_date"],
                    "end_date": event.get("end_date"),  # For multi-day events
                    "deadline_time": event.get("deadline_time"),
                    "is_all_day": event.get("deadline_time") is None,
                    "target_streams": event.get("target_streams", []),
                    "target_departments": event.get("target_departments", []),
                    "target_years": event.get("target_years", []),
                    "priority": event.get("priority", "normal"),
                    "status": "active",
                    "confidence_score": event.get("confidence", 0.5),
                    "extracted_text": event.get("extracted_text", "")[:500]
                }
                
                client.table("deadlines").insert(deadline_data).execute()
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to insert event '{event.get('title', 'unknown')}': {e}")
                continue
        
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
