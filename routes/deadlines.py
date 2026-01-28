"""
Deadline routes for Smart Calendar & Deadline Assassin feature.
Provides API endpoints for managing and retrieving deadlines.
"""
import logging
from datetime import date, datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.supabase_client import get_supabase_admin_client
from services.deadline_extractor import extract_deadlines_from_text

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class DeadlineResponse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    event_type: str
    deadline_date: str
    deadline_time: Optional[str] = None
    priority: str
    days_remaining: int
    is_urgent: bool
    target_streams: list[str] = []
    target_departments: list[str] = []
    target_years: list[int] = []


class DeadlineInteraction(BaseModel):
    interaction_type: str  # 'dismissed', 'completed', 'snoozed'
    snooze_until: Optional[str] = None  # ISO datetime string


class DeadlineStats(BaseModel):
    total_active: int
    urgent_count: int
    by_type: dict[str, int]


# Helper functions
def get_org_id_from_slug(slug: str = "sjit") -> Optional[str]:
    """Get organization ID from slug."""
    client = get_supabase_admin_client()
    try:
        result = client.table("organizations").select("id").eq("slug", slug).single().execute()
        return result.data.get("id") if result.data else None
    except Exception as e:
        logger.error(f"Error getting org ID: {e}")
        return None


@router.get("/upcoming", response_model=list[DeadlineResponse])
async def get_upcoming_deadlines(
    user_id: str = Query(..., description="User identifier (session ID or user ID)"),
    stream: Optional[str] = Query(None, description="Stream code filter"),
    department: Optional[str] = Query(None, description="Department code filter"),
    year: Optional[int] = Query(None, description="Year number filter"),
    limit: int = Query(10, ge=1, le=50, description="Max number of deadlines to return")
):
    """
    Get upcoming deadlines for a user, filtered by their context.
    Excludes dismissed deadlines and orders by urgency.
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            return []
        
        today = date.today()
        
        # Build query
        query = client.table("deadlines").select("*").eq(
            "org_id", org_id
        ).eq(
            "status", "active"
        ).gte(
            "deadline_date", today.isoformat()
        )
        
        # Execute query
        result = query.order("deadline_date").limit(limit * 2).execute()  # Get extra to filter
        
        if not result.data:
            return []
        
        # Get user's dismissed deadlines
        dismissed_result = client.table("user_deadline_interactions").select(
            "deadline_id"
        ).eq(
            "user_identifier", user_id
        ).eq(
            "interaction_type", "dismissed"
        ).execute()
        
        dismissed_ids = {d["deadline_id"] for d in (dismissed_result.data or [])}
        
        # Filter and transform
        deadlines = []
        for dl in result.data:
            # Skip dismissed
            if dl["id"] in dismissed_ids:
                continue
            
            # Filter by targeting
            if stream and dl.get("target_streams") and stream not in dl["target_streams"]:
                continue
            if department and dl.get("target_departments") and department not in dl["target_departments"]:
                continue
            if year and dl.get("target_years") and year not in dl["target_years"]:
                continue
            
            # Calculate days remaining
            deadline_date = datetime.strptime(dl["deadline_date"], "%Y-%m-%d").date()
            days_remaining = (deadline_date - today).days
            
            # Determine urgency
            is_urgent = days_remaining <= 3 or dl.get("priority") in ["critical", "high"]
            
            deadlines.append(DeadlineResponse(
                id=dl["id"],
                title=dl["title"],
                description=dl.get("description"),
                event_type=dl["event_type"],
                deadline_date=dl["deadline_date"],
                deadline_time=dl.get("deadline_time"),
                priority=dl.get("priority", "normal"),
                days_remaining=days_remaining,
                is_urgent=is_urgent,
                target_streams=dl.get("target_streams", []),
                target_departments=dl.get("target_departments", []),
                target_years=dl.get("target_years", [])
            ))
            
            if len(deadlines) >= limit:
                break
        
        # Sort by urgency: critical first, then by days remaining
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        deadlines.sort(key=lambda d: (priority_order.get(d.priority, 2), d.days_remaining))
        
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
        # Validate interaction type
        valid_types = ["dismissed", "completed", "snoozed", "reminded"]
        if interaction.interaction_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid interaction type. Must be one of: {valid_types}")
        
        # Check if deadline exists
        deadline = client.table("deadlines").select("id").eq("id", deadline_id).single().execute()
        if not deadline.data:
            raise HTTPException(status_code=404, detail="Deadline not found")
        
        # Upsert interaction
        interaction_data = {
            "deadline_id": deadline_id,
            "user_identifier": user_id,
            "interaction_type": interaction.interaction_type,
        }
        
        if interaction.snooze_until and interaction.interaction_type == "snoozed":
            interaction_data["snooze_until"] = interaction.snooze_until
        
        # Delete existing interaction of same type, then insert new one
        client.table("user_deadline_interactions").delete().eq(
            "deadline_id", deadline_id
        ).eq(
            "user_identifier", user_id
        ).eq(
            "interaction_type", interaction.interaction_type
        ).execute()
        
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
    Get deadline statistics for the dashboard.
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            return DeadlineStats(total_active=0, urgent_count=0, by_type={})
        
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
            return DeadlineStats(total_active=0, urgent_count=0, by_type={})
        
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
        
        return DeadlineStats(
            total_active=total_active,
            urgent_count=urgent_count,
            by_type=by_type
        )
        
    except Exception as e:
        logger.error(f"Error getting deadline stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Internal function for registering deadlines from circulars
def register_deadlines_from_document(
    document_id: str,
    document_text: str,
    org_id: str,
    circular_id: Optional[str] = None
) -> int:
    """
    Extract and register deadlines from any document (circular or otherwise).
    
    Args:
        document_id: The document's UUID
        document_text: Full text content of the document
        org_id: Organization ID
        circular_id: Optional UUID if this document is a circular
        
    Returns:
        Number of deadlines registered
    """
    client = get_supabase_admin_client()
    
    try:
        # Extract deadlines using AI
        extracted = extract_deadlines_from_text(document_text)
        
        if not extracted:
            logger.info(f"No deadlines extracted from document {document_id}")
            return 0
        
        # Insert each deadline
        inserted_count = 0
        for dl in extracted:
            try:
                deadline_data = {
                    "org_id": org_id,
                    "circular_id": circular_id,
                    "document_id": document_id,
                    "title": dl["title"][:100],  # Limit title length
                    "description": dl.get("description", "")[:500],
                    "event_type": dl.get("event_type", "other"),
                    "deadline_date": dl["deadline_date"],
                    "deadline_time": dl.get("deadline_time"),
                    "is_all_day": dl.get("deadline_time") is None,
                    "target_streams": dl.get("target_streams", []),
                    "target_departments": dl.get("target_departments", []),
                    "target_years": dl.get("target_years", []),
                    "priority": dl.get("priority", "normal"),
                    "status": "active",
                    "confidence_score": dl.get("confidence", 0.5),
                    "extracted_text": dl.get("extracted_text", "")[:500]
                }
                
                client.table("deadlines").insert(deadline_data).execute()
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to insert deadline: {e}")
                continue
        
        logger.info(f"Registered {inserted_count} deadlines from document {document_id}")
        return inserted_count
        
    except Exception as e:
        logger.error(f"Error registering deadlines from circular: {e}")
        return 0


@router.post("/reprocess-all")
async def reprocess_all_documents_for_deadlines():
    """
    Re-extract deadlines from all existing documents.
    Useful for processing documents uploaded before deadline extraction was added.
    """
    from services.supabase_client import get_supabase_admin_client
    
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Get all documents with text chunks
        docs_result = client.table("documents").select(
            "id, filename, category"
        ).eq("org_id", org_id).execute()
        
        if not docs_result.data:
            return {"message": "No documents found", "processed": 0, "deadlines_created": 0}
        
        total_processed = 0
        total_deadlines = 0
        
        for doc in docs_result.data:
            doc_id = doc["id"]
            
            # Get the document text from chunks
            chunks_result = client.table("document_chunks").select(
                "content"
            ).eq("document_id", doc_id).order("chunk_number").execute()
            
            if not chunks_result.data:
                continue
            
            # Combine chunks to get full text
            full_text = " ".join([c["content"] for c in chunks_result.data])
            
            if len(full_text.strip()) < 50:
                continue
            
            # Get circular_id if this is a circular
            circular_id = None
            if doc.get("category") == "circulars":
                circ_result = client.table("circulars").select("id").eq(
                    "document_id", doc_id
                ).maybe_single().execute()
                if circ_result.data:
                    circular_id = circ_result.data["id"]
            
            # Extract and register deadlines
            count = register_deadlines_from_document(
                document_id=doc_id,
                document_text=full_text,
                org_id=org_id,
                circular_id=circular_id
            )
            
            total_processed += 1
            total_deadlines += count
            logger.info(f"Processed {doc['filename']}: {count} deadlines")
        
        return {
            "message": "Reprocessing complete",
            "processed": total_processed,
            "deadlines_created": total_deadlines
        }
        
    except Exception as e:
        logger.error(f"Error reprocessing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug-document/{filename}")
async def debug_document_extraction(filename: str):
    """
    Debug endpoint to check document content and deadline extraction.
    """
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Find document by filename (partial match)
        docs_result = client.table("documents").select(
            "id, filename, category, created_at"
        ).eq("org_id", org_id).ilike("filename", f"%{filename}%").execute()
        
        if not docs_result.data:
            return {"error": f"No document found matching '{filename}'"}
        
        doc = docs_result.data[0]
        doc_id = doc["id"]
        
        # Get chunks
        chunks_result = client.table("document_chunks").select(
            "content, chunk_number"
        ).eq("document_id", doc_id).order("chunk_number").execute()
        
        if not chunks_result.data:
            return {
                "document": doc,
                "error": "No text chunks found - document may not have been OCR processed",
                "chunks_count": 0
            }
        
        # Combine chunks
        full_text = " ".join([c["content"] for c in chunks_result.data])
        
        # Try extracting deadlines
        from services.deadline_extractor import extract_deadlines_from_text
        extracted = extract_deadlines_from_text(full_text)
        
        return {
            "document": doc,
            "chunks_count": len(chunks_result.data),
            "text_length": len(full_text),
            "text_preview": full_text[:1000] + "..." if len(full_text) > 1000 else full_text,
            "extracted_deadlines": extracted,
            "deadlines_count": len(extracted) if extracted else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-deadline")
async def create_test_deadline():
    """
    Create a test deadline for calendar verification.
    Creates a deadline 7 days from now.
    """
    from datetime import timedelta
    
    client = get_supabase_admin_client()
    
    try:
        org_id = get_org_id_from_slug()
        if not org_id:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Create test deadline 7 days from now
        future_date = (date.today() + timedelta(days=7)).isoformat()
        
        deadline_data = {
            "org_id": org_id,
            "title": "Test Exam - Calendar Verification",
            "description": "This is a test deadline to verify calendar functionality. You can dismiss this.",
            "event_type": "exam",
            "deadline_date": future_date,
            "deadline_time": "10:00",
            "is_all_day": False,
            "target_streams": [],
            "target_departments": [],
            "target_years": [],
            "priority": "high",
            "status": "active",
            "confidence_score": 1.0,
            "extracted_text": "Test deadline created for calendar verification"
        }
        
        result = client.table("deadlines").insert(deadline_data).execute()
        
        if result.data:
            return {
                "message": "Test deadline created successfully",
                "deadline": {
                    "id": result.data[0]["id"],
                    "title": deadline_data["title"],
                    "date": future_date,
                    "event_type": "exam"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create test deadline")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating test deadline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
