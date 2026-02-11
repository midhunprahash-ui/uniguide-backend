"""
Circular routes for handling circular announcements and chat.
Updated to use Supabase for circular storage.
"""
import logging

from fastapi import APIRouter, HTTPException

from models.schemas import ChatResponse, CircularChatQuery
from services.chat_sessions import session_manager
from services.provider_error_mapper import map_provider_error
from services.rag_engine import rag_engine
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


def register_circular(
    doc_id: str,
    filename: str,
    year: str,
    department: str,
    upload_date: str,
    one_line_summary: str = None,
    brief_summary: str = None,
    chunk_count: int = 0,
    org_id: str = None,
    document_text: str = None  # Full text for deadline extraction
):
    """
    Register a circular in Supabase and extract deadlines.
    This is called from admin routes when a circular is uploaded.
    """
    client = get_supabase_admin_client()
    circular_id = None

    try:
        # Check if circular already exists for this document
        existing = client.table("circulars").select("id").eq("document_id", doc_id).execute()

        if not existing.data:
            # Create new circular entry with org_id for multi-tenant isolation
            circular_data = {
                "document_id": doc_id,
                "title": filename,
                "one_line_summary": one_line_summary,
                "brief_summary": brief_summary,
                "is_active": True
            }

            # Add org_id if provided
            if org_id:
                circular_data["org_id"] = org_id

            result = client.table("circulars").insert(circular_data).execute()
            circular_id = result.data[0]["id"] if result.data else None
            logger.info(f"Registered circular: {filename} for org: {org_id}")
        else:
            circular_id = existing.data[0]["id"]
        
        # Trigger deadline extraction if we have text and a circular ID
        return circular_id
                
    except Exception as e:
        logger.error(f"Error registering circular: {e}")
        return None


def get_latest_circular(org_id: str = None):
    """Get the most recently uploaded active circular from Supabase for a specific org."""
    client = get_supabase_admin_client()

    try:
        # If no org_id provided, get default org (sjit)
        if not org_id:
            org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
            org_id = org.data.get("id") if org.data else None

        if not org_id:
            return None

        # First, get document IDs for this org's circulars category (exclude soft-deleted)
        docs = client.table("documents").select("id").eq(
            "org_id", org_id
        ).eq("category", "circulars").is_("deleted_at", "null").execute()

        if not docs.data:
            return None

        doc_ids = [d["id"] for d in docs.data]

        # Query circulars for these documents
        result = client.table("circulars").select(
            "*, documents(*)"
        ).eq("is_active", True).in_(
            "document_id", doc_ids
        ).order(
            "published_at", desc=True
        ).limit(1).execute()

        if result.data:
            circular = result.data[0]
            doc = circular.get("documents", {})

            return {
                "id": circular["id"],
                "document_id": circular["document_id"],
                "filename": doc.get("filename", circular["title"]),
                "year": doc.get("year", "all"),
                "department": doc.get("department", "all"),
                "upload_date": circular["published_at"],
                "one_line_summary": circular.get("one_line_summary"),
                "brief_summary": circular.get("brief_summary"),
                "chunk_count": 0
            }

        # Fallback: If no circular entry exists, return latest document with category=circulars
        # This handles documents uploaded before circular registration was implemented
        doc_result = client.table("documents").select("*").eq(
            "org_id", org_id
        ).eq("category", "circulars").is_("deleted_at", "null").order(
            "created_at", desc=True
        ).limit(1).execute()

        if doc_result.data:
            doc = doc_result.data[0]
            return {
                "id": doc["id"],
                "document_id": doc["id"],
                "filename": doc.get("filename", "Circular"),
                "year": doc.get("year", "all"),
                "department": doc.get("department", "all"),
                "upload_date": doc.get("created_at"),
                "one_line_summary": doc.get("one_line_summary", "New circular available"),
                "brief_summary": doc.get("brief_summary", "Click to learn more about this circular."),
                "chunk_count": 0
            }

        return None
    except Exception as e:
        logger.error(f"Error getting latest circular: {e}")
        return None


@router.get("/latest")
async def get_latest():
    """
    Get the latest circular metadata and summaries.

    Returns:
        Circular details with one_line and brief summaries
    """
    latest = get_latest_circular()

    if not latest:
        return {
            "id": "",
            "filename": "",
            "year": "",
            "department": "",
            "upload_date": "",
            "one_line_summary": "No circulars uploaded yet",
            "brief_summary": "No circulars have been uploaded by the administration yet.",
            "has_circular": False
        }

    return {
        "id": latest["id"],
        "filename": latest["filename"],
        "year": latest["year"],
        "department": latest["department"],
        "upload_date": latest["upload_date"],
        "one_line_summary": latest.get("one_line_summary", "New circular available"),
        "brief_summary": latest.get("brief_summary", "Click to learn more about this circular."),
        "has_circular": True
    }


@router.post("/chat", response_model=ChatResponse)
async def chat_about_circular(query: CircularChatQuery):
    """
    Chat endpoint specifically for circular questions.
    Uses RAG with category filter set to 'circulars'.

    Args:
        query: CircularChatQuery with question and optional session_id

    Returns:
        ChatResponse with answer, sources, and session_id
    """
    try:
        # Get default org_id for circular chat (uses sjit as default)
        client = get_supabase_admin_client()
        org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
        org_id = org.data.get("id") if org.data else None

        if not org_id:
            raise HTTPException(status_code=500, detail="Could not determine organization")

        # Get or create session
        session_id = query.session_id
        if session_id:
            existing_session = session_manager.get_session(session_id)
            if not existing_session:
                session_id = session_manager.create_session(
                    category="circulars",
                    year="all",
                    department="all",
                    org_id=org_id,
                    user_id=None,
                )
            elif not session_manager.get_session_for_user(
                session_id,
                org_id=org_id,
                user_id=None,
                allow_anonymous=True,
            ):
                raise HTTPException(status_code=403, detail="Access denied for session")
        else:
            session_id = session_manager.create_session(
                category="circulars",
                year="all",
                department="all",
                org_id=org_id,
                user_id=None,
            )

        # Store user question in session
        session_manager.add_message(session_id, "user", query.question)

        # Get conversation history (limit to last 5 exchanges = 10 messages)
        history = session_manager.get_history(session_id, limit=10)

        # Query RAG with circulars category filter
        result = rag_engine.query(
            question=query.question,
            stream="all",  # Circulars apply to all streams
            year="all",  # Circulars apply to all years
            department="all",  # Circulars apply to all departments
            category="circulars",
            org_id=org_id,
            conversation_history=history
        )

        # Store assistant response in session
        session_manager.add_message(session_id, "assistant", result["answer"])

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing circular query")
        mapped = map_provider_error(
            e,
            fallback_message="Unable to process your circular query right now. Please try again shortly.",
        )
        raise HTTPException(status_code=mapped.status_code, detail=mapped.message)
