"""
Circular routes for handling circular announcements and chat.
Updated to use Supabase for circular storage.
"""
import logging

from fastapi import APIRouter, HTTPException

from models.schemas import ChatResponse, CircularChatQuery
from services.chat_sessions import session_manager
from services.rag_engine import rag_engine
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


def register_circular(doc_id: str, filename: str, year: str, department: str,
                      upload_date: str, one_line_summary: str = None,
                      brief_summary: str = None, chunk_count: int = 0):
    """
    Register a circular in Supabase.
    This is called from admin routes when a circular is uploaded.
    """
    client = get_supabase_admin_client()

    try:
        # Check if circular already exists for this document
        existing = client.table("circulars").select("id").eq("document_id", doc_id).execute()

        if not existing.data:
            # Create new circular entry
            client.table("circulars").insert({
                "document_id": doc_id,
                "title": filename,
                "one_line_summary": one_line_summary,
                "brief_summary": brief_summary,
                "is_active": True
            }).execute()
            logger.info(f"Registered circular: {filename}")
    except Exception as e:
        logger.error(f"Error registering circular: {e}")


def get_latest_circular():
    """Get the most recently uploaded active circular from Supabase."""
    client = get_supabase_admin_client()

    try:
        result = client.table("circulars").select(
            "*, documents(*)"
        ).eq("is_active", True).order(
            "published_at", desc=True
        ).limit(1).execute()

        if not result.data:
            return None

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
            "chunk_count": 0  # Could count from document_chunks if needed
        }
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
        # Get or create session
        session_id = query.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session()

        # Store user question in session
        session_manager.add_message(session_id, "user", query.question)

        # Get conversation history (limit to last 5 exchanges = 10 messages)
        history = session_manager.get_history(session_id, limit=10)

        # Query RAG with circulars category filter
        result = rag_engine.query(
            question=query.question,
            year="all",  # Circulars apply to all years
            department="all",  # Circulars apply to all departments
            category="circulars",
            conversation_history=history
        )

        # Store assistant response in session
        session_manager.add_message(session_id, "assistant", result["answer"])

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error processing circular query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing circular query: {str(e)}")
