"""
Notes Chat API routes for student-facing Notes RAG queries.
Isolated from institutional chat (routes/chat.py).
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from services.auth import require_auth, require_org_membership, require_valid_org_id
from services.chat_sessions import session_manager
from services.notes_rag_engine import notes_rag_engine
from services.rate_limiter import limiter, RATE_LIMITS, get_org_key

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class NotesQuery(BaseModel):
    question: str
    org_id: str
    year_id: Optional[str] = None
    subject_id: Optional[str] = None
    unit_id: Optional[str] = None
    session_id: Optional[str] = None
    context_key: Optional[str] = None


class NotesResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    session_id: str
    subject: Optional[dict] = None


# ============================================================================
# Session Endpoints
# ============================================================================

@router.get("/session/{session_id}")
async def get_notes_session_history(
    request: Request,
    session_id: str,
    current_user: dict = Depends(require_auth),
):
    """
    Get notes chat session with its message history.
    Used to restore notes chat after page refresh.
    """
    session_data = session_manager.get_session_with_history(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_data.get("user_id") != current_user.get("id"):
        raise HTTPException(status_code=403, detail="Access denied")

    session_org_id = session_data.get("org_id")
    if not session_org_id:
        raise HTTPException(status_code=403, detail="Session organization is missing")
    require_org_membership(current_user.get("id"), session_org_id)
    
    return session_data


# ============================================================================
# Query Endpoints
# ============================================================================

@router.post("/query-stream")
@limiter.limit(RATE_LIMITS["chat"], key_func=get_org_key)
async def query_notes_stream(
    request: Request,
    query: NotesQuery,
    current_user: dict = Depends(require_auth),
):
    """
    Stream notes chat response using Server-Sent Events (SSE).
    
    SECURITY: org_id is validated to prevent cross-tenant data access.
    """
    try:
        # CRITICAL: Validate org_id
        require_valid_org_id(query.org_id)
        require_org_membership(current_user.get("id"), query.org_id)
        
        # Get or create session
        session_id = query.session_id
        if session_id:
            existing_session = session_manager.get_session(session_id)
            if not existing_session:
                context_key = query.context_key
                if not context_key and query.subject_id:
                    context_key = f"note_{query.subject_id}_{query.unit_id or 'subject'}"
                session_id = session_manager.create_session(
                    category="notes",  # Special category for notes
                    year=query.year_id,
                    department=None,
                    org_id=query.org_id,
                    user_id=current_user.get("id"),
                    context_key=context_key,
                )
            elif not session_manager.get_session_for_user(
                session_id,
                org_id=query.org_id,
                user_id=current_user.get("id"),
            ):
                raise HTTPException(status_code=403, detail="Access denied for session")
        else:
            context_key = query.context_key
            if not context_key and query.subject_id:
                context_key = f"note_{query.subject_id}_{query.unit_id or 'subject'}"
            session_id = session_manager.create_session(
                category="notes",  # Special category for notes
                year=query.year_id,
                department=None,
                org_id=query.org_id,
                user_id=current_user.get("id"),
                context_key=context_key,
            )
        
        # Store user question
        session_manager.add_message(session_id, "user", query.question)
        
        # Get conversation history
        history = session_manager.get_history(session_id, limit=10)
        
        async def event_generator():
            full_answer = ""
            sources = []
            
            # Send session ID first
            yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"
            
            try:
                for chunk_json in notes_rag_engine.query_stream(
                    question=query.question,
                    org_id=query.org_id,
                    year_id=query.year_id,
                    subject_id=query.subject_id,
                    unit_id=query.unit_id,
                    conversation_history=history
                ):
                    chunk_data = json.loads(chunk_json)
                    
                    if chunk_data["type"] == "token":
                        full_answer += chunk_data["data"]
                    elif chunk_data["type"] == "sources":
                        sources = chunk_data["data"]
                    
                    yield f"data: {chunk_json}\n\n"
                
                # Save to history
                session_manager.add_message(
                    session_id,
                    "assistant",
                    full_answer,
                    sources=sources
                )
                
            except Exception as e:
                logger.error(f"Notes stream error: {e}")
                error_json = json.dumps({"type": "error", "data": "Error generating response."})
                yield f"data: {error_json}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting stream: {str(e)}")


@router.post("/query", response_model=NotesResponse)
@limiter.limit(RATE_LIMITS["chat"], key_func=get_org_key)
async def query_notes(request: Request, query: NotesQuery, current_user: dict = Depends(require_auth)):
    """
    Query notes RAG (non-streaming).
    
    SECURITY: org_id is validated to prevent cross-tenant data access.
    """
    try:
        # CRITICAL: Validate org_id
        require_valid_org_id(query.org_id)
        require_org_membership(current_user.get("id"), query.org_id)
        
        # Get or create session
        session_id = query.session_id
        if session_id:
            existing_session = session_manager.get_session(session_id)
            if not existing_session:
                context_key = query.context_key
                if not context_key and query.subject_id:
                    context_key = f"note_{query.subject_id}_{query.unit_id or 'subject'}"
                session_id = session_manager.create_session(
                    category="notes",
                    year=query.year_id,
                    department=None,
                    org_id=query.org_id,
                    user_id=current_user.get("id"),
                    context_key=context_key,
                )
            elif not session_manager.get_session_for_user(
                session_id,
                org_id=query.org_id,
                user_id=current_user.get("id"),
            ):
                raise HTTPException(status_code=403, detail="Access denied for session")
        else:
            context_key = query.context_key
            if not context_key and query.subject_id:
                context_key = f"note_{query.subject_id}_{query.unit_id or 'subject'}"
            session_id = session_manager.create_session(
                category="notes",
                year=query.year_id,
                department=None,
                org_id=query.org_id,
                user_id=current_user.get("id"),
                context_key=context_key,
            )
        
        # Store user question
        session_manager.add_message(session_id, "user", query.question)
        
        # Get conversation history
        history = session_manager.get_history(session_id, limit=10)
        
        # Query RAG
        result = notes_rag_engine.query(
            question=query.question,
            org_id=query.org_id,
            year_id=query.year_id,
            subject_id=query.subject_id,
            unit_id=query.unit_id,
            conversation_history=history
        )
        
        # Store response
        session_manager.add_message(
            session_id,
            "assistant",
            result["answer"],
            sources=result["sources"]
        )
        
        return NotesResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            session_id=session_id,
            subject=result.get("subject")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ============================================================================
# Browse Endpoints (for sidebar display)
# ============================================================================

@router.get("/browse/subjects")
async def browse_subjects(
    request: Request,
    org_id: str,
    year_id: Optional[str] = None,
    current_user: dict = Depends(require_auth),
):
    """
    Get subjects for browsing in the notes sidebar.
    Returns subjects with note counts.
    """
    from services.supabase_client import get_supabase_admin_client
    
    require_valid_org_id(org_id)
    require_org_membership(current_user.get("id"), org_id)
    
    client = get_supabase_admin_client()
    
    query = client.table("subjects").select(
        "id, name, code, unit_count"
    ).eq("org_id", org_id).eq("is_active", True)
    
    if year_id:
        query = query.eq("year_id", year_id)
    
    result = query.order("sort_order").execute()
    
    subjects = []
    for subject in (result.data or []):
        # Get note count for this subject
        notes_count = client.table("notes").select(
            "id", count="exact"
        ).eq("subject_id", subject["id"]).eq("org_id", org_id).is_("deleted_at", "null").execute()
        
        subjects.append({
            **subject,
            "notes_count": notes_count.count or 0
        })
    
    return subjects


@router.get("/browse/subjects/{subject_id}/units")
async def browse_subject_units(
    request: Request,
    subject_id: str,
    org_id: str,
    current_user: dict = Depends(require_auth),
):
    """
    Get units for a subject with their notes.
    """
    from services.supabase_client import get_supabase_admin_client
    
    require_valid_org_id(org_id)
    require_org_membership(current_user.get("id"), org_id)
    
    client = get_supabase_admin_client()
    
    # Get units
    units = client.table("subject_units").select(
        "id, unit_number, name"
    ).eq("subject_id", subject_id).eq("org_id", org_id).order("unit_number").execute()
    
    result = []
    for unit in (units.data or []):
        # Get notes for this unit
        notes = client.table("notes").select(
            "id, title, original_filename, one_line_summary"
        ).eq("unit_id", unit["id"]).eq("org_id", org_id).is_("deleted_at", "null").order("created_at").execute()
        
        result.append({
            **unit,
            "notes": notes.data or []
        })
    
    return result
