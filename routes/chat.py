import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.schemas import ChatQuery, ChatResponse
from services.auth import require_auth, require_org_membership, require_valid_org_id
from services.chat_sessions import session_manager
from services.provider_error_mapper import map_provider_error, provider_error_sse_payload
from services.rag_engine import rag_engine
from services.rate_limiter import RATE_LIMITS, get_org_key, limiter

logger = logging.getLogger(__name__)
router = APIRouter()


class RenameSessionRequest(BaseModel):
    org_id: str
    title: str = Field(..., min_length=1, max_length=240)


class ShareSessionRequest(BaseModel):
    org_id: str


class DeleteSessionRequest(BaseModel):
    org_id: str


def _delete_session_for_user(session_id: str, org_id: str, current_user: dict):
    """Shared delete logic for DELETE/POST compatibility endpoints."""
    require_valid_org_id(org_id)
    require_org_membership(current_user.get("id"), org_id)

    deleted = session_manager.soft_delete_session_for_user(
        session_id,
        org_id=org_id,
        user_id=current_user.get("id"),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "session_id": session_id}


def _resolve_general_chat_session_id(query: ChatQuery, user_id: str | None) -> str:
    """
    Resolve session id with ChatGPT-style thread reuse semantics.

    Priority:
    1) valid requested session id owned by the user
    2) latest existing session for the same user/org/context key
    3) create a new session
    """
    context_key = session_manager.normalize_context_key(query.context_key, query.category)
    requested_session_id = query.session_id

    if requested_session_id:
        existing_session = session_manager.get_session(requested_session_id)
        if existing_session:
            if not session_manager.get_session_for_user(
                requested_session_id,
                org_id=query.org_id,
                user_id=user_id,
            ):
                raise HTTPException(status_code=403, detail="Access denied for session")

            existing_context_key = session_manager.normalize_context_key(
                existing_session.get("context_key"),
                existing_session.get("category"),
            )
            if not context_key or existing_context_key == context_key:
                return requested_session_id

            logger.info(
                "Requested session context mismatch; using current context thread user_id=%s "
                "org_id=%s requested_session_id=%s requested_context=%s active_context=%s",
                user_id,
                query.org_id,
                requested_session_id,
                existing_context_key,
                context_key,
            )

        logger.info(
            "Requested chat session not found; trying context reuse user_id=%s org_id=%s "
            "requested_session_id=%s context_key=%s",
            user_id,
            query.org_id,
            requested_session_id,
            context_key,
        )

    if user_id and context_key:
        reusable_session = session_manager.get_latest_session_for_context(
            user_id=user_id,
            org_id=query.org_id,
            context_key=context_key,
            category=query.category,
        )
        if reusable_session and reusable_session.get("id"):
            return reusable_session["id"]

    return session_manager.create_session(
        category=query.category,
        year=query.year,
        department=query.department,
        org_id=query.org_id,
        user_id=user_id,
        context_key=context_key,
    )


@router.get("/session/{session_id}")
@limiter.limit(RATE_LIMITS["public"])
async def get_session_history(
    request: Request,
    session_id: str,
    current_user: dict = Depends(require_auth),
):
    """
    Get chat session with its message history.
    Used to restore chat after page refresh.
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


@router.patch("/session/{session_id}")
@limiter.limit(RATE_LIMITS["public"])
async def rename_session(
    request: Request,
    session_id: str,
    payload: RenameSessionRequest,
    current_user: dict = Depends(require_auth),
):
    """Rename a chat session owned by the current user in the given org."""
    require_valid_org_id(payload.org_id)
    require_org_membership(current_user.get("id"), payload.org_id)

    normalized_title = " ".join((payload.title or "").split()).strip()
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    try:
        renamed_session = session_manager.rename_session_for_user(
            session_id,
            org_id=payload.org_id,
            user_id=current_user.get("id"),
            title=normalized_title,
        )
        if not renamed_session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "id": renamed_session.get("id"),
            "title": renamed_session.get("title"),
            "updated_at": renamed_session.get("updated_at"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to rename chat session user_id=%s org_id=%s session_id=%s",
            current_user.get("id"),
            payload.org_id,
            session_id,
        )
        raise HTTPException(status_code=500, detail="Failed to rename chat session")


@router.delete("/session/{session_id}")
@limiter.limit(RATE_LIMITS["public"])
async def delete_session(
    request: Request,
    session_id: str,
    org_id: str = Query(..., description="Organization ID"),
    current_user: dict = Depends(require_auth),
):
    """Soft delete a chat session owned by the current user in the given org."""
    try:
        return _delete_session_for_user(session_id, org_id, current_user)
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to delete chat session user_id=%s org_id=%s session_id=%s",
            current_user.get("id"),
            org_id,
            session_id,
        )
        raise HTTPException(status_code=500, detail="Failed to delete chat session")


@router.post("/session/{session_id}/delete")
@limiter.limit(RATE_LIMITS["public"])
async def delete_session_compat(
    request: Request,
    session_id: str,
    payload: DeleteSessionRequest,
    current_user: dict = Depends(require_auth),
):
    """
    Backward-compatible delete endpoint for environments/proxies that disallow DELETE.
    """
    try:
        return _delete_session_for_user(session_id, payload.org_id, current_user)
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to delete chat session (compat) user_id=%s org_id=%s session_id=%s",
            current_user.get("id"),
            payload.org_id,
            session_id,
        )
        raise HTTPException(status_code=500, detail="Failed to delete chat session")


@router.post("/session/{session_id}")
@limiter.limit(RATE_LIMITS["public"])
async def delete_session_post_alias(
    request: Request,
    session_id: str,
    payload: DeleteSessionRequest,
    current_user: dict = Depends(require_auth),
):
    """
    Additional compatibility endpoint for clients/proxies restricted to POST.
    """
    try:
        return _delete_session_for_user(session_id, payload.org_id, current_user)
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to delete chat session (post alias) user_id=%s org_id=%s session_id=%s",
            current_user.get("id"),
            payload.org_id,
            session_id,
        )
        raise HTTPException(status_code=500, detail="Failed to delete chat session")


@router.post("/session/{session_id}/share")
@limiter.limit(RATE_LIMITS["public"])
async def share_session(
    request: Request,
    session_id: str,
    payload: ShareSessionRequest,
    current_user: dict = Depends(require_auth),
):
    """Create or reuse a share link for a chat session owned by the current user."""
    require_valid_org_id(payload.org_id)
    require_org_membership(current_user.get("id"), payload.org_id)

    try:
        share = session_manager.create_or_get_share_for_user_session(
            session_id,
            org_id=payload.org_id,
            user_id=current_user.get("id"),
        )
        if not share:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "share_id": share.get("share_id"),
            "created_at": share.get("created_at"),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to create share for chat session user_id=%s org_id=%s session_id=%s",
            current_user.get("id"),
            payload.org_id,
            session_id,
        )
        raise HTTPException(status_code=500, detail="Failed to create share link")


@router.get("/shared/{share_id}")
@limiter.limit(RATE_LIMITS["public"])
async def get_shared_session(
    request: Request,
    share_id: str,
):
    """
    Public read-only endpoint to load a shared chat conversation by share ID.

    This intentionally bypasses user auth and relies on high-entropy share IDs.
    """
    shared = session_manager.get_shared_session_by_share_id(share_id)
    if not shared:
        raise HTTPException(status_code=404, detail="Shared chat not found")
    return shared


@router.get("/sessions")
@limiter.limit(RATE_LIMITS["public"])
async def list_sessions(
    request: Request,
    org_id: str = Query(..., description="Organization ID"),
    current_user: dict = Depends(require_auth),
):
    """
    List recent chat sessions for the current user within an org.
    Returns the most recent session per context_key.
    """
    require_valid_org_id(org_id)
    require_org_membership(current_user.get("id"), org_id)

    sessions = session_manager.list_sessions_for_user(current_user.get("id"), org_id, limit=200)
    seen_keys: set[str] = set()
    results = []

    for session in sessions:
        context_key = session.get("context_key")
        if not context_key:
            category = session.get("category")
            if category and category != "notes":
                context_key = category
        if not context_key or context_key in seen_keys:
            continue
        seen_keys.add(context_key)
        results.append({
            "context_key": context_key,
            "session_id": session.get("id"),
            "category": session.get("category"),
            "updated_at": session.get("updated_at"),
        })

    return {"sessions": results}


@router.get("/history")
@limiter.limit(RATE_LIMITS["public"])
async def list_history(
    request: Request,
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(30, ge=1, le=100, description="Page size"),
    cursor: str | None = Query(None, description="Keyset cursor: previous page last updated_at"),
    category: str | None = Query(None, description="Optional category slug"),
    current_user: dict = Depends(require_auth),
):
    """
    List paginated chat history for the current user in an organization.
    Returns full sessions (not deduped by context key).
    """
    require_valid_org_id(org_id)
    require_org_membership(current_user.get("id"), org_id)

    normalized_cursor: str | None = None
    if cursor:
        try:
            cursor_value = cursor.strip().replace(" ", "+").replace("Z", "+00:00")
            normalized_cursor = datetime.fromisoformat(cursor_value).isoformat()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid cursor format") from exc

    try:
        page = session_manager.list_history_for_user(
            current_user.get("id"),
            org_id,
            limit=limit,
            cursor=normalized_cursor,
            category=category.strip().lower() if category else None,
        )
        return page
    except Exception:
        logger.exception(
            "Failed to list chat history user_id=%s org_id=%s category=%s limit=%s cursor=%s",
            current_user.get("id"),
            org_id,
            category,
            limit,
            cursor,
        )
        raise HTTPException(status_code=500, detail="Failed to load chat history")


@router.post("/query-stream")
@limiter.limit(RATE_LIMITS["chat"], key_func=get_org_key)
async def query_chat_stream(request: Request, query: ChatQuery, current_user: dict = Depends(require_auth)):
    """
    Stream chat response using Server-Sent Events (SSE).

    SECURITY: org_id is validated to prevent cross-tenant data access.
    """
    try:
        # CRITICAL: Validate org_id to prevent tenant isolation bypass
        require_valid_org_id(query.org_id)
        require_org_membership(current_user.get("id"), query.org_id)

        session_id = _resolve_general_chat_session_id(query, current_user.get("id"))

        # Store user question in session
        session_manager.add_message(session_id, "user", query.question)

        # Get conversation history (limit to last 5 exchanges = 10 messages)
        history = session_manager.get_history(session_id, limit=10)

        async def event_generator():
            full_answer = ""
            sources = []
            
            # Send session ID as the first event (custom event or metadata)
            # We'll send it as a special "session" event
            yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"

            try:
                # Run the synchronous generator in a thread is handled by FastAPI if we just iterate? 
                # Actually for async def, we should probably interact with sync code carefully.
                # But here we are inside the response generator.
                # Let's use a standard iterator.
                
                for chunk_json in rag_engine.query_stream(
                    question=query.question,
                    stream=query.stream,
                    year=query.year,
                    department=query.department,
                    category=query.category,
                    org_id=query.org_id,
                    conversation_history=history
                ):
                    # chunk_json is already a JSON string from rag_engine
                    chunk_data = json.loads(chunk_json)
                    
                    if chunk_data["type"] == "token":
                        full_answer += chunk_data["data"]
                    elif chunk_data["type"] == "sources":
                        sources = chunk_data["data"]
                        
                    yield f"data: {chunk_json}\n\n"
                
                # After stream finishes, save to history
                session_manager.add_message(
                    session_id, 
                    "assistant", 
                    full_answer,
                    sources=sources
                )
                
            except Exception as e:
                logger.exception("Chat stream error")
                error_json = json.dumps(
                    provider_error_sse_payload(
                        e,
                        fallback_message=(
                            "We could not generate a response right now. Please try again shortly."
                        ),
                    )
                )
                yield f"data: {error_json}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start chat stream")
        mapped = map_provider_error(
            e,
            fallback_message="Unable to start chat right now. Please try again shortly.",
        )
        raise HTTPException(status_code=mapped.status_code, detail=mapped.message)


@router.post("/query", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS["chat"], key_func=get_org_key)
async def query_chat(request: Request, query: ChatQuery, current_user: dict = Depends(require_auth)):
    """
    Student chat endpoint to query the RAG system with conversation history.

    SECURITY: org_id is validated to prevent cross-tenant data access.

    Args:
        query: ChatQuery containing question, year, department, and optional session_id

    Returns:
        ChatResponse with answer, sources, and session_id
    """
    try:
        # CRITICAL: Validate org_id to prevent tenant isolation bypass
        require_valid_org_id(query.org_id)
        require_org_membership(current_user.get("id"), query.org_id)

        session_id = _resolve_general_chat_session_id(query, current_user.get("id"))

        # Store user question in session
        session_manager.add_message(session_id, "user", query.question)

        # Get conversation history (limit to last 5 exchanges = 10 messages)
        history = session_manager.get_history(session_id, limit=10)

        # Query RAG with history and category filter
        result = rag_engine.query(
            question=query.question,
            stream=query.stream,
            year=query.year,
            department=query.department,
            category=query.category,
            org_id=query.org_id,
            conversation_history=history
        )

        # Store assistant response in session with sources
        session_manager.add_message(
            session_id, 
            "assistant", 
            result["answer"],
            sources=result["sources"]
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process chat query")
        mapped = map_provider_error(
            e,
            fallback_message="Unable to process your query right now. Please try again shortly.",
        )
        raise HTTPException(status_code=mapped.status_code, detail=mapped.message)
