import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from models.schemas import ChatQuery, ChatResponse
from services.auth import require_auth, require_org_membership, require_valid_org_id
from services.chat_sessions import session_manager
from services.rag_engine import rag_engine
from services.rate_limiter import limiter, RATE_LIMITS, get_org_key

logger = logging.getLogger(__name__)
router = APIRouter()


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
    
    return session_data


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

        # Get or create session
        session_id = query.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(
                category=query.category,
                year=query.year,
                department=query.department,
                org_id=query.org_id,
                user_id=current_user.get("id"),
            )

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
                import traceback
                logger.error(f"Stream error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                error_json = json.dumps({"type": "error", "data": f"Error: {str(e)}"})
                yield f"data: {error_json}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting stream: {str(e)}")


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

        # Get or create session
        session_id = query.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(
                category=query.category,
                year=query.year,
                department=query.department,
                org_id=query.org_id,
                user_id=current_user.get("id"),
            )

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
