from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from models.schemas import ChatQuery, ChatResponse
from services.chat_sessions import session_manager
from services.rag_engine import rag_engine

router = APIRouter()


@router.get("/session/{session_id}")
async def get_session_history(session_id: str):
    """
    Get chat session with its message history.
    Used to restore chat after page refresh.
    """
    session_data = session_manager.get_session_with_history(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_data


@router.post("/query-stream")
async def query_chat_stream(query: ChatQuery):
    """
    Stream chat response using Server-Sent Events (SSE).
    """
    try:
        # Get or create session
        session_id = query.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(
                category=query.category,
                year=query.year,
                department=query.department,
                org_id=query.org_id
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
                print(f"Stream error: {e}")
                error_json = json.dumps({"type": "error", "data": "Error generating response."})
                yield f"data: {error_json}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting stream: {str(e)}")


@router.post("/query", response_model=ChatResponse)
async def query_chat(query: ChatQuery):
    """
    Student chat endpoint to query the RAG system with conversation history.

    Args:
        query: ChatQuery containing question, year, department, and optional session_id

    Returns:
        ChatResponse with answer, sources, and session_id
    """
    try:
        # Get or create session
        session_id = query.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(
                category=query.category,
                year=query.year,
                department=query.department,
                org_id=query.org_id
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
