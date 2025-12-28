from fastapi import APIRouter, HTTPException
from typing import Optional

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
                department=query.department
            )

        # Store user question in session
        session_manager.add_message(session_id, "user", query.question)

        # Get conversation history (limit to last 5 exchanges = 10 messages)
        history = session_manager.get_history(session_id, limit=10)

        # Query RAG with history and category filter
        result = rag_engine.query(
            question=query.question,
            year=query.year,
            department=query.department,
            category=query.category,
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
