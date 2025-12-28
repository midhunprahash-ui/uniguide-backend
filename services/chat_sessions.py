"""Chat session management for conversation history."""
import uuid
from datetime import datetime, timedelta


class ChatSession:
    """Represents a single chat session with history."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[dict[str, str]] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def add_message(self, role: str, content: str):
        """Add a message to the session history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()

    def get_history(self, limit: int | None = None) -> list[dict[str, str]]:
        """Get conversation history, optionally limited to recent messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages

class ChatSessionManager:
    """Manages multiple chat sessions in memory."""

    def __init__(self, session_timeout_hours: int = 24):
        self.sessions: dict[str, ChatSession] = {}
        self.session_timeout = timedelta(hours=session_timeout_hours)

    def create_session(self) -> str:
        """Create a new chat session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id)
        return session_id

    def get_session(self, session_id: str) -> ChatSession | None:
        """Get a session by ID, return None if not found or expired."""
        session = self.sessions.get(session_id)
        if session:
            # Check if session has expired
            if datetime.now() - session.last_activity > self.session_timeout:
                del self.sessions[session_id]
                return None
        return session

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to a session. Returns False if session doesn't exist."""
        session = self.get_session(session_id)
        if session:
            session.add_message(role, content)
            return True
        return False

    def get_history(self, session_id: str, limit: int | None = 5) -> list[dict[str, str]]:
        """Get conversation history for a session."""
        session = self.get_session(session_id)
        if session:
            return session.get_history(limit)
        return []

    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory."""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_activity > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]

# Global session manager instance
session_manager = ChatSessionManager()
