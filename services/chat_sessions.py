"""
Chat session management for conversation history.
Uses Supabase for persistent storage across page refreshes.
"""
import uuid
import logging
from datetime import datetime
from typing import Optional

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


class SupabaseChatSessionManager:
    """Manages chat sessions persisted in Supabase."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = get_supabase_admin_client()
        return self._client

    def create_session(
        self,
        category: str = "rules",
        year: str = "all",
        department: str = "all",
        org_id: str = None,
        user_id: str = None,
        context_key: str | None = None
    ) -> str:
        """Create a new chat session and return its ID.

        Args:
            category: Chat category filter
            year: Year filter
            department: Department filter
            org_id: Organization ID for multi-tenant isolation (REQUIRED for proper isolation)
            user_id: User ID (None for anonymous sessions)
        """
        session_id = str(uuid.uuid4())

        # CRITICAL: org_id is REQUIRED for tenant isolation - NO FALLBACKS!
        if not org_id:
            logger.warning("No org_id provided for chat session - session will not be org-scoped")

        try:
            payload = {
                "id": session_id,
                "org_id": org_id,
                "user_id": user_id,
                "category": category,
                "year": year,
                "department": department,
                "updated_at": datetime.now().isoformat()
            }
            if context_key:
                payload["context_key"] = context_key

            self.client.table("chat_sessions").insert(payload).execute()

            logger.info(f"Created new chat session: {session_id} for org: {org_id}")
            return session_id

        except Exception as e:
            if context_key:
                try:
                    fallback_payload = {
                        "id": session_id,
                        "org_id": org_id,
                        "user_id": user_id,
                        "category": category,
                        "year": year,
                        "department": department,
                        "updated_at": datetime.now().isoformat()
                    }
                    self.client.table("chat_sessions").insert(fallback_payload).execute()
                    logger.warning(
                        "Created session without context_key (migration missing?): %s",
                        session_id
                    )
                    return session_id
                except Exception as fallback_error:
                    logger.error(f"Error creating session fallback: {fallback_error}")

            logger.error(f"Error creating session: {e}")
            # Fallback to UUID for now
            return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID, return None if not found."""
        try:
            result = self.client.table("chat_sessions").select("*").eq("id", session_id).single().execute()
            return result.data
        except Exception as e:
            logger.debug(f"Session not found: {session_id}")
            return None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list = None
    ) -> bool:
        """Add a message to a session. Returns False if failed."""
        try:
            # Insert message
            self.client.table("chat_messages").insert({
                "session_id": session_id,
                "role": role,
                "content": content,
                "sources": sources or []
            }).execute()
            
            # Update session's last activity
            self.client.table("chat_sessions").update({
                "updated_at": datetime.now().isoformat()
            }).eq("id", session_id).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False

    def get_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> list[dict[str, str]]:
        """Get conversation history for a session."""
        try:
            result = self.client.table("chat_messages").select(
                "role, content, sources, created_at"
            ).eq(
                "session_id", session_id
            ).order(
                "created_at", desc=False
            ).limit(limit).execute()
            
            return [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "sources": msg.get("sources", [])
                }
                for msg in result.data
            ]
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def get_session_with_history(self, session_id: str) -> Optional[dict]:
        """Get session details along with its message history."""
        session = self.get_session(session_id)
        if not session:
            return None
            
        history = self.get_history(session_id)
        return {
            **session,
            "messages": history
        }

    def list_sessions_for_user(
        self,
        user_id: str,
        org_id: str,
        limit: int = 100
    ) -> list[dict]:
        """List recent sessions for a user within an org."""
        try:
            result = (
                self.client.table("chat_sessions")
                .select("id, category, context_key, updated_at")
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .is_("deleted_at", None)
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.warning(f"Error listing sessions with context_key: {e}")
            try:
                result = (
                    self.client.table("chat_sessions")
                    .select("id, category, updated_at")
                    .eq("user_id", user_id)
                    .eq("org_id", org_id)
                    .is_("deleted_at", None)
                    .order("updated_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                return result.data or []
            except Exception as fallback_error:
                logger.error(f"Error listing sessions fallback: {fallback_error}")
                return []


# Global session manager instance
session_manager = SupabaseChatSessionManager()
