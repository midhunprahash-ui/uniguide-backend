"""
Chat session management for conversation history.
Uses Supabase for persistent storage across page refreshes.
"""
import uuid
import logging
import secrets
from datetime import datetime, timezone
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

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def normalize_context_key(context_key: str | None, category: str | None = None) -> str | None:
        """
        Normalize context keys so all clients hit the same thread identifier.

        Falls back to category when an explicit key is not provided.
        """
        raw_context_key = (context_key or "").strip()
        if raw_context_key:
            return raw_context_key.lower()

        raw_category = (category or "").strip()
        if raw_category:
            return raw_category.lower()

        return None

    def get_latest_session_for_context(
        self,
        *,
        user_id: str,
        org_id: str,
        context_key: str,
        category: str | None = None,
    ) -> Optional[dict]:
        """
        Get the most recently updated active session for a user context.

        This is the primary primitive for ChatGPT-style thread reuse:
        one persistent thread per user+org+context.
        """
        normalized_context_key = self.normalize_context_key(context_key, category)
        if not normalized_context_key:
            return None

        normalized_category = (category or "").strip().lower() or None

        try:
            query = (
                self.client.table("chat_sessions")
                .select("id, user_id, org_id, category, year, department, context_key, updated_at")
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .eq("context_key", normalized_context_key)
                .is_("deleted_at", None)
            )
            if normalized_category:
                query = query.eq("category", normalized_category)

            result = query.order("updated_at", desc=True).limit(1).execute()
            rows = result.data or []
            if rows:
                return rows[0]
            return None
        except Exception as e:
            logger.warning(
                "get_latest_session_for_context primary failed user_id=%s org_id=%s context_key=%s "
                "category=%s error=%s",
                user_id,
                org_id,
                normalized_context_key,
                normalized_category,
                e,
            )

        try:
            fallback_query = (
                self.client.table("chat_sessions")
                .select("id, user_id, org_id, category, year, department, context_key, updated_at")
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .eq("context_key", normalized_context_key)
            )
            if normalized_category:
                fallback_query = fallback_query.eq("category", normalized_category)

            fallback_result = fallback_query.order("updated_at", desc=True).limit(1).execute()
            rows = fallback_result.data or []
            if rows:
                return rows[0]
        except Exception as fallback_error:
            logger.warning(
                "get_latest_session_for_context fallback failed user_id=%s org_id=%s "
                "context_key=%s category=%s error=%s",
                user_id,
                org_id,
                normalized_context_key,
                normalized_category,
                fallback_error,
            )

        # Legacy schema fallback when context_key is missing:
        # only safe when context_key equals category slug.
        if normalized_category and normalized_context_key == normalized_category:
            try:
                legacy_result = (
                    self.client.table("chat_sessions")
                    .select("id, user_id, org_id, category, year, department, updated_at")
                    .eq("user_id", user_id)
                    .eq("org_id", org_id)
                    .eq("category", normalized_category)
                    .order("updated_at", desc=True)
                    .limit(1)
                    .execute()
                )
                rows = legacy_result.data or []
                if rows:
                    session = rows[0]
                    session["context_key"] = normalized_context_key
                    return session
            except Exception as legacy_error:
                logger.error(
                    "get_latest_session_for_context legacy fallback failed user_id=%s org_id=%s "
                    "context_key=%s category=%s error=%s",
                    user_id,
                    org_id,
                    normalized_context_key,
                    normalized_category,
                    legacy_error,
                )

        return None

    def create_session(
        self,
        category: str = "rules",
        year: str = "all",
        department: str = "all",
        org_id: str | None = None,
        user_id: str | None = None,
        context_key: str | None = None,
    ) -> str:
        """Create a new chat session and return its ID.

        Args:
            category: Chat category filter
            year: Year filter
            department: Department filter
            org_id: Organization ID for multi-tenant isolation (REQUIRED for proper isolation)
            user_id: User ID (None for anonymous sessions)
        """
        if not org_id:
            raise ValueError("org_id is required to create a chat session")

        normalized_category = (category or "rules").strip().lower() or "rules"
        normalized_context_key = self.normalize_context_key(context_key, normalized_category)
        session_id = str(uuid.uuid4())

        try:
            payload = {
                "id": session_id,
                "org_id": org_id,
                "user_id": user_id,
                "category": normalized_category,
                "year": year,
                "department": department,
                "updated_at": datetime.now().isoformat()
            }
            if normalized_context_key:
                payload["context_key"] = normalized_context_key

            self.client.table("chat_sessions").insert(payload).execute()

            logger.info(f"Created new chat session: {session_id} for org: {org_id}")
            return session_id

        except Exception as e:
            if user_id and normalized_context_key:
                existing_session = self.get_latest_session_for_context(
                    user_id=user_id,
                    org_id=org_id,
                    context_key=normalized_context_key,
                    category=normalized_category,
                )
                if existing_session and existing_session.get("id"):
                    reused_session_id = existing_session["id"]
                    logger.info(
                        "Reused existing chat session after create conflict user_id=%s org_id=%s "
                        "context_key=%s session_id=%s",
                        user_id,
                        org_id,
                        normalized_context_key,
                        reused_session_id,
                    )
                    return reused_session_id

            if normalized_context_key:
                try:
                    fallback_payload = {
                        "id": session_id,
                        "org_id": org_id,
                        "user_id": user_id,
                        "category": normalized_category,
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
            raise

    def get_session(self, session_id: str, *, include_deleted: bool = False) -> Optional[dict]:
        """Get a session by ID, return None if not found."""
        try:
            query = self.client.table("chat_sessions").select("*").eq("id", session_id)
            if not include_deleted:
                query = query.is_("deleted_at", None)
            result = query.single().execute()
            return result.data
        except Exception as e:
            if include_deleted:
                logger.debug("Session not found: %s", session_id)
                return None

            # Backward compatibility for environments where deleted_at isn't present.
            try:
                fallback_result = (
                    self.client.table("chat_sessions")
                    .select("*")
                    .eq("id", session_id)
                    .single()
                    .execute()
                )
                return fallback_result.data
            except Exception:
                logger.debug("Session not found: %s error=%s", session_id, e)
                return None

    def get_session_for_user(
        self,
        session_id: str,
        *,
        org_id: str,
        user_id: str | None,
        allow_anonymous: bool = False,
    ) -> Optional[dict]:
        """
        Return session only if org/user ownership checks pass.

        Args:
            session_id: Session UUID.
            org_id: Expected tenant org_id.
            user_id: Expected user id.
            allow_anonymous: Allow user_id NULL sessions when True.
        """
        session = self.get_session(session_id)
        if not session:
            return None
        if session.get("org_id") != org_id:
            return None
        session_user_id = session.get("user_id")
        if user_id and session_user_id == user_id:
            return session
        if allow_anonymous and session_user_id is None:
            return session
        return None

    def rename_session_for_user(
        self,
        session_id: str,
        *,
        org_id: str,
        user_id: str,
        title: str,
    ) -> Optional[dict]:
        """Rename a user-owned active session."""
        session = self.get_session_for_user(session_id, org_id=org_id, user_id=user_id)
        if not session:
            return None

        normalized_title = " ".join((title or "").split()).strip()
        if not normalized_title:
            raise ValueError("title cannot be empty")

        normalized_title = normalized_title[:120]
        now_iso = self._utc_now_iso()

        (
            self.client.table("chat_sessions")
            .update({"title": normalized_title, "updated_at": now_iso})
            .eq("id", session_id)
            .execute()
        )

        session["title"] = normalized_title
        session["updated_at"] = now_iso
        return session

    def soft_delete_session_for_user(
        self,
        session_id: str,
        *,
        org_id: str,
        user_id: str,
    ) -> bool:
        """Soft delete a user-owned active session and revoke active shares."""
        session = self.get_session_for_user(session_id, org_id=org_id, user_id=user_id)
        if not session:
            return False

        now_iso = self._utc_now_iso()
        used_hard_delete = False
        try:
            (
                self.client.table("chat_sessions")
                .update({"deleted_at": now_iso, "updated_at": now_iso})
                .eq("id", session_id)
                .is_("deleted_at", None)
                .execute()
            )
        except Exception as soft_delete_error:
            # Backward compatibility for environments missing deleted_at.
            logger.warning(
                "Soft delete failed; falling back to hard delete session_id=%s org_id=%s user_id=%s "
                "error=%s",
                session_id,
                org_id,
                user_id,
                soft_delete_error,
            )
            (
                self.client.table("chat_sessions")
                .delete()
                .eq("id", session_id)
                .eq("org_id", org_id)
                .eq("user_id", user_id)
                .execute()
            )
            used_hard_delete = True

        # Revoke active shares for deleted sessions.
        try:
            (
                self.client.table("chat_session_shares")
                .update({"revoked_at": now_iso})
                .eq("session_id", session_id)
                .is_("revoked_at", None)
                .execute()
            )
        except Exception as share_revoke_error:
            logger.warning(
                "Failed revoking chat shares for deleted session session_id=%s error=%s",
                session_id,
                share_revoke_error,
            )

        logger.info(
            "Deleted chat session session_id=%s org_id=%s user_id=%s mode=%s",
            session_id,
            org_id,
            user_id,
            "hard" if used_hard_delete else "soft",
        )
        return True

    def _generate_share_id(self) -> str:
        # 24+ chars URL-safe identifier with very low collision probability.
        return secrets.token_urlsafe(18)

    def create_or_get_share_for_user_session(
        self,
        session_id: str,
        *,
        org_id: str,
        user_id: str,
    ) -> Optional[dict]:
        """
        Create or reuse an active share link for a user-owned session.
        """
        session = self.get_session_for_user(session_id, org_id=org_id, user_id=user_id)
        if not session:
            return None

        try:
            existing_result = (
                self.client.table("chat_session_shares")
                .select("share_id, created_at")
                .eq("session_id", session_id)
                .eq("org_id", org_id)
                .eq("user_id", user_id)
                .is_("revoked_at", None)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            existing_rows = existing_result.data or []
            if existing_rows:
                return existing_rows[0]
        except Exception as e:
            logger.warning(
                "Failed to query existing share link session_id=%s org_id=%s user_id=%s error=%s",
                session_id,
                org_id,
                user_id,
                e,
            )

        last_error: Exception | None = None
        for _ in range(5):
            share_id = self._generate_share_id()
            try:
                insert_result = (
                    self.client.table("chat_session_shares")
                    .insert(
                        {
                            "session_id": session_id,
                            "org_id": org_id,
                            "user_id": user_id,
                            "share_id": share_id,
                        }
                    )
                    .execute()
                )
                rows = insert_result.data or []
                if rows:
                    return {"share_id": rows[0].get("share_id"), "created_at": rows[0].get("created_at")}
                return {"share_id": share_id, "created_at": self._utc_now_iso()}
            except Exception as e:
                last_error = e
                try:
                    race_result = (
                        self.client.table("chat_session_shares")
                        .select("share_id, created_at")
                        .eq("session_id", session_id)
                        .eq("org_id", org_id)
                        .eq("user_id", user_id)
                        .is_("revoked_at", None)
                        .order("created_at", desc=True)
                        .limit(1)
                        .execute()
                    )
                    race_rows = race_result.data or []
                    if race_rows:
                        return race_rows[0]
                except Exception:
                    pass
                continue

        if last_error:
            raise last_error
        return None

    def get_shared_session_by_share_id(self, share_id: str) -> Optional[dict]:
        """Resolve a public share id to session metadata and message history."""
        normalized_share_id = (share_id or "").strip()
        if not normalized_share_id:
            return None

        try:
            share_result = (
                self.client.table("chat_session_shares")
                .select("session_id, share_id, revoked_at, expires_at, created_at")
                .eq("share_id", normalized_share_id)
                .is_("revoked_at", None)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            share_rows = share_result.data or []
            if not share_rows:
                return None

            share = share_rows[0]
            expires_at = share.get("expires_at")
            if expires_at:
                try:
                    if datetime.fromisoformat(str(expires_at).replace("Z", "+00:00")) < datetime.now(
                        timezone.utc
                    ):
                        return None
                except Exception:
                    logger.warning("Invalid expires_at on chat share share_id=%s", normalized_share_id)

            session_id = share.get("session_id")
            if not session_id:
                return None

            session = self.get_session(session_id)
            if not session:
                return None

            history = self.get_history(session_id, limit=1000)

            return {
                "share_id": normalized_share_id,
                "session": {
                    "id": session.get("id"),
                    "title": session.get("title"),
                    "category": session.get("category"),
                    "context_key": session.get("context_key"),
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                    "last_message_at": session.get("last_message_at"),
                },
                "messages": history,
            }
        except Exception as e:
            logger.error("Failed to resolve chat share share_id=%s error=%s", normalized_share_id, e)
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
            session = self.get_session(session_id)
            if not session:
                logger.warning("Cannot add message: unknown session %s", session_id)
                return False

            now_iso = datetime.now(timezone.utc).isoformat()

            # Insert message
            self.client.table("chat_messages").insert({
                "session_id": session_id,
                "role": role,
                "content": content,
                "sources": sources or [],
                "org_id": session.get("org_id"),
            }).execute()

            # Update session's last activity and history metadata.
            update_payload = {"updated_at": now_iso}
            normalized_preview = " ".join((content or "").split())[:220]

            if normalized_preview:
                update_payload.update({
                    "last_message_preview": normalized_preview,
                    "last_message_role": role,
                    "last_message_at": now_iso,
                })

                if role == "user" and not session.get("title"):
                    update_payload["title"] = normalized_preview[:120]

            try:
                (
                    self.client.table("chat_sessions")
                    .update(update_payload)
                    .eq("id", session_id)
                    .execute()
                )
            except Exception:
                # Backward compatibility for environments where metadata columns are not yet migrated.
                self.client.table("chat_sessions").update({
                    "updated_at": now_iso
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
        safe_limit = max(1, min(limit, 500))
        try:
            result = (
                self.client.table("chat_sessions")
                .select(
                    "id, category, context_key, title, last_message_preview, "
                    "last_message_at, updated_at"
                )
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .is_("deleted_at", None)
                .order("updated_at", desc=True)
                .limit(safe_limit)
                .execute()
            )
            rows = result.data or []
            logger.debug(
                "list_sessions_for_user rows=%s stage=primary user_id=%s org_id=%s",
                len(rows),
                user_id,
                org_id,
            )
            return rows
        except Exception as e:
            logger.warning(
                "list_sessions_for_user primary query failed user_id=%s org_id=%s error=%s",
                user_id,
                org_id,
                e,
            )
            try:
                result = (
                    self.client.table("chat_sessions")
                    .select("id, category, updated_at")
                    .eq("user_id", user_id)
                    .eq("org_id", org_id)
                    .is_("deleted_at", None)
                    .order("updated_at", desc=True)
                    .limit(safe_limit)
                    .execute()
                )
                rows = result.data or []
                for row in rows:
                    row.setdefault("context_key", None)
                logger.info(
                    "list_sessions_for_user rows=%s stage=minimal user_id=%s org_id=%s",
                    len(rows),
                    user_id,
                    org_id,
                )
                return rows
            except Exception as fallback_error:
                logger.warning(
                    "list_sessions_for_user minimal fallback failed user_id=%s org_id=%s "
                    "error=%s",
                    user_id,
                    org_id,
                    fallback_error,
                )
                try:
                    legacy_result = (
                        self.client.table("chat_sessions")
                        .select("id, category, updated_at")
                        .eq("user_id", user_id)
                        .eq("org_id", org_id)
                        .order("updated_at", desc=True)
                        .limit(safe_limit)
                        .execute()
                    )
                    rows = legacy_result.data or []
                    for row in rows:
                        row.setdefault("context_key", None)
                    logger.info(
                        "list_sessions_for_user rows=%s stage=legacy user_id=%s org_id=%s",
                        len(rows),
                        user_id,
                        org_id,
                    )
                    return rows
                except Exception as legacy_error:
                    logger.error(
                        "list_sessions_for_user legacy fallback failed user_id=%s org_id=%s "
                        "error=%s",
                        user_id,
                        org_id,
                        legacy_error,
                    )
                    return []

    def list_history_for_user(
        self,
        user_id: str,
        org_id: str,
        *,
        limit: int = 30,
        cursor: str | None = None,
        category: str | None = None,
    ) -> dict:
        """
        Return paginated chat history for a user in an org.

        Uses keyset pagination on updated_at for stable ordering at scale.
        """
        safe_limit = max(1, min(limit, 100))
        fetch_limit = safe_limit + 1
        fallback_stage = "primary"

        try:
            query = (
                self.client.table("chat_sessions")
                .select(
                    "id, category, context_key, title, last_message_preview, "
                    "last_message_role, last_message_at, created_at, updated_at"
                )
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .is_("deleted_at", None)
            )
            if category:
                query = query.eq("category", category)

            if cursor:
                query = query.lt("updated_at", cursor)

            result = (
                query.order("updated_at", desc=True)
                .order("id", desc=True)
                .limit(fetch_limit)
                .execute()
            )
            rows = result.data or []
        except Exception as e:
            fallback_stage = "minimal"
            logger.warning(
                "list_history_for_user primary query failed user_id=%s org_id=%s category=%s "
                "error=%s",
                user_id,
                org_id,
                category,
                e,
            )
            try:
                fallback_query = (
                    self.client.table("chat_sessions")
                    .select("id, category, created_at, updated_at")
                    .eq("user_id", user_id)
                    .eq("org_id", org_id)
                    .is_("deleted_at", None)
                )
                if category:
                    fallback_query = fallback_query.eq("category", category)
                if cursor:
                    fallback_query = fallback_query.lt("updated_at", cursor)
                fallback_result = (
                    fallback_query.order("updated_at", desc=True)
                    .order("id", desc=True)
                    .limit(fetch_limit)
                    .execute()
                )
                rows = fallback_result.data or []
            except Exception as fallback_error:
                fallback_stage = "legacy"
                logger.warning(
                    "list_history_for_user minimal fallback failed user_id=%s org_id=%s "
                    "category=%s error=%s",
                    user_id,
                    org_id,
                    category,
                    fallback_error,
                )
                try:
                    legacy_query = (
                        self.client.table("chat_sessions")
                        .select("id, category, created_at, updated_at")
                        .eq("user_id", user_id)
                        .eq("org_id", org_id)
                    )
                    if category:
                        legacy_query = legacy_query.eq("category", category)
                    if cursor:
                        legacy_query = legacy_query.lt("updated_at", cursor)
                    legacy_result = (
                        legacy_query.order("updated_at", desc=True)
                        .order("id", desc=True)
                        .limit(fetch_limit)
                        .execute()
                    )
                    rows = legacy_result.data or []
                except Exception as legacy_error:
                    logger.error(
                        "list_history_for_user legacy fallback failed user_id=%s org_id=%s "
                        "category=%s error=%s",
                        user_id,
                        org_id,
                        category,
                        legacy_error,
                    )
                    rows = []

        normalized_rows = []
        for row in rows:
            normalized_rows.append({
                "id": row.get("id"),
                "category": row.get("category"),
                "context_key": row.get("context_key"),
                "title": row.get("title"),
                "last_message_preview": row.get("last_message_preview"),
                "last_message_role": row.get("last_message_role"),
                "last_message_at": row.get("last_message_at"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            })

        has_more = len(normalized_rows) > safe_limit
        page_rows = normalized_rows[:safe_limit]
        next_cursor = None
        if has_more and page_rows:
            next_cursor = page_rows[-1].get("updated_at") or page_rows[-1].get("created_at")

        logger.info(
            "list_history_for_user rows=%s has_more=%s stage=%s user_id=%s org_id=%s category=%s",
            len(page_rows),
            has_more,
            fallback_stage,
            user_id,
            org_id,
            category,
        )

        return {
            "sessions": page_rows,
            "next_cursor": next_cursor,
        }


# Global session manager instance
session_manager = SupabaseChatSessionManager()
