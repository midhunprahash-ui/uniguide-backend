"""
Supabase Authentication utilities.
Replaces custom JWT auth with Supabase Auth.
"""
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> dict | None:
    """
    Verify Supabase JWT and return user data.
    Returns None if no token provided (for anonymous access).

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        User data dict or None for anonymous

    Raises:
        HTTPException: If token is invalid
    """
    if not credentials:
        return None

    client = get_supabase_admin_client()

    try:
        # Verify the JWT token using Supabase
        user_response = client.auth.get_user(credentials.credentials)

        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        user = user_response.user
        return {
            "id": str(user.id),
            "email": user.email,
            "user_metadata": user.user_metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def require_auth(
    user: dict | None = Depends(get_current_user)
) -> dict:
    """
    Require authenticated user.

    Returns:
        User data dict

    Raises:
        HTTPException: If not authenticated
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


async def get_user_role(user_id: str) -> str:
    """
    Get user's role from profiles table.

    Args:
        user_id: User UUID

    Returns:
        Role string ('student', 'admin', 'superadmin')
    """
    client = get_supabase_admin_client()

    try:
        result = client.table("profiles").select("role").eq("id", user_id).single().execute()
        return result.data.get("role", "student") if result.data else "student"
    except Exception as e:
        logger.error(f"Error getting user role: {e}")
        return "student"


async def require_admin(
    user: dict = Depends(require_auth)
) -> dict:
    """
    Require admin or superadmin role.

    Returns:
        User data dict with role

    Raises:
        HTTPException: If not admin
    """
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found"
        )

    role = await get_user_role(user_id)

    if role not in ["admin", "superadmin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return {**user, "role": role}


# Legacy compatibility - for existing code that uses verify_token
def verify_token(token: str) -> dict | None:
    """
    Legacy token verification (deprecated).
    Use get_current_user dependency instead.
    """
    client = get_supabase_admin_client()

    try:
        user_response = client.auth.get_user(token)
        if user_response and user_response.user:
            return {
                "id": str(user_response.user.id),
                "email": user_response.user.email
            }
    except Exception as e:
        logger.error(f"Token verification failed: {e}")

    return None
