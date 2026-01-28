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
    Verify Supabase JWT and return user data including profile info.
    Returns None if no token provided (for anonymous access).

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        User data dict with org_id and role, or None for anonymous

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
        
        # Fetch profile data once during auth (eliminates redundant queries in routes)
        try:
            profile = client.table("profiles").select("org_id, role").eq("id", str(user.id)).single().execute()
            profile_data = profile.data or {}
        except Exception as e:
            logger.warning(f"Could not fetch profile for user {user.id}: {e}")
            profile_data = {}
        
        return {
            "id": str(user.id),
            "email": user.email,
            "user_metadata": user.user_metadata,
            "org_id": profile_data.get("org_id"),
            "role": profile_data.get("role", "student")
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
        User data dict with role and org_id (already cached from auth)

    Raises:
        HTTPException: If not admin
    """
    # org_id and role are now already in user dict from get_current_user
    role = user.get("role", "student")
    org_id = user.get("org_id")
    
    # Fallback: only query if role/org_id missing (shouldn't happen normally)
    if not role or not org_id:
        user_id = user.get("id")
        if user_id:
            client = get_supabase_admin_client()
            try:
                result = client.table("profiles").select("role, org_id").eq("id", user_id).single().execute()
                role = result.data.get("role", "student") if result.data else "student"
                org_id = result.data.get("org_id") if result.data else None
            except Exception as e:
                logger.error(f"Error getting user profile: {e}")
                role = "student"
                org_id = None

    if role not in ["admin", "superadmin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return {**user, "role": role, "org_id": org_id}


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
