"""
Authentication service using Supabase Auth.
Provides JWT verification and role-based access control.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client

from config import get_settings

security = HTTPBearer(auto_error=False)
settings = get_settings()


def get_supabase_client() -> Client:
    """Get Supabase client with service role key (bypasses RLS)."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_supabase_anon_client() -> Client:
    """Get Supabase client with anon key (respects RLS)."""
    return create_client(settings.supabase_url, settings.supabase_anon_key)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> dict | None:
    """
    Verify Supabase JWT and return user data.
    Returns None if no token provided (for public endpoints).
    """
    if not credentials:
        return None

    client = get_supabase_client()

    try:
        # Verify the JWT token with Supabase
        user_response = client.auth.get_user(credentials.credentials)

        if not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        return {
            "id": user_response.user.id,
            "email": user_response.user.email,
            "metadata": user_response.user.user_metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> dict:
    """Require valid authentication (any authenticated user)."""
    client = get_supabase_client()

    try:
        user_response = client.auth.get_user(credentials.credentials)

        if not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        return {
            "id": user_response.user.id,
            "email": user_response.user.email,
            "metadata": user_response.user.user_metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


async def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> dict:
    """
    Require admin role.
    Checks the profiles table for admin or superadmin role.
    """
    client = get_supabase_client()

    try:
        # First verify the JWT
        user_response = client.auth.get_user(credentials.credentials)

        if not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        user_id = user_response.user.id

        # Check user role in profiles table
        profile_response = client.table("profiles").select("role").eq("id", user_id).single().execute()

        if not profile_response.data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User profile not found"
            )

        role = profile_response.data.get("role", "student")

        if role not in ["admin", "superadmin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )

        return {
            "id": user_id,
            "email": user_response.user.email,
            "role": role,
            "metadata": user_response.user.user_metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


async def require_superadmin(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> dict:
    """Require superadmin role for sensitive operations."""
    admin = await require_admin(credentials)

    if admin.get("role") != "superadmin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required"
        )

    return admin
