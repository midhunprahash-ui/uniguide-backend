"""
Authentication service using Supabase Auth.
Provides JWT verification and role-based access control.
"""
import logging
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client

from config import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)
settings = get_settings()

# Cache for valid org_ids (reduces database lookups)
_valid_org_ids_cache: set[str] = set()
_org_cache_loaded = False


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


# =============================================================================
# Organization ID Validation (for tenant isolation)
# =============================================================================

def _load_valid_org_ids() -> set[str]:
    """Load all valid org_ids from the database (cached)."""
    global _valid_org_ids_cache, _org_cache_loaded

    if _org_cache_loaded:
        return _valid_org_ids_cache

    try:
        client = get_supabase_client()
        result = client.table("organizations").select("id").execute()

        if result.data:
            _valid_org_ids_cache = {str(org["id"]) for org in result.data}
            _org_cache_loaded = True
            logger.info(f"Loaded {len(_valid_org_ids_cache)} valid org_ids into cache")
        else:
            _valid_org_ids_cache = set()
            _org_cache_loaded = True

    except Exception as e:
        logger.error(f"Failed to load org_ids: {e}")
        # Don't cache on failure - retry next time
        return set()

    return _valid_org_ids_cache


def refresh_org_cache() -> None:
    """Force refresh the org_id cache (call when orgs are added/removed)."""
    global _org_cache_loaded
    _org_cache_loaded = False
    _load_valid_org_ids()


def validate_org_id(org_id: str) -> bool:
    """
    Validate that an org_id exists in the organizations table.

    This is a CRITICAL security function for multi-tenant isolation.
    It prevents users from accessing documents from other organizations
    by spoofing org_id in requests.

    Args:
        org_id: The organization ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not org_id:
        return False

    # Check cache first
    valid_orgs = _load_valid_org_ids()
    if org_id in valid_orgs:
        return True

    # If not in cache, do a direct lookup (might be newly created)
    try:
        client = get_supabase_client()
        result = client.table("organizations").select("id").eq("id", org_id).limit(1).execute()

        if result.data and len(result.data) > 0:
            # Add to cache
            _valid_org_ids_cache.add(org_id)
            return True

    except Exception as e:
        logger.error(f"Error validating org_id {org_id}: {e}")

    return False


def require_valid_org_id(org_id: str) -> str:
    """
    Validate org_id and raise HTTPException if invalid.

    Use this in route handlers to ensure tenant isolation.

    Args:
        org_id: The organization ID from the request

    Returns:
        The validated org_id

    Raises:
        HTTPException: If org_id is invalid
    """
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="org_id is required"
        )

    if not validate_org_id(org_id):
        logger.warning(f"Invalid org_id attempted: {org_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid organization"
        )

    return org_id


def get_user_org_id(user_id: str) -> str | None:
    """Fetch the org_id for a given user from profiles."""
    try:
        client = get_supabase_client()
        result = client.table("profiles").select("org_id").eq("id", user_id).single().execute()
        if result.data:
            return result.data.get("org_id")
    except Exception as e:
        logger.error(f"Error fetching org_id for user {user_id}: {e}")
    return None


def require_org_membership(user_id: str, org_id: str) -> str:
    """
    Ensure the user belongs to the requested organization.
    Returns the user's org_id if valid, otherwise raises HTTPException.
    """
    profile_org_id = get_user_org_id(user_id)
    if not profile_org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User organization not found"
        )

    if profile_org_id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to this organization"
        )

    return profile_org_id
