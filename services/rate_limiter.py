"""
Multi-tenant rate limiting for UniGuide API.

Implements industry-standard multi-layer rate limiting:
- Tenant (org_id): Fair quota per organization
- User (user_id): Prevent single user hogging tenant quota  
- IP (fallback): Public endpoints + abuse prevention
"""
import logging
from typing import Optional
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

logger = logging.getLogger(__name__)


def get_org_id_from_request(request: Request) -> Optional[str]:
    """
    Extract org_id from request.
    Checks request body first (for chat queries), then falls back to state (for admin).
    """
    # Try to get from request state (set by auth middleware)
    if hasattr(request.state, 'org_id') and request.state.org_id:
        return request.state.org_id
    
    # Try to get from body (for ChatQuery requests)
    if hasattr(request, '_body_cache'):
        try:
            import json
            body = json.loads(request._body_cache)
            if isinstance(body, dict) and 'org_id' in body:
                return body.get('org_id')
        except:
            pass
    
    return None


def get_user_id_from_request(request: Request) -> Optional[str]:
    """Extract user_id from request state (set by auth middleware)."""
    if hasattr(request.state, 'user_id') and request.state.user_id:
        return request.state.user_id
    return None


def get_org_key(request: Request) -> str:
    """
    Key function for tenant-level rate limiting.
    Returns org_id if available, otherwise falls back to IP.
    """
    org_id = get_org_id_from_request(request)
    if org_id:
        return f"org:{org_id}"
    # Fallback to IP if no org_id (e.g., public endpoints)
    return f"ip:{get_remote_address(request)}"


def get_user_key(request: Request) -> str:
    """
    Key function for user-level rate limiting.
    Returns user_id if available, otherwise falls back to org, then IP.
    """
    user_id = get_user_id_from_request(request)
    if user_id:
        return f"user:{user_id}"
    # Fallback to org-level
    return get_org_key(request)


def get_ip_key(request: Request) -> str:
    """Key function for IP-based rate limiting (fallback/public endpoints)."""
    return f"ip:{get_remote_address(request)}"


# Create rate limiter instance with org-based default
# Falls back to IP if org_id not available
limiter = Limiter(key_func=get_org_key, default_limits=["100/minute"])


# =============================================================================
# Rate Limit Configurations
# =============================================================================

# Per-endpoint rate limits (applied per org unless noted otherwise)
RATE_LIMITS = {
    # Chat endpoints - most expensive (AI calls)
    "chat": "300/minute",           # 60 chat queries/min per org
    "chat_hourly": "10000/hour",     # 500/hour cap for sustained usage
    
    # Upload endpoints - heavy I/O + embeddings
    "upload": "10/minute",         # 20 uploads/min per org
    "upload_hourly": "100/hour",   # 100/hour cap
    
    # Admin operations
    "admin": "120/minute",         # 120 admin ops/min per org
    
    # Public endpoints (login, health, etc.) - per IP
    "public": "60/minute",         # 60/min per IP
    
    # User-level limits (within an org)
    "user_chat": "30/minute",      # Single user: 30 chat queries/min
}


# Tier-based limits for future use (can be loaded from database)
TIER_LIMITS = {
    "free": {
        "chat": "100/hour",
        "upload": "10/hour",
    },
    "basic": {
        "chat": "500/hour",
        "upload": "50/hour",
    },
    "pro": {
        "chat": "2000/hour",
        "upload": "200/hour",
    },
    "enterprise": {
        "chat": "10000/hour",
        "upload": "1000/hour",
    },
}


def get_tier_limit(tier: str, limit_type: str) -> str:
    """Get rate limit for a specific tier and limit type."""
    tier_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    return tier_config.get(limit_type, RATE_LIMITS.get(limit_type, "100/minute"))


# =============================================================================
# Rate Limit Headers Helper
# =============================================================================

def add_rate_limit_headers(response, limit: int, remaining: int, reset_time: int):
    """
    Add standard rate limit headers to response.
    
    Headers:
        X-RateLimit-Limit: Maximum requests allowed
        X-RateLimit-Remaining: Requests remaining in window
        X-RateLimit-Reset: Unix timestamp when limit resets
    """
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Reset"] = str(reset_time)
    return response
