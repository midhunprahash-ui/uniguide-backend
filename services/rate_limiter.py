"""
Rate limiting configuration for API protection.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

# Create rate limiter instance
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# Rate limit configurations
RATE_LIMITS = {
    "chat": "20/minute",      # Chat queries (expensive AI calls)
    "upload": "10/minute",    # Document uploads
    "admin": "60/minute",     # Admin operations
    "public": "100/minute",   # General public endpoints
}
