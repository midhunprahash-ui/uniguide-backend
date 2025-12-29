"""
Health check routes for monitoring and load balancers.
"""
import logging
from fastapi import APIRouter, HTTPException
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check - returns 200 if the server is running.
    Used by load balancers for basic liveness checks.
    """
    return {
        "status": "healthy",
        "service": "uniguide-api"
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - verifies database connectivity.
    Used by orchestrators to know when the service is ready to accept traffic.
    """
    try:
        client = get_supabase_admin_client()
        # Simple query to verify database connection
        result = client.table("organizations").select("id").limit(1).execute()
        
        return {
            "status": "ready",
            "database": "connected",
            "service": "uniguide-api"
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail={
            "status": "not_ready",
            "database": "disconnected",
            "error": str(e)
        })
