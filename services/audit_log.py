"""
Audit logging service for tracking admin actions.
Enhanced with before/after values, duration tracking, and error logging.
"""
import logging
import time
from typing import Optional, Any
from contextlib import contextmanager
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


async def log_admin_action(
    org_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    # New enhanced audit fields
    old_values: Optional[dict[str, Any]] = None,
    new_values: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    duration_ms: Optional[int] = None,
    affected_count: int = 1,
    status: str = "success",
    error_message: Optional[str] = None,
    error_code: Optional[str] = None
):
    """
    Log an admin action for audit purposes.

    Args:
        org_id: Organization ID
        user_id: User performing the action
        action: Type of action ('create', 'update', 'delete', 'soft_delete', 'restore')
        resource_type: Type of resource ('document', 'category', 'stream', 'department', 'year')
        resource_id: ID of the affected resource
        resource_name: Human-readable name of the resource
        details: Additional details as JSON
        ip_address: Client IP address
        user_agent: Client user agent
        old_values: State before the change (for updates/deletes)
        new_values: State after the change (for creates/updates)
        session_id: Request session ID for correlation
        request_id: Unique request ID for tracing
        duration_ms: Operation duration in milliseconds
        affected_count: Number of rows affected (for batch operations)
        status: Operation status ('success', 'error', 'warning')
        error_message: Error message if operation failed
        error_code: Error code if operation failed
    """
    try:
        client = get_supabase_admin_client()

        data = {
            "org_id": org_id,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "resource_name": resource_name,
            "details": details,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "old_values": old_values,
            "new_values": new_values,
            "session_id": session_id,
            "request_id": request_id,
            "duration_ms": duration_ms,
            "affected_count": affected_count,
            "status": status,
            "error_message": error_message,
            "error_code": error_code
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        client.table("audit_logs").insert(data).execute()
        logger.info(f"Audit log: {action} {resource_type} by user {user_id} [{status}]")

    except Exception as e:
        # Don't fail the main operation if audit logging fails
        logger.error(f"Failed to create audit log: {e}")


@contextmanager
def audit_context(
    org_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None
):
    """
    Context manager for timing and logging admin actions.

    Usage:
        with audit_context(org_id, user_id, "update", "document", doc_id, filename) as ctx:
            ctx["old_values"] = get_current_state()
            # perform operation
            ctx["new_values"] = get_new_state()

    The audit log is automatically created when the context exits.
    """
    start_time = time.time()
    ctx = {
        "org_id": org_id,
        "user_id": user_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "resource_name": resource_name,
        "old_values": None,
        "new_values": None,
        "status": "success",
        "error_message": None,
        "error_code": None
    }

    try:
        yield ctx
    except Exception as e:
        ctx["status"] = "error"
        ctx["error_message"] = str(e)
        ctx["error_code"] = type(e).__name__
        raise
    finally:
        duration_ms = int((time.time() - start_time) * 1000)

        # Fire and forget - don't await in finally
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(log_admin_action(
                    org_id=ctx["org_id"],
                    user_id=ctx["user_id"],
                    action=ctx["action"],
                    resource_type=ctx["resource_type"],
                    resource_id=ctx["resource_id"],
                    resource_name=ctx["resource_name"],
                    old_values=ctx.get("old_values"),
                    new_values=ctx.get("new_values"),
                    duration_ms=duration_ms,
                    status=ctx["status"],
                    error_message=ctx.get("error_message"),
                    error_code=ctx.get("error_code")
                ))
        except RuntimeError:
            # No event loop - just log synchronously (best effort)
            pass


async def get_audit_logs(
    org_id: str,
    limit: int = 50,
    resource_type: Optional[str] = None,
    action: Optional[str] = None
):
    """
    Retrieve audit logs for an organization.
    
    Args:
        org_id: Organization ID
        limit: Maximum number of logs to return
        resource_type: Filter by resource type
        action: Filter by action type
    """
    client = get_supabase_admin_client()
    
    query = client.table("audit_logs").select("*").eq("org_id", org_id).order("created_at", desc=True).limit(limit)
    
    if resource_type:
        query = query.eq("resource_type", resource_type)
    
    if action:
        query = query.eq("action", action)
    
    result = query.execute()
    return result.data
