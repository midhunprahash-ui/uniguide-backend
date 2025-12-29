"""
Audit logging service for tracking admin actions.
"""
import logging
from typing import Optional, Any
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
    user_agent: Optional[str] = None
):
    """
    Log an admin action for audit purposes.
    
    Args:
        org_id: Organization ID
        user_id: User performing the action
        action: Type of action ('create', 'update', 'delete')
        resource_type: Type of resource ('document', 'category', 'stream', 'year')
        resource_id: ID of the affected resource
        resource_name: Human-readable name of the resource
        details: Additional details as JSON
        ip_address: Client IP address
        user_agent: Client user agent
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
            "user_agent": user_agent
        }
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        client.table("audit_logs").insert(data).execute()
        logger.info(f"Audit log: {action} {resource_type} by user {user_id}")
        
    except Exception as e:
        # Don't fail the main operation if audit logging fails
        logger.error(f"Failed to create audit log: {e}")


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
