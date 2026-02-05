"""Admin statistics utilities."""
import logging

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


def refresh_document_stats() -> None:
    """Refresh the admin_document_stats materialized view."""
    try:
        client = get_supabase_admin_client()
        client.rpc("refresh_admin_document_stats", {}).execute()
        logger.debug("Refreshed admin_document_stats materialized view")
    except Exception as e:
        logger.warning(f"Failed to refresh admin_document_stats: {e}")
