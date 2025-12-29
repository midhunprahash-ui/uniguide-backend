"""
Streams API routes for document hierarchy management.
New hierarchy: Stream -> Department -> Year
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.supabase_auth import require_admin, get_current_user
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class StreamCreate(BaseModel):
    name: str
    code: str
    max_years: int = 4  # Kept for reference/UI hints, but years are now per-department


class StreamUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    max_years: Optional[int] = None
    is_active: Optional[bool] = None
    sort_order: Optional[int] = None


class StreamResponse(BaseModel):
    id: str
    org_id: str
    name: str
    code: str
    max_years: int
    sort_order: int
    is_active: bool


@router.get("", response_model=list[StreamResponse])
async def list_streams(
    include_inactive: bool = False,
    user: dict = Depends(get_current_user)
):
    """List all streams for the organization. Accessible by anyone."""
    client = get_supabase_admin_client()

    # Get user's org or default to sjit for anonymous users
    if user:
        profile = client.table("profiles").select("org_id, role").eq("id", user["id"]).single().execute()
        org_id = profile.data.get("org_id") if profile.data else None
        is_admin = profile.data.get("role") in ["admin", "superadmin"] if profile.data else False
    else:
        # Anonymous user - get default org
        org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
        org_id = org.data.get("id") if org.data else None
        is_admin = False

    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")

    query = client.table("streams").select("*").eq("org_id", org_id).order("sort_order")

    # Only admins can see inactive streams
    if not include_inactive or not is_admin:
        query = query.eq("is_active", True)

    result = query.execute()
    return result.data


@router.post("", response_model=StreamResponse)
async def create_stream(
    stream: StreamCreate,
    admin: dict = Depends(require_admin)
):
    """Create a new stream. Years are now created per-department, not per-stream."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Check for duplicate code
    existing = client.table("streams").select("id").eq("org_id", org_id).eq("code", stream.code.upper()).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Stream with this code already exists")

    data = {
        "org_id": org_id,
        "name": stream.name,
        "code": stream.code.upper(),
        "max_years": stream.max_years,
    }

    result = client.table("streams").insert(data).execute()

    if result.data:
        # NOTE: Years are NO LONGER auto-created here
        # Years are now created per-department via DepartmentManagement or YearManagement
        logger.info(f"Created stream: {stream.name} ({stream.code})")
        return result.data[0]

    raise HTTPException(status_code=400, detail="Failed to create stream")


@router.put("/{stream_id}", response_model=StreamResponse)
async def update_stream(
    stream_id: str,
    stream: StreamUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a stream."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    update_data = {k: v for k, v in stream.dict().items() if v is not None}

    # Uppercase code if provided
    if "code" in update_data:
        update_data["code"] = update_data["code"].upper()

    result = client.table("streams").update(update_data).eq("id", stream_id).eq("org_id", org_id).execute()

    if result.data:
        return result.data[0]

    raise HTTPException(status_code=404, detail="Stream not found")


@router.delete("/{stream_id}")
async def delete_stream(
    stream_id: str,
    admin: dict = Depends(require_admin)
):
    """Delete a stream. Cascades to departments and years."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Get all departments for this stream (filter by org_id for safety)
    deps = client.table("departments").select("id").eq("stream_id", stream_id).eq("org_id", org_id).execute()

    # Delete years for each department (with org_id for safety)
    for dept in (deps.data or []):
        client.table("years").delete().eq("department_id", dept["id"]).eq("org_id", org_id).execute()

    # Delete all departments for this stream (with org_id for safety)
    client.table("departments").delete().eq("stream_id", stream_id).eq("org_id", org_id).execute()

    # Delete the stream
    result = client.table("streams").delete().eq("id", stream_id).eq("org_id", org_id).execute()

    logger.info(f"Deleted stream: {stream_id}")
    return {"success": True, "message": "Stream deleted"}
