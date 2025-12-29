"""
Streams API routes for document hierarchy management.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.supabase_auth import require_admin
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class StreamCreate(BaseModel):
    name: str
    code: str
    max_years: int = 4


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
    admin: dict = Depends(require_admin)
):
    """List all streams for the organization."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    query = client.table("streams").select("*").eq("org_id", org_id).order("sort_order")
    
    if not include_inactive:
        query = query.eq("is_active", True)
    
    result = query.execute()
    return result.data


@router.post("", response_model=StreamResponse)
async def create_stream(
    stream: StreamCreate,
    admin: dict = Depends(require_admin)
):
    """Create a new stream."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    data = {
        "org_id": org_id,
        "name": stream.name,
        "code": stream.code.upper(),
        "max_years": stream.max_years,
    }
    
    result = client.table("streams").insert(data).execute()
    
    if result.data:
        # Auto-create years for the stream
        stream_id = result.data[0]["id"]
        years_data = [
            {
                "org_id": org_id,
                "stream_id": stream_id,
                "year_number": i,
                "display_name": f"Year {i}",
                "code": str(i)
            }
            for i in range(1, stream.max_years + 1)
        ]
        client.table("years").insert(years_data).execute()
        
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
    
    result = client.table("streams").update(update_data).eq("id", stream_id).eq("org_id", org_id).execute()
    
    if result.data:
        return result.data[0]
    
    raise HTTPException(status_code=404, detail="Stream not found")


@router.delete("/{stream_id}")
async def delete_stream(
    stream_id: str,
    admin: dict = Depends(require_admin)
):
    """Delete a stream (and its years)."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    # First delete associated years
    client.table("years").delete().eq("stream_id", stream_id).execute()
    
    # Then delete the stream
    result = client.table("streams").delete().eq("id", stream_id).eq("org_id", org_id).execute()
    
    return {"success": True, "message": "Stream deleted"}
