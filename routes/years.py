"""
Years API routes for document hierarchy management.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.supabase_auth import require_admin
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class YearCreate(BaseModel):
    stream_id: str
    year_number: int
    display_name: str
    code: Optional[str] = None


class YearUpdate(BaseModel):
    display_name: Optional[str] = None
    is_active: Optional[bool] = None


class YearResponse(BaseModel):
    id: str
    org_id: str
    stream_id: str
    year_number: int
    display_name: str
    code: str
    is_active: bool


@router.get("", response_model=list[YearResponse])
async def list_years(
    stream_id: Optional[str] = None,
    include_inactive: bool = False,
    admin: dict = Depends(require_admin)
):
    """List all years, optionally filtered by stream."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    query = client.table("years").select("*").eq("org_id", org_id).order("year_number")
    
    if stream_id:
        query = query.eq("stream_id", stream_id)
    
    if not include_inactive:
        query = query.eq("is_active", True)
    
    result = query.execute()
    return result.data


@router.post("", response_model=YearResponse)
async def create_year(
    year: YearCreate,
    admin: dict = Depends(require_admin)
):
    """Create a new year."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    data = {
        "org_id": org_id,
        "stream_id": year.stream_id,
        "year_number": year.year_number,
        "display_name": year.display_name,
        "code": year.code or str(year.year_number)
    }
    
    result = client.table("years").insert(data).execute()
    
    if result.data:
        return result.data[0]
    
    raise HTTPException(status_code=400, detail="Failed to create year")


@router.put("/{year_id}", response_model=YearResponse)
async def update_year(
    year_id: str,
    year: YearUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a year."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    update_data = {k: v for k, v in year.dict().items() if v is not None}
    
    result = client.table("years").update(update_data).eq("id", year_id).eq("org_id", org_id).execute()
    
    if result.data:
        return result.data[0]
    
    raise HTTPException(status_code=404, detail="Year not found")


@router.delete("/{year_id}")
async def delete_year(
    year_id: str,
    admin: dict = Depends(require_admin)
):
    """Delete a year."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    result = client.table("years").delete().eq("id", year_id).eq("org_id", org_id).execute()
    
    return {"success": True, "message": "Year deleted"}
