"""
Years API routes for document hierarchy management.
Years now belong to departments (Stream -> Department -> Year hierarchy).
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.supabase_auth import require_admin, get_current_user
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class YearCreate(BaseModel):
    department_id: str  # Years now belong to departments
    year_number: int
    display_name: str
    code: Optional[str] = None


class YearUpdate(BaseModel):
    display_name: Optional[str] = None
    is_active: Optional[bool] = None


class YearResponse(BaseModel):
    id: str
    org_id: str
    department_id: Optional[str] = None  # Nullable until migration completes
    stream_id: Optional[str] = None  # Keep for backward compatibility during transition
    year_number: int
    display_name: str
    code: Optional[str] = None  # May be null in database
    is_active: bool


@router.get("", response_model=list[YearResponse])
async def list_years(
    department_id: Optional[str] = None,  # Primary filter
    stream_id: Optional[str] = None,  # Secondary filter (via departments)
    include_inactive: bool = False,
    user: dict = Depends(get_current_user)
):
    """List all years, optionally filtered by department or stream. Accessible by anyone."""
    client = get_supabase_admin_client()

    # Use cached org_id and role from auth (no redundant profile query)
    if user:
        org_id = user.get("org_id")
        is_admin = user.get("role") in ["admin", "superadmin"]
    else:
        # Anonymous user - get default org
        org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
        org_id = org.data.get("id") if org.data else None
        is_admin = False

    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")

    query = client.table("years").select("*").eq("org_id", org_id).order("year_number")

    if department_id:
        # Direct filter by department
        query = query.eq("department_id", department_id)
    elif stream_id:
        # Filter by stream: get all departments for this stream, then get years
        deps = client.table("departments").select("id").eq("stream_id", stream_id).eq("org_id", org_id).execute()
        dep_ids = [d["id"] for d in deps.data] if deps.data else []
        if dep_ids:
            query = query.in_("department_id", dep_ids)
        else:
            # No departments in this stream, return empty
            return []

    # Only admins can see inactive years
    if not include_inactive or not is_admin:
        query = query.eq("is_active", True)

    result = query.execute()
    return result.data


@router.post("", response_model=YearResponse)
async def create_year(
    year: YearCreate,
    admin: dict = Depends(require_admin)
):
    """Create a new year under a department."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Validate department exists and get its stream_id for backward compatibility
    dept = client.table("departments").select("id, stream_id").eq("id", year.department_id).eq("org_id", org_id).single().execute()
    if not dept.data:
        raise HTTPException(status_code=400, detail="Invalid department_id")

    # Check for duplicate year_number in this department
    existing = client.table("years").select("id").eq("department_id", year.department_id).eq("year_number", year.year_number).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail=f"Year {year.year_number} already exists in this department")

    data = {
        "org_id": org_id,
        "department_id": year.department_id,
        "stream_id": dept.data.get("stream_id"),  # Keep for backward compatibility
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
