"""
Department routes for dynamic department management.
Departments now belong to streams (Stream -> Department -> Year hierarchy).
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.supabase_auth import require_admin, get_current_user
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class DepartmentCreate(BaseModel):
    name: str
    code: str
    stream_id: str  # Required: department belongs to a stream
    sort_order: int = 0


class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    stream_id: Optional[str] = None  # Can move department to different stream
    is_active: Optional[bool] = None
    sort_order: Optional[int] = None


class DepartmentResponse(BaseModel):
    id: str
    stream_id: str  # Include stream_id in response
    name: str
    code: str
    is_active: bool
    sort_order: int


@router.get("", response_model=list[DepartmentResponse])
async def list_departments(
    stream_id: Optional[str] = None,  # Filter by stream
    include_inactive: bool = False,
    user: dict = Depends(get_current_user)
):
    """List departments for the current organization, optionally filtered by stream."""
    client = get_supabase_admin_client()

    # Get user's organization
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

    query = client.table("departments").select("*").eq("org_id", org_id)

    # Filter by stream if provided
    if stream_id:
        query = query.eq("stream_id", stream_id)

    if not is_admin or not include_inactive:
        query = query.eq("is_active", True)

    query = query.order("sort_order")
    result = query.execute()

    return result.data


@router.post("", response_model=DepartmentResponse)
async def create_department(
    department: DepartmentCreate,
    admin: dict = Depends(require_admin)
):
    """Create a new department (admin only). Requires stream_id."""
    client = get_supabase_admin_client()

    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")

    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Validate stream exists and belongs to org
    stream = client.table("streams").select("id").eq("id", department.stream_id).eq("org_id", org_id).single().execute()
    if not stream.data:
        raise HTTPException(status_code=400, detail="Invalid stream_id")

    # Check if code already exists for this stream (same code allowed in different streams)
    existing = client.table("departments").select("id").eq("org_id", org_id).eq("stream_id", department.stream_id).eq("code", department.code).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Department with this code already exists in this stream")

    result = client.table("departments").insert({
        "org_id": org_id,
        "stream_id": department.stream_id,
        "name": department.name,
        "code": department.code.upper(),
        "sort_order": department.sort_order
    }).execute()

    logger.info(f"Created department: {department.name} ({department.code}) in stream {department.stream_id}")
    return result.data[0]


@router.put("/{department_id}", response_model=DepartmentResponse)
async def update_department(
    department_id: str,
    department: DepartmentUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a department (admin only)."""
    client = get_supabase_admin_client()

    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")

    existing = client.table("departments").select("*").eq("id", department_id).eq("org_id", org_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Department not found")

    update_data = {}
    if department.name is not None:
        update_data["name"] = department.name
    if department.code is not None:
        update_data["code"] = department.code.upper()
    if department.stream_id is not None:
        # Validate new stream exists
        stream = client.table("streams").select("id").eq("id", department.stream_id).eq("org_id", org_id).single().execute()
        if not stream.data:
            raise HTTPException(status_code=400, detail="Invalid stream_id")
        update_data["stream_id"] = department.stream_id
    if department.is_active is not None:
        update_data["is_active"] = department.is_active
    if department.sort_order is not None:
        update_data["sort_order"] = department.sort_order

    if not update_data:
        return existing.data

    # Check for duplicate code in the target stream
    target_stream = update_data.get("stream_id", existing.data["stream_id"])
    target_code = update_data.get("code", existing.data["code"])
    if "code" in update_data or "stream_id" in update_data:
        dup_check = client.table("departments").select("id").eq("org_id", org_id).eq("stream_id", target_stream).eq("code", target_code).neq("id", department_id).execute()
        if dup_check.data:
            raise HTTPException(status_code=400, detail="Department with this code already exists in this stream")

    result = client.table("departments").update(update_data).eq("id", department_id).execute()
    logger.info(f"Updated department: {department_id}")
    return result.data[0]


@router.delete("/{department_id}")
async def delete_department(
    department_id: str,
    admin: dict = Depends(require_admin)
):
    """Delete a department (admin only). Also deletes associated years."""
    client = get_supabase_admin_client()

    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")

    existing = client.table("departments").select("*").eq("id", department_id).eq("org_id", org_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Department not found")

    # Check if department is used in documents (filter by org_id too)
    docs = client.table("documents").select("id", count="exact").eq("org_id", org_id).eq("department", existing.data["code"]).execute()
    if docs.count and docs.count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete department with {docs.count} documents."
        )

    # Delete associated years first (cascade) - with org_id for safety
    client.table("years").delete().eq("department_id", department_id).eq("org_id", org_id).execute()

    # Delete the department - with org_id for safety
    client.table("departments").delete().eq("id", department_id).eq("org_id", org_id).execute()
    logger.info(f"Deleted department: {department_id}")
    return {"message": "Department deleted successfully"}
