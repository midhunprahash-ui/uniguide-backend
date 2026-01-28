"""
Subjects API routes for Notes RAG curriculum management.
Hierarchy: Stream → Department → Year → Subject → Unit
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from services.supabase_auth import require_admin, get_current_user
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class SubjectCreate(BaseModel):
    year_id: str
    name: str
    code: str
    semester: int = Field(..., ge=1, le=8, description="Semester number (1-8). Sem 1-2 = Year 1, etc.")
    unit_count: int = Field(default=5, ge=1, le=20)
    description: Optional[str] = None


class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    year_id: Optional[str] = None  # Allow changing the year (which determines dept/stream)
    semester: Optional[int] = Field(None, ge=1, le=8)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    sort_order: Optional[int] = None


class UnitUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class SubjectResponse(BaseModel):
    id: str
    org_id: str
    year_id: str
    name: str
    code: str
    semester: Optional[int] = None
    unit_count: int
    description: Optional[str] = None
    is_active: bool
    sort_order: int
    created_at: str
    updated_at: str


class UnitResponse(BaseModel):
    id: str
    org_id: str
    subject_id: str
    unit_number: int
    name: Optional[str]
    description: Optional[str]


class SubjectWithUnitsResponse(SubjectResponse):
    units: list[UnitResponse]


# ============================================================================
# Subject Endpoints
# ============================================================================

@router.get("", response_model=list[SubjectResponse])
async def list_subjects(
    year_id: Optional[str] = Query(None, description="Filter by year ID"),
    department_id: Optional[str] = Query(None, description="Filter by department ID"),
    semester: Optional[int] = Query(None, ge=1, le=8, description="Filter by semester (1-8)"),
    include_inactive: bool = False,
    user: dict = Depends(get_current_user)
):
    """
    List subjects. Filter by year_id, department_id, or semester.
    Students see only active subjects; admins can see all.
    Supports both authenticated and anonymous users.
    """
    client = get_supabase_admin_client()
    
    # Get user's org or default to org for anonymous users
    if user:
        profile = client.table("profiles").select("org_id, role").eq("id", user["id"]).single().execute()
        org_id = profile.data.get("org_id") if profile.data else None
        is_admin = profile.data.get("role") in ["admin", "superadmin"] if profile.data else False
    else:
        # Anonymous user - get org from slug (passed via query param or default)
        org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
        org_id = org.data.get("id") if org.data else None
        is_admin = False
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Build query
    query = client.table("subjects").select("*").eq("org_id", org_id).order("sort_order")
    
    if year_id:
        query = query.eq("year_id", year_id)
    
    if semester:
        query = query.eq("semester", semester)
    
    # If filtering by department, we need to get years for that department first
    if department_id:
        years = client.table("years").select("id").eq("department_id", department_id).execute()
        year_ids = [y["id"] for y in (years.data or [])]
        if year_ids:
            query = query.in_("year_id", year_ids)
        else:
            return []  # No years in department means no subjects
    
    # Only admins can see inactive subjects
    if not include_inactive or not is_admin:
        query = query.eq("is_active", True)
    
    result = query.execute()
    return result.data


@router.get("/with-stats", response_model=None)
async def list_subjects_with_stats(
    year_id: Optional[str] = Query(None, description="Filter by year ID"),
    semester: Optional[int] = Query(None, ge=1, le=8, description="Filter by semester (1-8)"),
    user: Optional[dict] = Depends(get_current_user)
):
    """
    List subjects with unit counts and notes statistics.
    OPTIMIZED: Uses single RPC call instead of 6+ separate queries.
    Supports both authenticated and anonymous users.
    """
    client = get_supabase_admin_client()
    
    # Get user's org or default to org for anonymous users
    if user:
        profile = client.table("profiles").select("org_id, role").eq("id", user["id"]).single().execute()
        org_id = profile.data.get("org_id") if profile.data else None
        is_admin = profile.data.get("role") in ["admin", "superadmin"] if profile.data else False
    else:
        # Anonymous user - get org from slug (default org)
        org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
        org_id = org.data.get("id") if org.data else None
        is_admin = False
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # OPTIMIZED: Single RPC call replaces 6+ separate queries
    result = client.rpc("get_subjects_with_stats", {
        "p_org_id": org_id,
        "p_year_id": year_id,
        "p_semester": semester,
        "p_include_inactive": is_admin
    }).execute()
    
    return result.data or []



@router.get("/{subject_id}", response_model=SubjectWithUnitsResponse)
async def get_subject(
    subject_id: str,
    user: dict = Depends(get_current_user)
):
    """Get a subject with all its units."""
    client = get_supabase_admin_client()
    
    # Get user's org
    profile = client.table("profiles").select("org_id").eq("id", user["id"]).single().execute()
    org_id = profile.data.get("org_id") if profile.data else None
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Get subject
    subject = client.table("subjects").select("*").eq("id", subject_id).eq("org_id", org_id).single().execute()
    
    if not subject.data:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Get units
    units = client.table("subject_units").select("*").eq("subject_id", subject_id).order("unit_number").execute()
    
    return {
        **subject.data,
        "units": units.data or []
    }


@router.post("", response_model=SubjectWithUnitsResponse)
async def create_subject(
    subject: SubjectCreate,
    admin: dict = Depends(require_admin)
):
    """
    Create a new subject. Units are auto-created based on unit_count.
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    # Verify year exists and belongs to this org
    year = client.table("years").select("id, department_id").eq("id", subject.year_id).eq("org_id", org_id).single().execute()
    if not year.data:
        raise HTTPException(status_code=404, detail="Year not found")
    
    # Check for duplicate code in same year
    existing = client.table("subjects").select("id").eq("org_id", org_id).eq("year_id", subject.year_id).eq("code", subject.code.upper()).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail=f"Subject with code '{subject.code}' already exists in this year")
    
    # Create subject (units are auto-created by trigger)
    data = {
        "org_id": org_id,
        "year_id": subject.year_id,
        "name": subject.name,
        "code": subject.code.upper(),
        "semester": subject.semester,
        "unit_count": subject.unit_count,
        "description": subject.description,
    }
    
    result = client.table("subjects").insert(data).execute()
    
    if result.data:
        subject_id = result.data[0]["id"]
        
        # Fetch units (auto-created by trigger)
        units = client.table("subject_units").select("*").eq("subject_id", subject_id).order("unit_number").execute()
        
        logger.info(f"Created subject: {subject.name} ({subject.code}) with {subject.unit_count} units")
        
        return {
            **result.data[0],
            "units": units.data or []
        }
    
    raise HTTPException(status_code=400, detail="Failed to create subject")


@router.put("/{subject_id}", response_model=SubjectResponse)
async def update_subject(
    subject_id: str,
    subject: SubjectUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a subject. Note: unit_count cannot be changed after creation."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    try:
        # Use model_dump() for Pydantic v2 compatibility (dict() is deprecated)
        update_data = {k: v for k, v in subject.model_dump().items() if v is not None}
        
        # Uppercase code if provided
        if "code" in update_data:
            update_data["code"] = update_data["code"].upper()
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        result = client.table("subjects").update(update_data).eq("id", subject_id).eq("org_id", org_id).execute()
        
        if result.data:
            return result.data[0]
        
        raise HTTPException(status_code=404, detail="Subject not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update subject {subject_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update subject: {str(e)}")


@router.delete("/{subject_id}")
async def delete_subject(
    subject_id: str,
    admin: dict = Depends(require_admin)
):
    """
    Delete a subject. Manually cascades to units and notes.
    WARNING: This permanently deletes all notes for this subject.
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    try:
        # Verify subject exists
        subject = client.table("subjects").select("id, name, code").eq("id", subject_id).eq("org_id", org_id).single().execute()
        if not subject.data:
            raise HTTPException(status_code=404, detail="Subject not found")
        
        # Get all units for this subject (table is 'subject_units', not 'units')
        units = client.table("subject_units").select("id").eq("subject_id", subject_id).execute()
        unit_ids = [u["id"] for u in (units.data or [])]
        
        # Delete notes first (they reference units)
        notes_deleted = 0
        if unit_ids:
            notes_result = client.table("notes").delete().in_("unit_id", unit_ids).execute()
            notes_deleted = len(notes_result.data or [])
        
        # Also delete notes that reference subject directly
        direct_notes = client.table("notes").delete().eq("subject_id", subject_id).execute()
        notes_deleted += len(direct_notes.data or [])
        
        # Delete units (table is 'subject_units')
        if unit_ids:
            client.table("subject_units").delete().eq("subject_id", subject_id).execute()
        
        # Finally delete subject
        client.table("subjects").delete().eq("id", subject_id).eq("org_id", org_id).execute()
        
        logger.info(f"Deleted subject: {subject.data['name']} ({subject.data['code']}) and {notes_deleted} notes")
        
        return {
            "success": True,
            "message": f"Subject deleted along with {notes_deleted} notes"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete subject {subject_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete subject: {str(e)}")


# ============================================================================
# Unit Endpoints
# ============================================================================

@router.get("/{subject_id}/units", response_model=None)
async def list_units(
    subject_id: str,
    with_notes_only: bool = False,
    user: dict = Depends(get_current_user)
):
    """
    List all units for a subject with notes count.
    OPTIMIZED: Uses single RPC call instead of 3 separate queries.
    Supports both authenticated and anonymous users.
    """
    client = get_supabase_admin_client()
    
    # Get user's org or default to org for anonymous users
    if user:
        profile = client.table("profiles").select("org_id").eq("id", user["id"]).single().execute()
        org_id = profile.data.get("org_id") if profile.data else None
    else:
        # Anonymous user - use default org
        org = client.table("organizations").select("id").eq("slug", "sjit").single().execute()
        org_id = org.data.get("id") if org.data else None
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # OPTIMIZED: Single RPC call replaces 3 separate queries
    # (subject verification, units fetch, notes count aggregation)
    result = client.rpc("get_units_with_notes_count", {
        "p_org_id": org_id,
        "p_subject_id": subject_id,
        "p_with_notes_only": with_notes_only
    }).execute()
    
    # RPC returns empty array if subject not found
    if not result.data:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    return result.data



@router.put("/{subject_id}/units/{unit_id}", response_model=UnitResponse)
async def update_unit(
    subject_id: str,
    unit_id: str,
    unit: UnitUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a unit's name or description."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    # Verify unit belongs to subject and org
    existing = client.table("subject_units").select("id").eq("id", unit_id).eq("subject_id", subject_id).eq("org_id", org_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Unit not found")
    
    update_data = {k: v for k, v in unit.dict().items() if v is not None}
    
    result = client.table("subject_units").update(update_data).eq("id", unit_id).execute()
    
    if result.data:
        return result.data[0]
    
    raise HTTPException(status_code=400, detail="Failed to update unit")


# ============================================================================
# Statistics Endpoint
# ============================================================================

@router.get("/stats/overview")
async def get_subjects_stats(
    admin: dict = Depends(require_admin)
):
    """Get subject statistics for the organization."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    result = client.rpc("get_notes_stats", {"filter_org_id": org_id}).execute()
    
    if result.data:
        return result.data[0] if isinstance(result.data, list) else result.data
    
    return {
        "total_subjects": 0,
        "total_notes": 0,
        "total_chunks": 0,
        "subjects_by_year": []
    }

