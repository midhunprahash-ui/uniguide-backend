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
    unit_count: int = Field(default=5, ge=1, le=20)
    description: Optional[str] = None


class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
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
    unit_count: int
    description: Optional[str]
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
    include_inactive: bool = False,
    user: dict = Depends(get_current_user)
):
    """
    List subjects. Filter by year_id or department_id.
    Students see only active subjects; admins can see all.
    """
    client = get_supabase_admin_client()
    
    # Get user's org
    profile = client.table("profiles").select("org_id, role").eq("id", user["id"]).single().execute()
    org_id = profile.data.get("org_id") if profile.data else None
    is_admin = profile.data.get("role") in ["admin", "superadmin"] if profile.data else False
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Build query
    query = client.table("subjects").select("*").eq("org_id", org_id).order("sort_order")
    
    if year_id:
        query = query.eq("year_id", year_id)
    
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
    
    update_data = {k: v for k, v in subject.dict().items() if v is not None}
    
    # Uppercase code if provided
    if "code" in update_data:
        update_data["code"] = update_data["code"].upper()
    
    result = client.table("subjects").update(update_data).eq("id", subject_id).eq("org_id", org_id).execute()
    
    if result.data:
        return result.data[0]
    
    raise HTTPException(status_code=404, detail="Subject not found")


@router.delete("/{subject_id}")
async def delete_subject(
    subject_id: str,
    admin: dict = Depends(require_admin)
):
    """
    Delete a subject. Cascades to units and notes.
    WARNING: This permanently deletes all notes for this subject.
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    
    # Verify subject exists
    subject = client.table("subjects").select("id, name, code").eq("id", subject_id).eq("org_id", org_id).single().execute()
    if not subject.data:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Count notes that will be deleted
    notes_count = client.table("notes").select("id", count="exact").eq("subject_id", subject_id).execute()
    
    # Delete subject (cascades to units and notes due to FK constraints)
    client.table("subjects").delete().eq("id", subject_id).eq("org_id", org_id).execute()
    
    logger.info(f"Deleted subject: {subject.data['name']} ({subject.data['code']}) and {notes_count.count or 0} notes")
    
    return {
        "success": True,
        "message": f"Subject deleted along with {notes_count.count or 0} notes"
    }


# ============================================================================
# Unit Endpoints
# ============================================================================

@router.get("/{subject_id}/units", response_model=list[UnitResponse])
async def list_units(
    subject_id: str,
    user: dict = Depends(get_current_user)
):
    """List all units for a subject."""
    client = get_supabase_admin_client()
    
    # Get user's org
    profile = client.table("profiles").select("org_id").eq("id", user["id"]).single().execute()
    org_id = profile.data.get("org_id") if profile.data else None
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Verify subject belongs to user's org
    subject = client.table("subjects").select("id").eq("id", subject_id).eq("org_id", org_id).single().execute()
    if not subject.data:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    units = client.table("subject_units").select("*").eq("subject_id", subject_id).order("unit_number").execute()
    
    return units.data or []


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
