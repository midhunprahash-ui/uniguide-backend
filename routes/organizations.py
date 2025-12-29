"""
Organization routes for multi-tenant management.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.supabase_auth import get_current_user
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    logo_url: Optional[str] = None
    primary_color: Optional[str] = "#000000"  # Default if null
    secondary_color: Optional[str] = "#ffffff"  # Default if null
    settings: Optional[dict] = None


@router.get("/current", response_model=OrganizationResponse)
async def get_current_organization(user: dict = Depends(get_current_user)):
    """
    Get the current user's organization details.
    Used by frontend to customize branding and settings.
    """
    client = get_supabase_admin_client()
    
    if user:
        # Get user's organization
        profile = client.table("profiles").select("org_id").eq("id", user["id"]).single().execute()
        org_id = profile.data.get("org_id") if profile.data else None
        
        if org_id:
            org = client.table("organizations").select("*").eq("id", org_id).single().execute()
            if org.data:
                return org.data
    
    # Fallback to default organization
    org = client.table("organizations").select("*").eq("slug", "sjit").single().execute()
    if not org.data:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    return org.data


@router.get("/by-slug/{slug}", response_model=OrganizationResponse)
async def get_organization_by_slug(slug: str):
    """
    Get organization by slug (used for subdomain routing).
    This is a public endpoint for initial app load.
    """
    client = get_supabase_admin_client()
    
    org = client.table("organizations").select(
        "id, name, slug, logo_url, primary_color, secondary_color, settings"
    ).eq("slug", slug).single().execute()
    
    if not org.data:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    return org.data
