"""
Category routes for dynamic category management.
Categories can be created, updated, and published by admins.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.supabase_auth import require_admin, get_current_user
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()


class CategoryCreate(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    icon: str = "BookOpen"
    color: str = "blue"
    is_published: bool = False
    is_admin_only: bool = False
    sort_order: int = 0


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    is_published: Optional[bool] = None
    is_admin_only: Optional[bool] = None
    sort_order: Optional[int] = None


class CategoryResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str]
    icon: str
    color: str
    is_published: bool
    is_admin_only: bool
    sort_order: int


@router.get("", response_model=list[CategoryResponse])
async def list_categories(
    include_unpublished: bool = False,
    user: dict = Depends(get_current_user)
):
    """
    List categories for the current organization.
    
    Students only see published categories.
    Admins can see all categories if include_unpublished=true.
    """
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
    
    # Build query
    query = client.table("categories").select("*").eq("org_id", org_id)
    
    # Filter by published status
    if not is_admin or not include_unpublished:
        query = query.eq("is_published", True)
        if not is_admin:
            query = query.eq("is_admin_only", False)
    
    query = query.order("sort_order")
    result = query.execute()
    
    return result.data


@router.post("", response_model=CategoryResponse)
async def create_category(
    category: CategoryCreate,
    admin: dict = Depends(require_admin)
):
    """Create a new category (admin only)."""
    client = get_supabase_admin_client()
    
    # Get admin's organization
    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")
    
    if not org_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Check if slug already exists
    existing = client.table("categories").select("id").eq("org_id", org_id).eq("slug", category.slug).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Category with this slug already exists")
    
    # Insert category
    result = client.table("categories").insert({
        "org_id": org_id,
        "name": category.name,
        "slug": category.slug,
        "description": category.description,
        "icon": category.icon,
        "color": category.color,
        "is_published": category.is_published,
        "is_admin_only": category.is_admin_only,
        "sort_order": category.sort_order
    }).execute()
    
    logger.info(f"Created category: {category.name} (slug: {category.slug})")
    return result.data[0]


@router.put("/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: str,
    category: CategoryUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a category (admin only)."""
    client = get_supabase_admin_client()
    
    # Verify category exists and belongs to admin's org
    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")
    
    existing = client.table("categories").select("*").eq("id", category_id).eq("org_id", org_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Build update data
    update_data = {}
    if category.name is not None:
        update_data["name"] = category.name
    if category.description is not None:
        update_data["description"] = category.description
    if category.icon is not None:
        update_data["icon"] = category.icon
    if category.color is not None:
        update_data["color"] = category.color
    if category.is_published is not None:
        update_data["is_published"] = category.is_published
    if category.is_admin_only is not None:
        update_data["is_admin_only"] = category.is_admin_only
    if category.sort_order is not None:
        update_data["sort_order"] = category.sort_order
    
    if not update_data:
        return existing.data
    
    result = client.table("categories").update(update_data).eq("id", category_id).execute()
    logger.info(f"Updated category: {category_id}")
    return result.data[0]


@router.delete("/{category_id}")
async def delete_category(
    category_id: str,
    admin: dict = Depends(require_admin)
):
    """Delete a category (admin only). Fails if category has documents."""
    client = get_supabase_admin_client()
    
    # Verify category exists and belongs to admin's org
    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")
    
    existing = client.table("categories").select("*").eq("id", category_id).eq("org_id", org_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Check if category has documents
    docs = client.table("documents").select("id", count="exact").eq("category_id", category_id).execute()
    if docs.count and docs.count > 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete category with {docs.count} documents. Move or delete documents first."
        )
    
    client.table("categories").delete().eq("id", category_id).execute()
    logger.info(f"Deleted category: {category_id}")
    return {"message": "Category deleted successfully"}


@router.post("/{category_id}/publish")
async def publish_category(
    category_id: str,
    admin: dict = Depends(require_admin)
):
    """Publish a category to make it visible to students."""
    client = get_supabase_admin_client()
    
    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")
    
    result = client.table("categories").update({"is_published": True}).eq("id", category_id).eq("org_id", org_id).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Category not found")
    
    logger.info(f"Published category: {category_id}")
    return {"message": "Category published successfully"}


@router.post("/{category_id}/unpublish")
async def unpublish_category(
    category_id: str,
    admin: dict = Depends(require_admin)
):
    """Unpublish a category to hide it from students."""
    client = get_supabase_admin_client()
    
    profile = client.table("profiles").select("org_id").eq("id", admin["id"]).single().execute()
    org_id = profile.data.get("org_id")
    
    result = client.table("categories").update({"is_published": False}).eq("id", category_id).eq("org_id", org_id).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Category not found")
    
    logger.info(f"Unpublished category: {category_id}")
    return {"message": "Category unpublished successfully"}
