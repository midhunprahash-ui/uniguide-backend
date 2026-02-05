from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.auth import require_auth, require_valid_org_id
from services.supabase_client import get_supabase_admin_client

router = APIRouter()


class EnsureOrgRequest(BaseModel):
    org_id: str
    full_name: str | None = None


@router.post("/ensure-org")
async def ensure_org(request: EnsureOrgRequest, current_user: dict = Depends(require_auth)):
    """
    Ensure the authenticated user is attached to the requested org.

    - If profile has no org_id, set it to the requested org.
    - If profile org_id differs, reject.
    """
    org_id = require_valid_org_id(request.org_id)
    user_id = current_user.get("id")

    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    client = get_supabase_admin_client()
    profile = client.table("profiles").select("id, org_id, full_name").eq("id", user_id).single().execute()

    if not profile.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile_org_id = profile.data.get("org_id")

    if profile_org_id and profile_org_id != org_id:
        # Allow a first-time org assignment if the user has no chat sessions yet.
        sessions = client.table("chat_sessions").select("id", count="exact").eq("user_id", user_id).limit(1).execute()
        if (sessions.count or 0) == 0:
            updates["org_id"] = org_id
        else:
            raise HTTPException(status_code=403, detail="User does not belong to this organization")

    updates = {}
    if not profile_org_id:
        updates["org_id"] = org_id
    if request.full_name and not profile.data.get("full_name"):
        updates["full_name"] = request.full_name

    if updates:
        client.table("profiles").update(updates).eq("id", user_id).execute()

    return {"org_id": org_id}
