"""
Admin routes for document management.
Updated to use Supabase for document storage and auth.
"""
import logging
import os
import shutil
import uuid as uuid_module

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import get_settings
from models.schemas import (
    VALID_CATEGORIES,
    AdminLogin,
    AdminStats,
    CategoryStats,
    DocumentMetadata,
    DocumentsByCategory,
    RenameDocumentRequest,
    Token,
    UpdateDocumentRequest,
)
from services.admin_stats import refresh_document_stats
from services.upload_pipeline import process_upload_payload
from services.upload_queue import enqueue_upload, get_upload_status
from services import supabase_storage
from services.document_processor import document_processor
from services.supabase_auth import require_admin
from services.supabase_client import get_supabase_admin_client
from services.vector_store import vector_store
from services.rate_limiter import limiter, RATE_LIMITS, get_org_key, get_ip_key
from services.storage_paths import (
    build_object_key,
    build_storage_path,
    sanitize_filename,
    split_storage_path,
    validate_storage_path,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def is_valid_uuid(val: str | None) -> bool:
    """Check if a string is a valid UUID."""
    if not val:
        return False
    try:
        uuid_module.UUID(val)
        return True
    except (ValueError, TypeError):
        return False


class UploadUrlRequest(BaseModel):
    filename: str
    category: str
    content_type: str | None = None
    file_size_bytes: int | None = None


def _extract_file_ext(filename: str) -> str:
    return filename.split(".")[-1].lower() if "." in filename else ""


def get_document_registry() -> dict:
    """
    Get document registry from Supabase database.
    This replaces the in-memory document_registry.
    """
    client = get_supabase_admin_client()
    result = client.table("documents").select("*").execute()

    registry = {}
    for doc in result.data:
        # Count chunks for this document
        chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()
        chunk_count = chunk_result.count or 0

        registry[doc["id"]] = {
            "id": doc["id"],
            "filename": doc["filename"],
            "stream": doc.get("stream") or "all",
            "year": doc["year"],
            "department": doc["department"],
            "category": doc["category"],
            "upload_date": doc["created_at"],
            "chunk_count": chunk_count,
            "one_line_summary": doc.get("one_line_summary"),
            "brief_summary": doc.get("brief_summary")
        }

    return registry


@router.post("/login", response_model=Token)
@limiter.limit(RATE_LIMITS["public"], key_func=get_ip_key)
async def admin_login(request: Request, credentials: AdminLogin):
    """
    Admin login endpoint.

    For Supabase Auth, use the Supabase client directly.
    This endpoint is kept for backwards compatibility but redirects to Supabase.

    Args:
        credentials: AdminLogin with username (email) and password

    Returns:
        JWT access token from Supabase Auth
    """
    from services.supabase_client import get_supabase_client
    
    # Use anon client for login to avoid polluting admin client state
    auth_client = get_supabase_client()
    admin_client = get_supabase_admin_client()

    try:
        # Use email as username for Supabase Auth
        response = auth_client.auth.sign_in_with_password({
            "email": credentials.username,
            "password": credentials.password
        })

        if not response.session:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if user is admin
        user_id = response.user.id
        # Use admin client to verify role (bypasses RLS)
        profile = admin_client.table("profiles").select("role").eq("id", user_id).single().execute()

        if not profile.data or profile.data["role"] not in ["admin", "superadmin"]:
            raise HTTPException(status_code=403, detail="Admin access required")

        return Token(
            access_token=response.session.access_token,
            token_type="bearer"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/validate-token")
@limiter.limit(RATE_LIMITS["admin"], key_func=get_org_key)
async def validate_token(request: Request, admin: dict = Depends(require_admin)):
    """
    Validate the current token and return user info.
    Used by frontend to verify authentication state.
    
    Returns:
        valid: True if token is valid
        user: User information (id, email, role)
    """
    return {
        "valid": True,
        "user": {
            "id": admin.get("id"),
            "email": admin.get("email"),
            "role": admin.get("role", "admin")
        }
    }


@router.post("/upload-url")
@limiter.limit(RATE_LIMITS["admin"], key_func=get_org_key)
async def create_upload_url(
    request: Request,
    payload: UploadUrlRequest,
    admin: dict = Depends(require_admin)
):
    """
    Create a signed upload URL for direct-to-storage uploads.
    """
    org_id = admin.get("org_id")
    if not org_id:
        raise HTTPException(status_code=400, detail="User has no organization")
    if payload.category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category: {payload.category}")

    bucket = supabase_storage.get_bucket_for_category(payload.category)
    path = build_object_key(
        org_id=org_id,
        filename=sanitize_filename(payload.filename),
        namespace=f"documents/{payload.category}",
        user_id=admin.get("id"),
    )

    client = get_supabase_admin_client()
    try:
        signed = client.storage.from_(bucket).create_signed_upload_url(path)
    except Exception as e:
        logger.error(f"Failed to create signed upload URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to create upload URL")

    signed_url = None
    token = None
    if isinstance(signed, dict):
        signed_url = signed.get("signed_url") or signed.get("signedUrl") or signed.get("signedURL")
        token = signed.get("token")
    else:
        signed_url = getattr(signed, "signed_url", None) or getattr(signed, "signedUrl", None)
        token = getattr(signed, "token", None)

    if not signed_url:
        raise HTTPException(status_code=500, detail="Signed upload URL not returned by storage")

    return {
        "bucket": bucket,
        "path": path,
        "storage_path": build_storage_path(bucket, path),
        "signed_url": signed_url,
        "token": token,
    }



@router.post("/upload")
@limiter.limit(RATE_LIMITS["upload"], key_func=get_org_key)
async def upload_document(
    request: Request,
    file: UploadFile | None = File(None),
    year: str = Form(...),
    department: str = Form(...),
    category: str = Form(...),
    stream: str = Form("all"),
    semester: str = Form("all"),
    summary: str | None = Form(None),
    storage_path: str | None = Form(None),
    original_filename: str | None = Form(None),
    file_size_bytes: int | None = Form(None),
    mime_type: str | None = Form(None),
    process_async: bool = Form(False),
    # FK-based IDs (optional, will be looked up from codes if not provided)
    stream_id: str | None = Form(None),
    department_id: str | None = Form(None),
    year_id: str | None = Form(None),
    admin: dict = Depends(require_admin)
):
    """
    Upload and process a document.

    Args:
        file: Document file (PDF, image, or text)
        year: Year filter (e.g., "1", "2", "3", "4", "all" or comma-separated like "1,2,3")
        department: Department filter (e.g., "CSE", "ECE", "all" or comma-separated)
        category: Category (rules, admissions, schedules, abhs, circulars)
        stream: Stream filter (e.g., "UG", "PG", "all" or comma-separated)
        semester: Semester filter (e.g., "1", "2", "all" or comma-separated)
        summary: Optional summary for circulars

    Returns:
        Document metadata
    """
    # Validate FK IDs - if not valid UUIDs, set to None to trigger code-based resolution
    if not is_valid_uuid(stream_id):
        stream_id = None
    if not is_valid_uuid(department_id):
        department_id = None
    if not is_valid_uuid(year_id):
        year_id = None

    
    if not file and not storage_path:
        raise HTTPException(status_code=400, detail="file or storage_path is required")

    filename = original_filename or (file.filename if file else None)
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    # Validate file type
    allowed_extensions = ['pdf', 'png', 'jpg', 'jpeg', 'txt']
    file_ext = _extract_file_ext(filename)

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Validate category against database
    from services.supabase_client import get_supabase_admin_client
    db_client = get_supabase_admin_client()
    valid_cats = db_client.table("categories").select("slug").execute()
    valid_category_slugs = [c["slug"] for c in valid_cats.data] if valid_cats.data else []

    if category not in valid_category_slugs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed: {', '.join(valid_category_slugs)}"
        )

    # Check for duplicate filename in active documents for this org
    org_id = admin.get("org_id")
    if not org_id:
        raise HTTPException(status_code=400, detail="User has no organization")
    expected_bucket = supabase_storage.get_bucket_for_category(category)

    # For direct uploads, accept only storage paths in the caller's scoped namespace.
    if storage_path:
        in_user_scope = validate_storage_path(
            storage_path=storage_path,
            org_id=org_id,
            expected_bucket=expected_bucket,
            expected_user_id=admin.get("id"),
        )
        in_org_scope = validate_storage_path(
            storage_path=storage_path,
            org_id=org_id,
            expected_bucket=expected_bucket,
        )
        if not in_user_scope and not in_org_scope:
            raise HTTPException(
                status_code=400,
                detail="Invalid storage_path. Upload path does not match your organization scope.",
            )
        if in_org_scope and not in_user_scope:
            logger.warning("Accepting legacy org-scoped storage path for upload: %s", storage_path)

    try:
        existing_doc = db_client.table("documents").select("id, filename").eq("filename", filename).eq("org_id", org_id).execute()
        if existing_doc.data and len(existing_doc.data) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Document '{filename}' already exists. Delete it first or rename your file."
            )
    except HTTPException:
        raise
    except Exception as e:
        # Log but don't block upload for query errors
        logger.warning(f"Could not check for duplicate filename: {e}")

    # Note: Stream is now derived from selected departments by the frontend
    # The stream parameter reflects which streams the selected departments belong to
    # No additional validation needed here as frontend handles the logic

    # Save file temporarily for processing (if file uploaded via backend)
    os.makedirs(settings.upload_directory, exist_ok=True)
    file_path = os.path.join(settings.upload_directory, filename) if file else None

    try:
        if file and file_path:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Decide async vs sync
        threshold_mb = int(os.getenv("UPLOAD_ASYNC_THRESHOLD_MB", "10"))
        threshold_bytes = threshold_mb * 1024 * 1024
        payload_size = file_size_bytes
        if payload_size is None and file_path and os.path.exists(file_path):
            payload_size = os.path.getsize(file_path)
        resolved_mime_type = mime_type or (file.content_type if file else None)

        should_async = process_async or (payload_size is not None and payload_size >= threshold_bytes)

        payload = {
            "org_id": admin.get("org_id"),
            "uploaded_by": admin.get("id"),
            "file_path": file_path,
            "storage_path": storage_path,
            "original_filename": filename,
            "file_ext": file_ext,
            "file_size_bytes": payload_size,
            "mime_type": resolved_mime_type,
            "year": year,
            "department": department,
            "category": category,
            "stream": stream,
            "semester": semester,
            "stream_id": stream_id,
            "department_id": department_id,
            "year_id": year_id,
        }

        if should_async:
            # Ensure file is in storage for async processing
            if file_path and not storage_path:
                storage_path = supabase_storage.upload_file(
                    file_path=file_path,
                    category=category,
                    filename=filename,
                    org_id=org_id,
                    uploaded_by=admin.get("id"),
                    namespace=f"documents/{category}",
                )
                payload["storage_path"] = storage_path

            job_id = enqueue_upload(payload)
            if not job_id:
                # Queue not configured, fall back to sync
                result = process_upload_payload(payload)
                return result

            if file_path and os.path.exists(file_path):
                os.remove(file_path)

            return JSONResponse(
                status_code=202,
                content={
                    "status": "queued",
                    "job_id": job_id,
                    "filename": filename,
                    "storage_path": storage_path,
                },
            )

        result = process_upload_payload(payload)
        return result

    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Error processing document: {e}")
        detail = str(e)
        if "No text could be extracted" in detail or "Poppler is not installed" in detail:
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=500, detail=f"Error processing document: {detail}")


@router.get("/upload-status/{job_id}")
@limiter.limit(RATE_LIMITS["admin"], key_func=get_org_key)
async def upload_status(
    request: Request,
    job_id: str,
    admin: dict = Depends(require_admin)
):
    """
    Get upload job status for async processing.
    """
    status = get_upload_status(job_id, admin.get("org_id"))
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/documents", response_model=list[DocumentMetadata])
async def list_documents(
    category: str | None = None,
    include_deleted: bool = False,
    admin: dict = Depends(require_admin)
):
    """
    List all uploaded documents, optionally filtered by category.
    Excludes soft-deleted documents by default.
    """
    client = get_supabase_admin_client()

    org_id = admin.get("org_id")
    query = client.table("documents").select("*").eq("org_id", org_id)
    if category:
        query = query.eq("category", category)

    # Exclude soft-deleted documents unless explicitly requested
    if not include_deleted:
        query = query.is_("deleted_at", "null")

    result = query.order("created_at", desc=True).execute()

    # Format for response
    docs = []
    for doc in result.data:
        # Get chunk count
        chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()

        docs.append({
            "id": doc["id"],
            "filename": doc["filename"],
            "stream": doc.get("stream") or "all",
            "year": doc["year"],
            "department": doc["department"],
            "category": doc["category"],
            "upload_date": doc["created_at"],
            "chunk_count": chunk_result.count or 0
        })

    return docs


@router.get("/documents/by-category", response_model=DocumentsByCategory)
async def get_documents_by_category(admin: dict = Depends(require_admin)):
    """
    Get all documents grouped by category.
    Uses direct table query to ensure fresh data.
    """
    client = get_supabase_admin_client()
    admin_org_id = admin.get("org_id")

    if not admin_org_id:
        raise HTTPException(status_code=400, detail="User has no organization")

    # Direct query instead of RPC to ensure fresh data
    result = client.table("documents").select("*").eq("org_id", admin_org_id).is_("deleted_at", "null").order("created_at", desc=True).execute()
    
    # Get all categories
    cats_result = client.table("categories").select("slug").execute()
    all_category_slugs = [c["slug"] for c in cats_result.data] if cats_result.data else []
    
    # Group documents by category
    docs_by_cat: dict[str, list] = {slug: [] for slug in all_category_slugs}
    
    for doc in result.data or []:
        cat = doc.get("category", "general")
        if cat not in docs_by_cat:
            docs_by_cat[cat] = []
        
        # Get chunk count for this document
        chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()
        
        docs_by_cat[cat].append({
            "id": doc["id"],
            "filename": doc["filename"],
            "stream": doc.get("stream") or "all",
            "year": doc.get("year") or "all",
            "department": doc.get("department") or "all",
            "category": cat,
            "upload_date": doc["created_at"],
            "chunk_count": chunk_result.count or 0
        })
    
    categories_data = []
    total_documents = 0
    
    for cat_slug in all_category_slugs:
        cat_docs = docs_by_cat.get(cat_slug, [])
        total_documents += len(cat_docs)
        categories_data.append(CategoryStats(
            category=cat_slug,
            count=len(cat_docs),
            documents=cat_docs
        ))
    
    return DocumentsByCategory(
        categories=categories_data,
        total_documents=total_documents
    )


@router.get("/documents/category/{category}", response_model=list[DocumentMetadata])
async def get_documents_for_category(
    category: str,
    admin: dict = Depends(require_admin)
):
    """
    Get all documents for a specific category.
    """
    client = get_supabase_admin_client()
    
    # Validate category against database
    valid_cats = client.table("categories").select("slug").execute()
    valid_category_slugs = [c["slug"] for c in valid_cats.data] if valid_cats.data else []
    
    if category not in valid_category_slugs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed: {', '.join(valid_category_slugs)}"
        )

    org_id = admin.get("org_id")
    result = client.table("documents").select("*").eq("org_id", org_id).eq("category", category).order("created_at", desc=True).execute()

    docs = []
    for doc in result.data:
        chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()

        docs.append({
            "id": doc["id"],
            "filename": doc["filename"],
            "stream": doc.get("stream") or "all",
            "year": doc["year"],
            "department": doc["department"],
            "category": doc["category"],
            "upload_date": doc["created_at"],
            "chunk_count": chunk_result.count or 0
        })

    return docs


@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(admin: dict = Depends(require_admin)):
    """
    Get admin dashboard statistics.
    OPTIMIZED: Uses single RPC call instead of 8-12 separate queries.
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    if not org_id:
        raise HTTPException(status_code=400, detail="User has no organization")

    # Single RPC call replaces 8-12 queries + file system operations
    result = client.rpc("get_admin_stats", {"p_org_id": org_id}).execute()

    if result.data:
        stats = result.data
        total_size_bytes = stats.get("total_storage_bytes", 0) or 0
        total_size_mb = total_size_bytes / (1024 * 1024)

        return AdminStats(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            documents_by_category=stats.get("by_category", {}),
            total_size_mb=round(total_size_mb, 2)
        )

    return AdminStats(
        total_documents=0,
        total_chunks=0,
        documents_by_category={},
        total_size_mb=0.0
    )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    hard_delete: bool = False,
    admin: dict = Depends(require_admin)
):
    """
    Delete a document.
    - Default: Archives to deleted_documents table (can be restored)
    - hard_delete=True: Permanently deletes including storage file (superadmin only)

    Args:
        doc_id: Document ID to delete
        hard_delete: If True, permanently deletes the document (requires superadmin)
    """
    from services import supabase_storage
    from datetime import datetime

    client = get_supabase_admin_client()
    org_id = admin.get("org_id")
    user_id = admin.get("id")

    # Get document info - verify it belongs to this org
    doc_result = client.table("documents").select("*").eq("id", doc_id).eq("org_id", org_id).single().execute()

    if not doc_result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_info = doc_result.data

    try:
        if hard_delete:
            # Hard delete requires superadmin role
            if admin.get("role") != "superadmin":
                raise HTTPException(
                    status_code=403,
                    detail="Only superadmins can permanently delete documents"
                )

            # Delete from Supabase Storage if storage_path exists
            storage_path = doc_info.get("storage_path")
            if storage_path:
                supabase_storage.delete_file(storage_path)

            # Also remove from deleted_documents archive if exists
            client.table("deleted_documents").delete().eq("id", doc_id).execute()

            # Delete from Supabase database (chunks cascade automatically)
            client.table("documents").delete().eq("id", doc_id).execute()

            # Delete local file if exists (legacy cleanup)
            file_path = os.path.join(settings.upload_directory, doc_info["filename"])
            if os.path.exists(file_path):
                os.remove(file_path)

            # Refresh materialized view for updated stats
            refresh_document_stats()

            return {
                "message": "Document permanently deleted",
                "filename": doc_info["filename"],
                "hard_delete": True
            }
        else:
            # Archive delete - move to deleted_documents table
            # 1. Insert into deleted_documents archive
            archive_data = {
                "id": doc_info["id"],
                "filename": doc_info["filename"],
                "original_filename": doc_info.get("original_filename"),
                "file_path": doc_info.get("file_path"),
                "storage_path": doc_info.get("storage_path"),
                "file_size_bytes": doc_info.get("file_size_bytes"),
                "mime_type": doc_info.get("mime_type"),
                "year": doc_info.get("year"),
                "department": doc_info.get("department"),
                "category": doc_info.get("category"),
                "stream": doc_info.get("stream"),
                "semester": doc_info.get("semester"),
                "one_line_summary": doc_info.get("one_line_summary"),
                "brief_summary": doc_info.get("brief_summary"),
                "org_id": doc_info["org_id"],
                "stream_id": doc_info.get("stream_id"),
                "department_id": doc_info.get("department_id"),
                "year_id": doc_info.get("year_id"),
                "uploaded_by": doc_info.get("uploaded_by"),
                "created_at": doc_info.get("created_at"),
                "updated_at": doc_info.get("updated_at"),
                "deleted_at": datetime.now().isoformat(),
                "deleted_by": user_id
            }
            client.table("deleted_documents").insert(archive_data).execute()

            # 2. Delete from documents table (chunks cascade automatically)
            client.table("documents").delete().eq("id", doc_id).execute()

            # Refresh materialized view for updated stats
            refresh_document_stats()

            return {
                "message": "Document archived (can be restored)",
                "filename": doc_info["filename"],
                "hard_delete": False
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.post("/documents/{doc_id}/restore")
async def restore_document(doc_id: str, admin: dict = Depends(require_admin)):
    """
    Restore a document from the deleted_documents archive.
    Moves the document back to the documents table and re-processes to regenerate chunks.
    """
    from services.document_processor import document_processor
    from services import supabase_storage

    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Get document from archive
    archive_result = client.table("deleted_documents").select("*").eq("id", doc_id).eq("org_id", org_id).single().execute()

    if not archive_result.data:
        raise HTTPException(status_code=404, detail="Archived document not found")

    archived_doc = archive_result.data

    try:
        # Check if a document with the same filename already exists
        existing = client.table("documents").select("id").eq("filename", archived_doc["filename"]).eq("org_id", org_id).maybe_single().execute()
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail=f"A document named '{archived_doc['filename']}' already exists. Delete it first before restoring."
            )

        # 1. Insert back into documents table (without chunks - they'll be regenerated)
        doc_data = {
            "id": archived_doc["id"],
            "filename": archived_doc["filename"],
            "original_filename": archived_doc.get("original_filename"),
            "file_path": archived_doc.get("file_path"),
            "storage_path": archived_doc.get("storage_path"),
            "file_size_bytes": archived_doc.get("file_size_bytes"),
            "mime_type": archived_doc.get("mime_type"),
            "year": archived_doc.get("year"),
            "department": archived_doc.get("department"),
            "category": archived_doc.get("category"),
            "stream": archived_doc.get("stream"),
            "semester": archived_doc.get("semester"),
            "one_line_summary": archived_doc.get("one_line_summary"),
            "brief_summary": archived_doc.get("brief_summary"),
            "org_id": archived_doc["org_id"],
            "stream_id": archived_doc.get("stream_id"),
            "department_id": archived_doc.get("department_id"),
            "year_id": archived_doc.get("year_id"),
            "uploaded_by": archived_doc.get("uploaded_by"),
            "created_at": archived_doc.get("created_at")
        }
        client.table("documents").insert(doc_data).execute()

        # 2. Remove from archive
        client.table("deleted_documents").delete().eq("id", doc_id).execute()

        # 3. If storage_path exists, download and re-process to regenerate chunks
        storage_path = archived_doc.get("storage_path")
        if storage_path:
            # Download file from storage for re-processing
            local_path = os.path.join(settings.upload_directory, archived_doc["filename"])
            file_bytes = supabase_storage.download_file(storage_path)
            with open(local_path, "wb") as f:
                f.write(file_bytes)

            # Re-process to generate chunks and embeddings
            file_ext = archived_doc["filename"].split(".")[-1].lower()
            result = document_processor.process_and_store_document(
                file_path=local_path,
                filename=archived_doc["filename"],
                year=archived_doc.get("year", "all"),
                department=archived_doc.get("department", "all"),
                category=archived_doc.get("category", "general"),
                file_type=file_ext,
                org_id=archived_doc["org_id"],
                stream=archived_doc.get("stream", "all"),
                semester=archived_doc.get("semester", "all")
            )

            # Cleanup local file
            if os.path.exists(local_path):
                os.remove(local_path)

        # Refresh materialized view for updated stats
        refresh_document_stats()

        return {
            "message": "Document restored successfully",
            "filename": archived_doc["filename"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring document: {e}")
        raise HTTPException(status_code=500, detail=f"Error restoring document: {str(e)}")


@router.get("/documents/deleted")
async def list_deleted_documents(admin: dict = Depends(require_admin)):
    """
    List all archived (deleted) documents for recovery.
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    result = client.table("deleted_documents").select(
        "id, filename, category, year, department, deleted_at, deleted_by"
    ).eq("org_id", org_id).order("deleted_at", desc=True).execute()

    # Get deleter names
    docs = []
    for doc in result.data:
        deleter_name = None
        if doc.get("deleted_by"):
            profile = client.table("profiles").select("full_name").eq("id", doc["deleted_by"]).maybe_single().execute()
            deleter_name = profile.data.get("full_name") if profile.data else None

        docs.append({
            "id": doc["id"],
            "filename": doc["filename"],
            "category": doc["category"],
            "year": doc["year"],
            "department": doc["department"],
            "deleted_at": doc["deleted_at"],
            "deleted_by_name": deleter_name
        })

    return docs


@router.put("/documents/{doc_id}/rename")
async def rename_document(
    doc_id: str,
    request: RenameDocumentRequest,
    admin: dict = Depends(require_admin)
):
    """
    Rename a document file (both in Storage and DB).
    """
    from services import supabase_storage

    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Verify document belongs to this org
    doc_result = client.table("documents").select("*").eq("id", doc_id).eq("org_id", org_id).single().execute()

    if not doc_result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    old_filename = doc_result.data["filename"]
    storage_path = doc_result.data.get("storage_path")
    new_filename = request.new_filename

    if not new_filename or new_filename.strip() == "":
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Keep original extension if not provided
    old_ext = old_filename.split('.')[-1].lower()
    if '.' not in new_filename:
        new_filename = f"{new_filename}.{old_ext}"

    try:
        new_storage_path = storage_path
        storage_renamed = False

        # Try to rename in Supabase Storage if storage_path exists
        if storage_path:
            try:
                new_storage_path = supabase_storage.rename_file(storage_path, new_filename)
                storage_renamed = True
                logger.info(f"Storage file renamed: {storage_path} -> {new_storage_path}")
            except Exception as storage_err:
                # Storage rename failed (file might not exist) - continue with DB update only
                logger.warning(f"Storage rename failed (continuing with DB update): {storage_err}")
                # Update storage_path to new filename anyway (keep existing prefix when possible)
                try:
                    bucket, object_key = split_storage_path(storage_path)
                    object_dir = object_key.rsplit("/", 1)[0] if "/" in object_key else ""
                    target_key = f"{object_dir}/{new_filename}" if object_dir else new_filename
                    new_storage_path = build_storage_path(bucket, target_key)
                except ValueError:
                    new_storage_path = storage_path

        # Update in Supabase DB
        client.table("documents").update({
            "filename": new_filename,
            "original_filename": new_filename,
            "file_path": f"uploads/{new_filename}",
            "storage_path": new_storage_path
        }).eq("id", doc_id).execute()

        # Verify the update actually happened by fetching the updated record
        verify_result = client.table("documents").select("id, filename").eq("id", doc_id).single().execute()
        if not verify_result.data or verify_result.data.get("filename") != new_filename:
            logger.error(f"Document rename update verification failed for doc_id: {doc_id}")
            raise HTTPException(status_code=500, detail="Failed to update document in database")

        logger.info(f"Document renamed successfully: {old_filename} -> {new_filename}")

        # Also rename local file if exists (legacy)
        old_file_path = os.path.join(settings.upload_directory, old_filename)
        new_file_path = os.path.join(settings.upload_directory, new_filename)
        if os.path.exists(old_file_path):
            try:
                os.rename(old_file_path, new_file_path)
            except Exception as local_err:
                logger.warning(f"Local file rename failed: {local_err}")

        return {
            "message": "Document renamed successfully",
            "old_filename": old_filename,
            "new_filename": new_filename,
            "storage_renamed": storage_renamed
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming document: {e}")
        raise HTTPException(status_code=500, detail=f"Error renaming document: {str(e)}")


@router.put("/documents/{doc_id}/metadata")
async def update_document_metadata(
    doc_id: str,
    request: UpdateDocumentRequest,
    admin: dict = Depends(require_admin)
):
    """
    Update document metadata (year, department, category, stream, semester).
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Verify document exists and belongs to this org
    doc_result = client.table("documents").select("id").eq("id", doc_id).eq("org_id", org_id).single().execute()

    if not doc_result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate category against database if provided
    if request.category:
        valid_cats = client.table("categories").select("slug").execute()
        valid_category_slugs = [c["slug"] for c in valid_cats.data] if valid_cats.data else []
        if request.category not in valid_category_slugs:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Allowed: {', '.join(valid_category_slugs)}"
            )

    # Validate years - can be single value or comma-separated list
    if request.year:
        # Get valid year codes from database
        year_records = client.table("years").select("code").eq("org_id", org_id).execute()
        valid_year_codes = {y["code"] for y in year_records.data} if year_records.data else set()
        valid_year_codes.add("all")  # Always allow "all"

        # Split by comma and validate each year
        years_list = [y.strip() for y in request.year.split(',')]
        invalid_years = [y for y in years_list if y not in valid_year_codes]
        if invalid_years:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid year(s): {', '.join(invalid_years)}. Valid codes: {', '.join(sorted(valid_year_codes))}"
            )

    # Validate streams - can be single value or comma-separated list
    if request.stream:
        stream_records = client.table("streams").select("code").eq("org_id", org_id).execute()
        valid_stream_codes = {s["code"] for s in stream_records.data} if stream_records.data else set()
        valid_stream_codes.add("all")  # Always allow "all"

        streams_list = [s.strip() for s in request.stream.split(',')]
        invalid_streams = [s for s in streams_list if s not in valid_stream_codes]
        if invalid_streams:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stream(s): {', '.join(invalid_streams)}. Valid codes: {', '.join(sorted(valid_stream_codes))}"
            )

    # Validate semesters - can be single value or comma-separated list (1-8 typically)
    if request.semester and request.semester != "all":
        semesters_list = [s.strip() for s in request.semester.split(',')]
        # Semesters are numeric values, validate they are valid numbers
        for sem in semesters_list:
            if sem != "all" and not sem.isdigit():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid semester: {sem}. Semesters must be numeric values or 'all'."
                )

    try:
        updates = {}
        if request.year is not None:
            updates["year"] = request.year
        if request.department is not None:
            updates["department"] = request.department
        if request.category is not None:
            updates["category"] = request.category
        if request.stream is not None:
            updates["stream"] = request.stream
        if request.semester is not None:
            updates["semester"] = request.semester

        if not updates:
            return {"message": "No updates provided"}

        # Resolve FK IDs from updated TEXT codes
        stream_val = request.stream if request.stream else None
        dept_val = request.department if request.department else None
        year_val = request.year if request.year else None
        
        # Resolve stream_id
        if stream_val and stream_val != 'all' and ',' not in stream_val:
            stream_result = client.table("streams").select("id").eq("code", stream_val).eq("org_id", org_id).maybe_single().execute()
            if stream_result.data:
                updates["stream_id"] = stream_result.data["id"]
        elif stream_val == 'all':
            updates["stream_id"] = None
            
        # Resolve department_id 
        if dept_val and dept_val != 'all' and ',' not in dept_val:
            dept_query = client.table("departments").select("id").eq("code", dept_val).eq("org_id", org_id)
            if "stream_id" in updates and updates["stream_id"]:
                dept_query = dept_query.eq("stream_id", updates["stream_id"])
            dept_result = dept_query.maybe_single().execute()
            if dept_result.data:
                updates["department_id"] = dept_result.data["id"]
        elif dept_val == 'all':
            updates["department_id"] = None
            
        # Resolve year_id
        if year_val and year_val != 'all' and ',' not in year_val:
            if "department_id" in updates and updates["department_id"]:
                year_result = client.table("years").select("id").eq("code", year_val).eq("department_id", updates["department_id"]).maybe_single().execute()
                if year_result.data:
                    updates["year_id"] = year_result.data["id"]
        elif year_val == 'all':
            updates["year_id"] = None

        client.table("documents").update(updates).eq("id", doc_id).execute()

        return {
            "message": "Document metadata updated successfully",
            "updates": updates
        }

    except Exception as e:
        logger.error(f"Error updating document metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating document metadata: {str(e)}")


@router.post("/sync")
async def sync_vector_store(admin: dict = Depends(require_admin)):
    """
    Sync vector store with uploads folder.
    """
    try:
        sync_result = vector_store.sync_with_uploads(settings.upload_directory)

        return {
            "message": "Sync completed",
            "files_removed": sync_result.get("removed", []),
            "files_kept": sync_result.get("kept", []),
            "current_chunk_count": vector_store.get_collection_count()
        }
    except Exception as e:
        logger.error(f"Error syncing vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error syncing vector store: {str(e)}")


@router.post("/refresh-registry")
async def refresh_registry(admin: dict = Depends(require_admin)):
    """
    Refresh document registry from database.
    """
    try:
        registry = get_document_registry()

        return {
            "message": "Registry refreshed successfully",
            "document_count": len(registry),
            "documents": list(registry.values())
        }
    except Exception as e:
        logger.error(f"Error refreshing registry: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing registry: {str(e)}")


@router.delete("/clear-all")
async def clear_vector_store(
    confirm: str = None,
    admin: dict = Depends(require_admin)
):
    """
    Clear all documents from the vector store.
    WARNING: This removes all vectors and cannot be undone.
    
    Requires confirm=YES_DELETE_ALL to proceed.
    """
    if confirm != "YES_DELETE_ALL":
        raise HTTPException(
            status_code=400,
            detail="This action will DELETE ALL documents permanently. Pass confirm=YES_DELETE_ALL to proceed."
        )
    
    try:
        # Also clear from Supabase Storage - ONLY for this org
        from services import supabase_storage
        client = get_supabase_admin_client()
        org_id = admin.get("org_id")

        # Get all documents with storage paths for this org
        docs = client.table("documents").select("id, storage_path").eq("org_id", org_id).execute()
        doc_ids = []
        for doc in docs.data:
            doc_ids.append(doc["id"])
            if doc.get("storage_path"):
                supabase_storage.delete_file(doc["storage_path"])

        # Delete documents and chunks for this org only
        if doc_ids:
            client.table("documents").delete().in_("id", doc_ids).execute()

        return {
            "message": "Vector store cleared successfully",
            "chunk_count": vector_store.get_collection_count()
        }
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")


@router.get("/health-check")
async def health_check(admin: dict = Depends(require_admin)):
    """
    Check data consistency between Supabase Storage and database.
    Reports orphaned files (in Storage but not DB) and missing files (in DB but not Storage).
    """
    from services import supabase_storage

    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Get all files from Storage
    storage_files = supabase_storage.list_all_files()
    storage_paths = {f["storage_path"] for f in storage_files}

    # Get all documents from DB for this org
    db_result = client.table("documents").select("id, filename, storage_path, status, category").eq("org_id", org_id).execute()
    db_docs = db_result.data or []
    db_storage_paths = {doc.get("storage_path") for doc in db_docs if doc.get("storage_path")}
    
    # Find inconsistencies
    orphaned_files = storage_paths - db_storage_paths  # In Storage but not in DB
    missing_files = db_storage_paths - storage_paths   # In DB but not in Storage
    
    # Get pending/failed uploads
    pending_docs = [doc for doc in db_docs if doc.get("status") == "pending"]
    failed_docs = [doc for doc in db_docs if doc.get("status") == "failed"]
    
    is_healthy = len(orphaned_files) == 0 and len(missing_files) == 0 and len(pending_docs) == 0
    
    return {
        "healthy": is_healthy,
        "summary": {
            "total_storage_files": len(storage_files),
            "total_db_documents": len(db_docs),
            "orphaned_files_count": len(orphaned_files),
            "missing_files_count": len(missing_files),
            "pending_uploads": len(pending_docs),
            "failed_uploads": len(failed_docs)
        },
        "orphaned_files": list(orphaned_files),  # Files in Storage with no DB record
        "missing_files": list(missing_files),     # DB records with no Storage file
        "pending_documents": [{"id": d["id"], "filename": d["filename"]} for d in pending_docs],
        "failed_documents": [{"id": d["id"], "filename": d["filename"]} for d in failed_docs]
    }
