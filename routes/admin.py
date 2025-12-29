"""
Admin routes for document management.
Updated to use Supabase for document storage and auth.
"""
import logging
import os
import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

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
from services.document_processor import document_processor
from services.supabase_auth import require_admin
from services.supabase_client import get_supabase_admin_client
from services.vector_store import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


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
async def admin_login(credentials: AdminLogin):
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
async def validate_token(admin: dict = Depends(require_admin)):
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



@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    year: str = Form(...),
    department: str = Form(...),
    category: str = Form(...),
    summary: str | None = Form(None),
    admin: dict = Depends(require_admin)
):
    """
    Upload and process a document.

    Args:
        file: Document file (PDF, image, or text)
        year: Year filter (e.g., "1", "2", "3", "4", "all")
        department: Department filter (e.g., "CSE", "ECE", "all")
        category: Category (rules, admissions, schedules, abhs, circulars)
        summary: Optional summary for circulars

    Returns:
        Document metadata
    """
    from services import supabase_storage
    
    # Validate file type
    allowed_extensions = ['pdf', 'png', 'jpg', 'jpeg', 'txt']
    file_ext = file.filename.split('.')[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Validate category
    valid_categories = ['rules', 'admissions', 'schedules', 'timetables', 'abhs', 'circulars']
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Allowed: {', '.join(valid_categories)}"
        )

    # Save file temporarily for processing
    os.makedirs(settings.upload_directory, exist_ok=True)
    file_path = os.path.join(settings.upload_directory, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process and store document (creates embeddings in DB)
        result = document_processor.process_and_store_document(
            file_path=file_path,
            filename=file.filename,
            year=year,
            department=department,
            category=category,
            file_type=file_ext
        )

        # Upload file to Supabase Storage for permanent storage
        storage_path = supabase_storage.upload_file(
            file_path=file_path,
            category=category,
            filename=file.filename
        )

        # Update document record with storage path
        client = get_supabase_admin_client()
        doc_result = client.table("documents").select("id").eq("filename", file.filename).single().execute()
        if doc_result.data:
            document_id = doc_result.data["id"]
            client.table("documents").update({
                "storage_path": storage_path
            }).eq("id", document_id).execute()
            result["storage_path"] = storage_path

        # If it's a circular, generate summary and register it
        if category == "circulars":
            from services.rag_engine import rag_engine

            extracted_text = result.get("extracted_text", "")
            summaries = rag_engine.generate_circular_summary(extracted_text, file.filename)

            if doc_result.data:
                # Update document with summaries
                client.table("documents").update({
                    "one_line_summary": summaries["one_line"],
                    "brief_summary": summaries["brief"]
                }).eq("id", document_id).execute()

                # Create circular entry
                client.table("circulars").insert({
                    "document_id": document_id,
                    "title": file.filename,
                    "one_line_summary": summaries["one_line"],
                    "brief_summary": summaries["brief"],
                    "is_active": True
                }).execute()

        # Clean up local file (it's now in Supabase Storage)
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/documents", response_model=list[DocumentMetadata])
async def list_documents(
    category: str | None = None,
    admin: dict = Depends(require_admin)
):
    """
    List all uploaded documents, optionally filtered by category.
    """
    client = get_supabase_admin_client()

    query = client.table("documents").select("*")
    if category:
        query = query.eq("category", category)

    result = query.order("created_at", desc=True).execute()

    # Format for response
    docs = []
    for doc in result.data:
        # Get chunk count
        chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()

        docs.append({
            "id": doc["id"],
            "filename": doc["filename"],
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
    """
    client = get_supabase_admin_client()

    categories_data = []
    total_documents = 0

    for category in VALID_CATEGORIES:
        result = client.table("documents").select("*").eq("category", category).order("created_at", desc=True).execute()

        category_docs = []
        for doc in result.data:
            chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()

            category_docs.append({
                "id": doc["id"],
                "filename": doc["filename"],
                "year": doc["year"],
                "department": doc["department"],
                "category": doc["category"],
                "upload_date": doc["created_at"],
                "chunk_count": chunk_result.count or 0
            })

        total_documents += len(category_docs)

        categories_data.append(CategoryStats(
            category=category,
            count=len(category_docs),
            documents=category_docs
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
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {', '.join(VALID_CATEGORIES)}"
        )

    client = get_supabase_admin_client()
    result = client.table("documents").select("*").eq("category", category).order("created_at", desc=True).execute()

    docs = []
    for doc in result.data:
        chunk_result = client.table("document_chunks").select("id", count="exact").eq("document_id", doc["id"]).execute()

        docs.append({
            "id": doc["id"],
            "filename": doc["filename"],
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
    """
    client = get_supabase_admin_client()

    # Count documents by category
    docs_by_category = {}
    for category in VALID_CATEGORIES:
        result = client.table("documents").select("id", count="exact").eq("category", category).execute()
        docs_by_category[category] = result.count or 0

    # Total documents
    total_docs_result = client.table("documents").select("id", count="exact").execute()

    # Total chunks
    total_chunks_result = client.table("document_chunks").select("id", count="exact").execute()

    # Calculate total storage size
    docs_result = client.table("documents").select("filename").execute()
    total_size = 0
    for doc in docs_result.data:
        file_path = os.path.join(settings.upload_directory, doc["filename"])
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)

    total_size_mb = total_size / (1024 * 1024)

    return AdminStats(
        total_documents=total_docs_result.count or 0,
        total_chunks=total_chunks_result.count or 0,
        documents_by_category=docs_by_category,
        total_size_mb=round(total_size_mb, 2)
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, admin: dict = Depends(require_admin)):
    """
    Delete a document and all its chunks.
    """
    from services import supabase_storage
    
    client = get_supabase_admin_client()

    # Get document info
    doc_result = client.table("documents").select("*").eq("id", doc_id).single().execute()

    if not doc_result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_info = doc_result.data

    try:
        # Delete from Supabase Storage if storage_path exists
        storage_path = doc_info.get("storage_path")
        if storage_path:
            supabase_storage.delete_file(storage_path)

        # Delete from Supabase database (chunks cascade automatically)
        client.table("documents").delete().eq("id", doc_id).execute()

        # Delete local file if exists (legacy cleanup)
        file_path = os.path.join(settings.upload_directory, doc_info["filename"])
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"message": "Document deleted successfully", "filename": doc_info["filename"]}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


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

    doc_result = client.table("documents").select("*").eq("id", doc_id).single().execute()

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
        
        # Rename in Supabase Storage if storage_path exists
        if storage_path:
            new_storage_path = supabase_storage.rename_file(storage_path, new_filename)

        # Update in Supabase DB
        client.table("documents").update({
            "filename": new_filename,
            "original_filename": new_filename,
            "file_path": f"uploads/{new_filename}",
            "storage_path": new_storage_path
        }).eq("id", doc_id).execute()

        # Also rename local file if exists (legacy)
        old_file_path = os.path.join(settings.upload_directory, old_filename)
        new_file_path = os.path.join(settings.upload_directory, new_filename)
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)

        return {
            "message": "Document renamed successfully",
            "old_filename": old_filename,
            "new_filename": new_filename
        }

    except Exception as e:
        # Rollback file rename
        if os.path.exists(new_file_path) and not os.path.exists(old_file_path):
            try:
                os.rename(new_file_path, old_file_path)
            except Exception:
                pass
        logger.error(f"Error renaming document: {e}")
        raise HTTPException(status_code=500, detail=f"Error renaming document: {str(e)}")


@router.put("/documents/{doc_id}/metadata")
async def update_document_metadata(
    doc_id: str,
    request: UpdateDocumentRequest,
    admin: dict = Depends(require_admin)
):
    """
    Update document metadata (year, department, category).
    """
    client = get_supabase_admin_client()

    # Verify document exists
    doc_result = client.table("documents").select("id").eq("id", doc_id).single().execute()

    if not doc_result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate
    if request.category and request.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {', '.join(VALID_CATEGORIES)}"
        )

    valid_years = ['1', '2', '3', '4', 'all']
    if request.year and request.year not in valid_years:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid year. Must be one of: {', '.join(valid_years)}"
        )

    try:
        updates = {}
        if request.year is not None:
            updates["year"] = request.year
        if request.department is not None:
            updates["department"] = request.department
        if request.category is not None:
            updates["category"] = request.category

        if not updates:
            return {"message": "No updates provided"}

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
        # Also clear from Supabase Storage
        from services import supabase_storage
        client = get_supabase_admin_client()
        
        # Get all documents with storage paths
        docs = client.table("documents").select("storage_path").execute()
        for doc in docs.data:
            if doc.get("storage_path"):
                supabase_storage.delete_file(doc["storage_path"])
        
        vector_store.clear_all()

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
    
    # Get all files from Storage
    storage_files = supabase_storage.list_all_files()
    storage_paths = {f["storage_path"] for f in storage_files}
    
    # Get all documents from DB
    db_result = client.table("documents").select("id, filename, storage_path, status, category").execute()
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

