"""
Notes Admin API routes for uploading and managing academic notes.
Part of Notes RAG subsystem, isolated from institutional documents.
"""
import logging
import os
import uuid as uuid_module
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.document_processor import document_processor
from services.notes_rag_engine import notes_rag_engine
from services.notes_vector_store import notes_vector_store
from services.rag_engine import rag_engine
from services.storage_paths import build_object_key, sanitize_filename
from services.supabase_auth import require_admin
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Notes storage bucket name
NOTES_BUCKET = "notes"


def cleanup_failed_note_upload(note_id: str | None, file_path: str | None) -> None:
    """Best-effort cleanup for partially created note uploads."""
    client = get_supabase_admin_client()

    if note_id:
        try:
            notes_vector_store.delete_note_chunks(note_id)
        except Exception as exc:
            logger.warning(f"Failed to cleanup note chunks for {note_id}: {exc}")

        try:
            client.table("notes").delete().eq("id", note_id).execute()
        except Exception as exc:
            logger.warning(f"Failed to cleanup note record {note_id}: {exc}")

    if file_path:
        delete_note_from_storage(file_path)


def upload_note_to_storage(file_path: str, file_content: bytes, content_type: str) -> bool:
    """Upload a note file to Supabase Storage."""
    client = get_supabase_admin_client()
    try:
        client.storage.from_(NOTES_BUCKET).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        logger.info(f"✅ Uploaded note to storage: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload note to storage: {e}")
        return False


def download_note_from_storage(file_path: str) -> bytes | None:
    """Download a note file from Supabase Storage."""
    client = get_supabase_admin_client()
    try:
        # Backward-compat: old records might include "notes/" prefix in object key.
        if file_path.startswith(f"{NOTES_BUCKET}/"):
            path = file_path[len(f"{NOTES_BUCKET}/"):]
        else:
            path = file_path
        result = client.storage.from_(NOTES_BUCKET).download(path)
        return result
    except Exception as e:
        logger.error(f"❌ Failed to download note from storage: {e}")
        return None


def delete_note_from_storage(file_path: str) -> bool:
    """Delete a note file from Supabase Storage."""
    client = get_supabase_admin_client()
    try:
        if file_path.startswith(f"{NOTES_BUCKET}/"):
            path = file_path[len(f"{NOTES_BUCKET}/"):]
        else:
            path = file_path
        client.storage.from_(NOTES_BUCKET).remove([path])
        logger.info(f"✅ Deleted note from storage: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete note from storage: {e}")
        return False


def extract_note_text(file_path: str, content_type: str | None) -> str:
    """Extract text from a note file based on its content type."""
    normalized_content_type = (content_type or "").lower()

    if normalized_content_type == "application/pdf":
        return document_processor.process_pdf(file_path)

    if normalized_content_type in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        return document_processor.process_image(file_path)

    if normalized_content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }:
        try:
            return document_processor.process_pdf(file_path)
        except Exception:
            return document_processor.process_text(file_path)

    return document_processor.process_text(file_path)


def build_note_embeddings(text: str) -> tuple[list[dict], list[list[float]]]:
    """Chunk note text and generate embeddings that match note_chunks.embedding."""
    enhanced_chunks = document_processor.chunk_text_enhanced(text)
    if not enhanced_chunks:
        raise HTTPException(status_code=500, detail="Failed to chunk document")

    chunk_data: list[dict] = []
    embeddings: list[list[float]] = []
    for chunk in enhanced_chunks:
        chunk_data.append({
            "content": chunk.text,
            "token_count": chunk.token_count,
            "metadata": {
                "section_header": chunk.section_header,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
        })
        embeddings.append(notes_rag_engine.generate_embedding(chunk.text))

    return chunk_data, embeddings


# ============================================================================
# Pydantic Models
# ============================================================================

class NoteResponse(BaseModel):
    id: str
    org_id: str
    unit_id: str
    subject_id: str
    year_id: str
    department_id: str
    stream_id: str
    filename: str
    original_filename: str
    file_path: str
    file_size_bytes: int | None
    mime_type: str | None
    title: str | None
    one_line_summary: str | None
    brief_summary: str | None
    created_at: str
    updated_at: str


class NoteUpdateRequest(BaseModel):
    title: str | None = None


# ============================================================================
# Upload Endpoint
# ============================================================================

@router.post("/upload", response_model=NoteResponse)
async def upload_note(
    file: UploadFile = File(...),
    unit_id: str = Form(...),
    title: str | None = Form(None),
    admin: dict = Depends(require_admin)
):
    """
    Upload a note file (PDF, DOCX, PPTX, images) to a subject unit.

    The file is:
    1. Validated for type
    2. Stored in Supabase Storage (notes bucket)
    3. Text extracted and chunked
    4. Embedded using Gemini
    5. Stored in note_chunks table
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Validate file type
    ALLOWED_TYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    ]

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, PPTX, PNG, JPG, WEBP"
        )

    # Get unit and verify it belongs to this org
    unit = client.table("subject_units").select(
        "id, subject_id, org_id"
    ).eq("id", unit_id).eq("org_id", org_id).single().execute()

    if not unit.data:
        raise HTTPException(status_code=404, detail="Unit not found")

    # Get subject to get year_id
    subject = client.table("subjects").select(
        "id, year_id"
    ).eq("id", unit.data["subject_id"]).single().execute()

    if not subject.data:
        raise HTTPException(status_code=404, detail="Subject not found")

    # Get year to get department_id
    year = client.table("years").select(
        "id, department_id"
    ).eq("id", subject.data["year_id"]).single().execute()

    if not year.data:
        raise HTTPException(status_code=404, detail="Year not found")

    # Get department to get stream_id
    department = client.table("departments").select(
        "id, stream_id"
    ).eq("id", year.data["department_id"]).single().execute()

    if not department.data:
        raise HTTPException(status_code=404, detail="Department not found")

    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid_module.uuid4().hex}{file_ext}"
    object_key = build_object_key(
        org_id=org_id,
        filename=sanitize_filename(unique_filename),
        namespace="notes",
        user_id=admin.get("id"),
    )

    # Read file content
    file_content = await file.read()
    file_size = len(file_content)

    # Save to temporary file for processing
    temp_dir = "/tmp/notes_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, unique_filename)
    note_id: str | None = None
    uploaded_file_path: str | None = None

    try:
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # Upload to Supabase Storage
        storage_result = upload_note_to_storage(
            file_path=object_key,
            file_content=file_content,
            content_type=file.content_type
        )

        if not storage_result:
            raise HTTPException(status_code=500, detail="Failed to upload file to storage")
        uploaded_file_path = object_key

        # Process document (extract text based on file type)
        content_type = file.content_type
        text = extract_note_text(temp_path, content_type)

        if not text or not text.strip():
            raise HTTPException(status_code=500, detail="Failed to extract text from document")

        chunk_data, embeddings = build_note_embeddings(text)

        # Generate summary
        full_text = "\n".join([chunk["content"] for chunk in chunk_data[:5]])  # First 5 chunks for summary
        try:
            summaries = rag_engine.generate_circular_summary(full_text, file.filename)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            summaries = {"one_line": None, "brief": None}

        # Generate AI title if not provided
        if not title:
            try:
                # Get subject name for context
                subject_info = client.table("subjects").select("name").eq("id", unit.data["subject_id"]).single().execute()
                subject_name = subject_info.data.get("name") if subject_info.data else None

                # Get unit number for context
                unit_info = client.table("subject_units").select("unit_number").eq("id", unit_id).single().execute()
                unit_number = unit_info.data.get("unit_number") if unit_info.data else None

                title = rag_engine.generate_note_title(
                    document_text=full_text,
                    filename=file.filename,
                    subject_name=subject_name,
                    unit_number=unit_number
                )
                logger.info(f"Generated AI title: {title}")
            except Exception as e:
                logger.warning(f"Failed to generate AI title: {e}")
                title = file.filename

        # Create note record
        note_data = {
            "org_id": org_id,
            "unit_id": unit_id,
            "subject_id": unit.data["subject_id"],
            "year_id": subject.data["year_id"],
            "department_id": year.data["department_id"],
            "stream_id": department.data["stream_id"],
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_path": object_key,
            "file_size_bytes": file_size,
            "mime_type": file.content_type,
            "title": title,
            "one_line_summary": summaries.get("one_line"),
            "brief_summary": summaries.get("brief"),
            "uploaded_by": admin.get("id"),
        }

        note_result = client.table("notes").insert(note_data).execute()

        if not note_result.data:
            raise HTTPException(status_code=500, detail="Failed to create note record")

        note_id = note_result.data[0]["id"]

        # Add chunks to vector store
        chunks_added = notes_vector_store.add_note(
            note_id=note_id,
            org_id=org_id,
            subject_id=unit.data["subject_id"],
            year_id=subject.data["year_id"],
            department_id=year.data["department_id"],
            stream_id=department.data["stream_id"],
            unit_id=unit_id,
            chunks=chunk_data,
            embeddings=embeddings
        )

        if chunk_data and chunks_added == 0:
            raise HTTPException(status_code=500, detail="Failed to create note embeddings")

        # Check if this is the first note for this unit and update unit name
        existing_notes = client.table("notes").select("id").eq("unit_id", unit_id).is_("deleted_at", "null").execute()
        if len(existing_notes.data or []) == 1:  # This is the first note
            try:
                # Generate AI unit name from the note content
                unit_name_prompt = f"""Based on this academic document content from Unit {unit_number} of {subject_name},
generate a short descriptive name (3-5 words) for this unit topic.

Document content:
{full_text[:2000]}

Respond with ONLY the unit name, nothing else. Make it academic and descriptive."""

                unit_name = rag_engine.generate_text(unit_name_prompt)
                if unit_name and len(unit_name) < 100:
                    client.table("subject_units").update({"name": unit_name.strip()}).eq("id", unit_id).execute()
                    logger.info(f"Updated unit name to: {unit_name.strip()}")
            except Exception as e:
                logger.warning(f"Failed to generate AI unit name: {e}")

        logger.info(f"Uploaded note: {file.filename} -> {unique_filename} ({chunks_added} chunks)")

        return note_result.data[0]

    except Exception:
        cleanup_failed_note_upload(note_id, uploaded_file_path)
        raise

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================================
# List & Get Endpoints
# ============================================================================

@router.get("", response_model=list[NoteResponse])
async def list_notes(
    unit_id: str | None = None,
    subject_id: str | None = None,
    year_id: str | None = None,
    include_deleted: bool = False,
    admin: dict = Depends(require_admin)
):
    """List notes with optional filters."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    query = client.table("notes").select("*").eq("org_id", org_id).order("created_at", desc=True)

    if unit_id:
        query = query.eq("unit_id", unit_id)
    if subject_id:
        query = query.eq("subject_id", subject_id)
    if year_id:
        query = query.eq("year_id", year_id)
    if not include_deleted:
        query = query.is_("deleted_at", "null")

    result = query.execute()
    return result.data or []


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    admin: dict = Depends(require_admin)
):
    """Get a specific note by ID."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    result = client.table("notes").select("*").eq("id", note_id).eq("org_id", org_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Note not found")

    return result.data


# ============================================================================
# Update & Delete Endpoints
# ============================================================================

@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    request: NoteUpdateRequest,
    admin: dict = Depends(require_admin)
):
    """Update note metadata."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    update_data = {k: v for k, v in request.dict().items() if v is not None}

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = client.table("notes").update(update_data).eq("id", note_id).eq("org_id", org_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Note not found")

    return result.data[0]


@router.delete("/{note_id}")
async def delete_note(
    note_id: str,
    hard_delete: bool = False,
    admin: dict = Depends(require_admin)
):
    """
    Delete a note.
    - Soft delete (default): Sets deleted_at timestamp
    - Hard delete: Permanently removes note and chunks
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Verify note exists
    note = client.table("notes").select("id, filename, file_path").eq("id", note_id).eq("org_id", org_id).single().execute()
    if not note.data:
        raise HTTPException(status_code=404, detail="Note not found")

    if hard_delete:
        # Delete chunks
        notes_vector_store.delete_note_chunks(note_id)

        # Delete from storage
        try:
            delete_note_from_storage(note.data["file_path"])
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {e}")

        # Delete note record
        client.table("notes").delete().eq("id", note_id).execute()

        logger.info(f"Hard deleted note: {note.data['filename']}")
        return {"success": True, "message": "Note permanently deleted"}
    else:
        # Soft delete
        client.table("notes").update({"deleted_at": datetime.utcnow().isoformat()}).eq("id", note_id).execute()

        logger.info(f"Soft deleted note: {note.data['filename']}")
        return {"success": True, "message": "Note archived"}


@router.post("/{note_id}/restore")
async def restore_note(
    note_id: str,
    admin: dict = Depends(require_admin)
):
    """Restore a soft-deleted note."""
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    result = client.table("notes").update({"deleted_at": None}).eq("id", note_id).eq("org_id", org_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Note not found")

    logger.info(f"Restored note: {result.data[0]['filename']}")
    return {"success": True, "message": "Note restored"}


# ============================================================================
# Re-embed Endpoint
# ============================================================================

@router.post("/{note_id}/reembed")
async def reembed_note(
    note_id: str,
    admin: dict = Depends(require_admin)
):
    """
    Re-process and re-embed a note.
    Useful when embedding model changes or content needs reprocessing.
    """
    client = get_supabase_admin_client()
    org_id = admin.get("org_id")

    # Get note
    note = client.table("notes").select("*").eq("id", note_id).eq("org_id", org_id).single().execute()
    if not note.data:
        raise HTTPException(status_code=404, detail="Note not found")

    note_data = note.data

    # Download file from storage
    file_content = download_note_from_storage(note_data["file_path"])
    if not file_content:
        raise HTTPException(status_code=500, detail="Failed to download file from storage")

    # Save to temp file
    temp_dir = "/tmp/notes_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, note_data["filename"])

    try:
        with open(temp_path, "wb") as f:
            f.write(file_content)

        text = extract_note_text(temp_path, note_data.get("mime_type"))
        if not text or not text.strip():
            raise HTTPException(status_code=500, detail="Failed to extract text from document")

        chunk_data, embeddings = build_note_embeddings(text)

        # Delete old chunks
        notes_vector_store.delete_note_chunks(note_id)

        # Add new chunks
        chunks_added = notes_vector_store.add_note(
            note_id=note_id,
            org_id=org_id,
            subject_id=note_data["subject_id"],
            year_id=note_data["year_id"],
            department_id=note_data["department_id"],
            stream_id=note_data["stream_id"],
            unit_id=note_data["unit_id"],
            chunks=chunk_data,
            embeddings=embeddings
        )

        if chunk_data and chunks_added == 0:
            raise HTTPException(status_code=500, detail="Failed to recreate note embeddings")

        logger.info(f"Re-embedded note: {note_data['filename']} ({chunks_added} chunks)")

        return {
            "success": True,
            "message": f"Note re-embedded with {chunks_added} chunks"
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================================
# Statistics Endpoint
# ============================================================================

@router.get("/stats/overview")
async def get_notes_stats(
    admin: dict = Depends(require_admin)
):
    """Get notes statistics for the organization."""
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
