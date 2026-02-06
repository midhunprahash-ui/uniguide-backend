"""
Upload processing pipeline shared by sync and async workers.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from config import get_settings
from services import supabase_storage
from services.document_processor import document_processor
from services.supabase_client import get_supabase_admin_client
from services.admin_stats import refresh_document_stats
from services.storage_paths import validate_storage_path

logger = logging.getLogger(__name__)
settings = get_settings()


def _ensure_local_file(payload: dict) -> str:
    """Ensure the file exists locally and return its path."""
    file_path = payload.get("file_path")
    storage_path = payload.get("storage_path")

    if file_path and os.path.exists(file_path):
        return file_path

    if not storage_path:
        raise ValueError("No file_path or storage_path provided")

    # Download from Supabase Storage
    file_bytes = supabase_storage.download_file(storage_path)

    original_filename = payload.get("original_filename") or os.path.basename(storage_path)
    safe_filename = f"{uuid.uuid4()}-{original_filename}"
    os.makedirs(settings.upload_directory, exist_ok=True)
    local_path = os.path.join(settings.upload_directory, safe_filename)

    with open(local_path, "wb") as f:
        f.write(file_bytes)

    return local_path


def _resolve_fk_ids(
    client,
    org_id: str,
    stream: str | None,
    department: str | None,
    year: str | None,
    stream_id: str | None,
    department_id: str | None,
    year_id: str | None,
) -> dict[str, str | None]:
    """Resolve FK IDs from codes if not provided."""
    resolved_stream_id = stream_id
    resolved_department_id = department_id
    resolved_year_id = year_id

    if not resolved_stream_id and stream and stream != "all" and "," not in stream:
        stream_result = client.table("streams").select("id").eq("code", stream).eq("org_id", org_id).maybe_single().execute()
        if stream_result.data:
            resolved_stream_id = stream_result.data["id"]

    if not resolved_department_id and department and department != "all" and "," not in department:
        dept_query = client.table("departments").select("id").eq("code", department).eq("org_id", org_id)
        if resolved_stream_id:
            dept_query = dept_query.eq("stream_id", resolved_stream_id)
        dept_result = dept_query.maybe_single().execute()
        if dept_result.data:
            resolved_department_id = dept_result.data["id"]

    if not resolved_year_id and year and year != "all" and "," not in year and resolved_department_id:
        year_result = client.table("years").select("id").eq("code", year).eq("department_id", resolved_department_id).maybe_single().execute()
        if year_result.data:
            resolved_year_id = year_result.data["id"]

    return {
        "stream_id": resolved_stream_id,
        "department_id": resolved_department_id,
        "year_id": resolved_year_id,
    }


def process_upload_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Process a document upload payload and return result."""
    org_id = payload.get("org_id")
    if not org_id:
        raise ValueError("org_id is required")

    storage_path = payload.get("storage_path")
    if storage_path:
        category = payload.get("category", "rules")
        expected_bucket = supabase_storage.get_bucket_for_category(category)
        uploaded_by = payload.get("uploaded_by")
        is_user_scoped_path = validate_storage_path(
            storage_path=storage_path,
            org_id=org_id,
            expected_bucket=expected_bucket,
            expected_user_id=uploaded_by,
            allow_legacy_org_prefix=False,
        )
        is_org_scoped_path = validate_storage_path(
            storage_path=storage_path,
            org_id=org_id,
            expected_bucket=expected_bucket,
            allow_legacy_org_prefix=False,
        )
        is_legacy_path = validate_storage_path(
            storage_path=storage_path,
            org_id=org_id,
            expected_bucket=expected_bucket,
            allow_legacy_org_prefix=True,
        )
        if not is_user_scoped_path and not is_org_scoped_path and not is_legacy_path:
            raise ValueError("Invalid storage_path scope for upload payload")
        if is_legacy_path and not is_org_scoped_path:
            logger.warning("Processing upload payload with legacy storage path: %s", storage_path)

    file_path = None
    try:
        file_path = _ensure_local_file(payload)

        filename = payload.get("original_filename") or payload.get("filename")
        if not filename:
            filename = os.path.basename(file_path)

        file_ext = payload.get("file_ext") or filename.split(".")[-1].lower()

        # Process and store document (creates embeddings in DB)
        result = document_processor.process_and_store_document(
            file_path=file_path,
            filename=filename,
            year=payload.get("year", "all"),
            department=payload.get("department", "all"),
            category=payload.get("category", "rules"),
            file_type=file_ext,
            org_id=org_id,
            stream=payload.get("stream", "all"),
            semester=payload.get("semester", "all"),
        )

        # Ensure file is in Supabase Storage
        storage_path = payload.get("storage_path")
        if not storage_path:
            storage_path = supabase_storage.upload_file(
                file_path=file_path,
                category=payload.get("category", "rules"),
                filename=filename,
                org_id=org_id,
                uploaded_by=payload.get("uploaded_by"),
                namespace=f"documents/{payload.get('category', 'rules')}",
            )

        # Update document record with storage path and FK IDs
        client = get_supabase_admin_client()
        doc_result = (
            client.table("documents")
            .select("id, created_at")
            .eq("filename", filename)
            .eq("org_id", org_id)
            .maybe_single()
            .execute()
        )

        if doc_result.data:
            document_id = doc_result.data["id"]

            fk_ids = _resolve_fk_ids(
                client,
                org_id=org_id,
                stream=payload.get("stream"),
                department=payload.get("department"),
                year=payload.get("year"),
                stream_id=payload.get("stream_id"),
                department_id=payload.get("department_id"),
                year_id=payload.get("year_id"),
            )

            update_data = {"storage_path": storage_path}
            if payload.get("uploaded_by"):
                update_data["uploaded_by"] = payload.get("uploaded_by")
            if payload.get("original_filename"):
                update_data["original_filename"] = payload.get("original_filename")
            if payload.get("file_size_bytes") is not None:
                update_data["file_size_bytes"] = payload.get("file_size_bytes")
            if payload.get("mime_type"):
                update_data["mime_type"] = payload.get("mime_type")
            if fk_ids["stream_id"]:
                update_data["stream_id"] = fk_ids["stream_id"]
            if fk_ids["department_id"]:
                update_data["department_id"] = fk_ids["department_id"]
            if fk_ids["year_id"]:
                update_data["year_id"] = fk_ids["year_id"]

            client.table("documents").update(update_data).eq("id", document_id).execute()
            result.update({
                "storage_path": storage_path,
                "stream_id": fk_ids["stream_id"],
                "department_id": fk_ids["department_id"],
                "year_id": fk_ids["year_id"],
            })

            # Circular summary + register
            circular_id = None
            if payload.get("category") == "circulars":
                from services.rag_engine import rag_engine
                extracted_text = result.get("extracted_text", "")
                summaries = rag_engine.generate_circular_summary(extracted_text, filename)
                client.table("documents").update({
                    "one_line_summary": summaries["one_line"],
                    "brief_summary": summaries["brief"],
                }).eq("id", document_id).execute()

                from routes.circular import register_circular
                circular_id = register_circular(
                    doc_id=document_id,
                    filename=filename,
                    year=payload.get("year", "all"),
                    department=payload.get("department", "all"),
                    upload_date=doc_result.data["created_at"],
                    one_line_summary=summaries["one_line"],
                    brief_summary=summaries["brief"],
                    chunk_count=result.get("chunks", 0),
                    org_id=org_id,
                    document_text=extracted_text,
                )

            # Deadline extraction (best-effort)
            if result.get("extracted_text"):
                try:
                    from routes.deadlines import register_deadlines_from_document
                    register_deadlines_from_document(
                        document_id=document_id,
                        document_text=result["extracted_text"],
                        org_id=org_id,
                        circular_id=circular_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract deadlines from document: {e}")

        # Clean up local file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Refresh materialized view for updated stats
        refresh_document_stats()

        return result

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
