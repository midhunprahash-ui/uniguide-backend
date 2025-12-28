"""
Supabase Storage service for document file management.
Handles upload, download, and deletion of document files in Supabase Storage buckets.
"""
import logging
import os
from typing import Optional

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)

# Map categories to bucket names
CATEGORY_BUCKET_MAP = {
    "rules": "documents-rules",
    "admissions": "documents-admissions",
    "schedules": "documents-schedules",
    "timetables": "documents-timetables",
    "abhs": "documents-abhs",
    "circulars": "documents-circulars",
}


def get_bucket_for_category(category: str) -> str:
    """Get the storage bucket name for a given category."""
    bucket = CATEGORY_BUCKET_MAP.get(category.lower())
    if not bucket:
        # Default to rules bucket if category not found
        logger.warning(f"Unknown category '{category}', using documents-rules bucket")
        bucket = "documents-rules"
    return bucket


def upload_file(
    file_path: str,
    category: str,
    filename: str,
    content_type: Optional[str] = None
) -> str:
    """
    Upload a file to Supabase Storage.
    
    Args:
        file_path: Local path to the file
        category: Document category (determines bucket)
        filename: Name to store the file as
        content_type: Optional MIME type
    
    Returns:
        Storage path (bucket/filename format)
    """
    client = get_supabase_admin_client()
    bucket = get_bucket_for_category(category)
    
    # Read file content
    with open(file_path, "rb") as f:
        file_data = f.read()
    
    # Determine content type if not provided
    if not content_type:
        ext = filename.split(".")[-1].lower()
        content_type_map = {
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "txt": "text/plain",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")
    
    try:
        # Upload to Supabase Storage
        result = client.storage.from_(bucket).upload(
            path=filename,
            file=file_data,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        storage_path = f"{bucket}/{filename}"
        logger.info(f"✅ Uploaded file to Supabase Storage: {storage_path}")
        return storage_path
        
    except Exception as e:
        logger.error(f"❌ Failed to upload file to Supabase Storage: {e}")
        raise


def download_file(storage_path: str) -> bytes:
    """
    Download a file from Supabase Storage.
    
    Args:
        storage_path: Path in format "bucket/filename"
    
    Returns:
        File content as bytes
    """
    client = get_supabase_admin_client()
    
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {storage_path}")
    
    bucket, filename = parts
    
    try:
        result = client.storage.from_(bucket).download(filename)
        logger.info(f"✅ Downloaded file from Supabase Storage: {storage_path}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to download file from Supabase Storage: {e}")
        raise


def delete_file(storage_path: str) -> bool:
    """
    Delete a file from Supabase Storage.
    
    Args:
        storage_path: Path in format "bucket/filename"
    
    Returns:
        True if deleted successfully
    """
    client = get_supabase_admin_client()
    
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        logger.warning(f"Invalid storage path for deletion: {storage_path}")
        return False
    
    bucket, filename = parts
    
    try:
        client.storage.from_(bucket).remove([filename])
        logger.info(f"✅ Deleted file from Supabase Storage: {storage_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete file from Supabase Storage: {e}")
        return False


def get_public_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a signed URL for a file in Supabase Storage.
    
    Args:
        storage_path: Path in format "bucket/filename"
        expires_in: URL expiration time in seconds (default 1 hour)
    
    Returns:
        Signed URL for the file
    """
    client = get_supabase_admin_client()
    
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {storage_path}")
    
    bucket, filename = parts
    
    try:
        result = client.storage.from_(bucket).create_signed_url(
            path=filename,
            expires_in=expires_in
        )
        return result.get("signedURL", "")
        
    except Exception as e:
        logger.error(f"❌ Failed to get signed URL: {e}")
        raise


def file_exists(storage_path: str) -> bool:
    """
    Check if a file exists in Supabase Storage.
    
    Args:
        storage_path: Path in format "bucket/filename"
    
    Returns:
        True if file exists
    """
    client = get_supabase_admin_client()
    
    parts = storage_path.split("/", 1)
    if len(parts) != 2:
        return False
    
    bucket, filename = parts
    
    try:
        # List files to check existence
        result = client.storage.from_(bucket).list(path="", options={"search": filename})
        return any(f.get("name") == filename for f in result)
        
    except Exception:
        return False


def list_all_files(category: str = None) -> list[dict]:
    """
    List all files in storage buckets.
    
    Args:
        category: Optional category to filter by. If None, lists all buckets.
    
    Returns:
        List of dicts with bucket, filename, and size info
    """
    client = get_supabase_admin_client()
    all_files = []
    
    buckets_to_check = [get_bucket_for_category(category)] if category else CATEGORY_BUCKET_MAP.values()
    
    for bucket in buckets_to_check:
        try:
            files = client.storage.from_(bucket).list()
            for f in files:
                if f.get("name") and not f.get("name").startswith("."):
                    all_files.append({
                        "bucket": bucket,
                        "filename": f.get("name"),
                        "size": f.get("metadata", {}).get("size", 0),
                        "storage_path": f"{bucket}/{f.get('name')}"
                    })
        except Exception as e:
            logger.warning(f"Failed to list files in bucket {bucket}: {e}")
    
    return all_files


def rename_file(old_storage_path: str, new_filename: str) -> str:
    """
    Rename a file in Supabase Storage by copying to new name and deleting old.
    
    Args:
        old_storage_path: Current path in format "bucket/filename"
        new_filename: New filename (same bucket)
    
    Returns:
        New storage path
    """
    client = get_supabase_admin_client()
    
    parts = old_storage_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path: {old_storage_path}")
    
    bucket, old_filename = parts
    
    if old_filename == new_filename:
        return old_storage_path  # No change needed
    
    try:
        # Download file content
        file_data = client.storage.from_(bucket).download(old_filename)
        
        # Determine content type
        ext = new_filename.split(".")[-1].lower()
        content_type_map = {
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "txt": "text/plain",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")
        
        # Upload with new name
        client.storage.from_(bucket).upload(
            path=new_filename,
            file=file_data,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        # Delete old file
        client.storage.from_(bucket).remove([old_filename])
        
        new_storage_path = f"{bucket}/{new_filename}"
        logger.info(f"✅ Renamed file in Storage: {old_storage_path} -> {new_storage_path}")
        return new_storage_path
        
    except Exception as e:
        logger.error(f"❌ Failed to rename file in Storage: {e}")
        raise
