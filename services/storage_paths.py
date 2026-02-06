"""
Utilities for building and validating Supabase storage paths.

Path strategy:
- Shared org assets: <org_id>/shared/<namespace>/<uuid>-<filename>
- User-scoped assets: <org_id>/users/<user_id>/<namespace>/<uuid>-<filename>
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Optional


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(filename: str) -> str:
    """Return a safe filename that can be used as an object key segment."""
    name = os.path.basename((filename or "").strip())
    safe = _SAFE_CHARS.sub("_", name)
    return safe or "file"


def build_object_key(
    *,
    org_id: str,
    filename: str,
    namespace: str,
    user_id: Optional[str] = None,
) -> str:
    """
    Build an object key with tenant-safe prefixes.

    Examples:
      - <org_id>/shared/documents/<uuid>-file.pdf
      - <org_id>/users/<user_id>/chat-assets/<uuid>-file.png
    """
    safe_name = sanitize_filename(filename)
    object_id = uuid.uuid4().hex

    if user_id:
        return f"{org_id}/users/{user_id}/{namespace}/{object_id}-{safe_name}"
    return f"{org_id}/shared/{namespace}/{object_id}-{safe_name}"


def build_storage_path(bucket: str, object_key: str) -> str:
    """Join bucket and object key into canonical storage path."""
    return f"{bucket}/{object_key.lstrip('/')}"


def split_storage_path(storage_path: str) -> tuple[str, str]:
    """
    Split a storage path into (bucket, object_key).

    Raises:
      ValueError: if storage_path is malformed.
    """
    bucket, sep, object_key = (storage_path or "").partition("/")
    if not sep or not bucket or not object_key:
        raise ValueError("Invalid storage path format. Expected 'bucket/object_key'")
    return bucket, object_key


def is_org_scoped_object_key(object_key: str, org_id: str) -> bool:
    """
    Check whether object_key is scoped to org_id.

    Supports:
      - Current canonical format: <org_id>/...
      - Legacy notes format: notes/<org_id>/...
    """
    if not object_key or not org_id:
        return False
    if object_key.startswith(f"{org_id}/"):
        return True
    return object_key.startswith(f"notes/{org_id}/")


def is_user_scoped_object_key(object_key: str, org_id: str, user_id: str) -> bool:
    """Check whether object_key is scoped to (org_id, user_id) in canonical format."""
    if not object_key or not org_id or not user_id:
        return False
    return object_key.startswith(f"{org_id}/users/{user_id}/")


def validate_storage_path(
    *,
    storage_path: str,
    org_id: str,
    expected_bucket: Optional[str] = None,
    expected_user_id: Optional[str] = None,
    allow_legacy_org_prefix: bool = False,
) -> bool:
    """
    Validate bucket and org/user scoping of a storage_path.

    Args:
      storage_path: Full path in `bucket/object_key` form.
      org_id: Required tenant org_id.
      expected_bucket: Optional exact bucket match.
      expected_user_id: Optional user ownership requirement.
      allow_legacy_org_prefix: Allow legacy object keys without canonical prefixes.
    """
    try:
        bucket, object_key = split_storage_path(storage_path)
    except ValueError:
        return False

    if expected_bucket and bucket != expected_bucket:
        return False

    if expected_user_id:
        return is_user_scoped_object_key(object_key, org_id, expected_user_id)

    if is_org_scoped_object_key(object_key, org_id):
        return True

    # Optional compatibility escape hatch for old one-segment keys.
    if allow_legacy_org_prefix and "/" not in object_key:
        return True
    return False
