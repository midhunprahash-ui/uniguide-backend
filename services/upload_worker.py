"""RQ worker task for processing uploads."""
from __future__ import annotations

import logging

from rq import get_current_job

from services.upload_pipeline import process_upload_payload

logger = logging.getLogger(__name__)


def process_upload_job(payload: dict) -> dict:
    job = get_current_job()
    try:
        result = process_upload_payload(payload)
        if job:
            job.meta["result"] = {"status": "done", "document": result.get("doc_id")}
            job.save_meta()
        return result
    except Exception as e:
        logger.error(f"Upload job failed: {e}")
        if job:
            job.meta["error"] = str(e)
            job.save_meta()
        raise
