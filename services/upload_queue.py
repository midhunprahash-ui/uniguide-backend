"""Upload queue utilities using Redis + RQ."""
from __future__ import annotations

import os
import logging
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job

from services.upload_worker import process_upload_job

logger = logging.getLogger(__name__)


def _get_redis_url() -> str | None:
    return os.getenv("REDIS_URL")


def get_queue() -> Queue | None:
    redis_url = _get_redis_url()
    if not redis_url:
        return None
    return Queue("uploads", connection=Redis.from_url(redis_url))


def enqueue_upload(payload: dict[str, Any]) -> str | None:
    queue = get_queue()
    if not queue:
        logger.warning("Upload queue not configured; REDIS_URL missing")
        return None

    job = queue.enqueue(
        process_upload_job,
        payload,
        job_timeout=int(os.getenv("UPLOAD_JOB_TIMEOUT_SEC", "1800")),
        result_ttl=int(os.getenv("UPLOAD_JOB_RESULT_TTL_SEC", "3600")),
        failure_ttl=int(os.getenv("UPLOAD_JOB_FAILURE_TTL_SEC", "3600")),
    )
    job.meta["org_id"] = payload.get("org_id")
    job.meta["uploaded_by"] = payload.get("uploaded_by")
    job.save_meta()
    return job.id


def get_upload_status(job_id: str, org_id: str | None = None) -> dict[str, Any] | None:
    queue = get_queue()
    if not queue:
        return None

    try:
        job = Job.fetch(job_id, connection=queue.connection)
    except Exception:
        return None

    if org_id and job.meta.get("org_id") and job.meta.get("org_id") != org_id:
        return None

    status = job.get_status()
    payload: dict[str, Any] = {
        "job_id": job.id,
        "status": status,
    }
    if job.meta.get("error"):
        payload["error"] = job.meta.get("error")
    if job.meta.get("result"):
        payload["result"] = job.meta.get("result")
    return payload
