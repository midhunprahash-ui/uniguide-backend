import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from models.schemas import EventDiscoverQuery, SaveEventRequest
from services.auth import require_auth, require_valid_org_id
from services.event_cache import event_cache
from services.event_discovery import event_discovery
from services.event_store import event_store
from services.event_utils import enrich_event_payload
from services.rate_limiter import RATE_LIMITS, get_org_key, limiter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/discover-stream")
@limiter.limit(RATE_LIMITS["chat"], key_func=get_org_key)
async def discover_events_stream(request: Request, query: EventDiscoverQuery):
    """
    Stream event discovery results using SSE.
    """
    try:
        require_valid_org_id(query.org_id)
        max_results = max(1, min(query.max_results or 40, 60))

        cached = event_cache.get(
            query.question,
            query.org_id,
            query.nearby,
            query.nearby_location,
            query.nearby_lat,
            query.nearby_lng,
        )

        async def event_generator():
            try:
                if cached:
                    for citation in cached.citations:
                        yield f"data: {json.dumps({'type': 'citation', 'data': citation})}\n\n"
                    for event in cached.events:
                        enriched = enrich_event_payload(event)
                        yield f"data: {json.dumps({'type': 'event', 'data': enriched})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"
                    return

                citations: list[dict] = []
                events: list[dict] = []

                for kind, data in event_discovery.discover_stream(
                    question=query.question,
                    max_results=max_results,
                    nearby=query.nearby,
                    nearby_location=query.nearby_location,
                    nearby_lat=query.nearby_lat,
                    nearby_lng=query.nearby_lng,
                ):
                    if kind == "citation":
                        citations.append(data)
                    elif kind == "event":
                        enriched = enrich_event_payload(data)
                        events.append(enriched)
                        data = enriched

                    yield f"data: {json.dumps({'type': kind, 'data': data})}\n\n"

                event_cache.set(
                    query.question,
                    query.org_id,
                    query.nearby,
                    query.nearby_location,
                    query.nearby_lat,
                    query.nearby_lng,
                    events=events,
                    citations=citations,
                )

                yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"

            except Exception as e:
                logger.error("Event discovery stream error: %s", e)
                error_json = json.dumps({"type": "error", "data": str(e)})
                yield f"data: {error_json}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting event discovery: {str(e)}")


@router.get("/saved")
async def list_saved_events(current_user: dict = Depends(require_auth)):
    try:
        user_id = current_user.get("id")
        org_id = event_store.get_user_org_id(user_id)
        if not org_id:
            raise HTTPException(status_code=403, detail="User organization not found")

        return event_store.list_saved_events(user_id, org_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list saved events: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list saved events")


@router.post("/saved")
async def save_event(payload: SaveEventRequest, current_user: dict = Depends(require_auth)):
    try:
        user_id = current_user.get("id")
        org_id = event_store.get_user_org_id(user_id)
        if not org_id:
            raise HTTPException(status_code=403, detail="User organization not found")

        event_id, event_key = event_store.upsert_event(org_id, payload.model_dump())
        event_store.save_event_for_user(user_id, org_id, event_id)
        return {"event_id": event_id, "event_key": event_key}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save event: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save event")


@router.delete("/saved/{event_id}")
async def delete_saved_event(event_id: str, current_user: dict = Depends(require_auth)):
    try:
        user_id = current_user.get("id")
        org_id = event_store.get_user_org_id(user_id)
        if not org_id:
            raise HTTPException(status_code=403, detail="User organization not found")

        event_store.delete_saved_event(user_id, org_id, event_id)
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete saved event: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete saved event")
