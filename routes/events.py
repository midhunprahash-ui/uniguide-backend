import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from models.schemas import EventDiscoverQuery
from services.auth import require_valid_org_id
from services.event_cache import event_cache
from services.event_discovery import event_discovery
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
                        yield f"data: {json.dumps({'type': 'event', 'data': event})}\n\n"
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
                        events.append(data)

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
