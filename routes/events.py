import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from models.schemas import EventDiscoverQuery, SaveEventRequest
from services.auth import require_auth, require_org_membership, require_valid_org_id
from services.event_cache import event_cache
from services.event_discovery import get_event_discovery
from services.event_store import event_store
from services.event_utils import enrich_event_payload
from services.provider_error_mapper import map_provider_error, provider_error_sse_payload
from services.rate_limiter import RATE_LIMITS, get_org_key, limiter

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_user_id(current_user: dict) -> str:
    user_id = current_user.get("id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(status_code=401, detail="Invalid user context")
    return user_id


@router.post("/discover-stream")
@limiter.limit(RATE_LIMITS["chat"], key_func=get_org_key)
async def discover_events_stream(
    request: Request,
    query: EventDiscoverQuery,
    current_user: dict = Depends(require_auth),
):
    """
    Stream event discovery results using SSE.
    """
    try:
        require_valid_org_id(query.org_id)
        user_id = _require_user_id(current_user)
        require_org_membership(user_id, query.org_id)
        max_results = max(1, min(query.max_results or 40, 60))

        cached = event_cache.get(
            query.question,
            query.org_id,
            query.nearby,
            query.nearby_location,
            query.nearby_lat,
            query.nearby_lng,
            query.category_hint,
            query.strict_trust,
            query.accuracy_mode,
            query.geo_scope,
        )
        try:
            discovery = get_event_discovery()
        except RuntimeError as e:
            logger.error("Event discovery initialization failed: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Event discovery is temporarily unavailable. Please try again later.",
            )

        async def event_generator():
            try:
                if cached:
                    policy = discovery.evaluate_policy(
                        query.question,
                        query.category_hint,
                        nearby=query.nearby,
                        nearby_location=query.nearby_location,
                        geo_scope=query.geo_scope,
                    ).to_payload()
                    yield f"data: {json.dumps({'type': 'policy', 'data': policy})}\n\n"
                    if not policy.get("allowed", False):
                        yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"
                        return

                    search_plan = discovery.build_search_plan(
                        query.question,
                        policy.get("normalized_intent", "general_student_opportunity"),
                        query.nearby,
                        query.nearby_location,
                        max_results,
                        policy.get("intent_context"),
                    )
                    yield f"data: {json.dumps({'type': 'search_plan', 'data': search_plan})}\n\n"

                    for citation in cached.citations:
                        yield f"data: {json.dumps({'type': 'citation', 'data': citation})}\n\n"
                    for event in cached.events:
                        enriched = enrich_event_payload(event)
                        yield f"data: {json.dumps({'type': 'event', 'data': enriched})}\n\n"
                    metrics = {
                        "cached": True,
                        "policy_allowed": True,
                        "citations_emitted": len(cached.citations),
                        "events_emitted": len(cached.events),
                        "first_citation_ms": None,
                        "first_event_ms": None,
                        "total_ms": 0,
                    }
                    yield f"data: {json.dumps({'type': 'metrics', 'data': metrics})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"
                    return

                citations: list[dict] = []
                events: list[dict] = []
                policy_payload: dict | None = None

                for kind, data in discovery.discover_stream(
                    question=query.question,
                    max_results=max_results,
                    nearby=query.nearby,
                    nearby_location=query.nearby_location,
                    nearby_lat=query.nearby_lat,
                    nearby_lng=query.nearby_lng,
                    category_hint=query.category_hint,
                    strict_trust=query.strict_trust,
                    accuracy_mode=query.accuracy_mode,
                    geo_scope=query.geo_scope,
                ):
                    if kind == "citation":
                        citations.append(data)
                    elif kind == "event":
                        enriched = enrich_event_payload(data)
                        events.append(enriched)
                        data = enriched
                    elif kind == "policy":
                        policy_payload = data

                    yield f"data: {json.dumps({'type': kind, 'data': data})}\n\n"

                if policy_payload is None or policy_payload.get("allowed", False):
                    event_cache.set(
                        query.question,
                        query.org_id,
                        query.nearby,
                        query.nearby_location,
                        query.nearby_lat,
                        query.nearby_lng,
                        query.category_hint,
                        query.strict_trust,
                        query.accuracy_mode,
                        query.geo_scope,
                        events=events,
                        citations=citations,
                    )

                yield f"data: {json.dumps({'type': 'done', 'data': True})}\n\n"

            except Exception as e:
                logger.exception("Event discovery stream error")
                error_json = json.dumps(
                    provider_error_sse_payload(
                        e,
                        fallback_message=(
                            "We could not complete event discovery right now. Please try again shortly."
                        ),
                    )
                )
                yield f"data: {error_json}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start event discovery")
        mapped = map_provider_error(
            e,
            fallback_message="Unable to start event discovery right now. Please try again shortly.",
        )
        raise HTTPException(status_code=mapped.status_code, detail=mapped.message)


@router.get("/saved")
async def list_saved_events(current_user: dict = Depends(require_auth)):
    try:
        user_id = _require_user_id(current_user)
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
        user_id = _require_user_id(current_user)
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
        user_id = _require_user_id(current_user)
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
