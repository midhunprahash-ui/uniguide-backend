"""
Event persistence for saved events.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from services.event_utils import enrich_event_payload
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


class EventStore:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_supabase_admin_client()
        return self._client

    def get_user_org_id(self, user_id: str) -> Optional[str]:
        try:
            result = self.client.table("profiles").select("org_id").eq("id", user_id).single().execute()
            if result.data:
                return result.data.get("org_id")
        except Exception as e:
            logger.error("Failed to fetch org_id for user %s: %s", user_id, e)
        return None

    def upsert_event(self, org_id: str, event: dict) -> tuple[str, str]:
        enriched = enrich_event_payload(event)
        now = datetime.now(timezone.utc).isoformat()
        metadata = enriched.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if enriched.get("prize_type"):
            metadata["prize_type"] = enriched.get("prize_type")
        if enriched.get("cash_prize_currency"):
            metadata["cash_prize_currency"] = enriched.get("cash_prize_currency")
        if enriched.get("cash_prize_amount") is not None:
            metadata["cash_prize_amount"] = enriched.get("cash_prize_amount")

        payload = {
            "org_id": org_id,
            "event_key": enriched.get("event_key"),
            "name": enriched.get("name"),
            "start_date": enriched.get("start_date"),
            "end_date": enriched.get("end_date"),
            "date_text": enriched.get("date"),
            "location": enriched.get("location"),
            "cash_prize": enriched.get("cash_prize") or enriched.get("prize_display_text"),
            "short_description": enriched.get("short_description"),
            "url": enriched.get("url"),
            "source_url": enriched.get("source_url"),
            "source_domain": enriched.get("source_domain"),
            "status": enriched.get("status") or "unknown",
            "metadata": metadata,
            "updated_at": now,
        }

        try:
            self.client.table("events").upsert(
                payload,
                on_conflict="org_id,event_key",
            ).execute()
        except Exception as e:
            logger.error("Event upsert failed: %s", e)
            raise

        event_key = payload["event_key"]
        result = self.client.table("events").select("id").eq("org_id", org_id).eq("event_key", event_key).single().execute()
        if not result.data:
            raise RuntimeError("Failed to read event after upsert")
        return result.data["id"], event_key

    def save_event_for_user(self, user_id: str, org_id: str, event_id: str) -> None:
        try:
            self.client.table("saved_events").upsert(
                {
                    "user_id": user_id,
                    "org_id": org_id,
                    "event_id": event_id,
                },
                on_conflict="user_id,event_id",
            ).execute()
        except Exception as e:
            logger.error("Save event failed: %s", e)
            raise

    def delete_saved_event(self, user_id: str, org_id: str, event_id: str) -> None:
        try:
            self.client.table("saved_events").delete().eq("user_id", user_id).eq("org_id", org_id).eq("event_id", event_id).execute()
        except Exception as e:
            logger.error("Delete saved event failed: %s", e)
            raise

    def list_saved_events(self, user_id: str, org_id: str) -> list[dict]:
        try:
            result = (
                self.client.table("saved_events")
                .select("id, event_id, created_at, events(*)")
                .eq("user_id", user_id)
                .eq("org_id", org_id)
                .order("created_at", desc=True)
                .execute()
            )
        except Exception as e:
            logger.error("List saved events failed: %s", e)
            raise

        rows = result.data or []
        events: list[dict] = []
        for row in rows:
            event = row.get("events") or {}
            if not event:
                continue
            event["saved_id"] = row.get("id")
            event["saved_at"] = row.get("created_at")
            events.append(event)
        return events


event_store = EventStore()
