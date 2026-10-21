import hashlib
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import RLock

import requests


logger = logging.getLogger(__name__)
CACHE_WRITE_EXECUTOR = ThreadPoolExecutor(max_workers=2)


class AnalysisCache:
    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[datetime, dict]] = OrderedDict()
        self._lock = RLock()
        self._supabase_retry_after = 0.0

    @staticmethod
    def key(description: str) -> str:
        normalized = " ".join(description.casefold().split())
        return hashlib.sha256(
            f"community-analysis-v6:{normalized}".encode("utf-8")
        ).hexdigest()

    def get(self, key: str) -> dict | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            cached = self._items.get(key)
            if cached:
                expires_at, payload = cached
                if expires_at > now:
                    self._items.move_to_end(key)
                    return payload
                del self._items[key]

        base_url, api_key = self._supabase_config()
        if not base_url or not api_key or self._supabase_is_backing_off():
            return None
        try:
            response = requests.get(
                f"{base_url}/rest/v1/analysis_cache",
                headers=self._headers(api_key),
                params={
                    "cache_key": f"eq.{key}",
                    "expires_at": f"gt.{now.isoformat()}",
                    "select": "payload,expires_at",
                    "limit": "1",
                },
                timeout=(0.3, 0.8),
            )
            response.raise_for_status()
            self._mark_supabase_available()
            rows = response.json()
            if rows:
                payload = rows[0]["payload"]
                expires_at = datetime.fromisoformat(
                    rows[0]["expires_at"].replace("Z", "+00:00")
                )
                self._store_memory(key, payload, expires_at)
                return payload
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            self._mark_supabase_unavailable()
            logger.info("Shared cache unavailable; local cache active (%s).", type(exc).__name__)
        return None

    def set(self, key: str, payload: dict) -> None:
        ttl_seconds = max(30, int(os.getenv("ANALYSIS_CACHE_TTL_SECONDS", "600")))
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        self._store_memory(key, payload, expires_at)

        base_url, api_key = self._supabase_config()
        if not base_url or not api_key or self._supabase_is_backing_off():
            return
        CACHE_WRITE_EXECUTOR.submit(
            self._write_supabase,
            base_url,
            api_key,
            key,
            payload,
            expires_at,
        )

    def _write_supabase(
        self,
        base_url: str,
        api_key: str,
        key: str,
        payload: dict,
        expires_at: datetime,
    ) -> None:
        try:
            response = requests.post(
                f"{base_url}/rest/v1/analysis_cache",
                headers={
                    **self._headers(api_key),
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
                params={"on_conflict": "cache_key"},
                json={
                    "cache_key": key,
                    "payload": payload,
                    "expires_at": expires_at.isoformat(),
                },
                timeout=(0.3, 0.8),
            )
            response.raise_for_status()
            self._mark_supabase_available()
        except requests.RequestException as exc:
            self._mark_supabase_unavailable()
            logger.info("Shared cache write skipped; local cache remains active (%s).", type(exc).__name__)

    def _store_memory(self, key: str, payload: dict, expires_at: datetime) -> None:
        with self._lock:
            self._items[key] = (expires_at, payload)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    @staticmethod
    def _supabase_config() -> tuple[str | None, str | None]:
        url = os.getenv("SUPABASE_URL", "").rstrip("/")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.getenv("SUPABASE_KEY", "").strip()
        )
        return (url or None, key or None)

    def health(self) -> dict[str, str | bool]:
        """Report cache setup without exposing credentials or provider errors."""
        base_url, api_key = self._supabase_config()
        if not base_url and not api_key:
            return {"configured": False, "reachable": False, "status": "memory_only"}
        if not base_url or not api_key:
            return {"configured": False, "reachable": False, "status": "missing_credentials"}
        if self._supabase_is_backing_off():
            return {"configured": True, "reachable": False, "status": "unavailable"}
        try:
            response = requests.get(
                f"{base_url}/rest/v1/analysis_cache",
                headers=self._headers(api_key),
                params={"select": "cache_key", "limit": "1"},
                timeout=(0.5, 1.5),
            )
            response.raise_for_status()
            self._mark_supabase_available()
            return {"configured": True, "reachable": True, "status": "connected"}
        except requests.RequestException as exc:
            self._mark_supabase_unavailable()
            logger.info("Shared cache health check unavailable; local cache remains active (%s).", type(exc).__name__)
            return {"configured": True, "reachable": False, "status": "unavailable"}

    def _supabase_is_backing_off(self) -> bool:
        with self._lock:
            return time.monotonic() < self._supabase_retry_after

    def _mark_supabase_unavailable(self) -> None:
        with self._lock:
            self._supabase_retry_after = time.monotonic() + 60

    def _mark_supabase_available(self) -> None:
        with self._lock:
            self._supabase_retry_after = 0.0

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }


analysis_cache = AnalysisCache()
