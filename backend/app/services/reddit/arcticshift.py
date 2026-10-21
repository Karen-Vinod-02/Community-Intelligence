import time
import os
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from app.models.reddit import RedditPost, RedditComment
from app.services.reddit.base import RedditSource


ARCTIC_SHIFT_BASE = os.getenv(
    "ARCTIC_SHIFT_BASE",
    "https://arctic-shift.photon-reddit.com/api",
)

HEADERS = {
    "User-Agent": "community-intelligence/0.1"
}

REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 0
RETRY_BACKOFF_SECONDS = 0.5
MAX_RETRY_AFTER_SECONDS = 10
RATE_LIMIT_COOLDOWN_SECONDS = 30
INVALID_REQUEST_COOLDOWN_SECONDS = 120


class ArcticShiftUnavailable(RuntimeError):
    pass


class ArcticShiftSource(RedditSource):

    def __init__(
        self,
        base_url: str = ARCTIC_SHIFT_BASE,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._state_lock = threading.Lock()
        self._blocked_until = 0.0
        self._blocked_reason: str | None = None

    def _check_available(self) -> None:
        with self._state_lock:
            remaining = self._blocked_until - time.monotonic()
            reason = self._blocked_reason
        if remaining > 0:
            raise ArcticShiftUnavailable(
                f"Arctic Shift temporarily unavailable ({reason}); "
                f"retry in about {int(remaining) + 1}s."
            )

    def _block(self, seconds: float, reason: str) -> None:
        with self._state_lock:
            self._blocked_until = max(
                self._blocked_until,
                time.monotonic() + seconds,
            )
            self._blocked_reason = reason

    def _request(self, endpoint: str, params: dict):
        self._check_available()
        url = f"{self.base_url}/{endpoint}"

        last_exception = None

        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=HEADERS,
                    timeout=self.timeout,
                )

                response.raise_for_status()
                return response

            except requests.HTTPError as exc:
                status = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )

                if status is not None and 400 <= status < 500:
                    if status not in (408, 429):
                        raise

                if status == 429 and exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After")
                    delay = self._retry_delay(retry_after, attempt)
                    self._block(
                        self._rate_limit_cooldown(retry_after),
                        "rate limited (HTTP 429)",
                    )

                last_exception = exc

            except requests.RequestException as exc:
                last_exception = exc

            if attempt < self.max_retries:
                delay = self._retry_delay(retry_after, attempt)
                if delay is None:
                    raise last_exception
                time.sleep(delay)

        raise last_exception

    @staticmethod
    def _retry_delay(retry_after: str | None, attempt: int) -> float | None:
        if retry_after:
            delay = ArcticShiftSource._parse_retry_after(retry_after)
            if delay is None:
                delay = RETRY_BACKOFF_SECONDS * (attempt + 1)

            if delay > MAX_RETRY_AFTER_SECONDS:
                return None
            return max(0.0, delay)

        return RETRY_BACKOFF_SECONDS * (attempt + 1)

    @staticmethod
    def _parse_retry_after(retry_after: str) -> float | None:
        try:
            return float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return (
                    retry_at - datetime.now(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _rate_limit_cooldown(retry_after: str | None) -> float:
        delay = (
            ArcticShiftSource._parse_retry_after(retry_after)
            if retry_after
            else None
        )
        if delay is None:
            return RATE_LIMIT_COOLDOWN_SECONDS
        return min(
            max(delay, RATE_LIMIT_COOLDOWN_SECONDS),
            300.0,
        )

    def search_posts(
        self,
        subreddit: str,
        query: str | None = None,
        limit: int = 50,
    ) -> list[RedditPost]:

        params = {
            "subreddit": subreddit,
            "limit": min(limit, 100),
            "sort": "desc",
        }

        if query:
            params["query"] = query

        try:
            response = self._request(
                "posts/search",
                params,
            )

        except requests.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None
                else None
            )

            # If keyword search is rejected for a large/active
            # subreddit, fall back to recent posts.
            # Ranking layer will determine which posts & communities are actually relevant to the product.
            if status == 422 and query:
                params.pop("query", None)

                try:
                    response = self._request(
                        "posts/search",
                        params,
                    )
                except requests.HTTPError as fallback_exc:
                    fallback_status = (
                        fallback_exc.response.status_code
                        if fallback_exc.response is not None
                        else None
                    )
                    if fallback_status == 422:
                        self._block(
                            INVALID_REQUEST_COOLDOWN_SECONDS,
                            "recent-post fallback rejected (HTTP 422)",
                        )
                    raise
            else:
                if status == 422:
                    self._block(
                        INVALID_REQUEST_COOLDOWN_SECONDS,
                        "recent-post search rejected (HTTP 422)",
                    )
                raise

        payload = response.json()
        data = payload.get("data", [])

        return [
            RedditPost(
                id=post.get("id"),
                subreddit=post.get("subreddit") or subreddit,
                title=post.get("title") or "",
                body=post.get("selftext") or "",
                author=post.get("author"),
                score=post.get("score"),
                num_comments=post.get("num_comments"),
                created_utc=post.get("created_utc"),
                url=(
                    f"https://www.reddit.com{post['permalink']}"
                    if post.get("permalink", "").startswith("/")
                    else post.get("url")
                ),
            )
            for post in data
        ]

    def fetch_comments(
        self,
        post_id: str,
        limit: int = 100,
    ) -> list[RedditComment]:

        response = self._request(
            "comments/tree",
            {
                "link_id": post_id,
                "limit": limit,
            },
        )

        payload = response.json()
        data = payload.get("data", [])

        flattened = self._flatten_comment_tree(data)

        return [
            RedditComment(
                id=comment.get("id"),
                body=comment.get("body") or "",
                author=comment.get("author"),
                score=comment.get("score"),
            )
            for comment in flattened
        ]

    @staticmethod
    def _flatten_comment_tree(node) -> list[dict]:
        comments = []

        def visit(current):
            if isinstance(current, list):
                for item in current:
                    visit(item)
                return

            if not isinstance(current, dict):
                return

            payload = (
                current.get("data")
                if isinstance(current.get("data"), dict)
                else current
            )

            if (
                payload.get("id") is not None
                or payload.get("body") is not None
            ):
                comments.append(payload)

            for key in (
                "replies",
                "children",
                "comments",
            ):
                children = current.get(key)

                if children:
                    visit(children)

        visit(node)

        return comments