import time
import os

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

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1


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

    def _request(self, endpoint: str, params: dict):
        url = f"{self.base_url}/{endpoint}"

        last_exception = None

        for attempt in range(self.max_retries + 1):
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
                    raise

                last_exception = exc

            except requests.RequestException as exc:
                last_exception = exc

            if attempt < self.max_retries:
                time.sleep(
                    RETRY_BACKOFF_SECONDS * (attempt + 1)
                )

        raise last_exception

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

                response = self._request(
                    "posts/search",
                    params,
                )
            else:
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