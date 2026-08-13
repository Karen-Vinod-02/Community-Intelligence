"""
Minimal Arctic Shift client for the *retrieval* benchmark.

Arctic Shift supports full- text search within a known subreddit, but not Reddit-wide
search for subreddit discovery.
It fits better in the 'fetch_posts / fetch_comments' stage, once PullPush 
or a known-subreddit list has identified the relevant communities.

Docs: https://github.com/ArthurHeitmann/arctic_shift/tree/master/api
Known constraints to design around:
    - No auth required
    - Significantly higher throughput ceiling than PullPush
    - Data latency: minutes to hours behind live Reddit (scheduled pulls).
"""
import time
import sys
import os

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARCTIC_SHIFT_BASE, REQUEST_TIMEOUT_SECONDS, MAX_RETRIES, RETRY_BACKOFF_SECONDS
from metrics import CallResult, Timer


HEADERS = {"User-Agent": "community-intelligence-benchmark/0.1"}


def _request_with_retries(url: str, params: dict):
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc


def fetch_posts(subreddit: str, days_window: int = 90, limit: int = 50) -> CallResult:
    """Fetch posts from a known subreddit within the last N days."""
    url = f"{ARCTIC_SHIFT_BASE}/posts/search"
    after_ts = int(time.time()) - days_window * 86400
    params = {
        "subreddit": subreddit,
        "after": after_ts,
        "limit": limit,
        "sort": "desc",
    }

    timestamp = time.time()
    http_status = None
    error = None
    num_results = 0
    bytes_returned = 0
    sample = []

    with Timer() as t:
        try:
            resp = _request_with_retries(url, params)
            http_status = resp.status_code
            bytes_returned = len(resp.content)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", [])
            num_results = len(data)
            sample = [
                {"id": p.get("id"), "title": p.get("title"), "num_comments": p.get("num_comments")}
                for p in data[:3]
            ]
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    return CallResult(
        provider="arcticshift",
        call_type="fetch_posts",
        query_or_target=subreddit,
        timestamp=timestamp,
        latency_seconds=round(t.elapsed, 3),
        http_status=http_status,
        ok=(error is None and http_status == 200),
        num_results=num_results,
        bytes_returned=bytes_returned,
        error=error,
        raw_sample=sample,
    )


def _flatten_comment_tree(node) -> list[dict]:
    """
    Arctic Shift returns comments as a tree (nested replies), not a flat list.
    This walks the tree defensively: it doesn't assume one exact key name for
    children, since that wasn't confirmed against a live response, only against
    the docs' description of "tree structure, similar to how they're displayed
    on reddit." It checks the common candidates (replies/children) and recurses.
    """
    flat = []

    def _visit(n):
        if isinstance(n, list):
            for item in n:
                _visit(item)
            return
        if not isinstance(n, dict):
            return

        # Handle responses where a comment may be wrapped under a "data" key.
        payload = n.get("data") if isinstance(n.get("data"), dict) else n

        if payload.get("id") is not None or payload.get("body") is not None:
            flat.append({
                "id": payload.get("id"),
                "body": payload.get("body"),
                "author": payload.get("author"),
                "score": payload.get("score"),
            })

        # Recurse into whichever nesting key is actually present.
        for children_key in ("replies", "children", "comments"):
            children = n.get(children_key)
            if children:
                _visit(children)

    _visit(node)
    return flat


def fetch_comments(post_id: str) -> CallResult:
    url = f"{ARCTIC_SHIFT_BASE}/comments/tree"
    params = {"link_id": post_id, "limit": 100}

    timestamp = time.time()
    http_status = None
    error = None
    num_results = 0
    bytes_returned = 0
    sample = []
    raw_top_level_keys = None  #captured for debugging if shape is still unexpected

    with Timer() as t:
        try:
            resp = _request_with_retries(url, params)
            http_status = resp.status_code
            bytes_returned = len(resp.content)
            resp.raise_for_status()
            payload = resp.json()
            raw_top_level_keys = list(payload.keys()) if isinstance(payload, dict) else None
            raw_data = payload.get("data", [])
            flat_comments = _flatten_comment_tree(raw_data)
            num_results = len(flat_comments)
            sample = [
                {"id": c.get("id"), "body": (c.get("body") or "")[:80]}
                for c in flat_comments[:3]
            ]
            if num_results == 0:
                # Nothing matched our known shapes -> surface the raw top-level
                # keys so it's obvious in the JSON output that this needs a
                # manual look, rather than silently reporting "0 comments".
                sample = [{"_debug_top_level_keys": raw_top_level_keys, "_debug_raw_snippet": str(raw_data)[:200]}]
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    return CallResult(
        provider="arcticshift",
        call_type="fetch_comments",
        query_or_target=post_id,
        timestamp=timestamp,
        latency_seconds=round(t.elapsed, 3),
        http_status=http_status,
        ok=(error is None and http_status == 200),
        num_results=num_results,
        bytes_returned=bytes_returned,
        error=error,
        raw_sample=sample,
    )
