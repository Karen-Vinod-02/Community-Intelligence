"""
Minimal PullPush client for the *discovery* benchmark.

PullPush seems to be the only free option identified in this benchmark that provides Reddit-wide search over submission title/selftext, 
which enables subreddit discovery without requiring a predefined subreddit list.
It doesn't have a dedicated 'find relevant subreddits' endpoint, so discovery here would mean:
    1. Run keyword search across Reddit via the submission endpoint.
    2. Group returned matches by the `subreddit` field.
    3. Rank candidate subreddits by match count.


Docs: https://pullpush.io/ (see api/search/submission and api/search/comment)
Known constraints to design around:
    - ~30 req/min soft limit, documented outages
    - `after`/`before` support relative windows like "90d"
"""
import time
import sys
import os

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PULLPUSH_BASE, REQUEST_TIMEOUT_SECONDS, MAX_RETRIES, RETRY_BACKOFF_SECONDS
from metrics import CallResult, Timer


# Some Reddit-adjacent APIs reject requests with no/generic User-Agent.
HEADERS = {"User-Agent": "community-intelligence-benchmark/0.1 (university capstone project)"}


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


def discover(query: str, days_window: int = 90, size: int = 50) -> CallResult:
    """
    Keyword search across all of Reddit, filtered to the last `days_window` days.
    Returns a CallResult with the top subreddits ranked by simple match-count
    aggregation, plus a small sample of matching posts.
    """
    url = f"{PULLPUSH_BASE}/search/submission/"
    params = {
        "q": query,
        "after": f"{days_window}d",
        "size": size,
        "sort": "desc",
        "sort_type": "created_utc",
    }

    timestamp = time.time()
    http_status = None
    error = None
    num_results = 0
    bytes_returned = 0
    subreddit_counts: dict[str, int] = {}
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
            for post in data:
                sub = post.get("subreddit", "unknown")
                subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
            sample = [
                {"subreddit": p.get("subreddit"), "title": p.get("title")}
                for p in data[:3]
            ]
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    ranked_subreddits = sorted(subreddit_counts.items(), key=lambda kv: kv[1], reverse=True)

    return CallResult(
        provider="pullpush",
        call_type="discover",
        query_or_target=query,
        timestamp=timestamp,
        latency_seconds=round(t.elapsed, 3),
        http_status=http_status,
        ok=(error is None and http_status == 200),
        num_results=num_results,
        bytes_returned=bytes_returned,
        error=error,
        subreddits_seen=ranked_subreddits[:10],
        raw_sample=sample,
    )
