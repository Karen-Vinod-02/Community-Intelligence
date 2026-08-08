"""
Reddit official Data API discovery client, via PRAW, using a personal 'script' app 
Status: pending access authorization 

Setup (one-time, approx. 5 minutes):
    1. Create a Reddit "script" app at https://www.reddit.com/prefs/apps
    2. Copy the client ID and secret.
    3. Add them to experiments/benchmark/.env (see .env.example):
        REDDIT_CLIENT_ID = your_client_id
        REDDIT_CLIENT_SECRET = your_secret
        REDDIT_USER_AGENT = community-intelligence-benchmark/0.1 by u/yourusername

Known constraints to design around:
    - ~100 queries/minute, higher than PullPush's ~30/min
    - No arbitrary date-range parameter. time_filter only supports
      day/week/month/year/all, so a 90-day window uses year & filters created_utc client-side.
    - r/all search uses Reddit's relevance ranking rather than a pure keyword index,
      so results may be noisier than PullPush's search.

"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    REDDIT_CLIENT_ID_ENV,
    REDDIT_CLIENT_SECRET_ENV,
    REDDIT_USER_AGENT_ENV,
    REDDIT_DEFAULT_USER_AGENT,
)
from metrics import CallResult, Timer

try:
    import praw
    import prawcore
except ImportError:
    praw = None
    prawcore = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can also be set directly


class CredentialsMissingError(RuntimeError):
    pass


def _get_reddit():
    if praw is None:
        raise CredentialsMissingError(
            "praw is not installed. Run: pip install praw python-dotenv"
        )

    client_id = os.environ.get(REDDIT_CLIENT_ID_ENV)
    client_secret = os.environ.get(REDDIT_CLIENT_SECRET_ENV)
    user_agent = os.environ.get(REDDIT_USER_AGENT_ENV, REDDIT_DEFAULT_USER_AGENT)

    if not client_id or not client_secret:
        raise CredentialsMissingError(
            f"Missing {REDDIT_CLIENT_ID_ENV} / {REDDIT_CLIENT_SECRET_ENV} env vars. "
            "Create a Reddit 'script' app at https://www.reddit.com/prefs/apps and "
            "set them in a .env file — see clients/reddit_praw.py docstring for steps."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def discover(query: str, days_window: int = 90, size: int = 50) -> CallResult:
    """
    Keyword search across r/all via the official Reddit API, filtered
    client-side to the last `days_window` days, ranked by subreddit
    match count — same discovery pattern as the PullPush client, so this
    is a drop-in replacement in the CommunitySource abstraction.
    """
    timestamp = time.time()
    http_status = None
    error = None
    num_results = 0
    bytes_returned = 0  # PRAW doesn't expose raw byte counts; left at 0
    subreddit_counts: dict[str, int] = {}
    sample = []
    cutoff = time.time() - days_window * 86400

    with Timer() as t:
        try:
            reddit = _get_reddit()
            results = reddit.subreddit("all").search(
                query, sort="new", time_filter="year", limit=size
            )
            for post in results:
                if post.created_utc < cutoff:
                    continue
                num_results += 1
                sub = str(post.subreddit)
                subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
                if len(sample) < 3:
                    sample.append({"subreddit": sub, "title": post.title})
            http_status = 200
        except CredentialsMissingError as e:
            error = str(e)
            http_status = None
        except Exception as e:
            # Covers prawcore.exceptions.ResponseException (401/403/429) and
            # any other network/auth failure without needing prawcore imported
            # at module load time if it's unavailable.
            error = f"{type(e).__name__}: {e}"
            http_status = getattr(getattr(e, "response", None), "status_code", None)

    ranked_subreddits = sorted(subreddit_counts.items(), key=lambda kv: kv[1], reverse=True)

    return CallResult(
        provider="reddit_official",
        call_type="discover",
        query_or_target=query,
        timestamp=timestamp,
        latency_seconds=round(t.elapsed, 3),
        http_status=http_status,
        ok=(error is None),
        num_results=num_results,
        bytes_returned=bytes_returned,
        error=error,
        subreddits_seen=ranked_subreddits[:10],
        raw_sample=sample,
    )
