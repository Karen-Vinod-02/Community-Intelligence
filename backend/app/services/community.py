import logging
import time

import requests

from app.services.reddit.discovery.client import get_discovery_source
from app.services.reddit.client import get_reddit_source
from app.services.community_filter import CommunityFilter
from app.services.ranking import CommunityCandidate, CommunityRanker
from app.services.query_extraction import extract_search_queries


REMOVED_MARKERS = {
    "[removed]",
    "[deleted]",
}

MAX_POST_AGE_SECONDS = 90 * 24 * 60 * 60  # 90 days
MAX_COMMUNITY_CANDIDATES = 15

logger = logging.getLogger(__name__)


def _is_usable_post(post) -> bool:
    title = (post.title or "").strip().lower()
    body = (post.body or "").strip().lower()

    if title in REMOVED_MARKERS or body in REMOVED_MARKERS:
        return False
    if not title and not body:
        return False

    created = getattr(post, "created_utc", None)
    if created is not None:
        age_seconds = time.time() - created
        if age_seconds > MAX_POST_AGE_SECONDS:
            return False

    return True


def _belongs_to_subreddit(post, subreddit: str) -> bool:
    post_subreddit = getattr(post, "subreddit", None)
    return (
        isinstance(post_subreddit, str)
        and post_subreddit.casefold() == subreddit.casefold()
    )

class CommunityService:

    def __init__(self):
        self.discovery = get_discovery_source()
        self.reddit = get_reddit_source()
        self.ranker = CommunityRanker()
        self.community_filter = CommunityFilter(self.ranker.embeddings)

    def analyze(
        self,
        description: str,
        candidate_limit: int = 10,
        result_limit: int = 5,
        posts_per_community: int = 10,
    ):
        queries = extract_search_queries(description)

        # Store the query that discovered each subreddit. Discovery queries
        # identify communities; they are not reused as post-search filters.
        community_queries = {}
        community_names = set()

        for query in queries:
            try:
                subreddits = self.discovery.discover_communities(
                    query,
                    limit=candidate_limit,
                )

            except Exception as exc:
                logger.warning(
                    "Community discovery failed for query '%s': %s",
                    query,
                    exc,
                )
                continue

            for subreddit in subreddits:
                key = subreddit.casefold()
                if key not in community_names:
                    community_names.add(key)
                    community_queries[subreddit] = query
                    logger.debug(
                        "Discovered r/%s from community query '%s'",
                        subreddit,
                        query,
                    )

        candidates = []

        # Retrieve recent posts from a bounded candidate pool. The local
        # embedding/reranking stages decide relevance; Arctic Shift is not
        # asked to perform a second keyword search.
        for subreddit in list(community_queries)[:MAX_COMMUNITY_CANDIDATES]:

            try:
                raw_posts = self.reddit.search_posts(
                    subreddit=subreddit,
                    limit=posts_per_community * 2,
                )

            except requests.RequestException as exc:
                logger.warning(
                    "Skipping r/%s after Reddit retrieval failure: %s",
                    subreddit,
                    exc,
                )
                continue

            except Exception as exc:
                logger.warning(
                    "Skipping r/%s after unexpected retrieval failure: %s",
                    subreddit,
                    exc,
                )
                continue

            usable = [
                post
                for post in raw_posts
                if _is_usable_post(post)
                and _belongs_to_subreddit(post, subreddit)
            ][:posts_per_community]

            if not usable:
                continue

            candidates.append(
                CommunityCandidate(
                    subreddit=subreddit,
                    posts=usable,
                )
            )

        # Rerank all retrieved communities before post-level ranking. This
        # preserves recall while exposing intent signals for diagnostics.
        candidates = self.community_filter.rerank(
            description,
            candidates,
            queries=queries,
        )

        return self.ranker.rank(
            description,
            candidates,
            limit=result_limit,
            context="\n".join(queries),
        )