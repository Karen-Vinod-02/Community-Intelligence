import time

from app.services.reddit.discovery.client import get_discovery_source
from app.services.reddit.client import get_reddit_source
from app.services.ranking import CommunityCandidate, CommunityRanker
from app.services.query_extraction import extract_search_queries

REMOVED_MARKERS = {"[removed]", "[deleted]"}
MAX_POST_AGE_SECONDS = 90 * 24 * 60 * 60  # 90 days


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


class CommunityService:

    def __init__(self):
        self.discovery = get_discovery_source()
        self.reddit = get_reddit_source()
        self.ranker = CommunityRanker()

    def analyze(
        self,
        description: str,
        candidate_limit: int = 10,
        result_limit: int = 5,
        posts_per_community: int = 10,
    ):
        queries = extract_search_queries(description)

        seen = set()
        communities = []
        for query in queries:
            for subreddit in self.discovery.discover_communities(query, limit=candidate_limit):
                if subreddit not in seen:
                    seen.add(subreddit)
                    communities.append(subreddit)

        candidates = []
        for subreddit in communities:
            raw_posts = self.reddit.search_posts(
                subreddit=subreddit,
                limit=posts_per_community * 2,
            )
            usable = [p for p in raw_posts if _is_usable_post(p)][:posts_per_community]

            if not usable:
                continue

            candidates.append(CommunityCandidate(subreddit=subreddit, posts=usable))

        return self.ranker.rank(description, candidates, limit=result_limit)